"""One trial: one sampled configuration, trained once per seed and scored across them."""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import statistics
from collections.abc import Callable, Iterable
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import TrainingArguments, set_seed

from lab.ner.evaluation.metrics import build_compute_metrics
from lab.ner.hpo.space import resolve_batch_sizes
from lab.ner.hpo.variants import EncodedVariant
from lab.ner.models.registry import Architecture, build_model
from lab.ner.training.devices import require_single_device
from lab.ner.training.tracking import ResourceTracker
from lab.ner.training.trainer import train

CRF_DIMENSIONS = (
    "dropout",
    "aux_ce_weight",
    "constrain_transitions",
    "loss_type",
    "loss_reduction",
    "tag_scheme",
)
from lab.ner.hpo.binary import BINARY_DIMENSIONS

ARCHITECTURE_DIMENSIONS = tuple(
    f"architecture__{name}" for name in dict.fromkeys((*CRF_DIMENSIONS, *BINARY_DIMENSIONS))
)
RESERVED_DIMENSIONS = (
    "variant",
    "loss_variant",
    "effective_train_batch_size",
    *ARCHITECTURE_DIMENSIONS,
)


def sampled_architecture_kwargs(
    sampled: dict[str, Any], settings: dict[str, Any] | None
) -> dict[str, Any]:
    """Route namespaced architecture dimensions separately from optimizer arguments."""
    result = dict(settings or {})
    result.update(
        {
            name.removeprefix("architecture__"): value
            for name, value in sampled.items()
            if name in ARCHITECTURE_DIMENSIONS
        }
    )
    if "loss_variant" in sampled:
        result.update(json.loads(sampled["loss_variant"]))

    return result


def validate_architecture_dimensions(
    space: dict[str, Any],
    architecture: str | Architecture,
    settings: dict[str, Any] | None = None,
) -> None:
    """Reject unsupported or incompatible architecture search families before trials."""
    if architecture == "multilabel":
        from lab.ner.hpo.binary import validate_binary_dimensions

        validate_binary_dimensions(space, settings or {})
        return
    dimensions = set(space) & set(ARCHITECTURE_DIMENSIONS)
    if any(name.removeprefix("architecture__") not in CRF_DIMENSIONS for name in dimensions):
        raise ValueError("Binary loss dimensions require architecture=multilabel.")
    settings = settings or {}
    if dimensions and (
        architecture not in ("crf", "multihead_crf") or settings.get("crf_version", 2) != 2
    ):
        raise ValueError("CRF search dimensions require version-2 crf or multihead_crf.")
    allowed = {
        "tag_scheme": ["BIO2"],
        "loss_reduction": ["word_mean"],
        "loss_type": ["crf", "cross_entropy"],
        "constrain_transitions": [True, False],
    }
    for name, options in allowed.items():
        domain = space.get(f"architecture__{name}")
        if domain is not None and (
            getattr(domain, "categories", None) is None
            or any(value not in options for value in domain.categories)
        ):
            raise ValueError(f"architecture__{name} must choose from {options}.")
    for name in ("dropout", "aux_ce_weight"):
        domain = space.get(f"architecture__{name}")
        if domain is None:
            continue
        candidates = getattr(domain, "categories", None)
        bounds = candidates if candidates is not None else [domain.lower, domain.upper]
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or name == "dropout"
            and value > 1
            for value in bounds
        ):
            raise ValueError(f"Invalid nonnegative search bounds for architecture__{name}.")
    loss_domain = space.get("architecture__loss_type")
    losses = (
        loss_domain.categories if loss_domain is not None else [settings.get("loss_type", "crf")]
    )
    auxiliary = space.get("architecture__aux_ce_weight")
    values = (
        (getattr(auxiliary, "categories", None) or [auxiliary.upper])
        if auxiliary is not None
        else [settings.get("aux_ce_weight", 0)]
    )
    if "cross_entropy" in losses and any(value != 0 for value in values):
        raise ValueError("Use separate sweeps for constrained CE and nonzero auxiliary CE weights.")


OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "not enough memory",
    "defaultcpuallocator",
)


def validate_search_space(dimensions: Iterable[str]) -> None:
    """
    Reject dimensions no trial can apply, before the corpus is encoded.

    Every sampled dimension outside `RESERVED_DIMENSIONS` is unpacked into
    `TrainingArguments`, so a name it does not define raises `TypeError` inside a
    Ray worker — long after the variants have been windowed. Architecture settings
    such as `crf_dropout` are not fields: they are fixed for the sweep and passed
    through `architecture_kwargs`.
    """
    known = {field.name for field in dataclasses.fields(TrainingArguments)}
    unknown = sorted(set(dimensions) - known - set(RESERVED_DIMENSIONS))

    if not unknown:
        return

    reported = ", ".join(
        f"{name!r} (did you mean {close[0]!r}?)"
        if (close := get_close_matches(name, known, n=1))
        else repr(name)
        for name in unknown
    )

    raise ValueError(
        f"Unknown search dimension(s): {reported}. A dimension must name a "
        f"`TrainingArguments` field or one of {RESERVED_DIMENSIONS}."
    )


def metric_greater_is_better(arguments: TrainingArguments) -> bool:
    """Whether the selection metric is maximized, resolving HF's `None` the way HF does."""
    if arguments.greater_is_better is not None:
        return bool(arguments.greater_is_better)

    return not (arguments.metric_for_best_model or "").endswith("loss")


def top_k_epoch_mean(
    epoch_metrics: pd.DataFrame,
    metric: str,
    k: int = 3,
    greater_is_better: bool = True,
) -> float:
    """
    Mean of the k best per-epoch eval values of `metric`.

    A less noisy trial score than the single best epoch, which is an order
    statistic of a noisy signal and rewards a lucky epoch as much as a good
    configuration.
    """
    column = metric.removeprefix("eval_")
    evaluated = epoch_metrics

    if "split" in epoch_metrics.columns:
        evaluated = epoch_metrics[epoch_metrics["split"] == "eval"]

    values = (
        evaluated[column].dropna().astype(float).tolist() if column in evaluated.columns else []
    )

    if not values:
        raise ValueError(f"No {column!r} values in the epoch metrics.")

    top = sorted(values, reverse=greater_is_better)[:k]

    return sum(top) / len(top)


def is_oom_error(error: BaseException) -> bool:
    """True for CUDA and allocator out-of-memory signatures."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True

    text = str(error).lower()

    return any(pattern in text for pattern in OOM_PATTERNS)


def release_trial_memory() -> None:
    """Free as much accelerator memory as possible between seeds or after a failure."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_trial(
    sampled: dict[str, Any],
    variants: dict[str, EncodedVariant],
    base_arguments: TrainingArguments,
    trial_dir: str | Path,
    architecture: str | Architecture = "linear",
    architecture_kwargs: dict[str, Any] | None = None,
    max_micro_batch_size: int = 64,
    seeds_per_trial: int = 3,
    top_k_epochs: int = 3,
    min_overlap_percentage: float = 40.0,
    early_stopping_patience: int | None = 5,
    pad_to_multiple_of: int | None = 8,
    track_resources: bool = True,
) -> dict[str, Any]:
    """
    Run one sampled configuration once per seed and return what Ray should report.

    The score is the mean across seeds of each seed's mean-of-top-k epochs —
    deliberately not a single run's single best epoch, since epoch-to-epoch
    jitter and run-to-run nondeterminism are each the size of the search-space
    spread being measured.

    Only out-of-memory failures are caught — a legitimate, hyperparameter-dependent
    outcome of the sweep. Everything else propagates, so a bug halts the sweep
    instead of burning the remaining trials on it.

    An out-of-memory trial scores the seeds that did finish, and only falls to the
    worst possible score when none did. A configuration too large to train fails on
    the first seed and is ranked infeasible as it should be; one that trains several
    seeds and then hits a fragmented or contended device has shown it fits, and its
    completed runs are evidence rather than something to discard.
    """
    if seeds_per_trial < 1:
        raise ValueError("seeds_per_trial must be at least 1.")

    require_single_device(
        base_arguments,
        context=(
            "Under Ray each trial reserves exactly one GPU, so this means the "
            "reservation was bypassed. Running more trials in parallel beats "
            "splitting one trial across devices."
        ),
    )

    architecture_kwargs = sampled_architecture_kwargs(sampled, architecture_kwargs)
    if any(name in sampled for name in ARCHITECTURE_DIMENSIONS) and architecture not in (
        "crf",
        "multihead_crf",
        "multilabel",
    ):
        raise ValueError("CRF architecture dimensions require crf or multihead_crf.")
    metric = base_arguments.metric_for_best_model

    if not metric:
        raise ValueError("base_arguments.metric_for_best_model must name the trial objective.")

    metric_key = metric.removeprefix("eval_")
    greater_is_better = metric_greater_is_better(base_arguments)
    variant = variants[sampled["variant"]]
    if architecture == "multilabel":
        from lab.ner.training.binary import binary_architecture_kwargs

        architecture_kwargs = binary_architecture_kwargs(
            architecture_kwargs, variant.train_rows, variant.entity_labels
        )

    micro_batch_size, accumulation_steps = _batch_sizes(
        sampled, base_arguments, max_micro_batch_size
    )

    hyperparameters = {
        key: value for key, value in sampled.items() if key not in RESERVED_DIMENSIONS
    }

    if architecture == "multilabel":
        from lab.ner.training.binary import configure_binary_metrics

        configure_binary_metrics(variant.validation_rows, architecture_kwargs)

    compute_metrics = build_compute_metrics(
        rows=variant.validation_rows,
        tokenizer=variant.tokenizer,
        id2label=variant.id2label,
        min_overlap_percentage=min_overlap_percentage,
        include_tokens=False,
        include_by_entity=False,
    )

    scores: list[float] = []
    tracker = ResourceTracker() if track_resources else None
    resources: dict[str, Any] = {}
    oom = False

    if tracker is not None:
        tracker.start()

    try:
        for index in range(seeds_per_trial):
            seed = base_arguments.seed + index
            arguments = dataclasses.replace(
                base_arguments,
                output_dir=str(Path(trial_dir) / f"seed_{seed}"),
                seed=seed,
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=accumulation_steps,
                **hyperparameters,
            )

            scores.append(
                _seed_score(
                    variant=variant,
                    arguments=arguments,
                    architecture=architecture,
                    architecture_kwargs=architecture_kwargs,
                    compute_metrics=compute_metrics,
                    early_stopping_patience=early_stopping_patience,
                    pad_to_multiple_of=pad_to_multiple_of,
                    metric=metric,
                    top_k_epochs=top_k_epochs,
                    greater_is_better=greater_is_better,
                )
            )
            release_trial_memory()
    except Exception as error:
        if not is_oom_error(error):
            raise

        oom = True
        release_trial_memory()
    finally:
        if tracker is not None:
            resources = tracker.stop()

    worst = float("-inf") if greater_is_better else float("inf")

    return {
        **resources,
        "oom": oom,
        "n_seeds": len(scores),
        "per_seed_scores": scores,
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        f"{metric_key}_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        metric_key: statistics.mean(scores) if scores else worst,
    }


def _batch_sizes(
    sampled: dict[str, Any], base_arguments: TrainingArguments, max_micro_batch_size: int
) -> tuple[int, int]:
    if "effective_train_batch_size" in sampled:
        return resolve_batch_sizes(int(sampled["effective_train_batch_size"]), max_micro_batch_size)

    return base_arguments.per_device_train_batch_size, base_arguments.gradient_accumulation_steps


def _seed_score(
    variant: EncodedVariant,
    arguments: TrainingArguments,
    architecture: str | Architecture,
    architecture_kwargs: dict[str, Any] | None,
    compute_metrics: Callable[[Any], dict],
    early_stopping_patience: int | None,
    pad_to_multiple_of: int | None,
    metric: str,
    top_k_epochs: int,
    greater_is_better: bool,
) -> float:
    set_seed(arguments.seed)

    settings = dict(architecture_kwargs or {})
    if architecture in ("multihead", "multihead_crf", "multilabel"):
        settings["entity_labels"] = variant.entity_labels
    if architecture == "multilabel":
        from lab.ner.training.binary import binary_architecture_kwargs

        settings = binary_architecture_kwargs(settings, variant.train_rows, variant.entity_labels)
    model = build_model(
        base_model=variant.base_model,
        label2id=variant.label2id,
        id2label=variant.id2label,
        architecture=architecture,
        **settings,
    )

    result = train(
        model=model,
        tokenizer=variant.tokenizer,
        train_rows=variant.train_rows,
        validation_rows=variant.validation_rows,
        training_arguments=arguments,
        compute_metrics=compute_metrics,
        early_stopping_patience=early_stopping_patience,
        pad_to_multiple_of=pad_to_multiple_of,
        track_resources=False,
    )

    return top_k_epoch_mean(result.epoch_metrics, metric, top_k_epochs, greater_is_better)
