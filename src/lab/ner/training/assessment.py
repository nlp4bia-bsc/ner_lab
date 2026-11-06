"""Running one fixed training configuration against a prepared split."""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, set_seed

from lab.core.provenance import write_manifest
from lab.core.split import split_paths
from lab.ner.encoding.encoder import Encoder, WindowStrategy, describe_encoder, encode_partition
from lab.ner.encoding.overlaps import OverlapPolicy
from lab.ner.evaluation.metrics import build_compute_metrics
from lab.ner.models.capacity import check_max_length
from lab.ner.models.registry import Architecture, architecture_name, build_model
from lab.ner.training.arguments import training_arguments as build_training_arguments
from lab.ner.training.devices import (
    DevicePolicy,
    apply_device_policy,
    effective_train_batch_size,
)
from lab.ner.training.folds import (
    FOLD_METRICS_FILENAME,
    aggregate_metrics,
    fold_row,
    write_fold_metrics,
)
from lab.ner.training.multihead import architecture_arguments, head_architecture_kwargs
from lab.ner.training.runs import (
    RUN_MANIFEST_FILENAME,
    claim_run_dir,
    fold_rotations,
    read_data_manifest,
    split_provenance,
)
from lab.ner.training.tracking import gpu_hardware_info
from lab.ner.training.trainer import TrainingResult, train

ASSESSMENT_SUMMARY_FILENAME = "assessment_summary.json"


@dataclass(frozen=True)
class AssessmentResult:
    """What one assessment produced: per-run metrics, their aggregate, and where they landed."""

    run_dir: Path
    mode: str
    fold_metrics: pd.DataFrame
    aggregate: dict[str, dict[str, float]]
    summaries: list[dict[str, Any]]
    manifest: dict[str, Any]
    paths: dict[str, Path]


def train_model(
    split_dir: str | Path,
    output_dir: str | Path,
    base_model: str,
    target_label: str | None = None,
    language: str | None = None,
    architecture: str | Architecture = "linear",
    architecture_kwargs: dict[str, Any] | None = None,
    training_arguments: TrainingArguments | Mapping[str, Any] | None = None,
    max_length: int = 256,
    strategy: str | WindowStrategy = "greedy",
    context_tokens: int | None = None,
    overlap_policy: OverlapPolicy | None = None,
    min_sentence_tokens: int = 4,
    early_stopping_patience: int | None = 5,
    pad_to_multiple_of: int | None = 8,
    metrics_scope: str = "eval",
    min_overlap_percentage: float = 40.0,
    include_confusion: bool = False,
    save_model: bool = False,
    track_resources: bool = True,
    devices: DevicePolicy = 1,
    overwrite: bool = False,
    folds: Sequence[int] | None = None,
    random_state: int | None = None,
) -> AssessmentResult:
    """
    Train one configuration against the split in `split_dir` and record what it scored.

    The split's `data_manifest.json` decides the shape of the run: a
    `train_validation` split trains once, a `fixed_holdout_kfold` split trains
    once per rotatable fold and aggregates. The fixed holdout is never read.

    `output_dir` is the run directory: artifacts go into it directly, with each fold
    of a k-fold split in its own `fold_XX/` subdirectory. Nothing is derived and
    nothing is nested, so the layout is yours to decide — join
    `run_directory_name()` yourself for a timestamped one. A directory another run
    already claimed raises unless `overwrite` is set, checked before anything is
    created. `run_manifest.json` is written before training starts, so a crashed run
    still explains itself.

    `training_arguments` takes a `TrainingArguments`, a mapping of overrides
    applied to this library's defaults, or nothing for the defaults alone. Every
    other keyword configures the `Encoder` or `build_model` that this assembles;
    build those yourself and call `lab.ner.training.train` if the assembly is in
    the way.

    Weights are not kept unless `save_model` is set — this scores a
    configuration, it does not produce a deployment artifact. A saved fold
    carries an `encoding.json` beside its weights, which is what
    `lab.ner.inference` loads. Each fold gets a freshly built model, and the
    previous fold's is released before the next one is built.

    Each fold's model runs one forward pass on CPU at `max_length` before training, and
    prints a line when it passes, so a window longer than the model's positions raises
    naming the length rather than as a CUDA device-side assert in the first step.

    `devices` defaults to `1`, which pins the run to one GPU however many the
    process can see. Left to itself `Trainer` would train at
    `per_device_train_batch_size x n_gpu` on a schedule as many times shorter, and a
    configuration `lab.ner.hpo` searched on one device is not the same configuration
    on four. `"all"` spreads the run over every visible device and warns, for callers
    who want the throughput and accept that the batch size and the learning-rate
    schedule are no longer the ones that were tuned. The policy, the number of
    devices trained on and the resulting `effective_train_batch_size` all go into the
    run manifest, so a run that took the throughput says so on paper rather than only
    in its stderr.
    """
    if language is None:
        raise ValueError("language is required; pass an ISO 639-1 code explicitly.")
    if architecture in ("multihead", "multihead_crf") and target_label is not None:
        raise ValueError(
            f"architecture={architecture} requires target_label=None for automatic discovery."
        )
    if architecture not in ("multihead", "multihead_crf") and target_label is None:
        raise ValueError("target_label is required for single-label architectures.")
    training_arguments = architecture_arguments(training_arguments, architecture)
    run_dir = claim_run_dir(output_dir, overwrite)

    split_dir = Path(split_dir).resolve()
    data_manifest = read_data_manifest(split_dir)
    mode = data_manifest["split"]["mode"]
    rotations = fold_rotations(data_manifest, folds)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    encoder = Encoder(
        tokenizer=tokenizer,
        target_label=target_label,
        language=language,
        max_length=max_length,
        overlap_policy=overlap_policy,
        strategy=strategy,
        context_tokens=context_tokens,
        min_sentence_tokens=min_sentence_tokens,
        include_word_mask=architecture in ("crf", "multihead_crf")
        and (architecture_kwargs or {}).get("crf_version", 2) == 2,
    )

    resolved_arguments = resolve_training_arguments(training_arguments, run_dir, random_state)

    device_count = apply_device_policy(resolved_arguments, devices)

    manifest = {
        "split_dir": str(split_dir),
        "data_manifest": data_manifest,
        "mode": mode,
        "folds": rotations,
        **(
            {"head_discovery": "training annotations independently per fold"}
            if architecture in ("multihead", "multihead_crf")
            else {}
        ),
        **model_encoding(encoder, base_model, architecture, architecture_kwargs),
        "evaluation": {"min_overlap_percentage": min_overlap_percentage},
        "training_arguments": resolved_arguments.to_dict(),
        "hardware": gpu_hardware_info(),
        "devices": {
            "policy": devices,
            "trained_on": device_count,
            "effective_train_batch_size": effective_train_batch_size(
                resolved_arguments, device_count
            ),
        },
    }
    manifest_path = write_manifest(manifest, run_dir / RUN_MANIFEST_FILENAME)

    encoded: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    paths: dict[str, Path] = {"run_manifest": manifest_path}

    for validation_index in rotations:
        result = _train_fold(
            validation_index=validation_index,
            split_dir=split_dir,
            run_dir=run_dir,
            encoder=encoder,
            encoded=encoded,
            training_arguments=resolved_arguments,
            base_model=base_model,
            architecture=architecture,
            architecture_kwargs=architecture_kwargs,
            early_stopping_patience=early_stopping_patience,
            pad_to_multiple_of=pad_to_multiple_of,
            metrics_scope=metrics_scope,
            min_overlap_percentage=min_overlap_percentage,
            include_confusion=include_confusion,
            save_model=save_model,
            track_resources=track_resources,
        )

        summaries.append(result.summary)
        if architecture in ("multihead", "multihead_crf"):
            catalogues = manifest.setdefault("entity_labels_by_fold", {})
            catalogues[fold_directory_key(validation_index)] = result.summary["metadata"][
                "entity_labels"
            ]
            write_manifest(manifest, manifest_path)
        rows.append(
            fold_row(
                validation_index=validation_index,
                summary=result.summary,
                epoch_metrics=result.epoch_metrics,
                greater_is_better=bool(resolved_arguments.greater_is_better),
            )
        )
        paths[fold_directory_key(validation_index)] = result.output_dir

        del result
        release_accelerator_memory()

    fold_metrics = pd.DataFrame(rows)
    aggregate = aggregate_metrics(fold_metrics, exclude=("validation_fold",))

    summary = {
        "mode": mode,
        "n_runs": len(rows),
        "metric_for_best_model": resolved_arguments.metric_for_best_model,
        "aggregate": aggregate,
    }

    paths["fold_metrics"] = write_fold_metrics(fold_metrics, run_dir / FOLD_METRICS_FILENAME)
    paths["assessment_summary"] = write_manifest(summary, run_dir / ASSESSMENT_SUMMARY_FILENAME)

    return AssessmentResult(
        run_dir=run_dir,
        mode=mode,
        fold_metrics=fold_metrics,
        aggregate=aggregate,
        summaries=summaries,
        manifest=manifest,
        paths=paths,
    )


def model_encoding(
    encoder: Encoder,
    base_model: str,
    architecture: str | Architecture = "linear",
    architecture_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    What a saved model needs to state about itself: how it was built, how its input was windowed.

    Handed to `train(model_encoding=...)`, which writes it beside the weights as
    `encoding.json`. Inference reads it back rather than asking the caller to
    restate a `max_length` or a `strategy` that silently changes every window
    boundary if it is off by one.
    """
    structured = (
        architecture in ("crf", "multihead_crf")
        and (architecture_kwargs or {}).get("crf_version", 2) == 2
    )
    if structured and not encoder.include_word_mask:
        raise ValueError("New CRF encoding requires include_word_mask=True.")
    settings = dict(architecture_kwargs or {})
    if structured:
        settings["crf_version"] = 2
    catalogue = encoder.entity_labels if encoder.target_label is None else [encoder.target_label]
    return {
        **(
            {
                "format_version": 2,
                "crf_version": 2,
                "entity_labels": catalogue,
                "num_heads": len(catalogue or []),
                "head_to_label": dict(enumerate(catalogue or [])),
                "bio_schema": {"O": 0, "B": 1, "I": 2},
            }
            if structured
            else {}
        ),
        **(
            {
                "format_version": 2,
                "entity_labels": encoder.entity_labels,
                "num_heads": len(encoder.entity_labels or []),
                "head_to_label": dict(enumerate(encoder.entity_labels or [])),
                "bio_schema": {"O": 0, "B": 1, "I": 2},
            }
            if architecture in ("multihead", "multihead_crf") and encoder.entity_labels is not None
            else {}
        ),
        "model": {
            "base_model": base_model,
            "architecture": architecture_name(architecture),
            "architecture_kwargs": (
                head_architecture_kwargs(encoder, settings) if encoder.entity_labels else settings
            ),
        },
        "encoding": describe_encoder(encoder),
    }


def resolve_training_arguments(
    arguments: TrainingArguments | Mapping[str, Any] | None,
    output_dir: str | Path,
    random_state: int | None = None,
) -> TrainingArguments:
    """
    Settle a `TrainingArguments` from an instance, a mapping of overrides, or nothing.

    A mapping is applied on top of this library's defaults, which is how the YAML
    path supplies them. `random_state` overrides `seed` in every case.
    """
    overrides: dict[str, Any] = {} if random_state is None else {"seed": random_state}

    if arguments is None:
        return build_training_arguments(output_dir, **overrides)

    if isinstance(arguments, Mapping):
        return build_training_arguments(output_dir, **{**arguments, **overrides})

    return dataclasses.replace(arguments, output_dir=str(output_dir), **overrides)


def release_accelerator_memory() -> None:
    """Drop the previous run's model from the allocator before the next one is built."""
    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fold_directory_key(validation_index: int | None) -> str:
    """The `paths` key an individual run's directory is recorded under."""
    return "run" if validation_index is None else f"fold_{validation_index:02d}"


def _train_fold(
    validation_index: int | None,
    split_dir: Path,
    run_dir: Path,
    encoder: Encoder,
    encoded: dict[str, pd.DataFrame],
    training_arguments: TrainingArguments,
    base_model: str,
    architecture: str | Architecture,
    architecture_kwargs: dict[str, Any] | None,
    early_stopping_patience: int | None,
    pad_to_multiple_of: int | None,
    metrics_scope: str,
    min_overlap_percentage: float,
    include_confusion: bool,
    save_model: bool,
    track_resources: bool,
) -> TrainingResult:
    partitions = split_paths(split_dir, validation_index=validation_index)
    fold_dir = run_dir if validation_index is None else run_dir / f"fold_{validation_index:02d}"

    if architecture in ("multihead", "multihead_crf"):
        from lab.ner.encoding.encoder import encoder_from_description
        from lab.ner.encoding.multihead import training_entity_labels

        encoder = encoder_from_description(
            describe_encoder(encoder),
            encoder.tokenizer,
            strategy=encoder.strategy,
            entity_labels=training_entity_labels(partitions["train"]),
        )
        encoded = {}
    architecture_kwargs = head_architecture_kwargs(encoder, architecture_kwargs)
    train_rows = encode_partition(encoder, partitions["train"], encoded)
    validation_rows = encode_partition(encoder, partitions["validation"], encoded)

    set_seed(training_arguments.seed)

    model = build_model(
        base_model=base_model,
        label2id=encoder.label2id,
        id2label=encoder.id2label,
        architecture=architecture,
        **(architecture_kwargs or {}),
    )
    check_max_length(model, encoder.tokenizer, encoder.max_length, base_model)

    return train(
        model=model,
        tokenizer=encoder.tokenizer,
        train_rows=train_rows,
        validation_rows=validation_rows,
        training_arguments=dataclasses.replace(training_arguments, output_dir=str(fold_dir)),
        compute_metrics=build_compute_metrics(
            rows=validation_rows,
            tokenizer=encoder.tokenizer,
            id2label=encoder.id2label,
            min_overlap_percentage=min_overlap_percentage,
            include_confusion=include_confusion,
        ),
        train_compute_metrics=(
            build_compute_metrics(
                rows=train_rows,
                tokenizer=encoder.tokenizer,
                id2label=encoder.id2label,
                min_overlap_percentage=min_overlap_percentage,
                include_confusion=False,
            )
            if metrics_scope == "both"
            else None
        ),
        early_stopping_patience=early_stopping_patience,
        pad_to_multiple_of=pad_to_multiple_of,
        metrics_scope=metrics_scope,
        save_model=save_model,
        track_resources=track_resources,
        run_metadata={
            **split_provenance(validation_index, partitions),
            **(
                {"entity_labels": encoder.entity_labels, "encoding": describe_encoder(encoder)}
                if architecture in ("multihead", "multihead_crf")
                else {}
            ),
        },
        model_encoding=model_encoding(
            encoder=encoder,
            base_model=base_model,
            architecture=architecture,
            architecture_kwargs=architecture_kwargs,
        ),
    )
