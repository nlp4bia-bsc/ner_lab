"""Running one fixed training configuration against a prepared split."""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, set_seed

from ner_lab.data.dataset import DATA_MANIFEST_FILENAME
from ner_lab.data.split import split_paths
from ner_lab.encoding.encoder import Encoder, WindowStrategy, describe_encoder
from ner_lab.encoding.overlaps import OverlapPolicy
from ner_lab.evaluation.metrics import build_compute_metrics
from ner_lab.models.registry import Architecture, build_model
from ner_lab.provenance import read_manifest, write_manifest
from ner_lab.training.arguments import training_arguments as build_training_arguments
from ner_lab.training.tracking import gpu_hardware_info
from ner_lab.training.trainer import train

RUN_MANIFEST_FILENAME = "run_manifest.json"
FOLD_METRICS_FILENAME = "fold_metrics.parquet"
ASSESSMENT_SUMMARY_FILENAME = "assessment_summary.json"

SPLIT_MODES = ("train_validation", "fixed_holdout_kfold")


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
    target_label: str,
    language: str,
    architecture: str | Architecture = "linear",
    architecture_kwargs: dict[str, Any] | None = None,
    training_arguments: TrainingArguments | Mapping[str, Any] | None = None,
    max_length: int = 256,
    strategy: str | WindowStrategy = "greedy",
    context_tokens: int | None = None,
    overlap_policy: OverlapPolicy = "merge_same_label_then_keep_longest",
    min_sentence_tokens: int = 4,
    early_stopping_patience: int | None = 5,
    pad_to_multiple_of: int | None = 8,
    metrics_scope: str = "eval",
    min_overlap_percentage: float = 40.0,
    include_confusion: bool = False,
    save_model: bool = False,
    track_resources: bool = True,
    folds: Sequence[int] | None = None,
    random_state: int | None = None,
    run_name: str | None = None,
) -> AssessmentResult:
    """
    Train one configuration against the split in `split_dir` and record what it scored.

    The split's `data_manifest.json` decides the shape of the run: a
    `train_validation` split trains once, a `fixed_holdout_kfold` split trains
    once per rotatable fold and aggregates. The fixed holdout is never read.

    Artifacts go to `<output_dir>/<target_label>__<architecture>__<base_model>__<timestamp>/`,
    with each fold in its own `fold_XX/` subdirectory. `run_manifest.json` is
    written before training starts, so a crashed run still explains itself.

    `training_arguments` takes a `TrainingArguments`, a mapping of overrides
    applied to this library's defaults, or nothing for the defaults alone. Every
    other keyword configures the `Encoder` or `build_model` that this assembles;
    build those yourself and call `ner_lab.training.train` if the assembly is in
    the way.

    Weights are not kept unless `save_model` is set — this scores a
    configuration, it does not produce a deployment artifact. A saved fold
    carries an `encoding.json` beside its weights, which is what
    `ner_lab.inference` loads. Each fold gets a freshly built model, and the
    previous fold's is released before the next one is built.
    """
    split_dir = Path(split_dir).resolve()
    data_manifest = read_data_manifest(split_dir)
    mode = data_manifest["split"]["mode"]
    rotations = fold_rotations(data_manifest, folds)

    run_dir = Path(output_dir).resolve() / (
        run_name or run_directory_name(target_label, architecture, base_model)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

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
    )

    resolved_arguments = resolve_training_arguments(training_arguments, run_dir, random_state)

    manifest = {
        "split_dir": str(split_dir),
        "data_manifest": data_manifest,
        "mode": mode,
        "folds": rotations,
        "model": {
            "base_model": base_model,
            "architecture": architecture_name(architecture),
            "architecture_kwargs": architecture_kwargs or {},
        },
        "encoding": describe_encoder(encoder),
        "evaluation": {"min_overlap_percentage": min_overlap_percentage},
        "training_arguments": resolved_arguments.to_dict(),
        "hardware": gpu_hardware_info(),
    }
    manifest_path = write_manifest(manifest, run_dir / RUN_MANIFEST_FILENAME)

    encoded: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    paths: dict[str, Path] = {"run_manifest": manifest_path}

    for validation_index in rotations:
        partitions = split_paths(split_dir, validation_index=validation_index)
        fold_dir = run_dir if validation_index is None else run_dir / f"fold_{validation_index:02d}"

        train_rows = encode_partition(encoder, partitions["train"], encoded)
        validation_rows = encode_partition(encoder, partitions["validation"], encoded)

        set_seed(resolved_arguments.seed)

        model = build_model(
            base_model=base_model,
            label2id=encoder.label2id,
            id2label=encoder.id2label,
            architecture=architecture,
            **(architecture_kwargs or {}),
        )

        result = train(
            model=model,
            tokenizer=tokenizer,
            train_rows=train_rows,
            validation_rows=validation_rows,
            training_arguments=dataclasses.replace(
                resolved_arguments, output_dir=str(fold_dir)
            ),
            compute_metrics=build_compute_metrics(
                rows=validation_rows,
                tokenizer=tokenizer,
                id2label=encoder.id2label,
                min_overlap_percentage=min_overlap_percentage,
                include_confusion=include_confusion,
            ),
            train_compute_metrics=(
                build_compute_metrics(
                    rows=train_rows,
                    tokenizer=tokenizer,
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
            run_metadata=split_provenance(validation_index, partitions),
            model_encoding=model_encoding(
                encoder=encoder,
                base_model=base_model,
                architecture=architecture,
                architecture_kwargs=architecture_kwargs,
            ),
        )

        summaries.append(result.summary)
        rows.append(
            fold_row(
                validation_index=validation_index,
                summary=result.summary,
                epoch_metrics=result.epoch_metrics,
                greater_is_better=bool(resolved_arguments.greater_is_better),
            )
        )
        paths[fold_directory_key(validation_index)] = result.output_dir

        del result, model
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
    paths["assessment_summary"] = write_manifest(
        summary, run_dir / ASSESSMENT_SUMMARY_FILENAME
    )

    return AssessmentResult(
        run_dir=run_dir,
        mode=mode,
        fold_metrics=fold_metrics,
        aggregate=aggregate,
        summaries=summaries,
        manifest=manifest,
        paths=paths,
    )


def read_data_manifest(split_dir: str | Path) -> dict[str, Any]:
    """Read the `data_manifest.json` that `prepare_dataset` wrote beside a split."""
    manifest_path = Path(split_dir) / DATA_MANIFEST_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No {DATA_MANIFEST_FILENAME} in {split_dir} — expected the output directory of "
            "ner_lab.data.prepare_dataset."
        )

    return read_manifest(manifest_path)


def fold_rotations(
    data_manifest: dict[str, Any],
    folds: Sequence[int] | None = None,
) -> list[int | None]:
    """
    Which validation folds a split calls for: `[None]` for a plain train/validation split.

    With a fixed holdout, every non-holdout fold takes a turn as validation
    unless `folds` narrows it.
    """
    split = data_manifest.get("split", {})
    mode = split.get("mode")

    if mode == "train_validation":
        if folds is not None:
            raise ValueError(
                "folds only applies to a fixed_holdout_kfold split; this one is "
                "train_validation, which has a single rotation."
            )

        return [None]

    if mode != "fixed_holdout_kfold":
        raise ValueError(f"Unsupported split mode {mode!r}; expected one of {SPLIT_MODES}.")

    n_splits = int(split["n_splits"])
    holdout_fold = int(split["holdout_fold"])
    rotatable = [index for index in range(n_splits) if index != holdout_fold]

    if folds is None:
        return rotatable

    requested = [int(fold) for fold in folds]
    unknown = sorted(set(requested) - set(rotatable))

    if unknown:
        raise ValueError(
            f"folds={unknown} are not rotatable validation folds; this split has "
            f"{n_splits} folds with {holdout_fold} reserved as the fixed holdout."
        )

    return requested


def run_directory_name(
    target_label: str,
    architecture: str | Architecture,
    base_model: str,
    timestamp: str | None = None,
) -> str:
    """The directory one assessment writes into: what was trained, on what, when."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{target_label}__{architecture_name(architecture)}__{Path(base_model).name}__{stamp}"


def architecture_name(architecture: str | Architecture) -> str:
    """A recordable name for a builtin architecture or a user-supplied callable."""
    if isinstance(architecture, str):
        return architecture

    return getattr(architecture, "__name__", type(architecture).__name__)


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
    return {
        "model": {
            "base_model": base_model,
            "architecture": architecture_name(architecture),
            "architecture_kwargs": architecture_kwargs or {},
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


def encode_partition(
    encoder: Encoder,
    path_or_paths: Path | list[Path],
    cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Encode one partition, or concatenate several into one training frame.

    `cache` keeps each fold parquet's rows keyed by path, so a fold that is train
    in one rotation and validation in another is encoded once.
    """
    paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    frames = []

    for path in paths:
        key = str(path)

        if cache is None:
            frames.append(encoder.encode_parquet(path))
        else:
            if key not in cache:
                cache[key] = encoder.encode_parquet(path)

            frames.append(cache[key])

    if len(frames) == 1:
        return frames[0].reset_index(drop=True)

    return pd.concat(frames, ignore_index=True)


def split_provenance(
    validation_index: int | None,
    partitions: dict[str, Path | list[Path]],
) -> dict[str, Any]:
    """What this run actually trained and validated on, for its `training_summary.json`."""
    train = partitions["train"]

    return {
        "validation_fold": validation_index,
        "train_parquet": (
            [str(path) for path in train] if isinstance(train, list) else str(train)
        ),
        "validation_parquet": str(partitions["validation"]),
    }


def best_epoch_metrics(
    epoch_metrics: pd.DataFrame,
    metric: str,
    greater_is_better: bool = True,
) -> dict[str, Any]:
    """
    The evaluation row of whichever epoch scored best on `metric`.

    `EpochMetricsLogger` strips the `eval_` prefix the Trainer adds, so a
    `metric_for_best_model` given either way resolves to the same column.
    """
    if epoch_metrics.empty:
        return {}

    evaluated = epoch_metrics
    if "split" in epoch_metrics.columns:
        evaluated = epoch_metrics[epoch_metrics["split"] == "eval"]

    column = metric[len("eval_"):] if metric.startswith("eval_") else metric

    if evaluated.empty or column not in evaluated.columns:
        return {}

    index = evaluated[column].idxmax() if greater_is_better else evaluated[column].idxmin()

    return {
        name: value
        for name, value in evaluated.loc[index].to_dict().items()
        if name != "split"
    }


def fold_row(
    validation_index: int | None,
    summary: dict[str, Any],
    epoch_metrics: pd.DataFrame,
    greater_is_better: bool = True,
) -> dict[str, Any]:
    """
    One row of `fold_metrics.parquet`: the selection metric, the epoch behind it, and cost.

    `best_metric` is stored under its own fixed key rather than looked up by
    name, so aggregating across folds never has to guess the Trainer's prefix.
    """
    best = summary.get("best", {})
    metric = best.get("metric") or ""

    row: dict[str, Any] = {
        "validation_fold": validation_index,
        "best_metric": best.get("value"),
        "best_epoch": best.get("epoch"),
    }
    row.update(best_epoch_metrics(epoch_metrics, metric, greater_is_better))
    row.update(summary.get("resources") or {})

    return row


def aggregate_metrics(
    frame: pd.DataFrame,
    exclude: Sequence[str] = (),
) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of every numeric column, across runs."""
    if frame.empty:
        return {}

    excluded = set(exclude)
    numeric = frame.select_dtypes(include="number")

    return {
        column: {
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std()) if len(numeric) > 1 else 0.0,
        }
        for column in numeric.columns
        if column not in excluded
    }


def write_fold_metrics(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the per-run metric table to parquet."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(metrics_path, index=False)

    return metrics_path


def release_accelerator_memory() -> None:
    """Drop the previous run's model from the allocator before the next one is built."""
    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fold_directory_key(validation_index: int | None) -> str:
    """The `paths` key an individual run's directory is recorded under."""
    return "run" if validation_index is None else f"fold_{validation_index:02d}"
