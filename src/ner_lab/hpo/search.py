"""Searching hyperparameters with Ray Tune + Optuna over a prepared split."""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import TrainingArguments

from ner_lab.encoding.encoder import WindowStrategy
from ner_lab.encoding.overlaps import OverlapPolicy
from ner_lab.data.split import split_paths
from ner_lab.hpo.report import summarize_sweep
from ner_lab.hpo.space import (
    DEFAULT_SEARCH_SPACE,
    build_search_space,
    describe_search_space,
    smoke_configuration,
)
from ner_lab.hpo.trial import (
    RESERVED_DIMENSIONS,
    metric_greater_is_better,
    run_trial,
    validate_search_space,
)
from ner_lab.hpo.variants import (
    EncodedVariant,
    build_variants,
    describe_variants,
    encode_variants,
)
from ner_lab.models.registry import Architecture
from ner_lab.provenance import write_manifest
from ner_lab.training.arguments import training_arguments as build_training_arguments
from ner_lab.training.assessment import (
    RUN_MANIFEST_FILENAME,
    architecture_name,
    fold_rotations,
    read_data_manifest,
    run_directory_name,
)
from ner_lab.training.tracking import gpu_hardware_info

TRIALS_SUMMARY_FILENAME = "trials_summary.parquet"
HPO_SUMMARY_FILENAME = "hpo_summary.json"

HPO_ARGUMENT_DEFAULTS: dict[str, Any] = {
    "num_train_epochs": 40,
    "save_strategy": "no",
    "load_best_model_at_end": False,
}

DERIVED_ARGUMENT_KEYS = ("output_dir", "run_name", "logging_dir")

RESOURCE_COLUMNS = (
    "duration_sec",
    "energy_kwh",
    "emissions_kg_co2",
    "cpu_utilization_percent",
    "gpu_utilization_percent",
    "ram_used_gb",
    "gpu_peak_vram_allocated_gb",
    "gpu_peak_vram_reserved_gb",
)


@dataclass(frozen=True)
class HPOResult:
    """What one sweep produced: the winner, every trial's record, and where they landed."""

    run_dir: Path
    metric: str
    summary: dict[str, Any]
    best: dict[str, Any] | None
    trials: pd.DataFrame
    manifest: dict[str, Any]
    paths: dict[str, Path]
    report: str


def search_hyperparameters(
    split_dir: str | Path,
    output_dir: str | Path,
    checkpoints: str | Sequence[str],
    target_label: str,
    language: str,
    architecture: str | Architecture = "linear",
    architecture_kwargs: dict[str, Any] | None = None,
    search_space: Mapping[str, Any] | None = None,
    training_arguments: TrainingArguments | Mapping[str, Any] | None = None,
    n_trials: int = 40,
    seeds_per_trial: int = 5,
    top_k_epochs: int = 3,
    strategies: Sequence[str | WindowStrategy] = ("greedy",),
    context_tokens: Sequence[int] | None = None,
    max_lengths: Sequence[int] = (256,),
    overlap_policy: OverlapPolicy = "merge_same_label_then_keep_longest",
    min_sentence_tokens: int = 4,
    min_overlap_percentage: float = 40.0,
    early_stopping_patience: int | None = 5,
    pad_to_multiple_of: int | None = 8,
    max_micro_batch_size: int = 64,
    gpus_per_trial: float = 1.0,
    validation_index: int | None = None,
    random_state: int | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    smoke_test: bool = True,
    track_resources: bool = True,
    report: bool = True,
    run_name: str | None = None,
) -> HPOResult:
    """
    Search hyperparameters for one configuration family against a prepared split.

    The search space is `DEFAULT_SEARCH_SPACE` overridden by `search_space` —
    see `build_search_space` for the accepted forms. On top of it, every
    (checkpoint, strategy, context_tokens, max_length) combination becomes one
    `variant` dimension, windowed once before any trial starts and shared across
    all of them.

    A trial trains its sampled configuration once per seed with no checkpointing,
    scoring the mean across seeds of each seed's mean-of-top-k epochs on the
    metric `training_arguments` selects. Out-of-memory trials score worst instead
    of failing; any other error halts the sweep immediately.

    `smoke_test` runs one epoch per variant first, so an unloadable checkpoint or a
    search dimension the training arguments have no field for costs one epoch rather
    than the sweep's first parallel wave.

    `report` prints the end-of-sweep summary; it is on the result either way, as
    `HPOResult.report`.

    A k-fold split is searched against one rotation, `validation_index`
    (defaulting to the first rotatable fold) — searching across all folds would
    multiply an already GPU-day sweep by the fold count. The fixed holdout is
    never read.

    `hpo_summary.json` records the winner both raw and as a ready-to-run
    `train_model` YAML config; nothing ever reads it back.
    """
    import ray
    from ray import tune
    from ray.air import FailureConfig
    from ray.tune.search.optuna import OptunaSearch

    sweep_start = time.monotonic()

    split_dir = Path(split_dir).resolve()
    data_manifest = read_data_manifest(split_dir)
    requested = None if validation_index is None else [validation_index]
    validation_index = fold_rotations(data_manifest, requested)[0]
    partitions = split_paths(split_dir, validation_index=validation_index)

    checkpoints = [checkpoints] if isinstance(checkpoints, str) else list(checkpoints)
    checkpoint_slug = (
        Path(checkpoints[0]).name if len(checkpoints) == 1 else f"{len(checkpoints)}checkpoints"
    )

    run_dir = Path(output_dir).resolve() / (
        run_name or run_directory_name(target_label, architecture, checkpoint_slug)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variants(checkpoints, strategies, context_tokens, max_lengths)
    space = build_search_space(search_space)

    if "variant" in space:
        raise ValueError(
            "variant is a reserved search dimension, built from checkpoints/strategies/"
            "context_tokens/max_lengths."
        )

    validate_search_space(space)

    if gpus_per_trial > 1:
        raise ValueError(
            "gpus_per_trial must be at most 1. The searched effective_train_batch_size is "
            "per trial, but per_device_train_batch_size is per device, so extra GPUs "
            "multiply the batch a trial actually trains at and the winner stops being "
            "reproducible on one GPU. Run more trials in parallel instead."
        )

    base_arguments = resolve_base_arguments(training_arguments, run_dir, random_state)
    metric = base_arguments.metric_for_best_model

    if not metric:
        raise ValueError(
            "training_arguments must set metric_for_best_model — it is the sweep's objective."
        )

    metric_key = metric[len("eval_"):] if metric.startswith("eval_") else metric
    greater_is_better = metric_greater_is_better(base_arguments)

    full_space = {**space, "variant": tune.choice(list(variants))}

    manifest = {
        "split_dir": str(split_dir),
        "data_manifest": data_manifest,
        "validation_index": validation_index,
        "model": {
            "checkpoints": checkpoints,
            "architecture": architecture_name(architecture),
            "architecture_kwargs": architecture_kwargs or {},
        },
        "encoding": {
            "target_label": target_label,
            "language": language,
            "overlap_policy": overlap_policy,
            "min_sentence_tokens": min_sentence_tokens,
        },
        "variants": describe_variants(variants),
        "search_space": describe_search_space(full_space),
        "objective": {
            "metric": metric_key,
            "greater_is_better": greater_is_better,
            "min_overlap_percentage": min_overlap_percentage,
        },
        "n_trials": n_trials,
        "seeds_per_trial": seeds_per_trial,
        "top_k_epochs": top_k_epochs,
        "max_micro_batch_size": max_micro_batch_size,
        "gpus_per_trial": gpus_per_trial,
        "training_arguments": base_arguments.to_dict(),
        "hardware": gpu_hardware_info(),
    }
    manifest_path = write_manifest(manifest, run_dir / RUN_MANIFEST_FILENAME)
    paths: dict[str, Path] = {"run_manifest": manifest_path}

    encoded = encode_variants(
        partitions=partitions,
        variants=variants,
        target_label=target_label,
        language=language,
        overlap_policy=overlap_policy,
        min_sentence_tokens=min_sentence_tokens,
    )

    if not ray.is_initialized():
        ray.init()

    def build_trainable(seeds: int, arguments: TrainingArguments = base_arguments) -> Any:
        trainable = tune.with_parameters(
            _reported_trial,
            variants=encoded,
            base_arguments=arguments,
            architecture=architecture,
            architecture_kwargs=architecture_kwargs,
            max_micro_batch_size=max_micro_batch_size,
            seeds_per_trial=seeds,
            top_k_epochs=top_k_epochs,
            min_overlap_percentage=min_overlap_percentage,
            early_stopping_patience=early_stopping_patience,
            pad_to_multiple_of=pad_to_multiple_of,
            track_resources=track_resources,
        )

        if gpus_per_trial:
            trainable = tune.with_resources(trainable, {"gpu": gpus_per_trial})

        return trainable

    if smoke_test:
        smoke_result = tune.Tuner(
            build_trainable(
                seeds=1, arguments=dataclasses.replace(base_arguments, num_train_epochs=1)
            ),
            param_space={
                **smoke_configuration(space),
                "variant": tune.grid_search(list(variants)),
            },
            tune_config=tune.TuneConfig(num_samples=1),
            run_config=tune.RunConfig(name="smoke_test", storage_path=str(run_dir)),
        ).fit()

        if smoke_result.num_errors:
            raise RuntimeError(
                "Pre-flight smoke test failed, aborting before the real sweep: "
                f"{smoke_result.errors[0]}"
            )

    search_algorithm = OptunaSearch(
        metric=metric_key,
        mode="max" if greater_is_better else "min",
        seed=base_arguments.seed,
        study_name=study_name,
        storage=storage,
    )

    result_grid = tune.Tuner(
        build_trainable(seeds=seeds_per_trial),
        param_space=full_space,
        tune_config=tune.TuneConfig(
            search_alg=search_algorithm,
            num_samples=n_trials,
            trial_dirname_creator=trial_directory_name,
        ),
        run_config=tune.RunConfig(
            name="trials",
            storage_path=str(run_dir),
            failure_config=FailureConfig(max_failures=0, fail_fast=True),
        ),
    ).fit()

    trials = trials_table(result_grid, metric_key)
    paths["trials_summary"] = write_trials_table(trials, run_dir / TRIALS_SUMMARY_FILENAME)

    best = best_trial(result_grid, metric_key, greater_is_better)

    summary: dict[str, Any] = {
        "success": best is not None,
        "metric": metric_key,
        "greater_is_better": greater_is_better,
        "n_trials": len(result_grid),
        "num_errors": result_grid.num_errors,
    }

    if best is None:
        summary["error"] = "No successful trial was completed."
    else:
        summary.update(
            {
                "best_metric": best.metrics.get(metric_key),
                "best_metric_std": best.metrics.get(f"{metric_key}_std"),
                "n_seeds": best.metrics.get("n_seeds"),
                "per_seed_scores": best.metrics.get("per_seed_scores"),
                "best_trial_path": best.path,
                "sampled": dict(best.config),
                "train_model": winner_configuration(
                    split_dir=split_dir,
                    output_dir=output_dir,
                    variant=variants[best.config["variant"]],
                    sampled=best.config,
                    micro_batch_size=best.metrics.get("per_device_train_batch_size"),
                    accumulation_steps=best.metrics.get("gradient_accumulation_steps"),
                    target_label=target_label,
                    language=language,
                    architecture=architecture,
                    architecture_kwargs=architecture_kwargs,
                    overlap_policy=overlap_policy,
                    min_sentence_tokens=min_sentence_tokens,
                    min_overlap_percentage=min_overlap_percentage,
                    early_stopping_patience=early_stopping_patience,
                    base_arguments=base_arguments,
                    user_overrides=user_argument_overrides(training_arguments),
                ),
            }
        )

    summary["total_wall_time_sec"] = time.monotonic() - sweep_start
    paths["hpo_summary"] = write_manifest(summary, run_dir / HPO_SUMMARY_FILENAME)

    sweep_report = summarize_sweep(
        trials=trials,
        metric=metric_key,
        greater_is_better=greater_is_better,
        search_space=manifest["search_space"],
        epoch_cap=int(base_arguments.num_train_epochs),
    )

    if report:
        print(sweep_report)

    return HPOResult(
        run_dir=run_dir,
        metric=metric_key,
        summary=summary,
        best=None if best is None else summary["train_model"],
        trials=trials,
        manifest=manifest,
        paths=paths,
        report=sweep_report,
    )


def user_argument_overrides(
    arguments: TrainingArguments | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    What the caller changed from this library's defaults, for the winner block.

    A mapping already says exactly that. An instance has every field populated, so
    the caller's intent is only recoverable by diffing it against a default-built
    one — without which the winner block would silently fall back to the defaults
    for everything the sweep was actually run with.
    """
    if arguments is None:
        return {}

    if not isinstance(arguments, TrainingArguments):
        return dict(arguments)

    baseline = build_training_arguments(arguments.output_dir).to_dict()
    current = arguments.to_dict()

    return {
        key: value
        for key, value in current.items()
        if not key.startswith("_")
        and key not in DERIVED_ARGUMENT_KEYS
        and baseline.get(key) != value
    }


def resolve_base_arguments(
    arguments: TrainingArguments | Mapping[str, Any] | None,
    output_dir: str | Path,
    random_state: int | None = None,
) -> TrainingArguments:
    """
    The `TrainingArguments` every trial starts from.

    A mapping is applied over the library defaults plus `HPO_ARGUMENT_DEFAULTS` — a
    high epoch cap with no checkpointing, since the trial score is read from the epoch
    metrics and the weights are discarded. An instance is used as given, except for
    those same sweep defaults, which are forced: every field of an instance is
    populated, so a deliberate `num_train_epochs=10` cannot be told apart from the
    library default of the same value. Pass a mapping to override them per key.
    """
    overrides: dict[str, Any] = {} if random_state is None else {"seed": random_state}

    if isinstance(arguments, TrainingArguments):
        return dataclasses.replace(
            arguments, output_dir=str(output_dir), **HPO_ARGUMENT_DEFAULTS, **overrides
        )

    merged = {**HPO_ARGUMENT_DEFAULTS, **(dict(arguments) if arguments else {})}

    return build_training_arguments(output_dir, **{**merged, **overrides})


def trial_directory_name(trial: Any) -> str:
    """A readable per-trial directory: the id plus the dimensions that vary most."""
    configuration = trial.config

    return (
        f"trial{trial.trial_id}"
        f"_lr{configuration.get('learning_rate', 0):.1e}"
        f"_bs{configuration.get('effective_train_batch_size', 0)}"
        f"_wd{configuration.get('weight_decay', 0):.3f}"
    )


def trials_table(result_grid: Any, metric_key: str) -> pd.DataFrame:
    """
    One row per trial: sampled values, score and spread, batch resolution, resources.

    Everything needed to compare trials without opening each trial's own
    artifacts individually.
    """
    reported_columns = [
        metric_key,
        f"{metric_key}_std",
        "n_seeds",
        "per_seed_scores",
        "oom",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        *RESOURCE_COLUMNS,
    ]

    rows: list[dict[str, Any]] = []

    for result in result_grid:
        metrics = result.metrics or {}
        row: dict[str, Any] = {
            "trial_id": metrics.get("trial_id"),
            "trial_path": result.path,
            "error": str(result.error) if result.error else None,
        }
        row.update(result.config or {})

        for column in reported_columns:
            row[column] = metrics.get(column)

        rows.append(row)

    return pd.DataFrame(rows)


def write_trials_table(trials: pd.DataFrame, path: str | Path) -> Path:
    """Write the per-trial table to parquet."""
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    trials.to_parquet(table_path, index=False)

    return table_path


def best_trial(result_grid: Any, metric_key: str, greater_is_better: bool = True) -> Any | None:
    """The finished trial with the best finite score, or None when there is none."""
    scored = []

    for result in result_grid:
        value = (result.metrics or {}).get(metric_key)

        if value is not None and math.isfinite(value):
            scored.append((value, result))

    if not scored:
        return None

    scored.sort(key=lambda pair: pair[0], reverse=greater_is_better)

    return scored[0][1]


def winner_configuration(
    split_dir: str | Path,
    output_dir: str | Path,
    variant: dict[str, Any],
    sampled: dict[str, Any],
    micro_batch_size: int | None,
    accumulation_steps: int | None,
    target_label: str,
    language: str,
    architecture: str | Architecture,
    architecture_kwargs: dict[str, Any] | None,
    overlap_policy: str,
    min_sentence_tokens: int,
    min_overlap_percentage: float,
    early_stopping_patience: int | None,
    base_arguments: TrainingArguments,
    user_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    The winning trial as a ready-to-run `train_model` YAML config.

    An output only — `train_model` never reads it back, so no file format enters
    the API. Sweep-only settings (no checkpointing) are dropped; everything else
    the user overrode for the sweep is carried through, as reported by
    `user_argument_overrides`.
    """
    hyperparameters = {
        key: value for key, value in sampled.items() if key not in RESERVED_DIMENSIONS
    }
    carried = {
        key: value
        for key, value in (user_overrides or {}).items()
        if key not in ("save_strategy", "load_best_model_at_end")
    }

    configuration: dict[str, Any] = {
        "task": "train_model",
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "checkpoint": variant["checkpoint"],
        "target_label": target_label,
        "language": language,
        "architecture": architecture_name(architecture),
        "max_length": variant["max_length"],
        "strategy": (
            variant["strategy"]
            if isinstance(variant["strategy"], str)
            else architecture_name(variant["strategy"])
        ),
        "context_tokens": variant["context_tokens"],
        "overlap_policy": overlap_policy,
        "min_sentence_tokens": min_sentence_tokens,
        "min_overlap_percentage": min_overlap_percentage,
        "early_stopping_patience": early_stopping_patience,
        "training_arguments": {
            **carried,
            "num_train_epochs": base_arguments.num_train_epochs,
            **hyperparameters,
            "per_device_train_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
        },
    }

    if architecture_kwargs:
        configuration["architecture_kwargs"] = architecture_kwargs

    return configuration


def _reported_trial(
    tune_config: dict[str, Any],
    variants: dict[str, EncodedVariant],
    base_arguments: TrainingArguments,
    architecture: str | Architecture,
    architecture_kwargs: dict[str, Any] | None,
    max_micro_batch_size: int,
    seeds_per_trial: int,
    top_k_epochs: int,
    min_overlap_percentage: float,
    early_stopping_patience: int | None,
    pad_to_multiple_of: int | None,
    track_resources: bool,
) -> None:
    from ray import tune

    tune.report(
        run_trial(
            tune_config,
            variants=variants,
            base_arguments=base_arguments,
            trial_dir=tune.get_context().get_trial_dir(),
            architecture=architecture,
            architecture_kwargs=architecture_kwargs,
            max_micro_batch_size=max_micro_batch_size,
            seeds_per_trial=seeds_per_trial,
            top_k_epochs=top_k_epochs,
            min_overlap_percentage=min_overlap_percentage,
            early_stopping_patience=early_stopping_patience,
            pad_to_multiple_of=pad_to_multiple_of,
            track_resources=track_resources,
        )
    )
