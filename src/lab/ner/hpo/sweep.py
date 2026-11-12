"""Running a sweep over encoded variants with Ray Tune + Optuna, whatever the model."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import TrainingArguments

from lab.core.provenance import write_manifest
from lab.core.split import split_paths
from lab.ner.encoding.encoder import WindowStrategy
from lab.ner.hpo.arguments import resolve_base_arguments
from lab.ner.hpo.progress import SweepProgress
from lab.ner.hpo.report import summarize_sweep
from lab.ner.hpo.space import build_search_space, describe_search_space, smoke_configuration
from lab.ner.hpo.trial import metric_greater_is_better, run_trial, validate_search_space
from lab.ner.hpo.trials import (
    TRIALS_SUMMARY_FILENAME,
    best_trial,
    trial_directory_name,
    trials_table,
    write_trials_table,
)
from lab.ner.hpo.variants import EncodedVariant, build_variants, describe_variants
from lab.ner.hpo.winner import WINNER_CONFIG_FILENAME, write_winner_config
from lab.ner.models.capacity import check_max_length
from lab.ner.training.runs import fold_rotations, read_data_manifest
from lab.ner.training.tracking import gpu_hardware_info

HPO_SUMMARY_FILENAME = "hpo_summary.json"
TUNE_VERBOSITY = 0


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


@dataclass(frozen=True)
class SearchSetup:
    """What every sweep settles before it encodes anything: its split, variants and objective."""

    split_dir: Path
    data_manifest: dict[str, Any]
    validation_index: int | None
    partitions: dict[str, Path | list[Path]]
    base_models: list[str]
    variants: dict[str, dict[str, Any]]
    space: dict[str, Any]
    full_space: dict[str, Any]
    base_arguments: TrainingArguments
    metric_key: str
    greater_is_better: bool


def prepare_search(
    split_dir: str | Path,
    base_models: str | Sequence[str],
    strategies: Sequence[str | WindowStrategy],
    context_tokens: Sequence[int] | None,
    max_lengths: Sequence[int],
    search_space: Mapping[str, Any] | None,
    training_arguments: TrainingArguments | Mapping[str, Any] | None,
    validation_index: int | None,
    random_state: int | None,
    run_dir: Path,
) -> SearchSetup:
    """
    Settle everything a sweep needs before encoding, whatever kind of model it searches.

    Resolves the searched rotation of the split, the variants, the validated search
    space with its `variant` dimension, and the base arguments and the objective they
    name. Ray is not started yet, so a setting that fails here costs nothing.
    """
    from ray import tune

    split_dir = Path(split_dir).resolve()
    data_manifest = read_data_manifest(split_dir)
    requested = None if validation_index is None else [validation_index]
    validation_index = fold_rotations(data_manifest, requested)[0]

    base_models = [base_models] if isinstance(base_models, str) else list(base_models)
    variants = build_variants(base_models, strategies, context_tokens, max_lengths)
    space = _validated_search_space(search_space)

    base_arguments = resolve_base_arguments(training_arguments, run_dir, random_state)
    metric_key, greater_is_better = _objective(base_arguments)

    return SearchSetup(
        split_dir=split_dir,
        data_manifest=data_manifest,
        validation_index=validation_index,
        partitions=split_paths(split_dir, validation_index=validation_index),
        base_models=base_models,
        variants=variants,
        space=space,
        full_space={**space, "variant": tune.choice(list(variants))},
        base_arguments=base_arguments,
        metric_key=metric_key,
        greater_is_better=greater_is_better,
    )


def start_ray() -> int:
    """
    Start Ray unless it is running, and return the GPUs one trial reserves.

    One when Ray reports any GPU, none otherwise: a trial spanning two would train at
    twice the batch size it was scored on, so parallelism comes from running trials
    side by side.
    """
    import ray

    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    return 1 if ray.cluster_resources().get("GPU", 0) else 0


def check_variant_lengths(
    variants: dict[str, dict[str, Any]],
    build_for: Callable[..., Any],
    report: bool = True,
) -> None:
    """
    Run every base model once on CPU at each of its variants' `max_length`s.

    `build_for(base_model=...)` builds the searched kind of model over one base model.
    A window longer than a model's positions then raises naming the length, before Ray
    starts.
    """
    from transformers import AutoTokenizer

    lengths_by_model: dict[str, set[int]] = {}

    for specification in variants.values():
        lengths = lengths_by_model.setdefault(specification["base_model"], set())
        lengths.add(specification["max_length"])

    for base_model, max_lengths in lengths_by_model.items():
        model = build_for(base_model=base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        for max_length in sorted(max_lengths):
            check_max_length(model, tokenizer, max_length, base_model, report=report)


def sweep_manifest(
    setup: SearchSetup,
    gpus_per_trial: int,
    n_trials: int,
    seeds_per_trial: int,
    top_k_epochs: int,
    max_micro_batch_size: int,
    min_overlap_percentage: float,
) -> dict[str, Any]:
    """The run manifest every sweep writes; the caller adds its `model` and `encoding` blocks."""
    return {
        "split_dir": str(setup.split_dir),
        "data_manifest": setup.data_manifest,
        "validation_index": setup.validation_index,
        "variants": describe_variants(setup.variants),
        "search_space": describe_search_space(setup.full_space),
        "objective": {
            "metric": setup.metric_key,
            "greater_is_better": setup.greater_is_better,
            "min_overlap_percentage": min_overlap_percentage,
        },
        "n_trials": n_trials,
        "seeds_per_trial": seeds_per_trial,
        "top_k_epochs": top_k_epochs,
        "max_micro_batch_size": max_micro_batch_size,
        "gpus_per_trial": gpus_per_trial,
        "training_arguments": setup.base_arguments.to_dict(),
        "hardware": gpu_hardware_info(),
    }


def run_search(
    setup: SearchSetup,
    encoded: dict[str, EncodedVariant],
    gpus_per_trial: int,
    run_dir: Path,
    manifest: dict[str, Any],
    paths: dict[str, Path],
    n_trials: int,
    seeds_per_trial: int,
    top_k_epochs: int,
    max_micro_batch_size: int,
    early_stopping_patience: int | None,
    pad_to_multiple_of: int | None,
    track_resources: bool,
    study_name: str | None,
    storage: str | None,
    smoke_test: bool,
    report: bool,
    sweep_start: float,
    winner_task: str,
    winner: Callable[[Any], dict[str, Any]],
) -> HPOResult:
    """
    Run a sweep over encoded variants and write what it found.

    The part of a sweep that does not depend on the kind of model: the smoke test,
    the Optuna search under Ray, the trials table, the summary, the winner config and
    the report. `winner` turns the best trial into a config for `winner_task`, the
    task that retrains it, recorded under that name in `hpo_summary.json`.
    """
    trial_settings = {
        "variants": encoded,
        "max_micro_batch_size": max_micro_batch_size,
        "top_k_epochs": top_k_epochs,
        "early_stopping_patience": early_stopping_patience,
        "pad_to_multiple_of": pad_to_multiple_of,
        "track_resources": track_resources,
    }
    metric_key, greater_is_better = setup.metric_key, setup.greater_is_better

    if smoke_test:
        _run_smoke_test(
            trainable=_trainable(
                trial_settings,
                arguments=dataclasses.replace(setup.base_arguments, num_train_epochs=1),
                seeds_per_trial=1,
                gpus_per_trial=gpus_per_trial,
            ),
            space=setup.space,
            variants=setup.variants,
            run_dir=run_dir,
            callbacks=_progress_callbacks(
                report, metric_key, greater_is_better, total=len(setup.variants), stage="smoke"
            ),
        )

    result_grid = _run_sweep(
        trainable=_trainable(
            trial_settings,
            arguments=setup.base_arguments,
            seeds_per_trial=seeds_per_trial,
            gpus_per_trial=gpus_per_trial,
        ),
        full_space=setup.full_space,
        n_trials=n_trials,
        metric_key=metric_key,
        greater_is_better=greater_is_better,
        seed=setup.base_arguments.seed,
        study_name=study_name,
        storage=storage,
        run_dir=run_dir,
        callbacks=_progress_callbacks(
            report, metric_key, greater_is_better, total=n_trials, stage="trial"
        ),
    )

    trials = trials_table(result_grid, metric_key)
    paths["trials_summary"] = write_trials_table(trials, run_dir / TRIALS_SUMMARY_FILENAME)

    best = best_trial(result_grid, metric_key, greater_is_better)
    configuration = None if best is None else winner(best)

    summary = _sweep_summary(
        result_grid, best, metric_key, greater_is_better, winner_task, configuration
    )
    summary["total_wall_time_sec"] = time.monotonic() - sweep_start
    paths["hpo_summary"] = write_manifest(summary, run_dir / HPO_SUMMARY_FILENAME)

    if configuration is not None:
        paths["winner"] = write_winner_config(configuration, run_dir / WINNER_CONFIG_FILENAME)

    sweep_report = summarize_sweep(
        trials=trials,
        metric=metric_key,
        greater_is_better=greater_is_better,
        search_space=manifest["search_space"],
        epoch_cap=int(setup.base_arguments.num_train_epochs),
    )

    if report:
        print(sweep_report)

    return HPOResult(
        run_dir=run_dir,
        metric=metric_key,
        summary=summary,
        best=configuration,
        trials=trials,
        manifest=manifest,
        paths=paths,
        report=sweep_report,
    )


def _validated_search_space(search_space: Mapping[str, Any] | None) -> dict[str, Any]:
    space = build_search_space(search_space)

    if "variant" in space:
        raise ValueError(
            "variant is a reserved search dimension, built from base_models/strategies/"
            "context_tokens/max_lengths."
        )

    validate_search_space(space)

    return space


def _objective(base_arguments: TrainingArguments) -> tuple[str, bool]:
    metric = base_arguments.metric_for_best_model

    if not metric:
        raise ValueError(
            "training_arguments must set metric_for_best_model — it is the sweep's objective."
        )

    metric_key = metric.removeprefix("eval_")

    return metric_key, metric_greater_is_better(base_arguments)


def _trainable(
    trial_settings: dict[str, Any],
    arguments: TrainingArguments,
    seeds_per_trial: int,
    gpus_per_trial: int,
) -> Any:
    from ray import tune

    trainable = tune.with_parameters(
        _reported_trial,
        base_arguments=arguments,
        seeds_per_trial=seeds_per_trial,
        **trial_settings,
    )

    if gpus_per_trial:
        trainable = tune.with_resources(trainable, {"gpu": gpus_per_trial})

    return trainable


def _progress_callbacks(
    report: bool, metric_key: str, greater_is_better: bool, total: int, stage: str
) -> list[SweepProgress]:
    if not report:
        return []

    return [
        SweepProgress(
            metric=metric_key, greater_is_better=greater_is_better, total=total, stage=stage
        )
    ]


def _run_smoke_test(
    trainable: Any,
    space: dict[str, Any],
    variants: dict[str, Any],
    run_dir: Path,
    callbacks: list[SweepProgress],
) -> None:
    from ray import tune

    smoke_result = tune.Tuner(
        trainable,
        param_space={**smoke_configuration(space), "variant": tune.grid_search(list(variants))},
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=tune.RunConfig(
            name="smoke_test",
            storage_path=str(run_dir),
            verbose=TUNE_VERBOSITY,
            callbacks=callbacks,
        ),
    ).fit()

    if smoke_result.num_errors:
        raise RuntimeError(
            "Pre-flight smoke test failed, aborting before the real sweep: "
            f"{smoke_result.errors[0]}"
        )


def _run_sweep(
    trainable: Any,
    full_space: dict[str, Any],
    n_trials: int,
    metric_key: str,
    greater_is_better: bool,
    seed: int,
    study_name: str | None,
    storage: str | None,
    run_dir: Path,
    callbacks: list[SweepProgress],
) -> Any:
    from ray import tune
    from ray.air import FailureConfig
    from ray.tune.search.optuna import OptunaSearch

    search_algorithm = OptunaSearch(
        metric=metric_key,
        mode="max" if greater_is_better else "min",
        seed=seed,
        study_name=study_name,
        storage=storage,
    )

    return tune.Tuner(
        trainable,
        param_space=full_space,
        tune_config=tune.TuneConfig(
            search_alg=search_algorithm,
            num_samples=n_trials,
            trial_dirname_creator=trial_directory_name,
        ),
        run_config=tune.RunConfig(
            name="trials",
            storage_path=str(run_dir),
            verbose=TUNE_VERBOSITY,
            callbacks=callbacks,
            failure_config=FailureConfig(max_failures=0, fail_fast=True),
        ),
    ).fit()


def _sweep_summary(
    result_grid: Any,
    best: Any | None,
    metric_key: str,
    greater_is_better: bool,
    winner_task: str,
    winner: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "success": best is not None,
        "metric": metric_key,
        "greater_is_better": greater_is_better,
        "n_trials": len(result_grid),
        "num_errors": result_grid.num_errors,
    }

    if best is None:
        summary["error"] = "No successful trial was completed."

        return summary

    summary.update(
        {
            "best_metric": best.metrics.get(metric_key),
            "best_metric_std": best.metrics.get(f"{metric_key}_std"),
            "n_seeds": best.metrics.get("n_seeds"),
            "per_seed_scores": best.metrics.get("per_seed_scores"),
            "best_trial_path": best.path,
            "sampled": dict(best.config),
            winner_task: winner,
        }
    )

    return summary


def _reported_trial(
    tune_config: dict[str, Any],
    variants: dict[str, EncodedVariant],
    base_arguments: TrainingArguments,
    max_micro_batch_size: int,
    seeds_per_trial: int,
    top_k_epochs: int,
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
            max_micro_batch_size=max_micro_batch_size,
            seeds_per_trial=seeds_per_trial,
            top_k_epochs=top_k_epochs,
            early_stopping_patience=early_stopping_patience,
            pad_to_multiple_of=pad_to_multiple_of,
            track_resources=track_resources,
        )
    )
