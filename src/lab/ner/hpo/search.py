"""Searching hyperparameters with Ray Tune + Optuna over a prepared split."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import TrainingArguments

from lab.core.provenance import write_manifest
from lab.core.split import split_paths
from lab.ner.encoding.encoder import WindowStrategy
from lab.ner.encoding.overlaps import OverlapPolicy
from lab.ner.encoding.tagging import build_label_vocabulary
from lab.ner.hpo.arguments import resolve_base_arguments, user_argument_overrides
from lab.ner.hpo.progress import SweepProgress
from lab.ner.hpo.report import summarize_sweep
from lab.ner.hpo.space import build_search_space, describe_search_space, smoke_configuration
from lab.ner.hpo.trial import (
    metric_greater_is_better,
    run_trial,
    validate_search_space,
)
from lab.ner.hpo.trials import (
    TRIALS_SUMMARY_FILENAME,
    best_trial,
    optional_int,
    trial_directory_name,
    trials_table,
    write_trials_table,
)
from lab.ner.hpo.variants import (
    EncodedVariant,
    build_variants,
    describe_variants,
    encode_variants,
)
from lab.ner.hpo.winner import (
    FINAL_TRAIN_DIRNAME,
    WINNER_CONFIG_FILENAME,
    winner_configuration,
    write_winner_config,
)
from lab.ner.models.capacity import check_max_length
from lab.ner.models.registry import Architecture, architecture_name, build_model
from lab.ner.training.runs import (
    RUN_MANIFEST_FILENAME,
    claim_run_dir,
    fold_rotations,
    read_data_manifest,
)
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


def search_hyperparameters(
    split_dir: str | Path,
    output_dir: str | Path,
    base_models: str | Sequence[str],
    target_label: str,
    language: str,
    architecture: str | Architecture = "linear",
    architecture_kwargs: dict[str, Any] | None = None,
    search_space: Mapping[str, Any] | None = None,
    training_arguments: TrainingArguments | Mapping[str, Any] | None = None,
    n_trials: int = 40,
    seeds_per_trial: int = 3,
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
    validation_index: int | None = None,
    random_state: int | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    smoke_test: bool = True,
    track_resources: bool = True,
    report: bool = True,
    overwrite: bool = False,
) -> HPOResult:
    """
    Search hyperparameters for one configuration family against a prepared split.

    The search space is `DEFAULT_SEARCH_SPACE` overridden by `search_space` —
    see `build_search_space` for the accepted forms. On top of it, every
    (base_model, strategy, context_tokens, max_length) combination becomes one
    `variant` dimension, windowed once before any trial starts and shared across
    all of them.

    A trial trains its sampled configuration once per seed with no checkpointing,
    scoring the mean across seeds of each seed's mean-of-top-k epochs on the
    metric `training_arguments` selects. Out-of-memory trials score worst instead
    of failing; any other error halts the sweep immediately.

    A trial reserves one GPU when Ray reports any, and runs on CPU when it reports
    none. There is no knob for this: `per_device_train_batch_size` is per device, so
    a trial spanning two GPUs would train at twice the batch size it was scored on.
    Parallelism comes from running trials side by side.

    Before Ray starts, every base model is built on CPU and run once at each of its
    `max_lengths`, so a window longer than the model's positions raises naming the
    length, rather than as a CUDA device-side assert in the first trial. `report`
    prints a line per check passed.

    `smoke_test` runs one epoch per variant first, so an unloadable base model or a
    search dimension the training arguments have no field for costs one epoch rather
    than the sweep's first parallel wave.

    `report` prints two lines per trial while the sweep runs and the end-of-sweep
    summary when it finishes; the summary is on the result either way, as
    `HPOResult.report`. Ray Tune's own console output is off regardless, so
    `report=False` makes the sweep silent.

    A k-fold split is searched against one rotation, `validation_index`
    (defaulting to the first rotatable fold) — searching across all folds would
    multiply an already GPU-day sweep by the fold count. The fixed holdout is
    never read.

    `output_dir` is the sweep directory: the summary, the trials table, the winner
    config and Ray's per-trial tree all go into it directly, with nothing derived or
    nested on your behalf — join `run_directory_name()` yourself for a timestamped
    one. A directory another run already claimed raises unless `overwrite` is set,
    checked before anything is created so a refused sweep costs nothing.

    `hpo_summary.json` records the winner both raw and as a ready-to-run
    `train_model` YAML config pointed at `<output_dir>/final_train`; nothing ever
    reads it back.
    """
    import ray
    from ray import tune

    sweep_start = time.monotonic()

    run_dir = claim_run_dir(output_dir, overwrite)

    split_dir = Path(split_dir).resolve()
    data_manifest = read_data_manifest(split_dir)
    requested = None if validation_index is None else [validation_index]
    validation_index = fold_rotations(data_manifest, requested)[0]
    partitions = split_paths(split_dir, validation_index=validation_index)

    base_models = [base_models] if isinstance(base_models, str) else list(base_models)

    variants = build_variants(base_models, strategies, context_tokens, max_lengths)
    space = _validated_search_space(search_space)

    base_arguments = resolve_base_arguments(training_arguments, run_dir, random_state)
    metric_key, greater_is_better = _objective(base_arguments)

    _check_variant_lengths(variants, target_label, architecture, architecture_kwargs, report)

    full_space = {**space, "variant": tune.choice(list(variants))}

    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    gpus_per_trial = 1 if ray.cluster_resources().get("GPU", 0) else 0

    manifest = {
        "split_dir": str(split_dir),
        "data_manifest": data_manifest,
        "validation_index": validation_index,
        "model": {
            "base_models": base_models,
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

    trial_settings = {
        "variants": encode_variants(
            partitions=partitions,
            variants=variants,
            target_label=target_label,
            language=language,
            overlap_policy=overlap_policy,
            min_sentence_tokens=min_sentence_tokens,
        ),
        "architecture": architecture,
        "architecture_kwargs": architecture_kwargs,
        "max_micro_batch_size": max_micro_batch_size,
        "top_k_epochs": top_k_epochs,
        "min_overlap_percentage": min_overlap_percentage,
        "early_stopping_patience": early_stopping_patience,
        "pad_to_multiple_of": pad_to_multiple_of,
        "track_resources": track_resources,
    }

    if smoke_test:
        _run_smoke_test(
            trainable=_trainable(
                trial_settings,
                arguments=dataclasses.replace(base_arguments, num_train_epochs=1),
                seeds_per_trial=1,
                gpus_per_trial=gpus_per_trial,
            ),
            space=space,
            variants=variants,
            run_dir=run_dir,
            callbacks=_progress_callbacks(
                report, metric_key, greater_is_better, total=len(variants), stage="smoke"
            ),
        )

    result_grid = _run_sweep(
        trainable=_trainable(
            trial_settings,
            arguments=base_arguments,
            seeds_per_trial=seeds_per_trial,
            gpus_per_trial=gpus_per_trial,
        ),
        full_space=full_space,
        n_trials=n_trials,
        metric_key=metric_key,
        greater_is_better=greater_is_better,
        seed=base_arguments.seed,
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
    winner = None

    if best is not None:
        winner = winner_configuration(
            split_dir=split_dir,
            output_dir=run_dir / FINAL_TRAIN_DIRNAME,
            variant=variants[best.config["variant"]],
            sampled=best.config,
            micro_batch_size=optional_int(best.metrics.get("per_device_train_batch_size")),
            accumulation_steps=optional_int(best.metrics.get("gradient_accumulation_steps")),
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
        )

    summary = _sweep_summary(result_grid, best, metric_key, greater_is_better, winner)
    summary["total_wall_time_sec"] = time.monotonic() - sweep_start
    paths["hpo_summary"] = write_manifest(summary, run_dir / HPO_SUMMARY_FILENAME)

    if winner is not None:
        paths["winner"] = write_winner_config(winner, run_dir / WINNER_CONFIG_FILENAME)

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
        best=winner,
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


def _check_variant_lengths(
    variants: dict[str, dict[str, Any]],
    target_label: str,
    architecture: str | Architecture,
    architecture_kwargs: dict[str, Any] | None,
    report: bool,
) -> None:
    from transformers import AutoTokenizer

    vocabulary = build_label_vocabulary(target_label)
    lengths_by_model: dict[str, set[int]] = {}

    for specification in variants.values():
        lengths = lengths_by_model.setdefault(specification["base_model"], set())
        lengths.add(specification["max_length"])

    for base_model, max_lengths in lengths_by_model.items():
        model = build_model(
            base_model=base_model,
            label2id=vocabulary["label2id"],
            id2label=vocabulary["id2label"],
            architecture=architecture,
            **(architecture_kwargs or {}),
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        for max_length in sorted(max_lengths):
            check_max_length(model, tokenizer, max_length, base_model, report=report)


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
            "train_model": winner,
        }
    )

    return summary


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
