"""The `ner.search_hyperparameters` task: a single-label sweep over a prepared split."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from lab.core.provenance import write_manifest
from lab.ner.encoding.encoder import WindowStrategy
from lab.ner.encoding.overlaps import OverlapPolicy
from lab.ner.encoding.tagging import build_label_vocabulary
from lab.ner.hpo.arguments import user_argument_overrides
from lab.ner.hpo.sweep import (
    HPOResult,
    check_variant_lengths,
    prepare_search,
    run_search,
    start_ray,
    sweep_manifest,
)
from lab.ner.hpo.trials import optional_int
from lab.ner.hpo.variants import encode_variants
from lab.ner.hpo.winner import FINAL_TRAIN_DIRNAME, winner_configuration
from lab.ner.models.registry import Architecture, architecture_name, build_model
from lab.ner.training.runs import RUN_MANIFEST_FILENAME, claim_run_dir


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
    sweep_start = time.monotonic()

    run_dir = claim_run_dir(output_dir, overwrite)
    setup = prepare_search(
        split_dir=split_dir,
        base_models=base_models,
        strategies=strategies,
        context_tokens=context_tokens,
        max_lengths=max_lengths,
        search_space=search_space,
        training_arguments=training_arguments,
        validation_index=validation_index,
        random_state=random_state,
        run_dir=run_dir,
    )

    vocabulary = build_label_vocabulary(target_label)
    check_variant_lengths(
        variants=setup.variants,
        build_for=partial(
            build_model,
            label2id=vocabulary["label2id"],
            id2label=vocabulary["id2label"],
            architecture=architecture,
            **(architecture_kwargs or {}),
        ),
        report=report,
    )

    gpus_per_trial = start_ray()

    manifest = {
        **sweep_manifest(
            setup=setup,
            gpus_per_trial=gpus_per_trial,
            n_trials=n_trials,
            seeds_per_trial=seeds_per_trial,
            top_k_epochs=top_k_epochs,
            max_micro_batch_size=max_micro_batch_size,
            min_overlap_percentage=min_overlap_percentage,
        ),
        "model": {
            "base_models": setup.base_models,
            "architecture": architecture_name(architecture),
            "architecture_kwargs": architecture_kwargs or {},
        },
        "encoding": {
            "target_label": target_label,
            "language": language,
            "overlap_policy": overlap_policy,
            "min_sentence_tokens": min_sentence_tokens,
        },
    }
    manifest_path = write_manifest(manifest, run_dir / RUN_MANIFEST_FILENAME)

    encoded = encode_variants(
        partitions=setup.partitions,
        variants=setup.variants,
        target_label=target_label,
        language=language,
        overlap_policy=overlap_policy,
        min_sentence_tokens=min_sentence_tokens,
        architecture=architecture,
        architecture_kwargs=architecture_kwargs,
        min_overlap_percentage=min_overlap_percentage,
    )

    def winner(best: Any) -> dict[str, Any]:
        return winner_configuration(
            split_dir=setup.split_dir,
            output_dir=run_dir / FINAL_TRAIN_DIRNAME,
            variant=setup.variants[best.config["variant"]],
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
            base_arguments=setup.base_arguments,
            user_overrides=user_argument_overrides(training_arguments),
        )

    return run_search(
        setup=setup,
        encoded=encoded,
        gpus_per_trial=gpus_per_trial,
        run_dir=run_dir,
        manifest=manifest,
        paths={"run_manifest": manifest_path},
        n_trials=n_trials,
        seeds_per_trial=seeds_per_trial,
        top_k_epochs=top_k_epochs,
        max_micro_batch_size=max_micro_batch_size,
        early_stopping_patience=early_stopping_patience,
        pad_to_multiple_of=pad_to_multiple_of,
        track_resources=track_resources,
        study_name=study_name,
        storage=storage,
        smoke_test=smoke_test,
        report=report,
        sweep_start=sweep_start,
        winner_task="train_model",
        winner=winner,
    )
