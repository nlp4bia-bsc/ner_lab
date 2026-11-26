"""The `ner.search_multihead` task: a hyperparameter sweep for a multihead model."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, TrainingArguments

from lab.core.provenance import write_manifest
from lab.ner.encoding.encoder import WindowStrategy, encode_partition, strategy_name
from lab.ner.encoding.overlaps import OverlapPolicy
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
from lab.ner.hpo.variants import EncodedVariant
from lab.ner.hpo.winner import FINAL_TRAIN_DIRNAME, winner_training_arguments
from lab.ner.multihead.encoding import MultiHeadEncoder
from lab.ner.multihead.loading import (
    MODEL_CLASSES,
    MultiHeadArchitecture,
    chain_settings,
    recorded_architecture,
)
from lab.ner.multihead.scoring import build_multihead_compute_metrics
from lab.ner.multihead.training import resolve_entity_labels, with_default_metric
from lab.ner.training.runs import RUN_MANIFEST_FILENAME, claim_run_dir


def search_multihead(
    split_dir: str | Path,
    output_dir: str | Path,
    base_models: str | Sequence[str],
    language: str,
    entity_labels: Sequence[str] | None = None,
    architecture: MultiHeadArchitecture = "linear",
    constrain_transitions: bool | None = None,
    dropout: float = 0.1,
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
    Search hyperparameters for a multihead model against a prepared split.

    The `search_hyperparameters` counterpart, sweeping the same search space, variants,
    seeds, top-k epochs, smoke test and reports, with every trial training a multihead
    model. `entity_labels` is resolved as `train_multihead` resolves it, and the
    objective defaults to `span_strict_macro_f1` unless `training_arguments` names
    another metric. `architecture` and `constrain_transitions` are taken as
    `train_multihead` takes them, fixed for the sweep.

    `winner.yaml` is a ready-to-run `ner.train_multihead` config pointed at
    `<output_dir>/final_train`, with the head catalogue written out in full.
    """
    model_class = MODEL_CLASSES[recorded_architecture(architecture)]
    model_settings = chain_settings(architecture, constrain_transitions)
    sweep_start = time.monotonic()

    run_dir = claim_run_dir(output_dir, overwrite)
    setup = prepare_search(
        split_dir=split_dir,
        base_models=base_models,
        strategies=strategies,
        context_tokens=context_tokens,
        max_lengths=max_lengths,
        search_space=search_space,
        training_arguments=with_default_metric(training_arguments),
        validation_index=validation_index,
        random_state=random_state,
        run_dir=run_dir,
    )

    entity_labels = resolve_entity_labels(entity_labels, setup.split_dir, setup.validation_index)

    check_variant_lengths(
        variants=setup.variants,
        build_for=partial(
            model_class, entity_labels=entity_labels, dropout=dropout, **model_settings
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
            "architecture": recorded_architecture(architecture),
            "dropout": dropout,
            **model_settings,
        },
        "encoding": {
            "entity_labels": entity_labels,
            "language": language,
            "overlap_policy": overlap_policy,
            "min_sentence_tokens": min_sentence_tokens,
        },
    }
    manifest_path = write_manifest(manifest, run_dir / RUN_MANIFEST_FILENAME)

    encoded = encode_multihead_variants(
        partitions=setup.partitions,
        variants=setup.variants,
        entity_labels=entity_labels,
        language=language,
        architecture=architecture,
        constrain_transitions=constrain_transitions,
        dropout=dropout,
        overlap_policy=overlap_policy,
        min_sentence_tokens=min_sentence_tokens,
        min_overlap_percentage=min_overlap_percentage,
    )

    def winner(best: Any) -> dict[str, Any]:
        variant = setup.variants[best.config["variant"]]

        return {
            "task": "ner.train_multihead",
            "split_dir": str(setup.split_dir),
            "output_dir": str(run_dir / FINAL_TRAIN_DIRNAME),
            "base_model": variant["base_model"],
            "language": language,
            "entity_labels": entity_labels,
            "architecture": architecture,
            **model_settings,
            "dropout": dropout,
            "max_length": variant["max_length"],
            "strategy": strategy_name(variant["strategy"]),
            "context_tokens": variant["context_tokens"],
            "overlap_policy": overlap_policy,
            "min_sentence_tokens": min_sentence_tokens,
            "min_overlap_percentage": min_overlap_percentage,
            "early_stopping_patience": early_stopping_patience,
            "training_arguments": winner_training_arguments(
                sampled=best.config,
                micro_batch_size=optional_int(best.metrics.get("per_device_train_batch_size")),
                accumulation_steps=optional_int(best.metrics.get("gradient_accumulation_steps")),
                base_arguments=setup.base_arguments,
                user_overrides=user_argument_overrides(training_arguments),
            ),
        }

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
        winner_task="train_multihead",
        winner=winner,
    )


def encode_multihead_variants(
    partitions: dict[str, Path | list[Path]],
    variants: dict[str, dict[str, Any]],
    entity_labels: list[str],
    language: str,
    architecture: MultiHeadArchitecture = "linear",
    constrain_transitions: bool | None = None,
    dropout: float = 0.1,
    overlap_policy: OverlapPolicy = "merge_same_label_then_keep_longest",
    min_sentence_tokens: int = 4,
    min_overlap_percentage: float = 40.0,
) -> dict[str, EncodedVariant]:
    """
    Window the train and validation partitions once per variant, for multihead trials.

    The `encode_variants` counterpart: each variant builds a multihead model of
    `architecture` over its base model and scores per head with span metrics only.
    """
    model_class = MODEL_CLASSES[recorded_architecture(architecture)]
    model_settings = chain_settings(architecture, constrain_transitions)
    tokenizers: dict[str, Any] = {}
    encoded: dict[str, EncodedVariant] = {}

    for key, specification in variants.items():
        base_model = specification["base_model"]

        if base_model not in tokenizers:
            tokenizers[base_model] = AutoTokenizer.from_pretrained(base_model)

        encoder = MultiHeadEncoder(
            tokenizer=tokenizers[base_model],
            entity_labels=entity_labels,
            language=language,
            max_length=specification["max_length"],
            overlap_policy=overlap_policy,
            strategy=specification["strategy"],
            context_tokens=specification["context_tokens"],
            min_sentence_tokens=min_sentence_tokens,
        )
        validation_rows = encode_partition(encoder, partitions["validation"])

        encoded[key] = EncodedVariant(
            base_model=base_model,
            strategy=specification["strategy"],
            context_tokens=specification["context_tokens"],
            max_length=specification["max_length"],
            tokenizer=tokenizers[base_model],
            train_rows=encode_partition(encoder, partitions["train"]),
            validation_rows=validation_rows,
            build_model=partial(
                model_class,
                base_model=base_model,
                entity_labels=entity_labels,
                dropout=dropout,
                **model_settings,
            ),
            compute_metrics=build_multihead_compute_metrics(
                rows=validation_rows,
                tokenizer=tokenizers[base_model],
                entity_labels=entity_labels,
                min_overlap_percentage=min_overlap_percentage,
                include_tokens=False,
                include_by_entity=False,
            ),
        )

    return encoded
