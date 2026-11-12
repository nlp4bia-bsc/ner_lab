"""The `ner.train_multihead` task: one multihead configuration against a prepared split."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, set_seed

from lab.core.provenance import write_manifest
from lab.core.split import split_paths
from lab.ner.encoding.encoder import WindowStrategy, encode_partition
from lab.ner.encoding.overlaps import OverlapPolicy
from lab.ner.models.capacity import check_max_length
from lab.ner.multihead.encoding import (
    MultiHeadEncoder,
    describe_multihead_encoder,
    entity_labels_from_partitions,
)
from lab.ner.multihead.loading import ARCHITECTURE, multihead_model_encoding
from lab.ner.multihead.model import MultiHeadForTokenClassification
from lab.ner.multihead.scoring import BEST_METRIC, build_multihead_compute_metrics
from lab.ner.training.assessment import (
    ASSESSMENT_SUMMARY_FILENAME,
    AssessmentResult,
    fold_directory_key,
    release_accelerator_memory,
    resolve_training_arguments,
)
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
from lab.ner.training.runs import (
    RUN_MANIFEST_FILENAME,
    claim_run_dir,
    fold_rotations,
    read_data_manifest,
    split_provenance,
)
from lab.ner.training.tracking import gpu_hardware_info
from lab.ner.training.trainer import TrainingResult, train


def train_multihead(
    split_dir: str | Path,
    output_dir: str | Path,
    base_model: str,
    language: str,
    entity_labels: Sequence[str] | None = None,
    dropout: float = 0.1,
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
    devices: DevicePolicy = 1,
    overwrite: bool = False,
    folds: Sequence[int] | None = None,
    random_state: int | None = None,
) -> AssessmentResult:
    """
    Train one multihead configuration against the split in `split_dir` and record its scores.

    The `train_model` counterpart for a model with one head per entity type, run the
    same way: the split's mode decides one run or one per rotatable fold, the fixed
    holdout is never read, and the run directory, manifest, fold metrics, devices and
    `save_model` behave as `train_model` documents.

    `entity_labels` is the head catalogue, in head order. Left out, it is every type
    annotated anywhere in the split's train and validation partitions, sorted, so
    every fold trains the same heads; a listed type annotated nowhere in them raises.
    Each fold's validation is scored per head, micro and macro (see `score_heads`),
    and model selection defaults to `span_strict_macro_f1` unless `training_arguments`
    names another metric.

    `overlap_policy` resolves overlapping annotations within each type; different
    types are never resolved against each other.
    """
    run_dir = claim_run_dir(output_dir, overwrite)

    split_dir = Path(split_dir).resolve()
    data_manifest = read_data_manifest(split_dir)
    mode = data_manifest["split"]["mode"]
    rotations = fold_rotations(data_manifest, folds)

    entity_labels = resolve_entity_labels(entity_labels, split_dir, rotations[0])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    encoder = MultiHeadEncoder(
        tokenizer=tokenizer,
        entity_labels=entity_labels,
        language=language,
        max_length=max_length,
        overlap_policy=overlap_policy,
        strategy=strategy,
        context_tokens=context_tokens,
        min_sentence_tokens=min_sentence_tokens,
    )

    resolved_arguments = resolve_training_arguments(
        with_default_metric(training_arguments), run_dir, random_state
    )

    device_count = apply_device_policy(resolved_arguments, devices)

    manifest = {
        "split_dir": str(split_dir),
        "data_manifest": data_manifest,
        "mode": mode,
        "folds": rotations,
        "model": {"base_model": base_model, "architecture": ARCHITECTURE, "dropout": dropout},
        "encoding": describe_multihead_encoder(encoder),
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
            dropout=dropout,
            early_stopping_patience=early_stopping_patience,
            pad_to_multiple_of=pad_to_multiple_of,
            metrics_scope=metrics_scope,
            min_overlap_percentage=min_overlap_percentage,
            include_confusion=include_confusion,
            save_model=save_model,
            track_resources=track_resources,
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


def resolve_entity_labels(
    entity_labels: Sequence[str] | None, split_dir: Path, validation_index: int | None
) -> list[str]:
    """
    The head catalogue: the one given, or every type annotated in the split's folds.

    Read from one rotation's train and validation partitions, which together are every
    non-holdout fold. A given type annotated in none of them raises, as a typo would
    otherwise train a head that never sees a positive.
    """
    partitions = split_paths(split_dir, validation_index=validation_index)
    paths = [
        *(partitions["train"] if isinstance(partitions["train"], list) else [partitions["train"]]),
        partitions["validation"],
    ]
    annotated = entity_labels_from_partitions(paths)

    if entity_labels is None:
        return annotated

    missing = [label for label in entity_labels if label not in annotated]

    if missing:
        raise ValueError(
            f"entity_labels {missing[:10]} are annotated nowhere in the split's train and "
            f"validation partitions; annotated types are {annotated[:10]}. Fix the names, "
            "or leave entity_labels out to train every annotated type."
        )

    return list(entity_labels)


def with_default_metric(
    arguments: TrainingArguments | Mapping[str, Any] | None,
) -> TrainingArguments | Mapping[str, Any]:
    """
    Training arguments that select on `span_strict_macro_f1` unless they name a metric.

    A `TrainingArguments` instance is the caller's authority and comes back as given.
    """
    if isinstance(arguments, TrainingArguments):
        return arguments

    return {"metric_for_best_model": BEST_METRIC, **(arguments or {})}


def _train_fold(
    validation_index: int | None,
    split_dir: Path,
    run_dir: Path,
    encoder: MultiHeadEncoder,
    encoded: dict[str, pd.DataFrame],
    training_arguments: TrainingArguments,
    base_model: str,
    dropout: float,
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

    train_rows = encode_partition(encoder, partitions["train"], encoded)
    validation_rows = encode_partition(encoder, partitions["validation"], encoded)

    set_seed(training_arguments.seed)

    model = MultiHeadForTokenClassification(
        base_model=base_model, entity_labels=encoder.entity_labels, dropout=dropout
    )
    check_max_length(model, encoder.tokenizer, encoder.max_length, base_model)

    return train(
        model=model,
        tokenizer=encoder.tokenizer,
        train_rows=train_rows,
        validation_rows=validation_rows,
        training_arguments=dataclasses.replace(training_arguments, output_dir=str(fold_dir)),
        compute_metrics=build_multihead_compute_metrics(
            rows=validation_rows,
            tokenizer=encoder.tokenizer,
            entity_labels=encoder.entity_labels,
            min_overlap_percentage=min_overlap_percentage,
            include_confusion=include_confusion,
        ),
        train_compute_metrics=(
            build_multihead_compute_metrics(
                rows=train_rows,
                tokenizer=encoder.tokenizer,
                entity_labels=encoder.entity_labels,
                min_overlap_percentage=min_overlap_percentage,
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
        model_encoding=multihead_model_encoding(
            encoder=encoder,
            base_model=base_model,
            dropout=dropout,
            backbone_config=model.config,
        ),
    )
