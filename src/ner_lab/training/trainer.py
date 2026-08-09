"""Training a token-classification model on encoded rows."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ner_lab.encoding.rows import IGNORE_INDEX
from ner_lab.training.dataset import to_dataset, validate_rows
from ner_lab.training.tracking import EpochMetricsLogger, ResourceTracker, gpu_hardware_info

EPOCH_METRICS_FILENAME = "epoch_metrics.parquet"
SUMMARY_FILENAME = "training_summary.json"
BEST_MODEL_DIRNAME = "best_model"
MODEL_ENCODING_FILENAME = "encoding.json"


@dataclass
class TrainingResult:
    """What one training run produced, in memory and on disk."""

    trainer: Trainer
    model: Any
    metrics: dict[str, Any]
    epoch_metrics: pd.DataFrame
    summary: dict[str, Any]
    output_dir: Path
    resources: dict[str, Any] | None = None
    paths: dict[str, Path] = field(default_factory=dict)


class CRFTrainer(Trainer):
    """
    A Trainer that predicts by Viterbi decoding rather than emission argmax.

    The decoded path is re-encoded as one-hot logits so a CRF model and a linear
    one reach the evaluation stack in the same shape.
    """

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        from ner_lab.models.crf import decoded_paths_to_logits

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.detach() if has_labels and outputs.loss is not None else None

            if prediction_loss_only:
                return loss, None, None

            emissions = outputs.logits
            decoder = model.module if hasattr(model, "module") else model

            decoded = decoder.decode_from_emissions(
                emissions=emissions, attention_mask=inputs.get("attention_mask")
            )
            logits = decoded_paths_to_logits(
                decoded_paths=decoded,
                sequence_length=emissions.shape[1],
                num_labels=emissions.shape[2],
                device=emissions.device,
            )

        return loss, logits.detach(), labels.detach() if labels is not None else None


def resolve_trainer_class(model) -> type[Trainer]:
    """Pick `CRFTrainer` for a model that can Viterbi-decode, `Trainer` otherwise."""
    return CRFTrainer if hasattr(model, "decode_from_emissions") else Trainer


def train(
    model,
    tokenizer,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    training_arguments: TrainingArguments,
    compute_metrics: Callable[[Any], dict] | None = None,
    train_compute_metrics: Callable[[Any], dict] | None = None,
    trainer_class: type[Trainer] | None = None,
    early_stopping_patience: int | None = 5,
    pad_to_multiple_of: int | None = 8,
    ignore_index: int = IGNORE_INDEX,
    metrics_scope: str = "eval",
    save_model: bool = False,
    track_resources: bool = False,
    run_metadata: dict[str, Any] | None = None,
    model_encoding: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """
    Train `model` on encoded rows and return everything the run produced.

    `train_rows` and `validation_rows` are `Encoder` output. `training_arguments`
    carries every optimization and run setting — build one with
    `ner_lab.training.training_arguments` for this library's defaults, or pass
    your own.

    `trainer_class` defaults to `CRFTrainer` for a CRF model and `Trainer`
    otherwise; pass one to override. `compute_metrics` is handed straight to the
    Trainer, and with `metrics_scope="both"` a second pass over the training set
    runs each epoch using `train_compute_metrics`.

    Weights are not kept unless `save_model` is set: the checkpoints the Trainer
    writes to select the best epoch are removed afterwards, including when
    training raises. `epoch_metrics.parquet` and `training_summary.json` are
    always written to `training_arguments.output_dir`.

    `model_encoding` is written beside the saved weights as `encoding.json`, so
    the checkpoint says how its input was windowed and how it was built. Build
    one with `ner_lab.training.model_encoding`; without it a saved directory
    cannot be loaded by `ner_lab.inference` — and a CRF checkpoint, which the
    Trainer saves as a bare state dict, cannot be rebuilt at all.
    """
    validate_rows(train_rows, "train_rows")
    validate_rows(validation_rows, "validation_rows")

    _check_best_metric(training_arguments, compute_metrics)

    if (
        save_model
        and early_stopping_patience is not None
        and not training_arguments.load_best_model_at_end
    ):
        raise ValueError(
            "save_model=True with early stopping requires load_best_model_at_end=True on "
            "the training arguments — otherwise the saved weights are the last epoch's, "
            "not the best epoch's. Enable it, or pass early_stopping_patience=None."
        )

    output_dir = Path(training_arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = to_dataset(train_rows)
    validation_dataset = to_dataset(validation_rows)

    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_arguments=training_arguments,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        trainer_class=trainer_class or resolve_trainer_class(model),
        early_stopping_patience=early_stopping_patience,
        pad_to_multiple_of=pad_to_multiple_of,
        ignore_index=ignore_index,
    )

    metrics_logger = EpochMetricsLogger(
        trainer=trainer,
        scope=metrics_scope,
        train_dataset=train_dataset,
        train_compute_metrics=train_compute_metrics,
    )
    trainer.add_callback(metrics_logger)

    tracker = ResourceTracker() if track_resources else None
    resources = None
    paths: dict[str, Path] = {}

    if tracker is not None:
        tracker.start()

    try:
        train_output = trainer.train()

        if tracker is not None:
            resources = tracker.stop()
            tracker = None

        epoch_metrics_path = metrics_logger.write(output_dir / EPOCH_METRICS_FILENAME)

        if epoch_metrics_path is not None:
            paths["epoch_metrics"] = epoch_metrics_path

        if save_model:
            best_model_dir = output_dir / BEST_MODEL_DIRNAME
            best_model_dir.mkdir(parents=True, exist_ok=True)

            trainer.save_model(str(best_model_dir))
            tokenizer.save_pretrained(best_model_dir)
            paths["best_model"] = best_model_dir

            if model_encoding is not None:
                paths["model_encoding"] = write_model_encoding(model_encoding, best_model_dir)
    finally:
        if tracker is not None:
            resources = tracker.stop()

        if not save_model:
            remove_checkpoints(output_dir)

    summary = build_summary(
        trainer=trainer,
        train_output=train_output,
        train_rows=train_rows,
        validation_rows=validation_rows,
        resources=resources,
        run_metadata=run_metadata,
    )

    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    return TrainingResult(
        trainer=trainer,
        model=trainer.model,
        metrics=summary["best"],
        epoch_metrics=metrics_logger.frame(),
        summary=summary,
        output_dir=output_dir,
        resources=resources,
        paths=paths,
    )


def write_model_encoding(encoding: Mapping[str, Any], model_dir: str | Path) -> Path:
    """Write a model's `encoding.json` into the directory holding its weights."""
    path = Path(model_dir) / MODEL_ENCODING_FILENAME
    path.write_text(json.dumps(dict(encoding), ensure_ascii=False, indent=2), encoding="utf-8")

    return path


def read_model_encoding(model_dir: str | Path) -> dict[str, Any]:
    """Read the `encoding.json` written beside a saved model."""
    path = Path(model_dir) / MODEL_ENCODING_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"No {MODEL_ENCODING_FILENAME} in {model_dir} — it is written when a model is "
            "saved with a model_encoding, so this directory was produced without one."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(
    trainer: Trainer,
    train_output,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    resources: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the run record written next to the model."""
    arguments = trainer.args
    metric_name = arguments.metric_for_best_model

    return {
        "best": {
            "metric": metric_name,
            "value": trainer.state.best_metric,
            "checkpoint": trainer.state.best_model_checkpoint,
            "epoch": trainer.state.epoch,
        },
        "train_output": {
            "global_step": train_output.global_step,
            "training_loss": train_output.training_loss,
            **{key: value for key, value in train_output.metrics.items()},
        },
        "dataset": dataset_summary(train_rows, validation_rows),
        "arguments": arguments.to_dict(),
        "hardware": gpu_hardware_info(),
        "resources": resources,
        "log_history": trainer.state.log_history,
        "metadata": run_metadata or {},
    }


def dataset_summary(train_rows: pd.DataFrame, validation_rows: pd.DataFrame) -> dict[str, Any]:
    """Row and token-length counts for both partitions."""
    summary: dict[str, Any] = {}

    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        lengths = rows["input_ids"].map(len)

        summary[f"{name}_rows"] = int(len(rows))
        summary[f"{name}_documents"] = int(rows["doc_id"].nunique()) if "doc_id" in rows else None
        summary[f"{name}_token_count_mean"] = float(lengths.mean())
        summary[f"{name}_token_count_max"] = int(lengths.max())

    return summary


def remove_checkpoints(output_dir: str | Path) -> list[Path]:
    """
    Delete the `checkpoint-*` directories the Trainer wrote, returning what went.

    Called on the failure path too: an OOM two epochs in otherwise leaks a full
    set of checkpoints per crashed run.
    """
    removed = []

    for checkpoint in sorted(Path(output_dir).glob("checkpoint-*")):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint, ignore_errors=True)
            removed.append(checkpoint)

    return removed


def _check_best_metric(training_arguments: TrainingArguments, compute_metrics) -> None:
    metric = training_arguments.metric_for_best_model

    if compute_metrics is not None or metric in (None, "loss", "eval_loss"):
        return

    raise ValueError(
        f"metric_for_best_model={metric!r} needs a compute_metrics that produces it. "
        "Build one with ner_lab.evaluation.build_compute_metrics, or set "
        "metric_for_best_model='eval_loss' with greater_is_better=False."
    )


def _build_trainer(
    model,
    tokenizer,
    training_arguments: TrainingArguments,
    train_dataset,
    validation_dataset,
    compute_metrics,
    trainer_class: type[Trainer],
    early_stopping_patience: int | None,
    pad_to_multiple_of: int | None,
    ignore_index: int,
) -> Trainer:
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=ignore_index,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    arguments = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
    }

    try:
        trainer = trainer_class(processing_class=tokenizer, **arguments)
    except TypeError:
        trainer = trainer_class(tokenizer=tokenizer, **arguments)

    if early_stopping_patience is not None:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )

    return trainer
