"""This library's opinionated defaults for `transformers.TrainingArguments`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainingArguments

DEFAULTS: dict[str, Any] = {
    "learning_rate": 3e-5,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "logging_strategy": "epoch",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "span_strict_f1",
    "greater_is_better": True,
    "fp16": True,
    "bf16": False,
    "seed": 42,
    "report_to": "none",
    "dataloader_num_workers": 0,
    "ddp_find_unused_parameters": False,
}


def training_arguments(output_dir: str | Path, **overrides: Any) -> TrainingArguments:
    """
    Build `TrainingArguments` from `DEFAULTS`, overridden by whatever you pass.

    Every keyword `TrainingArguments` accepts is accepted here, so this is a
    starting point rather than a wrapper — build your own and hand it to `train`
    if these defaults are in the way.

    `metric_for_best_model` defaults to `span_strict_f1`, which requires the
    `compute_metrics` built by `ner_lab.evaluation.build_compute_metrics`. Pass
    `metric_for_best_model="eval_loss"` with `greater_is_better=False` to train
    without any span scoring.
    """
    settings = {**DEFAULTS, **overrides, "output_dir": str(output_dir)}

    if "eval_strategy" in settings:
        try:
            return TrainingArguments(**settings)
        except TypeError:
            settings["evaluation_strategy"] = settings.pop("eval_strategy")

    return TrainingArguments(**settings)


def for_inference(arguments: TrainingArguments) -> TrainingArguments:
    """
    A copy with evaluation, saving and best-model loading switched off.

    `Trainer` refuses to build without an `eval_dataset` when the eval strategy
    is not `no`, and refuses mismatched eval/save strategies under
    `load_best_model_at_end` — both of which an inference-only caller trips.
    """
    import dataclasses

    changes = {"save_strategy": "no", "load_best_model_at_end": False}

    if hasattr(arguments, "eval_strategy"):
        changes["eval_strategy"] = "no"
    else:
        changes["evaluation_strategy"] = "no"

    return dataclasses.replace(arguments, **changes)
