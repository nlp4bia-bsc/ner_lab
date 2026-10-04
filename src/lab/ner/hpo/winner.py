"""The winning trial as a ready-to-run `train_model` configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from transformers import TrainingArguments

from lab.ner.encoding.encoder import strategy_name
from lab.ner.hpo.trial import RESERVED_DIMENSIONS
from lab.ner.models.registry import Architecture, architecture_name

FINAL_TRAIN_DIRNAME = "final_train"
WINNER_CONFIG_FILENAME = "winner.yaml"


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

    `output_dir` becomes the config's own `output_dir`, and `train_model` writes
    into it directly, so this must name a directory no run owns yet — the sweep
    passes `<its own directory>/final_train`.
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
        "task": "ner.train_model",
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "base_model": variant["base_model"],
        "target_label": target_label,
        "language": language,
        "architecture": architecture_name(architecture),
        "max_length": variant["max_length"],
        "strategy": strategy_name(variant["strategy"]),
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


def write_winner_config(configuration: Mapping[str, Any], path: str | Path) -> Path:
    """
    Write the winner block as a YAML config `lab run` accepts unchanged.

    The same content `hpo_summary.json` already records, in the form the CLI
    takes, so the winning trial reaches `train_model` without being retyped.
    Still an output only: nothing reads it back, and `output_dir` is the sweep's
    own, so the training run lands beside the sweep unless `--output-dir` says
    otherwise.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(configuration), file, sort_keys=False, allow_unicode=True)

    return config_path
