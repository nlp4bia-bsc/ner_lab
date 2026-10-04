"""The training arguments a sweep runs with, and what the caller changed from the defaults."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from lab.ner.training.arguments import training_arguments as build_training_arguments

HPO_ARGUMENT_DEFAULTS: dict[str, Any] = {
    "num_train_epochs": 40,
    "save_strategy": "no",
    "load_best_model_at_end": False,
}

DERIVED_ARGUMENT_KEYS = ("output_dir", "run_name", "logging_dir")


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
