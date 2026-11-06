"""Architecture-specific defaults and head-aware factory arguments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import TrainingArguments

    from lab.ner.encoding.encoder import Encoder


def architecture_arguments(
    arguments: TrainingArguments | Mapping[str, Any] | None,
    architecture: Any,
) -> TrainingArguments | Mapping[str, Any] | None:
    """Choose macro F1 only when no explicit model selection was supplied."""
    if (
        architecture not in ("multihead", "multihead_crf")
        or arguments is not None
        and not isinstance(arguments, Mapping)
    ):
        return arguments
    settings = dict(arguments or {})
    if settings.get("metric_for_best_model") is None:
        settings["metric_for_best_model"] = "span_strict_macro_f1"
    settings.setdefault("greater_is_better", not settings["metric_for_best_model"].endswith("loss"))

    return settings


def head_architecture_kwargs(
    encoder: Encoder,
    architecture_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind the discovered catalogue, rejecting an inconsistent manual mapping."""
    settings = dict(architecture_kwargs or {})
    if encoder.target_label is None:
        if "entity_labels" in settings and settings["entity_labels"] != encoder.entity_labels:
            raise ValueError("architecture_kwargs.entity_labels disagrees with training labels.")
        settings["entity_labels"] = list(encoder.entity_labels)

    return settings
