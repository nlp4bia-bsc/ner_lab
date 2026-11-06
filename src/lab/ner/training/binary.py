"""Train-only binary class weights and shared decoder settings for model selection."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from lab.ner.models.binary_decoding import BinaryDecoding
from lab.ner.models.binary_loss import estimate_positive_weights


def binary_architecture_kwargs(
    settings: dict | None, train_rows: pd.DataFrame, entity_labels: list[str]
) -> dict:
    """Estimate requested positive weights solely from the current fold's training rows."""
    settings = dict(settings or {})
    if settings.get("loss", "bce") == "weighted_bce" and settings.get("pos_weight") is None:
        settings["pos_weight_policy"] = "train"
        settings["pos_weight"] = estimate_positive_weights(
            train_rows,
            entity_labels,
            settings.get("weight_smoothing", 1.0),
            settings.get("max_pos_weight", 20.0),
        )

    return settings


def configure_binary_metrics(rows: pd.DataFrame, settings: dict | None) -> None:
    """Attach explicit decoding defaults to rows before a metrics closure captures them."""
    selected = {
        field.name: (settings or {})[field.name]
        for field in fields(BinaryDecoding)
        if field.name in (settings or {})
    }
    BinaryDecoding(**selected)
    rows.attrs["multilabel_decoding"] = selected
