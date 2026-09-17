"""Span scoring keyed by a model's label vocabulary."""

from __future__ import annotations

import pandas as pd

from lab.core.scoring import flatten, score_spans
from lab.ner.evaluation.spans import entity_tags


def span_metrics(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    id2label: dict | None = None,
    tags: list[str] | None = None,
    min_overlap_percentage: float = 40.0,
) -> dict[str, float | int]:
    """Score spans and flatten the result into a metric dict."""
    if tags is None:
        if id2label is None:
            raise ValueError("Pass either tags or id2label.")

        tags = entity_tags(id2label)

    return flatten(score_spans(gold, predicted, tags, min_overlap_percentage))
