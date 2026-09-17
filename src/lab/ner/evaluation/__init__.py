"""Scoring NER predictions: span reconstruction from rows, token diagnostics, the official scorer."""

from __future__ import annotations

from lab.ner.evaluation.metrics import (
    BEST_METRIC,
    build_compute_metrics,
    evaluate_predictions,
)
from lab.ner.evaluation.scoring import span_metrics
from lab.ner.evaluation.spans import (
    bio_to_spans,
    entity_tags,
    expand_to_word_extent,
    gold_spans,
    predicted_spans,
    softmax,
    strip_bio_prefix,
)
from lab.ner.evaluation.tokens import (
    token_confusion_matrix,
    token_metrics,
    token_metrics_by_entity,
)

__all__ = [
    "BEST_METRIC",
    "bio_to_spans",
    "build_compute_metrics",
    "entity_tags",
    "evaluate_predictions",
    "expand_to_word_extent",
    "gold_spans",
    "predicted_spans",
    "softmax",
    "span_metrics",
    "strip_bio_prefix",
    "token_confusion_matrix",
    "token_metrics",
    "token_metrics_by_entity",
]
