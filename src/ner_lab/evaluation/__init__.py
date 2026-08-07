"""Scoring predictions: span reconstruction, SemEval scoring, token diagnostics."""

from __future__ import annotations

from ner_lab.evaluation.metrics import (
    BEST_METRIC,
    build_compute_metrics,
    evaluate_predictions,
)
from ner_lab.evaluation.scoring import (
    SCENARIO_PREFIXES,
    flatten,
    group_by_document,
    normalize_spans,
    safe_f1,
    score_spans,
    span_metrics,
)
from ner_lab.evaluation.spans import (
    SPAN_COLUMNS,
    bio_to_spans,
    entity_tags,
    expand_to_word_extent,
    gold_spans,
    predicted_spans,
    span_dataframe,
    strip_bio_prefix,
)
from ner_lab.evaluation.tokens import (
    token_confusion_matrix,
    token_metrics,
    token_metrics_by_entity,
)

__all__ = [
    "BEST_METRIC",
    "SCENARIO_PREFIXES",
    "SPAN_COLUMNS",
    "bio_to_spans",
    "build_compute_metrics",
    "entity_tags",
    "evaluate_predictions",
    "expand_to_word_extent",
    "flatten",
    "gold_spans",
    "group_by_document",
    "normalize_spans",
    "predicted_spans",
    "safe_f1",
    "score_spans",
    "span_dataframe",
    "span_metrics",
    "strip_bio_prefix",
    "token_confusion_matrix",
    "token_metrics",
    "token_metrics_by_entity",
]
