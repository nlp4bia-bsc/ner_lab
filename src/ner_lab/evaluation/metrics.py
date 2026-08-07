"""The `compute_metrics` a Trainer calls, and the same scoring outside training."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ner_lab.encoding.rows import IGNORE_INDEX
from ner_lab.evaluation.scoring import span_metrics
from ner_lab.evaluation.spans import entity_tags, gold_spans, predicted_spans
from ner_lab.evaluation.tokens import (
    token_confusion_matrix,
    token_metrics,
    token_metrics_by_entity,
)

BEST_METRIC = "span_strict_f1"


def evaluate_predictions(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
    min_overlap_percentage: float = 40.0,
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> dict[str, Any]:
    """
    Score one set of predictions against the rows they were made on.

    Span metrics are always computed and are the canonical result;
    `span_strict_f1` is what model selection should track. Token metrics are
    diagnostic — they say how a model is wrong when the span numbers say it is.
    """
    gold = gold_spans(rows, tokenizer, id2label, ignore_index)
    predicted = predicted_spans(rows, predictions, tokenizer, id2label, ignore_index)

    metrics = span_metrics(
        gold=gold,
        predicted=predicted,
        tags=entity_tags(id2label),
        min_overlap_percentage=min_overlap_percentage,
    )

    labels = np.array([_padded(row, predictions.shape[1], ignore_index) for row in rows["labels"]])

    if include_tokens:
        metrics.update(token_metrics(predictions, labels, id2label, ignore_index))

    if include_by_entity:
        metrics.update(token_metrics_by_entity(predictions, labels, id2label))

    if include_confusion:
        metrics.update(token_confusion_matrix(predictions, labels, id2label, ignore_index))

    return metrics


def build_compute_metrics(
    rows: pd.DataFrame,
    tokenizer,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
    min_overlap_percentage: float = 40.0,
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> Callable[[Any], dict[str, Any]]:
    """
    Build the `compute_metrics` callable to hand to `ner_lab.training.train`.

    `rows` must be the same frame, in the same order, that produced the dataset
    being evaluated — span reconstruction reads `doc_id`, `token_offsets` and
    `word_ids` from it positionally.
    """
    rows = rows.reset_index(drop=True)

    def compute_metrics(eval_prediction) -> dict[str, Any]:
        predictions = eval_prediction.predictions

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        return evaluate_predictions(
            rows=rows,
            predictions=np.asarray(predictions),
            tokenizer=tokenizer,
            id2label=id2label,
            ignore_index=ignore_index,
            min_overlap_percentage=min_overlap_percentage,
            include_tokens=include_tokens,
            include_by_entity=include_by_entity,
            include_confusion=include_confusion,
        )

    return compute_metrics


def _padded(labels, width: int, ignore_index: int) -> list[int]:
    values = [int(label) for label in labels][:width]

    return values + [ignore_index] * (width - len(values))
