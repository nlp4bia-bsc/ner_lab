"""The `compute_metrics` a Trainer calls, and the same scoring outside training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from lab.ner.encoding.rows import IGNORE_INDEX
from lab.ner.evaluation.scoring import span_metrics
from lab.ner.evaluation.spans import entity_tags, gold_spans, predicted_spans
from lab.ner.evaluation.tokens import (
    token_confusion_matrix,
    token_metrics,
    token_metrics_by_entity,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

BEST_METRIC = "span_strict_f1"


def evaluate_predictions(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
    min_overlap_percentage: float = 40.0,
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> dict[str, Any]:
    """
    Score one set of predictions against the rows they were made on.

    Span and character metrics are always computed and are the canonical
    result; `span_strict_f1` is what model selection should track. Token metrics
    are diagnostic — they say how a model is wrong when the span numbers say it
    is. Gold here is reconstructed from the rows, so it is the gold the model was
    shown: after overlap resolution and windowing.
    """
    gold = gold_spans(rows, tokenizer, id2label, ignore_index)
    predicted = predicted_spans(rows, predictions, tokenizer, id2label, ignore_index)

    metrics = span_metrics(
        gold=gold,
        predicted=predicted,
        tags=entity_tags(id2label),
        min_overlap_percentage=min_overlap_percentage,
    )

    metrics.update(
        token_diagnostics(
            rows=rows,
            predictions=predictions,
            id2label=id2label,
            ignore_index=ignore_index,
            include_tokens=include_tokens,
            include_by_entity=include_by_entity,
            include_confusion=include_confusion,
        )
    )

    return metrics


def token_diagnostics(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> dict[str, Any]:
    """
    The token-level half of `evaluate_predictions`: how a model is wrong, token by token.

    Reads the gold labels off `rows`, so it needs rows encoded from an annotated
    corpus; it is what inference adds when its input carried gold.
    """
    labels = np.array([_padded(row, predictions.shape[1], ignore_index) for row in rows["labels"]])
    metrics: dict[str, Any] = {}

    if include_tokens:
        metrics.update(token_metrics(predictions, labels, id2label, ignore_index))

    if include_by_entity:
        metrics.update(token_metrics_by_entity(predictions, labels, id2label))

    if include_confusion:
        metrics.update(token_confusion_matrix(predictions, labels, id2label, ignore_index))

    return metrics


def build_compute_metrics(
    rows: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
    min_overlap_percentage: float = 40.0,
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> Callable[[Any], dict[str, Any]]:
    """
    Build the `compute_metrics` callable to hand to `lab.ner.training.train`.

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
