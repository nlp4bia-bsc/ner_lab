"""Scoring multihead predictions: each head as its own single-type model, then across heads."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lab.core.scoring import (
    SCENARIO_PREFIXES,
    flatten,
    score_characters,
    score_entity_characters,
    score_spans,
    sum_span_results,
)
from lab.core.spans import span_dataframe
from lab.ner.encoding.tagging import build_label_vocabulary
from lab.ner.evaluation.metrics import token_diagnostics
from lab.ner.evaluation.spans import gold_spans, predicted_spans
from lab.ner.training.dataset import ensure_int_list

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizerBase

BEST_METRIC = "span_strict_macro_f1"

MACRO_STATISTICS = ("precision", "recall", "f1")


def head_rows(rows: pd.DataFrame, head: int) -> pd.DataFrame:
    """
    The rows as a single-type model's would be: `labels` cut down to one head's ids.

    Each head numbers its tags as `build_label_vocabulary` does, so these rows go
    through the single-label span and token functions unchanged.
    """
    view = rows.copy()
    view["labels"] = [
        [position[head] for position in ensure_int_list(labels)] for labels in rows["labels"]
    ]

    return view


def multihead_gold_spans(
    rows: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, entity_labels: list[str]
) -> pd.DataFrame:
    """Rebuild every head's annotated spans from the encoded rows, as one span table."""
    return pd.concat(
        [
            gold_spans(
                rows=head_rows(rows, head),
                tokenizer=tokenizer,
                id2label=build_label_vocabulary(label)["id2label"],
            )
            for head, label in enumerate(entity_labels)
        ],
        ignore_index=True,
    )


def multihead_predicted_spans(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    texts: dict[str, str] | None = None,
    include_scores: bool = False,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Rebuild every head's predicted spans, as one span table.

    `predictions` is `[rows, length, heads, 3]`, and `probabilities`, when given, the
    same shape. Each head is decoded on its own by `predicted_spans`, so spans of
    different types may overlap, and `texts`, `include_scores` and `probabilities`
    work as they do there.
    """
    if predictions.ndim != 4 or predictions.shape[2] != len(entity_labels):
        raise ValueError(
            f"predictions shaped {predictions.shape} do not have one head for each of the "
            f"{len(entity_labels)} entity_labels; expected [rows, length, heads, 3]."
        )

    frames = [
        predicted_spans(
            rows=head_rows(rows, head),
            predictions=predictions[:, :, head, :],
            tokenizer=tokenizer,
            id2label=build_label_vocabulary(label)["id2label"],
            texts=texts,
            include_scores=include_scores,
            probabilities=None if probabilities is None else probabilities[:, :, head, :],
        )
        for head, label in enumerate(entity_labels)
    ]

    spans = pd.concat(frames, ignore_index=True).to_dict("records")

    return span_dataframe(spans, scored=include_scores)


def score_heads(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    entity_labels: list[str],
    min_overlap_percentage: float = 40.0,
) -> dict[str, float | int]:
    """
    Span and character metrics for a multihead model, per head, micro and macro.

    Each head is scored with nervaluate on its own type's spans, exactly as a
    single-label model for that type is, and keyed `head_<label>_<metric>`. The
    plain keys (`span_strict_f1`, ...) are the micro scores, the heads' counts summed
    by `sum_span_results`; `span_<scenario>_macro_<precision|recall|f1>` average the
    heads, each type weighing the same. Gold of a type with no head is left out, as
    a single-label model's gold is restricted to its type.
    """
    gold = gold[gold["label"].isin(entity_labels)]
    results = {
        label: score_spans(
            gold=gold[gold["label"] == label],
            predicted=predicted[predicted["label"] == label],
            tags=[label],
            min_overlap_percentage=min_overlap_percentage,
        )
        for label in entity_labels
    }

    metrics = flatten(sum_span_results(list(results.values())))

    for scenario, prefix in SCENARIO_PREFIXES.items():
        for statistic in MACRO_STATISTICS:
            metrics[f"{prefix}_macro_{statistic}"] = round(
                statistics.mean(
                    float(getattr(result[scenario], statistic)) for result in results.values()
                ),
                4,
            )

    for label, result in results.items():
        metrics.update({f"head_{label}_{key}": value for key, value in flatten(result).items()})

    metrics.update(score_characters(gold, predicted))
    metrics.update(score_entity_characters(gold, predicted))

    return metrics


def multihead_token_diagnostics(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    entity_labels: list[str],
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> dict[str, Any]:
    """`token_diagnostics` for each head, keyed `head_<label>_<metric>`."""
    metrics: dict[str, Any] = {}

    for head, label in enumerate(entity_labels):
        diagnostics = token_diagnostics(
            rows=head_rows(rows, head),
            predictions=predictions[:, :, head, :],
            id2label=build_label_vocabulary(label)["id2label"],
            include_tokens=include_tokens,
            include_by_entity=include_by_entity,
            include_confusion=include_confusion,
        )
        metrics.update({f"head_{label}_{key}": value for key, value in diagnostics.items()})

    return metrics


def evaluate_multihead_predictions(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    min_overlap_percentage: float = 40.0,
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> dict[str, Any]:
    """
    Score multihead predictions against the rows they were made on.

    The `evaluate_predictions` counterpart: gold is reconstructed from the rows, so
    it is the gold the model was shown, after overlap resolution and windowing.
    `span_strict_macro_f1` is what model selection tracks by default.
    """
    metrics = score_heads(
        gold=multihead_gold_spans(rows, tokenizer, entity_labels),
        predicted=multihead_predicted_spans(rows, predictions, tokenizer, entity_labels),
        entity_labels=entity_labels,
        min_overlap_percentage=min_overlap_percentage,
    )

    metrics.update(
        multihead_token_diagnostics(
            rows=rows,
            predictions=predictions,
            entity_labels=entity_labels,
            include_tokens=include_tokens,
            include_by_entity=include_by_entity,
            include_confusion=include_confusion,
        )
    )

    return metrics


def build_multihead_compute_metrics(
    rows: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    min_overlap_percentage: float = 40.0,
    include_tokens: bool = True,
    include_by_entity: bool = True,
    include_confusion: bool = False,
) -> Callable[[Any], dict[str, Any]]:
    """
    Build the `compute_metrics` callable a multihead training run hands to `train`.

    `rows` must be the frame, in the same order, that produced the dataset being
    evaluated, as for `build_compute_metrics`.
    """
    rows = rows.reset_index(drop=True)

    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, Any]:
        predictions = eval_prediction.predictions

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        return evaluate_multihead_predictions(
            rows=rows,
            predictions=np.asarray(predictions),
            tokenizer=tokenizer,
            entity_labels=entity_labels,
            min_overlap_percentage=min_overlap_percentage,
            include_tokens=include_tokens,
            include_by_entity=include_by_entity,
            include_confusion=include_confusion,
        )

    return compute_metrics
