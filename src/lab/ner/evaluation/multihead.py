"""Independent BIO reconstruction and exact set scoring across heads."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lab.core.scoring import score_characters
from lab.core.spans import span_dataframe
from lab.ner.encoding.rows import special_token_template
from lab.ner.evaluation.spans import _aligned_offsets, bio_to_spans, ensure_int_list, softmax

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

IDENTITY = ["filename", "label", "start_span", "end_span"]


def multihead_spans(
    rows: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    predictions: np.ndarray | None = None,
    ignore_index: int = -100,
    texts: dict[str, str] | None = None,
    include_scores: bool = False,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Decode each head using the shared offsets and existing BIO decoder."""
    if predictions is not None:
        if predictions.ndim != 4 or predictions.shape[2:] != (len(entity_labels), 3):
            raise ValueError("Multihead predictions must have shape [B,L,K,3] matching heads.")
        if len(predictions) != len(rows):
            raise ValueError("rows and predictions must have the same length.")
    ids = None if predictions is None else predictions.argmax(axis=-1)
    if include_scores and probabilities is None:
        probabilities = softmax(predictions)
    prefix, suffix = special_token_template(tokenizer)
    spans = []
    for index, row in enumerate(rows.itertuples(index=False)):
        labels = np.asarray(ensure_int_list(row.labels))
        if labels.ndim != 2 or labels.shape[1] != len(entity_labels):
            raise ValueError("Encoded labels disagree with entity_labels.")
        aligned = _aligned_offsets(row, prefix, suffix)
        for head, label in enumerate(entity_labels):
            positions = [p for p in aligned if labels[p, head] != ignore_index]
            chosen = labels[:, head] if ids is None else ids[index, : len(labels), head]
            vocabulary = {0: "O", 1: f"B-{label}", 2: f"I-{label}"}
            spans.extend(
                bio_to_spans(
                    str(row.doc_id),
                    [aligned[p] for p in positions],
                    [vocabulary[int(chosen[p])] for p in positions],
                    scores=(
                        [float(probabilities[index, p, head, chosen[p]]) for p in positions]
                        if probabilities is not None
                        else None
                    ),
                    text=None if texts is None else texts.get(str(row.doc_id), ""),
                )
            )

    return span_dataframe(spans, scored=include_scores)


def strict_multihead_metrics(
    gold: pd.DataFrame, predicted: pd.DataFrame, entity_labels: list[str]
) -> dict[str, float | int]:
    """Exact identities are sets; zero-support heads score zero and stay in macro.

    Unknown gold types cannot add heads. They count as micro false negatives and
    receive their own metrics, but the macro denominator stays the trained catalogue.
    """
    gold_set = set(gold[IDENTITY].itertuples(index=False, name=None))
    predicted_set = set(predicted[IDENTITY].itertuples(index=False, name=None))
    unknown = sorted({item[1] for item in gold_set} - set(entity_labels))
    if unknown:
        warnings.warn(
            f"Unknown gold labels {unknown[:10]} count as micro false negatives; "
            "macro uses trained heads only.",
            stacklevel=2,
        )
    metrics = {"unknown_gold_entities": sum(item[1] in unknown for item in gold_set)}
    head_scores = []
    for label in [*entity_labels, *unknown]:
        gold_head = {item for item in gold_set if item[1] == label}
        predicted_head = {item for item in predicted_set if item[1] == label}
        score = _exact_counts(gold_head, predicted_head)
        for key, value in score.items():
            metrics[f"span_strict_{key}__{label}"] = value
        if label in entity_labels:
            head_scores.append(score)
    micro = _exact_counts(gold_set, predicted_set)
    for key, value in micro.items():
        metrics[f"span_strict_micro_{key}"] = value
        metrics[f"span_strict_{key}"] = value
    for key in ("precision", "recall", "f1"):
        metrics[f"span_strict_macro_{key}"] = float(np.mean([s[key] for s in head_scores]))
    metrics.update(score_characters(gold, predicted))

    return metrics


def _exact_counts(
    gold: set[tuple[Any, ...]], predicted: set[tuple[Any, ...]]
) -> dict[str, float | int]:
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
