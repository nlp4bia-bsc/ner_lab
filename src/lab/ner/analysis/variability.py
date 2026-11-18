"""Variability across documents: per-document scores and the document bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lab.core.scoring import safe_f1, score_characters, score_entity_characters

DOCUMENT_COUNTS = (
    "strict_tp",
    "strict_fp",
    "strict_fn",
    "char_correct",
    "char_spurious",
    "char_missed",
    "entity_char_gold_credit",
    "entity_char_gold",
    "entity_char_predicted_credit",
    "entity_char_predicted",
)
INTERVAL_METRICS = {
    "span_strict_precision": ("strict_tp", ("strict_tp", "strict_fp")),
    "span_strict_recall": ("strict_tp", ("strict_tp", "strict_fn")),
    "span_strict_f1": ("strict_tp", ("strict_tp", "strict_fp", "strict_fn")),
    "char_precision": ("char_correct", ("char_correct", "char_spurious")),
    "char_recall": ("char_correct", ("char_correct", "char_missed")),
    "char_f1": ("char_correct", ("char_correct", "char_spurious", "char_missed")),
    "entity_char_precision": ("entity_char_predicted_credit", ("entity_char_predicted",)),
    "entity_char_recall": ("entity_char_gold_credit", ("entity_char_gold",)),
}


def document_scores(
    events: pd.DataFrame, gold: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    """
    Strict, character and entity-weighted character scores per document, one row each.

    One row per document with a gold span or prediction. Strict counts are read off
    the events, so they sum to the overall strict counts; character and
    entity-weighted character scores are `score_characters` and
    `score_entity_characters` on the document alone. A span's entity-weighted credit
    depends only on its own document, so those credits and counts sum to the overall
    ones too.
    """
    strict = (
        events.groupby(["filename", "strict_outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["TP", "FP", "FN"], fill_value=0)
    )
    gold_by_document = dict(tuple(gold.groupby(gold["filename"].astype(str))))
    predicted_by_document = dict(tuple(predicted.groupby(predicted["filename"].astype(str))))
    empty = gold.iloc[0:0]

    rows = []

    for filename, (tp, fp, fn) in zip(strict.index, strict[["TP", "FP", "FN"]].to_numpy()):
        document_gold = gold_by_document.get(filename, empty)
        document_predicted = predicted_by_document.get(filename, empty)
        characters = score_characters(document_gold, document_predicted)
        entity_char = score_entity_characters(document_gold, document_predicted)
        strict_scores = safe_f1(int(tp), int(fp), int(fn))

        rows.append(
            {
                "filename": filename,
                "n_gold": len(document_gold),
                "n_predicted": len(document_predicted),
                "strict_tp": int(tp),
                "strict_fp": int(fp),
                "strict_fn": int(fn),
                "strict_precision": strict_scores["precision"],
                "strict_recall": strict_scores["recall"],
                "strict_f1": strict_scores["f1"],
                **characters,
                **entity_char,
            }
        )

    return pd.DataFrame(rows)


def bootstrap_draws(n_documents: int, samples: int, random_state: int) -> np.ndarray:
    """
    How often each document is drawn in each resample, as a `samples × n_documents` array.

    Each row is one resample of `n_documents` documents with replacement, so a
    resampled total is the row times a per-document count column.
    """
    generator = np.random.default_rng(random_state)

    return generator.multinomial(n_documents, np.full(n_documents, 1 / n_documents), size=samples)


def ratio_interval(
    numerators: np.ndarray, denominators: np.ndarray, draws: np.ndarray, confidence: float
) -> tuple[float, float]:
    """
    The percentile interval of `Σ numerator / Σ denominator` over the resamples.

    `numerators` and `denominators` are per-document counts in the documents' order.
    A resample whose denominator is zero has no value and is left out; with none
    left the interval is NaN.
    """
    return percentile_interval(resampled_ratios(numerators, denominators, draws), confidence)


def resampled_ratios(
    numerators: np.ndarray, denominators: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    """`Σ numerator / Σ denominator` in each resample, NaN where the denominator is zero."""
    totals = draws @ np.asarray(denominators, dtype=float)

    return np.divide(
        draws @ np.asarray(numerators, dtype=float),
        totals,
        out=np.full(len(draws), np.nan),
        where=totals > 0,
    )


def percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """The central `confidence` interval of `values`, NaN ones left out; NaN when none are left."""
    if np.isnan(values).all():
        return float("nan"), float("nan")

    tail = (1 - confidence) / 2

    return float(np.nanquantile(values, tail)), float(np.nanquantile(values, 1 - tail))


def bootstrap_intervals(
    documents: pd.DataFrame, draws: np.ndarray, confidence: float
) -> pd.DataFrame:
    """
    Intervals for the overall strict, character and entity-weighted character P, R and F1.

    Strict and character F1 is `2·TP / (2·TP + FP + FN)`, a ratio of totals, so it is
    resampled the same way as precision and recall: its numerator counts true
    positives twice. Entity-weighted F1 is the harmonic mean of two ratios, not one,
    so it is computed from each resample's entity-weighted precision and recall. The estimate is the
    value on all documents.
    """
    rows = []

    for metric, (numerator, denominator_columns) in INTERVAL_METRICS.items():
        weight = 2 if metric.endswith("f1") else 1
        numerators = weight * documents[numerator].to_numpy()
        denominators = documents[list(denominator_columns)].sum(axis=1).to_numpy()
        denominators = denominators + (weight - 1) * documents[numerator].to_numpy()
        total = denominators.sum()
        low, high = ratio_interval(numerators, denominators, draws, confidence)

        rows.append(
            {
                "metric": metric,
                "estimate": numerators.sum() / total if total else float("nan"),
                "low": low,
                "high": high,
            }
        )

    rows.append(_entity_char_f1_interval(documents, draws, confidence))

    return pd.DataFrame(rows)


def _entity_char_f1_interval(
    documents: pd.DataFrame, draws: np.ndarray, confidence: float
) -> dict[str, Any]:
    all_documents = np.ones((1, len(documents)), dtype=int)
    resamples = np.vstack([draws, all_documents])
    precision = resampled_ratios(
        documents["entity_char_predicted_credit"].to_numpy(),
        documents["entity_char_predicted"].to_numpy(),
        resamples,
    )
    recall = resampled_ratios(
        documents["entity_char_gold_credit"].to_numpy(),
        documents["entity_char_gold"].to_numpy(),
        resamples,
    )
    summed = precision + recall

    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(summed == 0, 0.0, 2 * precision * recall / summed)

    low, high = percentile_interval(f1[:-1], confidence)

    return {"metric": "entity_char_f1", "estimate": float(f1[-1]), "low": low, "high": high}
