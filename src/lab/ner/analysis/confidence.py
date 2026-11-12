"""Prediction confidence: its spread by outcome, its calibration, and what filtering on it costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.ner.evaluation.scoring import span_metrics

MIN_SCORE_RANGE = 0.01
CALIBRATION_BINS = 10
SWEEP_QUANTILES = np.linspace(0.0, 1.0, 21)
SCORE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def has_varying_scores(predicted: pd.DataFrame) -> bool:
    """
    Whether predictions carry a `score` that varies enough to analyse.

    A spread under `MIN_SCORE_RANGE` says nothing about which predictions to trust,
    and every confidence table and figure is skipped.
    """
    if "score" not in predicted.columns:
        return False

    scores = pd.to_numeric(predicted["score"], errors="coerce").dropna()

    return len(scores) > 1 and scores.max() - scores.min() >= MIN_SCORE_RANGE


def confidence_by_outcome(predicted_mentions: pd.DataFrame) -> pd.DataFrame:
    """The distribution of `score` among the predictions of each outcome: count, mean, quantiles."""
    rows = []

    for outcome, members in predicted_mentions.groupby("outcome", sort=True):
        scores = members["score"].dropna()
        quantiles = scores.quantile(list(SCORE_QUANTILES))

        rows.append(
            {
                "outcome": outcome,
                "count": len(scores),
                "mean": scores.mean(),
                **{f"q{round(q * 100):02d}": quantiles[q] for q in SCORE_QUANTILES},
            }
        )

    return pd.DataFrame(rows)


def calibration(predicted_mentions: pd.DataFrame, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """
    How often a prediction is a strict match, against its score, in equal-count bins.

    Scores usually crowd near 1, so equal-width bins would leave most of them in
    one; quantile bins give each point about the same support. A calibrated model
    puts every row's `exact_share` near its `mean_score`.
    """
    scored = predicted_mentions.dropna(subset=["score"])
    binned = pd.qcut(scored["score"], q=bins, duplicates="drop")
    exact = scored["outcome"] == "CORRECT"

    table = (
        pd.DataFrame({"bin": binned, "score": scored["score"], "exact": exact})
        .groupby("bin", observed=True)
        .agg(count=("exact", "size"), mean_score=("score", "mean"), exact_share=("exact", "mean"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)

    return table


def confidence_sweep(
    gold: pd.DataFrame, predicted: pd.DataFrame, target_label: str, min_overlap_percentage: float
) -> pd.DataFrame:
    """
    Strict precision, recall and F1 when predictions below each threshold are dropped.

    Thresholds are the scores' 0th to 100th percentiles in steps of 5, so they
    follow the scores wherever they crowd. Each one is scored afresh: dropping a
    prediction can change which one nervaluate credits.
    """
    scores = pd.to_numeric(predicted["score"], errors="coerce")
    rows = []

    for threshold in np.unique(np.quantile(scores.dropna(), SWEEP_QUANTILES)):
        kept = predicted[scores >= threshold]
        metrics = span_metrics(
            gold=gold,
            predicted=kept,
            tags=[target_label],
            min_overlap_percentage=min_overlap_percentage,
        )

        rows.append(
            {
                "threshold": float(threshold),
                "kept": len(kept),
                "kept_share": len(kept) / len(predicted),
                "strict_precision": metrics["span_strict_precision"],
                "strict_recall": metrics["span_strict_recall"],
                "strict_f1": metrics["span_strict_f1"],
            }
        )

    return pd.DataFrame(rows)
