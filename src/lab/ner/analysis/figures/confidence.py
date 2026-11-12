"""Figures of prediction confidence: by outcome, its calibration, and filtering on it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from lab.ner.analysis.figures.style import (
    BLUE,
    DOUBLE_COLUMN,
    MARKER_SIZE,
    METRIC_COLORS,
    MUTED,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    RULE,
    SINGLE_COLUMN,
    SURFACE,
    figure_title,
    new_figure,
    rate_axis,
    value_grid,
)
from lab.ner.analysis.subgroups import PREDICTED_OUTCOMES


def confidence_by_outcome_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """The distribution of `score` for correct, boundary-error, spurious and uncredited predictions."""
    if "confidence_by_outcome" not in tables:
        return None

    mentions = tables["predicted_mentions"].dropna(subset=["score"])
    figure, axes = new_figure(SINGLE_COLUMN, 2.5)
    panel = axes[0, 0]
    low = float(mentions["score"].min())

    for outcome in PREDICTED_OUTCOMES:
        scores = np.sort(
            mentions.loc[mentions["outcome"] == outcome, "score"].to_numpy(dtype=float)
        )

        if not len(scores):
            continue

        shares = np.arange(1, len(scores) + 1) / len(scores)
        panel.step(
            np.concatenate([[low], scores, [1.0]]),
            np.concatenate([[0.0], shares, [1.0]]),
            where="post",
            color=OUTCOME_COLORS[outcome],
            label=f"{OUTCOME_LABELS[outcome]} (n={len(scores):,})",
        )

    rate_axis(panel, "y", "Share of predictions at or below")
    panel.set_xlim(low - (1.0 - low) * 0.02, 1.0)
    panel.set_xlabel("Prediction score")
    value_grid(panel, "both")
    panel.legend(loc="upper left")
    panel.set_title("Prediction scores by outcome")

    return figure


def calibration_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """A reliability diagram: the share of strict matches against the mean score, per score bin."""
    if "calibration" not in tables:
        return None

    bins = tables["calibration"]
    figure, axes = new_figure(SINGLE_COLUMN, SINGLE_COLUMN * 0.9)
    panel = axes[0, 0]
    low = min(float(bins["mean_score"].min()), float(bins["exact_share"].min()))
    low = max(0.0, np.floor(low * 10) / 10)

    panel.plot([low, 1.0], [low, 1.0], color=RULE, linewidth=0.8, zorder=1)
    panel.plot(
        bins["mean_score"],
        bins["exact_share"],
        color=BLUE,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        zorder=3,
    )

    panel.set_xlim(low, 1.0)
    panel.set_ylim(low, 1.02)
    panel.set_aspect("equal")
    panel.set_xlabel("Mean prediction score in the bin")
    panel.set_ylabel("Share that are strict matches")
    value_grid(panel, "both")
    panel.set_title(
        f"Calibration, {len(bins)} bins of about {int(bins['count'].median()):,} predictions"
    )

    return figure


def confidence_sweep_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Strict P/R/F1 and the share of predictions kept, as the score threshold rises."""
    if "confidence_sweep" not in tables:
        return None

    sweep = tables["confidence_sweep"]
    figure, axes = new_figure(DOUBLE_COLUMN, 2.6, 1, 2, sharex=True)
    scores_panel, kept_panel = axes[0]
    thresholds = sweep["threshold"].to_numpy()

    best = sweep.loc[sweep["strict_f1"].idxmax()]
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": f"F1 (best {best['strict_f1']:.3f} at {best['threshold']:.3f})",
    }

    for metric, label in labels.items():
        scores_panel.plot(
            thresholds, sweep[f"strict_{metric}"], color=METRIC_COLORS[metric], label=label
        )

    scores_panel.plot(
        [best["threshold"]],
        [best["strict_f1"]],
        linestyle="none",
        marker="o",
        markersize=MARKER_SIZE,
        color=METRIC_COLORS["f1"],
        markeredgecolor=SURFACE,
    )
    rate_axis(scores_panel, "y", "Strict score")
    scores_panel.set_xlabel("Score threshold")
    scores_panel.legend(loc="lower left")
    scores_panel.set_title("Strict scores")

    kept_panel.plot(thresholds, sweep["kept_share"], color=MUTED)
    rate_axis(kept_panel, "y", "Share of predictions kept")
    kept_panel.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    kept_panel.set_xlabel("Score threshold")
    kept_panel.set_title("Predictions kept")
    kept_panel.set_xlim(thresholds.min(), thresholds.max())
    figure_title(figure, "Dropping predictions scored below a threshold")

    return figure
