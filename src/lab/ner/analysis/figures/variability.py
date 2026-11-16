"""Figures of variability: how the score spreads across documents."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from lab.ner.analysis.figures.style import LEVEL_COLORS, SINGLE_COLUMN, new_figure, rate_axis

MIN_DOCUMENTS = 2


def documents_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """
    The empirical distribution of per-document strict and character F1.

    A step at x reads as the share of documents scoring x or less, so a long flat
    start shows a block of documents the model fails on.
    """
    documents = tables["documents"]

    if len(documents) < MIN_DOCUMENTS:
        return None

    figure, axes = new_figure(SINGLE_COLUMN, 2.5)
    panel = axes[0, 0]

    for level, column, label in (
        ("strict", "strict_f1", "Strict F1"),
        ("char", "char_f1", "Character F1"),
    ):
        values = np.sort(documents[column].to_numpy(dtype=float))
        shares = np.arange(1, len(values) + 1) / len(values)
        panel.step(
            np.concatenate([[0.0], values, [1.0]]),
            np.concatenate([[0.0], shares, [1.0]]),
            where="post",
            color=LEVEL_COLORS[level],
            label=f"{label} (median {np.median(values):.2f})",
        )

    rate_axis(panel, "x", "F1 of a document")
    rate_axis(panel, "y", "Share of documents at or below")
    panel.legend(loc="upper left")
    panel.set_title(f"Per-document F1 across {len(documents):,} documents")

    return figure
