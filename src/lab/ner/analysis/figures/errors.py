"""Figures of errors: their types, boundary errors, threshold sensitivity, the oracle budget, surfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from lab.ner.analysis.errors import ONE_TO_ONE_KINDS
from lab.ner.analysis.figures.style import (
    BAR_HEIGHT,
    BLUE,
    CONTEXT,
    DOUBLE_COLUMN,
    MARKER_SIZE,
    MUTED,
    ORANGE,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    RULE,
    SEGMENT_GAP,
    SINGLE_COLUMN,
    SURFACE,
    TEXT,
    bar_end_labels,
    new_figure,
    rate_axis,
    value_grid,
)

OFFSET_FOLD = 10
TOP_SURFACES = 15
KIND_LABELS = {
    ("TOO_LONG", "START"): "Too long, past the end",
    ("TOO_LONG", "END"): "Too long, before the start",
    ("TOO_LONG", "NEITHER"): "Too long, at both edges",
    ("TOO_SHORT", "START"): "Too short, ends early",
    ("TOO_SHORT", "END"): "Too short, starts late",
    ("TOO_SHORT", "NEITHER"): "Too short, at both edges",
    ("SHIFTED_LEFT", "NEITHER"): "Shifted left",
    ("SHIFTED_RIGHT", "NEITHER"): "Shifted right",
    ("FRAGMENTED", None): "Fragmented (gold split)",
    ("MERGED", None): "Merged (gold joined)",
}
KIND_GAP = 0.5
OFFSET_COLORS = LinearSegmentedColormap.from_list(
    "boundary_offsets", ["#F3C877", ORANGE, "#6B4100"]
)
QUADRANT_LABELS = (
    (0.03, 0.97, "left", "top", "Too long"),
    (0.97, 0.03, "right", "bottom", "Too short"),
    (0.03, 0.03, "left", "bottom", "Shifted left"),
    (0.97, 0.97, "right", "top", "Shifted right"),
)
SCENARIO_LABELS = {
    "fix_boundary_errors": "Fix boundary errors",
    "recover_missed": "Recover missed",
    "remove_spurious": "Remove spurious",
    "fix_uncredited_exact": "Credit uncredited exact",
    "fix_all": "Fix every error",
}
SENSITIVITY_SERIES = ("BOUNDARY_ERROR", "MISSED", "SPURIOUS")
SURFACE_ERRORS = ("BOUNDARY_ERROR", "MISSED", "SPURIOUS", "UNCREDITED_EXACT")


def error_taxonomy_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Diagnostic units by type, correct ones first, errors by count, each with its share."""
    taxonomy = tables["error_taxonomy"]
    taxonomy = taxonomy[taxonomy["count"] > 0].reset_index(drop=True)

    if taxonomy.empty:
        return None

    figure, axes = new_figure(SINGLE_COLUMN, 0.6 + 0.3 * len(taxonomy))
    panel = axes[0, 0]
    positions = np.arange(len(taxonomy))
    counts = taxonomy["count"].to_numpy()
    bars = panel.barh(
        positions,
        counts,
        height=BAR_HEIGHT,
        color=[OUTCOME_COLORS[kind] for kind in taxonomy["error_type"]],
    )

    labels = [
        f"{count:,} ({share:.0%})"
        if error_type == "CORRECT"
        else f"{count:,} ({error_share:.0%} of errors)"
        for error_type, count, share, error_share in taxonomy[
            ["error_type", "count", "share", "error_share"]
        ].itertuples(index=False)
    ]
    bar_end_labels(panel, bars, labels, pad=counts.max() * 0.02)
    panel.set_yticks(positions, [OUTCOME_LABELS[kind] for kind in taxonomy["error_type"]])
    panel.invert_yaxis()
    panel.set_xlim(0, counts.max() * 1.45)
    panel.set_xlabel("Diagnostic units")
    value_grid(panel, "x")
    panel.set_title("What the model got right and wrong")

    return figure


def boundary_kinds_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """
    Boundary errors by kind and by the edge they got right, one bar each, grouped by kind.

    A one-to-one kind splits into the prediction running past one edge, falling short
    of one, or missing both, so the edge the model gets wrong reads off the labels.
    """
    summary = tables["boundary_summary"]

    if summary.empty:
        return None

    edges = summary["matched_edge"].astype(object).where(summary["matched_edge"].notna(), None)
    labels = [KIND_LABELS[kind, edge] for kind, edge in zip(summary["boundary_kind"], edges)]
    new_kind = (summary["boundary_kind"] != summary["boundary_kind"].shift()).to_numpy()
    positions = np.arange(len(summary)) + KIND_GAP * (np.cumsum(new_kind) - 1)

    figure, axes = new_figure(SINGLE_COLUMN, 0.75 + 0.26 * (positions[-1] + 1))
    panel = axes[0, 0]
    counts = summary["count"].to_numpy()
    bars = panel.barh(positions, counts, height=BAR_HEIGHT, color=ORANGE)

    bar_end_labels(
        panel,
        bars,
        [f"{count:,} ({share:.0%})" for count, share in zip(counts, summary["share"])],
        pad=counts.max() * 0.02,
    )
    panel.set_yticks(positions, labels)
    panel.invert_yaxis()
    panel.set_xlim(0, counts.max() * 1.35)
    panel.set_xlabel("Boundary errors")
    value_grid(panel, "x")
    panel.set_title("How predicted boundaries miss the gold")

    return figure


def boundary_offsets_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """
    Start against end offset of each one-to-one boundary error, as counts on a grid.

    Offsets are prediction minus gold, folded at ±`OFFSET_FOLD`. The vertical line
    through zero holds the errors that got the start right, the horizontal one those
    that got the end right, and the quadrants are the kinds.
    """
    boundary = tables["boundary_errors"]
    boundary = boundary[boundary["boundary_kind"].isin(ONE_TO_ONE_KINDS)]

    if boundary.empty:
        return None

    edges = np.arange(-OFFSET_FOLD, OFFSET_FOLD + 2) - 0.5
    counts, _, _ = np.histogram2d(
        boundary["start_offset"].astype(int).clip(-OFFSET_FOLD, OFFSET_FOLD),
        boundary["end_offset"].astype(int).clip(-OFFSET_FOLD, OFFSET_FOLD),
        bins=[edges, edges],
    )
    counts = np.ma.masked_equal(counts.T, 0)

    figure, axes = new_figure(SINGLE_COLUMN + 0.5, SINGLE_COLUMN)
    panel = axes[0, 0]
    cells = panel.pcolormesh(
        edges,
        edges,
        counts,
        cmap=OFFSET_COLORS,
        norm=LogNorm(vmin=1, vmax=max(float(counts.max()), 2.0)),
        edgecolors=SURFACE,
        linewidth=0.3,
    )
    panel.axvline(0, color=RULE, linewidth=0.5, zorder=3)
    panel.axhline(0, color=RULE, linewidth=0.5, zorder=3)
    panel.text(0.4, -OFFSET_FOLD, "start right", ha="left", va="center", fontsize=7, color=MUTED)
    panel.text(-OFFSET_FOLD, 0.4, "end right", ha="left", va="bottom", fontsize=7, color=MUTED)

    for x, y, ha, va, text in QUADRANT_LABELS:
        panel.text(x, y, text, transform=panel.transAxes, ha=ha, va=va, fontsize=7, color=MUTED)

    ticks = [-OFFSET_FOLD, -5, 0, 5, OFFSET_FOLD]
    tick_labels = [f"≤−{OFFSET_FOLD}", "−5", "0", "5", f"≥{OFFSET_FOLD}"]
    panel.set_xticks(ticks, tick_labels)
    panel.set_yticks(ticks, tick_labels)
    panel.set_aspect("equal")
    panel.set_xlabel("Start offset, prediction − gold (characters)")
    panel.set_ylabel("End offset, prediction − gold (characters)")
    panel.set_title("Where predicted edges fall")

    colorbar = figure.colorbar(cells, ax=panel, shrink=0.8)
    colorbar.set_label("One-to-one boundary errors")
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    return figure


def overlap_sensitivity_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Units of each error type as the Dice threshold for pairing moves, the chosen one marked."""
    sensitivity = tables["overlap_sensitivity"]

    if len(sensitivity) < 2:
        return None

    figure, axes = new_figure(SINGLE_COLUMN, 2.5)
    panel = axes[0, 0]
    thresholds = sensitivity["threshold"].to_numpy()

    for error_type in SENSITIVITY_SERIES:
        counts = sensitivity[error_type.lower()].to_numpy()
        panel.plot(
            thresholds,
            counts,
            color=OUTCOME_COLORS[error_type],
            marker="o",
            markersize=MARKER_SIZE - 1.5,
            markeredgecolor=SURFACE,
            label=OUTCOME_LABELS[error_type],
        )

    chosen = float(sensitivity.loc[sensitivity["chosen"], "threshold"].iloc[0])
    panel.axvline(chosen, color=RULE, linewidth=0.8, zorder=0)
    panel.annotate(
        f"chosen {chosen:.2f}",
        (chosen, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, 3),
        textcoords="offset points",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )
    panel.set_xticks(thresholds)
    panel.set_xlabel("Dice needed to pair a gold span\nwith a prediction")
    panel.set_ylabel("Diagnostic units")
    panel.set_ylim(bottom=0)
    value_grid(panel, "y")
    panel.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    panel.set_title("How the error types depend on the threshold")

    return figure


def oracle_error_budget_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Observed strict F1 against the F1 with each kind of error fixed, largest gain first."""
    budget = tables["oracle_error_budget"]
    budget = (
        budget[budget["units"] > 0]
        .sort_values("gain", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    if budget.empty:
        return None

    figure, axes = new_figure(SINGLE_COLUMN, 0.75 + 0.3 * len(budget))
    panel = axes[0, 0]
    positions = np.arange(len(budget))
    observed = float(budget["observed_f1"].iloc[0])

    panel.hlines(positions, observed, budget["oracle_f1"], color=CONTEXT, linewidth=2.0, zorder=1)
    panel.plot(
        np.full(len(budget), observed),
        positions,
        linestyle="none",
        marker="o",
        markersize=MARKER_SIZE,
        color=CONTEXT,
        markeredgecolor=SURFACE,
        label="Observed",
        zorder=2,
    )
    panel.plot(
        budget["oracle_f1"],
        positions,
        linestyle="none",
        marker="o",
        markersize=MARKER_SIZE,
        color=BLUE,
        markeredgecolor=SURFACE,
        label="With the errors fixed",
        zorder=3,
    )

    for position, oracle, gain in zip(positions, budget["oracle_f1"], budget["gain"]):
        panel.text(
            oracle + 0.015, position, f"+{gain:.3f}", va="center", ha="left", fontsize=7, color=TEXT
        )

    rate_axis(panel, "x", "Strict F1")
    panel.set_xlim(max(0.0, observed - 0.1), 1.08)
    panel.set_yticks(positions, [SCENARIO_LABELS[scenario] for scenario in budget["scenario"]])
    panel.invert_yaxis()
    panel.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.0)
    panel.set_title("Strict F1 if one kind of error were fixed", pad=18)

    return figure


def error_surfaces_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """The surfaces most often in an error, stacked by error type."""
    surfaces = tables["error_surfaces"].head(TOP_SURFACES)

    if surfaces.empty:
        return None

    figure, axes = new_figure(DOUBLE_COLUMN, 0.8 + 0.24 * len(surfaces))
    panel = axes[0, 0]
    positions = np.arange(len(surfaces))
    left = np.zeros(len(surfaces))
    present = [
        error_type for error_type in SURFACE_ERRORS if surfaces[error_type.lower()].sum() > 0
    ]

    for error_type in present:
        counts = surfaces[error_type.lower()].to_numpy()
        panel.barh(
            positions,
            counts,
            left=left,
            height=BAR_HEIGHT,
            color=OUTCOME_COLORS[error_type],
            edgecolor=SURFACE,
            linewidth=SEGMENT_GAP,
            label=OUTCOME_LABELS[error_type],
        )
        left += counts

    for position, total in zip(positions, surfaces["total"]):
        panel.text(
            total + left.max() * 0.01,
            position,
            f"{total:,}",
            va="center",
            ha="left",
            fontsize=7,
            color=TEXT,
        )

    panel.set_yticks(positions, [_shorten(surface) for surface in surfaces["surface"]])
    panel.invert_yaxis()
    panel.set_xlim(0, left.max() * 1.08)
    panel.set_xlabel("Errors involving the surface")
    value_grid(panel, "x")
    panel.legend(loc="lower right")
    panel.set_title(f"The {len(surfaces)} surfaces most often in an error (normalized)")

    return figure


def _shorten(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"
