"""The figures' shared look: Okabe–Ito colours fixed per meaning, sizes, and the marks they reuse."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREEN = "#009E73"
TEXT = "#1A1A1A"
MUTED = "#5F5F5F"
RULE = "#9E9E9E"
GRID = "#E6E6E6"
CONTEXT = "#BDBDBD"
SURFACE = "#FFFFFF"

OUTCOME_COLORS = {
    "CORRECT": BLUE,
    "BOUNDARY_ERROR": ORANGE,
    "MISSED": VERMILLION,
    "SPURIOUS": PURPLE,
    "UNCREDITED_EXACT": SKY,
}
OUTCOME_LABELS = {
    "CORRECT": "Correct",
    "BOUNDARY_ERROR": "Boundary error",
    "MISSED": "Missed",
    "SPURIOUS": "Spurious",
    "UNCREDITED_EXACT": "Uncredited exact",
}
METRIC_COLORS = {"precision": BLUE, "recall": ORANGE, "f1": GREEN}
LEVEL_COLORS = {"strict": BLUE, "char": ORANGE, "dice": GREEN}

SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.2
ROW_HEIGHT = 0.28
BAR_HEIGHT = 0.62
MARKER_SIZE = 5.5
SEGMENT_GAP = 0.8
MIN_SEGMENT_LABEL = 0.08

STYLE = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 8,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": TEXT,
    "ytick.labelcolor": TEXT,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.handlelength": 1.2,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "savefig.facecolor": SURFACE,
}


def new_figure(
    width: float, height: float, rows: int = 1, columns: int = 1, **subplots
) -> tuple[Figure, np.ndarray]:
    """A figure of `width × height` inches with a grid of axes, always returned as an array."""
    figure = Figure(figsize=(width, height), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False, **subplots)

    return figure, axes


def value_grid(axes: Axes, axis: str) -> None:
    """Gridlines on the value axis only, as hairlines behind the marks."""
    axes.grid(True, axis=axis)
    axes.grid(False, axis="y" if axis == "x" else "x")


def interval_marks(
    axes: Axes,
    positions: Sequence[float],
    rows: pd.DataFrame,
    measure: str,
    color: str,
    horizontal: bool = True,
    label: str | None = None,
) -> None:
    """
    A dot per row at `rows[measure]`, with a whisker from its `_low` to its `_high` column.

    The form every rate with a bootstrap interval takes in these figures.
    """
    positions = np.asarray(positions, dtype=float)
    estimates = rows[measure].to_numpy(dtype=float)
    lines = axes.hlines if horizontal else axes.vlines
    lines(
        positions,
        rows[f"{measure}_low"],
        rows[f"{measure}_high"],
        color=color,
        linewidth=1.2,
        zorder=2,
    )

    x, y = (estimates, positions) if horizontal else (positions, estimates)
    axes.plot(
        x,
        y,
        linestyle="none",
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        color=color,
        label=label,
        zorder=3,
    )


def figure_title(figure: Figure, title: str) -> None:
    """The figure's bold title, flush left above every panel."""
    figure.suptitle(title, x=0.0, ha="left", fontweight="bold", fontsize=9)


def rate_axis(axes: Axes, axis: str, label: str) -> None:
    """A 0–1 rate axis with ticks at every 0.2 and its gridlines."""
    limits, ticks = (
        (axes.set_xlim, axes.set_xticks) if axis == "x" else (axes.set_ylim, axes.set_yticks)
    )
    limits(0.0, 1.0)
    ticks(np.linspace(0.0, 1.0, 6))
    (axes.set_xlabel if axis == "x" else axes.set_ylabel)(label)
    value_grid(axes, axis)


def bar_end_labels(axes: Axes, bars, texts: Sequence[str], pad: float) -> None:
    """Write each bar's value just past its end, in text ink."""
    for bar, text in zip(bars, texts):
        axes.text(
            bar.get_x() + bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            color=TEXT,
            fontsize=7,
        )


def segment_label_color(color: str) -> str:
    """White or near-black text, whichever reads on a fill of `color`."""
    red, green, blue = (int(color[index : index + 2], 16) / 255 for index in (1, 3, 5))

    return TEXT if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.5 else SURFACE


def support_label(name: str, support: int) -> str:
    """A category label carrying its support: `Seen (n=123)`."""
    return f"{name} (n={support:,})"
