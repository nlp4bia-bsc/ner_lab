"""Drawing every figure an analysis's tables can support, in one style, to one directory."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from lab.ner.analysis.figures.confidence import (
    calibration_figure,
    confidence_by_outcome_figure,
    confidence_sweep_figure,
)
from lab.ner.analysis.figures.errors import (
    boundary_kinds_figure,
    boundary_offsets_figure,
    error_surfaces_figure,
    error_taxonomy_figure,
    oracle_error_budget_figure,
    overlap_sensitivity_figure,
)
from lab.ner.analysis.figures.style import STYLE
from lab.ner.analysis.figures.subgroups import (
    exposure_figure,
    mention_properties_figure,
    mention_traits_figure,
    predicted_exposure_figure,
)
from lab.ner.analysis.figures.variability import documents_figure

FIGURE_FORMATS = ("svg", "pdf", "png")
FIGURES = (
    ("error_taxonomy", error_taxonomy_figure),
    ("exposure", exposure_figure),
    ("predicted_exposure", predicted_exposure_figure),
    ("mention_properties", mention_properties_figure),
    ("mention_traits", mention_traits_figure),
    ("boundary_kinds", boundary_kinds_figure),
    ("boundary_offsets", boundary_offsets_figure),
    ("overlap_sensitivity", overlap_sensitivity_figure),
    ("oracle_error_budget", oracle_error_budget_figure),
    ("error_surfaces", error_surfaces_figure),
    ("documents", documents_figure),
    ("confidence_by_outcome", confidence_by_outcome_figure),
    ("calibration", calibration_figure),
    ("confidence_sweep", confidence_sweep_figure),
)


def draw_figures(
    tables: dict[str, pd.DataFrame], figures_dir: str | Path, figure_format: str = "svg"
) -> list[Path]:
    """
    Draw each of `FIGURES` the tables can support into `figures_dir`, as `figure_format`.

    A figure with nothing to show — no boundary errors, one document, no varying
    score — is skipped; the paths returned are the figures written. Styling is
    applied for the drawing only and leaves matplotlib's global settings alone.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    with matplotlib.rc_context(STYLE):
        for name, draw in FIGURES:
            figure = draw(tables)

            if figure is None:
                continue

            path = figures_dir / f"{name}.{figure_format}"
            figure.savefig(path, format=figure_format)
            paths.append(path)

    return paths
