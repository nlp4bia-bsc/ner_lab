"""Publication-quality SVG figures for NER error and generalization analysis.

The plotting layer is intentionally analytical rather than decorative: every
figure answers a specific model-diagnostic question, exposes support whenever
possible, avoids rotated category labels, and suppresses plots that would be
misleading because the underlying data have negligible variation.
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Okabe-Ito palette + neutral greys. Semantic assignments below stay stable
# across figures so readers do not have to relearn the visual language.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#111111"
DARK_GREY = "#4D4D4D"
MID_GREY = "#8C8C8C"
LIGHT_GREY = "#D9D9D9"
PALE_GREY = "#F2F2F2"
WHITE = "#FFFFFF"

METRIC_COLORS = {
    "Precision": BLUE,
    "Recall": ORANGE,
    "F1": GREEN,
}

ERROR_COLORS = {
    "CORRECT": GREEN,
    "MISSED": VERMILLION,
    "SPURIOUS": PURPLE,
    "BOUNDARY_ERROR": SKY,
    "LABEL_ERROR": ORANGE,
    "BOUNDARY_AND_LABEL_ERROR": YELLOW,
    "UNKNOWN": MID_GREY,
}

ERROR_LABELS = {
    "CORRECT": "Correct",
    "MISSED": "Missed",
    "SPURIOUS": "Spurious",
    "BOUNDARY_ERROR": "Boundary error",
    "LABEL_ERROR": "Label error",
    "BOUNDARY_AND_LABEL_ERROR": "Boundary + label",
    "UNKNOWN": "Unknown",
}

GENERALIZATION_LABELS = {
    "SEEN": "Seen",
    "FEW_SHOT": "Few-shot",
    "ZERO_SHOT": "Zero-shot",
}

ORACLE_LABELS = {
    "recover_all_missed": "Recover missed entities",
    "remove_all_spurious": "Remove spurious entities",
    "boundary_correction": "Fix boundary errors",
    "label_correction": "Fix label errors",
    "boundary_and_label_correction": "Fix boundary + label errors",
    "all_diagnostic_errors": "Fix all diagnosable errors",
}

MIN_RELIABLE_SUPPORT = 10
MIN_CONFIDENCE_RANGE = 0.01


@dataclass(frozen=True)
class PublicationTheme:
    """Consistent single/double-column-friendly style for SVG output."""

    width: float = 7.25
    height: float = 4.7
    base_font_size: float = 9.5
    title_font_size: float = 12.5
    subtitle_font_size: float = 8.5
    axis_label_size: float = 10.0
    tick_size: float = 9.0
    legend_size: float = 8.8
    annotation_size: float = 8.2
    spine_width: float = 0.9
    line_width: float = 1.8
    marker_size: float = 7.0


THEME = PublicationTheme()


@dataclass(frozen=True)
class PlotDecision:
    filename: str
    status: str
    caption: str
    reason: str | None = None


FIGURE_CAPTIONS: dict[str, str] = {
    "generalization_strict_prf.svg": (
        "STRICT precision, recall and F1 across lexical exposure groups. Gold-support counts "
        "and proportions are shown with each group; F1 markers include document-bootstrap "
        "confidence intervals when available, and deltas quantify the generalization gap from seen mentions."
    ),
    "generalization_character_prf.svg": (
        "Character-level precision, recall and F1 across lexical exposure groups, with gold-support "
        "counts/proportions and document-bootstrap confidence intervals for F1 when available."
    ),
    "generalization_error_profile.svg": (
        "Gold-conditioned outcome composition across seen, few-shot and zero-shot mentions. "
        "The 100% stacked bars separate correct detections from missed, boundary, label and combined errors."
    ),
    "generalization_strict_f1_by_label.svg": (
        "STRICT F1 by entity label and lexical exposure group. Cells report F1 and gold support; "
        "low-support cells remain visible so uncertainty is not hidden."
    ),
    "generalization_character_f1_by_label.svg": (
        "Character F1 by entity label and lexical exposure group. Cells report F1 and gold support."
    ),
    "similarity_vs_strict_f1.svg": (
        "STRICT F1 across lexical-similarity bins to training mentions. Discrete lollipops avoid implying "
        "a continuous trend; support is printed for every bin and low-support bins use hollow markers."
    ),
    "training_frequency_vs_strict_f1.svg": (
        "STRICT F1 by exact training-mention frequency. Support is reported for every bucket and low-support "
        "buckets are visually flagged."
    ),
    "entity_length_strict_f1.svg": (
        "STRICT F1 by mention word length, exposing whether span complexity increases with longer mentions."
    ),
    "structural_effects_strict_f1.svg": (
        "Difference in STRICT F1 between mentions with and without structural properties (acronym-like, "
        "overlapping or nested). Negative values indicate a performance penalty when the property is present."
    ),
    "error_taxonomy.svg": (
        "Diagnostic event composition. Counts and percentages distinguish correct detections from missed, "
        "spurious, boundary and label-related errors without changing official STRICT scoring."
    ),
    "error_type_pareto.svg": (
        "Errors-only Pareto view ordered by frequency. Each bar reports the error count, share of all errors "
        "and cumulative share, identifying the highest-yield failure modes."
    ),
    "error_surface_pareto.svg": (
        "Most recurrent mention surfaces among diagnostic errors. Cumulative proportions reveal whether "
        "a small lexical set accounts for a large fraction of failures."
    ),
    "span_start_delta.svg": (
        "Signed start-boundary deviation (prediction minus gold). Negative values start too early; positive "
        "values start too late. Tail offsets are collapsed only when necessary for legibility."
    ),
    "span_end_delta.svg": (
        "Signed end-boundary deviation (prediction minus gold). Negative values end too early; positive values "
        "end too late."
    ),
    "boundary_absolute_delta.svg": (
        "Absolute start/end boundary deviation among boundary errors. Offsets are grouped into interpretable "
        "character-distance bins to expose whether failures are predominantly near-misses."
    ),
    "boundary_relations.svg": (
        "Geometric relation between paired gold and predicted spans for boundary-related errors."
    ),
    "label_confusion_counts.svg": (
        "Exact-boundary label confusion matrix in counts. Text is kept horizontal and cell values are printed explicitly."
    ),
    "label_confusion_normalized.svg": (
        "Row-normalized exact-boundary label confusion matrix, showing where each gold label is reassigned."
    ),
    "label_confusion_extended.svg": (
        "Extended label-flow matrix including missed gold entities and spurious predictions."
    ),
    "oracle_error_budget.svg": (
        "Counterfactual STRICT F1 ceilings for independently correcting each diagnosable error family. "
        "Dumbbells connect the observed baseline to each oracle score; scenarios are not treated as additive."
    ),
    "dataset_shift_labels.svg": (
        "Training versus evaluation label composition as 100% stacked bars. Segment labels report meaningful "
        "shares, and Jensen-Shannon distance is displayed when available."
    ),
    "confidence_by_error.svg": (
        "Prediction-confidence distribution summaries by error type: min-max range, interquartile range, median "
        "and mean. This plot is omitted automatically when score variation is negligible."
    ),
    "confidence_threshold_tradeoff.svg": (
        "STRICT F1 and retained-prediction coverage across confidence thresholds. The plot is omitted when "
        "confidence scores contain too little variation to support threshold analysis."
    ),
    "document_performance_distribution.svg": (
        "Distribution of per-document STRICT and character F1. Box summaries and individual-document points "
        "expose heterogeneity that aggregate corpus metrics can hide."
    ),
}


def write_analysis_figures(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    multiclass: bool,
) -> list[Path]:
    """Generate analytical, publication-quality SVGs from persisted tables.

    The function preserves the historical filenames where possible and adds
    complementary diagnostic figures. It also writes ``figures/plot_manifest.json``
    recording both generated and deliberately omitted plots so a constant or
    underpowered signal is never silently presented as informative.
    """
    root = Path(output_dir) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    _remove_previous_generated(root)

    figures: list[Path] = []
    decisions: list[PlotDecision] = []

    def generate(filename: str, builder, *args, caption: str | None = None, **kwargs) -> None:
        path = root / filename
        builder(*args, path=path, **kwargs)
        figures.append(path)
        decisions.append(
            PlotDecision(
                filename=filename,
                status="generated",
                caption=caption or FIGURE_CAPTIONS.get(filename, "Generated NER analysis figure."),
            )
        )

    def omit(filename: str, reason: str) -> None:
        decisions.append(
            PlotDecision(
                filename=filename,
                status="omitted",
                caption=FIGURE_CAPTIONS.get(filename, "NER analysis figure."),
                reason=reason,
            )
        )

    bootstrap = tables.get("bootstrap_intervals", pd.DataFrame())
    overall = tables.get("overall_metrics", pd.DataFrame())

    generalization = tables.get("metrics_by_generalization", pd.DataFrame())
    if _has_rows(generalization):
        generate(
            "generalization_strict_prf.svg",
            generalization_prf_figure,
            generalization,
            prefix="strict",
            bootstrap=bootstrap,
            title="Lexical generalization performance — STRICT",
        )
        generate(
            "generalization_character_prf.svg",
            generalization_prf_figure,
            generalization,
            prefix="char",
            bootstrap=bootstrap,
            title="Lexical generalization performance — character",
        )
    else:
        omit("generalization_strict_prf.svg", "No generalization table was available.")
        omit("generalization_character_prf.svg", "No generalization table was available.")

    generalization_profile = tables.get("generalization_error_profile", pd.DataFrame())
    if _has_rows(generalization_profile):
        generate(
            "generalization_error_profile.svg",
            generalization_error_profile_figure,
            generalization_profile,
            title="What fails when lexical novelty increases?",
        )
    else:
        omit("generalization_error_profile.svg", "No gold-conditioned generalization error profile was available.")

    label_generalization = tables.get("metrics_by_label_generalization", pd.DataFrame())
    if multiclass and _has_rows(label_generalization):
        generate(
            "generalization_strict_f1_by_label.svg",
            label_generalization_heatmap,
            label_generalization,
            metric="strict_f1",
            title="STRICT F1 by label and lexical exposure",
        )
        generate(
            "generalization_character_f1_by_label.svg",
            label_generalization_heatmap,
            label_generalization,
            metric="char_f1",
            title="Character F1 by label and lexical exposure",
        )
    else:
        reason = "Single-label evaluation; a label-by-generalization matrix would be redundant." if not multiclass else "No label-generalization table was available."
        omit("generalization_strict_f1_by_label.svg", reason)
        omit("generalization_character_f1_by_label.svg", reason)

    omit("similarity_vs_strict_f1.svg", "Disabled by design: lexical-similarity subgroup plots were judged redundant and visually weak for this analysis profile.")

    frequency = tables.get("metrics_by_training_frequency", pd.DataFrame())
    if _has_rows(frequency):
        generate(
            "training_frequency_vs_strict_f1.svg",
            training_frequency_figure,
            frequency,
            overall=_overall_score(overall, "strict"),
            title="Training exposure frequency and STRICT F1",
        )

    entity_length = tables.get("metrics_by_entity_length", pd.DataFrame())
    if _has_rows(entity_length) and entity_length["support"].sum() > 0:
        generate(
            "entity_length_strict_f1.svg",
            subgroup_lollipop_figure,
            entity_length,
            category="value",
            metric="strict_f1",
            overall=_overall_score(overall, "strict"),
            title="Mention length and STRICT F1",
            xlabel="Mention length (words)",
        )

    structural_tables = {
        "Acronym-like": tables.get("metrics_by_acronym", pd.DataFrame()),
        "Overlapping": tables.get("metrics_by_overlap", pd.DataFrame()),
        "Nested": tables.get("metrics_by_nested", pd.DataFrame()),
    }
    structural = _structural_effect_frame(structural_tables)
    if _has_rows(structural):
        generate(
            "structural_effects_strict_f1.svg",
            structural_effects_figure,
            structural,
            title="Structural mention properties and STRICT F1",
        )

    taxonomy = tables.get("error_taxonomy", pd.DataFrame())
    if _has_rows(taxonomy):
        generate(
            "error_taxonomy.svg",
            error_taxonomy_figure,
            taxonomy,
            title="NER diagnostic outcome taxonomy",
        )

    error_type_pareto = tables.get("error_type_pareto", pd.DataFrame())
    if _has_rows(error_type_pareto):
        generate(
            "error_type_pareto.svg",
            error_type_pareto_figure,
            error_type_pareto,
            title="Which error families dominate?",
        )

    surface_pareto = tables.get("error_pareto", pd.DataFrame())
    if _has_rows(surface_pareto):
        generate(
            "error_surface_pareto.svg",
            error_surface_pareto_figure,
            surface_pareto,
            title="Recurrent mention surfaces among errors",
        )

    boundary = tables.get("boundary_errors", pd.DataFrame())
    if _has_rows(boundary):
        generate(
            "span_start_delta.svg",
            signed_boundary_figure,
            boundary["span_start_delta"],
            title="Start-boundary deviation",
            explanation="Prediction − gold; negative = starts too early, positive = starts too late",
        )
        generate(
            "span_end_delta.svg",
            signed_boundary_figure,
            boundary["span_end_delta"],
            title="End-boundary deviation",
            explanation="Prediction − gold; negative = ends too early, positive = ends too late",
        )
        generate(
            "boundary_absolute_delta.svg",
            absolute_boundary_figure,
            boundary,
            title="How large are boundary errors?",
        )
        omit("boundary_relations.svg", "Disabled by design: span-relation panels were judged low-value compared with deviation-range summaries.")
    else:
        for filename in ("span_start_delta.svg", "span_end_delta.svg", "boundary_absolute_delta.svg", "boundary_relations.svg"):
            omit(filename, "No boundary-related diagnostic pairs were available.")

    if multiclass:
        counts = tables.get("label_confusion_counts", pd.DataFrame())
        normalized = tables.get("label_confusion_normalized", pd.DataFrame())
        extended = tables.get("label_confusion_extended", pd.DataFrame())
        if _has_rows(counts):
            generate(
                "label_confusion_counts.svg",
                confusion_heatmap,
                counts,
                value="count",
                title="Exact-boundary label confusion — counts",
                normalized=False,
            )
        if _has_rows(normalized):
            generate(
                "label_confusion_normalized.svg",
                confusion_heatmap,
                normalized,
                value="proportion",
                title="Exact-boundary label confusion — row normalized",
                normalized=True,
            )
        if _has_rows(extended):
            generate(
                "label_confusion_extended.svg",
                extended_confusion_heatmap,
                extended,
                title="Label flow including missed and spurious entities",
            )
    else:
        for filename in ("label_confusion_counts.svg", "label_confusion_normalized.svg", "label_confusion_extended.svg"):
            omit(filename, "Single-label evaluation; a label confusion matrix is not informative.")

    oracle = tables.get("oracle_error_budget", pd.DataFrame())
    if _has_rows(oracle):
        generate(
            "oracle_error_budget.svg",
            oracle_dumbbell_figure,
            oracle,
            title="Counterfactual STRICT F1 error budget",
        )

    shift = tables.get("dataset_shift_labels", pd.DataFrame())
    shift_summary = tables.get("dataset_shift_summary", pd.DataFrame())
    if _has_rows(shift):
        generate(
            "dataset_shift_labels.svg",
            dataset_shift_figure,
            shift,
            summary=shift_summary,
            title="Label composition shift: training vs evaluation",
        )

    confidence = tables.get("confidence_by_error", pd.DataFrame())
    confidence_summary = tables.get("confidence_summary", pd.DataFrame())
    confidence_range = _confidence_range(confidence_summary, confidence)
    if _has_rows(confidence) and confidence_range >= MIN_CONFIDENCE_RANGE:
        generate(
            "confidence_by_error.svg",
            confidence_distribution_figure,
            confidence,
            title="Prediction confidence by diagnostic outcome",
        )
    elif _has_rows(confidence):
        omit(
            "confidence_by_error.svg",
            f"Prediction scores span only {confidence_range:.4f}; the confidence signal is effectively constant and a plot would be misleading.",
        )

    sweep = tables.get("confidence_threshold_sweep", pd.DataFrame())
    if _has_rows(sweep) and confidence_range >= MIN_CONFIDENCE_RANGE:
        generate(
            "confidence_threshold_tradeoff.svg",
            confidence_threshold_figure,
            sweep,
            title="Confidence threshold: performance–coverage trade-off",
        )
    elif _has_rows(sweep):
        omit(
            "confidence_threshold_tradeoff.svg",
            "Confidence scores have negligible variation, so threshold sweeps do not provide a meaningful operating-point analysis.",
        )

    documents = tables.get("document_metrics", pd.DataFrame())
    if _has_rows(documents) and len(documents) >= 3:
        generate(
            "document_performance_distribution.svg",
            document_distribution_figure,
            documents,
            title="Performance heterogeneity across documents",
        )

    manifest_path = root / "plot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "theme": asdict(THEME),
                "generated": [Path(path).name for path in figures],
                "decisions": [asdict(decision) for decision in decisions],
                "design": {
                    "vector_format": "SVG",
                    "palette": "Okabe-Ito",
                    "rotated_category_tick_labels": False,
                    "top_spine": False,
                    "right_spine": False,
                    "default_grid": False,
                    "low_support_threshold": MIN_RELIABLE_SUPPORT,
                    "confidence_range_required": MIN_CONFIDENCE_RANGE,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return figures


def write_figure_captions(output_dir: str | Path, figures: Sequence[Path]) -> Path:
    """Write informative captions plus reasons for deliberately omitted plots."""
    root = Path(output_dir)
    manifest_path = root / "figures" / "plot_manifest.json"
    decisions: list[dict] = []
    if manifest_path.exists():
        decisions = json.loads(manifest_path.read_text(encoding="utf-8")).get("decisions", [])

    generated = {Path(path).name for path in figures}
    lines = [
        "# Figure captions",
        "",
        (
            "Figures are deterministic SVG vectors designed for scientific publication: Okabe–Ito color-blind-safe semantics, "
            "horizontal category labels, no top/right spines, no default grid, direct numerical annotation where useful, and explicit support/uncertainty cues."
        ),
        "",
    ]
    for filename in sorted(generated):
        caption = FIGURE_CAPTIONS.get(filename, "Generated from persisted NER analysis tables.")
        lines.append(f"- **{filename}** — {caption}")

    omitted = [entry for entry in decisions if entry.get("status") == "omitted"]
    if omitted:
        lines.extend(["", "## Deliberately omitted figures", ""])
        for entry in omitted:
            lines.append(f"- **{entry['filename']}** — {entry.get('reason', 'Not informative for this run.')}")

    path = root / "captions.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generalization_prf_figure(
    frame: pd.DataFrame,
    *,
    prefix: str,
    bootstrap: pd.DataFrame,
    title: str,
    path: str | Path,
) -> Path:
    order = [value for value in ("SEEN", "FEW_SHOT", "ZERO_SHOT") if value in set(frame["value"].astype(str))]
    ordered = frame.set_index(frame["value"].astype(str)).reindex(order).reset_index(drop=True)
    supports = pd.to_numeric(ordered.get("support", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    total = max(int(supports.sum()), 1)

    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, THEME.height))
        x = np.arange(len(order), dtype=float)
        width = 0.24
        metrics = [
            ("Precision", f"{prefix}_precision", -width),
            ("Recall", f"{prefix}_recall", 0.0),
            ("F1", f"{prefix}_f1", width),
        ]
        for name, column, offset in metrics:
            values = pd.to_numeric(ordered[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                values,
                width=width * 0.9,
                label=name,
                color=METRIC_COLORS[name],
                edgecolor=WHITE,
                linewidth=0.8,
                zorder=2,
            )
            for rect, value in zip(bars, values):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    min(value + 0.02, 1.03),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=THEME.annotation_size,
                    color=BLACK,
                    fontweight="bold" if name == "F1" else "normal",
                )

        f1_values = pd.to_numeric(ordered[f"{prefix}_f1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.scatter(x + width, f1_values, s=36, marker="D", color=BLACK, zorder=4, label="F1 point estimate")

        seen_f1 = f1_values[0] if len(f1_values) and np.isfinite(f1_values[0]) else None
        for idx, value in enumerate(f1_values):
            if seen_f1 is not None and idx > 0:
                ax.text(
                    x[idx] + width,
                    min(value + 0.105, 1.07),
                    f"Δ vs Seen\n{value - seen_f1:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=THEME.annotation_size - 0.15,
                    color=DARK_GREY,
                    linespacing=1.0,
                )

        labels = []
        for name, support in zip(order, supports):
            share = 100.0 * int(support) / total
            labels.append(f"{GENERALIZATION_LABELS.get(name, name)}\n(n={int(support)}, {share:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylim(0.0, 1.11)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_ylabel("Score")
        ax.set_title(title, loc="left", pad=26)
        handles = [Patch(facecolor=METRIC_COLORS[name], edgecolor="none", label=name) for name in ("Precision", "Recall", "F1")]
        ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
        _clean_axes(ax)
        return _save(fig, path)



def generalization_error_profile_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    order = [value for value in ("SEEN", "FEW_SHOT", "ZERO_SHOT") if value in set(frame["generalization_class"].astype(str))]
    error_order = [
        value
        for value in ("CORRECT", "MISSED", "BOUNDARY_ERROR", "LABEL_ERROR", "BOUNDARY_AND_LABEL_ERROR")
        if value in set(frame["error_type"].astype(str))
    ]
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 4.2))
        y = np.arange(len(order))
        left = np.zeros(len(order), dtype=float)
        supports = []
        for group in order:
            subset = frame.loc[frame["generalization_class"].astype(str) == group]
            supports.append(int(pd.to_numeric(subset["count"], errors="coerce").fillna(0).sum()))
        for error in error_order:
            values = []
            for group, support in zip(order, supports):
                subset = frame.loc[
                    (frame["generalization_class"].astype(str) == group)
                    & (frame["error_type"].astype(str) == error)
                ]
                count = int(pd.to_numeric(subset.get("count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                values.append(100.0 * count / support if support else 0.0)
            bars = ax.barh(
                y,
                values,
                left=left,
                height=0.58,
                color=ERROR_COLORS.get(error, MID_GREY),
                edgecolor=WHITE,
                linewidth=0.8,
                label=ERROR_LABELS.get(error, error),
            )
            for rect, value in zip(bars, values):
                if value >= 7.5:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=THEME.annotation_size,
                        color=WHITE if error not in {"BOUNDARY_AND_LABEL_ERROR"} else BLACK,
                        fontweight="bold",
                    )
            left += np.asarray(values)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{GENERALIZATION_LABELS.get(group, group)}  ·  n={n}" for group, n in zip(order, supports)])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
        ax.set_xlabel("Gold mentions")
        ax.set_title(title, loc="left", pad=34)
        ax.legend(ncol=min(3, len(error_order)), loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, columnspacing=1.5)
        _clean_axes(ax)
        return _save(fig, path)


def label_generalization_heatmap(frame: pd.DataFrame, *, metric: str, title: str, path: str | Path) -> Path:
    classes = [value for value in ("SEEN", "FEW_SHOT", "ZERO_SHOT") if value in set(frame["generalization_class"].astype(str))]
    labels = sorted(frame["label"].astype(str).unique(), key=lambda value: value.lower())
    matrix = np.full((len(labels), len(classes)), np.nan)
    support = np.zeros((len(labels), len(classes)), dtype=int)
    for i, label in enumerate(labels):
        for j, group in enumerate(classes):
            subset = frame.loc[
                (frame["label"].astype(str) == label)
                & (frame["generalization_class"].astype(str) == group)
            ]
            if len(subset):
                matrix[i, j] = float(pd.to_numeric(subset.iloc[0][metric], errors="coerce"))
                support[i, j] = int(subset.iloc[0].get("support", 0))

    height = max(3.5, min(9.0, 1.25 + 0.42 * len(labels)))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(6.3, height))
        masked = np.ma.masked_invalid(matrix)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("f1", [WHITE, SKY, BLUE])
        image = ax.imshow(masked, vmin=0, vmax=1, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels([GENERALIZATION_LABELS.get(group, group) for group in classes], rotation=0)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels([_wrap(label, 24) for label in labels])
        for i in range(len(labels)):
            for j in range(len(classes)):
                if not np.isfinite(matrix[i, j]):
                    continue
                value = matrix[i, j]
                color = WHITE if value >= 0.58 else BLACK
                ax.text(j, i - 0.08, f"{value:.3f}", ha="center", va="center", fontsize=THEME.annotation_size, color=color, fontweight="bold")
                ax.text(j, i + 0.22, f"n={support[i, j]}", ha="center", va="center", fontsize=THEME.annotation_size - 1.0, color=color)
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.025)
        cbar.set_label(metric.replace("_", " ").upper())
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, loc="left", pad=16)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelrotation=0)
        return _save(fig, path)


def subgroup_lollipop_figure(
    frame: pd.DataFrame,
    *,
    category: str,
    metric: str,
    overall: float | None,
    title: str,
    xlabel: str,
    path: str | Path,
) -> Path:
    labels = frame[category].astype(str).tolist()
    values = pd.to_numeric(frame[metric], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    support = pd.to_numeric(frame.get("support", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    x = np.arange(len(labels))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 4.55))
        for index, (value, n) in enumerate(zip(values, support)):
            color = BLUE if n >= MIN_RELIABLE_SUPPORT else MID_GREY
            ax.vlines(index, 0, value, color=LIGHT_GREY, linewidth=2.0, zorder=1)
            ax.scatter(
                index,
                value,
                s=74,
                facecolor=color if n >= MIN_RELIABLE_SUPPORT else WHITE,
                edgecolor=color,
                linewidth=1.6,
                zorder=3,
            )
            ax.text(index, min(value + 0.035, 1.03), f"F1 {value:.3f}", ha="center", va="bottom", fontsize=THEME.annotation_size, fontweight="bold")
            ax.text(index, max(value - 0.055, 0.025), f"n={int(n)}", ha="center", va="top", fontsize=THEME.annotation_size - 0.4, color=DARK_GREY)
        if overall is not None and np.isfinite(overall):
            ax.axhline(overall, color=DARK_GREY, linestyle=(0, (4, 3)), linewidth=1.1, zorder=0)
            ax.text(len(labels) - 0.5, overall + 0.012, f"Overall F1 {overall:.3f}", ha="right", va="bottom", fontsize=THEME.annotation_size, color=DARK_GREY)
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap(_pretty_bucket(label), 14) for label in labels], rotation=0, ha="center")
        ax.set_ylim(0, 1.06)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_ylabel("STRICT F1")
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_title(title, loc="left", pad=18)
        if "frequency" in title.lower():
            fig.text(
                0.13,
                0.01,
                "Each point summarises mentions whose exact surface form was observed in the indicated training-frequency bucket. The F1 score therefore reflects performance for mention groups, not for individual words.",
                ha="left",
                va="bottom",
                fontsize=THEME.annotation_size - 0.2,
                color=DARK_GREY,
            )
        if (support < MIN_RELIABLE_SUPPORT).any():
            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=WHITE,
                markeredgecolor=MID_GREY,
                markeredgewidth=1.5,
                label=f"Low support (n < {MIN_RELIABLE_SUPPORT})",
                markersize=6,
            )
            ax.legend(handles=[handle], loc="upper right", frameon=False)
        _clean_axes(ax)
        return _save(fig, path)



def training_frequency_figure(
    frame: pd.DataFrame,
    *,
    overall: float | None,
    title: str,
    path: str | Path,
) -> Path:
    """Show STRICT F1 by exact mention-surface frequency in training."""
    work = frame.copy()
    work["_bucket_key"] = work["value"].astype(str).map(_frequency_bucket_key)
    work = work.sort_values("_bucket_key", kind="stable").reset_index(drop=True)

    values = pd.to_numeric(work["strict_f1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    support = pd.to_numeric(work.get("support", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    raw_labels = work["value"].astype(str).tolist()
    x = np.arange(len(work), dtype=float) + 0.18

    point_palette = [VERMILLION, ORANGE, YELLOW, SKY, BLUE, GREEN, PURPLE, MID_GREY]

    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width + 0.35, 5.5))
        fig.subplots_adjust(bottom=0.28, top=0.80)
        for index, (raw, value, n) in enumerate(zip(raw_labels, values, support)):
            xpos = x[index]
            color = point_palette[index % len(point_palette)]
            if overall is not None and np.isfinite(overall):
                low, high = sorted((float(overall), float(value)))
                ax.vlines(xpos, low, high, color=LIGHT_GREY, linewidth=2.4, zorder=1)
            else:
                ax.vlines(xpos, 0.0, value, color=LIGHT_GREY, linewidth=2.4, zorder=1)
            reliable = int(n) >= MIN_RELIABLE_SUPPORT
            ax.scatter(xpos, value, s=92, facecolor=color if reliable else WHITE, edgecolor=color, linewidth=1.8, zorder=3)
            ax.text(xpos, min(value + 0.045, 1.035), f"{value:.3f}", ha="center", va="bottom", fontsize=THEME.annotation_size + 0.2, fontweight="bold", color=BLACK)
            if overall is not None and np.isfinite(overall):
                delta = float(value) - float(overall)
                ax.text(xpos, max(value - 0.058, 0.045), f"Δ {delta:+.3f}", ha="center", va="top", fontsize=THEME.annotation_size - 0.35, color=DARK_GREY)

        if overall is not None and np.isfinite(overall):
            ax.axhline(float(overall), color=DARK_GREY, linestyle=(0, (4, 3)), linewidth=1.1, zorder=0)
            ax.text(x[-1] + 0.25, float(overall) + 0.012, f"Overall STRICT F1 = {float(overall):.3f}", ha="right", va="bottom", fontsize=THEME.annotation_size, color=DARK_GREY)

        tick_labels = [_frequency_bucket_label_short(raw) for raw in raw_labels]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")
        ax.set_xlim(-0.10, x[-1] + 0.30)
        ax.set_ylim(0.0, 1.06)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_ylabel("STRICT F1")
        ax.set_xlabel("Exact mention-surface exposure in the training corpus", labelpad=8)
        ax.set_title(title, loc="left", pad=34)
        ax.text(0.0, 1.02, "Each point is STRICT F1 for a group of evaluation mentions, bucketed by how often the exact mention surface appeared in training.", transform=ax.transAxes, ha="left", va="bottom", fontsize=THEME.subtitle_font_size, color=DARK_GREY)

        support_note = "Supports by bucket: " + "; ".join(f"{_frequency_bucket_name(raw)} n={int(n)}" for raw, n in zip(raw_labels, support)) + "."
        fig.text(0.13, 0.06, support_note, ha="left", va="bottom", fontsize=THEME.annotation_size - 0.2, color=DARK_GREY)
        handles = []
        if (support < MIN_RELIABLE_SUPPORT).any():
            handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=WHITE, markeredgecolor=MID_GREY, markeredgewidth=1.5, label=f"Open marker: low support (n < {MIN_RELIABLE_SUPPORT})", markersize=6))
        if handles:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False)
        _clean_axes(ax)
        return _save(fig, path)



def structural_effects_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    ordered = frame.sort_values("delta_f1", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    values = ordered["delta_f1"].to_numpy(dtype=float)
    labels = [f"{row.feature}\n(n={int(row.present_support)} present; n={int(row.absent_support)} absent)" for row in ordered.itertuples(index=False)]
    with _publication_context():
        fig, ax = plt.subplots(figsize=(8.0, max(4.0, 1.35 + 0.95 * len(ordered))))
        colors = [_label_color(i) for i in range(len(ordered))]
        bars = ax.barh(y, values, height=0.48, color=colors, edgecolor=WHITE, linewidth=0.8, zorder=2)
        ax.axvline(0, color=DARK_GREY, linewidth=1.0, zorder=1)
        bound = max(0.15, float(np.nanmax(np.abs(values))) * 1.42 if len(values) else 0.2)
        ax.set_xlim(-bound, bound)
        xmax = ax.get_xlim()[1]
        xmin = ax.get_xlim()[0]
        for rect, value in zip(bars, values):
            xpos = min(value + 0.04 * bound, xmax - 0.07 * bound) if value >= 0 else max(value - 0.04 * bound, xmin + 0.07 * bound)
            ha = "left" if value >= 0 else "right"
            ax.text(xpos, rect.get_y() + rect.get_height() / 2, f"Δ {value:+.3f}", ha=ha, va="center", fontsize=THEME.annotation_size, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("ΔSTRICT F1 vs baseline (property present − absent)")
        ax.set_title(title, loc="left", pad=18)
        fig.text(0.13, 0.015, "Negative values indicate that mentions with the property are harder for the model than mentions without that property. Positive values indicate the opposite.", ha="left", va="bottom", fontsize=THEME.annotation_size - 0.2, color=DARK_GREY)
        _clean_axes(ax)
        return _save(fig, path)



def error_taxonomy_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    """Summarise correct outcomes and rank the model's dominant error families."""
    work = frame.copy()
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0).astype(int)
    work["proportion"] = pd.to_numeric(work.get("proportion", 0.0), errors="coerce").fillna(0.0)
    work["error_type"] = work["error_type"].astype(str)
    correct = work.loc[work["error_type"] == "CORRECT"].copy()
    errors = work.loc[work["error_type"] != "CORRECT"].copy().sort_values("count", ascending=False, kind="stable")
    ordered = pd.concat([correct, errors], ignore_index=True)
    total = max(int(ordered["count"].sum()), 1)
    correct_count = int(correct["count"].sum())
    error_count = total - correct_count
    error_rate = 100.0 * error_count / total
    dominant_name = None
    dominant_count = 0
    if not errors.empty:
        dominant = errors.iloc[0]
        dominant_name = ERROR_LABELS.get(str(dominant["error_type"]), str(dominant["error_type"]).replace("_", " ").title())
        dominant_count = int(dominant["count"])
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, max(4.4, 1.4 + 0.58 * len(ordered))))
        y = np.arange(len(ordered))
        counts = ordered["count"].to_numpy(dtype=float)
        colors = [ERROR_COLORS.get(str(value), MID_GREY) for value in ordered["error_type"]]
        bars = ax.barh(y, counts, height=0.56, color=colors, edgecolor=WHITE, linewidth=0.8, zorder=2)
        labels = [ERROR_LABELS.get(str(value), str(value).replace("_", " ").title()) for value in ordered["error_type"]]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        xmax = max(float(counts.max()) * 1.36 if len(counts) else 1.0, 1.0)
        ax.set_xlim(0, xmax)
        for rect, count in zip(bars, counts):
            share = 100.0 * float(count) / total
            ax.text(rect.get_width() + xmax * 0.012, rect.get_y() + rect.get_height() / 2, f"{int(count)}  ·  {share:.1f}%", ha="left", va="center", fontsize=THEME.annotation_size, fontweight="bold")
        if len(correct) and len(errors):
            ax.axhline(0.5, color=LIGHT_GREY, linewidth=1.0, zorder=0)
        summary_lines = [f"Errors: {error_count:,}/{total:,} ({error_rate:.1f}%)"]
        if dominant_name is not None and error_count > 0:
            summary_lines.append(f"Dominant error: {dominant_name} ({dominant_count:,}; {100.0 * dominant_count / error_count:.1f}% of errors)")
        ax.text(0.985, 0.09, "\n".join(summary_lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=THEME.annotation_size, color=BLACK, bbox={"boxstyle": "round,pad=0.45", "facecolor": PALE_GREY, "edgecolor": "none"})
        ax.set_xlabel("Number of diagnostic outcomes")
        ax.set_title(title, loc="left", pad=24)
        fig.text(0.13, 0.012, "Correct = exact span and label; Missed = unmatched gold mention; Spurious = unmatched prediction. Remaining categories describe matched diagnostic errors.", ha="left", va="bottom", fontsize=THEME.annotation_size - 0.25, color=DARK_GREY)
        _clean_axes(ax)
        return _save(fig, path)



def error_type_pareto_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    work = frame.sort_values("count", ascending=True).reset_index(drop=True)
    y = np.arange(len(work))
    total_errors = max(int(pd.to_numeric(work["count"], errors="coerce").fillna(0).sum()), 1)
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, max(3.5, 1.5 + 0.58 * len(work))))
        values = pd.to_numeric(work["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        colors = [ERROR_COLORS.get(str(value), BLUE) for value in work["error_type"]]
        bars = ax.barh(y, values, height=0.58, color=colors, edgecolor=WHITE, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([ERROR_LABELS.get(str(value), str(value).replace("_", " ").title()) for value in work["error_type"]])
        xmax = max(float(values.max()) * 1.52 if len(values) else 1.0, 1.0)
        ax.set_xlim(0, xmax)
        for rect, row in zip(bars, work.itertuples(index=False)):
            share = 100.0 * float(row.proportion)
            cumulative = 100.0 * float(row.cumulative_proportion)
            ax.text(rect.get_width() + xmax * 0.012, rect.get_y() + rect.get_height() / 2, f"{int(row.count)} · {share:.1f}% · cumulative {cumulative:.1f}%", ha="left", va="center", fontsize=THEME.annotation_size)
        ax.set_xlabel("Errors")
        ax.set_title(title, loc="left", pad=16)
        _clean_axes(ax)
        return _save(fig, path)


def error_surface_pareto_figure(frame: pd.DataFrame, *, title: str, path: str | Path, top_n: int = 12) -> Path:
    work = frame.head(top_n).copy().sort_values("count", ascending=True)
    y = np.arange(len(work))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, max(4.2, 1.4 + 0.42 * len(work))))
        values = pd.to_numeric(work["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        colors = [_label_color(i) for i in range(len(work))]
        bars = ax.barh(y, values, height=0.62, color=colors, edgecolor=WHITE, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(str(value), 28) for value in work["surface"]])
        xmax = max(float(values.max()) * 1.42 if len(values) else 1.0, 1.0)
        ax.set_xlim(0, xmax)
        for rect, row in zip(bars, work.itertuples(index=False)):
            ax.text(rect.get_width() + xmax * 0.012, rect.get_y() + rect.get_height() / 2, f"{int(row.count)} · cumulative {100*float(row.cumulative_proportion):.1f}%", ha="left", va="center", fontsize=THEME.annotation_size)
        ax.set_xlabel("Error occurrences")
        ax.set_title(title, loc="left", pad=16)
        _clean_axes(ax)
        return _save(fig, path)


def signed_boundary_figure(values: pd.Series, *, title: str, explanation: str, path: str | Path) -> Path:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    series = series.loc[series != 0]
    labels, counts = _signed_offset_counts(series)
    x = np.arange(len(labels))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 4.5))
        colors = [SKY if label.startswith("−") or label.startswith("≤") else ORANGE for label in labels]
        bars = ax.bar(x, counts, width=0.72, color=colors, edgecolor=WHITE, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Boundary-error pairs")
        ax.set_xlabel("Character deviation ranges for the mismatched boundary (prediction − gold)")
        ax.set_title(title, loc="left", pad=16)
        fig.text(0.13, 0.015, explanation + ". Exact matches on that boundary are excluded; the histogram therefore focuses only on the boundary side that actually failed.", ha="left", va="bottom", fontsize=THEME.annotation_size - 0.2, color=DARK_GREY)
        maximum = max(counts) if counts else 1
        ax.set_ylim(0, maximum * 1.20 if maximum else 1)
        for rect, count in zip(bars, counts):
            if count:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + maximum * 0.025, str(int(count)), ha="center", va="bottom", fontsize=THEME.annotation_size)
        _clean_axes(ax)
        return _save(fig, path)



def absolute_boundary_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    bins = ("1–5", "6–10", "11–15", "16–20", "21+")
    start_series = pd.to_numeric(frame["span_abs_start_delta"], errors="coerce")
    end_series = pd.to_numeric(frame["span_abs_end_delta"], errors="coerce")
    start = _absolute_offset_distribution(start_series.loc[start_series > 0], bins)
    end = _absolute_offset_distribution(end_series.loc[end_series > 0], bins)
    x = np.arange(len(bins), dtype=float)
    width = 0.34
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 4.75))
        start_bars = ax.bar(x - width / 2, start * 100, width=width, color=BLUE, edgecolor=WHITE, linewidth=0.8, label="Start boundary")
        end_bars = ax.bar(x + width / 2, end * 100, width=width, color=ORANGE, edgecolor=WHITE, linewidth=0.8, label="End boundary")
        for bars, values in ((start_bars, start), (end_bars, end)):
            for rect, value in zip(bars, values):
                if value >= 0.035:
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.2, f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=THEME.annotation_size)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=0)
        ax.set_ylabel("Boundary-error pairs (%)")
        ax.set_xlabel("Absolute character deviation range")
        ymax = max(22.0, float(max(np.max(start), np.max(end), 0.0)) * 118)
        ax.set_ylim(0, ymax)
        ax.set_title(title, loc="left", pad=20)
        ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(0.985, 1.13), frameon=False)
        fig.text(0.13, 0.015, "Start and end deviations are summarised separately in five-character ranges. Only the boundary side that actually deviated is counted, so exact matches on that side are excluded from this panel.", ha="left", va="bottom", fontsize=THEME.annotation_size - 0.2, color=DARK_GREY)
        _clean_axes(ax)
        return _save(fig, path)



def boundary_relation_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    labels = frame["span_relation"].fillna("UNKNOWN").astype(str).map(_pretty_relation)
    counts = labels.value_counts().sort_values(ascending=True)
    total = max(int(counts.sum()), 1)
    y = np.arange(len(counts))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, max(3.4, 1.3 + 0.55 * len(counts))))
        bars = ax.barh(y, counts.to_numpy(dtype=float), height=0.58, color=SKY, edgecolor=WHITE, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(label, 25) for label in counts.index])
        xmax = max(float(counts.max()) * 1.28, 1.0)
        ax.set_xlim(0, xmax)
        for rect, count in zip(bars, counts):
            ax.text(rect.get_width() + xmax * 0.012, rect.get_y() + rect.get_height() / 2, f"{int(count)} · {100*count/total:.1f}%", ha="left", va="center", fontsize=THEME.annotation_size)
        ax.set_xlabel("Boundary-error pairs")
        ax.set_title(title, loc="left", pad=16)
        _clean_axes(ax)
        return _save(fig, path)


def confusion_heatmap(frame: pd.DataFrame, *, value: str, title: str, path: str | Path, normalized: bool) -> Path:
    matrix = frame.pivot(index="gold_label", columns="pred_label", values=value).fillna(0.0)
    return _matrix_heatmap(matrix, title=title, path=path, normalized=normalized)


def extended_confusion_heatmap(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    work = frame.groupby(["gold_label", "pred_label"], as_index=False)["count"].sum()
    gold_order = _special_label_order(work["gold_label"].astype(str).unique(), "[SPURIOUS]")
    pred_order = _special_label_order(work["pred_label"].astype(str).unique(), "[MISSED]")
    matrix = work.pivot(index="gold_label", columns="pred_label", values="count").fillna(0.0).reindex(index=gold_order, columns=pred_order, fill_value=0.0)
    return _matrix_heatmap(matrix, title=title, path=path, normalized=False)


def oracle_dumbbell_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    work = frame.copy()
    work["delta_f1"] = pd.to_numeric(work["delta_f1"], errors="coerce").fillna(0.0)
    work["oracle_f1"] = pd.to_numeric(work["oracle_f1"], errors="coerce").fillna(0.0)
    work["baseline_f1"] = pd.to_numeric(work["baseline_f1"], errors="coerce").fillna(0.0)
    work = work.sort_values("delta_f1", ascending=True).reset_index(drop=True)
    y = np.arange(len(work))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, max(4.0, 1.4 + 0.58 * len(work))))
        for idx, row in work.iterrows():
            baseline = float(row["baseline_f1"])
            oracle = float(row["oracle_f1"])
            ax.hlines(idx, baseline, oracle, color=LIGHT_GREY, linewidth=3.0, zorder=1)
            ax.scatter(baseline, idx, s=48, color=MID_GREY, edgecolor=BLACK, linewidth=0.7, zorder=3)
            ax.scatter(oracle, idx, s=72, color=GREEN, edgecolor=BLACK, linewidth=0.8, zorder=4)
            ax.text(oracle + 0.008, idx, f"{oracle:.3f}  (Δ {float(row['delta_f1']):+.3f})", ha="left", va="center", fontsize=THEME.annotation_size, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(ORACLE_LABELS.get(str(value), str(value).replace("_", " ").title()), 28) for value in work["scenario"]])
        minimum = max(0.0, float(work["baseline_f1"].min()) - 0.05)
        maximum = min(1.0, float(work["oracle_f1"].max()) + 0.12)
        if maximum - minimum < 0.2:
            minimum = max(0.0, minimum - 0.08)
            maximum = min(1.0, maximum + 0.08)
        ax.set_xlim(minimum, maximum)
        ax.set_xlabel("STRICT F1")
        ax.set_title(title, loc="left", pad=20)
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MID_GREY, markeredgecolor=BLACK, label="Observed baseline", markersize=6),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=BLACK, label="Counterfactual oracle", markersize=7),
        ]
        ax.set_ylim(-0.5, len(work) + 0.55)
        ax.legend(handles=handles, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.995), frameon=False)
        _clean_axes(ax)
        return _save(fig, path)



def dataset_shift_figure(
    frame: pd.DataFrame,
    *,
    summary: pd.DataFrame,
    title: str,
    path: str | Path,
) -> Path:
    """Compare label composition in the exact training corpus and evaluation gold."""
    work = frame.copy()
    work["dataset"] = work["dataset"].astype(str)
    work["label"] = work["label"].astype(str)
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0).astype(int)
    work["proportion"] = pd.to_numeric(work["proportion"], errors="coerce").fillna(0.0)
    labels = sorted(work["label"].unique(), key=lambda value: value.lower())
    datasets = [value for value in ("training", "evaluation") if value in set(work["dataset"])]
    totals = {dataset: int(work.loc[work["dataset"] == dataset, "count"].sum()) for dataset in datasets}
    y = np.arange(len(datasets))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 4.25))
        fig.subplots_adjust(top=0.80, bottom=0.18)
        left = np.zeros(len(datasets), dtype=float)
        for idx, label in enumerate(labels):
            percentages = []
            counts = []
            for dataset in datasets:
                subset = work.loc[(work["dataset"] == dataset) & (work["label"] == label)]
                if len(subset):
                    percentages.append(100.0 * float(subset.iloc[0]["proportion"]))
                    counts.append(int(subset.iloc[0]["count"]))
                else:
                    percentages.append(0.0)
                    counts.append(0)
            color = _label_color(idx)
            bars = ax.barh(y, percentages, left=left, height=0.52, color=color, edgecolor=WHITE, linewidth=0.9, label=label)
            for rect, percentage, count in zip(bars, percentages, counts):
                if percentage >= 18.0:
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2, f"{percentage:.0f}%\n(n={count:,})", ha="center", va="center", fontsize=THEME.annotation_size - 0.1, color=WHITE if idx in {0, 3, 5} else BLACK, fontweight="bold", linespacing=1.05)
                elif percentage >= 8.0:
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2, f"{percentage:.0f}%", ha="center", va="center", fontsize=THEME.annotation_size - 0.1, color=WHITE if idx in {0, 3, 5} else BLACK, fontweight="bold")
            left += np.asarray(percentages)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{dataset.capitalize()}\n(n={totals[dataset]:,} entities)" for dataset in datasets])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
        ax.set_xlabel("Share of annotated entities")
        ax.set_title(title, loc="left", y=1.18, pad=0)
        ax.text(0.0, 1.115, "Entity-label composition of the exact training corpus used by the model versus the evaluation gold standard.", transform=ax.transAxes, ha="left", va="bottom", fontsize=THEME.subtitle_font_size, color=DARK_GREY)
        leg = ax.legend(title="Entity label", ncol=min(4, max(1, len(labels))), loc="center", bbox_to_anchor=(0.5, 0.50), frameon=True, columnspacing=1.25, handletextpad=0.45)
        leg.get_frame().set_facecolor(WHITE)
        leg.get_frame().set_alpha(0.95)
        leg.get_frame().set_edgecolor(LIGHT_GREY)
        eval_subset = work.loc[work["dataset"] == "evaluation"].copy()
        if not eval_subset.empty:
            positive = eval_subset.loc[eval_subset["count"] > 0]
            if len(positive) == 1:
                row = positive.iloc[0]
                ax.text(0.99, -0.24, f"Evaluation gold is single-label: {row['label']} = {100.0 * float(row['proportion']):.0f}% (n={int(row['count']):,}).", transform=ax.transAxes, ha="right", va="top", fontsize=THEME.annotation_size, color=DARK_GREY)
        _clean_axes(ax)
        return _save(fig, path)



def confidence_distribution_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    work = frame.copy().sort_values("median", ascending=True).reset_index(drop=True)
    y = np.arange(len(work))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, max(3.6, 1.5 + 0.62 * len(work))))
        for idx, row in work.iterrows():
            minimum, q1, median, q3, maximum, mean = [float(row[name]) for name in ("min", "q1", "median", "q3", "max", "mean")]
            ax.hlines(idx, minimum, maximum, color=LIGHT_GREY, linewidth=1.8)
            ax.hlines(idx, q1, q3, color=BLUE, linewidth=8.0)
            ax.scatter(median, idx, s=48, color=WHITE, edgecolor=BLACK, linewidth=1.1, zorder=4)
            ax.scatter(mean, idx, s=36, color=ORANGE, marker="D", edgecolor=BLACK, linewidth=0.7, zorder=5)
            ax.text(maximum + max(0.002, (work["max"].max() - work["min"].min()) * 0.02), idx, f"n={int(row['support'])}", ha="left", va="center", fontsize=THEME.annotation_size)
        ax.set_yticks(y)
        ax.set_yticklabels([ERROR_LABELS.get(str(value), str(value).replace("_", " ").title()) for value in work["error_type"]])
        lower = float(work["min"].min())
        upper = float(work["max"].max())
        margin = max((upper - lower) * 0.18, 0.01)
        ax.set_xlim(max(0.0, lower - margin), min(1.0, upper + margin))
        ax.set_xlabel("Prediction score")
        ax.set_title(title, loc="left", pad=18)
        handles = [
            Line2D([0], [0], color=BLUE, linewidth=7, label="Interquartile range"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=WHITE, markeredgecolor=BLACK, label="Median", markersize=6),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=ORANGE, markeredgecolor=BLACK, label="Mean", markersize=5),
        ]
        ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
        _clean_axes(ax)
        return _save(fig, path)


def confidence_threshold_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    work = frame.sort_values("threshold").copy()
    thresholds = pd.to_numeric(work["threshold"], errors="coerce").to_numpy(dtype=float)
    f1 = pd.to_numeric(work["strict_f1"], errors="coerce").to_numpy(dtype=float)
    coverage = pd.to_numeric(work["coverage"], errors="coerce").to_numpy(dtype=float)
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 4.45))
        ax.plot(thresholds, f1, color=GREEN, linewidth=THEME.line_width, marker="o", markersize=5, label="STRICT F1")
        ax.plot(thresholds, coverage, color=BLUE, linewidth=THEME.line_width, marker="s", markersize=5, label="Prediction coverage")
        if len(f1) and np.isfinite(f1).any():
            best = int(np.nanargmax(f1))
            ax.scatter(thresholds[best], f1[best], s=70, facecolor=WHITE, edgecolor=BLACK, linewidth=1.2, zorder=5)
            ax.text(thresholds[best], min(1.03, f1[best] + 0.045), f"Best F1 {f1[best]:.3f}\nthreshold {thresholds[best]:.3f}", ha="center", va="bottom", fontsize=THEME.annotation_size)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score / coverage")
        ax.set_xlabel("Confidence threshold")
        ax.set_title(title, loc="left", pad=18)
        ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
        _clean_axes(ax)
        return _save(fig, path)


def document_distribution_figure(frame: pd.DataFrame, *, title: str, path: str | Path) -> Path:
    work = frame.copy()
    strict = pd.to_numeric(work["strict_f1"], errors="coerce").dropna().to_numpy(dtype=float)
    char = pd.to_numeric(work["char_f1"], errors="coerce").dropna().to_numpy(dtype=float)
    data = [strict, char]
    labels = ["STRICT F1", "Character F1"]
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 3.9))
        bp = ax.boxplot(
            data,
            vert=False,
            widths=0.42,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": SKY, "edgecolor": BLUE, "linewidth": 1.2},
            medianprops={"color": BLACK, "linewidth": 1.6},
            whiskerprops={"color": DARK_GREY, "linewidth": 1.0},
            capprops={"color": DARK_GREY, "linewidth": 1.0},
            meanprops={"marker": "D", "markerfacecolor": ORANGE, "markeredgecolor": BLACK, "markersize": 5},
            flierprops={"marker": "o", "markerfacecolor": WHITE, "markeredgecolor": MID_GREY, "markersize": 3},
        )
        ax.set_yticks(np.arange(1, len(labels) + 1))
        ax.set_yticklabels(labels)
        rng = np.random.default_rng(13)
        for idx, values in enumerate(data, start=1):
            if len(values) <= 120:
                jitter = rng.uniform(-0.10, 0.10, size=len(values))
                ax.scatter(values, np.full(len(values), idx) + jitter, s=12, facecolor=WHITE, edgecolor=MID_GREY, linewidth=0.5, alpha=0.75, zorder=2)
            if len(values):
                ax.text(1.015, idx, f"n={len(values)} · median {np.median(values):.3f}", ha="left", va="center", fontsize=THEME.annotation_size, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_xlabel("Document-level F1")
        ax.set_title(title, loc="left", pad=16)
        _clean_axes(ax)
        return _save(fig, path)


# ---------------------------------------------------------------------------
# Compatibility wrappers for the first analysis release. Keeping these names
# avoids breaking downstream code that imported plotting helpers directly.
# ---------------------------------------------------------------------------

def grouped_prf_figure(frame, *, category="value", prefix="strict", title, path) -> Path:
    return generalization_prf_figure(frame, prefix=prefix, bootstrap=pd.DataFrame(), title=title, path=path)


def grouped_label_generalization_figure(frame, *, metric, title, path) -> Path:
    return label_generalization_heatmap(frame, metric=metric, title=title, path=path)


def metric_line_figure(frame, *, category, metric, title, ylabel=None, path) -> Path:
    return subgroup_lollipop_figure(frame, category=category, metric=metric, overall=None, title=title, xlabel=category.replace("_", " ").title(), path=path)


def horizontal_bar_figure(frame, *, label, value, title, xlabel, path) -> Path:
    work = frame[[label, value]].copy().sort_values(value, ascending=True).tail(15)
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, max(3.5, 1.3 + 0.46 * len(work))))
        y = np.arange(len(work))
        values = pd.to_numeric(work[value], errors="coerce").fillna(0).to_numpy(dtype=float)
        bars = ax.barh(y, values, height=0.58, color=BLUE, edgecolor=WHITE, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(str(value), 28) for value in work[label]])
        xmax = max(float(values.max()) * 1.2 if len(values) else 1.0, 1.0)
        ax.set_xlim(0, xmax)
        for rect, raw in zip(bars, values):
            shown = f"{raw:.3f}" if abs(raw) < 1 else f"{raw:.0f}"
            ax.text(rect.get_width() + xmax * 0.012, rect.get_y() + rect.get_height() / 2, shown, ha="left", va="center", fontsize=THEME.annotation_size)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=16)
        _clean_axes(ax)
        return _save(fig, path)


def histogram_figure(values, *, title, xlabel, path, bins=15) -> Path:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    with _publication_context():
        fig, ax = plt.subplots(figsize=(THEME.width, 4.3))
        ax.hist(series.to_numpy(dtype=float), bins=min(int(bins), max(1, int(series.nunique()))), color=SKY, edgecolor=WHITE, linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title, loc="left", pad=16)
        ax.tick_params(axis="x", labelrotation=0)
        _clean_axes(ax)
        return _save(fig, path)


def grouped_shift_figure(frame, *, title, path) -> Path:
    return dataset_shift_figure(frame, summary=pd.DataFrame(), title=title, path=path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _publication_context():
    return plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": THEME.base_font_size,
            "axes.titlesize": THEME.title_font_size,
            "axes.titleweight": "bold",
            "axes.labelsize": THEME.axis_label_size,
            "xtick.labelsize": THEME.tick_size,
            "ytick.labelsize": THEME.tick_size,
            "legend.fontsize": THEME.legend_size,
            "axes.linewidth": THEME.spine_width,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(THEME.spine_width)
    ax.spines["bottom"].set_linewidth(THEME.spine_width)
    ax.tick_params(axis="x", rotation=0, direction="out", length=4, width=0.8, color=BLACK, pad=6)
    ax.tick_params(axis="y", rotation=0, direction="out", length=4, width=0.8, color=BLACK, pad=5)
    ax.grid(False)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.93))
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.06,
        metadata={"Creator": "lab.ner.analysis", "Title": fig._suptitle.get_text() if fig._suptitle else "NER analysis"},
    )
    plt.close(fig)
    return path


def _has_rows(frame: pd.DataFrame | None) -> bool:
    return frame is not None and isinstance(frame, pd.DataFrame) and not frame.empty


def _bootstrap_ci(frame: pd.DataFrame, scope: str, metric: str) -> tuple[float, float] | None:
    if not _has_rows(frame):
        return None
    subset = frame.loc[
        (frame["scope"].astype(str) == str(scope))
        & (frame["metric"].astype(str) == str(metric))
        & (frame.get("status", "ok").astype(str) == "ok")
    ]
    if subset.empty:
        return None
    low = float(subset.iloc[0]["ci_low"])
    high = float(subset.iloc[0]["ci_high"])
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return low, high


def _overall_score(frame: pd.DataFrame, kind: str) -> float | None:
    if not _has_rows(frame):
        return None
    candidates = (
        ["strict_f1", "span_strict_f1"] if kind == "strict" else ["char_f1"]
    )
    for column in candidates:
        if column in frame.columns:
            value = pd.to_numeric(frame[column], errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def _structural_effect_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for feature, frame in tables.items():
        if not _has_rows(frame) or "value" not in frame.columns:
            continue
        index = frame.set_index(frame["value"].astype(str))
        if "True" not in index.index or "False" not in index.index:
            continue
        present = index.loc["True"]
        absent = index.loc["False"]
        if isinstance(present, pd.DataFrame):
            present = present.iloc[0]
        if isinstance(absent, pd.DataFrame):
            absent = absent.iloc[0]
        rows.append(
            {
                "feature": feature,
                "present_f1": float(present.get("strict_f1", 0.0)),
                "absent_f1": float(absent.get("strict_f1", 0.0)),
                "delta_f1": float(present.get("strict_f1", 0.0)) - float(absent.get("strict_f1", 0.0)),
                "present_support": int(present.get("support", 0)),
                "absent_support": int(absent.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def _matrix_heatmap(matrix: pd.DataFrame, *, title: str, path: str | Path, normalized: bool) -> Path:
    rows = [str(value) for value in matrix.index]
    columns = [str(value) for value in matrix.columns]
    values = matrix.to_numpy(dtype=float)
    height = max(3.8, min(9.0, 1.8 + 0.48 * len(rows)))
    width = max(5.5, min(10.5, 2.8 + 0.62 * len(columns)))
    with _publication_context():
        fig, ax = plt.subplots(figsize=(width, height))
        vmax = 1.0 if normalized else max(float(np.nanmax(values)) if values.size else 1.0, 1.0)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("confusion", [WHITE, SKY, BLUE])
        image = ax.imshow(values, vmin=0, vmax=vmax, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels([_wrap(value, 18) for value in columns], rotation=0, ha="center")
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([_wrap(value, 22) for value in rows], rotation=0)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Gold label")
        ax.set_title(title, loc="left", pad=16)
        for i in range(len(rows)):
            for j in range(len(columns)):
                raw = values[i, j]
                intensity = raw / vmax if vmax else 0.0
                shown = f"{100*raw:.1f}%" if normalized else f"{int(round(raw))}"
                ax.text(j, i, shown, ha="center", va="center", fontsize=THEME.annotation_size, color=WHITE if intensity > 0.55 else BLACK, fontweight="bold" if raw else "normal")
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.025)
        cbar.set_label("Row proportion" if normalized else "Count")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelrotation=0)
        return _save(fig, path)


def _signed_offset_counts(series: pd.Series) -> tuple[list[str], list[int]]:
    series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if series.empty:
        return ["0"], [0]
    counts = {
        "≤−11": int((series <= -11).sum()),
        "−10 to −6": int(((series >= -10) & (series <= -6)).sum()),
        "−5 to −1": int(((series >= -5) & (series <= -1)).sum()),
        "0": int((series == 0).sum()),
        "+1 to +5": int(((series >= 1) & (series <= 5)).sum()),
        "+6 to +10": int(((series >= 6) & (series <= 10)).sum()),
        "≥+11": int((series >= 11).sum()),
    }
    labels = [label for label, count in counts.items() if count > 0 or label == "0"]
    return labels, [counts[label] for label in labels]


def _absolute_offset_distribution(values: pd.Series, bins: Sequence[str]) -> np.ndarray:
    series = pd.to_numeric(values, errors="coerce").dropna().abs()
    if series.empty:
        return np.zeros(len(bins), dtype=float)
    counts = [
        int(((series >= 0) & (series <= 5)).sum()),
        int(((series >= 6) & (series <= 10)).sum()),
        int(((series >= 11) & (series <= 15)).sum()),
        int(((series >= 16) & (series <= 20)).sum()),
        int((series >= 21).sum()),
    ]
    total = max(sum(counts), 1)
    return np.asarray(counts, dtype=float) / total



def _frequency_bucket_key(value: str) -> int:
    """Stable scientific ordering for training-frequency buckets."""
    normalized = str(value).strip().replace("–", "-").replace("—", "-").replace(" ", "")
    mapping = {"0": 0, "1": 1, "2-5": 2, "6-10": 3, "11-20": 4, ">20": 5, "21+": 5}
    return mapping.get(normalized, 99)


def _frequency_bucket_name(value: str) -> str:
    normalized = str(value).strip().replace("–", "-").replace("—", "-").replace(" ", "")
    mapping = {
        "0": "Unseen",
        "1": "Seen once",
        "2-5": "Low exposure",
        "6-10": "Moderate exposure",
        "11-20": "High exposure",
        ">20": "Very high exposure",
        "21+": "Very high exposure",
    }
    return mapping.get(normalized, _pretty_bucket(str(value)))


def _frequency_bucket_label_short(value: str) -> str:
    normalized = str(value).strip().replace("–", "-").replace("—", "-").replace(" ", "")
    mapping = {
        "0": "Unseen\n0× in training",
        "1": "Seen once\n1× in training",
        "2-5": "Low exposure\n2–5× in training",
        "6-10": "Moderate exposure\n6–10× in training",
        "11-20": "High exposure\n11–20× in training",
        ">20": "Very high exposure\n>20× in training",
        "21+": "Very high exposure\n21+× in training",
    }
    return mapping.get(normalized, _pretty_bucket(str(value)))


def _frequency_bucket_label(value: str, support: int) -> str:
    name = _frequency_bucket_name(value)
    return f"{name}\nn={int(support)} mentions"


def _signed_offset_counts(series: pd.Series) -> tuple[list[str], list[int]]:
    series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if series.empty:
        return ["−10 to −5", "−4 to −1", "+1 to +4", "+5 to +10"], [0, 0, 0, 0]
    counts = {
        "≤−11": int((series <= -11).sum()),
        "−10 to −5": int(((series >= -10) & (series <= -5)).sum()),
        "−4 to −1": int(((series >= -4) & (series <= -1)).sum()),
        "+1 to +4": int(((series >= 1) & (series <= 4)).sum()),
        "+5 to +10": int(((series >= 5) & (series <= 10)).sum()),
        "≥+11": int((series >= 11).sum()),
    }
    labels = [label for label, count in counts.items() if count > 0]
    if not labels:
        labels = ["−10 to −5", "−4 to −1", "+1 to +4", "+5 to +10"]
    return labels, [counts.get(label, 0) for label in labels]


def _absolute_offset_distribution(values: pd.Series, bins: Sequence[str]) -> np.ndarray:
    series = pd.to_numeric(values, errors="coerce").dropna().abs()
    if series.empty:
        return np.zeros(len(bins), dtype=float)
    counts = [
        int(((series >= 1) & (series <= 5)).sum()),
        int(((series >= 6) & (series <= 10)).sum()),
        int(((series >= 11) & (series <= 15)).sum()),
        int(((series >= 16) & (series <= 20)).sum()),
        int((series >= 21).sum()),
    ]
    total = max(sum(counts), 1)
    return np.asarray(counts, dtype=float) / total


def _pretty_bucket(value: str) -> str:
    return (
        str(value)
        .replace("-<", "–<")
        .replace("-", "–")
        .replace("FEW_SHOT", "Few-shot")
        .replace("ZERO_SHOT", "Zero-shot")
    )


def _pretty_relation(value: str) -> str:
    mapping = {
        "EXACT_BOUNDARY": "Exact boundary",
        "PRED_CONTAINS_GOLD": "Prediction contains gold",
        "GOLD_CONTAINS_PRED": "Gold contains prediction",
        "LEFT_OVERLAP": "Left-side overlap",
        "RIGHT_OVERLAP": "Right-side overlap",
        "OVERLAP": "Partial overlap",
        "DISJOINT": "Disjoint",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def _special_label_order(values: Iterable[str], special: str) -> list[str]:
    values = [str(value) for value in values]
    ordinary = sorted([value for value in values if value != special], key=lambda value: value.lower())
    if special in values:
        ordinary.append(special)
    return ordinary


def _label_color(index: int) -> str:
    palette = [BLUE, ORANGE, GREEN, PURPLE, SKY, VERMILLION, YELLOW, MID_GREY]
    return palette[index % len(palette)]


def _confidence_range(summary: pd.DataFrame, confidence: pd.DataFrame) -> float:
    if _has_rows(summary) and {"min", "max"}.issubset(summary.columns):
        return float(summary.iloc[0]["max"] - summary.iloc[0]["min"])
    if _has_rows(confidence) and {"min", "max"}.issubset(confidence.columns):
        return float(pd.to_numeric(confidence["max"], errors="coerce").max() - pd.to_numeric(confidence["min"], errors="coerce").min())
    return 0.0


def _wrap(value: str, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _remove_previous_generated(root: Path) -> None:
    manifest = root / "plot_manifest.json"
    if not manifest.exists():
        return
    try:
        generated = json.loads(manifest.read_text(encoding="utf-8")).get("generated", [])
    except (json.JSONDecodeError, OSError):
        return
    for filename in generated:
        path = root / str(filename)
        if path.exists() and path.suffix.lower() == ".svg":
            path.unlink()
