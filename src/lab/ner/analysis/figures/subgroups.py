"""Figures of subgroups: recall and outcomes by exposure, mention properties and traits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from lab.ner.analysis.features import GOLD_TRAITS, TRAITS
from lab.ner.analysis.figures.style import (
    BAR_HEIGHT,
    BLUE,
    CONTEXT,
    DOUBLE_COLUMN,
    MIN_SEGMENT_LABEL,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    SEGMENT_GAP,
    SURFACE,
    figure_title,
    interval_marks,
    new_figure,
    rate_axis,
    segment_label_color,
    support_label,
)
from lab.ner.analysis.subgroups import GOLD_OUTCOMES, PREDICTED_OUTCOMES, family_values

VALUE_LABELS = {
    "SEEN": "Seen",
    "LEXICALLY_SIMILAR": "Lexically similar",
    "NOVEL": "Novel",
    "LOWERCASE": "Lower",
    "UPPERCASE": "Upper",
    "TITLECASE": "Title",
    "MIXEDCASE": "Mixed",
    "UNCASED": "Uncased",
}
PROPERTY_TITLES = {
    "frequency_bin": "Training frequency of the surface",
    "similarity_bin": "Similarity to nearest training mention",
    "word_count_bin": "Words in the mention",
    "casing": "Casing",
}
TRAIT_LABELS = {
    "acronym_like": "Acronym-like",
    "has_digit": "Digit",
    "has_punctuation": "Punctuation",
    "has_parentheses": "Parentheses",
    "has_hyphen": "Hyphen or dash",
    "has_slash": "Slash",
    "has_greek": "Greek letter",
    "overlapping": "Overlaps other gold",
    "nested": "Nested in or around gold",
}


def exposure_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Gold recall per exposure class, beside what became of each class's gold mentions."""
    return _profile_figure(
        tables["gold_subgroups"],
        measure="strict_recall",
        outcomes=GOLD_OUTCOMES,
        title="Gold mentions by lexical exposure to training",
        measure_label="Strict recall",
        share_label="Share of gold mentions",
    )


def predicted_exposure_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Precision per exposure class of the predicted surface, beside each class's outcomes."""
    return _profile_figure(
        tables["predicted_subgroups"],
        measure="strict_precision",
        outcomes=PREDICTED_OUTCOMES,
        title="Predictions by lexical exposure of their surface",
        measure_label="Strict precision",
        share_label="Share of predictions",
    )


def mention_properties_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Gold recall across training frequency, similarity, length and casing, one panel each."""
    subgroups = tables["gold_subgroups"]

    if subgroups.empty:
        return None

    figure, axes = new_figure(DOUBLE_COLUMN, 4.6, 2, 2, sharey=True)

    for index, (panel, family) in enumerate(zip(axes.flat, PROPERTY_TITLES)):
        rows = _ordered(subgroups, family)
        positions = np.arange(len(rows))

        interval_marks(panel, positions, rows, "strict_recall", BLUE, horizontal=False)
        rate_axis(panel, "y", "Strict recall" if index % 2 == 0 else "")
        panel.set_xticks(
            positions,
            [
                f"{VALUE_LABELS.get(value, value)}\nn={support:,}"
                for value, support in zip(rows["value"], rows["support"])
            ],
        )
        panel.set_xlim(-0.6, max(len(rows), 1) - 0.4)
        panel.set_title(PROPERTY_TITLES[family])

    figure_title(figure, "Gold recall by mention property")

    return figure


def mention_traits_figure(tables: dict[str, pd.DataFrame]) -> Figure | None:
    """Gold recall without and with each surface or structural trait, side by side per trait."""
    subgroups = tables["gold_subgroups"]

    if subgroups.empty:
        return None

    traits = [
        trait
        for trait in (*TRAITS, *GOLD_TRAITS)
        if {"absent", "present"} <= set(subgroups.loc[subgroups["family"] == trait, "value"])
    ]

    if not traits:
        return None

    figure, axes = new_figure(DOUBLE_COLUMN, 0.7 + 0.32 * len(traits))
    panel = axes[0, 0]
    positions = np.arange(len(traits))
    absent = pd.DataFrame([_row(subgroups, trait, "absent") for trait in traits])
    present = pd.DataFrame([_row(subgroups, trait, "present") for trait in traits])

    interval_marks(
        panel, positions + 0.15, absent, "strict_recall", CONTEXT, label="Without the trait"
    )
    interval_marks(panel, positions - 0.15, present, "strict_recall", BLUE, label="With the trait")
    rate_axis(panel, "x", "Strict recall")
    panel.set_yticks(
        positions,
        [
            support_label(TRAIT_LABELS[trait], support)
            for trait, support in zip(traits, present["support"])
        ],
    )
    panel.invert_yaxis()
    figure.legend(loc="outside lower center", ncol=2)
    figure_title(figure, "Gold recall without and with each trait (n = gold mentions with it)")

    return figure


def _profile_figure(
    subgroups: pd.DataFrame,
    measure: str,
    outcomes: tuple[str, ...],
    title: str,
    measure_label: str,
    share_label: str,
) -> Figure | None:
    if subgroups.empty:
        return None

    rows = _ordered(subgroups, "exposure")
    positions = np.arange(len(rows))
    figure, axes = new_figure(
        DOUBLE_COLUMN, 0.95 + 0.42 * len(rows), 1, 2, sharey=True, width_ratios=[1.0, 1.5]
    )
    rate_panel, share_panel = axes[0]

    interval_marks(rate_panel, positions, rows, measure, BLUE)
    rate_axis(rate_panel, "x", measure_label)
    rate_panel.set_yticks(
        positions,
        [
            support_label(VALUE_LABELS[value], support)
            for value, support in zip(rows["value"], rows["support"])
        ],
    )
    rate_panel.invert_yaxis()

    left = np.zeros(len(rows))

    for outcome in outcomes:
        shares = rows[outcome.lower()].to_numpy() / rows["support"].to_numpy()
        share_panel.barh(
            positions,
            shares,
            left=left,
            height=BAR_HEIGHT,
            color=OUTCOME_COLORS[outcome],
            edgecolor=SURFACE,
            linewidth=SEGMENT_GAP,
        )

        for position, start, share in zip(positions, left, shares):
            if share >= MIN_SEGMENT_LABEL:
                share_panel.text(
                    start + share / 2,
                    position,
                    f"{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=segment_label_color(OUTCOME_COLORS[outcome]),
                )

        left += shares

    rate_axis(share_panel, "x", share_label)
    share_panel.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    share_panel.tick_params(axis="y", length=0)
    share_panel.spines["left"].set_visible(False)

    figure.legend(
        handles=[
            Patch(color=OUTCOME_COLORS[outcome], label=OUTCOME_LABELS[outcome])
            for outcome in outcomes
        ],
        loc="outside lower center",
        ncol=len(outcomes),
    )
    figure_title(figure, title)

    return figure


def _ordered(subgroups: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = subgroups[subgroups["family"] == family].set_index("value")
    order = [value for value in family_values(family) if value in rows.index]

    return rows.loc[order].reset_index()


def _row(subgroups: pd.DataFrame, family: str, value: str) -> pd.Series:
    return subgroups[(subgroups["family"] == family) & (subgroups["value"] == value)].iloc[0]
