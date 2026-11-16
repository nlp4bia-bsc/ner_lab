"""Subgroups on the side that defines them: gold mentions by recall, predictions by precision."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.core.scoring import group_by_document, merge_intervals, normalize_spans
from lab.ner.analysis.exposure import EXPOSURE_CLASSES, FREQUENCY_BINS, SIMILARITY_BINS
from lab.ner.analysis.features import CASINGS, GOLD_TRAITS, TRAITS, WORD_COUNT_BINS
from lab.ner.analysis.variability import ratio_interval

SHARED_FAMILIES = (
    "exposure",
    "similarity_bin",
    "frequency_bin",
    "word_count_bin",
    "casing",
    *TRAITS,
)
GOLD_FAMILIES = (*SHARED_FAMILIES, *GOLD_TRAITS)
PREDICTED_FAMILIES = SHARED_FAMILIES
GOLD_OUTCOMES = ("CORRECT", "BOUNDARY_ERROR", "MISSED", "UNCREDITED_EXACT")
PREDICTED_OUTCOMES = ("CORRECT", "BOUNDARY_ERROR", "SPURIOUS", "UNCREDITED_EXACT")
TRAIT_VALUES = ("absent", "present")


def family_values(family: str) -> tuple[str, ...]:
    """The values a subgroup family takes, in the order tables and figures show them."""
    ordered = {
        "exposure": EXPOSURE_CLASSES,
        "similarity_bin": tuple(label for label, _ in SIMILARITY_BINS),
        "frequency_bin": tuple(label for label, _, _ in FREQUENCY_BINS),
        "word_count_bin": WORD_COUNT_BINS,
        "casing": CASINGS,
    }

    return ordered.get(family, TRAIT_VALUES)


def covered_characters(spans: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """
    How many of each span's characters some span of `other` in its document covers.

    Aligned to `spans`' rows. Summed over a set of gold spans and divided by their
    length, it is that set's character recall; over predictions, character precision.
    """
    coverage = {
        filename: merge_intervals([(row["off0"], row["off1"]) for row in rows])
        for filename, rows in group_by_document(normalize_spans(other)).items()
    }

    return np.array(
        [
            _overlap(int(start), int(end), coverage.get(str(filename), []))
            for filename, start, end in zip(
                spans["filename"], spans["start_span"], spans["end_span"]
            )
        ],
        dtype=int,
    )


def gold_subgroups(
    gold_mentions: pd.DataFrame, documents: pd.Index, draws: np.ndarray, confidence: float
) -> pd.DataFrame:
    """
    Strict and character recall of gold mentions per subgroup, and how the rest failed.

    One row per family and value of `GOLD_FAMILIES` with any support: the count of
    each gold outcome, strict recall as the share `CORRECT`, character recall as the
    share of the subgroup's characters some prediction covers, and a bootstrap
    interval for each over `documents`.
    """
    return _subgroups(
        gold_mentions,
        families=GOLD_FAMILIES,
        outcomes=GOLD_OUTCOMES,
        measure="recall",
        documents=documents,
        draws=draws,
        confidence=confidence,
    )


def predicted_subgroups(
    predicted_mentions: pd.DataFrame, documents: pd.Index, draws: np.ndarray, confidence: float
) -> pd.DataFrame:
    """
    Strict and character precision of predictions per subgroup of their own surface.

    The mirror of `gold_subgroups` over `PREDICTED_FAMILIES`: strict precision is the
    share of the subgroup's predictions that are strict matches, character precision
    the share of their characters some gold span covers.
    """
    return _subgroups(
        predicted_mentions,
        families=PREDICTED_FAMILIES,
        outcomes=PREDICTED_OUTCOMES,
        measure="precision",
        documents=documents,
        draws=draws,
        confidence=confidence,
    )


def subgroup_value(mentions: pd.DataFrame, family: str) -> pd.Series:
    """A family's value per mention, traits written as `absent` or `present`."""
    if family in TRAITS or family in GOLD_TRAITS:
        return mentions[family].map({False: "absent", True: "present"})

    return mentions[family].astype(str)


def _subgroups(
    mentions: pd.DataFrame,
    families: tuple[str, ...],
    outcomes: tuple[str, ...],
    measure: str,
    documents: pd.Index,
    draws: np.ndarray,
    confidence: float,
) -> pd.DataFrame:
    rows = []

    for family in families:
        values = subgroup_value(mentions, family)

        for value in family_values(family):
            members = mentions[values == value]

            if members.empty:
                continue

            rows.append(
                {
                    "family": family,
                    "value": value,
                    "support": len(members),
                    **{
                        outcome.lower(): int((members["outcome"] == outcome).sum())
                        for outcome in outcomes
                    },
                    **_measure(
                        members, f"strict_{measure}", documents, draws, confidence, strict=True
                    ),
                    **_measure(
                        members, f"char_{measure}", documents, draws, confidence, strict=False
                    ),
                }
            )

    return pd.DataFrame(rows)


def _measure(
    members: pd.DataFrame,
    name: str,
    documents: pd.Index,
    draws: np.ndarray,
    confidence: float,
    strict: bool,
) -> dict[str, float]:
    if strict:
        numerator = (members["outcome"] == "CORRECT").astype(int)
        denominator = pd.Series(1, index=members.index)
    else:
        numerator = members["covered_characters"]
        denominator = members["characters"]

    numerators = numerator.groupby(members["filename"]).sum().reindex(documents, fill_value=0)
    denominators = denominator.groupby(members["filename"]).sum().reindex(documents, fill_value=0)
    low, high = ratio_interval(
        numerators=numerators.to_numpy(),
        denominators=denominators.to_numpy(),
        draws=draws,
        confidence=confidence,
    )

    return {name: numerator.sum() / denominator.sum(), f"{name}_low": low, f"{name}_high": high}


def _overlap(start: int, end: int, ranges: list[tuple[int, int]]) -> int:
    return sum(
        max(0, min(end, range_end) - max(start, range_start)) for range_start, range_end in ranges
    )
