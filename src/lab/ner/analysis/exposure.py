"""Lexical exposure: how close a mention's surface comes to the training mentions."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

EXPOSURE_CLASSES = ("SEEN", "LEXICALLY_SIMILAR", "NOVEL")
SIMILARITY_BINS = (
    ("1.00", 1.0),
    ("[0.95, 1)", 0.95),
    ("[0.90, 0.95)", 0.90),
    ("[0.85, 0.90)", 0.85),
    ("[0.80, 0.85)", 0.80),
    ("< 0.80", 0.0),
)
FREQUENCY_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    (">20", 21, None),
)
EXPOSURE_COLUMNS = [
    "exposure",
    "seen_verbatim",
    "similarity",
    "nearest_training",
    "training_frequency",
    "similarity_bin",
    "frequency_bin",
]

WHITESPACE = re.compile(r"\s+")


def normalize_mention(text: str) -> str:
    """The form mentions are compared in: NFKC, casefolded, whitespace collapsed and stripped."""
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def describe_exposure(
    texts: pd.Series, training_texts: pd.Series, similarity_threshold: float = 0.80
) -> pd.DataFrame:
    """
    One row of lexical exposure per mention, aligned to `texts`.

    A mention is `SEEN` when its normalized form occurs among the training mentions,
    `LEXICALLY_SIMILAR` when its nearest training form reaches `similarity_threshold`
    in normalized Levenshtein similarity, `1 − edits / max(len)`, and `NOVEL`
    otherwise; `SEEN` is exactly a similarity of 1.0. `training_frequency` counts the
    training mentions sharing its normalized form, and `seen_verbatim` says whether
    the surface occurs in training character for character.

    Each distinct normalized form is searched once, over the distinct training forms,
    and ties go to the first training form in sorted order.
    """
    training_forms = Counter(normalize_mention(text) for text in training_texts)
    choices = sorted(training_forms)
    verbatim = set(training_texts.astype(str))
    forms = texts.astype(str).map(normalize_mention)
    nearest = {form: _nearest(form, choices) for form in forms.unique()}

    similarity = forms.map(lambda form: nearest[form][1])
    exposure = pd.DataFrame(
        {
            "exposure": similarity.map(lambda value: _exposure_class(value, similarity_threshold)),
            "seen_verbatim": texts.astype(str).isin(verbatim),
            "similarity": similarity,
            "nearest_training": forms.map(lambda form: nearest[form][0]),
            "training_frequency": forms.map(lambda form: training_forms.get(form, 0)),
        },
        index=texts.index,
    )
    exposure["similarity_bin"] = exposure["similarity"].map(similarity_bin)
    exposure["frequency_bin"] = exposure["training_frequency"].map(frequency_bin)

    return exposure[EXPOSURE_COLUMNS]


def similarity_bin(similarity: float) -> str:
    """The `SIMILARITY_BINS` label a similarity falls in."""
    return next(label for label, lower in SIMILARITY_BINS if similarity >= lower)


def frequency_bin(frequency: int) -> str:
    """The `FREQUENCY_BINS` label a training frequency falls in."""
    return next(
        label
        for label, lower, upper in FREQUENCY_BINS
        if frequency >= lower and (upper is None or frequency <= upper)
    )


def _nearest(form: str, choices: list[str]) -> tuple[str, float]:
    match = process.extractOne(form, choices, scorer=Levenshtein.normalized_similarity)

    return match[0], float(match[1])


def _exposure_class(similarity: float, similarity_threshold: float) -> str:
    if similarity == 1.0:
        return "SEEN"

    return "LEXICALLY_SIMILAR" if similarity >= similarity_threshold else "NOVEL"
