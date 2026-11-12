"""A mention's surface features, and how gold mentions sit against each other."""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from lab.core.segmentation import split_into_words

CASINGS = ("LOWERCASE", "UPPERCASE", "TITLECASE", "MIXEDCASE", "UNCASED")
WORD_COUNT_BINS = ("1", "2", "3", "4", "5+")
TRAITS = (
    "acronym_like",
    "has_digit",
    "has_punctuation",
    "has_parentheses",
    "has_hyphen",
    "has_slash",
    "has_greek",
)
GOLD_TRAITS = ("overlapping", "nested")
ACRONYM_MAX_CHARACTERS = 12
ACRONYM_MAX_WORDS = 3
ACRONYM_MIN_UPPERCASE = 0.6


def mention_features(texts: pd.Series) -> pd.DataFrame:
    """
    The surface features of each mention, aligned to `texts`.

    `word_count` is the mention's UAX #29 words, punctuation included: the tagging
    decisions a strict match needs. `casing` is one of `CASINGS`, and each of
    `TRAITS` is a boolean. A mention is acronym-like when it has letters, at most
    12 characters and 3 words, and at least 60% of its letters are uppercase.
    """
    return pd.DataFrame([_features(str(text)) for text in texts], index=texts.index)


def gold_structure(gold: pd.DataFrame) -> pd.DataFrame:
    """
    Whether each gold span overlaps, or nests with, another gold span of its document.

    A span is `nested` when it contains another or lies inside one, and
    `overlapping` when it shares any character with another; nesting implies
    overlap. Aligned to `gold`'s rows.
    """
    structure = pd.DataFrame(False, index=gold.index, columns=list(GOLD_TRAITS))

    for _, document in gold.groupby(gold["filename"].astype(str), sort=False):
        starts = document["start_span"].to_numpy(dtype=int)
        ends = document["end_span"].to_numpy(dtype=int)
        others = ~np.eye(len(document), dtype=bool)

        overlaps = (starts[:, None] < ends[None, :]) & (starts[None, :] < ends[:, None]) & others
        contains = (starts[:, None] <= starts[None, :]) & (ends[None, :] <= ends[:, None]) & others

        structure.loc[document.index, "overlapping"] = overlaps.any(axis=1)
        structure.loc[document.index, "nested"] = (contains | contains.T).any(axis=1)

    return structure


def word_count_bin(word_count: int) -> str:
    """The `WORD_COUNT_BINS` label of a word count."""
    return WORD_COUNT_BINS[min(word_count, len(WORD_COUNT_BINS)) - 1]


def _features(text: str) -> dict:
    word_count = max(len(split_into_words(text)), 1)
    letters = [character for character in text if character.isalpha()]
    categories = [unicodedata.category(character) for character in text]
    uppercase_share = sum(letter.isupper() for letter in letters) / len(letters) if letters else 0.0

    return {
        "word_count": word_count,
        "word_count_bin": word_count_bin(word_count),
        "casing": _casing(text, letters),
        "acronym_like": bool(letters)
        and len(text) <= ACRONYM_MAX_CHARACTERS
        and word_count <= ACRONYM_MAX_WORDS
        and uppercase_share >= ACRONYM_MIN_UPPERCASE,
        "has_digit": any(character.isdigit() for character in text),
        "has_punctuation": any(category.startswith("P") for category in categories),
        "has_parentheses": any(category in ("Ps", "Pe") for category in categories),
        "has_hyphen": "Pd" in categories,
        "has_slash": "/" in text,
        "has_greek": any("GREEK" in unicodedata.name(character, "") for character in text),
    }


def _casing(text: str, letters: list[str]) -> str:
    if not letters:
        return "UNCASED"

    if text.isupper():
        return "UPPERCASE"

    if text.islower():
        return "LOWERCASE"

    return "TITLECASE" if text.istitle() else "MIXEDCASE"
