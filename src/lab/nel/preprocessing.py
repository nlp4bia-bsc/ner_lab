"""Normalizing mention and gazetteer text before it is compared."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    normalize_punct: bool = False,
) -> str:
    """
    Collapse whitespace and, optionally, lowercase, strip accents and blank out punctuation.

    `strip_accents` drops the Unicode combining marks left by NFD decomposition;
    `normalize_punct` replaces every non-word, non-space character with a space.
    """
    normalized = " ".join(str(text).split())

    if lowercase:
        normalized = normalized.lower()

    if strip_accents:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(character) != "Mn"
        )

    if normalize_punct:
        normalized = " ".join(re.sub(r"[^\w\s]", " ", normalized).split())

    return normalized
