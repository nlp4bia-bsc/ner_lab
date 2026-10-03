"""Training-exposure and lexical-generalization analysis for NER spans."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class NormalizationConfig:
    """Conservative normalization used only for lexical-neighbour comparison."""

    unicode_form: str | None = "NFKC"
    strip: bool = True
    casefold: bool = True
    collapse_whitespace: bool = True

    def normalize(self, text: str) -> str:
        value = str(text)
        if self.unicode_form:
            value = unicodedata.normalize(self.unicode_form, value)
        if self.casefold:
            value = value.casefold()
        if self.collapse_whitespace:
            value = re.sub(r"\s+", " ", value)
        return value.strip() if self.strip else value


DEFAULT_SIMILARITY_BINS = (
    ("EXACT", 1.0, 1.0),
    ("0.95-<1.00", 0.95, 1.0),
    ("0.90-<0.95", 0.90, 0.95),
    ("0.85-<0.90", 0.85, 0.90),
    ("0.80-<0.85", 0.80, 0.85),
    ("<0.80", 0.0, 0.80),
)


@dataclass(frozen=True)
class ExposureConfig:
    """Configuration for lexical training-exposure analysis.

    Canonical exposure classes are:

    * ``SEEN``: raw mention surface occurs exactly in training;
    * ``LEXICALLY_SIMILAR``: no raw exact match, but lexical similarity reaches
      the configured threshold;
    * ``NOVEL``: no raw exact match and lexical similarity is below threshold.

    ``similarity_mode='hybrid'`` is intentionally conservative: the combined
    score is the minimum of normalized Levenshtein and Jaro-Winkler similarity,
    so both comparators must support a high-similarity decision.  This gives
    genuinely novel mentions more protection from being absorbed into the
    similar class. ``levenshtein`` is retained as a backward-compatible mode.

    Jaro reference: Jaro, M. A. (1989), JASA 84(406), 414–420,
    https://doi.org/10.1080/01621459.1989.10478785.
    """

    levenshtein_threshold: float = 0.80
    similarity_mode: str = "hybrid"
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    similarity_bins: tuple[tuple[str, float, float], ...] = DEFAULT_SIMILARITY_BINS
    frequency_bins: tuple[tuple[str, int, int | None], ...] = (
        ("0", 0, 0),
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        (">20", 21, None),
    )

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.levenshtein_threshold) <= 1.0:
            raise ValueError("levenshtein_threshold must be on the 0.0-1.0 scale.")
        if self.similarity_mode not in {"hybrid", "levenshtein"}:
            raise ValueError("similarity_mode must be 'hybrid' or 'levenshtein'.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Neighbour:
    text: str | None
    label: str | None
    similarity: float | None
    levenshtein_similarity: float | None = None
    jaro_winkler_similarity: float | None = None


class ExposureIndex:
    """Cached training-mention index for lexical generalization analysis."""

    def __init__(self, training: pd.DataFrame, config: ExposureConfig | None = None) -> None:
        self.config = config or ExposureConfig()
        self.raw_counts = Counter(str(value) for value in training["text"])
        self.raw_labels: dict[str, set[str]] = defaultdict(set)
        self.normalized_counts: Counter[str] = Counter()
        self.normalized_labels: dict[str, set[str]] = defaultdict(set)
        self.entries: list[tuple[str, str, str]] = []
        seen_entries: set[tuple[str, str]] = set()

        for row in training.itertuples(index=False):
            text = str(row.text)
            label = str(row.label)
            normalized = self.config.normalization.normalize(text)
            self.raw_labels[text].add(label)
            self.normalized_counts[normalized] += 1
            self.normalized_labels[normalized].add(label)
            key = (normalized, label)
            if key not in seen_entries:
                self.entries.append((normalized, text, label))
                seen_entries.add(key)

        self.entries.sort(key=lambda item: (item[0], item[1], item[2]))
        self.backend, self._similarity = _resolve_similarity(self.config.similarity_mode)
        self._cache: dict[tuple[str, str | None, bool], Neighbour] = {}
        self.cache_hits = 0

    @property
    def unique_training_mentions(self) -> int:
        return len(self.entries)

    def nearest(self, text: str, label: str | None = None, *, other_label: bool = False) -> Neighbour:
        normalized = self.config.normalization.normalize(text)
        key = (normalized, label, other_label)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        candidates = [
            entry
            for entry in self.entries
            if label is None
            or ((entry[2] != label) if other_label else (entry[2] == label))
        ]

        if not candidates:
            result = Neighbour(None, None, None, None, None)
        else:
            scored = []
            for entry in candidates:
                combined, levenshtein, jaro_winkler = self._similarity(normalized, entry[0])
                scored.append(
                    (
                        round(float(combined), 6),
                        round(float(levenshtein), 6),
                        round(float(jaro_winkler), 6),
                        entry,
                    )
                )
            similarity, levenshtein, jaro_winkler, (_, original, candidate_label) = min(
                scored,
                key=lambda item: (-item[0], -item[1], -item[2], item[3][1], item[3][2], item[3][0]),
            )
            result = Neighbour(
                original,
                candidate_label,
                similarity,
                levenshtein,
                jaro_winkler,
            )

        self._cache[key] = result
        return result

    def describe(self, text: str, label: str) -> dict:
        text = str(text)
        label = str(label)
        normalized = self.config.normalization.normalize(text)
        exact_labels = sorted(self.raw_labels.get(text, set()))
        normalized_labels = sorted(self.normalized_labels.get(normalized, set()))
        nearest = self.nearest(text)
        nearest_same = self.nearest(text, label)
        nearest_other = self.nearest(text, label, other_label=True)

        exact_seen = bool(exact_labels)
        normalized_seen = bool(normalized_labels)
        similarity = nearest.similarity
        threshold = float(self.config.levenshtein_threshold)

        if exact_seen:
            generalization = "SEEN"
        elif similarity is not None and similarity >= threshold:
            generalization = "LEXICALLY_SIMILAR"
        else:
            generalization = "NOVEL"

        legacy_generalization = {
            "SEEN": "SEEN",
            "LEXICALLY_SIMILAR": "FEW_SHOT",
            "NOVEL": "ZERO_SHOT",
        }[generalization]

        return {
            "generalization_class": generalization,
            "generalization_class_legacy": legacy_generalization,
            "train_exact_seen": exact_seen,
            "train_exact_seen_same_label": label in exact_labels,
            "train_exact_seen_other_label": any(value != label for value in exact_labels),
            "train_exact_frequency": int(self.raw_counts[text]),
            "train_exact_labels": exact_labels,
            "train_normalized_text": normalized,
            "train_normalized_seen": normalized_seen,
            "train_normalized_seen_same_label": label in normalized_labels,
            "train_normalized_seen_other_label": any(value != label for value in normalized_labels),
            "train_normalized_frequency": int(self.normalized_counts[normalized]),
            "train_nearest_text": nearest.text,
            "train_nearest_label": nearest.label,
            "train_nearest_similarity": similarity,
            "train_nearest_levenshtein_similarity": nearest.levenshtein_similarity,
            "train_nearest_jaro_winkler_similarity": nearest.jaro_winkler_similarity,
            "train_nearest_same_label_text": nearest_same.text,
            "train_nearest_same_label_similarity": nearest_same.similarity,
            "train_nearest_other_label_text": nearest_other.text,
            "train_nearest_other_label": nearest_other.label,
            "train_nearest_other_label_similarity": nearest_other.similarity,
            "similarity_bin": similarity_bin(similarity, self.config.similarity_bins),
            "train_frequency_bin": frequency_bin(
                int(self.raw_counts[text]), self.config.frequency_bins
            ),
            # Canonical flags.
            "lexically_similar": generalization == "LEXICALLY_SIMILAR",
            "lexically_novel": generalization == "NOVEL",
            # Backward-compatible aliases retained for downstream users.
            "zero_shot_exact": not exact_seen,
            "zero_shot_normalized": not normalized_seen,
            "zero_shot_levenshtein": generalization == "NOVEL",
            "few_shot_levenshtein": generalization == "LEXICALLY_SIMILAR",
        }


def describe_frame(frame: pd.DataFrame, exposure: ExposureIndex) -> pd.DataFrame:
    """Return one exposure-description row per span, preserving frame order."""
    records = [
        exposure.describe(str(row.text), str(row.label))
        for row in frame.itertuples(index=False)
    ]
    if records:
        return pd.DataFrame.from_records(records, index=frame.index)

    template = exposure.describe("", "")
    return pd.DataFrame(columns=list(template), index=frame.index)


def similarity_bin(
    value: float | None,
    bins: tuple[tuple[str, float, float], ...] = DEFAULT_SIMILARITY_BINS,
) -> str | None:
    if value is None:
        return None
    for name, lower, upper in bins:
        if name == "EXACT" and value == 1.0:
            return name
        if name != "EXACT" and lower <= value < upper:
            return name
    return None


def frequency_bin(
    value: int,
    bins: tuple[tuple[str, int, int | None], ...],
) -> str:
    for name, lower, upper in bins:
        if value >= lower and (upper is None or value <= upper):
            return name
    raise ValueError(f"No training-frequency bin covers {value}.")


def _resolve_similarity(
    mode: str,
) -> tuple[str, Callable[[str, str], tuple[float, float, float]]]:
    try:
        from rapidfuzz.distance import JaroWinkler, Levenshtein
    except ImportError as error:
        raise ImportError(
            "Lexical exposure analysis requires rapidfuzz; install lab[ner]."
        ) from error

    def compare(first: str, second: str) -> tuple[float, float, float]:
        levenshtein = float(Levenshtein.normalized_similarity(first, second))
        jaro_winkler = float(JaroWinkler.normalized_similarity(first, second))
        combined = (
            levenshtein
            if mode == "levenshtein"
            else min(levenshtein, jaro_winkler)
        )
        return combined, levenshtein, jaro_winkler

    backend = (
        "rapidfuzz.Levenshtein.normalized_similarity"
        if mode == "levenshtein"
        else "min(rapidfuzz.Levenshtein.normalized_similarity, rapidfuzz.JaroWinkler.normalized_similarity)"
    )
    return backend, compare
