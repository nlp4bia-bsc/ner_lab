"""
Corpus statistics for a single-language ner_lab corpus.

Two independent entry points, one per metric table:

    compute_text_stats(frame, base_model, language)  size and shape of the raw documents
    compute_annotation_stats(frame, base_model)       size and shape of the entity layer

Both read the canonical `text` / `entities_json` columns `data.read_corpus` returns and
each return a single-row DataFrame; neither needs the other, and each loads its own
tokenizer from `base_model`. Pass `output_dir` to either to also write it as
`<stem>.json` / `<stem>.parquet`, via the public `write_stats`.

Unit conventions
-----------------
tokens     Subwords, via `encoding.tokenize_document` -- the counts the encoding stage
           actually sees.
sentences  pysbd, via `encoding.split_into_sentences` -- the same segmenter `Encoder` uses.
words      Unicode `\\w+` runs, lowercased. Used only for vocabulary and MATTR, where
           subword types would measure the tokenizer rather than the corpus. Both word-
           and subword-level figures are reported.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from ner_lab.data.io import DEFAULT_PARQUET_COMPRESSION
from ner_lab.data.labels import LABEL_ALIASES, normalize_entity_labels
from ner_lab.encoding.segmentation import (
    sentence_token_ranges,
    split_into_sentences,
    tokenize_document,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_MATTR_WINDOW = 100

WORD = re.compile(r"\w+", re.UNICODE)
WHITESPACE = re.compile(r"\s+")


def write_stats(frame: pd.DataFrame, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Write a stats DataFrame as `<stem>.json` and `<stem>.parquet`, returning both paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, parquet_path = directory / f"{stem}.json", directory / f"{stem}.parquet"

    frame.to_json(json_path, orient="records", indent=2, force_ascii=False)
    frame.to_parquet(parquet_path, engine="pyarrow", compression=DEFAULT_PARQUET_COMPRESSION, index=False)

    return json_path, parquet_path


# --------------------------------------------------------------------------- #
# shared primitives
# --------------------------------------------------------------------------- #


def _load_tokenizer(base_model: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if not tokenizer.is_fast:
        raise ValueError(f"{base_model!r} has no fast tokenizer; offsets drive every token metric.")

    # Documents are tokenized whole and never fed to the model, so the
    # "longer than the maximum sequence length" warning is noise here.
    tokenizer.model_max_length = int(1e9)

    return tokenizer


def words_of(text: str) -> list[str]:
    """Lowercased Unicode word runs, punctuation excluded."""
    return WORD.findall(text.lower())


def entities_of(raw: object, normalize_labels: bool = False) -> list[dict]:
    """Decode an `entities_json` cell, optionally folding aliases via `LABEL_ALIASES`."""
    entities = json.loads(str(raw))

    return normalize_entity_labels(entities, LABEL_ALIASES) if normalize_labels else entities


def mattr(sequence: Sequence, window: int) -> float:
    """
    Moving-average type-token ratio: the mean TTR over every window-sized slice.

    Sequences shorter than the window have no slice to average, so their plain TTR
    is used, which keeps short documents in the figure instead of dropping them.
    """
    length = len(sequence)

    if length == 0:
        return float("nan")

    if length <= window:
        return len(set(sequence)) / length

    counts = Counter(sequence[:window])
    types_seen = len(counts)

    for index in range(window, length):
        leaving = sequence[index - window]
        counts[leaving] -= 1

        if not counts[leaving]:
            del counts[leaving]

        counts[sequence[index]] += 1
        types_seen += len(counts)

    return types_seen / ((length - window + 1) * window)


def quantiles(prefix: str, values: Sequence[float], points: Sequence[int]) -> dict[str, float]:
    """Named percentile and mean fields, e.g. `tokens_per_document_p50`, `tokens_per_document_mean`."""
    if not values:
        return {f"{prefix}_p{point}": float("nan") for point in points} | {f"{prefix}_mean": float("nan")}

    return {
        f"{prefix}_p{point}": float(value)
        for point, value in zip(points, np.percentile(values, points))
    } | {f"{prefix}_mean": float(np.mean(values))}


def share(count: int, total: int) -> float:
    """Percentage, or NaN when there is nothing to take a share of."""
    return count / total * 100 if total else float("nan")


# --------------------------------------------------------------------------- #
# table 1: text corpus
# --------------------------------------------------------------------------- #


@dataclass
class TextMeasurements:
    """Raw text measurements, accumulated document by document."""

    sentence_counts: list[int] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    word_counts: list[int] = field(default_factory=list)
    sentence_lengths: list[int] = field(default_factory=list)
    word_types: set[str] = field(default_factory=set)
    subword_types: set[int] = field(default_factory=set)

    # MATTR is accumulated as a running length-weighted sum, so the corpus figure
    # is a mean over tokens rather than over documents, and no corpus-wide token
    # stream is ever held in memory.
    weighted_word_mattr: float = 0.0
    weighted_subword_mattr: float = 0.0

    def add(
        self, text: str, language: str, tokenizer: PreTrainedTokenizerBase, window: int
    ) -> None:
        tokens = tokenize_document(text, tokenizer)
        token_ids = [token["token_id"] for token in tokens]
        words = words_of(text)
        spans = sentence_token_ranges(split_into_sentences(text, language), tokens)

        self.sentence_counts.append(len(spans))
        self.token_counts.append(len(tokens))
        self.word_counts.append(len(words))
        self.sentence_lengths += [span["token_end"] - span["token_start"] for span in spans]

        self.word_types.update(words)
        self.subword_types.update(token_ids)

        self.weighted_word_mattr += mattr(words, window) * len(words)
        self.weighted_subword_mattr += mattr(token_ids, window) * len(token_ids)

    def row(self, language: str) -> dict:
        tokens = sum(self.token_counts)
        words = sum(self.word_counts)

        return {
            "language": language,
            "documents": len(self.token_counts),
            "sentences": sum(self.sentence_counts),
            "tokens": tokens,
            "words": words,
            "vocabulary_words": len(self.word_types),
            "vocabulary_subwords": len(self.subword_types),
            **quantiles("sentences_per_document", self.sentence_counts, (25, 50, 75)),
            **quantiles("tokens_per_document", self.token_counts, (5, 25, 50, 75, 95)),
            **quantiles("tokens_per_sentence", self.sentence_lengths, (25, 50, 75)),
            "mattr_words": self.weighted_word_mattr / words if words else float("nan"),
            "mattr_subwords": self.weighted_subword_mattr / tokens if tokens else float("nan"),
        }


def compute_text_stats(
    frame: pd.DataFrame,
    base_model: str,
    language: str,
    *,
    mattr_window: int = DEFAULT_MATTR_WINDOW,
    output_dir: str | Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Text-corpus metrics: documents, sentences, tokens, vocabulary, length
    distributions and lexical diversity, as a single-row DataFrame.

    Reads `text` only; the entity layer is ignored. Pass `output_dir` to also write
    `text_stats.json` / `text_stats.parquet` there.
    """
    tokenizer = _load_tokenizer(base_model)
    measurements = TextMeasurements()

    if progress:
        print(f"    text stats: {len(frame):,} documents", flush=True)

    for text in frame["text"]:
        measurements.add(text or "", language, tokenizer, mattr_window)

    result = pd.DataFrame([measurements.row(language)])

    if output_dir is not None:
        write_stats(result, output_dir, "text_stats")

    return result


# --------------------------------------------------------------------------- #
# table 2: annotated corpus
# --------------------------------------------------------------------------- #


def mention_token_length(entity: dict, starts: Sequence[int], ends: Sequence[int]) -> int:
    """
    How many tokens the entity's character span touches.

    Tokens are ordered and non-overlapping, so both boundaries are a binary
    search: the first token ending after the span starts, and the first token
    starting at or after the span ends.
    """
    first = bisect_right(ends, entity["start"])
    last = bisect_left(starts, entity["end"])

    return max(last - first, 0)


def span_relations(entities: Sequence[dict]) -> dict[str, int]:
    """
    Mentions involved in an identical, nested, or crossing span relation.

    A mention is counted once per relation type it takes part in. Spans are swept
    in start order and the inner loop stops at the first span beginning after the
    current one ends, since no later span can overlap it either.
    """
    spans = sorted((entity["start"], entity["end"], index) for index, entity in enumerate(entities))
    involved: dict[str, set[int]] = {"identical": set(), "nested": set(), "crossing": set()}

    for position, (start, end, index) in enumerate(spans):
        for offset in range(position + 1, len(spans)):
            other_start, other_end, other_index = spans[offset]

            if other_start >= end:
                break

            if (start, end) == (other_start, other_end):
                relation = "identical"
            elif other_end <= end or other_start <= start:
                relation = "nested"
            else:
                relation = "crossing"

            involved[relation].update((index, other_index))

    return {relation: len(mentions) for relation, mentions in involved.items()}


@dataclass
class AnnotationMeasurements:
    """Entity-layer measurements, accumulated document by document."""

    documents: int = 0
    tokens: int = 0
    annotated_documents: int = 0
    mention_counts: list[int] = field(default_factory=list)
    mention_lengths: list[int] = field(default_factory=list)
    labels: Counter[str] = field(default_factory=Counter)
    surface_forms: Counter[str] = field(default_factory=Counter)
    relations: Counter[str] = field(default_factory=Counter)

    def add(
        self, text: str, entities: Sequence[dict], tokenizer: PreTrainedTokenizerBase
    ) -> None:
        tokens = tokenize_document(text, tokenizer)

        self.documents += 1
        self.tokens += len(tokens)
        self.mention_counts.append(len(entities))

        if not entities:
            return

        self.annotated_documents += 1
        starts = [token["start"] for token in tokens]
        ends = [token["end"] for token in tokens]

        for entity in entities:
            self.labels[entity["label"]] += 1
            self.surface_forms[WHITESPACE.sub(" ", entity["text"].strip().lower())] += 1
            self.mention_lengths.append(mention_token_length(entity, starts, ends))

        self.relations.update(span_relations(entities))

    def row(self) -> dict:
        mentions = sum(self.mention_counts)
        unique_forms = len(self.surface_forms)
        singletons = sum(1 for count in self.surface_forms.values() if count == 1)
        multi_token = sum(1 for length in self.mention_lengths if length > 1)

        return {
            "mentions": mentions,
            "unique_surface_forms": unique_forms,
            **quantiles("mentions_per_document", self.mention_counts, (25, 50, 75)),
            "mentions_per_1k_tokens": mentions / self.tokens * 1000 if self.tokens else float("nan"),
            "entity_types": len(self.labels),
            **quantiles("entity_length_tokens", self.mention_lengths, (25, 50, 75)),
            "multi_token_pct": share(multi_token, mentions),
            "unique_surface_form_ratio": unique_forms / mentions if mentions else float("nan"),
            "singleton_surface_form_ratio": (
                singletons / unique_forms if unique_forms else float("nan")
            ),
            "documents_with_entities_pct": share(self.annotated_documents, self.documents),
            "nested_mentions": self.relations["nested"],
            "nested_pct": share(self.relations["nested"], mentions),
            "crossing_mentions": self.relations["crossing"],
            "crossing_pct": share(self.relations["crossing"], mentions),
            "identical_span_mentions": self.relations["identical"],
            "identical_span_pct": share(self.relations["identical"], mentions),
            # A `{start, end}` span cannot express a gap, so the format itself
            # rules out discontinuous mentions.
            "discontinuous_mentions": 0,
        }


def compute_annotation_stats(
    frame: pd.DataFrame,
    base_model: str,
    *,
    normalize_labels: bool = False,
    output_dir: str | Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Annotated-corpus metrics: mention counts, surface-form diversity, entity
    length, span-relation complexity and the per-entity-type breakdown, as a
    single-row DataFrame.

    The breakdown is spread into flat `mentions_<LABEL>` / `pct_<LABEL>` columns
    ordered by corpus-wide frequency, so the table round-trips through parquet and
    JSON without nesting. Pass `output_dir` to also write `annotation_stats.json` /
    `annotation_stats.parquet` there.
    """
    tokenizer = _load_tokenizer(base_model)
    measurements = AnnotationMeasurements()

    if progress:
        print(f"    annotation stats: {len(frame):,} documents", flush=True)

    for text, raw in zip(frame["text"], frame["entities_json"]):
        measurements.add(text or "", entities_of(raw, normalize_labels), tokenizer)

    row = measurements.row()
    mentions = row["mentions"]

    for label, count in measurements.labels.most_common():
        row[f"mentions_{label}"] = count
        row[f"pct_{label}"] = share(count, mentions)

    result = pd.DataFrame([row])

    if output_dir is not None:
        write_stats(result, output_dir, "annotation_stats")

    return result
