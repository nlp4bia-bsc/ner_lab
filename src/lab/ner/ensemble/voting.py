"""Combining members' word tables into one decision per word, and decoding those into spans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from lab.core.segmentation import split_into_words
from lab.core.spans import span_dataframe
from lab.ner.ensemble.tables import PROBABILITY_COLUMNS, TAGS

Votes = Literal["hard", "soft"]

VOTES = ("hard", "soft")
TOLERANCE = 1e-9

OUTSIDE, BEGIN, INSIDE = (TAGS.index(tag) for tag in ("O", "B", "I"))


@dataclass(frozen=True)
class WordVotes:
    """
    Every member's vote at every word any member labelled, on one shared index.

    `words` holds `doc_id` and `word_id` in document order; `present` says which
    member labelled each word, `tags` holds each member's own tag as an index into
    `TAGS`, and `probabilities` its probability of each. A member's absent entries
    are zero and are never read, because its weight there is zero.
    """

    words: pd.DataFrame
    present: np.ndarray
    tags: np.ndarray
    probabilities: np.ndarray

    @property
    def n_members(self) -> int:
        return self.present.shape[1]


@dataclass(frozen=True)
class CombinedWords:
    """The combined vote at each word: entity share, begin and inside shares, tag probabilities."""

    words: pd.DataFrame
    entity: np.ndarray
    begin: np.ndarray
    inside: np.ndarray
    probabilities: np.ndarray


def align_votes(tables: Sequence[pd.DataFrame]) -> WordVotes:
    """Put members' word tables on the union of the words they labelled."""
    if not tables:
        raise ValueError("An ensemble needs at least one member's word table.")

    keys = pd.concat([table[["doc_id", "word_id"]] for table in tables], ignore_index=True)
    words_df = keys.drop_duplicates().sort_values(["doc_id", "word_id"]).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(words_df)

    n_words, n_members = len(words_df), len(tables)
    present = np.zeros((n_words, n_members), dtype=bool)
    tags = np.full((n_words, n_members), OUTSIDE, dtype=np.int8)
    probabilities = np.zeros((n_words, n_members, len(TAGS)), dtype=np.float64)

    for member, table in enumerate(tables):
        positions = index.get_indexer(pd.MultiIndex.from_frame(table[["doc_id", "word_id"]]))

        present[positions, member] = True
        tags[positions, member] = table["tag"].map(TAGS.index).to_numpy()
        probabilities[positions, member] = table[PROBABILITY_COLUMNS].to_numpy()

    return WordVotes(words=words_df, present=present, tags=tags, probabilities=probabilities)


def combine_votes(
    word_votes: WordVotes,
    weights: Sequence[float],
    votes: Votes,
) -> CombinedWords:
    """
    Weigh the members' votes at every word, over the members that labelled it.

    Weights are renormalised word by word over the members present, so a member
    that did not label a word is no evidence either way. Hard votes count a
    member's own tag; soft votes its probabilities. The tag probabilities are
    always the weighted probabilities, whichever kind of vote decides.
    """
    if votes not in VOTES:
        raise ValueError(f"votes must be one of {VOTES}, got {votes!r}.")

    if len(weights) != word_votes.n_members:
        raise ValueError(f"Received {len(weights)} weights for {word_votes.n_members} members.")

    present_weights = word_votes.present * np.asarray(weights, dtype=np.float64)
    totals = present_weights.sum(axis=1)

    if votes == "hard":
        begin = (word_votes.tags == BEGIN).astype(np.float64)
        inside = (word_votes.tags == INSIDE).astype(np.float64)
    else:
        begin = word_votes.probabilities[:, :, BEGIN]
        inside = word_votes.probabilities[:, :, INSIDE]

    def share(values: np.ndarray) -> np.ndarray:
        weighted = (values * present_weights).sum(axis=1)

        return np.divide(weighted, totals, out=np.zeros_like(weighted), where=totals > 0)

    weighted_probabilities = np.einsum("wm,wmt->wt", present_weights, word_votes.probabilities)
    probabilities = np.divide(
        weighted_probabilities,
        totals[:, None],
        out=np.zeros_like(weighted_probabilities),
        where=totals[:, None] > 0,
    )

    return CombinedWords(
        words=word_votes.words,
        entity=share(begin + inside),
        begin=share(begin),
        inside=share(inside),
        probabilities=probabilities,
    )


def decode_combined(
    combined: CombinedWords,
    threshold: float,
    texts: dict[str, str],
    target_label: str,
) -> pd.DataFrame:
    """
    Decode combined votes into scored spans, document by document.

    A word is inside an entity when its entity share is strictly above `threshold`.
    An entity word begins a new entity when the word before it is not in one, or
    when its begin share beats its inside share; a tie continues the entity. "The
    word before" is the previous word any member labelled, so a word nobody
    labelled is passed over rather than breaking a span. A span runs from its first
    word's start to its last word's end, as `split_into_words` draws them, and its
    score is the mean combined probability of the tag each of its words took.
    """
    spans: list[dict[str, Any]] = []
    words_df = combined.words
    boundaries = np.flatnonzero(words_df["doc_id"].ne(words_df["doc_id"].shift()))

    for start, end in zip(boundaries, [*boundaries[1:], len(words_df)]):
        doc_id = str(words_df["doc_id"].iat[start])
        text = texts[doc_id]
        words = split_into_words(text)
        current: dict[str, Any] | None = None
        scores: list[float] = []

        for position in range(start, end):
            if combined.entity[position] - threshold <= TOLERANCE:
                current = None
                continue

            word = words[int(words_df["word_id"].iat[position])]
            begins = (
                current is None or combined.begin[position] - combined.inside[position] > TOLERANCE
            )

            if begins:
                current = {
                    "filename": doc_id,
                    "label": target_label,
                    "start_span": word["start"],
                    "end_span": word["end"],
                    "text": "",
                }
                scores = []
                spans.append(current)

            current["end_span"] = word["end"]
            scores.append(float(combined.probabilities[position, BEGIN if begins else INSIDE]))
            current["text"] = text[current["start_span"] : current["end_span"]]
            current["score"] = float(np.mean(scores))

    return span_dataframe(spans, scored=True)


def ensemble_spans(
    word_votes: WordVotes,
    weights: Sequence[float],
    threshold: float,
    votes: Votes,
    texts: dict[str, str],
    target_label: str,
) -> pd.DataFrame:
    """The spans an ensemble predicts: combine the votes, then decode them."""
    combined = combine_votes(word_votes, weights=weights, votes=votes)

    return decode_combined(combined, threshold=threshold, texts=texts, target_label=target_label)


def normalize_weights(weights: Sequence[float]) -> list[float]:
    """Scale non-negative weights to sum to one."""
    values = np.asarray(weights, dtype=np.float64)

    if values.size == 0 or (values < 0).any() or values.sum() <= 0:
        raise ValueError(f"Weights must be non-negative with a positive sum, got {list(weights)}.")

    return (values / values.sum()).tolist()
