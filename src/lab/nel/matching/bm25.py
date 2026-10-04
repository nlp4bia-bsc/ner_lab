"""The BM25 pieces both BM25 implementations share: the inverted index and the term score."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log


def bm25_index(
    tokenized_terms: list[list[str]],
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, float]]:
    """
    The postings and the IDF of a tokenized gazetteer.

    Postings map each token to `(term index, token frequency)` pairs. IDF is the
    smoothed `log(1 + (n - df + 0.5) / (df + 0.5))`, never negative.
    """
    document_frequency: Counter[str] = Counter()
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for term_index, tokens in enumerate(tokenized_terms):
        frequencies = Counter(tokens)
        document_frequency.update(frequencies.keys())

        for token, frequency in frequencies.items():
            postings[token].append((term_index, frequency))

    n_terms = len(tokenized_terms)
    idf = {
        token: log(1.0 + (n_terms - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }

    return postings, idf


def bm25_term_score(
    idf: float,
    term_frequency: int,
    term_length: float,
    average_length: float,
    k1: float,
    b: float,
) -> float:
    """One query token's contribution to a term's BM25 score."""
    denominator = term_frequency + k1 * (1.0 - b + b * term_length / average_length)

    return idf * term_frequency * (k1 + 1.0) / denominator


def validate_bm25_parameters(k1: float, b: float) -> None:
    """Raise unless `k1 > 0` and `0 <= b <= 1`."""
    if k1 <= 0:
        raise ValueError("k1 must be greater than zero.")

    if not 0.0 <= b <= 1.0:
        raise ValueError("b must be between 0 and 1.")
