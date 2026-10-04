"""Sparse TF-IDF retrieval: character n-grams, optionally blended with word n-grams."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lab.nel.preprocessing import normalize_text
from lab.nel.schemas import Candidate, GazetteerEntry, Mention, labels_compatible


class SparseRetriever:
    """
    TF-IDF retrieval over a gazetteer, returning up to `top_k` unique codes per mention.

    Scores are the character n-gram similarity weighted by `char_weight`, plus the
    word n-gram similarity weighted by `word_weight` when `use_word_ngrams` is set.
    Only the best non-zero entries of each query row are sorted, `overfetch_factor`
    times `top_k` of them, falling back to the whole row when that leaves fewer
    than `top_k` codes. With `threshold <= 0`, zero-score entries fill the list up to
    `top_k` codes when the gazetteer has that many.
    """

    method = "matrix_biencoder"

    def __init__(
        self,
        similarity: str = "cosine",
        char_ngram_range: tuple[int, int] = (3, 5),
        word_ngram_range: tuple[int, int] = (1, 2),
        use_word_ngrams: bool = True,
        char_weight: float = 1.0,
        word_weight: float = 0.25,
        max_char_features: int | None = None,
        max_word_features: int | None = 200_000,
        strip_accents: bool = False,
        normalize_punct: bool = False,
        overfetch_factor: int = 10,
        *,
        top_k: int = 25,
        threshold: float = 0.0,
        label_aware: bool = True,
    ) -> None:
        self.top_k = top_k
        self.threshold = threshold
        self.label_aware = label_aware
        self.gazetteer: list[GazetteerEntry] = []
        self.similarity = similarity
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.use_word_ngrams = use_word_ngrams
        self.char_weight = char_weight
        self.word_weight = word_weight
        self.max_char_features = max_char_features
        self.max_word_features = max_word_features
        self.strip_accents = strip_accents
        self.normalize_punct = normalize_punct
        self.overfetch_factor = max(1, overfetch_factor)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.char_ngram_range,
            max_features=self.max_char_features,
            dtype=np.float32,
        )
        self.word_vectorizer = (
            TfidfVectorizer(
                analyzer="word",
                ngram_range=self.word_ngram_range,
                max_features=self.max_word_features,
                dtype=np.float32,
            )
            if self.use_word_ngrams
            else None
        )
        self.term_matrix = None
        self.word_term_matrix = None

    def build_index(self, gazetteer: list[GazetteerEntry]) -> None:
        """Keep the gazetteer and fit the vectorizers on its terms."""
        self.gazetteer = gazetteer
        self._build_index(gazetteer)

    def search(
        self,
        mentions: list[Mention],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[list[Candidate]]:
        """Each mention's candidates: unique codes, best first, at or above the threshold."""
        k = top_k or self.top_k
        threshold = self.threshold if threshold is None else threshold
        similarities = self._score_queries(mentions)
        results: list[list[Candidate]] = []

        for row_id, mention in enumerate(mentions):
            row = similarities.getrow(row_id)
            indices, scores = self._top_entries(row, k)
            ranked = self._unique_code_rank(mention, indices, scores, k, threshold, fill_zero=False)

            if len(ranked) < k and len(indices) < row.nnz:
                all_indices, all_scores = self._top_entries(row, row.nnz)
                ranked = self._unique_code_rank(mention, all_indices, all_scores, k, threshold)
            elif len(ranked) < k and threshold <= 0.0:
                seen = {candidate.code for candidate in ranked if candidate.code is not None}
                self._fill_zero_score_codes(mention, ranked, seen, k)

            results.append(ranked)

        return results

    def _normalize(self, text: str) -> str:
        return normalize_text(
            text, strip_accents=self.strip_accents, normalize_punct=self.normalize_punct
        )

    def _build_index(self, gazetteer: list[GazetteerEntry]) -> None:
        terms = [self._normalize(entry.term) for entry in gazetteer]
        self.term_matrix = self.vectorizer.fit_transform(terms)

        if self.word_vectorizer is not None:
            self.word_term_matrix = self.word_vectorizer.fit_transform(terms)

    def _score_queries(self, mentions: list[Mention]) -> Any:
        if self.term_matrix is None:
            raise RuntimeError("Index not built")

        texts = [self._normalize(mention.text) for mention in mentions]
        similarities = (self.vectorizer.transform(texts) @ self.term_matrix.T) * self.char_weight

        if (
            self.word_vectorizer is not None
            and self.word_term_matrix is not None
            and self.word_weight
        ):
            word_queries = self.word_vectorizer.transform(texts)
            similarities = (
                similarities + (word_queries @ self.word_term_matrix.T) * self.word_weight
            )

        return similarities.tocsr()

    def _top_entries(self, row: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        if row.nnz == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        fetch = min(row.nnz, max(k * self.overfetch_factor, k))

        if fetch < row.nnz:
            positions = np.argpartition(row.data, -fetch)[-fetch:]
            positions = positions[np.argsort(row.data[positions])[::-1]]
        else:
            positions = np.argsort(row.data)[::-1]

        return row.indices[positions], row.data[positions]

    def _candidate(
        self, mention: Mention, entry: GazetteerEntry, score: float, rank: int
    ) -> Candidate:
        return Candidate(
            mention_id=None,
            filename=mention.filename,
            text=mention.text,
            label=mention.label,
            code=entry.code,
            term=entry.term,
            score=score,
            method=self.method,
            rank=rank,
            metadata={
                "candidate_span": entry.term,
                "char_weight": self.char_weight,
                "word_weight": self.word_weight if self.word_vectorizer is not None else 0.0,
            },
        )

    def _unique_code_rank(
        self,
        mention: Mention,
        indices: np.ndarray,
        scores: np.ndarray,
        k: int,
        threshold: float,
        fill_zero: bool = True,
    ) -> list[Candidate]:
        ranked: list[Candidate] = []
        seen: set[str] = set()

        for index, score in zip(indices, scores):
            entry = self.gazetteer[int(index)]
            value = float(score)

            if not self._allowed(mention, entry) or value < threshold or entry.code in seen:
                continue

            seen.add(entry.code)
            ranked.append(self._candidate(mention, entry, value, len(ranked) + 1))

            if len(ranked) >= k:
                break

        if fill_zero and len(ranked) < k and threshold <= 0.0:
            self._fill_zero_score_codes(mention, ranked, seen, k)

        return ranked

    def _fill_zero_score_codes(
        self, mention: Mention, ranked: list[Candidate], seen: set[str], k: int
    ) -> None:
        for entry in self.gazetteer:
            if len(ranked) >= k:
                break

            if not self._allowed(mention, entry) or entry.code in seen:
                continue

            seen.add(entry.code)
            ranked.append(self._candidate(mention, entry, 0.0, len(ranked) + 1))

    def _allowed(self, mention: Mention, entry: GazetteerEntry) -> bool:
        return not self.label_aware or labels_compatible(mention.label, entry.label)
