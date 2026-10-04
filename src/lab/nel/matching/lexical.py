"""The lexical matchers: exact, edit-distance, token-set, TF-IDF and BM25 term scoring."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lab.nel.matching.base import LexicalMatcher
from lab.nel.matching.bm25 import bm25_index, bm25_term_score, validate_bm25_parameters
from lab.nel.schemas import GazetteerEntry, Mention, labels_compatible

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
except ImportError:
    fuzz = Levenshtein = JaroWinkler = None


class ExactMatcher(LexicalMatcher):
    """Score 1.0 when the normalized mention equals the normalized term, 0.0 otherwise."""

    method = "string_match"

    def _score_candidates(self, mention: Mention) -> Iterator[tuple[GazetteerEntry, float]]:
        mention_text = self.normalizer(mention.text)

        for entry in self._compatible_entries(mention):
            yield entry, float(self.normalizer(entry.term) == mention_text)


class LevenshteinMatcher(LexicalMatcher):
    """Normalized Levenshtein similarity, or `difflib`'s ratio without rapidfuzz."""

    method = "levenshtein"

    def _score_candidates(self, mention: Mention) -> Iterator[tuple[GazetteerEntry, float]]:
        mention_text = self.normalizer(mention.text)

        for entry in self._compatible_entries(mention):
            entry_text = self.normalizer(entry.term)

            if Levenshtein is not None:
                score = Levenshtein.normalized_similarity(mention_text, entry_text)
            else:
                score = SequenceMatcher(None, mention_text, entry_text).ratio()

            yield entry, float(score)


class JaroWinklerMatcher(LexicalMatcher):
    """Jaro-Winkler similarity, or `difflib`'s ratio without rapidfuzz."""

    method = "jaro_winkler"

    def _score_candidates(self, mention: Mention) -> Iterator[tuple[GazetteerEntry, float]]:
        mention_text = self.normalizer(mention.text)

        for entry in self._compatible_entries(mention):
            entry_text = self.normalizer(entry.term)

            if JaroWinkler is not None:
                score = JaroWinkler.similarity(mention_text, entry_text)
            else:
                score = SequenceMatcher(None, mention_text, entry_text).ratio()

            yield entry, float(score / 100.0 if score > 1.0 else score)


class TokenSetMatcher(LexicalMatcher):
    """Token-set similarity, or `difflib`'s ratio over sorted tokens without rapidfuzz."""

    method = "token_set"

    def _score_candidates(self, mention: Mention) -> Iterator[tuple[GazetteerEntry, float]]:
        mention_text = self.normalizer(mention.text)

        for entry in self._compatible_entries(mention):
            entry_text = self.normalizer(entry.term)

            if fuzz is not None:
                score = fuzz.token_set_ratio(mention_text, entry_text) / 100.0
            else:
                score = SequenceMatcher(
                    None, _sorted_tokens(mention_text), _sorted_tokens(entry_text)
                ).ratio()

            yield entry, float(score)


class TfidfCharNgramMatcher(LexicalMatcher):
    """Cosine similarity of character n-gram TF-IDF vectors, indexed on first use."""

    method = "tfidf_char_ngram"

    def __init__(
        self,
        *args: Any,
        ngram_range: tuple[int, int] = (3, 5),
        analyzer: str = "char_wb",
        **kwargs: Any,
    ) -> None:
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self._entries: list[GazetteerEntry] | None = None
        super().__init__(*args, **kwargs)

    def _score_candidates(self, mention: Mention) -> Iterator[tuple[GazetteerEntry, float]]:
        if self._entries is None:
            self._build_index()

        query = self._vectorizer.transform([self.normalizer(mention.text) or ""])
        scores = (self._matrix @ query.T).toarray().ravel()

        for entry, score in zip(self._entries, scores):
            if not self.label_aware or labels_compatible(mention.label, entry.label):
                yield entry, float(score)

    def _build_index(self) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=False,
            norm="l2",
            dtype=np.float32,
        )
        self._matrix = self._vectorizer.fit_transform(
            [self.normalizer(entry.term) or "" for entry in self.gazetteer]
        )
        self._entries = list(self.gazetteer)


class BM25Matcher(LexicalMatcher):
    """Classic BM25 over whitespace tokens of the normalized terms, indexed on first use."""

    method = "bm25"

    def __init__(
        self,
        *args: Any,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        validate_bm25_parameters(k1, b)
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or str.split
        self._entries: list[GazetteerEntry] | None = None
        super().__init__(*args, **kwargs)

    def _score_candidates(self, mention: Mention) -> Iterator[tuple[GazetteerEntry, float]]:
        if self._entries is None:
            self._build_index()

        if self._average_length <= 0.0:
            yield from ((entry, 0.0) for entry in self._entries)
            return

        scores = [0.0] * len(self._entries)

        for token in set(self.tokenizer(self.normalizer(mention.text) or "")):
            if token not in self._idf:
                continue

            for term_index, term_frequency in self._postings[token]:
                scores[term_index] += bm25_term_score(
                    self._idf[token],
                    term_frequency,
                    self._lengths[term_index],
                    self._average_length,
                    self.k1,
                    self.b,
                )

        for entry, score in zip(self._entries, scores):
            if not self.label_aware or labels_compatible(mention.label, entry.label):
                yield entry, float(score)

    def _build_index(self) -> None:
        entries = list(self.gazetteer)
        tokenized = [self.tokenizer(self.normalizer(entry.term) or "") for entry in entries]
        self._lengths = [len(tokens) for tokens in tokenized]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._postings, self._idf = bm25_index(tokenized)
        self._entries = entries


def _sorted_tokens(text: str) -> str:
    return " ".join(sorted(text.split()))
