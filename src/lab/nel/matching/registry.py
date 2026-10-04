"""The built-in matchers by method name, and building one from its name."""

from __future__ import annotations

from typing import Any

from lab.nel.matching.base import LexicalMatcher
from lab.nel.matching.lexical import (
    BM25Matcher,
    ExactMatcher,
    JaroWinklerMatcher,
    LevenshteinMatcher,
    TfidfCharNgramMatcher,
    TokenSetMatcher,
)

MATCHER_REGISTRY: dict[str, type[LexicalMatcher]] = {
    "string_match": ExactMatcher,
    "levenshtein": LevenshteinMatcher,
    "jaro_winkler": JaroWinklerMatcher,
    "token_set": TokenSetMatcher,
    "tfidf_char": TfidfCharNgramMatcher,
    "bm25": BM25Matcher,
}


def build_matcher(method: str, **kwargs: Any) -> LexicalMatcher:
    """The built-in matcher called `method`, constructed with `kwargs`."""
    if method not in MATCHER_REGISTRY:
        available = ", ".join(sorted(MATCHER_REGISTRY))
        raise ValueError(f"Unknown matcher '{method}'. Available methods: {available}")

    return MATCHER_REGISTRY[method](**kwargs)
