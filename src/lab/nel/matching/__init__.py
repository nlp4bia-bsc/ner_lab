"""Lexical entity matching methods."""

from lab.nel.matching.base import BaseEntityMatcher
from lab.nel.matching.lexical import (
    BM25Matcher,
    BM25Retriever,
    JaroWinklerMatcher,
    LevenshteinMatcher,
    StringMatchMatcher,
    TfidfCharNgramMatcher,
    TfidfCharNgramRetriever,
    TokenSetMatcher,
    WhooshContextMatcher,
)
from lab.nel.matching.registry import MATCHER_REGISTRY, build_matcher

__all__ = [
    "BaseEntityMatcher",
    "BM25Matcher",
    "BM25Retriever",
    "JaroWinklerMatcher",
    "LevenshteinMatcher",
    "MATCHER_REGISTRY",
    "StringMatchMatcher",
    "TfidfCharNgramMatcher",
    "TfidfCharNgramRetriever",
    "TokenSetMatcher",
    "WhooshContextMatcher",
    "build_matcher",
]
