"""Lexical matching: scoring gazetteer terms against mentions, as records or as DataFrames."""

from __future__ import annotations

from lab.nel.matching.base import LexicalMatcher
from lab.nel.matching.bm25 import bm25_index, bm25_term_score
from lab.nel.matching.dataframe import (
    BM25Retriever,
    CandidateRecord,
    DataFrameLexicalRetriever,
    TfidfCharNgramRetriever,
    combine_gazetteer_and_train,
    default_normalizer,
)
from lab.nel.matching.lexical import (
    BM25Matcher,
    ExactMatcher,
    JaroWinklerMatcher,
    LevenshteinMatcher,
    TfidfCharNgramMatcher,
    TokenSetMatcher,
)
from lab.nel.matching.registry import MATCHER_REGISTRY, build_matcher

__all__ = [
    "BM25Matcher",
    "BM25Retriever",
    "CandidateRecord",
    "DataFrameLexicalRetriever",
    "ExactMatcher",
    "JaroWinklerMatcher",
    "LevenshteinMatcher",
    "LexicalMatcher",
    "MATCHER_REGISTRY",
    "TfidfCharNgramMatcher",
    "TfidfCharNgramRetriever",
    "TokenSetMatcher",
    "bm25_index",
    "bm25_term_score",
    "build_matcher",
    "combine_gazetteer_and_train",
    "default_normalizer",
]
