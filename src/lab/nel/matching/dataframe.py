"""Lexical retrieval over DataFrames: a gazetteer plus training aliases, queried row by row."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from lab.nel.matching.bm25 import bm25_index, bm25_term_score, validate_bm25_parameters
from lab.nel.schemas import labels_compatible

TOKEN_RE = re.compile(r"\w+(?:[-_/]\w+)*", flags=re.UNICODE)
TERM_COLUMN_CANDIDATES = ("term", "text", "span", "candidate", "descriptor")
CATALOGUE_COLUMNS = ["term", "code", "label", "source", "norm_term"]


def default_normalizer(text: object) -> str:
    """Lowercase, strip, and collapse whitespace runs to one space."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def combine_gazetteer_and_train(
    gazetteer_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    *,
    gazetteer_term_col: str | None = None,
    train_term_col: str | None = None,
    code_col: str = "code",
    label_col: str = "label",
    normalizer: Callable[[object], str] = default_normalizer,
) -> pd.DataFrame:
    """
    One candidate catalogue from a gazetteer and, optionally, the aliases in training data.

    One row per `term, code, label, source, norm_term`. The term column is found
    among `TERM_COLUMN_CANDIDATES` unless named. Aliases repeated within or across
    sources are dropped; where the same normalized alias maps to the same code and
    label in both, the gazetteer's copy is kept.
    """
    parts = [
        _catalogue_part(
            gazetteer_df, "gazetteer", gazetteer_term_col, code_col, label_col, normalizer
        )
    ]

    if train_df is not None:
        parts.append(
            _catalogue_part(train_df, "train", train_term_col, code_col, label_col, normalizer)
        )

    catalogue = pd.concat(parts, ignore_index=True)
    catalogue = catalogue.sort_values(["source_priority", "term", "code"], kind="stable")
    catalogue = catalogue.drop_duplicates(subset=["norm_term", "code", "label"], keep="first")

    return catalogue.reset_index(drop=True)[CATALOGUE_COLUMNS]


@dataclass(frozen=True)
class CandidateRecord:
    """One ranked alias: its code, its surface form, and its score."""

    code: str
    candidate: str
    score: float


class DataFrameLexicalRetriever:
    """
    Rank catalogue aliases against each row of a query DataFrame.

    The catalogue is `combine_gazetteer_and_train` of the inputs, indexed once at
    construction. A subclass names its `method`, builds its index in
    `_build_index` and ranks one query in `_rank_one`. With `label_aware`, a
    labelled query only matches aliases with the same label or none.
    """

    method = "base"

    def __init__(
        self,
        gazetteer_df: pd.DataFrame,
        train_df: pd.DataFrame | None = None,
        *,
        gazetteer_term_col: str | None = None,
        train_term_col: str | None = None,
        code_col: str = "code",
        label_col: str = "label",
        normalizer: Callable[[object], str] = default_normalizer,
        label_aware: bool = True,
    ) -> None:
        self.code_col = code_col
        self.label_col = label_col
        self.normalizer = normalizer
        self.label_aware = label_aware

        self.catalogue = combine_gazetteer_and_train(
            gazetteer_df=gazetteer_df,
            train_df=train_df,
            gazetteer_term_col=gazetteer_term_col,
            train_term_col=train_term_col,
            code_col=code_col,
            label_col=label_col,
            normalizer=normalizer,
        )

        if self.catalogue.empty:
            raise ValueError("The combined gazetteer/train catalogue is empty.")

        self._terms = self.catalogue["term"].to_numpy(dtype=object)
        self._norm_terms = self.catalogue["norm_term"].to_numpy(dtype=object)
        self._codes = self.catalogue["code"].to_numpy(dtype=object)
        self._labels = self.catalogue["label"].to_numpy(dtype=object)
        self._build_index()

    def predict(
        self,
        query_df: pd.DataFrame,
        *,
        text_col: str = "text",
        top_k: int = 200,
        min_score: float = 0.0,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> pd.DataFrame:
        """
        `query_df` with aligned `candidates`, `codes` and `scores` lists appended.

        Each list holds at most `top_k` codes, one alias per code, scores at or
        above `min_score`. The input rows and columns are left as they are.
        """
        if text_col not in query_df.columns:
            raise ValueError(
                f"query_df must contain {text_col!r}. Available columns: {list(query_df.columns)}"
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        output = query_df.copy()
        has_label = self.label_col in output.columns
        rows: Any = output.iterrows()

        if show_progress:
            rows = tqdm(
                rows,
                total=len(output),
                desc=progress_desc or f"{self.method}: queries",
                unit="mention",
                dynamic_ncols=True,
                leave=True,
            )

        rankings = [
            self._rank_one(
                normalized_query=self.normalizer(row[text_col]),
                query_label=_safe_string(row[self.label_col]) if has_label else None,
                top_k=top_k,
                min_score=min_score,
            )
            for _, row in rows
        ]

        output["candidates"] = [[item.candidate for item in ranking] for ranking in rankings]
        output["codes"] = [[item.code for item in ranking] for ranking in rankings]
        output["scores"] = [[float(item.score) for item in ranking] for ranking in rankings]

        return output

    def _build_index(self) -> None:
        raise NotImplementedError

    def _rank_one(
        self,
        normalized_query: str,
        query_label: str | None,
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        raise NotImplementedError

    def _deduplicate_by_code(
        self,
        alias_indices: np.ndarray,
        alias_scores: np.ndarray,
        query_label: str | None,
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        best_by_code: dict[str, tuple[float, str]] = {}

        for alias_index, alias_score in zip(alias_indices, alias_scores):
            alias_index = int(alias_index)
            score = float(alias_score)

            if score < min_score or not self._is_allowed(alias_index, query_label):
                continue

            code = str(self._codes[alias_index])
            candidate = str(self._terms[alias_index])
            current = best_by_code.get(code)

            if (
                current is None
                or score > current[0]
                or (score == current[0] and candidate < current[1])
            ):
                best_by_code[code] = (score, candidate)

        ranking = [
            CandidateRecord(code=code, candidate=candidate, score=score)
            for code, (score, candidate) in best_by_code.items()
        ]
        ranking.sort(key=lambda item: (-item.score, item.code, item.candidate))

        return ranking[:top_k]

    def _is_allowed(self, alias_index: int, query_label: str | None) -> bool:
        if not self.label_aware or query_label is None:
            return True

        return labels_compatible(query_label, _safe_string(self._labels[alias_index]))


class TfidfCharNgramRetriever(DataFrameLexicalRetriever):
    """TF-IDF character n-gram retrieval; scores are cosine similarities."""

    method = "tfidf_char_ngram"

    def __init__(
        self,
        *args: Any,
        ngram_range: tuple[int, int] = (3, 5),
        analyzer: str = "char_wb",
        **kwargs: Any,
    ) -> None:
        if analyzer not in {"char", "char_wb"}:
            raise ValueError("analyzer must be 'char' or 'char_wb'.")

        if ngram_range[0] <= 0 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range must be a valid positive (min_n, max_n).")

        self.ngram_range = ngram_range
        self.analyzer = analyzer
        super().__init__(*args, **kwargs)

    def _build_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=False,
            norm="l2",
            dtype=np.float32,
        )
        self.term_matrix = self.vectorizer.fit_transform(self._norm_terms)

    def _rank_one(
        self,
        normalized_query: str,
        query_label: str | None,
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        """Only aliases sharing at least one n-gram with the query are ranked."""
        if not normalized_query:
            return []

        similarities = (self.vectorizer.transform([normalized_query]) @ self.term_matrix.T).tocsr()

        return self._deduplicate_by_code(
            alias_indices=similarities.indices,
            alias_scores=similarities.data,
            query_label=query_label,
            top_k=top_k,
            min_score=min_score,
        )


class BM25Retriever(DataFrameLexicalRetriever):
    """Classic BM25 over word tokens of the normalized aliases."""

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
        self.tokenizer = tokenizer or TOKEN_RE.findall
        super().__init__(*args, **kwargs)

    def _build_index(self) -> None:
        tokenized = [self.tokenizer(term) for term in self._norm_terms]
        self.doc_lengths = np.asarray([len(tokens) for tokens in tokenized], dtype=np.float32)
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self.postings, self.idf = bm25_index(tokenized)

    def _rank_one(
        self,
        normalized_query: str,
        query_label: str | None,
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        if not normalized_query or self.avg_doc_length <= 0.0:
            return []

        scores_by_alias: dict[int, float] = defaultdict(float)

        for token in set(self.tokenizer(normalized_query)):
            if token not in self.idf:
                continue

            for alias_index, term_frequency in self.postings[token]:
                scores_by_alias[alias_index] += bm25_term_score(
                    self.idf[token],
                    term_frequency,
                    float(self.doc_lengths[alias_index]),
                    self.avg_doc_length,
                    self.k1,
                    self.b,
                )

        if not scores_by_alias:
            return []

        return self._deduplicate_by_code(
            alias_indices=np.fromiter(scores_by_alias.keys(), dtype=np.int64),
            alias_scores=np.fromiter(scores_by_alias.values(), dtype=np.float32),
            query_label=query_label,
            top_k=top_k,
            min_score=min_score,
        )


def _safe_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    return text or None


def _resolve_term_column(
    dataframe: pd.DataFrame,
    requested_column: str | None,
    dataframe_name: str,
) -> str:
    if requested_column is not None:
        if requested_column not in dataframe.columns:
            raise ValueError(
                f"{dataframe_name} does not contain {requested_column!r}. "
                f"Available columns: {list(dataframe.columns)}"
            )

        return requested_column

    for column in TERM_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not identify a term column in {dataframe_name}. "
        f"Expected one of {TERM_COLUMN_CANDIDATES}; "
        f"available: {list(dataframe.columns)}"
    )


def _catalogue_part(
    dataframe: pd.DataFrame,
    source: str,
    term_col: str | None,
    code_col: str,
    label_col: str,
    normalizer: Callable[[object], str],
) -> pd.DataFrame:
    dataframe_name = f"{source}_df"
    resolved_term_col = _resolve_term_column(dataframe, term_col, dataframe_name)

    if code_col not in dataframe.columns:
        raise ValueError(f"{dataframe_name} must contain {code_col!r}.")

    has_label = label_col in dataframe.columns
    columns = [resolved_term_col, code_col, *([label_col] if has_label else [])]
    part = dataframe.loc[:, columns].rename(
        columns={resolved_term_col: "term", code_col: "code", label_col: "label"}
    )

    if not has_label:
        part["label"] = None

    part = part.dropna(subset=["term", "code"])
    part["term"] = part["term"].astype(str).str.strip()
    part["code"] = part["code"].astype(str).str.strip()
    part["label"] = part["label"].map(_safe_string)
    part = part[(part["term"] != "") & (part["code"] != "")].copy()
    part["norm_term"] = part["term"].map(normalizer)
    part = part[part["norm_term"] != ""].copy()
    part["source"] = source
    part["source_priority"] = 0 if source == "gazetteer" else 1

    return part
