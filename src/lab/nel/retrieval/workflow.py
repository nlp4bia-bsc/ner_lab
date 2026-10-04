"""Candidate retrieval over a mention table: build a vocabulary, retrieve, score against gold."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lab.nel.evaluation import evaluate_candidate_dataframe, read_ontology_graph_pickle
from lab.nel.io import read_table, write_table

TableSource = pd.DataFrame | str | Path

TERM_COLUMNS = ("term", "text", "mention")
CODE_COLUMNS = ("code", "gold_code", "concept_id")


@dataclass(slots=True)
class RetrievalResult:
    """The predictions of a retrieval run, and its metrics when there was gold to score."""

    predictions: pd.DataFrame
    metrics: dict[str, Any] | None

    @property
    def evaluated(self) -> bool:
        """Whether at least one row carried a gold code to score."""
        return self.metrics is not None


def build_vocabulary(
    gazetteer: TableSource,
    additional_sources: Sequence[TableSource] | TableSource | None = None,
    *,
    term_column: str | None = None,
    code_column: str | None = None,
    deduplicate: bool = True,
    remove_guillemets: bool = True,
) -> pd.DataFrame:
    """
    One `term`/`code` vocabulary from a gazetteer and any additional tables.

    Each source names its surface form `term`, `text` or `mention` and its code
    `code`, `gold_code` or `concept_id`, unless `term_column`/`code_column` say
    otherwise. Blank rows are dropped, `«»` are stripped from terms unless
    `remove_guillemets=False`, and with `deduplicate` a repeated term-code pair is
    kept once, additional sources first.
    """
    if additional_sources is None:
        sources: list[TableSource] = []
    elif isinstance(additional_sources, (pd.DataFrame, str, Path)):
        sources = [additional_sources]
    else:
        sources = list(additional_sources)

    vocabulary = pd.concat(
        [
            _vocabulary_part(source, term_column, code_column, remove_guillemets)
            for source in [*sources, gazetteer]
        ],
        ignore_index=True,
    )

    if vocabulary.empty:
        raise ValueError("Vocabulary contains no valid term-code pairs")

    if deduplicate:
        vocabulary = vocabulary.drop_duplicates(subset=["code", "term"], keep="first")

    return vocabulary.reset_index(drop=True)


class CandidateRetrievalPipeline:
    """
    Index a vocabulary, retrieve candidates for a mention table, and score them against gold.

    The retriever indexes with `fit_faiss(vocab=..., batch_size=...)` or
    `build_index(vocabulary)` and retrieves with `get_candidates(texts, k, batch_size)`.
    Every input column is kept, and aligned `codes`, `candidates` and `scores` lists
    are appended with the top prediction of each. Rows are scored only when the gold
    column exists and at least one of its values is non-empty.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        top_k: int = 200,
        k_values: Sequence[int] = (1, 5, 25, 100, 200),
        graph_path: str | Path | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.retriever = retriever
        self.top_k = int(top_k)
        self.k_values = tuple(int(value) for value in k_values)
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self._graph: Any | None = None
        self._undirected_graph: Any | None = None

    def fit(
        self, vocabulary: pd.DataFrame, *, batch_size: int | None = None
    ) -> CandidateRetrievalPipeline:
        """Index `vocabulary` with the retriever."""
        if hasattr(self.retriever, "fit_faiss"):
            self.retriever.fit_faiss(vocab=vocabulary, batch_size=batch_size)
        elif hasattr(self.retriever, "build_index"):
            self.retriever.build_index(vocabulary)
        else:
            raise TypeError("Retriever must implement fit_faiss or build_index")

        return self

    def run(
        self,
        data: TableSource,
        *,
        mention_column: str = "text",
        gold_column: str | None = "code",
        batch_size: int | None = None,
        output_path: str | Path | None = None,
        metrics_output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve candidates for every row and score the rows with a gold code.

        `output_path` and `metrics_output_path` also write the predictions and the
        metrics; either refuses an existing file unless `overwrite` is set.
        """
        mentions_df = _read_source(data)

        if mentions_df.empty:
            raise ValueError("Input data contains no rows")

        if output_path is not None and Path(output_path).exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {Path(output_path)}")

        if mention_column not in mentions_df.columns:
            raise ValueError(f"Mention column '{mention_column}' is not present")

        predictions = self._predictions(mentions_df, mention_column, batch_size)
        metrics = self._metrics(predictions, gold_column)

        if output_path is not None:
            write_table(predictions, output_path, overwrite=overwrite)

        if metrics is not None and metrics_output_path is not None:
            _write_metrics(metrics, Path(metrics_output_path), overwrite)

        return RetrievalResult(predictions=predictions, metrics=metrics)

    def _predictions(
        self, mentions_df: pd.DataFrame, mention_column: str, batch_size: int | None
    ) -> pd.DataFrame:
        if not hasattr(self.retriever, "get_candidates"):
            raise TypeError("Retriever must implement get_candidates(texts, k, batch_size)")

        terms, codes, scores = self.retriever.get_candidates(
            mentions_df[mention_column].fillna("").astype(str).tolist(),
            k=self.top_k,
            batch_size=batch_size,
        )

        predictions = mentions_df.copy()
        predictions["codes"] = codes
        predictions["candidates"] = terms
        predictions["scores"] = scores
        predictions["predicted_code"] = [items[0] if items else None for items in codes]
        predictions["predicted_term"] = [items[0] if items else None for items in terms]
        predictions["predicted_score"] = [items[0] if items else None for items in scores]
        predictions["retrieval_method"] = getattr(
            self.retriever, "method", type(self.retriever).__name__
        )

        return predictions

    def _metrics(self, predictions: pd.DataFrame, gold_column: str | None) -> dict[str, Any] | None:
        if gold_column is None or gold_column not in predictions.columns:
            return None

        gold = predictions[gold_column]
        valid_gold = gold.notna() & gold.astype(str).str.strip().ne("")

        if not valid_gold.any():
            return None

        graph, undirected = self._load_graph()
        metrics = evaluate_candidate_dataframe(
            predictions.loc[valid_gold, [gold_column, "codes"]].copy(),
            self.k_values,
            gold_col=gold_column,
            codes_col="codes",
            method=str(predictions["retrieval_method"].iloc[0]),
            graph=graph,
            undirected_graph=undirected,
        )
        metrics["n_input_rows"] = int(len(predictions))
        metrics["n_evaluated_rows"] = int(valid_gold.sum())
        metrics["n_rows_without_gold"] = int((~valid_gold).sum())

        return metrics

    def _load_graph(self) -> tuple[Any | None, Any | None]:
        if self.graph_path is None:
            return None, None

        if self._graph is None or self._undirected_graph is None:
            self._graph, self._undirected_graph = read_ontology_graph_pickle(self.graph_path)

        return self._graph, self._undirected_graph


def _read_source(source: TableSource) -> pd.DataFrame:
    return source.copy() if isinstance(source, pd.DataFrame) else read_table(source)


def _resolve_column(
    dataframe: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    role: str,
) -> str:
    if explicit is not None:
        if explicit not in dataframe.columns:
            raise ValueError(f"Configured {role} column '{explicit}' is not present")

        return explicit

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(f"Could not infer the {role} column. Tried: {', '.join(candidates)}")


def _vocabulary_part(
    source: TableSource,
    term_column: str | None,
    code_column: str | None,
    remove_guillemets: bool,
) -> pd.DataFrame:
    dataframe = _read_source(source)
    resolved_term = _resolve_column(dataframe, term_column, TERM_COLUMNS, "vocabulary term")
    resolved_code = _resolve_column(dataframe, code_column, CODE_COLUMNS, "vocabulary code")

    part = dataframe[[resolved_term, resolved_code]].dropna().copy()
    part.columns = ["term", "code"]
    part["term"] = part["term"].astype(str)
    part["code"] = part["code"].astype(str)
    part = part[(part["term"].str.strip() != "") & (part["code"].str.strip() != "")]

    if remove_guillemets:
        part["term"] = part["term"].map(lambda value: re.sub(r"[«»]", "", value))

    return part


def _write_metrics(metrics: dict[str, Any], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
