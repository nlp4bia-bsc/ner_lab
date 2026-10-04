"""Retrieval metrics over ranked code lists: recall@k, MRR, coverage."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd


def parse_code_list(value: Any) -> list[str]:
    """
    A list of codes from a list, a JSON or Python-literal string, or a scalar.

    Missing values and empty strings give an empty list; a string that is not a
    list literal is one code.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    if isinstance(value, str):
        text = value.strip()

        return parse_code_list(_literal(text)) if text else []

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item) != ""]

    return [str(value)]


def unique_preserve_order(values: Iterable[Any]) -> list[str]:
    """The distinct non-empty values as strings, in order of first occurrence."""
    seen: set[str] = set()
    unique: list[str] = []

    for value in values:
        item = str(value)

        if item and item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


def retrieval_metrics_from_codes(
    gold_codes: Sequence[Any],
    predicted_codes: Sequence[Sequence[Any] | Any],
    k_values: Sequence[int],
) -> dict[str, float | int]:
    """
    Recall@k for each k, MRR, coverage and mean candidate count over ranked code lists.

    Each prediction is parsed with `parse_code_list` and deduplicated before scoring;
    coverage is the share of mentions with at least one candidate.
    """
    n_mentions = len(gold_codes)
    metrics: dict[str, float | int] = {"n_mentions": n_mentions}
    metrics.update({f"recall@{k}": 0.0 for k in k_values})
    metrics.update({"mrr": 0.0, "coverage": 0.0, "mean_candidates": 0.0})

    if n_mentions == 0:
        return metrics

    hits_at_k = {int(k): 0 for k in k_values}
    reciprocal_rank = 0.0
    covered = 0
    total_candidates = 0

    for gold, predicted in zip(gold_codes, predicted_codes):
        gold = str(gold)
        ranked = unique_preserve_order(parse_code_list(predicted))
        total_candidates += len(ranked)
        covered += bool(ranked)

        if gold in ranked:
            reciprocal_rank += 1.0 / (ranked.index(gold) + 1)

        for k in k_values:
            hits_at_k[int(k)] += gold in ranked[: int(k)]

    for k in k_values:
        metrics[f"recall@{int(k)}"] = hits_at_k[int(k)] / n_mentions

    metrics["mrr"] = reciprocal_rank / n_mentions
    metrics["coverage"] = covered / n_mentions
    metrics["mean_candidates"] = total_candidates / n_mentions

    return metrics


def evaluate_candidate_rows(
    rows: Sequence[dict[str, Any]],
    k_values: Sequence[int],
    gold_col: str = "gold_code",
    codes_col: str = "codes",
    method: str | None = None,
    graph: Any | None = None,
    undirected_graph: Any | None = None,
) -> dict[str, float | int | str]:
    """
    Retrieval metrics over prediction rows, each with a gold code and a candidate list.

    `method` is recorded first when given; with both graphs, the ontology summary
    of each top candidate is added.
    """
    return _evaluate(
        gold=[row.get(gold_col) for row in rows],
        predicted=[row.get(codes_col, []) for row in rows],
        k_values=k_values,
        method=method,
        graph=graph,
        undirected_graph=undirected_graph,
    )


def evaluate_candidate_dataframe(
    df: pd.DataFrame,
    k_values: Sequence[int],
    gold_col: str = "gold_code",
    codes_col: str = "codes",
    method: str | None = None,
    graph: Any | None = None,
    undirected_graph: Any | None = None,
) -> dict[str, float | int | str]:
    """`evaluate_candidate_rows` over a DataFrame, its gold codes read as strings."""
    return _evaluate(
        gold=df[gold_col].astype(str).tolist() if len(df) else [],
        predicted=df[codes_col].tolist() if len(df) else [],
        k_values=k_values,
        method=method,
        graph=graph,
        undirected_graph=undirected_graph,
    )


def _evaluate(
    gold: list[Any],
    predicted: list[Any],
    k_values: Sequence[int],
    method: str | None,
    graph: Any | None,
    undirected_graph: Any | None,
) -> dict[str, float | int | str]:
    from lab.nel.evaluation.hierarchy import ontology_summary_from_codes

    metrics: dict[str, float | int | str] = dict(
        retrieval_metrics_from_codes(gold, predicted, k_values)
    )

    if method is not None:
        metrics = {"method": method, **metrics}

    if graph is not None and undirected_graph is not None:
        metrics.update(ontology_summary_from_codes(gold, predicted, graph, undirected_graph))

    return metrics


def _literal(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return [text]
