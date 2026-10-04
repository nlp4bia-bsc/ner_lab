"""How a predicted code sits relative to the gold code in an ontology graph."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx
import pandas as pd

from lab.nel.evaluation.metrics import (
    parse_code_list,
    retrieval_metrics_from_codes,
    unique_preserve_order,
)

SUMMARY_DIRECTIONS = ("exact", "narrow", "broad", "non_rel")


def graph_distance_and_direction(
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    gold_code: str | int | None,
    pred_code: str | int | None,
    return_path: bool = False,
) -> dict[str, Any]:
    """
    Classify a prediction against the gold code as exact, narrow, broad or unrelated.

    In a `parent -> child` graph: `exact` is the same code; `narrow` a descendant
    of the gold code; `broad` an ancestor, or a relative with the same semantic tag
    that does not descend from the gold code; `unrelated` anything missing,
    disconnected, or a relative with a different tag. `distance` is the length of
    the path found; `return_path` adds the path itself.
    """
    gold = str(gold_code) if gold_code is not None else None
    pred = str(pred_code) if pred_code is not None else None
    gold_tag = graph.nodes[gold].get("semantic_tag") if gold in graph else None
    pred_tag = graph.nodes[pred].get("semantic_tag") if pred in graph else None
    same_tag = bool(gold_tag and pred_tag and gold_tag == pred_tag)

    direction, distance, path = _relation(graph, undirected_graph, gold, pred, same_tag)
    result: dict[str, Any] = {
        "gold": gold,
        "pred": pred,
        "distance": distance,
        "direction": direction,
        "gold_semantic_tag": gold_tag,
        "pred_semantic_tag": pred_tag,
        "same_semantic_tag": same_tag,
    }

    if return_path:
        result["path"] = path

    return result


def ontology_summary_from_codes(
    gold_codes: Iterable[Any],
    predicted_codes: Iterable[Any],
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
) -> dict[str, float | int]:
    """
    Count and rate of exact, narrow, broad and unrelated (`non_rel`) top-1 predictions.

    Also the mean graph distance over the predictions that have one.
    """
    counts = dict.fromkeys(SUMMARY_DIRECTIONS, 0)
    distances: list[float] = []
    gold_list = list(gold_codes)

    for gold, predictions in zip(gold_list, list(predicted_codes)):
        ranked = unique_preserve_order(parse_code_list(predictions))
        result = graph_distance_and_direction(
            graph, undirected_graph, gold, ranked[0] if ranked else None
        )
        direction = str(result.get("direction"))
        counts[direction if direction in counts else "non_rel"] += 1

        if result.get("distance") is not None:
            distances.append(float(result["distance"]))

    n_mentions = len(gold_list)
    summary: dict[str, float | int] = {}

    for direction, count in counts.items():
        summary[f"{direction}_count"] = count
        summary[f"{direction}_rate"] = count / n_mentions if n_mentions else 0.0

    summary["mean_graph_distance"] = sum(distances) / len(distances) if distances else 0.0

    return summary


def graph_metrics_for_dataframe(
    df: pd.DataFrame,
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    gold_col: str = "code",
    codes_col: str = "codes",
    label_col: str | None = "label",
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Top-1 graph relations and recall@k over a prediction table, overall and by label.

    `per_row` holds each row's `graph_distance_and_direction`; the summaries count
    and average them by direction and by whether the semantic tags match.
    """
    k_values = k_values or [1, 5]
    per_row = df.apply(
        lambda row: graph_distance_and_direction(
            graph,
            undirected_graph,
            gold_code=row[gold_col],
            pred_code=_first_code(row[codes_col]),
        ),
        axis=1,
        result_type="expand",
    )

    direction_summary = _summary(per_row, "direction")
    direction_summary["normalized"] = (
        direction_summary["count"] / len(per_row) if len(per_row) else 0.0
    )

    report: dict[str, Any] = {
        "direction_summary": direction_summary.to_dict("records"),
        "same_semantic_tag_summary": _summary(per_row, "same_semantic_tag").to_dict("records"),
        "mean_distance_global": float(per_row["distance"].mean()) if len(per_row) else 0.0,
        "recall_at_k": _recall_at_k(df, k_values, gold_col, codes_col),
        "per_row": per_row,
    }

    if label_col and label_col in df.columns:
        report["by_label"] = {
            str(label): {
                "n": len(label_df),
                "direction_summary": _summary(per_row.loc[label_df.index], "direction").to_dict(
                    "records"
                ),
                "recall_at_k": _recall_at_k(label_df, k_values, gold_col, codes_col),
                "mean_distance": (
                    float(per_row.loc[label_df.index, "distance"].mean()) if len(label_df) else 0.0
                ),
            }
            for label, label_df in df.groupby(label_col)
        }

    return report


def _relation(
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    gold: str | None,
    pred: str | None,
    same_tag: bool,
) -> tuple[str, int | None, list[str] | None]:
    if gold is None or pred is None:
        return "unrelated", None, None

    if gold == pred:
        return "exact", 0, [gold]

    if gold not in graph or pred not in graph:
        return "unrelated", None, None

    if nx.has_path(graph, gold, pred):
        path = nx.shortest_path(graph, gold, pred)

        return "narrow", len(path) - 1, path

    if nx.has_path(graph, pred, gold):
        path = nx.shortest_path(graph, pred, gold)

        return "broad", len(path) - 1, path

    try:
        path = nx.shortest_path(undirected_graph, gold, pred)
    except nx.NetworkXNoPath:
        return "unrelated", None, None

    return _relative_direction(graph, gold, pred, same_tag), len(path) - 1, path


def _relative_direction(graph: nx.DiGraph, gold: str, pred: str, same_tag: bool) -> str:
    if not same_tag:
        return "unrelated"

    common = ({gold} | nx.ancestors(graph, gold)) & ({pred} | nx.ancestors(graph, pred))

    if not common:
        return "unrelated"

    best_ancestor = None
    best_score = None
    best_gold_distance = None

    for ancestor in common:
        try:
            distance_to_gold = nx.shortest_path_length(graph, ancestor, gold)
            distance_to_pred = nx.shortest_path_length(graph, ancestor, pred)
        except nx.NetworkXNoPath:
            continue

        score = distance_to_gold + distance_to_pred

        if best_score is None or score < best_score:
            best_score = score
            best_ancestor = ancestor
            best_gold_distance = distance_to_gold

    if best_ancestor is None or best_gold_distance is None:
        return "unrelated"

    if best_ancestor != gold and best_gold_distance > 0:
        return "broad"

    return "narrow" if nx.has_path(graph, gold, best_ancestor) else "broad"


def _summary(per_row: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        per_row.groupby(column, dropna=False)
        .agg(count=(column, "size"), mean_distance=("distance", "mean"))
        .reset_index()
    )


def _first_code(value: Any) -> str | None:
    codes = parse_code_list(value)

    return codes[0] if codes else None


def _recall_at_k(
    df: pd.DataFrame, k_values: list[int], gold_col: str, codes_col: str
) -> dict[int, float]:
    metrics = retrieval_metrics_from_codes(
        df[gold_col].astype(str).tolist(), df[codes_col].tolist(), k_values
    )

    return {k: metrics[f"recall@{k}"] for k in k_values}
