"""Scoring linked spans against the gold codes they carry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from lab.nel.evaluation import (
    ontology_summary_from_codes,
    read_ontology_graph_pickle,
    retrieval_metrics_from_codes,
)
from lab.nel.linking.tables import GOLD_COLUMN, optional_str
from lab.nel.schemas import LinkedEntity


def score_linking(
    result_df: pd.DataFrame,
    linked: list[LinkedEntity],
    k_values: Sequence[int],
    hierarchy: str | Path | None = None,
) -> dict[str, Any] | None:
    """
    Recall@k, MRR and coverage over the rows that carry a gold code, or None without any.

    `hierarchy` is a pickled SNOMED graph; given one, how each top candidate sits
    relative to its gold code (exact, narrower, broader, unrelated) and how far
    apart they are is reported too.
    """
    if GOLD_COLUMN not in result_df.columns:
        return None

    gold = result_df[GOLD_COLUMN].map(optional_str)
    valid = gold.notna()

    if not valid.any():
        return None

    gold_codes = gold[valid].tolist()
    candidate_codes = [
        [candidate.code for candidate in entity.candidates]
        for entity, has_gold in zip(linked, valid)
        if has_gold
    ]
    metrics: dict[str, Any] = retrieval_metrics_from_codes(
        gold_codes, candidate_codes, list(k_values)
    )

    if hierarchy is not None:
        graph, undirected = read_ontology_graph_pickle(hierarchy)
        metrics.update(ontology_summary_from_codes(gold_codes, candidate_codes, graph, undirected))

    metrics["n_input_rows"] = int(len(result_df))
    metrics["n_evaluated_rows"] = int(valid.sum())
    metrics["n_rows_without_gold"] = int((~valid).sum())

    return metrics
