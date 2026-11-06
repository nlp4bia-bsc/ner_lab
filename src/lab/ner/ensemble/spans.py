"""Exact span union and consensus without suppressing overlapping mentions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from lab.core.segmentation import split_into_words
from lab.core.spans import span_dataframe
from lab.ner.ensemble.tables import MemberPredictions

SPAN_METHODS = ("span_union", "span_consensus", "span_intersection")
IDENTITY = ["filename", "label", "start_span", "end_span"]


def combine_spans(
    predictions: Sequence[MemberPredictions],
    entity_labels: Sequence[str],
    method: str,
    texts: dict[str, str],
    min_support: int | None = None,
) -> pd.DataFrame:
    """Combine exact identities; absent label/word coverage is never a negative vote.

    A non-supporting member is eligible only when it covers every UAX word the
    candidate overlaps. Supporting members are eligible by construction. Union scores
    and consensus scores are the mean original confidence among supporting members.
    """
    if method not in SPAN_METHODS:
        raise ValueError(f"Span method must be one of {SPAN_METHODS}.")
    if min_support is not None and (type(min_support) is not int or min_support < 1):
        raise ValueError("min_support must be a positive integer.")
    grouped: dict[tuple, dict[int, dict]] = {}
    coverage = []
    for member, prediction in enumerate(predictions):
        covered = {
            key: set(group["word_id"].astype(int))
            for key, group in prediction.words.groupby(["doc_id", "entity_label"])
        }
        coverage.append(covered)
        for span in prediction.spans.to_dict("records"):
            if span["label"] not in entity_labels:
                continue
            key = tuple(span[column] for column in IDENTITY)
            prior = grouped.setdefault(key, {}).get(member)
            if prior is None or prior["score"] < span["score"]:
                grouped[key][member] = span
    boundaries = {doc_id: split_into_words(text) for doc_id, text in texts.items()}
    results = []
    for (doc_id, label, start, end), supporters in sorted(grouped.items()):
        word_ids = {
            index
            for index, word in enumerate(boundaries[doc_id])
            if word["start"] < end and word["end"] > start
        }
        eligible = sum(
            member in supporters
            or (
                label in prediction.entity_labels
                and bool(word_ids)
                and word_ids.issubset(coverage[member].get((doc_id, label), set()))
            )
            for member, prediction in enumerate(predictions)
        )
        support = len(supporters)
        keep = method == "span_union"
        if method != "span_union":
            required = (
                min_support
                if min_support is not None
                else (eligible if method == "span_intersection" else eligible // 2 + 1)
            )
            keep = support >= required
        if keep:
            results.append(
                {
                    "filename": doc_id,
                    "label": label,
                    "start_span": start,
                    "end_span": end,
                    "text": texts[doc_id][start:end],
                    "score": float(np.mean([span["score"] for span in supporters.values()])),
                }
            )

    return span_dataframe(results, scored=True)
