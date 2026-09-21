"""The span table: one row per mention, the contract every subpackage reads and writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SPAN_COLUMNS = ["filename", "label", "start_span", "end_span", "text"]
SCORED_SPAN_COLUMNS = [*SPAN_COLUMNS, "score"]


def span_dataframe(
    spans: list[dict[str, Any]],
    scored: bool | None = None,
) -> pd.DataFrame:
    """
    Build the canonical span frame, dropping duplicate spans.

    Scored spans keep the highest-scoring copy of a span two windows both
    predicted, and come back in document order; unscored ones keep the first
    copy and stay in the order the rows produced them.

    `scored` is read off the spans themselves unless it is given. A caller that
    decoded scores passes it, so predicting nothing still returns a frame with
    the `score` column its next step reads.
    """
    scored = (bool(spans) and "score" in spans[0]) if scored is None else scored
    columns = SCORED_SPAN_COLUMNS if scored else SPAN_COLUMNS

    if not spans:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(spans, columns=columns)

    if scored:
        frame = frame.sort_values("score", ascending=False)

    frame = frame.drop_duplicates(subset=["filename", "label", "start_span", "end_span"])

    if scored:
        frame = frame.sort_values(["filename", "start_span", "end_span"])

    return frame.reset_index(drop=True)


def spans_from_corpus(
    corpus: pd.DataFrame | str | Path,
    label: str | None = None,
) -> pd.DataFrame:
    """
    Turn a canonical corpus into a span table, one row per entity.

    Labels are taken verbatim and overlaps are left as annotated: this is the
    gold as distributed, not as an overlap policy would rewrite it. Span text is
    re-derived from the document rather than trusted from the entity.
    """
    if not isinstance(corpus, pd.DataFrame):
        corpus = pd.read_parquet(corpus, columns=["doc_id", "text", "entities_json"])

    rows: list[dict[str, Any]] = []

    for document in corpus.itertuples(index=False):
        text = str(document.text)
        entities = json.loads(document.entities_json) if document.entities_json else []

        for entity in entities:
            entity_label = str(entity["label"])

            if label and entity_label != label:
                continue

            start, end = int(entity["start"]), int(entity["end"])
            rows.append(
                {
                    "filename": str(document.doc_id),
                    "label": entity_label,
                    "start_span": start,
                    "end_span": end,
                    "text": text[start:end],
                }
            )

    if not rows:
        return pd.DataFrame(columns=SPAN_COLUMNS)

    return (
        pd.DataFrame(rows, columns=SPAN_COLUMNS)
        .sort_values(["filename", "start_span", "end_span"])
        .reset_index(drop=True)
    )
