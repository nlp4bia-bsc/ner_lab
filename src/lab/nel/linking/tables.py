"""Converting the span table and the gazetteer to NEL's records and back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.spans import SPAN_COLUMNS
from lab.nel.io import read_table
from lab.nel.schemas import Candidate, GazetteerEntry, LinkedEntity, Mention

GOLD_COLUMN = "gold_code"
LINKED_COLUMNS = ["code", "code_term", "code_score", "candidates_json"]
SOURCE_COLUMN = "code_source"
GAZETTEER_COLUMNS = ["term", "code"]


def read_spans(spans: pd.DataFrame | str | Path) -> pd.DataFrame:
    """
    Read a span table, taking any `code` column on it as gold.

    A table that was linked before carries `gold_code` already; its linking
    columns are dropped so they can be written afresh.
    """
    spans_df = spans.copy() if isinstance(spans, pd.DataFrame) else read_table(spans)
    missing = [column for column in SPAN_COLUMNS if column not in spans_df.columns]

    if missing:
        raise ValueError(f"Span table is missing columns {missing}. Required: {SPAN_COLUMNS}")

    if GOLD_COLUMN in spans_df.columns:
        linked_columns = [*LINKED_COLUMNS, SOURCE_COLUMN]
        spans_df = spans_df.drop(
            columns=[column for column in linked_columns if column in spans_df.columns]
        )
    elif "code" in spans_df.columns:
        spans_df = spans_df.rename(columns={"code": GOLD_COLUMN})

    return spans_df.reset_index(drop=True)


def read_gazetteer(gazetteer: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Read a gazetteer table with `term` and `code` columns."""
    gazetteer_df = (
        gazetteer.copy() if isinstance(gazetteer, pd.DataFrame) else read_table(gazetteer)
    )
    missing = [column for column in GAZETTEER_COLUMNS if column not in gazetteer_df.columns]

    if missing:
        raise ValueError(f"Gazetteer is missing columns {missing}. Required: {GAZETTEER_COLUMNS}")

    gazetteer_df = gazetteer_df.dropna(subset=GAZETTEER_COLUMNS).reset_index(drop=True)
    gazetteer_df["term"] = gazetteer_df["term"].astype(str)
    gazetteer_df["code"] = gazetteer_df["code"].astype(str)

    if gazetteer_df.empty:
        raise ValueError("Gazetteer contains no term-code pairs.")

    return gazetteer_df


def gazetteer_entries(gazetteer_df: pd.DataFrame) -> list[GazetteerEntry]:
    """Turn a gazetteer table into the records the matchers and retrievers index."""
    known = {"term", "code", "label", "semantic_tag", "sem_tag"}

    return [
        GazetteerEntry(
            term=str(row["term"]),
            code=str(row["code"]),
            label=optional_str(row.get("label")),
            semantic_tag=optional_str(row.get("semantic_tag", row.get("sem_tag"))),
            metadata={key: value for key, value in row.items() if key not in known},
        )
        for row in gazetteer_df.to_dict("records")
    ]


def mentions_from_spans(spans_df: pd.DataFrame) -> list[Mention]:
    """Turn span rows into the mention records the pipeline links."""
    if GOLD_COLUMN in spans_df.columns:
        gold_codes = [optional_str(value) for value in spans_df[GOLD_COLUMN]]
    else:
        gold_codes = [None] * len(spans_df)

    return [
        Mention(
            filename=str(row.filename),
            label=optional_str(row.label),
            start_span=int(row.start_span),
            end_span=int(row.end_span),
            text=str(row.text),
            code=gold_code,
        )
        for row, gold_code in zip(spans_df.itertuples(index=False), gold_codes)
    ]


def linked_frame(spans_df: pd.DataFrame, linked: list[LinkedEntity]) -> pd.DataFrame:
    """Append the linking columns to the span rows they were linked from."""
    result_df = spans_df.copy()
    result_df["code"] = [entity.predicted_code for entity in linked]
    result_df["code_term"] = [entity.predicted_term for entity in linked]
    result_df["code_score"] = [entity.score for entity in linked]
    result_df["candidates_json"] = [
        json.dumps(
            [candidate_record(candidate) for candidate in entity.candidates],
            ensure_ascii=False,
        )
        for entity in linked
    ]

    return result_df


def completed_frame(
    spans_df: pd.DataFrame,
    uncoded: list[bool],
    linked: list[LinkedEntity],
) -> pd.DataFrame:
    """Gold codes where the spans carry them, the linking columns everywhere else."""
    result_df = spans_df.copy()
    result_df[LINKED_COLUMNS] = None

    if linked:
        linked_df = linked_frame(spans_df.loc[uncoded], linked)
        result_df.loc[uncoded, LINKED_COLUMNS] = linked_df[LINKED_COLUMNS]

    if GOLD_COLUMN in spans_df.columns:
        coded = [not is_uncoded for is_uncoded in uncoded]
        result_df.loc[coded, "code"] = spans_df.loc[coded, GOLD_COLUMN].map(optional_str)

    result_df[SOURCE_COLUMN] = ["predicted" if is_uncoded else "gold" for is_uncoded in uncoded]

    return result_df


def candidate_record(candidate: Candidate) -> dict[str, Any]:
    """One candidate as plain data, without the mention fields the row already carries."""
    return {
        "code": candidate.code,
        "term": candidate.term,
        "score": candidate.score,
        "method": candidate.method,
        "rank": candidate.rank,
        "metadata": candidate.metadata,
    }


def optional_str(value: Any) -> str | None:
    """A cell as stripped text, or None when it is missing or blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()

    return text or None
