"""Turning encoded rows into the Dataset a Trainer consumes."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd
from datasets import Dataset

REQUIRED_COLUMNS = ("input_ids", "attention_mask", "labels")
OPTIONAL_COLUMNS = ("token_type_ids",)


def to_dataset(rows: pd.DataFrame) -> Dataset:
    """
    Convert `Encoder` output into a `datasets.Dataset`.

    Nothing is re-tokenized: the rows already hold model-ready integer fields.
    Traceability columns (`doc_id`, `window_start`, `window_end`) are dropped
    here rather than passed to the Trainer with `remove_unused_columns=False`.
    """
    validate_rows(rows)

    records = []

    for row in rows.itertuples(index=False):
        record = {column: ensure_int_list(getattr(row, column)) for column in REQUIRED_COLUMNS}

        for column in OPTIONAL_COLUMNS:
            value = getattr(row, column, None)

            if value is not None:
                record[column] = ensure_int_list(value)

        records.append(record)

    return Dataset.from_list(records)


def validate_rows(rows: pd.DataFrame, name: str = "rows") -> None:
    """Check that a frame carries the columns a Trainer needs."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(rows.columns))

    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")

    if rows.empty:
        raise ValueError(f"{name} is empty.")


def is_encoded(rows: pd.DataFrame) -> bool:
    """True when a frame already holds model-ready token-level columns."""
    return set(REQUIRED_COLUMNS).issubset(rows.columns)


def ensure_int_list(value: Any) -> list[int]:
    """Coerce one cell to a list of ints, parsing the string form parquet can yield."""
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    if isinstance(value, str):
        parsed = ast.literal_eval(value)

        if not isinstance(parsed, list):
            raise TypeError(f"Expected a list-like string, received: {value!r}")

        return [int(item) for item in parsed]

    if hasattr(value, "tolist"):
        return [int(item) for item in value.tolist()]

    raise TypeError(f"Expected a list-like value, received: {type(value).__name__}")
