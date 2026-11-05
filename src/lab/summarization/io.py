"""The summaries table: one row per summarised document, read from JSONL, parquet or a frame."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

SUMMARY_COLUMNS = ("doc_id", "language", "text", "reference_summary", "generated_summary")
JSONL_RENAMES = {"case_id": "doc_id", "full_case": "text"}
FAILED_SUMMARY_PREFIX = "ERROR:"

FailedPolicy = Literal["raise", "drop"]

FAILED_POLICIES = ("raise", "drop")


def read_summaries(
    summaries: pd.DataFrame | str | Path | Sequence[str | Path],
    on_failed: FailedPolicy = "raise",
) -> pd.DataFrame:
    """
    Read summaries to evaluate into a summaries frame.

    `summaries` is a frame, a `.jsonl` or `.parquet` file, or a list of such files,
    which are concatenated so that several languages can be evaluated in one run.
    A row is one document: `doc_id`, `language`, the source `text`, the
    `reference_summary` and the `generated_summary`. The JSONL files the
    MultiClinSum baselines and `prepare_submission.py` write name the first and
    third `case_id` and `full_case`; they are renamed on reading.

    A generated summary starting with `ERROR:` is how those inference scripts
    record a failed generation. Such rows raise by default; `on_failed="drop"`
    evaluates without them and warns with their count.
    """
    if on_failed not in FAILED_POLICIES:
        raise ValueError(f"on_failed must be one of {FAILED_POLICIES}, got {on_failed!r}.")

    if isinstance(summaries, pd.DataFrame):
        summaries_df = summaries.rename(columns=JSONL_RENAMES)
    elif isinstance(summaries, (str, Path)):
        summaries_df = _read_file(Path(summaries))
    else:
        summaries_df = pd.concat([_read_file(Path(path)) for path in summaries], ignore_index=True)

    summaries_df = validate_summaries(summaries_df)

    return _resolve_failed(summaries_df, on_failed)


def validate_summaries(summaries_df: pd.DataFrame) -> pd.DataFrame:
    """Check a summaries frame's columns, values and keys, and return it with string ids."""
    missing = [column for column in SUMMARY_COLUMNS if column not in summaries_df.columns]

    if missing:
        raise ValueError(
            f"Summaries are missing columns {missing}; expected {list(SUMMARY_COLUMNS)} "
            f"(or case_id/full_case in place of doc_id/text). Got {list(summaries_df.columns)}."
        )

    if summaries_df.empty:
        raise ValueError("The summaries table is empty; there is nothing to evaluate.")

    summaries_df = summaries_df.loc[:, list(SUMMARY_COLUMNS)].reset_index(drop=True)
    summaries_df["doc_id"] = summaries_df["doc_id"].astype(str)

    for column in SUMMARY_COLUMNS[1:]:
        not_text = ~summaries_df[column].map(lambda value: isinstance(value, str))

        if not_text.any():
            raise ValueError(
                f"{int(not_text.sum())} row(s) have a non-string {column}, e.g. doc_id "
                f"{summaries_df.loc[not_text, 'doc_id'].tolist()[:10]}."
            )

    repeated = summaries_df.duplicated(subset=["doc_id", "language"], keep=False)

    if repeated.any():
        pairs = summaries_df.loc[repeated, ["doc_id", "language"]].drop_duplicates()

        raise ValueError(
            f"{len(pairs)} (doc_id, language) pair(s) occur more than once, e.g. "
            f"{list(pairs.itertuples(index=False, name=None))[:10]}. Each document is "
            "evaluated once per language."
        )

    return summaries_df


def _read_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No summaries file at {path}.")

    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False).rename(columns=JSONL_RENAMES)

    if path.suffix == ".parquet":
        return pd.read_parquet(path).rename(columns=JSONL_RENAMES)

    raise ValueError(f"{path}: summaries are read from .jsonl or .parquet, not {path.suffix!r}.")


def _resolve_failed(summaries_df: pd.DataFrame, on_failed: str) -> pd.DataFrame:
    failed = summaries_df["generated_summary"].str.startswith(FAILED_SUMMARY_PREFIX)

    if not failed.any():
        return summaries_df

    examples = summaries_df.loc[failed, ["doc_id", "language"]]
    examples = list(examples.itertuples(index=False, name=None))[:10]

    if on_failed == "raise":
        raise ValueError(
            f"{int(failed.sum())} generated summaries start with {FAILED_SUMMARY_PREFIX!r}, "
            f"which marks a failed generation: {examples}. Pass on_failed='drop' to evaluate "
            "without them."
        )

    warnings.warn(
        f"Dropped {int(failed.sum())} of {len(summaries_df)} summaries whose generation "
        f"failed ({FAILED_SUMMARY_PREFIX!r}), e.g. {examples}.",
        stacklevel=3,
    )

    return summaries_df.loc[~failed].reset_index(drop=True)
