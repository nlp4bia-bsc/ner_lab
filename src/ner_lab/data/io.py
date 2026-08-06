"""Reading and writing corpus parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ner_lab.data.corpus import validate_corpus

DEFAULT_PARQUET_COMPRESSION = "zstd"


def read_corpus(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """Read a corpus parquet, validated against the canonical schema by default."""
    parquet_path = Path(path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    documents_df = pd.read_parquet(parquet_path)

    return validate_corpus(documents_df) if validate else documents_df


def write_corpus(
    documents_df: pd.DataFrame,
    path: str | Path,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> Path:
    """Write a corpus DataFrame to parquet, creating parent directories."""
    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    documents_df.to_parquet(parquet_path, engine="pyarrow", compression=compression, index=False)

    return parquet_path
