"""What inference reads and writes: documents, gold references, span tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lab.core.brat import read_annotation_tsv, resolve_documents
from lab.core.corpus import DOCUMENT_COLUMNS, validate_corpus
from lab.core.spans import SCORED_SPAN_COLUMNS, SPAN_COLUMNS, spans_from_corpus


def read_documents(
    documents: pd.DataFrame | dict[str, str] | str | Path,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read what is to be predicted on into a canonical corpus frame.

    A corpus frame or parquet keeps its `entities_json`, which is what makes it
    scorable. Raw text has none, so every document gets an empty entity list —
    the encoder still labels every content token `O`, which is exactly the
    trustworthy position span reconstruction decodes.
    """
    if isinstance(documents, pd.DataFrame):
        return validate_corpus(documents)

    path = Path(documents) if not isinstance(documents, dict) else None

    if path is not None and path.is_file():
        return validate_corpus(pd.read_parquet(path))

    texts = resolve_documents(documents, pattern=pattern, encoding=encoding)

    if not texts:
        raise FileNotFoundError(f"No documents matching {pattern!r} in {documents}.")

    return validate_corpus(
        pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "text": text,
                    "entities_json": json.dumps([]),
                    "n_entities": 0,
                }
                for doc_id, text in sorted(texts.items())
            ],
            columns=DOCUMENT_COLUMNS,
        )
    )


def has_gold(corpus: pd.DataFrame) -> bool:
    """Whether a corpus carries any annotation to score predictions against."""
    return bool(corpus["n_entities"].sum())


def read_reference(reference: str | Path, target_label: str | None = None) -> pd.DataFrame:
    """Read a gold span table to score against, from an annotation TSV or a corpus parquet."""
    path = Path(reference)

    if not path.exists():
        raise FileNotFoundError(f"Reference does not exist: {path}")

    if path.suffix == ".parquet":
        return spans_from_corpus(path, label=target_label)

    annotations = read_annotation_tsv(path)

    if target_label:
        annotations = annotations[annotations["label"] == target_label].reset_index(drop=True)

    return annotations


def write_predictions(spans: pd.DataFrame, path: str | Path) -> Path:
    """Write predicted entities as a TSV: the span table plus its `score` column."""
    path = Path(path)
    spans[SCORED_SPAN_COLUMNS].to_csv(path, sep="\t", index=False)

    return path


def write_gold(spans: pd.DataFrame, path: str | Path) -> Path:
    """Write the gold spans a run was scored against, as a span table TSV."""
    path = Path(path)
    spans[SPAN_COLUMNS].to_csv(path, sep="\t", index=False)

    return path
