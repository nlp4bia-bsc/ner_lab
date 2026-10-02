"""The span table: one row per mention, the contract every subpackage reads and writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPAN_COLUMNS = ["filename", "label", "start_span", "end_span", "text"]
SCORED_SPAN_COLUMNS = [*SPAN_COLUMNS, "score"]

# Common aliases produced by scorers/shared-task resources. Canonical names win
# when both are present. Keeping this here makes span ingestion reusable by NER
# evaluation, analysis and any future task that consumes span tables.
SPAN_ALIASES = {
    "doc_id": "filename",
    "document_id": "filename",
    "off0": "start_span",
    "start": "start_span",
    "off1": "end_span",
    "end": "end_span",
    "span": "text",
    "entity_type": "label",
    "type": "label",
    "confidence": "score",
}


def validate_spans(
    spans: pd.DataFrame,
    *,
    scored: bool | None = None,
) -> pd.DataFrame:
    """Validate a span table without deduplicating or reordering its rows.

    Parameters
    ----------
    spans:
        Input span frame. Canonical columns are ``filename | label |
        start_span | end_span | text``. A small set of unambiguous aliases is
        accepted when the canonical column is absent.
    scored:
        ``True`` requires a numeric ``score`` column in ``[0, 1]``;
        ``False`` returns only the unscored canonical columns; ``None`` keeps
        and validates ``score`` when it is present.

    Notes
    -----
    This validator deliberately preserves duplicates. Duplicate removal is an
    output-construction policy of :func:`span_dataframe`, whereas evaluation
    and error analysis must be able to detect duplicate gold/prediction rows.
    """
    if not isinstance(spans, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, received {type(spans).__name__}.")

    frame = spans.copy()
    rename: dict[str, str] = {}

    for source, target in SPAN_ALIASES.items():
        if target not in frame.columns and source in frame.columns:
            rename[source] = target

    if rename:
        frame = frame.rename(columns=rename)

    missing = [column for column in SPAN_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Span frame is missing columns {missing}. Required: {SPAN_COLUMNS}."
        )

    canonical = frame[SPAN_COLUMNS].copy()
    canonical["filename"] = canonical["filename"].astype(str)
    canonical["label"] = canonical["label"].astype(str)
    canonical["text"] = canonical["text"].astype(str)

    for column in ("start_span", "end_span"):
        numeric = pd.to_numeric(canonical[column], errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Span column {column!r} contains non-finite values.")
        if not np.allclose(numeric.to_numpy(dtype=float), numeric.to_numpy(dtype=float).astype(np.int64)):
            raise ValueError(f"Span column {column!r} must contain integer offsets.")
        canonical[column] = numeric.astype("int64")

    if not canonical.empty:
        if (canonical["start_span"] < 0).any():
            raise ValueError("Invalid spans detected: start_span < 0.")
        if (canonical["end_span"] <= canonical["start_span"]).any():
            raise ValueError("Invalid spans detected: end_span <= start_span.")

    has_score = "score" in frame.columns
    if scored is True and not has_score:
        raise ValueError("A scored span frame requires a 'score' column.")

    keep_score = has_score and scored is not False
    if keep_score:
        score = pd.to_numeric(frame["score"], errors="raise").astype("float64")
        values = score.to_numpy(dtype=float)
        if score.isna().any() or not np.isfinite(values).all():
            raise ValueError("Prediction scores must be finite numeric values.")
        if ((score < 0.0) | (score > 1.0)).any():
            raise ValueError("Prediction scores must lie in the closed interval [0, 1].")
        canonical["score"] = score.to_numpy()

    return canonical.reset_index(drop=True)


def read_spans(
    value: pd.DataFrame | str | Path,
    *,
    scored: bool | None = None,
) -> pd.DataFrame:
    """Read canonical spans from a DataFrame, TSV or Parquet file.

    The function is intentionally lossless with respect to row multiplicity:
    duplicate annotations/predictions are preserved so downstream audits can
    report them. Use :func:`span_dataframe` when constructing inference output
    where window-level duplicates should be collapsed.
    """
    if isinstance(value, pd.DataFrame):
        return validate_spans(value, scored=scored)

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Span input does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a span file, received: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t", dtype=None, keep_default_na=False)
    else:
        raise ValueError(
            f"Unsupported span input {path}. Expected .tsv or .parquet."
        )

    return validate_spans(frame, scored=scored)


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

    A corpus whose entities carry gold codes adds a `code` column, missing on
    the uncoded rows.
    """
    if not isinstance(corpus, pd.DataFrame):
        corpus = pd.read_parquet(corpus, columns=["doc_id", "text", "entities_json"])

    rows: list[dict[str, Any]] = []
    has_codes = False

    for document in corpus.itertuples(index=False):
        text = str(document.text)
        entities = json.loads(document.entities_json) if document.entities_json else []

        for entity in entities:
            entity_label = str(entity["label"])

            if label and entity_label != label:
                continue

            start, end = int(entity["start"]), int(entity["end"])
            has_codes = has_codes or "code" in entity
            rows.append(
                {
                    "filename": str(document.doc_id),
                    "label": entity_label,
                    "start_span": start,
                    "end_span": end,
                    "text": text[start:end],
                    "code": entity.get("code"),
                }
            )

    columns = [*SPAN_COLUMNS, "code"] if has_codes else SPAN_COLUMNS

    if not rows:
        return pd.DataFrame(columns=SPAN_COLUMNS)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["filename", "start_span", "end_span"])
        .reset_index(drop=True)
    )
