"""The canonical corpus: one row per document, `doc_id | text | entities_json | n_entities`."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from lab.core.brat import NO_CODE, read_annotations, resolve_documents
from lab.core.sources import (
    CodeMismatchPolicy,
    ConflictPolicy,
    MismatchPolicy,
    resolve_codes,
    resolve_conflicts,
    resolve_mismatch,
)
from lab.core.stratification import parse_document_label_counts

DOCUMENT_COLUMNS = ["doc_id", "text", "entities_json", "n_entities"]

DOCUMENT_DTYPES = {
    "doc_id": "string",
    "text": "string",
    "entities_json": "string",
    "n_entities": "int32",
}


def count_codes(documents_df: pd.DataFrame) -> dict[str, object]:
    """How many entities carry a gold code, overall and by label, and what those codes look like."""
    by_label: dict[str, int] = {}
    codes: list[str] = []

    for entities_json in documents_df["entities_json"]:
        for entity in json.loads(entities_json) if entities_json else []:
            if "code" in entity:
                codes.append(entity["code"])
                by_label[entity["label"]] = by_label.get(entity["label"], 0) + 1

    return {
        "has_codes": bool(codes),
        "n_entities_with_code": len(codes),
        "n_entities_with_code_by_label": dict(sorted(by_label.items())),
        "n_no_code": sum(code == NO_CODE for code in codes),
        "n_composite_codes": sum("+" in code for code in codes),
        "n_unique_codes": len(set(codes)),
    }


def build_corpus(
    documents: dict[str, str] | str | Path,
    annotations: pd.DataFrame | str | Path,
    normalize_labels: bool = False,
    on_mismatch: MismatchPolicy = "error",
    on_conflict: ConflictPolicy = "raise",
    codes: pd.DataFrame | str | Path | None = None,
    on_code_mismatch: CodeMismatchPolicy = "raise",
    validate: bool = True,
) -> pd.DataFrame:
    """
    Build the canonical corpus DataFrame from documents and their annotations.

    `entities_json` holds a JSON list of `{id, start, end, label, text}` per
    document, plus `code` on the entities that carry one. Entity offsets are
    round-trip validated against the document text, so an offset bug surfaces
    here rather than as a mislabelled token later.
    Overlapping entities are preserved as annotated; resolving them is a
    modelling choice made at encoding time.

    Source label strings are kept verbatim. Set `normalize_labels=True` to map
    them through `LABEL_ALIASES` onto this library's clinical vocabulary, and
    `validate=False` to skip the whole-corpus check on a corpus you trust.

    Documents and annotations that name different files raise by default; see
    `resolve_mismatch` for the `on_mismatch` policies that reconcile them
    instead. An annotation whose text disagrees with the document it points into
    also raises by default; set `on_conflict="rewrite"` to take the document
    text as authoritative, or `on_conflict="drop"` to drop the documents that
    carry a conflict. Call either resolver directly to learn what a policy
    dropped or rewrote.

    `codes` is a linking TSV whose gold codes are added to the entities they
    name, as a `code` field; see `resolve_codes` for `on_code_mismatch`.
    """
    documents_dict = resolve_documents(documents)
    annotations_df = read_annotations(annotations, normalize_labels=normalize_labels)

    if codes is not None:
        annotations_df, _ = resolve_codes(annotations_df, codes, on_code_mismatch)

    documents_dict, annotations_df, _ = resolve_mismatch(
        documents_dict, annotations_df, on_mismatch
    )
    documents_dict, annotations_df, _ = resolve_conflicts(
        documents_dict, annotations_df, on_conflict
    )

    rows: list[dict[str, object]] = []

    for filename in sorted(documents_dict):
        text = documents_dict[filename]

        if not text.strip():
            raise ValueError(f"Document {filename!r} has empty text.")

        entities = _build_entities(
            filename, text, annotations_df[annotations_df["filename"] == filename]
        )

        rows.append(
            {
                "doc_id": filename,
                "text": text,
                "entities_json": json.dumps(entities, ensure_ascii=False),
                "n_entities": len(entities),
            }
        )

    corpus = pd.DataFrame(rows, columns=DOCUMENT_COLUMNS).astype(DOCUMENT_DTYPES)

    return validate_corpus(corpus) if validate else corpus


def validate_corpus(documents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a corpus DataFrame and return it reduced to the canonical columns.

    Checks unique non-null `doc_id`, non-empty text, non-negative `n_entities`,
    that every entity's offsets round-trip against its document text, and that
    any `code` an entity carries is a non-empty string.
    """
    missing = set(DOCUMENT_COLUMNS) - set(documents_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    corpus = documents_df[DOCUMENT_COLUMNS].copy()
    corpus["doc_id"] = corpus["doc_id"].astype("string")
    corpus["n_entities"] = pd.to_numeric(corpus["n_entities"], errors="raise").astype("int32")

    if corpus["doc_id"].isna().any():
        raise ValueError("doc_id contains missing values.")

    if corpus["doc_id"].duplicated().any():
        examples = corpus.loc[corpus["doc_id"].duplicated(keep=False), "doc_id"].head(10).tolist()
        raise ValueError(f"doc_id must be unique. Examples: {examples}")

    if (corpus["n_entities"] < 0).any():
        raise ValueError("n_entities cannot be negative.")

    blank_text = corpus["text"].isna() | corpus["text"].astype(str).str.strip().eq("")

    if blank_text.any():
        examples = corpus.loc[blank_text, "doc_id"].head(10).tolist()
        raise ValueError(
            f"{int(blank_text.sum())} document(s) have missing or empty text: {examples}"
        )

    for row in corpus.itertuples(index=False):
        _validate_entities(str(row.doc_id), str(row.text), str(row.entities_json))

    return corpus.reset_index(drop=True)


def count_labels(documents_df: pd.DataFrame) -> dict[str, int]:
    """
    How many entities carry each label across the corpus, keyed by label.

    Counts reflect the corpus as annotated: overlapping entities are resolved at
    encoding time, so the totals describe the corpus rather than the entity set
    any training run sees. Raises if a document's `n_entities` disagrees with its
    `entities_json`.
    """
    labels, per_document = parse_document_label_counts(documents_df)
    totals = per_document.sum(axis=0)

    return {label: int(totals[index]) for index, label in enumerate(labels)}


def document_fingerprints(documents_df: pd.DataFrame) -> pd.Series:
    """Stable per-document hash over the canonical columns, for split provenance."""
    return pd.Series(
        [_document_fingerprint(row) for row in documents_df.itertuples(index=False)],
        index=documents_df.index,
        dtype="string",
    )


def _build_entities(
    filename: str,
    text: str,
    document_annotations: pd.DataFrame,
) -> list[dict[str, object]]:
    entities: list[dict[str, object]] = []
    has_codes = "code" in document_annotations.columns

    sorted_rows = sorted(
        document_annotations.itertuples(index=False),
        key=lambda row: (int(row.start_span), int(row.end_span)),
    )

    for index, row in enumerate(sorted_rows, start=1):
        start, end = int(row.start_span), int(row.end_span)
        mention = text[start:end]

        if mention != row.text:
            raise ValueError(
                f"Offset round-trip mismatch in document {filename!r}: "
                f"expected {row.text!r} at [{start}:{end}], got {mention!r}."
            )

        entity: dict[str, object] = {
            "id": f"T{index}",
            "start": start,
            "end": end,
            "label": str(row.label),
            "text": mention,
        }

        if has_codes and pd.notna(row.code):
            entity["code"] = str(row.code)

        entities.append(entity)

    return entities


def _validate_entities(doc_id: str, text: str, entities_json: str) -> None:
    try:
        entities = json.loads(entities_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid entities_json for document {doc_id!r}.") from error

    if not isinstance(entities, list):
        raise ValueError(f"entities_json must be a JSON list for {doc_id!r}.")

    for position, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"Entity #{position} in document {doc_id!r} is not a JSON object.")

        name = entity.get("id", f"#{position}")
        start = entity.get("start")
        end = entity.get("end")

        valid_offsets = isinstance(start, int) and isinstance(end, int)

        if not valid_offsets or not 0 <= start < end <= len(text):
            raise ValueError(
                f"Entity {name} in document {doc_id!r} has invalid offsets "
                f"start={start!r}, end={end!r} (document has {len(text)} characters)."
            )

        code = entity.get("code")

        if "code" in entity and (not isinstance(code, str) or not code.strip()):
            raise ValueError(
                f"Entity {name} in document {doc_id!r} has an invalid code {code!r}; "
                "a code is a non-empty string, and an uncoded entity has no `code` key."
            )

        mention = entity.get("text")

        if mention is not None and text[start:end] != mention:
            raise ValueError(
                f"Offset round-trip mismatch in document {doc_id!r}, entity {name}: "
                f"expected {mention!r} at [{start}:{end}], got {text[start:end]!r}."
            )


def _document_fingerprint(row: object) -> str:
    payload = json.dumps(
        {
            "doc_id": str(row.doc_id),
            "text": str(row.text),
            "entities_json": str(row.entities_json),
            "n_entities": int(row.n_entities),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
