"""Reconciling a corpus's sources before it is built: codes, mismatched files, conflicting text."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from lab.core.brat import is_usable_code, read_codes

MismatchPolicy = Literal["error", "documents", "annotations"]

ConflictPolicy = Literal["raise", "rewrite", "drop"]

CodeMismatchPolicy = Literal["raise", "drop"]

SPAN_KEY = ["filename", "start_span", "end_span"]


@dataclass(frozen=True)
class SourceMismatch:
    dropped_documents: list[str] = field(default_factory=list)
    dropped_annotation_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceConflict:
    rewritten_documents: list[str] = field(default_factory=list)
    n_rewritten_entities: int = 0
    dropped_documents: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CodeResolution:
    n_rows: int = 0
    n_unusable: int = 0
    dropped_unmatched: list[str] = field(default_factory=list)
    dropped_conflicting: list[str] = field(default_factory=list)


def resolve_codes(
    annotations: pd.DataFrame,
    codes: pd.DataFrame | str | Path,
    on_code_mismatch: CodeMismatchPolicy = "raise",
) -> tuple[pd.DataFrame, CodeResolution]:
    """
    Attach gold codes to annotations, joined on `(filename, start_span, end_span)`.

    Unusable codes (see `is_usable_code`) are skipped and identical rows count
    once, leaving the entity uncoded. A code row naming no annotated span, or
    two rows giving one span different codes, raises by default;
    `on_code_mismatch="drop"` skips them instead, leaving those spans uncoded.

    Returns the annotations with a `code` column, missing where uncoded. Call it
    before `resolve_mismatch` and `resolve_conflicts`, so rows of a document
    they drop go with it.
    """
    if on_code_mismatch not in ("raise", "drop"):
        raise ValueError(
            f"Unknown on_code_mismatch policy: {on_code_mismatch!r}. Expected 'raise' or 'drop'."
        )

    raw = read_codes(codes)
    usable = raw.loc[raw["code"].map(is_usable_code)].drop_duplicates()

    keys = pd.MultiIndex.from_frame(usable[SPAN_KEY])
    annotated = pd.MultiIndex.from_frame(
        annotations[SPAN_KEY].astype({"start_span": int, "end_span": int})
    )
    unmatched = ~keys.isin(annotated)
    conflicting = keys.duplicated(keep=False)

    unmatched_names = _span_names(usable.loc[unmatched])
    conflicting_names = _span_names(usable.loc[conflicting & ~unmatched])

    if unmatched_names or conflicting_names:
        problems = []

        if unmatched_names:
            problems.append(
                f"{len(unmatched_names)} code row(s) name no annotated span: {unmatched_names[:5]}"
            )

        if conflicting_names:
            problems.append(
                f"{len(conflicting_names)} span(s) carry different codes: {conflicting_names[:5]}"
            )

        if on_code_mismatch == "raise":
            raise ValueError(
                f"{'; '.join(problems)}. Pass on_code_mismatch='drop' to leave them uncoded."
            )

        warnings.warn(f"on_code_mismatch='drop' skipped {'; '.join(problems)}.", stacklevel=2)

    attached = usable.loc[~unmatched & ~conflicting]
    resolved = annotations.drop(columns="code", errors="ignore").merge(
        attached, on=SPAN_KEY, how="left"
    )

    return resolved, CodeResolution(
        n_rows=len(raw),
        n_unusable=int(len(raw) - raw["code"].map(is_usable_code).sum()),
        dropped_unmatched=unmatched_names,
        dropped_conflicting=conflicting_names,
    )


def resolve_mismatch(
    documents: dict[str, str],
    annotations: pd.DataFrame,
    on_mismatch: MismatchPolicy = "error",
) -> tuple[dict[str, str], pd.DataFrame, SourceMismatch]:
    """
    Reconcile a document set with an annotation set that names different files.

    `error` raises when an annotated filename has no document, which is the
    default: a missing document is usually a broken export rather than an
    intentional subset. `documents` lets the document set win, dropping
    annotations that name no document. `annotations` lets the annotation set
    win, additionally dropping documents that carry no annotation at all.

    An annotated filename with no document is dropped under both non-raising
    policies — there is no text for its offsets to refer to.
    """
    if on_mismatch not in ("error", "documents", "annotations"):
        raise ValueError(
            f"Unknown on_mismatch policy: {on_mismatch!r}. "
            "Expected 'error', 'documents', or 'annotations'."
        )

    filenames = annotations["filename"].astype(str)
    orphans = sorted(set(filenames) - set(documents))

    if on_mismatch == "error":
        if orphans:
            raise ValueError(
                f"{len(orphans)} annotated filename(s) have no matching document and "
                f"would be silently dropped: {orphans[:10]}. Pass "
                "on_mismatch='documents' to drop them, or on_mismatch='annotations' "
                "to keep only the documents that are annotated."
            )

        return documents, annotations, SourceMismatch()

    unannotated = sorted(set(documents) - set(filenames)) if on_mismatch == "annotations" else []
    dropped = set(unannotated)

    kept_documents = {name: text for name, text in documents.items() if name not in dropped}

    if documents and not kept_documents:
        raise ValueError(
            f"on_mismatch={on_mismatch!r} dropped every document: none of the "
            f"{len(documents)} document(s) carry an annotation."
        )

    kept_annotations = annotations.loc[filenames.isin(list(kept_documents))].reset_index(drop=True)

    return (
        kept_documents,
        kept_annotations,
        SourceMismatch(dropped_documents=unannotated, dropped_annotation_files=orphans),
    )


def resolve_conflicts(
    documents: dict[str, str],
    annotations: pd.DataFrame,
    on_conflict: ConflictPolicy = "raise",
) -> tuple[dict[str, str], pd.DataFrame, SourceConflict]:
    """
    Reconcile annotations whose text disagrees with the document it points into.

    `raise` is the default: an annotation whose text is not what its own offsets
    select is a broken export rather than a subset of the corpus. `rewrite`
    makes the document authoritative, replacing the annotated surface form with
    `text[start:end]` and keeping the offsets as annotated. `drop` removes every
    document that carries a conflict, along with all of its annotations.

    The document is never edited to match an annotation, so a rewritten corpus
    always trains on what its documents actually say. A dropped document takes
    its clean annotations with it: keeping them would leave a document that
    looks fully annotated while a real entity is silently missing.

    Expects the aligned pair `resolve_mismatch` returns: every annotated
    filename must have a document.
    """
    if on_conflict not in ("raise", "rewrite", "drop"):
        raise ValueError(
            f"Unknown on_conflict policy: {on_conflict!r}. Expected 'raise', 'rewrite', or 'drop'."
        )

    resolved = annotations.copy()
    orphans = sorted(set(resolved["filename"].astype(str)) - set(documents))

    if orphans:
        raise ValueError(
            f"{len(orphans)} annotated filename(s) have no matching document: "
            f"{orphans[:10]}. Reconcile the sources with resolve_mismatch first."
        )

    mentions = pd.Series(
        [
            documents[str(row.filename)][int(row.start_span) : int(row.end_span)]
            for row in resolved.itertuples(index=False)
        ],
        index=resolved.index,
        dtype=object,
    )
    conflicting = mentions != resolved["text"].astype(str)

    if not conflicting.any():
        return documents, resolved, SourceConflict()

    affected = sorted(set(resolved.loc[conflicting, "filename"].astype(str)))

    if on_conflict == "raise":
        examples = [
            f"{resolved.at[index, 'filename']}"
            f"[{resolved.at[index, 'start_span']}:{resolved.at[index, 'end_span']}] "
            f"annotated {resolved.at[index, 'text']!r}, document has {mentions[index]!r}"
            for index in resolved.index[conflicting.to_numpy()][:5]
        ]
        raise ValueError(
            f"{int(conflicting.sum())} annotation(s) in {len(affected)} document(s) "
            f"disagree with the document text: {examples}. Pass on_conflict='rewrite' "
            "to take the document text as authoritative, or on_conflict='drop' to "
            "drop the affected document(s)."
        )

    if on_conflict == "drop":
        dropped = set(affected)
        kept_documents = {name: text for name, text in documents.items() if name not in dropped}

        if documents and not kept_documents:
            raise ValueError(
                f"on_conflict='drop' dropped every document: all {len(documents)} "
                "document(s) carry an annotation that disagrees with their text."
            )

        kept_annotations = resolved.loc[
            ~resolved["filename"].astype(str).isin(dropped)
        ].reset_index(drop=True)

        warnings.warn(
            f"on_conflict='drop' dropped {len(affected)} document(s) carrying "
            f"{int(conflicting.sum())} annotation(s) that disagree with their "
            f"text: {affected[:10]}.",
            stacklevel=2,
        )

        return kept_documents, kept_annotations, SourceConflict(dropped_documents=affected)

    resolved.loc[conflicting, "text"] = mentions[conflicting]

    warnings.warn(
        f"on_conflict='rewrite' replaced the annotated text of "
        f"{int(conflicting.sum())} annotation(s) in {len(affected)} document(s) "
        f"with the document text: {affected[:10]}.",
        stacklevel=2,
    )

    return (
        documents,
        resolved,
        SourceConflict(
            rewritten_documents=affected,
            n_rewritten_entities=int(conflicting.sum()),
        ),
    )


def _span_names(rows: pd.DataFrame) -> list[str]:
    return sorted(
        {f"{row.filename}[{row.start_span}:{row.end_span}]" for row in rows.itertuples(index=False)}
    )
