"""Reading BRAT-style source corpora: plain-text documents plus .ann or TSV annotations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ner_lab.data.labels import normalize_annotation_labels

ANNOTATION_COLUMNS = ["filename", "label", "start_span", "end_span", "text"]


def resolve_documents(
    documents: dict[str, str] | str | Path,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
) -> dict[str, str]:
    """Return a `{document name: text}` mapping from a directory or an existing mapping."""
    if isinstance(documents, dict):
        return {str(name): str(text) for name, text in documents.items()}

    documents_dir = Path(documents)

    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents path does not exist: {documents_dir}")

    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory of documents: {documents_dir}")

    return {
        file_path.stem: file_path.read_text(encoding=encoding)
        for file_path in sorted(documents_dir.glob(pattern))
        if file_path.is_file()
    }


def read_annotations(
    annotations: pd.DataFrame | str | Path,
    normalize_labels: bool = True,
) -> pd.DataFrame:
    """
    Read annotations into the canonical schema, from a DataFrame, a TSV file, a
    single .ann file, or a directory of .ann files.

    Set `normalize_labels=False` to keep the source label strings verbatim, for
    corpora whose vocabulary is not the clinical one this library defaults to.
    """
    if isinstance(annotations, pd.DataFrame):
        return _canonicalize_annotations(annotations, normalize_labels)

    annotations_path = Path(annotations)

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations path does not exist: {annotations_path}")

    if annotations_path.is_dir() or annotations_path.suffix == ".ann":
        return read_ann(annotations_path, normalize_labels=normalize_labels)

    if annotations_path.suffix == ".tsv":
        return read_annotation_tsv(annotations_path, normalize_labels=normalize_labels)

    raise ValueError(
        f"Unsupported annotations input: {annotations_path}. "
        "Expected a DataFrame, a .tsv file, a .ann file, or a directory of .ann files."
    )


def read_annotation_tsv(path: str | Path, normalize_labels: bool = True) -> pd.DataFrame:
    """Read a TSV holding the canonical annotation columns."""
    annotations = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    _require_columns(annotations, ANNOTATION_COLUMNS, f"annotations TSV {path}")

    return _canonicalize_annotations(annotations, normalize_labels)


def read_ann(
    path: str | Path,
    pattern: str = "*.ann",
    normalize_labels: bool = True,
) -> pd.DataFrame:
    """Read a single BRAT .ann file or a directory of them."""
    ann_path = Path(path)

    if not ann_path.exists():
        raise FileNotFoundError(f"Path does not exist: {ann_path}")

    ann_files = [ann_path] if ann_path.is_file() else sorted(ann_path.glob(pattern))

    rows: list[dict[str, object]] = []

    for file_path in ann_files:
        rows.extend(_parse_ann_file(file_path))

    return _canonicalize_annotations(
        pd.DataFrame(rows, columns=ANNOTATION_COLUMNS), normalize_labels
    )


def _parse_ann_file(ann_file: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    with ann_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")

            if line.startswith("T"):
                rows.extend(_parse_ann_line(line, ann_file.stem, ann_file, line_number))

    return rows


def _parse_ann_line(
    line: str,
    filename: str,
    ann_file: Path,
    line_number: int,
) -> list[dict[str, object]]:
    parts = line.split("\t")

    if len(parts) < 3:
        raise ValueError(f"Invalid .ann line in {ann_file} at line {line_number}: {line}")

    annotation_info, annotation_text = parts[1], parts[2]
    info_parts = annotation_info.split(" ", maxsplit=1)

    if len(info_parts) != 2:
        raise ValueError(f"Invalid annotation span in {ann_file} at line {line_number}: {line}")

    label, spans_raw = info_parts
    spans: list[tuple[int, int]] = []

    for span_raw in spans_raw.split(";"):
        span_parts = span_raw.split()

        if len(span_parts) != 2:
            raise ValueError(f"Invalid span in {ann_file} at line {line_number}: {line}")

        spans.append((int(span_parts[0]), int(span_parts[1])))

    fragment_texts = _split_discontinuous_text(spans, annotation_text, ann_file, line_number, line)

    return [
        {
            "filename": filename,
            "label": label,
            "start_span": start_span,
            "end_span": end_span,
            "text": fragment_text,
        }
        for (start_span, end_span), fragment_text in zip(spans, fragment_texts)
    ]


def _split_discontinuous_text(
    spans: list[tuple[int, int]],
    annotation_text: str,
    ann_file: Path,
    line_number: int,
    line: str,
) -> list[str]:
    if len(spans) == 1:
        return [annotation_text]

    fragment_lengths = [end - start for start, end in spans]
    expected_length = sum(fragment_lengths) + len(spans) - 1

    if len(annotation_text) != expected_length:
        raise ValueError(
            f"Discontinuous annotation in {ann_file} at line {line_number}: "
            f"spans {spans} imply a text of {expected_length} characters "
            f"(fragments joined by single spaces), but the annotation text "
            f"has {len(annotation_text)}: {line}"
        )

    fragments: list[str] = []
    cursor = 0

    for fragment_length in fragment_lengths:
        fragments.append(annotation_text[cursor:cursor + fragment_length])
        cursor += fragment_length + 1

    return fragments


def _canonicalize_annotations(
    annotations: pd.DataFrame,
    normalize_labels: bool = True,
) -> pd.DataFrame:
    _require_columns(annotations, ANNOTATION_COLUMNS, "annotations")

    canonical = annotations[ANNOTATION_COLUMNS].copy()
    canonical["filename"] = canonical["filename"].astype(str)
    canonical["label"] = canonical["label"].astype(str)
    canonical["start_span"] = canonical["start_span"].astype(int)
    canonical["end_span"] = canonical["end_span"].astype(int)
    canonical["text"] = canonical["text"].astype(str)

    if (canonical["end_span"] <= canonical["start_span"]).any():
        raise ValueError("Invalid spans detected: end_span <= start_span.")

    if normalize_labels:
        canonical = normalize_annotation_labels(canonical)

    return canonical.reset_index(drop=True)


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in {context}: {sorted(missing)}")
