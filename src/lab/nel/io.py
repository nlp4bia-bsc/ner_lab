"""Reading NEL's inputs into records, and tables to and from TSV, JSONL or parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lab.nel.brat import read_brat_ann
from lab.nel.schemas import Concept, GazetteerEntry, HierarchyEdge, Mention

MENTION_COLUMNS = ["filename", "label", "start_span", "end_span", "text", "code"]
MENTION_FLAG_COLUMNS = {"is_abbreviation", "is_composite", "is_compossite", "need_context"}
GAZETTEER_FIELDS = {"term", "code", "label", "semantic_tag", "sem_tag"}

TSV_SUFFIXES = {".tsv", ".txt"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}


def read_mentions_tsv(path: str | Path, encoding: str = "utf-8") -> list[Mention]:
    """
    Read a mention TSV with a gold code on every row.

    The `is_abbreviation`, `is_composite` (or the misspelt `is_compossite`) and
    `need_context` flags are parsed with `parse_bool`; any other column goes into
    the mention's `metadata`. A row with an empty code raises.
    """
    mentions_df = pd.read_csv(path, sep="\t", encoding=encoding)
    require_columns(mentions_df.columns.tolist(), MENTION_COLUMNS, "annotations")
    known_columns = {*MENTION_COLUMNS, *MENTION_FLAG_COLUMNS}
    mentions: list[Mention] = []

    for row in mentions_df.to_dict("records"):
        if pd.isna(row["code"]) or not str(row["code"]).strip():
            raise ValueError("Empty code in annotations row")

        mentions.append(
            Mention(
                filename=str(row["filename"]),
                label=None if pd.isna(row.get("label")) else str(row.get("label")),
                start_span=int(row["start_span"]) if pd.notna(row["start_span"]) else None,
                end_span=int(row["end_span"]) if pd.notna(row["end_span"]) else None,
                text=str(row["text"]),
                code=str(row["code"]),
                is_abbreviation=parse_bool(row.get("is_abbreviation")),
                is_composite=parse_bool(row.get("is_composite", row.get("is_compossite"))),
                need_context=parse_bool(row.get("need_context")),
                metadata={key: value for key, value in row.items() if key not in known_columns},
            )
        )

    return mentions


def read_gazetteer_entries(path: str | Path, encoding: str = "utf-8") -> list[GazetteerEntry]:
    """Read a gazetteer TSV with at least `term` and `code` columns into entries."""
    gazetteer_df = pd.read_csv(path, sep="\t", encoding=encoding)
    require_columns(gazetteer_df.columns.tolist(), ["term", "code"], "gazetteer")

    return [
        GazetteerEntry(
            term=str(row["term"]),
            code=str(row["code"]),
            label=None if pd.isna(row.get("label")) else row.get("label"),
            semantic_tag=row.get("semantic_tag", row.get("sem_tag")),
            metadata={key: value for key, value in row.items() if key not in GAZETTEER_FIELDS},
        )
        for row in gazetteer_df.to_dict("records")
    ]


def read_concepts_tsv(path: str | Path) -> list[Concept]:
    """Read ontology concepts from a TSV; `aliases` is a `|`-separated list."""
    concepts_df = pd.read_csv(path, sep="\t")
    require_columns(concepts_df.columns.tolist(), ["code", "term"], "concepts")

    return [
        Concept(
            code=str(row["code"]),
            term=str(row["term"]),
            label=None if pd.isna(row.get("label")) else row.get("label"),
            semantic_tag=row.get("semantic_tag", row.get("sem_tag")),
            aliases=_aliases(row.get("aliases")),
        )
        for row in concepts_df.to_dict("records")
    ]


def read_hierarchy_tsv(path: str | Path) -> list[HierarchyEdge]:
    """Read directed parent-to-child ontology edges from a TSV."""
    hierarchy_df = pd.read_csv(path, sep="\t")
    require_columns(hierarchy_df.columns.tolist(), ["parent_code", "child_code"], "hierarchy")

    return [
        HierarchyEdge(
            parent_code=str(row["parent_code"]),
            child_code=str(row["child_code"]),
            relation=None if pd.isna(row.get("relation")) else row.get("relation"),
        )
        for row in hierarchy_df.to_dict("records")
    ]


def read_text_dir(path: str | Path, encoding: str = "utf-8") -> dict[str, str]:
    """Every `.txt` file in a directory, keyed by filename stem."""
    return {
        text_path.stem: text_path.read_text(encoding=encoding)
        for text_path in sorted(Path(path).glob("*.txt"))
    }


def read_brat_dir(path: str | Path) -> list[Mention]:
    """The mentions of every BRAT `.ann` file in a directory, file by file in name order."""
    return [
        mention
        for annotation_path in sorted(Path(path).glob("*.ann"))
        for mention in read_brat_ann(annotation_path)
    ]


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a TSV, JSONL or parquet table, chosen by the file extension."""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix in TSV_SUFFIXES:
        return pd.read_csv(input_path, sep="\t", **kwargs)

    if suffix in JSONL_SUFFIXES:
        return pd.read_json(input_path, lines=True, **kwargs)

    if suffix == ".parquet":
        return pd.read_parquet(input_path, **kwargs)

    raise ValueError(f"Unsupported input format: {suffix}")


def write_table(
    dataframe: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Write a table as TSV, JSONL or parquet, chosen by the file extension.

    An existing file raises unless `overwrite` is set. In a TSV, list, tuple and
    dict cells are written as JSON.
    """
    output_path = Path(path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in TSV_SUFFIXES:
        _with_json_cells(dataframe).to_csv(output_path, sep="\t", index=False, **kwargs)
    elif suffix in JSONL_SUFFIXES:
        dataframe.to_json(output_path, orient="records", lines=True, force_ascii=False, **kwargs)
    elif suffix == ".parquet":
        dataframe.to_parquet(output_path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    return output_path


def require_columns(columns: list[str], required: list[str], name: str) -> None:
    """Raise naming the columns of `required` that `columns` lacks."""
    missing = [column for column in required if column not in columns]

    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def parse_bool(value: object) -> bool | None:
    """`true`/`1`/`yes` as True, `false`/`0`/`no` as False, anything else as None."""
    if value is None:
        return None

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    return None


def _aliases(value: object) -> list[str]:
    return str(value).split("|") if pd.notna(value) and str(value) else []


def _is_nested(value: object) -> bool:
    return isinstance(value, (list, dict, tuple))


def _with_json_cells(dataframe: pd.DataFrame) -> pd.DataFrame:
    serializable = dataframe.copy()

    for column in serializable.columns:
        if serializable[column].map(_is_nested).any():
            serializable[column] = serializable[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if _is_nested(value) else value
            )

    return serializable
