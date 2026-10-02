"""High-level API for task-independent document selection.

This module owns orchestration and selection-specific artifacts.  Canonical corpus
validation, reading, writing, hashing and manifests are delegated to :mod:`lab.core`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from lab.core import DOCUMENT_COLUMNS, file_sha256, read_corpus, validate_corpus, write_corpus, write_manifest
from lab.core.io import DEFAULT_PARQUET_COMPRESSION
from lab.selection.methods import MethodResult, get_method, method_info
from lab.selection.representations import RepresentationOutput, get_representation


@dataclass(frozen=True)
class SelectionRun:
    output_dir: Path
    selected_path: Path
    ranking_path: Path
    history_path: Path
    manifest_path: Path
    selected_doc_ids: tuple[str, ...]
    round_number: int


def validate_selection_input(input_path: str | Path) -> dict[str, Any]:
    """Validate a canonical corpus and return a compact selection-oriented summary."""
    corpus = read_corpus(input_path)
    annotated = int((corpus["n_entities"] > 0).sum())
    return {
        "path": str(Path(input_path)),
        "n_documents": int(len(corpus)),
        "n_annotated_documents": annotated,
        "n_unannotated_documents": int(len(corpus) - annotated),
        "n_entities": int(corpus["n_entities"].sum()),
        "columns": list(corpus.columns),
        "sha256": file_sha256(input_path),
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        path,
        engine="pyarrow",
        compression=DEFAULT_PARQUET_COMPRESSION,
        index=False,
    )
    return path


def _load_previous_selection(
    sources: Sequence[str | Path] | None,
    corpus_doc_ids: set[str],
) -> tuple[list[str], pd.DataFrame, list[dict[str, str]]]:
    if not sources:
        empty = pd.DataFrame(columns=["doc_id", "round", "method", "rank", "score", "source"])
        return [], empty, []

    ordered_ids: list[str] = []
    history_parts: list[pd.DataFrame] = []
    provenance: list[dict[str, str]] = []

    for source in sources:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Previous selection file not found: {path}")
        frame = pd.read_parquet(path)
        if "doc_id" not in frame.columns:
            raise ValueError(f"{path}: previous selection must contain a doc_id column.")

        if set(DOCUMENT_COLUMNS).issubset(frame.columns):
            canonical = validate_corpus(frame)
            ids = canonical["doc_id"].astype(str).tolist()
            part = pd.DataFrame(
                {
                    "doc_id": ids,
                    "round": 0,
                    "method": "previous",
                    "rank": range(1, len(ids) + 1),
                    "score": np.nan,
                    "source": str(path),
                }
            )
        else:
            if "selected" in frame.columns:
                frame = frame.loc[frame["selected"].fillna(False).astype(bool)].copy()
            ids = frame["doc_id"].astype(str).tolist()
            part = pd.DataFrame({"doc_id": ids})
            part["round"] = frame["round"].to_numpy() if "round" in frame.columns else 0
            part["method"] = frame["method"].astype(str).to_numpy() if "method" in frame.columns else "previous"
            part["rank"] = frame["rank"].to_numpy() if "rank" in frame.columns else np.arange(1, len(frame) + 1)
            part["score"] = frame["score"].to_numpy() if "score" in frame.columns else np.nan
            part["source"] = str(path)

        unknown = sorted(set(ids) - corpus_doc_ids)
        if unknown:
            raise ValueError(
                f"{path}: {len(unknown)} selected doc_id value(s) are absent from the input corpus: {unknown[:10]}"
            )
        for doc_id in ids:
            if doc_id not in ordered_ids:
                ordered_ids.append(doc_id)
        history_parts.append(part)
        provenance.append({"path": str(path), "sha256": file_sha256(path)})

    history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    history = history.drop_duplicates("doc_id", keep="first").reset_index(drop=True)
    return ordered_ids, history, provenance


def _read_representations(path: str | Path) -> RepresentationOutput:
    frame = pd.read_parquet(path)
    missing = {"doc_id", "embedding"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: representation parquet is missing columns {sorted(missing)}.")
    ids = frame["doc_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: doc_id must be unique in a representation parquet.")
    vectors = [np.asarray(value, dtype=np.float32).reshape(-1) for value in frame["embedding"]]
    if not vectors:
        raise ValueError(f"{path}: representation parquet is empty.")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"{path}: embeddings do not all have the same dimension.")
    return RepresentationOutput(ids, np.vstack(vectors), {"source": str(path), "sha256": file_sha256(path)})


def _align_matrix(output: RepresentationOutput, ids: Sequence[str], name: str) -> np.ndarray:
    lookup = {doc_id: index for index, doc_id in enumerate(output.doc_ids)}
    missing = [doc_id for doc_id in ids if doc_id not in lookup]
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} document(s): {missing[:10]}")
    return output.matrix[[lookup[doc_id] for doc_id in ids]]


def _read_scores(path: str | Path, candidate_ids: Sequence[str]) -> np.ndarray:
    frame = pd.read_parquet(path)
    missing = {"doc_id", "score"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: score parquet is missing columns {sorted(missing)}.")
    if frame["doc_id"].astype(str).duplicated().any():
        raise ValueError(f"{path}: score parquet contains duplicate doc_id values.")
    lookup = dict(zip(frame["doc_id"].astype(str), pd.to_numeric(frame["score"], errors="raise")))
    absent = [doc_id for doc_id in candidate_ids if doc_id not in lookup]
    if absent:
        raise ValueError(f"{path}: scores are missing {len(absent)} candidate document(s): {absent[:10]}")
    return np.asarray([lookup[doc_id] for doc_id in candidate_ids], dtype=np.float64)


def _read_probabilities(path: str | Path, candidate_ids: Sequence[str]) -> np.ndarray:
    frame = pd.read_parquet(path)
    missing = {"doc_id", "probabilities"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: probability parquet is missing columns {sorted(missing)}.")
    if frame["doc_id"].astype(str).duplicated().any():
        raise ValueError(f"{path}: probability parquet contains duplicate doc_id values.")
    lookup = {
        str(row.doc_id): np.asarray(row.probabilities, dtype=np.float64).reshape(-1)
        for row in frame.itertuples(index=False)
    }
    absent = [doc_id for doc_id in candidate_ids if doc_id not in lookup]
    if absent:
        raise ValueError(
            f"{path}: probabilities are missing {len(absent)} candidate document(s): {absent[:10]}"
        )
    values = [lookup[doc_id] for doc_id in candidate_ids]
    dimensions = {len(value) for value in values}
    if len(dimensions) != 1:
        raise ValueError(f"{path}: probability vectors have inconsistent dimensions.")
    return np.vstack(values)


def build_representations(
    input_path: str | Path,
    output_path: str | Path,
    representation: str,
    *,
    random_state: int = 13,
    representation_params: dict[str, Any] | None = None,
) -> Path:
    """Precompute one document vector per canonical-corpus row and save it as Parquet."""
    corpus = read_corpus(input_path)
    encoder = get_representation(representation)
    output = encoder.encode(
        corpus["doc_id"].astype(str).tolist(),
        corpus["text"].astype(str).tolist(),
        random_state=random_state,
        **(representation_params or {}),
    )
    if len(output.doc_ids) != len(corpus) or output.matrix.shape[0] != len(corpus):
        raise ValueError("Representation method did not return exactly one vector per input document.")
    if output.matrix.ndim != 2 or not np.isfinite(output.matrix).all():
        raise ValueError("Representation matrix must be finite and two-dimensional.")

    path = Path(output_path)
    frame = pd.DataFrame(
        {
            "doc_id": output.doc_ids,
            "embedding": [vector.astype(np.float32).tolist() for vector in output.matrix],
        }
    )
    _write_parquet(frame, path)
    write_manifest(
        {
            "artifact": "selection_representations",
            "input_path": str(input_path),
            "input_sha256": file_sha256(input_path),
            "representation": representation,
            "representation_params": representation_params or {},
            "random_state": random_state,
            "n_documents": len(frame),
            "embedding_dimension": int(output.matrix.shape[1]),
            "metadata": output.metadata,
            "output_path": str(path),
            "output_sha256": file_sha256(path),
        },
        path.with_suffix(".manifest.json"),
    )
    return path


def _resolve_representations(
    corpus: pd.DataFrame,
    candidate_ids: Sequence[str],
    previous_ids: Sequence[str],
    *,
    representation: str | None,
    representations_path: str | Path | None,
    method_name: str,
    random_state: int,
    representation_params: dict[str, Any] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    info = method_info(method_name)
    if not info.requires_representation:
        return None, None, {}
    if representations_path is not None and representation is not None:
        raise ValueError("Give either representation or representations_path, not both.")

    if representations_path is not None:
        output = _read_representations(representations_path)
    else:
        name = representation or info.default_representation
        if name is None:
            raise ValueError(
                f"Method {method_name!r} requires representations. Pass representation=... "
                "or representations_path=...."
            )
        encoder = get_representation(name)
        output = encoder.encode(
            corpus["doc_id"].astype(str).tolist(),
            corpus["text"].astype(str).tolist(),
            random_state=random_state,
            **(representation_params or {}),
        )
        output.metadata = {"name": name, **output.metadata}

    candidate_matrix = _align_matrix(output, candidate_ids, "representations")
    selected_matrix = (
        _align_matrix(output, previous_ids, "representations")
        if previous_ids
        else np.empty((0, candidate_matrix.shape[1]), dtype=np.float32)
    )
    return candidate_matrix, selected_matrix, output.metadata


def _validate_method_result(result: MethodResult, n: int, n_candidates: int) -> None:
    selected = list(map(int, result.selected_indices))
    if len(selected) != n:
        raise ValueError(f"Selection method returned {len(selected)} documents; expected exactly {n}.")
    if len(selected) != len(set(selected)):
        raise ValueError("Selection method returned duplicate candidate indices.")
    if any(index < 0 or index >= n_candidates for index in selected):
        raise ValueError("Selection method returned an index outside the candidate pool.")
    if result.scores is not None and len(result.scores) != n_candidates:
        raise ValueError("Selection method scores must contain one value per candidate.")
    for name, values in result.extras.items():
        if len(values) != n_candidates:
            raise ValueError(f"Selection extra column {name!r} must contain one value per candidate.")


def _ranking_frame(candidate_ids: Sequence[str], method: str, result: MethodResult) -> pd.DataFrame:
    frame = pd.DataFrame({"doc_id": list(candidate_ids)})
    frame["method"] = method
    frame["selected"] = False
    frame["rank"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    frame["score"] = np.nan if result.scores is None else np.asarray(result.scores, dtype=np.float64)
    for name, values in result.extras.items():
        frame[name] = list(values)
    for rank, index in enumerate(result.selected_indices, start=1):
        frame.at[int(index), "selected"] = True
        frame.at[int(index), "rank"] = rank
    frame = frame.sort_values(["selected", "rank", "doc_id"], ascending=[False, True, True], na_position="last")
    return frame.reset_index(drop=True)


def select_documents(
    input_path: str | Path,
    output_dir: str | Path,
    method: str,
    n_select: int,
    *,
    selected: Sequence[str | Path] | None = None,
    representation: str | None = None,
    representations_path: str | Path | None = None,
    scores_path: str | Path | None = None,
    probabilities_path: str | Path | None = None,
    random_state: int = 13,
    round_number: int | None = None,
    method_params: dict[str, Any] | None = None,
    representation_params: dict[str, Any] | None = None,
) -> SelectionRun:
    """Select the next batch of documents from a canonical lab corpus."""
    corpus = read_corpus(input_path)
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    previous_ids, previous_history, previous_provenance = _load_previous_selection(
        selected, set(corpus_ids)
    )
    previous_set = set(previous_ids)
    candidate_ids = [doc_id for doc_id in corpus_ids if doc_id not in previous_set]
    if n_select <= 0:
        raise ValueError("n_select must be greater than zero.")
    if n_select > len(candidate_ids):
        raise ValueError(
            f"Requested {n_select} documents but only {len(candidate_ids)} remain after exclusions."
        )

    info = method_info(method)
    candidate_rep, selected_rep, representation_metadata = _resolve_representations(
        corpus,
        candidate_ids,
        previous_ids,
        representation=representation,
        representations_path=representations_path,
        method_name=method,
        random_state=random_state,
        representation_params=representation_params,
    )
    scores = _read_scores(scores_path, candidate_ids) if scores_path is not None else None
    probabilities = (
        _read_probabilities(probabilities_path, candidate_ids)
        if probabilities_path is not None
        else None
    )
    if info.requires_scores and scores is None:
        raise ValueError(f"Method {method!r} requires scores_path (doc_id | score Parquet).")
    if info.requires_probabilities and probabilities is None:
        raise ValueError(
            f"Method {method!r} requires probabilities_path (doc_id | probabilities Parquet)."
        )

    selector = get_method(method)
    result = selector.select(
        candidate_ids,
        int(n_select),
        representations=candidate_rep,
        selected_representations=selected_rep,
        scores=scores,
        probabilities=probabilities,
        random_state=random_state,
        **(method_params or {}),
    )
    _validate_method_result(result, int(n_select), len(candidate_ids))
    selected_doc_ids = [candidate_ids[index] for index in result.selected_indices]

    if round_number is None:
        numeric_round = pd.to_numeric(previous_history.get("round", pd.Series(dtype=float)), errors="coerce")
        maximum = int(numeric_round.dropna().max()) if not numeric_round.dropna().empty else 0
        round_number = maximum + 1 if previous_ids else 1
    if round_number <= 0:
        raise ValueError("round_number must be greater than zero.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_path = output_root / "selected.parquet"
    ranking_path = output_root / "ranking.parquet"
    history_path = output_root / "history.parquet"
    manifest_path = output_root / "manifest.json"

    indexed = corpus.set_index(corpus["doc_id"].astype(str), drop=False)
    selected_corpus = indexed.loc[selected_doc_ids, DOCUMENT_COLUMNS].reset_index(drop=True)
    write_corpus(selected_corpus, selected_path)

    ranking = _ranking_frame(candidate_ids, method, result)
    ranking.insert(1, "round", int(round_number))
    _write_parquet(ranking, ranking_path)

    score_lookup = dict(zip(ranking["doc_id"].astype(str), ranking["score"]))
    new_history = pd.DataFrame(
        {
            "doc_id": selected_doc_ids,
            "round": int(round_number),
            "method": method,
            "rank": range(1, len(selected_doc_ids) + 1),
            "score": [score_lookup.get(doc_id, np.nan) for doc_id in selected_doc_ids],
            "source": str(selected_path),
        }
    )
    history = pd.concat([previous_history, new_history], ignore_index=True)
    history = history.drop_duplicates("doc_id", keep="first").reset_index(drop=True)
    _write_parquet(history, history_path)

    manifest: dict[str, Any] = {
        "artifact": "document_selection",
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "method": method,
        "method_family": info.family,
        "method_params": method_params or {},
        "method_reference": info.reference,
        "method_implementation_note": info.implementation_note,
        "representation": representation,
        "representations_path": str(representations_path) if representations_path is not None else None,
        "representation_params": representation_params or {},
        "representation_metadata": representation_metadata,
        "scores_path": str(scores_path) if scores_path is not None else None,
        "probabilities_path": str(probabilities_path) if probabilities_path is not None else None,
        "random_state": int(random_state),
        "round": int(round_number),
        "n_input_documents": int(len(corpus)),
        "n_previous_selected": int(len(previous_ids)),
        "n_candidates": int(len(candidate_ids)),
        "n_requested": int(n_select),
        "n_selected": int(len(selected_doc_ids)),
        "n_remaining_after": int(len(candidate_ids) - len(selected_doc_ids)),
        "previous_selection_sources": previous_provenance,
        "diagnostics": result.diagnostics,
        "outputs": {
            "selected": str(selected_path),
            "ranking": str(ranking_path),
            "history": str(history_path),
        },
    }
    write_manifest(manifest, manifest_path)
    return SelectionRun(
        output_root,
        selected_path,
        ranking_path,
        history_path,
        manifest_path,
        tuple(selected_doc_ids),
        int(round_number),
    )


def selection_history(path: str | Path) -> pd.DataFrame:
    """Read and validate a selection history artifact."""
    frame = pd.read_parquet(path)
    required = {"doc_id", "round", "method", "rank", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: selection history is missing columns {sorted(missing)}.")
    if frame["doc_id"].astype(str).duplicated().any():
        raise ValueError(f"{path}: a selection history cannot contain duplicate doc_id values.")
    return frame.sort_values(["round", "rank", "doc_id"], na_position="last").reset_index(drop=True)


def compare_methods(
    input_path: str | Path,
    output_dir: str | Path,
    methods: Sequence[str],
    n_select: int,
    *,
    selected: Sequence[str | Path] | None = None,
    representation: str | None = None,
    representations_path: str | Path | None = None,
    scores_path: str | Path | None = None,
    probabilities_path: str | Path | None = None,
    random_state: int = 13,
    method_params: dict[str, Any] | None = None,
    representation_params: dict[str, Any] | None = None,
) -> Path:
    """Run several methods on the same corpus and write pairwise selection overlap."""
    if len(methods) < 2:
        raise ValueError("compare_methods requires at least two methods.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selections: dict[str, set[str]] = {}
    for name in methods:
        run = select_documents(
            input_path,
            root / name,
            name,
            n_select,
            selected=selected,
            representation=representation,
            representations_path=representations_path,
            scores_path=scores_path,
            probabilities_path=probabilities_path,
            random_state=random_state,
            method_params=method_params,
            representation_params=representation_params,
        )
        selections[name] = set(run.selected_doc_ids)

    rows: list[dict[str, Any]] = []
    for left in methods:
        for right in methods:
            intersection = selections[left] & selections[right]
            union = selections[left] | selections[right]
            rows.append(
                {
                    "method_a": left,
                    "method_b": right,
                    "intersection": len(intersection),
                    "union": len(union),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                }
            )
    output = root / "overlap.parquet"
    _write_parquet(pd.DataFrame(rows), output)
    write_manifest(
        {
            "artifact": "selection_method_comparison",
            "input_path": str(input_path),
            "input_sha256": file_sha256(input_path),
            "methods": list(methods),
            "n_select": int(n_select),
            "random_state": int(random_state),
            "overlap_path": str(output),
        },
        root / "manifest.json",
    )
    return output
