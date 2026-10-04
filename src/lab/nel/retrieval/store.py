"""A gazetteer's encoder embeddings on disk, so an encoder retriever encodes it once."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.core.provenance import read_manifest, write_manifest

EMBEDDINGS_FILENAME = "embeddings.npy"
VOCABULARY_FILENAME = "vocabulary.parquet"
INDEX_MANIFEST_FILENAME = "index_manifest.json"


def gazetteer_fingerprint(vocabulary: pd.DataFrame) -> str:
    """Hash the `term` and `code` columns in row order, whatever the table came from."""
    hasher = hashlib.sha256()

    for term, code in zip(vocabulary["term"], vocabulary["code"]):
        hasher.update(f"{term}\t{code}\n".encode("utf-8"))

    return hasher.hexdigest()


def load_embeddings(
    index_dir: str | Path,
    vocabulary: pd.DataFrame,
    settings: dict[str, Any],
) -> np.ndarray | None:
    """
    Load the embeddings stored in `index_dir`, or None when it holds no index.

    `settings` are what the vectors depend on (method, model, encoding options).
    An index built for other settings or another gazetteer is refused, as is a
    directory that holds files but no manifest.
    """
    index_dir = Path(index_dir)
    manifest_path = index_dir / INDEX_MANIFEST_FILENAME

    if not manifest_path.exists():
        if index_dir.is_file() or (index_dir.is_dir() and any(index_dir.iterdir())):
            raise ValueError(
                f"{index_dir} holds files but no {INDEX_MANIFEST_FILENAME}: either an interrupted "
                "build or not a gazetteer index. Delete it or pass another index_dir."
            )

        return None

    stored = read_manifest(manifest_path)
    expected = {**settings, **_vocabulary_fields(vocabulary)}
    differences = [
        f"{key}: stored {stored.get(key)!r}, requested {expected.get(key)!r}"
        for key in sorted((set(stored) - {"dim"}) | set(expected))
        if stored.get(key) != expected.get(key)
    ]

    if differences:
        raise ValueError(
            f"The index in {index_dir} was built for a different run ({'; '.join(differences)}). "
            "Delete it or pass another index_dir."
        )

    embeddings = np.load(index_dir / EMBEDDINGS_FILENAME)

    if embeddings.shape != (stored["n_terms"], stored["dim"]):
        raise ValueError(
            f"{index_dir / EMBEDDINGS_FILENAME} has shape {embeddings.shape}, "
            f"its manifest says ({stored['n_terms']}, {stored['dim']})."
        )

    return embeddings


def save_embeddings(
    index_dir: str | Path,
    embeddings: np.ndarray,
    vocabulary: pd.DataFrame,
    settings: dict[str, Any],
) -> Path:
    """Write the embeddings, their vocabulary and the manifest, the manifest last."""
    index_dir = Path(index_dir)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(embeddings) != len(vocabulary):
        raise ValueError(f"{len(embeddings)} embeddings for {len(vocabulary)} vocabulary rows.")

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / EMBEDDINGS_FILENAME, embeddings)
    vocabulary[["term", "code"]].to_parquet(index_dir / VOCABULARY_FILENAME, index=False)

    manifest = {**settings, **_vocabulary_fields(vocabulary), "dim": int(embeddings.shape[1])}
    write_manifest(manifest, index_dir / INDEX_MANIFEST_FILENAME)

    return index_dir


def _vocabulary_fields(vocabulary: pd.DataFrame) -> dict[str, Any]:
    return {"gazetteer_fingerprint": gazetteer_fingerprint(vocabulary), "n_terms": int(len(vocabulary))}
