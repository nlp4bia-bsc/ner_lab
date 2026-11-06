"""Document-level evidence for separation of training and ensemble calibration."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.provenance import file_sha256, read_manifest


def describe_calibration(
    members: list[dict[str, Any]],
    corpus: pd.DataFrame,
    tuned: bool,
) -> dict[str, Any]:
    """Reject known document overlap when tuning; record missing provenance explicitly."""
    documents = dict(zip(corpus["doc_id"].astype(str), corpus["text"].map(_text_digest)))
    records = []
    for member in members:
        summary_path = Path(member["model_dir"]).parent / "training_summary.json"
        sources = _training_documents(summary_path)
        ids, texts, hashes, complete = sources
        overlap = sorted(
            doc_id for doc_id, text in documents.items() if doc_id in ids or text in texts
        )
        status = "overlap" if overlap else "disjoint" if complete else "unknown"
        if tuned and overlap:
            raise ValueError(
                f"Calibration overlaps training for {member['model_dir']}: {overlap[:10]}. "
                "Use independent validation documents."
            )
        if status == "unknown":
            warnings.warn(
                f"Training/calibration separation cannot be verified for {member['model_dir']}.",
                stacklevel=2,
            )
        records.append(
            {
                "model_dir": member["model_dir"],
                "status": status,
                "overlap_count": len(overlap),
                "training_sha256": hashes,
            }
        )
    fingerprint = hashlib.sha256(json.dumps(sorted(documents.items())).encode()).hexdigest()

    return {
        "n_documents": len(documents),
        "documents_sha256": fingerprint,
        "separation_basis": "document_ids_and_exact_text_hashes",
        "members": records,
        "independent": all(item["status"] == "disjoint" for item in records),
    }


def _training_documents(summary_path: Path) -> tuple[set[str], set[str], dict[str, str], bool]:
    ids, texts, hashes = set(), set(), {}
    try:
        metadata = read_manifest(summary_path).get("metadata", {})
        paths = metadata.get("train_parquet", [])
        paths = [paths] if isinstance(paths, str) else list(paths or [])
    except (OSError, ValueError, TypeError, AttributeError):
        return ids, texts, hashes, False
    complete = bool(paths)
    for path in paths:
        try:
            training = pd.read_parquet(path, columns=["doc_id", "text"])
            ids.update(training["doc_id"].astype(str))
            texts.update(training["text"].map(_text_digest))
            hashes[str(path)] = file_sha256(path)
        except (OSError, ValueError, KeyError, TypeError):
            complete = False

    return ids, texts, hashes, complete


def _text_digest(text: str) -> str:
    return hashlib.sha256(str(text).encode()).hexdigest()
