"""Checkpoint-bound persistence for explicitly selected post-hoc binary decoders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab.core.provenance import file_sha256
from lab.ner.models.binary_decoding import BinaryDecoding


def model_fingerprint(model_dir: str | Path) -> str:
    """Bind a decoder to original weights, encoder, catalogue and tokenizer files."""
    root = Path(model_dir)
    paths = [
        root / "encoding.json",
        *root.glob("*.safetensors"),
        *root.glob("pytorch_model*.bin"),
        *root.glob("tokenizer*"),
        *root.glob("vocab.*"),
        *root.glob("special_tokens_map.json"),
    ]
    if not any(path.suffix == ".safetensors" or path.suffix == ".bin" for path in paths):
        raise ValueError("Decoder binding requires a checkpoint with local weights.")
    records = [(path.name, file_sha256(path)) for path in sorted(set(paths)) if path.is_file()]

    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def read_binary_decoding(
    path: str | Path, model_dir: str | Path, entity_labels: list[str]
) -> BinaryDecoding:
    """Read fixed choices, rejecting use with a changed checkpoint or entity mapping."""
    recorded = json.loads(Path(path).read_text())
    if (
        recorded.get("format_version") != 1
        or recorded.get("entity_labels") != entity_labels
        or recorded.get("model_fingerprint") != model_fingerprint(model_dir)
    ):
        raise ValueError(
            "Post-hoc decoder disagrees with checkpoint weights or entity catalogue; refit it."
        )

    return BinaryDecoding(**recorded["decoding"])
