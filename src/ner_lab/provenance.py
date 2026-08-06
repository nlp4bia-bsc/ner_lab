"""Hashing and manifest helpers used to record how an artifact was produced."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's contents, read in chunks so large parquets stay off the heap."""
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write a manifest as indented JSON, creating parent directories."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return manifest_path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a JSON manifest."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
