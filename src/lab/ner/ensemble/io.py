"""A fitted ensemble on disk: `ensemble.json`, and the saved models it points to."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lab.core.provenance import file_sha256, read_manifest, write_manifest
from lab.ner.encoding.encoder import BUILTIN_STRATEGIES
from lab.ner.training.trainer import read_model_encoding

ENSEMBLE_FILENAME = "ensemble.json"
ENSEMBLE_FORMAT_VERSION = 1
WEIGHT_FILE_PATTERNS = ("*.safetensors", "pytorch_model*.bin")


def describe_member(model_dir: str | Path) -> dict[str, Any]:
    """
    One saved model as an ensemble member: where it is, what it is, and its weights' digest.

    A model trained with a callable window strategy is refused: its `encoding.json`
    records the strategy by name only, and an ensemble has no way to hand a callable
    back to one member in particular.
    """
    model_dir = Path(model_dir).resolve()
    description = read_model_encoding(model_dir)
    strategy = description["encoding"].get("strategy", "greedy")

    if strategy not in BUILTIN_STRATEGIES:
        raise ValueError(
            f"Member {model_dir} was trained with the callable window strategy {strategy!r}, "
            f"which cannot be rebuilt from its encoding.json; an ensemble member must use one "
            f"of {BUILTIN_STRATEGIES}."
        )

    return {
        "model_dir": str(model_dir),
        "weights_sha256": weights_sha256(model_dir),
        "model": description["model"],
        "encoding": description["encoding"],
    }


def shared_target_label(members: Sequence[dict[str, Any]]) -> str:
    """The entity type every member predicts, raising when they disagree."""
    labels = {member["model_dir"]: member["encoding"]["target_label"] for member in members}

    if len(set(labels.values())) > 1:
        raise ValueError(
            f"Ensemble members must predict the same target_label; these disagree: {labels}."
        )

    return next(iter(labels.values()))


def weights_sha256(model_dir: str | Path) -> str:
    """
    A digest of a saved model's weight files, taken together in name order.

    Covers a single `model.safetensors`, a sharded checkpoint and a bare
    `pytorch_model.bin` alike, so retraining a model in place changes the digest
    whichever way it was saved.
    """
    model_dir = Path(model_dir)
    weight_paths = sorted(
        {path for pattern in WEIGHT_FILE_PATTERNS for path in model_dir.glob(pattern)}
    )

    if not weight_paths:
        raise FileNotFoundError(
            f"No weight files in {model_dir}; expected one of {WEIGHT_FILE_PATTERNS}."
        )

    hasher = hashlib.sha256()

    for path in weight_paths:
        hasher.update(f"{path.name}:{file_sha256(path)}\n".encode())

    return hasher.hexdigest()


def write_ensemble(ensemble: dict[str, Any], path: str | Path) -> Path:
    """Write a fitted ensemble, the one file `predict_ensemble` reads back."""
    return write_manifest(ensemble, path)


def read_ensemble(ensemble_dir: str | Path) -> dict[str, Any]:
    """Read the `ensemble.json` that `fit_ensemble` wrote, refusing a format it does not know."""
    path = Path(ensemble_dir) / ENSEMBLE_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"No {ENSEMBLE_FILENAME} in {ensemble_dir} — expected the output directory of "
            "lab.ner.fit_ensemble."
        )

    ensemble = read_manifest(path)
    version = ensemble.get("format_version")

    if version != ENSEMBLE_FORMAT_VERSION:
        raise ValueError(
            f"{path} has format_version {version!r}; this library reads "
            f"{ENSEMBLE_FORMAT_VERSION}. Refit the ensemble with fit_ensemble."
        )

    return ensemble


def check_member_weights(ensemble: dict[str, Any]) -> None:
    """Raise, naming the member, when a member's weights changed since the ensemble was fitted."""
    for member in ensemble["members"]:
        current = weights_sha256(member["model_dir"])

        if current != member["weights_sha256"]:
            raise ValueError(
                f"The weights in {member['model_dir']} changed since this ensemble was "
                f"fitted (sha256 {member['weights_sha256'][:12]}… then, {current[:12]}… now). "
                "Refit the ensemble with fit_ensemble."
            )
