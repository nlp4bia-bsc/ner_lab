"""A fitted ensemble on disk: `ensemble.json`, and the saved models it points to."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lab.ner.ensemble.entities import VotingParameters

from lab.core.provenance import file_sha256, read_manifest, write_manifest
from lab.ner.encoding.encoder import BUILTIN_STRATEGIES
from lab.ner.training.trainer import read_model_encoding

ENSEMBLE_FILENAME = "ensemble.json"
ENSEMBLE_FORMAT_VERSION = 2
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
        "settings_sha256": settings_sha256(model_dir),
        "entity_labels": member_labels(description),
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

    if version not in (1, ENSEMBLE_FORMAT_VERSION):
        raise ValueError(
            f"{path} has format_version {version!r}; this library reads "
            f"{ENSEMBLE_FORMAT_VERSION}. Refit the ensemble with fit_ensemble."
        )

    if version == 2:
        _validate_ensemble(ensemble)

    return ensemble


def check_member_weights(ensemble: dict[str, Any]) -> None:
    """Raise, naming the member, when a member's weights changed since the ensemble was fitted."""
    for member in ensemble["members"]:
        current = weights_sha256(member["model_dir"])
        if ensemble["format_version"] == 1:
            labels = member_labels(read_model_encoding(member["model_dir"]))
            if labels != [ensemble["target_label"]]:
                raise ValueError(
                    f"Member catalogue in {member['model_dir']} changed since fitting."
                )

        if member.get("settings_sha256") is not None:
            description = read_model_encoding(member["model_dir"])
            if (
                member_labels(description) != member["entity_labels"]
                or description["model"].get("architecture", "linear") != member["architecture"]
            ):
                raise ValueError(f"Member metadata in {member['model_dir']} changed since fitting.")
            if settings_sha256(member["model_dir"]) != member["settings_sha256"]:
                raise ValueError(
                    f"Configuration or tokenizer in {member['model_dir']} changed since fitting."
                )
        if current != member["weights_sha256"]:
            raise ValueError(
                f"The weights in {member['model_dir']} changed since this ensemble was "
                f"fitted (sha256 {member['weights_sha256'][:12]}… then, {current[:12]}… now). "
                "Refit the ensemble with fit_ensemble."
            )


def member_labels(description: dict[str, Any]) -> list[str]:
    """Read supported types from checkpoint metadata, never from evaluation gold."""
    from lab.ner.encoding.tagging import validate_entity_labels

    target = description["encoding"]["target_label"]
    labels = description["encoding"].get("entity_labels") if target is None else [target]

    return validate_entity_labels(labels or [])


def settings_sha256(model_dir: str | Path) -> str:
    """Bind encoder/model metadata and tokenizer files to a fitted ensemble."""
    root = Path(model_dir)
    paths = sorted(
        {path for pattern in ("*.json", "*.model", "*.txt") for path in root.glob(pattern)}
    )
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(f"{path.name}:{file_sha256(path)}\n".encode())

    return hasher.hexdigest()


def parameters_from_ensemble(ensemble: dict[str, Any]) -> VotingParameters:
    """Restore explicit scoped values without recalibrating on prediction data."""
    from lab.ner.ensemble.entities import VotingParameters

    return VotingParameters(
        [member["weight"] for member in ensemble["members"]],
        ensemble["threshold"],
        ensemble["entity_weights"],
        ensemble["entity_thresholds"],
    )


def _validate_ensemble(ensemble: dict[str, Any]) -> None:
    import numpy as np

    from lab.ner.encoding.tagging import validate_entity_labels
    from lab.ner.ensemble.options import METHODS

    labels = validate_entity_labels(ensemble.get("entity_labels", []))
    members = ensemble.get("members", [])
    if not members or ensemble.get("method") not in METHODS:
        raise ValueError("Invalid ensemble method or empty members in format 2.")
    for member in members:
        validate_entity_labels(member.get("entity_labels", []))
        if not member.get("settings_sha256") or not member.get("weights_sha256"):
            raise ValueError("Format 2 requires weight and settings digests for every member.")
    for key in ("entity_weights", "entity_thresholds", "selected_members"):
        if set(ensemble.get(key, {})) - set(labels):
            raise ValueError(f"{key} contains labels outside the ensemble catalogue.")
    if ensemble.get("weight_scope") not in ("global", "entity") or ensemble.get(
        "threshold_scope"
    ) not in ("global", "entity"):
        raise ValueError("Invalid persisted weight_scope or threshold_scope.")
    support = ensemble.get("min_support")
    if support is not None and (type(support) is not int or support < 1):
        raise ValueError("Persisted min_support must be a positive integer.")
    if ensemble.get("missing_policy") not in ("skip", "break"):
        raise ValueError("Invalid ensemble missing_policy; use skip or break.")
    if ensemble.get("votes") not in ("hard", "soft"):
        raise ValueError("Invalid ensemble votes; use hard or soft.")
    for label in labels:
        eligible = [i for i, member in enumerate(members) if label in member["entity_labels"]]
        if not eligible:
            raise ValueError(f"No persisted member supports {label!r}.")
        weights = ensemble.get("entity_weights", {}).get(label)
        if weights is not None and (len(weights) != len(eligible) or sum(weights) <= 0):
            raise ValueError(f"Entity weights for {label!r} disagree with eligible members.")
        selected = ensemble.get("selected_members", {}).get(label)
        if ensemble["method"] == "best" and selected not in eligible:
            raise ValueError(f"Selected specialist for {label!r} is not eligible.")
    values = [member["weight"] for member in members]
    values.extend(
        value for weights in ensemble.get("entity_weights", {}).values() for value in weights
    )
    if not np.isfinite(values).all() or min(values) < 0 or sum(m["weight"] for m in members) <= 0:
        raise ValueError("Ensemble weights must be finite, non-negative and have positive total.")
    thresholds = [ensemble["threshold"], *ensemble.get("entity_thresholds", {}).values()]
    if not np.isfinite(thresholds).all() or min(thresholds) < 0 or max(thresholds) > 1:
        raise ValueError("Ensemble thresholds must be finite values in [0,1].")
