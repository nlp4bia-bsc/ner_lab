"""Fit heterogeneous NER ensembles using metadata catalogues and cached predictions."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.provenance import file_sha256, write_manifest
from lab.ner.ensemble.calibration import describe_calibration
from lab.ner.ensemble.entities import VotingParameters, align_entities
from lab.ner.ensemble.fitting import fit_predictions
from lab.ner.ensemble.io import (
    ENSEMBLE_FILENAME,
    ENSEMBLE_FORMAT_VERSION,
    describe_member,
    write_ensemble,
)
from lab.ner.ensemble.options import METHODS as METHODS
from lab.ner.ensemble.options import Method, check_options
from lab.ner.ensemble.tables import predict_members
from lab.ner.inference.io import read_documents, resolve_gold

ENSEMBLE_MANIFEST_FILENAME = "ensemble_manifest.json"
ENSEMBLE_TRIALS_FILENAME = "ensemble_trials.parquet"
ENSEMBLE_METRICS_FILENAME = "ensemble_metrics.json"


@dataclass(frozen=True)
class EnsembleFit:
    """Fitted parameters, validation evidence and artifact paths."""

    output_dir: Path
    ensemble: dict[str, Any]
    metrics: dict[str, Any] | None
    trials: pd.DataFrame | None
    manifest: dict[str, Any]
    paths: dict[str, Path]

    def summary(self) -> str:
        """A concise account, using the actual selection metric when scored."""
        line = f"{self.ensemble['method']} ensemble of {len(self.ensemble['members'])} members"
        if self.metrics is None:
            return line
        metric = self.ensemble["metric"]

        return (
            f"{line} | validation {metric} = {self.metrics['ensemble'][metric]}"
            f" (best member {self.metrics['best_member_global']['value']})"
        )


def fit_ensemble(
    model_dirs: Sequence[str | Path],
    method: Method,
    output_dir: str | Path,
    validation: pd.DataFrame | dict[str, str] | str | Path | None = None,
    reference: str | Path | None = None,
    votes: str | None = None,
    tune_threshold: bool = False,
    metric: str | None = None,
    n_trials: int = 200,
    random_state: int | None = None,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    min_overlap_percentage: float = 40.0,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
    entity_labels: Sequence[str] | None = None,
    weight_scope: str = "global",
    threshold_scope: str = "global",
    min_support: int | None = None,
    missing_policy: str | None = None,
    min_gold_entities: int = 5,
    compare_methods: bool = True,
) -> EnsembleFit:
    """Fit word voting, specialists or exact span consensus through one inference path.

    Labels are the union of checkpoint catalogues unless entity_labels selects a
    subset. Unknown labels and missing positions never vote O. Scoped tuning uses
    only validation predictions, cached once for all Optuna trials and baselines.
    New artifacts use format 2; prediction still reads historical format 1.
    """
    votes, tuned = check_options(
        method,
        votes,
        tune_threshold,
        validation,
        weight_scope,
        threshold_scope,
        min_support,
        missing_policy,
        min_gold_entities,
        n_trials,
    )
    if validation is not None and compare_methods and (type(n_trials) is not int or n_trials < 1):
        raise ValueError("n_trials must be positive for compare_methods=True.")
    started = time.perf_counter()
    members, labels = _resolve_members(model_dirs, entity_labels)
    legacy = len(labels) == 1 and all(m["encoding"]["target_label"] is not None for m in members)
    legacy = legacy and method in ("majority", "soft", "weighted")
    missing_policy = missing_policy or ("skip" if legacy else "break")
    metric = metric or ("span_strict_f1" if legacy else "span_strict_macro_f1")
    parameters = VotingParameters([1 / len(members)] * len(members))
    metrics, trials, calibration, validation_record = None, None, None, None
    selected, timings = {}, {}
    if validation is not None:
        corpus = read_documents(validation, pattern=pattern, encoding=encoding)
        gold = resolve_gold(reference, corpus, labels[0] if legacy else None)
        if gold is None:
            raise ValueError(
                "validation carries no gold; supply annotated validation= or reference=."
            )
        if entity_labels is not None:
            gold = gold[gold["label"].isin(labels)].reset_index(drop=True)
        calibration = describe_calibration(members, corpus, tuned or compare_methods)
        predictions = predict_members(
            [m["model_dir"] for m in members],
            corpus,
            batch_size,
            device,
            pad_to_multiple_of,
            use_annotations=legacy,
        )
        aligned_started = time.perf_counter()
        aligned = align_entities(predictions, labels)
        timings["alignment_sec"] = time.perf_counter() - aligned_started
        fitted = fit_predictions(
            predictions,
            aligned,
            gold,
            dict(zip(corpus["doc_id"], corpus["text"])),
            method,
            votes,
            tuned,
            metric,
            n_trials,
            random_state,
            weight_scope,
            threshold_scope,
            min_gold_entities,
            missing_policy,
            legacy,
            min_overlap_percentage,
            min_support,
            compare_methods,
        )
        parameters, metrics, trials, selected = (
            fitted.parameters,
            fitted.metrics,
            fitted.trials,
            fitted.selected_members,
        )
        for index, member in enumerate(members):
            metrics["members"][index].update(
                model_dir=member["model_dir"],
                supported_labels=member["entity_labels"],
                evaluation_scope="ensemble_catalogue",
            )
            metrics["specialists"][index].update(
                model_dir=member["model_dir"], evaluation_scope="supported_labels_only"
            )
        timings.update(fitted.timings)
        timings["members"] = [prediction.timings for prediction in predictions]
        validation_record = _describe_validation(validation, reference, corpus, gold)
    ensemble = {
        "format_version": ENSEMBLE_FORMAT_VERSION,
        "method": method,
        "votes": votes,
        "entity_labels": labels,
        "target_label": labels[0] if len(labels) == 1 else None,
        "threshold": parameters.threshold,
        "entity_thresholds": parameters.entity_thresholds,
        "entity_weights": parameters.entity_weights,
        "weight_scope": weight_scope,
        "threshold_scope": threshold_scope,
        "missing_policy": missing_policy,
        "min_support": min_support,
        "metric": metric,
        "selected_members": selected,
        "legacy_semantics": legacy,
        "restrict_gold": entity_labels is not None,
        "min_gold_entities": min_gold_entities,
        "calibration": calibration,
        "members": [
            {
                "model_dir": member["model_dir"],
                "weight": parameters.weights[index],
                "architecture": member["model"].get("architecture", "linear"),
                "entity_labels": member["entity_labels"],
                "weights_sha256": member["weights_sha256"],
                "settings_sha256": member["settings_sha256"],
            }
            for index, member in enumerate(members)
        ],
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"ensemble": write_ensemble(ensemble, output_dir / ENSEMBLE_FILENAME)}
    if trials is not None:
        paths["trials"] = output_dir / ENSEMBLE_TRIALS_FILENAME
        trials.to_parquet(paths["trials"], index=False)
    if metrics is not None:
        paths["metrics"] = write_manifest(metrics, output_dir / ENSEMBLE_METRICS_FILENAME)
    manifest = {
        "method": method,
        "members": members,
        "validation": validation_record,
        "calibration": calibration,
        "timings": timings,
        "entity_labels": labels,
        "search": (
            {
                "n_trials": n_trials,
                "metric": metric,
                "random_state": random_state,
                "tuned": ["weights", "threshold"] if method == "weighted" else ["threshold"],
            }
            if tuned
            else None
        ),
        "min_overlap_percentage": min_overlap_percentage,
        "duration_sec": time.perf_counter() - started,
    }
    paths["ensemble_manifest"] = write_manifest(manifest, output_dir / ENSEMBLE_MANIFEST_FILENAME)

    return EnsembleFit(output_dir, ensemble, metrics, trials, manifest, paths)


def _resolve_members(
    model_dirs: Sequence[str | Path], requested: Sequence[str] | None
) -> tuple[list[dict[str, Any]], list[str]]:
    if not model_dirs:
        raise ValueError("model_dirs is empty; provide at least one checkpoint.")
    members = [describe_member(path) for path in model_dirs]
    paths = [member["model_dir"] for member in members]
    if len(paths) != len(set(paths)):
        raise ValueError("model_dirs names the same checkpoint more than once.")
    available = sorted({label for member in members for label in member["entity_labels"]})
    labels = available if requested is None else list(requested)
    if not labels or len(set(labels)) != len(labels) or set(labels) - set(available):
        raise ValueError(f"entity_labels must be a unique, nonempty subset of {available}.")
    members = [member for member in members if set(labels) & set(member["entity_labels"])]

    return members, labels


def _describe_validation(
    validation: Any,
    reference: str | Path | None,
    corpus: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, Any]:
    path = Path(validation).resolve() if isinstance(validation, (str, Path)) else None
    reference_path = None if reference is None else Path(reference).resolve()

    return {
        "path": None if path is None else str(path),
        "sha256": file_sha256(path) if path is not None and path.is_file() else None,
        "reference": None if reference_path is None else str(reference_path),
        "reference_sha256": None if reference_path is None else file_sha256(reference_path),
        "n_documents": int(corpus["doc_id"].nunique()),
        "n_gold_entities": int(len(gold)),
    }
