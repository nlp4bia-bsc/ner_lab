"""The `ner.fit_ensemble` task: saved models, and validation data when tuning, to a fitted ensemble."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from lab.core.provenance import file_sha256, write_manifest
from lab.ner.ensemble.io import (
    ENSEMBLE_FILENAME,
    ENSEMBLE_FORMAT_VERSION,
    describe_member,
    shared_target_label,
    write_ensemble,
)
from lab.ner.ensemble.search import (
    SAMPLER,
    THRESHOLD_BOUNDS,
    WEIGHT_BOUNDS,
    search_ensemble,
)
from lab.ner.ensemble.tables import predict_word_tables
from lab.ner.ensemble.voting import (
    VOTES,
    Votes,
    WordVotes,
    align_votes,
    ensemble_spans,
    normalize_weights,
)
from lab.ner.evaluation.scoring import span_metrics
from lab.ner.inference.io import read_documents, resolve_gold
from lab.ner.inference.loading import resolve_device

Method = Literal["majority", "soft", "weighted"]

METHODS = ("majority", "soft", "weighted")
DEFAULT_THRESHOLD = 0.5
ENSEMBLE_MANIFEST_FILENAME = "ensemble_manifest.json"
ENSEMBLE_TRIALS_FILENAME = "ensemble_trials.parquet"
ENSEMBLE_METRICS_FILENAME = "ensemble_metrics.json"


@dataclass(frozen=True)
class EnsembleFit:
    """What fitting an ensemble produced: the ensemble, any validation scores and trials."""

    output_dir: Path
    ensemble: dict[str, Any]
    metrics: dict[str, Any] | None
    trials: pd.DataFrame | None
    manifest: dict[str, Any]
    paths: dict[str, Path]

    def summary(self) -> str:
        """A one-line account of the fit, with the validation score when there is one."""
        line = (
            f"{self.ensemble['method']} ensemble of {len(self.ensemble['members'])} members, "
            f"{self.ensemble['votes']} votes, threshold {self.ensemble['threshold']:.3f}"
        )

        if self.metrics is None:
            return line

        best_member = max(member["span_strict_f1"] for member in self.metrics["members"])

        return (
            f"{line} | validation strict F1 = {self.metrics['ensemble']['span_strict_f1']}"
            f" (best member {best_member})"
        )


def fit_ensemble(
    model_dirs: Sequence[str | Path],
    method: Method,
    output_dir: str | Path,
    validation: pd.DataFrame | dict[str, str] | str | Path | None = None,
    reference: str | Path | None = None,
    votes: Votes | None = None,
    tune_threshold: bool = False,
    metric: str = "span_strict_f1",
    n_trials: int = 200,
    random_state: int | None = None,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    min_overlap_percentage: float = 40.0,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
) -> EnsembleFit:
    """
    Fit an ensemble of saved models and write it to `output_dir`.

    `model_dirs` are saved models, each with the `encoding.json` beside its weights;
    they may differ in backbone, architecture and windowing, but must predict the
    same `target_label`. Members vote word by word on UAX #29 words, and the
    combined votes decode to spans.

    `method` picks how votes combine. `"majority"` counts each member's own tag
    with equal weights, and a word is in an entity when strictly more than half the
    members that labelled it say so. `"soft"` averages the members' probabilities
    with equal weights at a threshold of 0.5, or at a tuned one with
    `tune_threshold`. `"weighted"` tunes one weight per member and the threshold;
    its votes are the members' probabilities, or their tags with `votes="hard"`.

    `validation` is gold-annotated data in any form `predict_entities` reads,
    with `reference` supplying the gold when it does not carry its own. Anything
    tuned needs it; otherwise it is optional, and when given the ensemble and each
    member alone are scored on it. Which documents to tune on, and whether a member
    trained on them, is the caller's to decide.

    Tuning runs `n_trials` of Optuna's TPE sampler, seeded by `random_state`,
    maximising `metric` — any key `predict_entities` reports. The members run once
    over the validation data; every trial after that is CPU work.

    `ensemble.json` is what `predict_ensemble` reads: the method, the threshold,
    and each member's path, weight and weights' digest. `ensemble_manifest.json`
    records how it was fitted, `ensemble_trials.parquet` every trial of a search,
    and `ensemble_metrics.json` the validation scores.
    """
    votes = _resolve_votes(method, votes)
    tuned = _check_tuning(method, tune_threshold, validation)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    members = [describe_member(model_dir) for model_dir in model_dirs]
    _check_members(members)
    target_label = shared_target_label(members)

    weights = normalize_weights([1.0] * len(members))
    threshold = DEFAULT_THRESHOLD
    metrics: dict[str, Any] | None = None
    trials: pd.DataFrame | None = None
    validation_record: dict[str, Any] | None = None

    if validation is not None:
        corpus = read_documents(validation, pattern=pattern, encoding=encoding)
        gold = resolve_gold(reference, corpus, target_label)

        if gold is None:
            raise ValueError(
                "validation carries no gold to fit or score against. Pass a corpus with "
                "entities, or its gold as reference=."
            )

        texts = dict(zip(corpus["doc_id"], corpus["text"]))
        tables = predict_word_tables(
            [member["model_dir"] for member in members],
            corpus,
            batch_size=batch_size,
            device=device,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        word_votes = align_votes(tables)
        scoring = {
            "gold": gold,
            "texts": texts,
            "target_label": target_label,
            "min_overlap_percentage": min_overlap_percentage,
        }
        member_metrics = _member_metrics(word_votes, members, **scoring)

        if tuned:
            _check_metric(metric, member_metrics[0])
            result = search_ensemble(
                word_votes,
                votes=votes,
                tune_weights=method == "weighted",
                n_trials=n_trials,
                metric=metric,
                random_state=random_state,
                **scoring,
            )
            weights, threshold, trials = result.weights, result.threshold, result.trials_df

        metrics = {
            "ensemble": _score(word_votes, weights, threshold, votes, **scoring),
            "members": member_metrics,
        }
        validation_record = _describe_validation(validation, reference, corpus, gold)

    ensemble = {
        "format_version": ENSEMBLE_FORMAT_VERSION,
        "method": method,
        "votes": votes,
        "target_label": target_label,
        "threshold": threshold,
        "members": [
            {
                "model_dir": member["model_dir"],
                "weight": weight,
                "weights_sha256": member["weights_sha256"],
            }
            for member, weight in zip(members, weights)
        ],
    }
    paths = {"ensemble": write_ensemble(ensemble, output_dir / ENSEMBLE_FILENAME)}

    if trials is not None:
        paths["trials"] = output_dir / ENSEMBLE_TRIALS_FILENAME
        trials.to_parquet(paths["trials"], index=False)

    if metrics is not None:
        paths["metrics"] = write_manifest(metrics, output_dir / ENSEMBLE_METRICS_FILENAME)

    manifest = {
        "method": method,
        "votes": votes,
        "tune_threshold": tune_threshold,
        "target_label": target_label,
        "output_dir": str(output_dir),
        "members": members,
        "validation": validation_record,
        "search": _describe_search(method, tuned, n_trials, metric, random_state),
        "min_overlap_percentage": float(min_overlap_percentage),
        "device": str(resolve_device(device)),
        "duration_sec": round(time.perf_counter() - started, 3),
    }
    paths["ensemble_manifest"] = write_manifest(manifest, output_dir / ENSEMBLE_MANIFEST_FILENAME)

    return EnsembleFit(
        output_dir=output_dir,
        ensemble=ensemble,
        metrics=metrics,
        trials=trials,
        manifest=manifest,
        paths=paths,
    )


def _resolve_votes(method: str, votes: str | None) -> Votes:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")

    if method != "weighted":
        if votes is not None:
            raise ValueError(
                f"votes only applies to method='weighted'; {method!r} fixes its own votes. "
                "Leave votes unset."
            )

        return "hard" if method == "majority" else "soft"

    if votes is None:
        return "soft"

    if votes not in VOTES:
        raise ValueError(f"votes must be one of {VOTES}, got {votes!r}.")

    return votes


def _check_tuning(method: str, tune_threshold: bool, validation: Any) -> bool:
    if tune_threshold and method != "soft":
        raise ValueError(
            f"tune_threshold only applies to method='soft', not {method!r}: 'weighted' "
            "always tunes its threshold and 'majority' keeps a strict majority."
        )

    tuned = method == "weighted" or tune_threshold

    if tuned and validation is None:
        raise ValueError(
            f"This ensemble tunes its threshold{' and weights' if method == 'weighted' else ''}, "
            "so it needs validation= data with gold to tune on."
        )

    return tuned


def _check_members(members: list[dict[str, Any]]) -> None:
    if not members:
        raise ValueError("model_dirs is empty; an ensemble needs at least one member.")

    model_dirs = [member["model_dir"] for member in members]
    repeated = sorted({path for path in model_dirs if model_dirs.count(path) > 1})

    if repeated:
        raise ValueError(f"model_dirs names the same model more than once: {repeated[:10]}.")


def _check_metric(metric: str, metrics: dict[str, Any]) -> None:
    if metric not in metrics:
        available = sorted(key for key in metrics if key != "model_dir")

        raise ValueError(f"metric {metric!r} is not a reported metric. Choose one of {available}.")


def _score(
    word_votes: WordVotes,
    weights: list[float],
    threshold: float,
    votes: Votes,
    gold: pd.DataFrame,
    texts: dict[str, str],
    target_label: str,
    min_overlap_percentage: float,
) -> dict[str, Any]:
    spans = ensemble_spans(
        word_votes,
        weights=weights,
        threshold=threshold,
        votes=votes,
        texts=texts,
        target_label=target_label,
    )

    return span_metrics(
        gold=gold,
        predicted=spans,
        tags=[target_label],
        min_overlap_percentage=min_overlap_percentage,
    )


def _member_metrics(
    word_votes: WordVotes, members: list[dict[str, Any]], **scoring: Any
) -> list[dict[str, Any]]:
    return [
        {
            "model_dir": member["model_dir"],
            **_score(
                word_votes,
                [1.0 if other == index else 0.0 for other in range(len(members))],
                DEFAULT_THRESHOLD,
                "hard",
                **scoring,
            ),
        }
        for index, member in enumerate(members)
    ]


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


def _describe_search(
    method: str, tuned: bool, n_trials: int, metric: str, random_state: int | None
) -> dict[str, Any] | None:
    if not tuned:
        return None

    return {
        "tuned": ["weights", "threshold"] if method == "weighted" else ["threshold"],
        "sampler": SAMPLER,
        "n_trials": int(n_trials),
        "metric": metric,
        "random_state": random_state,
        "weight_bounds": list(WEIGHT_BOUNDS) if method == "weighted" else None,
        "threshold_bounds": list(THRESHOLD_BOUNDS),
    }
