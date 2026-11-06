"""Tuning an ensemble's weights and threshold on validation data, with Optuna's TPE sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from lab.ner.ensemble.entities import (
    EntityVotes,
    MissingPolicy,
    Scope,
    VotingParameters,
    decode_entities,
    score_ensemble,
)
from lab.ner.ensemble.voting import Votes, WordVotes, ensemble_spans, normalize_weights
from lab.ner.evaluation.scoring import span_metrics

WEIGHT_BOUNDS = (0.0, 1.0)
THRESHOLD_BOUNDS = (0.05, 0.95)
SAMPLER = "optuna.samplers.TPESampler"


@dataclass(frozen=True)
class SearchResult:
    """The best weights and threshold a search found, its score, and every trial it ran."""

    weights: list[float]
    threshold: float
    score: float
    trials_df: pd.DataFrame


def search_ensemble(
    word_votes: WordVotes,
    gold: pd.DataFrame,
    texts: dict[str, str],
    target_label: str,
    votes: Votes,
    tune_weights: bool,
    n_trials: int = 200,
    metric: str = "span_strict_f1",
    random_state: int | None = None,
    min_overlap_percentage: float = 40.0,
) -> SearchResult:
    """
    Maximise `metric` on the validation spans over the threshold, and the weights if asked.

    Each member's weight is sampled in `WEIGHT_BOUNDS` and the set normalised to sum
    to one; a trial whose weights are all zero falls back to equal weights. Without
    `tune_weights` the weights stay equal and only the threshold is searched. Every
    trial combines and decodes the same word votes on the CPU, so trials are cheap.
    Of trials that tie on the best score, the first wins.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    equal_weights = normalize_weights([1.0] * word_votes.n_members)
    records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        weights = equal_weights

        if tune_weights:
            raw = [
                trial.suggest_float(f"raw_weight_{member}", *WEIGHT_BOUNDS)
                for member in range(word_votes.n_members)
            ]
            weights = normalize_weights(raw) if sum(raw) > 0 else equal_weights

        threshold = trial.suggest_float("threshold", *THRESHOLD_BOUNDS)
        spans = ensemble_spans(
            word_votes,
            weights=weights,
            threshold=threshold,
            votes=votes,
            texts=texts,
            target_label=target_label,
        )
        score = span_metrics(
            gold=gold,
            predicted=spans,
            tags=[target_label],
            min_overlap_percentage=min_overlap_percentage,
        )[metric]

        trial.set_user_attr("weights", weights)
        records.append(
            {
                "trial": trial.number,
                **{f"weight_{member}": weight for member, weight in enumerate(weights)},
                "threshold": threshold,
                metric: score,
            }
        )

        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial

    return SearchResult(
        weights=list(best.user_attrs["weights"]),
        threshold=float(best.params["threshold"]),
        score=float(best.value),
        trials_df=pd.DataFrame(records),
    )


@dataclass(frozen=True)
class EntitySearchResult:
    """Scoped voting parameters selected against an equal-weight baseline."""

    parameters: VotingParameters
    score: float
    baseline_score: float
    trials_df: pd.DataFrame


def search_entity_parameters(
    aligned: dict[str, EntityVotes],
    gold: pd.DataFrame,
    texts: dict[str, str],
    n_members: int,
    votes: str,
    tune_weights: bool,
    weight_scope: Scope = "global",
    threshold_scope: Scope = "global",
    n_trials: int = 200,
    metric: str = "span_strict_macro_f1",
    random_state: int | None = None,
    min_gold_entities: int = 5,
    missing_policy: MissingPolicy = "break",
    legacy: bool = False,
    min_overlap_percentage: float = 40.0,
) -> EntitySearchResult:
    """Tune only identifiable, sufficiently supported parameters on cached predictions."""
    import json

    import optuna

    if n_trials < 1:
        raise ValueError("n_trials must be positive when tuning.")
    equal = normalize_weights([1.0] * n_members)
    baseline = VotingParameters(equal)
    support = gold.drop_duplicates(["filename", "label", "start_span", "end_span"])[
        "label"
    ].value_counts()
    sufficient = {label for label in aligned if support.get(label, 0) >= min_gold_entities}
    active = sorted(
        {
            index
            for label, entity in aligned.items()
            if label in sufficient
            for index in entity.member_indices
        }
    )

    from lab.core.segmentation import split_into_words

    word_spans = {doc_id: split_into_words(text) for doc_id, text in texts.items()}

    def score(parameters: VotingParameters) -> float:
        predicted = decode_entities(aligned, parameters, votes, texts, missing_policy, word_spans)
        metrics = score_ensemble(gold, predicted, list(aligned), legacy, min_overlap_percentage)
        if metric not in metrics:
            raise ValueError(f"metric {metric!r} is not reported; choose one of {sorted(metrics)}.")

        return float(metrics[metric])

    baseline_score = score(baseline)
    records = []
    candidates = []

    def objective(trial: optuna.Trial) -> float:
        weights = equal
        if tune_weights and weight_scope == "global" and len(active) > 1:
            weights = [0.0] * n_members
            for member, weight in zip(active, _simplex_weights(trial, "global", len(active))):
                weights[member] = weight
        by_entity = {
            label: [1 / len(entity.member_indices)] * len(entity.member_indices)
            for label, entity in aligned.items()
            if label not in sufficient or len(entity.member_indices) == 1
        }
        if tune_weights and weight_scope == "entity":
            for index, (label, entity) in enumerate(aligned.items()):
                count = len(entity.member_indices)
                by_entity[label] = (
                    _simplex_weights(trial, f"entity_{index}", count)
                    if label in sufficient and count > 1
                    else [1 / count] * count
                )
        threshold = (
            trial.suggest_float("threshold", *THRESHOLD_BOUNDS)
            if (threshold_scope == "global" and sufficient)
            else 0.5
        )
        thresholds = {}
        if threshold_scope == "entity":
            thresholds = {
                label: trial.suggest_float(f"threshold_{index}", *THRESHOLD_BOUNDS)
                if label in sufficient
                else 0.5
                for index, label in enumerate(aligned)
            }
        parameters = VotingParameters(weights, threshold, by_entity, thresholds)
        value = score(parameters)
        candidates.append(parameters)
        records.append(
            {
                "trial": trial.number,
                metric: value,
                "threshold": threshold,
                "parameters_json": json.dumps(trial.params, sort_keys=True),
                "weights_json": json.dumps(weights),
                "entity_weights_json": json.dumps(by_entity, sort_keys=True),
                "entity_thresholds_json": json.dumps(thresholds, sort_keys=True),
            }
        )

        return value

    has_parameters = bool(sufficient) and (threshold_scope in ("global", "entity"))
    if not has_parameters:
        return EntitySearchResult(baseline, baseline_score, baseline_score, pd.DataFrame(records))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    parameters = candidates[best.number] if best.value > baseline_score else baseline

    return EntitySearchResult(
        parameters, max(baseline_score, best.value), baseline_score, pd.DataFrame(records)
    )


def _simplex_weights(trial: Any, prefix: str, count: int) -> list[float]:
    remainder = 1.0
    weights = []
    for member in range(count - 1):
        fraction = trial.suggest_float(f"{prefix}_weight_{member}", 0.0, 1.0)
        weights.append(remainder * fraction)
        remainder *= 1 - fraction
    weights.append(remainder)

    return weights
