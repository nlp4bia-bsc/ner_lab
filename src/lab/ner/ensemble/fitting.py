"""Validation-only fitting and method comparisons over cached member predictions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from lab.ner.ensemble.assessment import evaluate_baselines
from lab.ner.ensemble.combination import combine_predictions
from lab.ner.ensemble.entities import EntityVotes, VotingParameters, score_ensemble
from lab.ner.ensemble.search import search_ensemble, search_entity_parameters
from lab.ner.ensemble.tables import MemberPredictions
from lab.ner.evaluation.scoring import span_metrics


@dataclass(frozen=True)
class FittedPredictions:
    """Selected parameters, comparable validation scores and search trace."""

    parameters: VotingParameters
    metrics: dict[str, Any]
    trials: pd.DataFrame | None
    selected_members: dict[str, int]
    timings: dict[str, float]


def fit_predictions(
    predictions: list[MemberPredictions],
    aligned: dict[str, EntityVotes],
    gold: pd.DataFrame,
    texts: dict[str, str],
    method: str,
    votes: str,
    tuned: bool,
    metric: str,
    n_trials: int,
    random_state: int | None,
    weight_scope: str,
    threshold_scope: str,
    min_gold_entities: int,
    missing_policy: str,
    legacy: bool,
    min_overlap_percentage: float,
    min_support: int | None,
    compare_methods: bool,
) -> FittedPredictions:
    """Fit one requested method; compare scoped baselines without redoing inference."""
    started = time.perf_counter()
    assessment = evaluate_baselines(
        predictions,
        aligned,
        gold,
        texts,
        legacy,
        missing_policy,
        min_gold_entities,
        min_overlap_percentage,
    )
    support = gold.drop_duplicates(["filename", "label", "start_span", "end_span"])[
        "label"
    ].value_counts()
    assessment["calibration_support"] = {label: int(support.get(label, 0)) for label in aligned}
    assessment["parameter_fallbacks"] = {
        label: {"weights": "uniform", "threshold": "global" if threshold_scope == "global" else 0.5}
        for label in aligned
        if support.get(label, 0) < min_gold_entities
    }
    parameters = VotingParameters([1 / len(predictions)] * len(predictions))
    trials = None
    selected = assessment["selected_members"]
    tuning_time = 0.0
    if tuned and method != "best":
        tuning_started = time.perf_counter()
        legacy_metric = legacy and metric in span_metrics(
            gold,
            predictions[0].spans,
            tags=list(aligned),
            min_overlap_percentage=min_overlap_percentage,
        )
        legacy_search = (
            legacy_metric
            and weight_scope == "global"
            and threshold_scope == "global"
            and missing_policy == "skip"
            and (method != "weighted" or len(predictions) > 1)
        )
        if legacy_search:
            label, entity = next(iter(aligned.items()))
            original = search_ensemble(
                entity.votes,
                gold,
                texts,
                label,
                votes,
                method == "weighted",
                n_trials,
                metric,
                random_state,
                min_overlap_percentage,
            )
            parameters = VotingParameters(original.weights, original.threshold)
            trials = original.trials_df
            assessment["selection"] = {
                "search": "legacy_global",
                "equal_weight_baseline": assessment["baselines"][
                    "majority" if votes == "hard" else "soft"
                ][metric],
            }
        else:
            result = search_entity_parameters(
                aligned,
                gold,
                texts,
                len(predictions),
                votes,
                method == "weighted",
                weight_scope,
                threshold_scope,
                n_trials,
                metric,
                random_state,
                min_gold_entities,
                missing_policy,
                legacy,
                min_overlap_percentage,
            )
            parameters, trials = result.parameters, result.trials_df
            assessment["selection"] = {
                "search": "scoped",
                "equal_weight_baseline": result.baseline_score,
                "selected_baseline": result.score == result.baseline_score,
            }
        tuning_time += time.perf_counter() - tuning_started
    if compare_methods:
        for weights in ("global", "entity"):
            for thresholds in ("global", "entity"):
                tuning_started = time.perf_counter()
                result = search_entity_parameters(
                    aligned,
                    gold,
                    texts,
                    len(predictions),
                    "soft",
                    True,
                    weights,
                    thresholds,
                    n_trials,
                    metric,
                    random_state,
                    min_gold_entities,
                    missing_policy,
                    legacy,
                    min_overlap_percentage,
                )
                tuning_time += time.perf_counter() - tuning_started
                spans = combine_predictions(
                    predictions,
                    aligned,
                    "weighted",
                    result.parameters,
                    "soft",
                    texts,
                    missing_policy=missing_policy,
                )
                name = f"weighted_{weights}_weights_{thresholds}_thresholds"
                assessment["baselines"][name] = score_ensemble(
                    gold, spans, list(aligned), legacy, min_overlap_percentage
                )
                assessment.setdefault("baseline_parameters", {})[name] = {
                    "weights": result.parameters.weights,
                    "threshold": result.parameters.threshold,
                    "entity_weights": result.parameters.entity_weights,
                    "entity_thresholds": result.parameters.entity_thresholds,
                }
    spans = combine_predictions(
        predictions,
        aligned,
        method,
        parameters,
        votes,
        texts,
        selected,
        min_support,
        missing_policy,
    )
    metrics = score_ensemble(gold, spans, list(aligned), legacy, min_overlap_percentage)
    if metric not in metrics:
        raise ValueError(f"metric {metric!r} is not reported; choose one of {sorted(metrics)}.")
    assessment["ensemble"] = metrics
    best = max(range(len(predictions)), key=lambda index: assessment["members"][index][metric])
    assessment["best_member_global"] = {
        "index": best,
        "metric": metric,
        "value": assessment["members"][best][metric],
    }

    return FittedPredictions(
        parameters,
        assessment,
        trials,
        selected,
        {"fit_sec": time.perf_counter() - started, "optuna_sec": tuning_time},
    )
