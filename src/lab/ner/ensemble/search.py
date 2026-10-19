"""Tuning an ensemble's weights and threshold on validation data, with Optuna's TPE sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

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
