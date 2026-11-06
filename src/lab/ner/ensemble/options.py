"""Validation of explicit ensemble method and parameter scopes."""

from __future__ import annotations

from typing import Literal

Method = Literal[
    "majority", "soft", "weighted", "best", "span_union", "span_consensus", "span_intersection"
]
METHODS = (
    "majority",
    "soft",
    "weighted",
    "best",
    "span_union",
    "span_consensus",
    "span_intersection",
)


def check_options(
    method: str,
    votes: str | None,
    tune_threshold: bool,
    validation: object,
    weight_scope: str,
    threshold_scope: str,
    min_support: int | None,
    missing_policy: str | None,
    min_gold_entities: int,
    n_trials: int,
) -> tuple[str, bool]:
    """Reject irrelevant options before reading checkpoints or starting inference."""
    if method == "stacking":
        raise ValueError(
            "Stacking requires out-of-fold or independently calibrated predictions; "
            "this API does not accept such provenance and does not train a meta-model."
        )
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    if votes is not None and method != "weighted":
        raise ValueError("votes only applies to method='weighted'; leave votes unset.")
    if votes not in (None, "hard", "soft"):
        raise ValueError("votes must be 'hard' or 'soft'.")
    if tune_threshold and method != "soft":
        raise ValueError("tune_threshold only applies to method='soft'; weighted always tunes it.")
    if weight_scope not in ("global", "entity") or threshold_scope not in ("global", "entity"):
        raise ValueError("weight_scope and threshold_scope must be 'global' or 'entity'.")
    if weight_scope != "global" and method != "weighted":
        raise ValueError("weight_scope='entity' requires method='weighted'.")
    if threshold_scope != "global" and not (
        method == "weighted" or method == "soft" and tune_threshold
    ):
        raise ValueError("threshold_scope='entity' requires weighted or soft with tune_threshold.")
    if min_support is not None and method not in ("span_consensus", "span_intersection"):
        raise ValueError("min_support only applies to span_consensus or span_intersection.")
    if min_support is not None and (type(min_support) is not int or min_support < 1):
        raise ValueError("min_support must be a positive integer.")
    if type(min_gold_entities) is not int or min_gold_entities < 1:
        raise ValueError("min_gold_entities must be a positive integer.")
    if missing_policy not in (None, "skip", "break"):
        raise ValueError("missing_policy must be 'skip' or 'break'.")
    if missing_policy is not None and method not in ("majority", "soft", "weighted"):
        raise ValueError("missing_policy only applies to BIO voting methods.")
    tuned = method in ("weighted", "best") or tune_threshold
    if tuned and validation is None:
        raise ValueError("This method needs validation= with gold to select its parameters.")
    if tuned and (type(n_trials) is not int or n_trials < 1):
        raise ValueError("n_trials must be positive when tuning.")

    return ("hard" if method == "majority" else votes or "soft"), tuned
