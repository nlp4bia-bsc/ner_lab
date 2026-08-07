"""Declaring what a sweep searches, and resolving the batch-size dimension."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ray import tune
from ray.tune.search.sample import Categorical, Domain, Float, Integer

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 2e-5, "high": 6e-5},
    "weight_decay": {"type": "uniform", "low": 0.0, "high": 0.1},
    "warmup_ratio": {"type": "uniform", "low": 0.0, "high": 0.1},
    "effective_train_batch_size": {"type": "choice", "categories": [16, 32, 64]},
    "lr_scheduler_type": {"type": "choice", "categories": ["linear", "cosine"]},
}

DOMAIN_TYPES = ("choice", "uniform", "loguniform", "randint")


def build_search_space(
    overrides: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] = DEFAULT_SEARCH_SPACE,
) -> dict[str, Domain]:
    """
    Build the Ray Tune search space from `defaults`, overridden by whatever you pass.

    Each value is a Ray Tune domain (used as given), a declarative mapping such as
    `{"type": "loguniform", "low": 2e-5, "high": 6e-5}` (the YAML form), a plain
    scalar (pinned to that constant instead of searched), or `None` (the dimension
    is removed entirely).
    """
    merged = {**defaults, **(overrides or {})}

    return {
        name: build_domain(name, specification)
        for name, specification in merged.items()
        if specification is not None
    }


def build_domain(name: str, specification: Any) -> Domain:
    """One search dimension from a domain, a declarative mapping, or a pinned scalar."""
    if isinstance(specification, Domain):
        return specification

    if isinstance(specification, Mapping):
        kind = specification.get("type")

        if kind == "choice":
            return tune.choice(list(specification["categories"]))
        if kind == "uniform":
            return tune.uniform(specification["low"], specification["high"])
        if kind == "loguniform":
            return tune.loguniform(specification["low"], specification["high"])
        if kind == "randint":
            return tune.randint(specification["low"], specification["high"])

        raise ValueError(
            f"Search dimension {name!r} has unknown type {kind!r}; expected one of "
            f"{DOMAIN_TYPES}, a Ray Tune domain, or a scalar to pin the value."
        )

    return tune.choice([specification])


def describe_search_space(space: Mapping[str, Domain]) -> dict[str, dict[str, Any]]:
    """The declarative form of a search space, as recorded in the run manifest."""
    described: dict[str, dict[str, Any]] = {}

    for name, domain in space.items():
        if isinstance(domain, Categorical):
            described[name] = {"type": "choice", "categories": list(domain.categories)}
        elif isinstance(domain, Float):
            kind = "loguniform" if "Log" in type(domain.sampler).__name__ else "uniform"
            described[name] = {"type": kind, "low": domain.lower, "high": domain.upper}
        elif isinstance(domain, Integer):
            described[name] = {"type": "randint", "low": domain.lower, "high": domain.upper}
        else:
            described[name] = {"type": type(domain).__name__, "repr": repr(domain)}

    return described


def smoke_configuration(space: Mapping[str, Domain]) -> dict[str, Any]:
    """
    One deterministic configuration from inside the space, for the pre-flight run.

    Categorical dimensions take their first category, numeric ones their lower
    bound — always a value the space itself declares sane.
    """
    configuration: dict[str, Any] = {}

    for name, domain in space.items():
        if isinstance(domain, Categorical):
            configuration[name] = domain.categories[0]
        elif isinstance(domain, (Float, Integer)):
            configuration[name] = domain.lower
        else:
            configuration[name] = domain.sample()

    return configuration


def resolve_batch_sizes(effective_batch_size: int, max_micro_batch_size: int) -> tuple[int, int]:
    """
    Split an effective batch size into (per-device micro batch, accumulation steps).

    The effective size is what the sweep searches; the micro batch is capped by
    `max_micro_batch_size` so the same sweep stays reachable on a smaller GPU
    without changing what is compared.
    """
    if effective_batch_size < 1 or max_micro_batch_size < 1:
        raise ValueError("effective_batch_size and max_micro_batch_size must be at least 1.")

    micro = min(effective_batch_size, max_micro_batch_size)

    while effective_batch_size % micro:
        micro -= 1

    return micro, effective_batch_size // micro
