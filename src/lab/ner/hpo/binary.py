"""Conditional binary-loss families for native Ray/Optuna training trials."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

from lab.ner.models.binary_loss import BinaryLoss

BINARY_DIMENSIONS = tuple(field.name for field in fields(BinaryLoss)) + ("training_temperature",)


def loss_family_space(
    space: dict[str, Any],
    variants: Sequence[BinaryLoss | Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Represent complete conditional loss families as one categorical profile dimension."""
    if variants is None:
        return space
    if not variants or any(
        name.startswith("architecture__") and name != "architecture__dropout" for name in space
    ):
        raise ValueError(
            "loss_variants must be nonempty; sample loss-specific settings inside profiles."
        )
    from ray import tune

    options = [item if isinstance(item, BinaryLoss) else BinaryLoss(**item) for item in variants]

    return {
        **space,
        "loss_variant": tune.choice(
            [json.dumps(item.arguments(), sort_keys=True) for item in options]
        ),
    }


def validate_binary_dimensions(space: dict[str, Any], settings: dict[str, Any]) -> None:
    """Only fixed applicable loss families may sample their own parameters independently."""
    from lab.ner.hpo.trial import ARCHITECTURE_DIMENSIONS

    unknown = {
        name
        for name in space
        if name in ARCHITECTURE_DIMENSIONS
        and name.removeprefix("architecture__") not in (*BINARY_DIMENSIONS, "dropout")
    }
    if unknown:
        raise ValueError(
            f"Invalid multilabel dimensions {sorted(unknown)}; decoder searches reuse a checkpoint."
        )
    domain = space.get("architecture__loss")
    if domain is not None and getattr(domain, "categories", None) is None:
        raise ValueError("architecture__loss must be a categorical choice.")
    losses = domain.categories if domain is not None else [settings.get("loss", "bce")]
    for loss in losses:
        BinaryLoss(loss=loss)
    groups = {
        "focal": ("focal_gamma", "focal_alpha"),
        "asymmetric": ("gamma_pos", "gamma_neg", "clip"),
        "weighted_bce": ("pos_weight_policy", "weight_smoothing", "max_pos_weight"),
    }
    for family, names in groups.items():
        if any(f"architecture__{name}" in space for name in names) and losses != [family]:
            raise ValueError(
                f"Search {names} only with fixed loss={family}, or use typed loss_variants."
            )

    import math
    from dataclasses import fields

    names = {field.name for field in fields(BinaryLoss)} | {"dropout", "training_temperature"}
    for dimension, domain in space.items():
        name = dimension.removeprefix("architecture__")
        if not dimension.startswith("architecture__") or name not in names or name == "loss":
            continue
        candidates = getattr(domain, "categories", None)
        values = candidates if candidates is not None else [domain.lower, domain.upper]
        for value in values:
            if name == "dropout":
                valid = isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
            elif name == "training_temperature":
                valid = isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            else:
                BinaryLoss(
                    **{
                        **{
                            key: val
                            for key, val in settings.items()
                            if key in BinaryLoss.__dataclass_fields__
                        },
                        "loss": losses[0],
                        name: value,
                    }
                )
                valid = True
            if not valid:
                raise ValueError(f"Invalid bounds for {dimension}.")
