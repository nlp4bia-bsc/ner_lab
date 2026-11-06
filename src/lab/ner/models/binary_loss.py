"""Stable masked binary objectives and train-only positive-weight estimation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BinaryLoss:
    """One discoverable loss family, including only applicable nondefault settings."""

    loss: str = "bce"
    pos_weight_policy: str = "none"
    weight_smoothing: float = 1.0
    max_pos_weight: float = 20.0
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    gamma_pos: float = 0.0
    gamma_neg: float = 4.0
    clip: float = 0.05

    def __post_init__(self) -> None:
        if self.loss not in ("bce", "weighted_bce", "focal", "asymmetric"):
            raise ValueError(f"Unknown binary loss {self.loss!r}.")
        if self.pos_weight_policy not in ("none", "train"):
            raise ValueError("pos_weight_policy must be none or train.")
        for name in ("focal_gamma", "gamma_pos", "gamma_neg", "weight_smoothing", "max_pos_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.weight_smoothing <= 0 or self.max_pos_weight < 1:
            raise ValueError("Use positive weight_smoothing and max_pos_weight >= 1.")
        if not math.isfinite(self.clip) or not 0 <= self.clip < 1:
            raise ValueError("clip must lie in [0,1).")
        if self.focal_alpha is not None and not 0 <= self.focal_alpha <= 1:
            raise ValueError("focal_alpha must lie in [0,1].")
        defaults = BinaryLoss.__dataclass_fields__
        groups = {
            "focal": ("focal_gamma", "focal_alpha"),
            "asymmetric": ("gamma_pos", "gamma_neg", "clip"),
            "weighted_bce": ("pos_weight_policy", "weight_smoothing", "max_pos_weight"),
        }
        for family, names in groups.items():
            if self.loss != family and any(getattr(self, n) != defaults[n].default for n in names):
                raise ValueError(f"Options {names} apply only to loss={family}.")

    def arguments(self) -> dict:
        """Return explicit constructor arguments without a generic configuration layer."""
        return asdict(self)


def binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    options: BinaryLoss,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean valid element loss per entity, then mean over supervised entities."""
    if logits.shape != labels.shape or logits.ndim != 4 or logits.shape[-1] != 2:
        raise ValueError("Binary logits and labels must share shape [B,L,K,2].")
    valid = labels != -100
    if not torch.equal(valid[..., 0], valid[..., 1]):
        raise ValueError("Both B/I channels must share supervision masks.")
    if (valid & (labels != 0) & (labels != 1)).any():
        raise ValueError("Binary gold must be 0/1 or -100.")
    if ((labels == 1).sum(-1) > 1).any():
        raise ValueError("One BIO track cannot supervise B and I simultaneously.")
    if options.loss == "weighted_bce" and pos_weight is None:
        raise ValueError(
            "weighted_bce requires positive weights estimated from train or supplied explicitly."
        )
    if pos_weight is not None and (
        pos_weight.shape != logits.shape[-2:]
        or not torch.isfinite(pos_weight).all()
        or (pos_weight <= 0).any()
    ):
        raise ValueError("pos_weight must be finite positive [K,2].")
    if options.loss != "weighted_bce" and pos_weight is not None:
        raise ValueError("pos_weight applies only to weighted_bce.")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.float()
        targets = labels.float().masked_fill(~valid, 0)
        if options.loss in ("bce", "weighted_bce", "focal"):
            losses = F.binary_cross_entropy_with_logits(
                values, targets, reduction="none", pos_weight=pos_weight
            )
            if options.loss == "focal":
                probability = torch.sigmoid(values)
                chosen = torch.where(targets.bool(), probability, 1 - probability)
                losses = losses * (1 - chosen).pow(options.focal_gamma)
                if options.focal_alpha is not None:
                    losses = losses * torch.where(
                        targets.bool(), options.focal_alpha, 1 - options.focal_alpha
                    )
        else:
            probability = torch.sigmoid(values)
            negative = (1 - probability + options.clip).clamp(max=1)
            log_negative = (
                F.logsigmoid(-values)
                if options.clip == 0
                else negative.clamp_min(torch.finfo(torch.float32).tiny).log()
            )
            log_chosen = torch.where(targets.bool(), F.logsigmoid(values), log_negative)
            chosen = torch.where(targets.bool(), probability, negative)
            exponent = torch.where(targets.bool(), options.gamma_pos, options.gamma_neg)
            losses = -log_chosen * (1 - chosen).pow(exponent)
        counts = valid.sum((0, 1, 3))
        head_loss = losses.masked_fill(~valid, 0).sum((0, 1, 3)) / counts.clamp_min(1)

        return head_loss.sum() / (counts > 0).sum().clamp_min(1)


def estimate_positive_weights(
    train_rows: pd.DataFrame,
    entity_labels: list[str],
    smoothing: float = 1.0,
    maximum: float = 20.0,
) -> dict[str, list[float]]:
    """Clip (negative+smoothing)/(positive+smoothing) to [1,maximum], using train only."""
    if smoothing <= 0 or maximum < 1 or not math.isfinite(smoothing + maximum):
        raise ValueError("Use finite smoothing > 0 and maximum >= 1.")
    positive = np.zeros((len(entity_labels), 2), dtype=np.int64)
    negative = np.zeros_like(positive)
    for values in train_rows.labels:
        values = np.asarray(values)
        if values.shape[1:] != positive.shape:
            raise ValueError("Training targets disagree with the binary entity catalogue.")
        positive += (values == 1).sum(0)
        negative += (values == 0).sum(0)
    weights = np.clip((negative + smoothing) / (positive + smoothing), 1, maximum)

    return dict(zip(entity_labels, weights.tolist()))
