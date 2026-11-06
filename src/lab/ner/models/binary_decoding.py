"""Independent sigmoid evidence, legal-state adaptation and simultaneous BIO decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from lab.ner.models.chain import ConstrainedChain
from lab.ner.models.packing import pack_words


@dataclass(frozen=True)
class BinaryDecoding:
    """Fixed post-training score transformation and decision rule; never a loss setting."""

    activation: str = "sigmoid"
    temperature: float | dict[str, float] = 1.0
    decoder: str = "threshold_bio"
    entity_threshold: float | dict[str, float] = 0.5
    entity_bias: float | dict[str, float] = 0.0

    def __post_init__(self) -> None:
        if self.activation not in ("sigmoid", "temperature_sigmoid", "hard_sigmoid"):
            raise ValueError("Use sigmoid, temperature_sigmoid or hard_sigmoid.")
        if self.decoder not in ("threshold_bio", "constrained_viterbi"):
            raise ValueError("Use threshold_bio or constrained_viterbi.")
        for name in ("temperature", "entity_threshold", "entity_bias"):
            setting = getattr(self, name)
            for value in setting.values() if isinstance(setting, dict) else [setting]:
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite.")
                if name == "temperature" and value <= 0:
                    raise ValueError("temperature must be positive.")
                if name == "entity_threshold" and not 0 <= value <= 1:
                    raise ValueError("entity_threshold must lie in [0,1].")
        if self.activation != "temperature_sigmoid" and self.temperature != 1.0:
            raise ValueError("temperature applies only to temperature_sigmoid.")
        if self.decoder == "constrained_viterbi" and self.entity_threshold != 0.5:
            raise ValueError("Viterbi uses entity_bias, not entity_threshold.")
        if self.decoder == "threshold_bio" and self.entity_bias != 0.0:
            raise ValueError("entity_bias applies only to constrained_viterbi.")


def entity_values(
    value: float | dict[str, float], labels: list[str], device: torch.device, default: float
) -> torch.Tensor:
    """Resolve named settings without assuming accidental head-index correspondence."""
    if isinstance(value, dict):
        if set(value) - set(labels):
            raise ValueError(f"Unknown entity settings: {sorted(set(value) - set(labels))[:10]}.")
        value = [value.get(label, default) for label in labels]

    return torch.as_tensor(value, device=device, dtype=torch.float32)


def binary_probabilities(
    logits: torch.Tensor, settings: BinaryDecoding, entity_labels: list[str]
) -> torch.Tensor:
    """Return two independent bounded channel scores, with no categorical softmax."""
    values = logits.float()
    if values.shape[-2:] != (len(entity_labels), 2) or not torch.isfinite(values).all():
        raise ValueError("Finite logits must end in [K,2] matching entity_labels.")
    if settings.activation == "temperature_sigmoid":
        temperature = entity_values(settings.temperature, entity_labels, values.device, 1.0)
        values = values / temperature[..., None]
    if settings.activation == "hard_sigmoid":
        return ((values + 3) / 6).clamp(0, 1)

    return torch.sigmoid(values)


def legal_log_scores(
    logits: torch.Tensor, settings: BinaryDecoding, entity_labels: list[str]
) -> torch.Tensor:
    """Bernoulli O/B/I log weights, excluding simultaneous B+I before normalization."""
    values = logits.float()
    if settings.activation == "hard_sigmoid":
        probability = binary_probabilities(values, settings, entity_labels).clamp(1e-7, 1 - 1e-7)
        positive, negative = probability.log(), torch.log1p(-probability)
    else:
        if values.shape[-2:] != (len(entity_labels), 2) or not torch.isfinite(values).all():
            raise ValueError("Finite logits must end in [K,2] matching entity_labels.")
        if settings.activation == "temperature_sigmoid":
            values = (
                values
                / entity_values(settings.temperature, entity_labels, values.device, 1)[..., None]
            )
        positive, negative = F.logsigmoid(values), F.logsigmoid(-values)

    return torch.stack(
        (
            negative.sum(-1),
            positive[..., 0] + negative[..., 1],
            negative[..., 0] + positive[..., 1],
        ),
        -1,
    )


def adapted_probabilities(
    logits: torch.Tensor, settings: BinaryDecoding, entity_labels: list[str]
) -> torch.Tensor:
    """Condition independent binary scores on legal local states; not the model activation."""
    scores = legal_log_scores(logits, settings, entity_labels)

    scores = scores - scores.amax(-1, keepdim=True)

    return (scores - scores.logsumexp(-1, keepdim=True)).exp()


def decode_binary(
    logits: torch.Tensor,
    word_mask: torch.Tensor,
    settings: BinaryDecoding,
    entity_labels: list[str],
) -> torch.Tensor:
    """Decode all entities together; orphan I starts B and B/I ties continue an open span."""
    if word_mask.shape != logits.shape[:2]:
        raise ValueError("word_mask must have shape [B,L].")
    if settings.decoder == "constrained_viterbi":
        scores = legal_log_scores(logits, settings, entity_labels)
        bias = entity_values(settings.entity_bias, entity_labels, logits.device, 0)
        scores = scores + torch.stack((torch.zeros_like(bias), bias, bias), -1)
        packed = pack_words(scores, word_mask)
        chain = ConstrainedChain({0: "O", 1: "B", 2: "I"}, len(entity_labels)).to(logits.device)
        chain.requires_grad_(False)

        return packed.restore(chain.decode(packed.emissions, packed.mask))
    probabilities = binary_probabilities(logits, settings, entity_labels)
    evidence = 1 - (1 - probabilities).prod(-1)
    threshold = entity_values(settings.entity_threshold, entity_labels, logits.device, 0.5)
    active = evidence >= threshold
    begins = probabilities[..., 0] > probabilities[..., 1]
    if logits.shape[1] == 0:
        return torch.full(logits.shape[:-1], -100, device=logits.device, dtype=torch.long)
    positions = torch.arange(logits.shape[1], device=logits.device)[None] + 1
    last_valid = torch.where(word_mask.bool(), positions, 0).cummax(dim=1).values
    previous_position = torch.cat((torch.zeros_like(last_valid[:, :1]), last_valid[:, :-1]), 1) - 1
    previous = active.gather(1, previous_position.clamp_min(0)[..., None].expand_as(active))
    previous = previous & (previous_position >= 0)[..., None]
    tags = torch.where(~active, 0, torch.where(~previous | begins, 1, 2))

    return tags.masked_fill(~word_mask.bool()[..., None], -100)
