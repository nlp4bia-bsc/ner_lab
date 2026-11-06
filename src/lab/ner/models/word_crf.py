"""Shared Transformer emissions with compact, independent word-level CRF heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput

from lab.ner.models.bio import bio_entity
from lab.ner.models.chain import ConstrainedChain
from lab.ner.models.multihead import MultiHeadForTokenClassification
from lab.ner.models.packing import pack_gold, pack_words


@dataclass(frozen=True)
class StructuredPrediction:
    """Original-position Viterbi tags and optional log marginals; absence is explicit."""

    tags: torch.Tensor
    log_marginals: torch.Tensor | None = None


def decision_logits(tags: torch.Tensor, states: int = 3) -> torch.Tensor:
    """Compatibility transport for APIs requiring argmax arrays, never confidence."""
    result = torch.full((*tags.shape, states), -10000.0, device=tags.device)
    return result.scatter(-1, tags.clamp_min(0)[..., None], 10000.0)


class WordCRFForTokenClassification(MultiHeadForTokenClassification):
    """One encoder shared by classic or Multi-Head compact BIO2 chains."""

    def __init__(
        self,
        base_model: str,
        label2id: dict,
        id2label: dict,
        entity_labels: list[str] | None = None,
        dropout: float = 0.1,
        backbone_config: dict | None = None,
        constrain_transitions: bool = True,
        aux_ce_weight: float = 0.0,
        head_weights: dict[str, float] | None = None,
        tag_scheme: str = "BIO2",
        crf_version: int = 2,
        loss_type: str = "crf",
        loss_reduction: str = "word_mean",
    ) -> None:
        if tag_scheme != "BIO2" or crf_version != 2:
            raise ValueError("The word-level CRF supports crf_version=2 and tag_scheme=BIO2 only.")
        if not math.isfinite(aux_ce_weight) or aux_ce_weight < 0:
            raise ValueError("aux_ce_weight must be finite and non-negative.")
        if loss_type not in ("crf", "cross_entropy") or loss_reduction != "word_mean":
            raise ValueError("Use loss_type=crf/cross_entropy and loss_reduction=word_mean.")
        if loss_type == "cross_entropy" and aux_ce_weight:
            raise ValueError("aux_ce_weight applies only to loss_type=crf.")
        self.loss_type = loss_type
        self.multihead = entity_labels is not None
        labels = (
            entity_labels if self.multihead else [bio_entity(id2label.get(1, id2label.get("1")))]
        )
        if not self.multihead and label2id != {"O": 0, f"B-{labels[0]}": 1, f"I-{labels[0]}": 2}:
            raise ValueError("Classic word CRF requires one entity with O=0, B=1, I=2.")
        super().__init__(
            base_model,
            {"O": 0, "B": 1, "I": 2},
            {0: "O", 1: "B", 2: "I"},
            labels,
            dropout,
            backbone_config,
        )
        self.chain = ConstrainedChain({0: "O", 1: "B", 2: "I"}, len(labels), constrain_transitions)
        if loss_type == "cross_entropy":
            self.chain.requires_grad_(False)
        weights = head_weights or {}
        if set(weights) - set(labels):
            raise ValueError("head_weights contains unknown entity names.")
        values = torch.tensor([weights.get(label, 1.0) for label in labels], dtype=torch.float32)
        if not torch.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("head_weights must be finite and nonnegative with positive total.")
        self.register_buffer("loss_head_weights", values)
        self.aux_ce_weight = aux_ce_weight
        self.constrain_transitions = constrain_transitions
        if not self.multihead:
            self.label2id = dict(label2id)
            self.id2label = dict(id2label)
            self.config.label2id = label2id
            self.config.id2label = id2label

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        crf_word_mask: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        emissions = super().forward(input_ids, attention_mask, token_type_ids).logits
        logits = emissions if self.multihead else emissions[:, :, 0]
        loss = None
        if labels is not None:
            if crf_word_mask is None:
                raise ValueError("Supply structural crf_word_mask independently of labels.")
            packed = pack_words(logits, crf_word_mask, attention_mask)
            gold = pack_gold(labels, packed)
            if self.loss_type == "crf":
                loss = self.chain(packed.emissions, gold, packed.mask, self.loss_head_weights)
            if self.aux_ce_weight or self.loss_type == "cross_entropy":
                local = F.cross_entropy(
                    packed.emissions.float().reshape(-1, 3),
                    gold.masked_fill(~packed.mask, -100).reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                )
                counts = packed.mask.sum((0, 1))
                local = local.reshape_as(gold).sum((0, 1)) / counts.clamp_min(1)
                active = self.loss_head_weights * (counts > 0)
                local_loss = (local * active).sum() / active.sum().clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                loss = (
                    local_loss
                    if self.loss_type == "cross_entropy"
                    else loss + self.aux_ce_weight * local_loss
                )
        return TokenClassifierOutput(loss=loss, logits=logits)

    def structured_from_emissions(
        self,
        emissions: torch.Tensor,
        crf_word_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        probabilities: bool = False,
    ) -> StructuredPrediction:
        """Reuse emissions for Viterbi and, only when requested, forward-backward."""
        packed = pack_words(emissions, crf_word_mask, attention_mask)
        tags = packed.restore(self.chain.decode(packed.emissions, packed.mask))
        marginals = None
        if probabilities:
            marginals = packed.restore(
                self.chain.log_marginals(packed.emissions, packed.mask), absent=-torch.inf
            )
        return StructuredPrediction(tags, marginals)

    def decode_from_emissions(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        crf_word_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode without rerunning the backbone; attention alone cannot define a chain."""
        if crf_word_mask is None:
            raise ValueError("Word-level decoding requires explicit crf_word_mask.")
        return self.structured_from_emissions(emissions, crf_word_mask, attention_mask).tags

    def marginals_from_emissions(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        crf_word_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return log marginals with -inf at absent positions, using the explicit word mask."""
        if crf_word_mask is None:
            raise ValueError("Word-level marginals require explicit crf_word_mask.")
        return self.structured_from_emissions(
            emissions, crf_word_mask, attention_mask, probabilities=True
        ).log_marginals
