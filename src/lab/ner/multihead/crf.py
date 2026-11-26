"""A Transformer shared by one O/B/I head per entity type, each with its own CRF chain."""

from __future__ import annotations

from typing import Any

import torch
from transformers.modeling_outputs import TokenClassifierOutput

from lab.ner.encoding.tagging import build_label_vocabulary
from lab.ner.models.chain import ConstrainedChain
from lab.ner.models.packing import pack_chains, unpack_chains
from lab.ner.multihead.model import MultiHeadForTokenClassification


class MultiHeadCRFForTokenClassification(MultiHeadForTokenClassification):
    """
    The multihead model with a linear-chain CRF over each head's tags.

    Backbone, classifier and head catalogue are `MultiHeadForTokenClassification`'s, so
    `logits` are the same `[batch, length, heads, 3]` emissions. Each head is a chain
    over the window's labelled words with its own start, transition and end scores, all
    computed at once by one `ConstrainedChain` (D155, D160). The loss is each head's
    negative log-likelihood per labelled word, averaged over the heads with any (D158),
    the rule `head_balanced_loss` applies to cross-entropy, so the two multiheads weigh
    entity types alike.

    `decode_from_emissions` and `marginals_from_emissions` take the rows' `labels` for
    where each chain runs (D156) and return `[batch, length, heads]` tags, `-100` off
    the chain, and `[batch, length, heads, 3]` log-marginals, zero off the chain.
    """

    def __init__(
        self,
        base_model: str,
        entity_labels: list[str],
        dropout: float = 0.1,
        constrain_transitions: bool = True,
        backbone_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            base_model=base_model,
            entity_labels=entity_labels,
            dropout=dropout,
            backbone_config=backbone_config,
        )

        self.constrain_transitions = constrain_transitions
        self.chain = ConstrainedChain(
            build_label_vocabulary(self.entity_labels[0])["id2label"],
            num_heads=len(self.entity_labels),
            constrain_transitions=constrain_transitions,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ) -> TokenClassifierOutput:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if labels is not None:
            packed = pack_chains(outputs.logits, labels)
            outputs.loss = self.chain.loss(packed.emissions, packed.tags, packed.mask)

        return outputs

    def decode_from_emissions(
        self, emissions: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
        """Each head's Viterbi tag at every position, `[batch, length, heads]`, `-100` off it."""
        packed = pack_chains(emissions, labels)
        tags = self.chain.viterbi(packed.emissions, packed.mask)

        return unpack_chains(tags, packed, fill=-100)

    def marginals_from_emissions(
        self, emissions: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Each head's log-probability of each tag at every position, given its whole chain.

        `[batch, length, heads, 3]` in float32; a position off a head's chain is zero
        throughout, a filler no reader uses (D161).
        """
        packed = pack_chains(emissions, labels)
        log_marginals = self.chain.log_marginals(packed.emissions, packed.mask)

        return unpack_chains(log_marginals, packed, fill=0.0)
