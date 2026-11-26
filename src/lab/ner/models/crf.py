"""Transformer encoder with a CRF over the BIO tags of a window's labelled words."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from lab.ner.encoding.rows import IGNORE_INDEX
from lab.ner.models.bio import normalize_id2label, normalize_label2id
from lab.ner.models.chain import ConstrainedChain
from lab.ner.models.packing import pack_chains, unpack_chains

DECISION_LOGIT = 10000.0


class CRFForTokenClassification(nn.Module):
    """
    A pretrained encoder, a linear emission layer, and a CRF over BIO tags.

    The chain runs over the window's labelled words — the positions whose label is not
    `ignore_index`, each word's first subword — not over its tokens (D154, D156): the
    encoder still reads every token, so each word's emission carries its subwords'
    context, but transitions link one word to the next. `forward` returns the emissions
    as `logits` and, given `labels`, the chain's negative log-likelihood per labelled
    word as `loss` (D158), so a standard `Trainer` drives it unchanged.

    `decode_from_emissions` and `marginals_from_emissions` take `labels` too, for where
    the chain runs; prediction rows carry them like training rows, `O` where the gold is
    unknown. With `constrain_transitions` the BIO transitions no well-formed sequence
    takes are excluded outright (D155). The backbone loads in float32 whatever its
    checkpoint's dtype, as `build_linear_model` explains.
    """

    def __init__(
        self,
        base_model: str,
        label2id: dict,
        id2label: dict,
        dropout: float = 0.1,
        ignore_index: int = IGNORE_INDEX,
        constrain_transitions: bool = True,
    ) -> None:
        super().__init__()

        self.label2id = normalize_label2id(label2id)
        self.id2label = normalize_id2label(id2label)
        self.num_labels = len(self.id2label)
        self.ignore_index = ignore_index
        self.constrain_transitions = constrain_transitions

        self.config = AutoConfig.from_pretrained(
            base_model,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.backbone = AutoModel.from_pretrained(
            base_model, config=self.config, dtype=torch.float32
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.chain = ConstrainedChain(
            self.id2label, num_heads=1, constrain_transitions=constrain_transitions
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))

        loss = None

        if labels is not None:
            packed = self._pack(emissions, labels)
            loss = self.chain.loss(packed.emissions, packed.tags, packed.mask)

        return TokenClassifierOutput(
            loss=loss,
            logits=emissions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ) -> torch.Tensor:
        """Viterbi-decode the best tag path for a batch of inputs, as `decode_from_emissions`."""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        return self.decode_from_emissions(outputs.logits, labels)

    def decode_from_emissions(
        self, emissions: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
        """
        The Viterbi tag of every position, `[batch, length]`, `ignore_index` off the chain.
        """
        packed = self._pack(emissions, labels)
        tags = self.chain.viterbi(packed.emissions, packed.mask)

        return unpack_chains(tags, packed, fill=self.ignore_index).squeeze(2)

    def marginals_from_emissions(
        self, emissions: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Each position's log-probability of each tag given the whole chain, `[batch, length, tags]`.

        Forward-backward over the emissions and the learned start, transition and end
        scores, in float32. Log rather than plain probabilities, so a softmax over the
        result returns the marginals unchanged and they can stand where logits are
        expected. A position off the chain is zero throughout, a filler no reader uses
        (D161).
        """
        packed = self._pack(emissions, labels)
        log_marginals = self.chain.log_marginals(packed.emissions, packed.mask)

        return unpack_chains(log_marginals, packed, fill=0.0).squeeze(2)

    def _pack(self, emissions: torch.Tensor, labels: torch.Tensor | None):
        if labels is not None:
            labels = labels.masked_fill(labels == self.ignore_index, IGNORE_INDEX).unsqueeze(2)

        return pack_chains(emissions.unsqueeze(2), labels)


def tags_to_logits(tags: torch.Tensor, num_labels: int) -> torch.Tensor:
    """
    Re-encode decoded tags as one-hot logits over a last axis of `num_labels`.

    Decoding returns tag ids, but the evaluation stack reads `argmax` over logits.
    Rebuilding that shape here keeps CRF and non-CRF models on one metrics path. Any
    shape of `tags` works — `[batch, length]`, or `[batch, length, heads]` for a
    multihead CRF; a position off the chain reads as tag 0, and no reader uses it.
    """
    logits = torch.full(
        (*tags.shape, num_labels), -DECISION_LOGIT, dtype=torch.float32, device=tags.device
    )

    return logits.scatter_(-1, tags.clamp_min(0).unsqueeze(-1), DECISION_LOGIT)
