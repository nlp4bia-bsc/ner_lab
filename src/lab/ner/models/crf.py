"""Transformer encoder with a CRF token-classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from lab.ner.models.bio import bio_constraint_masks, normalize_id2label, normalize_label2id

try:
    from torchcrf import CRF
except ImportError as error:
    raise ImportError(
        "pytorch-crf is required for the CRF architecture. Install it with: pip install pytorch-crf"
    ) from error

CONSTRAINT_VALUE = -10000.0


class CRFForTokenClassification(nn.Module):
    """
    A pretrained encoder, a linear emission layer, and a CRF over BIO tags.

    `forward` returns emissions as `logits` and the negative CRF log-likelihood as
    `loss`, so a standard `Trainer` drives it unchanged; `decode` runs Viterbi.
    With `constrain_transitions`, structurally impossible BIO transitions are
    pinned to a large negative score and held there — gradient hooks zero their
    updates, and the pin is reapplied before every loss and decode, since
    `load_state_dict` would otherwise restore learned values over them.
    """

    def __init__(
        self,
        base_model: str,
        label2id: dict,
        id2label: dict,
        dropout: float = 0.1,
        ignore_index: int = -100,
        constrain_transitions: bool = True,
        constraint_value: float = CONSTRAINT_VALUE,
    ) -> None:
        super().__init__()

        self.label2id = normalize_label2id(label2id)
        self.id2label = normalize_id2label(id2label)
        self.num_labels = len(self.id2label)
        self.ignore_index = ignore_index
        self.constrain_transitions = constrain_transitions
        self.constraint_value = constraint_value

        self.config = AutoConfig.from_pretrained(
            base_model,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.backbone = AutoModel.from_pretrained(base_model, config=self.config)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

        self.outside_label_id = self.label2id.get("O", 0)

        invalid_start, invalid_transitions = bio_constraint_masks(self.id2label)

        self.register_buffer("invalid_start_transitions", torch.tensor(invalid_start))
        self.register_buffer("invalid_transitions", torch.tensor(invalid_transitions))

        if self.constrain_transitions:
            self._register_transition_gradient_hooks()
            self.apply_transition_constraints()

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
            mask = torch.ones_like(labels, dtype=torch.bool) if attention_mask is None \
                else attention_mask.bool()

            crf_labels = labels.masked_fill(labels == self.ignore_index, self.outside_label_id)

            if self.constrain_transitions:
                self.apply_transition_constraints()

            loss = -self.crf(emissions=emissions, tags=crf_labels, mask=mask, reduction="mean")

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
        **kwargs,
    ) -> list[list[int]]:
        """Viterbi-decode the best tag path for a batch of inputs."""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None,
            **kwargs,
        )

        return self.decode_from_emissions(outputs.logits, attention_mask)

    def decode_from_emissions(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Viterbi-decode emissions that have already been computed."""
        if attention_mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            mask = attention_mask.bool()

        if self.constrain_transitions:
            self.apply_transition_constraints()

        return self.crf.decode(emissions=emissions, mask=mask)

    def apply_transition_constraints(self) -> None:
        """Pin structurally invalid transition scores to `constraint_value`."""
        with torch.no_grad():
            self.crf.start_transitions.data = self.crf.start_transitions.data.masked_fill(
                self.invalid_start_transitions, self.constraint_value
            )
            self.crf.transitions.data = self.crf.transitions.data.masked_fill(
                self.invalid_transitions, self.constraint_value
            )

    def _register_transition_gradient_hooks(self) -> None:
        def zero_invalid_start(gradient: torch.Tensor) -> torch.Tensor:
            return gradient.masked_fill(self.invalid_start_transitions, 0.0)

        def zero_invalid_transitions(gradient: torch.Tensor) -> torch.Tensor:
            return gradient.masked_fill(self.invalid_transitions, 0.0)

        self.crf.start_transitions.register_hook(zero_invalid_start)
        self.crf.transitions.register_hook(zero_invalid_transitions)


def decoded_paths_to_logits(
    decoded_paths: list[list[int]],
    sequence_length: int,
    num_labels: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Re-encode Viterbi paths as one-hot-ish logits.

    Decoding returns tag ids, but the evaluation stack reads `argmax` over a
    logit tensor. Rebuilding that shape here keeps CRF and non-CRF models on one
    metrics path instead of forking it.
    """
    decoded_ids = torch.zeros((len(decoded_paths), sequence_length), dtype=torch.long, device=device)

    for row, path in enumerate(decoded_paths):
        length = min(len(path), sequence_length)
        decoded_ids[row, :length] = torch.tensor(path[:length], dtype=torch.long, device=device)

    logits = torch.full(
        (len(decoded_paths), sequence_length, num_labels),
        fill_value=CONSTRAINT_VALUE,
        dtype=torch.float32,
        device=device,
    )

    return logits.scatter_(dim=-1, index=decoded_ids.unsqueeze(-1), value=-CONSTRAINT_VALUE)
