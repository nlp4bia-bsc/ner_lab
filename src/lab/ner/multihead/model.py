"""A Transformer shared by one O/B/I classification head per entity type."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from lab.ner.encoding.rows import IGNORE_INDEX

BIO_CLASSES = 3


class MultiHeadForTokenClassification(nn.Module):
    """
    A pretrained encoder and one independent O/B/I head per entity type.

    `logits` come back shaped `[batch, length, heads, 3]`, head `k` classifying
    `entity_labels[k]`, so a token can be inside entities of several types at once.
    The loss is each head's cross-entropy over its labelled positions, averaged
    over the heads that have any in the batch: every type weighs the same however
    frequent it is.

    The backbone is built from `backbone_config` when given, with no weights, which
    is how a saved model is rebuilt before its fine-tuned weights are loaded; without
    it the pretrained weights at `base_model` are loaded, in float32 whatever the
    checkpoint's dtype, as `build_linear_model` explains. A buffer holds a digest of
    `entity_labels`, so weights loaded under a reordered catalogue fail to load.
    """

    def __init__(
        self,
        base_model: str,
        entity_labels: list[str],
        dropout: float = 0.1,
        backbone_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.entity_labels = list(entity_labels)

        if backbone_config is None:
            self.config = AutoConfig.from_pretrained(base_model)
            self.backbone = AutoModel.from_pretrained(
                base_model, config=self.config, dtype=torch.float32
            )
        else:
            settings = dict(backbone_config)
            self.config = AutoConfig.for_model(settings.pop("model_type"), **settings)
            self.backbone = AutoModel.from_config(self.config, dtype=torch.float32)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, len(self.entity_labels) * BIO_CLASSES)
        self.register_buffer("catalogue_digest", catalogue_digest(self.entity_labels))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ) -> TokenClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        logits = logits.reshape(*logits.shape[:2], len(self.entity_labels), BIO_CLASSES)

        return TokenClassifierOutput(
            loss=None if labels is None else head_balanced_loss(logits, labels),
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def head_balanced_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy averaged within each head, then over the heads with a labelled position.

    `labels` is `[batch, length, heads]`, `IGNORE_INDEX` where no loss is computed. A
    batch with no labelled position at all gives a zero that still carries a gradient.
    """
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            f"labels shaped {tuple(labels.shape)} do not match logits {tuple(logits.shape)}; "
            "rows must be encoded with the model's entity_labels, in its order."
        )

    losses = F.cross_entropy(
        logits.reshape(-1, BIO_CLASSES),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(labels)

    labelled = (labels != IGNORE_INDEX).sum(dim=(0, 1))
    head_losses = losses.sum(dim=(0, 1)) / labelled.clamp_min(1)

    return head_losses.sum() / (labelled > 0).sum().clamp_min(1)


def catalogue_digest(entity_labels: list[str]) -> torch.Tensor:
    """The SHA-256 of a head catalogue, in order, as a byte tensor a state dict can carry."""
    digest = hashlib.sha256(json.dumps(list(entity_labels)).encode("utf-8")).digest()

    return torch.tensor(list(digest), dtype=torch.uint8)
