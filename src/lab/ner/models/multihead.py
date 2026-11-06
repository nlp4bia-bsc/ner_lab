"""One Transformer and independent categorical BIO heads."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from lab.ner.encoding.tagging import validate_entity_labels


class MultiHeadForTokenClassification(nn.Module):
    """Shared backbone with logits [batch, length, heads, 3] and balanced head loss.

    Empty heads contribute no loss; an entirely masked batch returns a differentiable
    zero. The BIO class order is O, B, I; entity identity is in entity_labels.
    """

    def __init__(
        self,
        base_model: str,
        label2id: dict,
        id2label: dict,
        entity_labels: Sequence[str],
        dropout: float = 0.1,
        backbone_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.entity_labels = validate_entity_labels(entity_labels)
        if label2id != {"O": 0, "B": 1, "I": 2}:
            raise ValueError("Multihead requires label2id={'O': 0, 'B': 1, 'I': 2}.")
        if {int(k): v for k, v in id2label.items()} != {0: "O", 1: "B", 2: "I"}:
            raise ValueError("Multihead id2label disagrees with its BIO schema.")
        if backbone_config is None:
            self.config = AutoConfig.from_pretrained(base_model)
            self.backbone = AutoModel.from_pretrained(
                base_model, config=self.config, dtype=torch.float32
            )
        else:
            settings = dict(backbone_config)
            model_type = settings.pop("model_type")
            self.config = AutoConfig.for_model(model_type, **settings)
            self.backbone = AutoModel.from_config(self.config)
        if hasattr(self.backbone, "pooler"):
            self.backbone.pooler = None
        digest = hashlib.sha256(json.dumps(self.entity_labels).encode()).digest()
        self.register_buffer("head_catalogue_digest", torch.tensor(list(digest), dtype=torch.uint8))
        self.num_entity_types = len(self.entity_labels)
        self.num_labels = 3
        self.label2id = dict(label2id)
        self.id2label = {int(index): label for index, label in id2label.items()}
        self.config.label2id = self.label2id
        self.config.id2label = self.id2label
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_entity_types * 3)
        self._accepts_token_types = (
            "token_type_ids" in inspect.signature(self.backbone.forward).parameters
        )
        self.accepts_loss_kwargs = False

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None
    ) -> None:
        """Delegate activation checkpointing to the shared backbone."""
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._accepts_token_types and token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**inputs)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        logits = logits.reshape(*logits.shape[:2], self.num_entity_types, 3)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:-1]:
                raise ValueError(f"labels shape {labels.shape} != {logits.shape[:-1]}.")
            losses = F.cross_entropy(
                logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100, reduction="none"
            ).reshape_as(labels)
            counts = (labels != -100).sum(dim=(0, 1))
            head_loss = losses.sum(dim=(0, 1)) / counts.clamp_min(1)
            loss = head_loss.sum() / (counts > 0).sum().clamp_min(1)

        return TokenClassifierOutput(loss=loss, logits=logits)
