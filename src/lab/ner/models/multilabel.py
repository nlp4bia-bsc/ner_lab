"""A shared Transformer with independent binary B/I outputs for every entity."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
from transformers.modeling_outputs import TokenClassifierOutput

from lab.ner.models.binary_decoding import BinaryDecoding, entity_values
from lab.ner.models.binary_loss import BinaryLoss, binary_loss
from lab.ner.models.multihead import MultiHeadForTokenClassification


class MultilabelForTokenClassification(MultiHeadForTokenClassification):
    """One backbone and vectorized 2K projection; O is implicit, not a competing class."""

    label_schema = "multilabel"
    _output_channels = 2

    def __init__(
        self,
        base_model: str,
        label2id: dict,
        id2label: dict,
        entity_labels: list[str],
        dropout: float = 0.1,
        backbone_config: dict | None = None,
        loss: str = "bce",
        pos_weight_policy: str = "none",
        pos_weight: dict[str, list[float]] | None = None,
        weight_smoothing: float = 1.0,
        max_pos_weight: float = 20.0,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        training_temperature: float = 1.0,
        activation: str = "sigmoid",
        temperature: float | dict[str, float] = 1.0,
        decoder: str = "threshold_bio",
        entity_threshold: float | dict[str, float] = 0.5,
        entity_bias: float | dict[str, float] = 0.0,
    ) -> None:
        if not math.isfinite(training_temperature) or training_temperature <= 0:
            raise ValueError("training_temperature must be positive and finite.")
        if loss == "weighted_bce" and pos_weight_policy == "none" and pos_weight is None:
            pos_weight_policy = "train"
        options = BinaryLoss(
            loss,
            pos_weight_policy,
            weight_smoothing,
            max_pos_weight,
            focal_gamma,
            focal_alpha,
            gamma_pos,
            gamma_neg,
            clip,
        )
        decoding = BinaryDecoding(activation, temperature, decoder, entity_threshold, entity_bias)
        super().__init__(base_model, label2id, id2label, entity_labels, dropout, backbone_config)
        self.num_labels = 2
        self.config.label_schema = "multilabel"
        self.config.binary_channels = {"B": 0, "I": 1}
        self.config.label2id = {"B": 0, "I": 1}
        self.config.id2label = {0: "B", 1: "I"}
        self.loss_options = options
        self.decoding = decoding
        self.training_temperature = training_temperature
        self.requires_weight_estimation = loss == "weighted_bce" and pos_weight is None
        if pos_weight is not None and (
            loss != "weighted_bce" or set(pos_weight) != set(entity_labels)
        ):
            raise ValueError("pos_weight must name every entity and requires weighted_bce.")
        weights = (
            torch.tensor([pos_weight[label] for label in entity_labels])
            if pos_weight
            else torch.ones(len(entity_labels), 2)
        )
        if (
            weights.shape != (len(entity_labels), 2)
            or not torch.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("pos_weight must contain finite positive B/I pairs for every entity.")
        self.register_buffer("positive_weights", weights.float())
        for name, default in (("temperature", 1), ("entity_threshold", 0.5), ("entity_bias", 0)):
            entity_values(getattr(decoding, name), self.entity_labels, weights.device, default)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        word_mask: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        logits = self._project(input_ids, attention_mask, token_type_ids)
        logits = logits.reshape(*logits.shape[:2], len(self.entity_labels), 2)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape:
                raise ValueError("Multilabel labels must match logits [B,L,K,2].")
            if (
                word_mask is not None
                and attention_mask is not None
                and (word_mask.bool() & ~attention_mask.bool()).any()
            ):
                raise ValueError("Eligible words cannot be Transformer padding.")
            if self.requires_weight_estimation:
                raise ValueError(
                    "Estimate pos_weight from encoded training rows before training weighted_bce."
                )
            if word_mask is None or word_mask.shape != logits.shape[:2]:
                raise ValueError("Multilabel training requires structural word_mask [B,L].")
            if ((labels != -100).any((-1, -2)) & ~word_mask.bool()).any():
                raise ValueError(
                    "Gold supervision cannot include structurally ineligible positions."
                )
            weights = self.positive_weights if self.loss_options.loss == "weighted_bce" else None
            loss = binary_loss(
                logits.float() / self.training_temperature, labels, self.loss_options, weights
            )

        return TokenClassifierOutput(loss=loss, logits=logits)

    def architecture_arguments(self) -> dict:
        """All explicit constructor options required to reproduce learned outputs and decisions."""
        weights = (
            dict(zip(self.entity_labels, self.positive_weights.tolist()))
            if self.loss_options.loss == "weighted_bce"
            else None
        )

        return {
            **asdict(self.loss_options),
            **asdict(self.decoding),
            "pos_weight": weights,
            "training_temperature": self.training_temperature,
            "dropout": self.dropout.p,
            "entity_labels": self.entity_labels,
        }
