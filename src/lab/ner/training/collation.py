"""Padding shared inputs and multidimensional BIO targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase


@dataclass
class MultiHeadCollator:
    """Pad [length, heads] labels on the tokenizer's padding side."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        inputs = [
            {k: v for k, v in feature.items() if k not in ("labels", "crf_word_mask")}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            inputs, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        width = batch["input_ids"].shape[1]
        heads = len(labels[0][0])
        if any(len(vector) != heads for row in labels for vector in row):
            raise ValueError("All label vectors must have the same number of heads.")
        padded = torch.full((len(labels), width, heads), self.label_pad_token_id, dtype=torch.long)
        for index, row in enumerate(labels):
            start = width - len(row) if self.tokenizer.padding_side == "left" else 0
            padded[index, start : start + len(row)] = torch.tensor(row)
        batch["labels"] = padded

        return batch


def classification_collator(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    pad_to_multiple_of: int | None = 8,
    ignore_index: int = -100,
) -> (
    MultiHeadCollator | DataCollatorForTokenClassification | WordChainCollator | MultilabelCollator
):
    """Choose padding by the model's target representation."""
    if getattr(model, "label_schema", None) == "multilabel":
        return MultilabelCollator(tokenizer, pad_to_multiple_of, ignore_index)
    if hasattr(model, "chain"):
        return WordChainCollator(tokenizer, pad_to_multiple_of, ignore_index, model.multihead)
    collator = (
        MultiHeadCollator
        if hasattr(model, "entity_labels")
        else (DataCollatorForTokenClassification)
    )

    return collator(
        tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of, label_pad_token_id=ignore_index
    )


@dataclass
class WordChainCollator:
    """Pad eligibility separately from token attention and gold supervision."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8
    label_pad_token_id: int = -100
    multihead: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if any("crf_word_mask" not in feature for feature in features):
            raise ValueError("CRF rows require crf_word_mask; encode with include_word_mask=True.")
        masks = [feature["crf_word_mask"] for feature in features]
        clean = [
            {key: value for key, value in feature.items() if key != "crf_word_mask"}
            for feature in features
        ]
        collator = MultiHeadCollator if self.multihead else DataCollatorForTokenClassification
        batch = collator(
            self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
            label_pad_token_id=self.label_pad_token_id,
        )(clean)
        mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        for index, values in enumerate(masks):
            if len(values) != len(features[index]["input_ids"]):
                raise ValueError("crf_word_mask length disagrees with input_ids.")
            start = mask.shape[1] - len(values) if self.tokenizer.padding_side == "left" else 0
            mask[index, start : start + len(values)] = torch.as_tensor(values, dtype=torch.bool)
        batch["crf_word_mask"] = mask
        return batch


@dataclass
class MultilabelCollator:
    """Pad binary [length,entity,channel] targets and structural word eligibility."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if any("word_mask" not in feature for feature in features):
            raise ValueError("Binary rows require word_mask; use Encoder(label_schema=multilabel).")
        supervised = ["labels" in feature for feature in features]
        if any(supervised) != all(supervised):
            raise ValueError("Do not mix supervised and unlabelled rows in one batch.")
        batch = self.tokenizer.pad(
            [
                {key: value for key, value in feature.items() if key not in ("labels", "word_mask")}
                for feature in features
            ],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        width = batch["input_ids"].shape[1]
        mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        labels = None
        if all(supervised):
            first = torch.as_tensor(features[0]["labels"])
            if first.ndim != 3 or first.shape[-1] != 2:
                raise ValueError("Multilabel targets require [L,K,2].")
            labels = torch.full(
                (len(features), width, *first.shape[1:]),
                self.label_pad_token_id,
                dtype=torch.float32,
            )
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            eligible = torch.as_tensor(feature["word_mask"], dtype=torch.bool)
            if len(eligible) != length:
                raise ValueError("word_mask and input_ids must have the same length.")
            start = width - length if self.tokenizer.padding_side == "left" else 0
            mask[index, start : start + length] = eligible
            if labels is not None:
                values = torch.as_tensor(feature["labels"])
                if values.shape != (length, *labels.shape[2:]):
                    raise ValueError("Inconsistent binary target length, catalogue or channels.")
                labels[index, start : start + length] = values
        if labels is not None:
            batch["labels"] = labels
        batch["word_mask"] = mask

        return batch
