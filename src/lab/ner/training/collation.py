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
        inputs = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]
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
) -> MultiHeadCollator | DataCollatorForTokenClassification:
    """Choose padding by the model's target representation."""
    collator = (
        MultiHeadCollator
        if hasattr(model, "entity_labels")
        else (DataCollatorForTokenClassification)
    )

    return collator(
        tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of, label_pad_token_id=ignore_index
    )
