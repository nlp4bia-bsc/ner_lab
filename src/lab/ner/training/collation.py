"""Padding encoded rows into a batch, always on the right."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from lab.ner.encoding.rows import IGNORE_INDEX


@dataclass
class TokenClassificationCollator:
    """
    Pad a batch of encoded rows to its longest row, on the right whatever the tokenizer says.

    Span reconstruction reads a row's predictions from position 0, against the offsets
    stored in the row. Padding on the left, as many decoder tokenizers do by default,
    would shift every shorter row's logits past them; padding right keeps each row's
    positions where they are read.

    `labels` may hold one id per position, or a list of ids per position (one per head,
    for a multihead model); they are padded with `label_pad_token_id` to the same width
    as the inputs, `pad_to_multiple_of` included.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
        inputs = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]

        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding_side="right",
            return_tensors="pt",
        )

        width = batch["input_ids"].shape[1]
        padded = torch.full(
            (len(labels), width, *labels[0].shape[1:]), self.label_pad_token_id, dtype=torch.long
        )

        for index, row in enumerate(labels):
            padded[index, : len(row)] = row

        batch["labels"] = padded

        return batch
