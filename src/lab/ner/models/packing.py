"""Vectorized compaction and restoration of eligible CRF word positions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedWords:
    """Compact emissions with right padding and a mapping to original token positions."""

    emissions: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor
    original_length: int
    single_head: bool

    def restore(self, values: torch.Tensor, absent: float | int = -100) -> torch.Tensor:
        """Scatter compact decisions or distributions without inventing absent evidence."""
        trailing = values.shape[3:]
        expected = self.mask.shape
        if values.shape[:3] != expected:
            raise ValueError("Compact values do not match packed batch/time/head dimensions.")
        width = expected[1]
        slots = torch.arange(width, device=values.device)[None, :, None]
        positions = torch.where(self.mask, self.positions, self.original_length + slots)
        shape = (expected[0], self.original_length + width, expected[2], *trailing)
        padded = torch.full(shape, absent, device=values.device, dtype=values.dtype)
        indices = positions.reshape(*positions.shape, *((1,) * len(trailing))).expand_as(values)
        padded.scatter_(1, indices, values)
        output = padded[:, : self.original_length]

        return output[:, :, 0] if self.single_head else output


def pack_words(
    emissions: torch.Tensor, word_mask: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> PackedWords:
    """Compact structural word eligibility, independently of observed gold values."""
    single = emissions.ndim == 3
    if single:
        emissions = emissions.unsqueeze(2)
    if word_mask.ndim == 2:
        word_mask = word_mask[:, :, None].expand(emissions.shape[:-1])
    if emissions.ndim != 4 or word_mask.shape != emissions.shape[:-1]:
        raise ValueError("word_mask must align with [B,L,K] emission positions.")
    valid = word_mask.bool()
    if attention_mask is not None:
        if attention_mask.shape != emissions.shape[:2]:
            raise ValueError("attention_mask must have shape [B,L].")
        if (valid & ~attention_mask.bool()[:, :, None]).any():
            raise ValueError("A CRF word position cannot be Transformer padding.")
    batch, length, heads, classes = emissions.shape
    counts = valid.sum(1)
    width = int(counts.max().item()) if counts.numel() else 0
    ranks = valid.long().cumsum(1) - 1
    tokens = torch.arange(length, device=emissions.device)[None, :, None].expand_as(valid)
    slots = torch.where(valid, ranks, length + tokens)
    inverse = torch.zeros((batch, 2 * length, heads), device=emissions.device, dtype=torch.long)
    inverse.scatter_(1, slots, tokens)
    positions = inverse[:, :width]
    compact_mask = torch.arange(width, device=emissions.device)[None, :, None] < counts[:, None, :]
    indices = positions[..., None].expand(batch, width, heads, classes)
    compact = emissions.gather(1, indices).masked_fill(~compact_mask[..., None], 0)

    return PackedWords(compact, compact_mask, positions, length, single)


def pack_gold(tags: torch.Tensor, packed: PackedWords) -> torch.Tensor:
    """Reject missing supervision at eligible words instead of connecting unknown segments."""
    if tags.ndim == 2:
        tags = tags[:, :, None]
    if tags.shape != (packed.mask.shape[0], packed.original_length, packed.mask.shape[2]):
        raise ValueError("Gold shape disagrees with the structural word mask.")
    values = tags.gather(1, packed.positions)
    if ((values == -100) & packed.mask).any():
        bad = ((values == -100) & packed.mask).nonzero()[:10].tolist()
        raise ValueError(
            f"Unobserved eligible gold words at [batch,word,head] {bad}; "
            "partial annotation requires an explicit partial-supervision policy."
        )

    return values.masked_fill(~packed.mask, 0)
