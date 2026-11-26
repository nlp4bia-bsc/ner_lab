"""Compacting a window's labelled words into CRF chains, and writing results back."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lab.ner.encoding.rows import IGNORE_INDEX


@dataclass(frozen=True)
class PackedChains:
    """
    Each row's and head's chain positions, compacted to the left and padded on the right.

    `emissions` are `[batch, width, heads, tags]` and `tags`, `mask` and `positions`
    `[batch, width, heads]`, `width` being the longest chain in the batch. `tags` holds
    the gold tags, `O` past a chain's end; `positions` is where each chain step sits in
    the window, so results can be written back; `length` is the window's.
    """

    emissions: torch.Tensor
    tags: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor
    length: int


def pack_chains(emissions: torch.Tensor, labels: torch.Tensor | None) -> PackedChains:
    """
    Compact the positions whose label is not `IGNORE_INDEX` into one chain per row and head.

    `emissions` are `[batch, length, heads, tags]` and `labels` `[batch, length, heads]`.
    A position is in the chain exactly when its label is not `IGNORE_INDEX`: the first
    subword of each labelled word, never a special token, continuation subword,
    unlabelled flank or padding (D156). Order is kept, so a chain is the window's
    labelled words in reading order. A row and head with no labelled word gives an
    empty chain. Without `labels` there is no chain to run, so it raises.
    """
    if labels is None:
        raise ValueError(
            "A CRF runs over the window's labelled words, which labels mark: pass the "
            f"rows' labels, {IGNORE_INDEX} everywhere but each word's first subword, as the "
            "encoder builds them for prediction too."
        )

    if labels.shape != emissions.shape[:-1]:
        raise ValueError(
            f"labels shaped {tuple(labels.shape)} do not match emissions "
            f"{tuple(emissions.shape)}; pass one label per position and head."
        )

    batch_size, length, num_heads, num_tags = emissions.shape
    in_chain = labels != IGNORE_INDEX
    chain_lengths = in_chain.sum(dim=1)
    width = int(chain_lengths.max()) if chain_lengths.numel() else 0

    rows, positions, heads = in_chain.nonzero(as_tuple=True)
    slots = (in_chain.long().cumsum(dim=1) - 1)[rows, positions, heads]

    packed_emissions = emissions.new_zeros(batch_size, width, num_heads, num_tags)
    packed_emissions[rows, slots, heads] = emissions[rows, positions, heads]

    packed_tags = labels.new_zeros(batch_size, width, num_heads)
    packed_tags[rows, slots, heads] = labels[rows, positions, heads]

    packed_positions = labels.new_zeros(batch_size, width, num_heads)
    packed_positions[rows, slots, heads] = positions

    steps = torch.arange(width, device=labels.device)
    mask = steps[None, :, None] < chain_lengths[:, None, :]

    return PackedChains(
        emissions=packed_emissions,
        tags=packed_tags,
        mask=mask,
        positions=packed_positions,
        length=length,
    )


def unpack_chains(values: torch.Tensor, packed: PackedChains, fill: float) -> torch.Tensor:
    """
    Write per-step chain values back to their window positions, `fill` everywhere else.

    `values` are `[batch, width, heads, ...]`, one entry per chain step — Viterbi tags
    or marginals — and come back `[batch, length, heads, ...]`.
    """
    batch_size, _, num_heads, *trailing = values.shape
    output = values.new_full((batch_size, packed.length, num_heads, *trailing), fill)

    rows, slots, heads = packed.mask.nonzero(as_tuple=True)
    output[rows, packed.positions[rows, slots, heads], heads] = values[rows, slots, heads]

    return output
