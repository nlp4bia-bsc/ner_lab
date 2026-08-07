"""The variant dimension: every (checkpoint, windowing) combination, encoded once."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from ner_lab.encoding.encoder import Encoder, WindowStrategy
from ner_lab.encoding.overlaps import OverlapPolicy
from ner_lab.training.assessment import architecture_name, encode_partition


@dataclass(frozen=True)
class EncodedVariant:
    """One variant's windowed rows, plus everything a trial needs to train on them."""

    checkpoint: str
    strategy: str | WindowStrategy
    context_tokens: int | None
    max_length: int
    tokenizer: Any
    label2id: dict[str, int]
    id2label: dict[int, str]
    train_rows: pd.DataFrame
    validation_rows: pd.DataFrame


def variant_key(
    checkpoint: str,
    strategy: str | WindowStrategy,
    context_tokens: int | None,
    max_length: int,
) -> str:
    """The readable string one variant is sampled and recorded as."""
    strategy_name = strategy if isinstance(strategy, str) else architecture_name(strategy)

    return f"{Path(checkpoint).name}|{strategy_name}|{context_tokens}|{max_length}"


def build_variants(
    checkpoints: Sequence[str],
    strategies: Sequence[str | WindowStrategy] = ("greedy",),
    context_tokens: Sequence[int] | None = None,
    max_lengths: Sequence[int] = (256,),
) -> dict[str, dict[str, Any]]:
    """
    Every (checkpoint, strategy, context_tokens, max_length) combination, keyed by
    `variant_key`.

    These form one search dimension rather than four: the checkpoint's tokenizer
    decides the windowing, so they cannot be sampled independently.
    `context_tokens` only varies for the context strategy; every other strategy
    gets a single `None` entry per (checkpoint, max_length).
    """
    if "context" in strategies and not context_tokens:
        raise ValueError("context_tokens is required when the context strategy is searched.")

    if context_tokens and "context" not in strategies:
        raise ValueError("context_tokens only applies when the context strategy is searched.")

    variants: dict[str, dict[str, Any]] = {}

    for checkpoint in checkpoints:
        for max_length in max_lengths:
            for strategy in strategies:
                context_options = list(context_tokens) if strategy == "context" else [None]

                for context in context_options:
                    key = variant_key(checkpoint, strategy, context, max_length)

                    if key in variants:
                        raise ValueError(
                            f"Duplicate variant {key!r} — two checkpoints share the "
                            "name the key is built from."
                        )

                    variants[key] = {
                        "checkpoint": checkpoint,
                        "strategy": strategy,
                        "context_tokens": context,
                        "max_length": max_length,
                    }

    return variants


def describe_variants(variants: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The manifest form of the variants, with callable strategies named."""
    return {
        key: {
            **specification,
            "strategy": (
                specification["strategy"]
                if isinstance(specification["strategy"], str)
                else architecture_name(specification["strategy"])
            ),
        }
        for key, specification in variants.items()
    }


def encode_variants(
    partitions: dict[str, Path | list[Path]],
    variants: dict[str, dict[str, Any]],
    target_label: str,
    language: str,
    overlap_policy: OverlapPolicy = "merge_same_label_then_keep_longest",
    min_sentence_tokens: int = 4,
) -> dict[str, EncodedVariant]:
    """
    Window the train and validation partitions once per variant, before any trial.

    Every trial that samples a variant reuses these frames, so the corpus is
    windowed `len(variants)` times per sweep rather than once per trial.
    Tokenizers are loaded once per checkpoint and shared across its variants.
    """
    tokenizers: dict[str, Any] = {}
    encoded: dict[str, EncodedVariant] = {}

    for key, specification in variants.items():
        checkpoint = specification["checkpoint"]

        if checkpoint not in tokenizers:
            tokenizers[checkpoint] = AutoTokenizer.from_pretrained(checkpoint)

        encoder = Encoder(
            tokenizer=tokenizers[checkpoint],
            target_label=target_label,
            language=language,
            max_length=specification["max_length"],
            overlap_policy=overlap_policy,
            strategy=specification["strategy"],
            context_tokens=specification["context_tokens"],
            min_sentence_tokens=min_sentence_tokens,
        )

        encoded[key] = EncodedVariant(
            checkpoint=checkpoint,
            strategy=specification["strategy"],
            context_tokens=specification["context_tokens"],
            max_length=specification["max_length"],
            tokenizer=tokenizers[checkpoint],
            label2id=encoder.label2id,
            id2label=encoder.id2label,
            train_rows=encode_partition(encoder, partitions["train"]),
            validation_rows=encode_partition(encoder, partitions["validation"]),
        )

    return encoded
