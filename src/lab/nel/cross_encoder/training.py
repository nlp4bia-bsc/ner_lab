"""Training a cross-encoder safely: DataLoader settings that survive CUDA, and quiet logs."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import transformers

from lab.nel.device import resolve_torch_device

DEFAULT_TRAINING_LOGGING_STEPS = 10_000_000


def configure_quiet_transformers_logging() -> None:
    """Silence transformers, SentenceTransformers and accelerate below errors; tqdm stays."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("WANDB_DISABLED", "true")

    for name in ("transformers", "sentence_transformers", "accelerate"):
        logging.getLogger(name).setLevel(logging.ERROR)

    transformers.utils.logging.set_verbosity_error()


def resolve_cross_encoder_dataloader_num_workers(
    requested_num_workers: int,
    device: str = "auto",
    use_legacy_fit_loop: bool = True,
) -> int:
    """
    The DataLoader worker count that is safe for cross-encoder training: 0 on CUDA.

    SentenceTransformers' collate function tokenizes the batch and moves it to the
    model's device; in a forked worker on CUDA that raises `Cannot re-initialize CUDA
    in forked subprocess`. CPU training keeps the requested count. The rule holds
    for both fit loops, since versions differ in where batching happens.
    """
    requested = max(0, int(requested_num_workers))

    if str(resolve_torch_device(device)).startswith("cuda") and requested > 0:
        return 0

    return requested


def resolve_cross_encoder_dataloader_pin_memory(
    requested_pin_memory: bool,
    device: str = "auto",
) -> bool:
    """
    The `pin_memory` that is safe for cross-encoder training: off on CUDA.

    The collate function already returns tensors on the model's device, and only
    dense CPU tensors can be pinned.
    """
    if str(resolve_torch_device(device)).startswith("cuda"):
        return False

    return bool(requested_pin_memory)


def training_dataloader(
    examples: list[Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool | None,
    device: str,
) -> torch.utils.data.DataLoader:
    """
    A shuffling DataLoader over training examples, with CUDA-safe workers and pinning.

    `pin_memory=None` pins on CUDA, which `resolve_cross_encoder_dataloader_pin_memory`
    then turns off again.
    """
    requested_pin_memory = (
        bool(pin_memory) if pin_memory is not None else str(device).startswith("cuda")
    )

    return torch.utils.data.DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        num_workers=resolve_cross_encoder_dataloader_num_workers(num_workers, device),
        pin_memory=resolve_cross_encoder_dataloader_pin_memory(requested_pin_memory, device),
    )


def supported_kwargs(function: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The `kwargs` `function` accepts, all of them when it takes `**kwargs`."""
    parameters = inspect.signature(function).parameters

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in parameters}
