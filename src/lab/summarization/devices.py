"""Which torch device a summarization model runs on."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> str:
    """Resolve `"auto"` against what this machine actually has."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested but CUDA is not available.")

    return device
