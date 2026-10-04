"""Optional dependencies and the device a torch model or a FAISS index runs on."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def optional_module(name: str) -> Any | None:
    """The module called `name`, or None when it is not installed."""
    try:
        if importlib.util.find_spec(name) is None:
            return None
    except ModuleNotFoundError:
        return None

    return importlib.import_module(name)


def resolve_torch_device(device: str = "auto") -> str:
    """`"auto"` as CUDA when torch can see a GPU, CPU otherwise; any other value as given."""
    if device != "auto":
        return device

    torch = optional_module("torch")

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def resolve_faiss_device(device: str = "auto") -> str:
    """`"auto"` as CUDA when FAISS can see a GPU, CPU otherwise; any other value as given."""
    if device != "auto":
        return device

    faiss = optional_module("faiss")

    if faiss is not None and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        return "cuda"

    return "cpu"
