"""Building a FAISS index of each supported kind, and moving it to a GPU."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

INNER_PRODUCT_KINDS = {"flatip", "sq8", "sq4", "ivfflatip", "ivfsq8", "ivfpq"}
IVF_KINDS = {"ivfflatip", "ivfflatl2", "ivfsq8", "ivfpq"}


def build_cpu_index(
    faiss: Any,
    kind: str,
    embeddings: np.ndarray,
    nlist: int,
    nprobe: int,
    pq_m: int,
    pq_bits: int,
) -> Any:
    """
    An empty CPU index of `kind`, trained on `embeddings` when the kind needs training.

    `kind` is a normalized f_type. IVF indexes use at most one list per embedding
    and probe at most every list; IVFPQ uses the largest sub-quantizer count up to
    `pq_m` that divides the dimension.
    """
    dimension = embeddings.shape[1]

    if kind == "flatip":
        return faiss.IndexFlatIP(dimension)

    if kind == "flatl2":
        return faiss.IndexFlatL2(dimension)

    if kind in {"sq8", "sq4"}:
        index = faiss.IndexScalarQuantizer(
            dimension, _scalar_quantizer_type(faiss, kind), faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)

        return index

    if kind not in IVF_KINDS:
        raise ValueError(f"Unsupported normalized f_type: {kind}")

    index = _ivf_index(faiss, kind, dimension, max(1, min(nlist, len(embeddings))), pq_m, pq_bits)
    index.train(embeddings)
    index.nprobe = min(nprobe, index.nlist)

    return index


def move_index_to_gpu(faiss: Any, index: Any, requested_device: str) -> tuple[Any, str, Any | None]:
    """
    `index` on the GPU `requested_device` names, with the device and the GPU resources used.

    Stays on the CPU when this FAISS build has no GPU, unless CUDA was requested
    explicitly, which raises. Keep the returned resources alive as long as the index.
    """
    gpu_count = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0
    gpu_api = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")

    if not gpu_api or gpu_count < 1:
        if requested_device.startswith("cuda"):
            raise RuntimeError(
                "CUDA was requested, but the installed FAISS build cannot access a GPU. "
                "Install faiss-gpu and verify faiss.get_num_gpus()."
            )

        logger.info("FAISS GPU is unavailable; keeping the index on CPU")

        return index, "cpu", None

    gpu_id = int(requested_device.split(":", 1)[1]) if requested_device.startswith("cuda:") else 0

    if gpu_id >= gpu_count:
        raise RuntimeError(
            f"FAISS GPU {gpu_id} was requested, but only {gpu_count} GPU(s) are available"
        )

    resources = faiss.StandardGpuResources()

    return faiss.index_cpu_to_gpu(resources, gpu_id, index), f"cuda:{gpu_id}", resources


def _ivf_index(faiss: Any, kind: str, dimension: int, nlist: int, pq_m: int, pq_bits: int) -> Any:
    if kind == "ivfflatl2":
        return faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, nlist, faiss.METRIC_L2)

    quantizer = faiss.IndexFlatIP(dimension)

    if kind == "ivfflatip":
        return faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    if kind == "ivfsq8":
        return faiss.IndexIVFScalarQuantizer(
            quantizer,
            dimension,
            nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )

    return faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        _divisor_at_most(dimension, pq_m),
        pq_bits,
        faiss.METRIC_INNER_PRODUCT,
    )


def _scalar_quantizer_type(faiss: Any, kind: str) -> Any:
    if kind == "sq8":
        return faiss.ScalarQuantizer.QT_8bit

    if hasattr(faiss.ScalarQuantizer, "QT_4bit"):
        return faiss.ScalarQuantizer.QT_4bit

    raise ValueError("SQ4 requires a FAISS build exposing ScalarQuantizer.QT_4bit; use SQ8 instead")


def _divisor_at_most(dimension: int, limit: int) -> int:
    if dimension % limit == 0:
        return limit

    for candidate in range(min(limit, dimension), 0, -1):
        if dimension % candidate == 0:
            return candidate

    return 1
