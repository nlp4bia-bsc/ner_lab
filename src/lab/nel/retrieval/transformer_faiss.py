"""Transformer embeddings searched with FAISS: the `transformer_faiss` linking method."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lab.nel.device import optional_module
from lab.nel.retrieval.faiss_index import (
    INNER_PRODUCT_KINDS,
    build_cpu_index,
    move_index_to_gpu,
)

SUPPORTED_INDEX_TYPES = (
    "FlatIP",
    "FlatL2",
    "SQ8",
    "SQ4",
    "IVFFlatIP",
    "IVFFlatL2",
    "IVFSQ8",
    "IVFPQ",
)

F_TYPE_ALIASES = {
    "flatip": "flatip",
    "ip": "flatip",
    "inner_product": "flatip",
    "flatl2": "flatl2",
    "l2": "flatl2",
    "sq8": "sq8",
    "scalarquantizer": "sq8",
    "scalar_quantizer": "sq8",
    "quantized": "sq8",
    "quantizedip": "sq8",
    "sq4": "sq4",
    "ivfflatip": "ivfflatip",
    "ivf_flat_ip": "ivfflatip",
    "ivf32,flat": "ivfflatip",
    "ivfflatl2": "ivfflatl2",
    "ivf_flat_l2": "ivfflatl2",
    "ivfsq8": "ivfsq8",
    "ivfsq8ip": "ivfsq8",
    "ivf_sq8_ip": "ivfsq8",
    "ivf32,sq8": "ivfsq8",
    "ivfpq": "ivfpq",
    "ivf_pq": "ivfpq",
    "ivf32,pq": "ivfpq",
}


class TransformerFaissRetriever:
    """
    Embed a `term`/`code` vocabulary with a transformer and retrieve unique codes with FAISS.

    The default `FlatIP`/`FlatL2` path is the HERBERT `FaissEncoder` contract: the
    terms are embedded with an `AutoModel`, pooled over `last_hidden_state`, and
    added to a FAISS `IndexIDMap` keyed by vocabulary row. The quantized
    (`SQ8`/`SQ4`) and inverted-file (`IVFFlatIP`, `IVFFlatL2`, `IVFSQ8`, `IVFPQ`)
    kinds are opt-in, trading exactness for memory and speed.

    `pooling="mean"` averages over every position, padding included, so the
    vectors depend on `batch_size`; `"attention_mask_mean"` averages over real
    tokens only, and `"cls"` takes the first position.
    """

    def __init__(
        self,
        model_name_or_path: str,
        f_type: str = "FlatIP",
        max_length: int | None = None,
        vocab: pd.DataFrame | None = None,
        batch_size: int = 64,
        device: str = "auto",
        verbose: int = 0,
        nlist: int = 100,
        nprobe: int = 10,
        pq_m: int = 8,
        pq_bits: int = 8,
        overfetch_factor: int = 5,
        use_gpu: bool = True,
        pooling: str = "mean",
    ) -> None:
        transformers = optional_module("transformers")
        torch = optional_module("torch")

        if transformers is None or torch is None:
            raise ImportError("transformers and torch are required for TransformerFaissRetriever")

        self.faiss = _require_faiss()
        self.torch = torch
        self.model_name_or_path = model_name_or_path
        self.index_kind = self.normalize_f_type(f_type)
        self.method = f"transformer_faiss_{self.index_kind}"
        self.model = transformers.AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False
        )
        self.f_type = f_type
        self.max_length = max_length or self.tokenizer.model_max_length
        self.batch_size = batch_size
        self.verbose = verbose
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.pq_bits = pq_bits
        self.overfetch_factor = max(1, overfetch_factor)
        self.use_gpu = use_gpu
        self.pooling = pooling
        self.requested_device = device
        self.device = (
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.device.startswith("cuda") and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
        self.model.eval()
        self.vocab: pd.DataFrame | None = None
        self.faiss_index: Any | None = None
        self.resolved_faiss_device = "cpu"
        self._gpu_resources: Any | None = None

        if vocab is not None:
            self.set_vocab(vocab)

    @classmethod
    def supported_index_types(cls) -> tuple[str, ...]:
        """The FAISS index kinds `f_type` accepts, by their public names."""
        return SUPPORTED_INDEX_TYPES

    @classmethod
    def normalize_f_type(cls, f_type: str) -> str:
        """The index kind an `f_type` name or alias means, ignoring case, spaces and dashes."""
        key = f_type.replace(" ", "").replace("-", "_").lower()

        if key not in F_TYPE_ALIASES:
            supported = ", ".join(SUPPORTED_INDEX_TYPES)
            raise ValueError(f"Unsupported f_type '{f_type}'. Supported values: {supported}")

        return F_TYPE_ALIASES[key]

    def set_vocab(self, vocab: pd.DataFrame) -> None:
        """Set the vocabulary: its non-missing `term`/`code` rows, as strings, in order."""
        if not {"term", "code"}.issubset(vocab.columns):
            raise ValueError("vocab must contain 'term' and 'code' columns")

        self.vocab = vocab[["term", "code"]].dropna().copy().reset_index(drop=True)
        self.vocab["term"] = self.vocab["term"].astype(str)
        self.vocab["code"] = self.vocab["code"].astype(str)
        self.arr_text = self.vocab["term"].tolist()
        self.arr_codes = self.vocab["code"].tolist()
        self.arr_text_id = np.arange(len(self.vocab), dtype=np.int64)

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed `texts` in batches, pooled as `pooling` says, as one float32 array."""
        if not texts:
            raise ValueError("texts cannot be empty")

        batch_size = batch_size or self.batch_size
        model = self.model.module if hasattr(self.model, "module") else self.model
        starts: Any = range(0, len(texts), batch_size)
        tqdm_module = optional_module("tqdm.auto") if self.verbose else None

        if tqdm_module is not None:
            starts = tqdm_module.tqdm(starts, desc="Encoding texts", disable=self.verbose == 0)

        embeddings = [
            self._encode_batch(model, texts[start : start + batch_size]) for start in starts
        ]

        return np.vstack(embeddings).astype("float32")

    def fit_faiss(self, vocab: pd.DataFrame | None = None, batch_size: int | None = None) -> None:
        """Embed the vocabulary, setting it first if `vocab` is given, and index it."""
        if vocab is not None:
            self.set_vocab(vocab)

        if self.vocab is None:
            raise ValueError("vocab is required before fitting FAISS")

        self.fit_faiss_from_embeddings(
            self.encode(self.arr_text, batch_size=batch_size or self.batch_size)
        )

    def fit_faiss_from_embeddings(
        self, embeddings: np.ndarray, vocab: pd.DataFrame | None = None
    ) -> None:
        """
        Index precomputed embeddings, row `i` embedding vocabulary row `i`.

        The embeddings are copied, so the caller's array is never normalized in place.
        """
        if vocab is not None:
            self.set_vocab(vocab)

        if self.vocab is None:
            raise ValueError("vocab is required before fitting FAISS")

        if len(embeddings) != len(self.arr_text):
            raise ValueError(
                f"{len(embeddings)} embeddings for {len(self.arr_text)} vocabulary rows"
            )

        embeddings = self._prepared(np.array(embeddings, dtype="float32"))
        cpu_index = build_cpu_index(
            self.faiss,
            self.index_kind,
            embeddings,
            nlist=self.nlist,
            nprobe=self.nprobe,
            pq_m=self.pq_m,
            pq_bits=self.pq_bits,
        )
        id_map = (
            self.faiss.IndexIDMap2 if hasattr(self.faiss, "IndexIDMap2") else self.faiss.IndexIDMap
        )
        id_index = id_map(cpu_index)
        id_index.add_with_ids(embeddings, self.arr_text_id)
        self.faiss_index = self._placed(id_index)

    def get_candidates(
        self, texts: list[str], k: int = 200, batch_size: int | None = None
    ) -> tuple[list[list[str]], list[list[str]], list[list[float]]]:
        """Each text's best `k` unique codes, as aligned term, code and score lists."""
        if self.faiss_index is None:
            raise AttributeError("FAISS index is not initialized. Run fit_faiss first.")

        queries = self._prepared(
            self.encode(texts, batch_size=batch_size).astype("float32", copy=False)
        )
        overfetch = min(len(self.arr_text), max(k * self.overfetch_factor, k))
        similarities, indices = self.faiss_index.search(queries, overfetch)
        rankings = [
            self._unique_codes(row_indices, row_scores, k)
            for row_indices, row_scores in zip(indices, similarities)
        ]

        return (
            [terms for terms, _, _ in rankings],
            [codes for _, codes, _ in rankings],
            [scores for _, _, scores in rankings],
        )

    def _encode_batch(self, model: Any, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            pooled = _pool(hidden, inputs, self.pooling)

        return pooled.detach().cpu().numpy()

    def _prepared(self, embeddings: np.ndarray) -> np.ndarray:
        if self.index_kind in INNER_PRODUCT_KINDS:
            self.faiss.normalize_L2(embeddings)

        return embeddings

    def _placed(self, index: Any) -> Any:
        self.resolved_faiss_device = "cpu"

        if not self.use_gpu or self.requested_device == "cpu":
            return index

        index, self.resolved_faiss_device, resources = move_index_to_gpu(
            self.faiss, index, self.requested_device
        )

        if resources is not None:
            self._gpu_resources = resources

        return index

    def _unique_codes(
        self, indices: np.ndarray, scores: np.ndarray, k: int
    ) -> tuple[list[str], list[str], list[float]]:
        seen: set[str] = set()
        terms: list[str] = []
        codes: list[str] = []
        kept_scores: list[float] = []

        for index, score in zip(indices, scores):
            if index < 0 or self.arr_codes[int(index)] in seen:
                continue

            seen.add(self.arr_codes[int(index)])
            terms.append(self.arr_text[int(index)])
            codes.append(self.arr_codes[int(index)])
            kept_scores.append(float(score))

            if len(codes) >= k:
                break

        return terms, codes, kept_scores


def _pool(hidden: Any, inputs: dict[str, Any], pooling: str) -> Any:
    if pooling == "mean":
        return hidden.mean(dim=1)

    if pooling in {"attention_mask_mean", "masked_mean"}:
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()

        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    if pooling == "cls":
        return hidden[:, 0]

    raise ValueError("pooling must be one of: mean, attention_mask_mean, cls")


def _require_faiss() -> Any:
    faiss = optional_module("faiss")

    if faiss is None:
        raise ImportError(
            "faiss-gpu (preferred) or faiss-cpu is required for TransformerFaissRetriever"
        )

    return faiss
