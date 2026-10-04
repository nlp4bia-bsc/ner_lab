"""Sparse n-gram vectors searched with FAISS, on the GPU when one is available."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from lab.nel.device import optional_module, resolve_faiss_device
from lab.nel.preprocessing import normalize_text
from lab.nel.retrieval.sparse import SparseRetriever
from lab.nel.schemas import Candidate, GazetteerEntry, Mention

VECTORIZER_BACKENDS = ("hashing", "tfidf")


class SparseFaissRetriever(SparseRetriever):
    """
    N-gram vectors in a flat FAISS index, returning up to `top_k` unique codes per mention.

    FAISS needs dense vectors, and a TF-IDF vocabulary over a biomedical gazetteer
    can run to hundreds of thousands of n-grams, so the default `"hashing"` backend
    bounds the dimension at `n_features`; `"tfidf"` caps it at `max_features`.
    `similarity` is `"cosine"` (inner product of L2-normalized vectors), `"dot"`, or
    L2 distance for anything else.
    """

    method = "faiss_biencoder"

    def __init__(
        self,
        similarity: str = "cosine",
        device: str = "auto",
        vectorizer_backend: str = "hashing",
        n_features: int = 4096,
        max_features: int = 50000,
        analyzer: str = "char",
        ngram_range: tuple[int, int] = (3, 5),
        **kwargs: Any,
    ) -> None:
        if vectorizer_backend not in VECTORIZER_BACKENDS:
            raise ValueError("vectorizer_backend must be 'hashing' or 'tfidf'")

        super().__init__(similarity=similarity, **kwargs)
        self.device = device
        self.resolved_device = "cpu"
        self.index: Any | None = None
        self._term_vectors: np.ndarray | None = None
        self.vectorizer_backend = vectorizer_backend
        self.n_features = n_features
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.vectorizer = self._build_vectorizer()

    def search(
        self,
        mentions: list[Mention],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[list[Candidate]]:
        """Each mention's candidates: unique codes, best first, at or above the threshold."""
        faiss = _require_faiss()

        if self.index is None:
            raise RuntimeError("FAISS index not built")

        k = top_k or self.top_k
        threshold = self.threshold if threshold is None else threshold
        overfetch = min(len(self.gazetteer), max(k * 5, k))
        queries = self._vectors([normalize_text(mention.text) for mention in mentions], fit=False)

        if self.similarity == "cosine":
            faiss.normalize_L2(queries)

        scores, indices = self.index.search(queries, overfetch)

        return [
            self._ranked(mention, row_scores, row_indices, k, threshold)
            for mention, row_scores, row_indices in zip(mentions, scores, indices)
        ]

    def _build_vectorizer(self) -> HashingVectorizer | TfidfVectorizer:
        if self.vectorizer_backend == "hashing":
            return HashingVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                n_features=self.n_features,
                alternate_sign=False,
                norm=None,
                dtype=np.float32,
            )

        return TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            dtype=np.float32,
        )

    def _vectors(self, texts: list[str], fit: bool) -> np.ndarray:
        sparse_matrix = (
            self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        )

        return sparse_matrix.astype(np.float32).toarray()

    def _build_index(self, gazetteer: list[GazetteerEntry]) -> None:
        faiss = _require_faiss()
        vectors = self._vectors([normalize_text(entry.term) for entry in gazetteer], fit=True)

        if self.similarity == "cosine":
            faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        if self.similarity in {"cosine", "dot"}:
            cpu_index = faiss.IndexFlatIP(dimension)
        else:
            cpu_index = faiss.IndexFlatL2(dimension)

        cpu_index.add(vectors)
        self.resolved_device = resolve_faiss_device(self.device)

        if self.resolved_device.startswith("cuda") and hasattr(faiss, "StandardGpuResources"):
            gpu_id = (
                int(self.resolved_device.split(":", 1)[1]) if ":" in self.resolved_device else 0
            )
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), gpu_id, cpu_index)
        else:
            self.index = cpu_index

        self._term_vectors = vectors

    def _ranked(
        self,
        mention: Mention,
        scores: np.ndarray,
        indices: np.ndarray,
        k: int,
        threshold: float,
    ) -> list[Candidate]:
        ranked: list[Candidate] = []
        seen: set[str] = set()

        for score, index in zip(scores, indices):
            if index < 0:
                continue

            entry = self.gazetteer[int(index)]
            value = float(score)

            if value < threshold or not self._allowed(mention, entry) or entry.code in seen:
                continue

            seen.add(entry.code)
            ranked.append(
                Candidate(
                    mention_id=None,
                    filename=mention.filename,
                    text=mention.text,
                    label=mention.label,
                    code=entry.code,
                    term=entry.term,
                    score=value,
                    method=self.method,
                    rank=len(ranked) + 1,
                    metadata={
                        "device": self.resolved_device,
                        "vectorizer_backend": self.vectorizer_backend,
                        "n_features": self.n_features,
                        "max_features": self.max_features,
                    },
                )
            )

            if len(ranked) >= k:
                break

        return ranked


def _require_faiss() -> Any:
    faiss = optional_module("faiss")

    if faiss is None:
        raise ImportError("faiss-gpu (preferred) or faiss-cpu is required for SparseFaissRetriever")

    return faiss
