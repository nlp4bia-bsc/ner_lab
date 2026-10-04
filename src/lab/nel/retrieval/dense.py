"""Dense retrieval: SentenceTransformers embeddings compared by a matrix product."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lab.nel.device import optional_module, resolve_torch_device

INPUT_FORMATS = ("text", "vector")


class SentenceTransformerBiEncoder:
    """A SentenceTransformer, built from a name or path or wrapped as given, on one device."""

    def __init__(self, model_or_path: str | Any, device: str = "auto") -> None:
        self.device = resolve_torch_device(device)

        if not isinstance(model_or_path, str):
            self.model = model_or_path
            self.to(self.device)
            return

        sentence_transformers = optional_module("sentence_transformers")

        if sentence_transformers is None:
            raise ImportError("sentence-transformers is required for SentenceTransformerBiEncoder")

        self.model = sentence_transformers.SentenceTransformer(model_or_path, device=self.device)

    def cuda(self, device: int | None = None) -> SentenceTransformerBiEncoder:
        """Move the encoder to CUDA, or to the CUDA device numbered `device`."""
        return self.to(f"cuda:{device}" if device is not None else "cuda")

    def to(self, device: str | Any) -> SentenceTransformerBiEncoder:
        """Move the encoder to `device`."""
        self.device = str(device)

        if hasattr(self.model, "to"):
            self.model.to(device)

        return self

    def encode(
        self, texts: list[str], batch_size: int = 32, normalize_embeddings: bool = True
    ) -> Any:
        """Embed `texts` as one tensor, row per text."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
            device=self.device,
        )


class DenseRetriever:
    """
    Rank `term`/`code` rows by the similarity of their embeddings to each query's.

    The rows are embedded at construction unless `vector_db` supplies their
    embeddings already, row for row. With `normalize`, every embedding is
    L2-normalized, so similarity is cosine.
    """

    def __init__(
        self,
        df_candidates: pd.DataFrame,
        model_or_path: str | Any,
        normalize: bool = True,
        vector_db: Any | None = None,
        vector_db_batch_size: int = 256,
        device: str = "auto",
    ) -> None:
        self.torch = optional_module("torch")

        if self.torch is None:
            raise ImportError("torch is required for DenseRetriever")

        self.normalize = normalize
        self.device = resolve_torch_device(device)
        self.df_candidates = df_candidates.copy()

        if "term" not in self.df_candidates.columns or "code" not in self.df_candidates.columns:
            raise ValueError("df_candidates must contain term and code columns")

        self.encoder = SentenceTransformerBiEncoder(model_or_path, device=self.device)

        if vector_db is None:
            self.vector_db = self.encoder.encode(
                self.df_candidates["term"].astype(str).tolist(),
                batch_size=vector_db_batch_size,
                normalize_embeddings=self.normalize,
            )
        else:
            self.vector_db = self._normalized(vector_db).to(self.device)

    def cuda(self, device: int | None = None) -> DenseRetriever:
        """Move the retriever to CUDA, or to the CUDA device numbered `device`."""
        return self.to(f"cuda:{device}" if device is not None else "cuda")

    def to(self, device: str | Any) -> DenseRetriever:
        """Move the encoder and the embeddings to `device`."""
        self.device = str(device)
        self.encoder.to(device)
        self.vector_db = self.vector_db.to(device)

        return self

    def get_distances(
        self, data: list[str] | Any, input_format: str = "text"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The full query-by-row similarity matrix, and each query's row indices best first.

        `data` is a list of texts with `input_format="text"`, or a tensor of query
        embeddings with `input_format="vector"`.
        """
        if input_format not in INPUT_FORMATS:
            raise ValueError("input_format must be text or vector")

        if input_format == "text":
            queries = self.encoder.encode(data, normalize_embeddings=self.normalize)
        else:
            queries = self._normalized(data.to(self.device))

        distances = self.torch.mm(queries, self.vector_db.T).detach().cpu().numpy()

        return distances, distances.argsort(axis=1)[:, ::-1]

    def retrieve_top_k(
        self, data: list[str] | Any, k: int = 10, input_format: str = "text"
    ) -> list[dict[str, list[str | float]]]:
        """Each query's best `k` rows, as aligned `codes`, `terms` and `similarity` lists."""
        distances, indices = self.get_distances(data, input_format=input_format)

        return [
            {
                "codes": [str(self.df_candidates.iloc[index]["code"]) for index in row_indices[:k]],
                "terms": [str(self.df_candidates.iloc[index]["term"]) for index in row_indices[:k]],
                "similarity": [float(row_distances[index]) for index in row_indices[:k]],
            }
            for row_distances, row_indices in zip(distances, indices)
        ]

    def normalize_vector(self, vector: Any, p: int | float = 2) -> Any:
        """`vector` divided row-wise by its `p`-norm, leaving all-zero rows as they are."""
        norm = self.torch.norm(vector, p=p, dim=1, keepdim=True)
        norm[norm == 0] = 1.0

        return vector / norm

    def _normalized(self, vectors: Any) -> Any:
        return self.normalize_vector(vectors) if self.normalize else vectors
