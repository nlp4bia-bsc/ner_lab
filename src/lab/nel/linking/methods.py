"""Building what a method name refers to: the candidate generator and the reranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from lab.nel.linking.tables import GAZETTEER_COLUMNS
from lab.nel.matching import MATCHER_REGISTRY, build_matcher
from lab.nel.retrieval import (
    DenseRetriever,
    SparseFaissRetriever,
    SparseRetriever,
    TransformerFaissRetriever,
    read_embeddings,
    write_embeddings,
)
from lab.nel.schemas import Candidate, GazetteerEntry, Mention

IndexSource = Literal["loaded", "built", "in_memory"]

RETRIEVERS = {"matrix": SparseRetriever, "faiss": SparseFaissRetriever}
ENCODER_METHODS = ("transformer_faiss", "dense")
METHODS = (*sorted(MATCHER_REGISTRY), *sorted(RETRIEVERS), *ENCODER_METHODS)

INDEX_DIRNAME = "gazetteer_index"


def build_candidate_generator(
    method: str,
    entries: list[GazetteerEntry],
    gazetteer_df: pd.DataFrame,
    base_model: str | None,
    top_k: int,
    method_kwargs: dict[str, Any],
    index_dir: Path | None = None,
    save_index: bool = True,
) -> tuple[Any, IndexSource | None]:
    """
    Construct and index the candidate generator a method name refers to.

    Also returns where an encoder method's gazetteer embeddings came from:
    `loaded` from `index_dir`, `built` and saved there, or `in_memory`.
    None for the other methods, which have no embeddings to keep.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Available methods: {', '.join(METHODS)}.")

    if method in MATCHER_REGISTRY:
        return build_matcher(method, gazetteer=entries, top_k=top_k, **method_kwargs), None

    if method in RETRIEVERS:
        retriever = RETRIEVERS[method](top_k=top_k, **method_kwargs)
        retriever.build_index(entries)

        return retriever, None

    vocabulary_df = gazetteer_df[GAZETTEER_COLUMNS]

    if method == "transformer_faiss":
        retriever, source = _transformer_faiss_retriever(
            base_model, vocabulary_df, method_kwargs, index_dir, save_index
        )

        return EncoderRetriever(retriever, retriever.method, top_k), source

    retriever, source = _dense_retriever(
        base_model, vocabulary_df, method_kwargs, index_dir, save_index
    )

    return EncoderRetriever(retriever, method, top_k), source


def resolve_index_dir(method: str, output_dir: Path, index_dir: str | Path | None) -> Path | None:
    """Where an encoder method keeps the gazetteer's embeddings; None for the other methods."""
    if method not in ENCODER_METHODS:
        return None

    return output_dir / INDEX_DIRNAME if index_dir is None else Path(index_dir)


def transformer_faiss_settings(
    retriever: TransformerFaissRetriever, base_model: str
) -> dict[str, Any]:
    """
    What a `transformer_faiss` embedding depends on.

    `mean` pooling averages over padding too, so the batch size that decides the
    padding changes the vectors and is part of the settings.
    """
    settings = {
        "method": "transformer_faiss",
        "base_model": base_model,
        "pooling": retriever.pooling,
        "max_length": int(retriever.max_length),
    }

    if retriever.pooling == "mean":
        settings["batch_size"] = int(retriever.batch_size)

    return settings


class EncoderRetriever:
    """Gives the text-in, lists-out encoder retrievers the `search` the pipeline expects."""

    def __init__(self, retriever: Any, method: str, top_k: int) -> None:
        self.retriever = retriever
        self.method = method
        self.top_k = top_k

    def search(self, mentions: list[Mention]) -> list[list[Candidate]]:
        """The `top_k` candidates for each mention, best first."""
        texts = [mention.text for mention in mentions]

        if hasattr(self.retriever, "get_candidates"):
            terms, codes, scores = self.retriever.get_candidates(texts, k=self.top_k)
        else:
            hits = self.retriever.retrieve_top_k(texts, k=self.top_k)
            terms = [hit["terms"] for hit in hits]
            codes = [hit["codes"] for hit in hits]
            scores = [hit["similarity"] for hit in hits]

        return [
            [
                Candidate(
                    mention_id=None,
                    filename=mention.filename,
                    text=mention.text,
                    label=mention.label,
                    code=str(code),
                    term=str(term),
                    score=float(score),
                    method=self.method,
                    rank=rank,
                )
                for rank, (code, term, score) in enumerate(
                    zip(mention_codes, mention_terms, mention_scores), 1
                )
            ]
            for mention, mention_codes, mention_terms, mention_scores in zip(
                mentions, codes, terms, scores
            )
        ]


def build_reranker(reranker: str | Path | None, reranker_kwargs: dict[str, Any]) -> Any | None:
    """Load the cross-encoder reranker a model path refers to."""
    if reranker is None:
        return None

    from lab.nel.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(str(reranker), **reranker_kwargs)


def _transformer_faiss_retriever(
    base_model: str,
    vocabulary_df: pd.DataFrame,
    method_kwargs: dict[str, Any],
    index_dir: Path | None,
    save_index: bool,
) -> tuple[TransformerFaissRetriever, IndexSource]:
    retriever = TransformerFaissRetriever(base_model, **method_kwargs)
    retriever.set_vocab(vocabulary_df)
    settings = transformer_faiss_settings(retriever, base_model)
    embeddings = _stored_embeddings(index_dir, retriever.vocab, settings)
    source: IndexSource = "loaded"

    if embeddings is None:
        embeddings = retriever.encode(retriever.arr_text)
        source = _store_embeddings(index_dir, embeddings, retriever.vocab, settings, save_index)

    retriever.fit_faiss_from_embeddings(embeddings)

    return retriever, source


def _dense_retriever(
    base_model: str,
    vocabulary_df: pd.DataFrame,
    method_kwargs: dict[str, Any],
    index_dir: Path | None,
    save_index: bool,
) -> tuple[DenseRetriever, IndexSource]:
    settings = {
        "method": "dense",
        "base_model": base_model,
        "normalize": bool(method_kwargs.get("normalize", True)),
    }
    embeddings = _stored_embeddings(index_dir, vocabulary_df, settings)

    if embeddings is None:
        retriever = DenseRetriever(vocabulary_df, base_model, **method_kwargs)
        vectors = retriever.vector_db.detach().cpu().numpy()

        return retriever, _store_embeddings(index_dir, vectors, vocabulary_df, settings, save_index)

    import torch

    retriever = DenseRetriever(
        vocabulary_df, base_model, vector_db=torch.from_numpy(embeddings), **method_kwargs
    )

    return retriever, "loaded"


def _stored_embeddings(
    index_dir: Path | None, vocabulary_df: pd.DataFrame, settings: dict[str, Any]
) -> np.ndarray | None:
    if index_dir is None:
        return None

    return read_embeddings(index_dir, vocabulary_df, settings)


def _store_embeddings(
    index_dir: Path | None,
    embeddings: np.ndarray,
    vocabulary_df: pd.DataFrame,
    settings: dict[str, Any],
    save_index: bool,
) -> IndexSource:
    if index_dir is None or not save_index:
        return "in_memory"

    write_embeddings(index_dir, embeddings, vocabulary_df, settings)

    return "built"
