"""Candidate retrieval: sparse, sparse-in-FAISS, dense and transformer-in-FAISS retrievers."""

from __future__ import annotations

from lab.nel.retrieval.dense import DenseRetriever, SentenceTransformerBiEncoder
from lab.nel.retrieval.faiss_index import build_cpu_index, move_index_to_gpu
from lab.nel.retrieval.sparse import SparseRetriever
from lab.nel.retrieval.sparse_faiss import SparseFaissRetriever
from lab.nel.retrieval.store import gazetteer_fingerprint, read_embeddings, write_embeddings
from lab.nel.retrieval.transformer_faiss import TransformerFaissRetriever
from lab.nel.retrieval.workflow import CandidateRetrievalPipeline, RetrievalResult, build_vocabulary

__all__ = [
    "CandidateRetrievalPipeline",
    "DenseRetriever",
    "RetrievalResult",
    "SentenceTransformerBiEncoder",
    "SparseFaissRetriever",
    "SparseRetriever",
    "TransformerFaissRetriever",
    "build_cpu_index",
    "build_vocabulary",
    "gazetteer_fingerprint",
    "move_index_to_gpu",
    "read_embeddings",
    "write_embeddings",
]
