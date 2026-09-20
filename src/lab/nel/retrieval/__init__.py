"""Candidate retrieval methods, one module per method family."""

from lab.nel.retrieval.base import BaseBiEncoder
from lab.nel.retrieval.faiss import FaissBiEncoder
from lab.nel.retrieval.matrix import MatrixBiEncoder
from lab.nel.retrieval.sentence_transformer import DenseRetriever, SentenceTransformerBiEncoder
from lab.nel.retrieval.transformer_faiss import HerbertFaissBiEncoder
from lab.nel.retrieval.workflow import CandidateRetrievalPipeline, RetrievalResult, build_vocabulary

__all__ = [
    "BaseBiEncoder",
    "CandidateRetrievalPipeline",
    "DenseRetriever",
    "FaissBiEncoder",
    "HerbertFaissBiEncoder",
    "MatrixBiEncoder",
    "RetrievalResult",
    "SentenceTransformerBiEncoder",
    "build_vocabulary",
]
