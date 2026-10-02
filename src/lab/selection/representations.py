"""Document representations used by :mod:`lab.selection`.

All representation implementations live in this file deliberately.  Selection
methods consume only matrices, so representations can evolve independently and
can also be precomputed once and reused across methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


BIOLORD_EN = "FremyCompany/BioLORD-2023"
BIOLORD_MULTILINGUAL = "FremyCompany/BioLORD-2023-M"
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"


@dataclass(frozen=True)
class RepresentationInfo:
    name: str
    family: str
    description: str
    requires_model: bool = False
    implementation_note: str | None = None
    parameters: Mapping[str, str] = field(default_factory=dict)


@dataclass
class RepresentationOutput:
    doc_ids: list[str]
    matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class RepresentationMethod(ABC):
    @abstractmethod
    def encode(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        random_state: int = 13,
        **params: Any,
    ) -> RepresentationOutput:
        """Produce one row per document in the same order as ``doc_ids``."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _word_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    chunk_size = max(1, int(chunk_size))
    overlap = min(max(0, int(overlap)), chunk_size - 1)
    step = max(1, chunk_size - overlap)
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step)]


def _resolved_model_limit(tokenizer: Any, model: Any, requested: int) -> int:
    limits: list[int] = []
    for value in (
        requested,
        getattr(tokenizer, "model_max_length", None),
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
    ):
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if 1 < integer < 1_000_000:
            limits.append(integer)
    return min(limits) if limits else int(requested)


def _single_sequence_special_affixes(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Infer the special-token prefix/suffix for a single input sequence.

    Recent Transformers tokenizers do not expose
    ``build_inputs_with_special_tokens`` uniformly.  We therefore infer the
    single-sequence template using the stable ``encode(...,
    add_special_tokens=...)`` interface.  Typical results are ``[CLS] ...
    [SEP]`` for BERT and ``<s> ... </s>`` for RoBERTa/XLM-R.
    """
    probe = "selection tokenizer compatibility probe"
    content = list(tokenizer.encode(probe, add_special_tokens=False))
    wrapped = list(tokenizer.encode(probe, add_special_tokens=True))

    if not content:
        raise ValueError(
            "Tokenizer produced no tokens for the special-token template probe."
        )

    width = len(content)
    for start in range(len(wrapped) - width + 1):
        if wrapped[start : start + width] == content:
            return wrapped[:start], wrapped[start + width :]

    raise ValueError(
        "Could not infer the tokenizer's single-sequence special-token template. "
        "The encoding with add_special_tokens=True does not contain the plain encoding."
    )


def _token_chunks(
    tokenizer: Any,
    text: str,
    max_length: int,
    overlap: int,
) -> tuple[list[list[int]], list[int]]:
    """Tokenize one document into overlapping, model-ready token-ID chunks.

    The function is compatible with current Transformers tokenizer backends and
    avoids backend-specific helpers such as ``build_inputs_with_special_tokens``.
    The returned counts contain the number of content tokens in each chunk and
    are used later for token-weighted document pooling.
    """
    prefix, suffix = _single_sequence_special_affixes(tokenizer)
    content_ids = list(tokenizer.encode(str(text), add_special_tokens=False))

    special_count = len(prefix) + len(suffix)
    if special_count >= int(max_length):
        raise ValueError(
            "Tokenizer special tokens occupy the complete requested sequence length: "
            f"{special_count} >= {max_length}."
        )

    budget = int(max_length) - special_count
    overlap = min(max(0, int(overlap)), budget - 1)
    step = max(1, budget - overlap)

    if not content_ids:
        return [[*prefix, *suffix]], [0]

    chunks: list[list[int]] = []
    counts: list[int] = []

    for start in range(0, len(content_ids), step):
        piece = content_ids[start : start + budget]
        if not piece:
            continue

        chunk = [*prefix, *piece, *suffix]
        if len(chunk) > int(max_length):
            raise RuntimeError(
                "Internal chunk construction exceeded max_length: "
                f"{len(chunk)} > {max_length}."
            )

        chunks.append(chunk)
        counts.append(len(piece))

        if start + budget >= len(content_ids):
            break

    return chunks, counts


class TFIDFRepresentation(RepresentationMethod):
    """Word n-gram TF-IDF, averaged over long-document word chunks."""

    def encode(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        language: str = "en",
        chunk_size: int = 350,
        chunk_overlap: int = 50,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_min: int = 1,
        ngram_max: int = 2,
        max_features: int | None = None,
        **_: Any,
    ) -> RepresentationOutput:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as error:
            raise ImportError("TF-IDF requires scikit-learn: pip install 'lab[selection]'.") from error

        chunks: list[str] = []
        owners: list[int] = []
        counts: list[int] = []
        for owner, text in enumerate(texts):
            document_chunks = _word_chunks(str(text), chunk_size, chunk_overlap)
            counts.append(len(document_chunks))
            chunks.extend(document_chunks)
            owners.extend([owner] * len(document_chunks))

        stop_words = "english" if str(language).lower().startswith("en") else None
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            ngram_range=(int(ngram_min), int(ngram_max)),
            min_df=int(min_df),
            max_df=float(max_df),
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        chunk_matrix = vectorizer.fit_transform(chunks)
        document_matrix = np.zeros((len(doc_ids), chunk_matrix.shape[1]), dtype=np.float32)
        document_counts = np.zeros(len(doc_ids), dtype=np.float32)
        for row, owner in enumerate(owners):
            document_matrix[owner] += chunk_matrix.getrow(row).toarray().ravel().astype(np.float32)
            document_counts[owner] += 1
        document_matrix /= np.clip(document_counts[:, None], 1.0, None)
        document_matrix = _normalize(document_matrix)
        return RepresentationOutput(
            list(map(str, doc_ids)),
            document_matrix,
            {
                "representation": "tfidf",
                "vocabulary_size": int(len(vectorizer.vocabulary_)),
                "chunk_counts": counts,
                "chunk_size_words": int(chunk_size),
                "chunk_overlap_words": int(chunk_overlap),
                "language": language,
                "ngram_range": [int(ngram_min), int(ngram_max)],
                "min_df": int(min_df),
                "max_df": float(max_df),
                "max_features": max_features,
            },
        )


TRANSFORMER_PRESETS: dict[str, dict[str, Any]] = {
    "biolord": {
        "model_en": BIOLORD_EN,
        "model_multilingual": BIOLORD_MULTILINGUAL,
        "pooling": "mean",
    },
    "medcpt": {
        "model_en": MEDCPT_ARTICLE,
        "model_multilingual": MEDCPT_ARTICLE,
        "pooling": "cls",
    },
}


class TransformerRepresentation(RepresentationMethod):
    """Generic Hugging Face encoder with token-aware document chunking."""

    def encode(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        model: str | None = None,
        preset: str | None = None,
        language: str = "en",
        pooling: str | None = None,
        max_length: int = 512,
        chunk_overlap: int = 64,
        batch_size: int = 16,
        device: str | None = None,
        **_: Any,
    ) -> RepresentationOutput:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Transformer representations require torch and transformers; install lab[torch]."
            ) from error

        preset_values: dict[str, Any] = {}
        if preset:
            if preset not in TRANSFORMER_PRESETS:
                raise ValueError(
                    f"Unknown transformer preset {preset!r}. Available: {', '.join(TRANSFORMER_PRESETS)}."
                )
            preset_values = TRANSFORMER_PRESETS[preset]
        if model is None:
            if not preset_values:
                raise ValueError("Transformer representation requires --representation-param model=...")
            key = "model_en" if str(language).lower().startswith("en") else "model_multilingual"
            model = str(preset_values[key])
        pooling = str(pooling or preset_values.get("pooling", "mean")).lower()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'.")
        if preset == "medcpt" and not str(language).lower().startswith("en"):
            warnings.warn(
                "The MedCPT Article Encoder is English/PubMed oriented; consider a multilingual model.",
                RuntimeWarning,
                stacklevel=2,
            )

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model)
        encoder = AutoModel.from_pretrained(model).to(resolved_device)
        encoder.eval()
        effective_max = _resolved_model_limit(tokenizer, encoder, int(max_length))

        document_vectors: list[np.ndarray] = []
        chunk_counts: list[int] = []
        for text in texts:
            chunks, token_counts = _token_chunks(tokenizer, str(text), effective_max, chunk_overlap)
            chunk_counts.append(len(chunks))
            vectors: list[np.ndarray] = []
            for start in range(0, len(chunks), int(batch_size)):
                batch_ids = chunks[start : start + int(batch_size)]
                width = max(len(ids) for ids in batch_ids)
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                input_ids = np.full((len(batch_ids), width), pad_id, dtype=np.int64)
                attention = np.zeros((len(batch_ids), width), dtype=np.int64)
                for row, ids in enumerate(batch_ids):
                    input_ids[row, : len(ids)] = ids
                    attention[row, : len(ids)] = 1
                input_tensor = torch.tensor(input_ids, device=resolved_device)
                attention_tensor = torch.tensor(attention, device=resolved_device)
                with torch.no_grad():
                    hidden = encoder(input_ids=input_tensor, attention_mask=attention_tensor).last_hidden_state
                    if pooling == "cls":
                        batch_vectors = hidden[:, 0, :]
                    else:
                        weights = attention_tensor.unsqueeze(-1).to(hidden.dtype)
                        batch_vectors = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
                    batch_vectors = torch.nn.functional.normalize(batch_vectors, p=2, dim=1)
                vectors.extend(batch_vectors.detach().cpu().numpy().astype(np.float32))

            weights = np.asarray(token_counts, dtype=np.float64)
            if not len(weights) or float(weights.sum()) <= 0:
                weights = np.ones(len(vectors), dtype=np.float64)
            document = np.average(np.vstack(vectors), axis=0, weights=weights).astype(np.float32)
            document /= max(float(np.linalg.norm(document)), 1e-12)
            document_vectors.append(document)

        matrix = np.vstack(document_vectors)
        return RepresentationOutput(
            list(map(str, doc_ids)),
            matrix,
            {
                "representation": "transformer",
                "preset": preset,
                "model": model,
                "pooling": pooling,
                "language": language,
                "device": resolved_device,
                "effective_max_length": int(effective_max),
                "chunk_overlap_tokens": int(chunk_overlap),
                "chunk_counts": chunk_counts,
            },
        )


class BioLORDRepresentation(RepresentationMethod):
    """Official SentenceTransformers-style BioLORD document representation."""

    def encode(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        model: str | None = None,
        language: str = "en",
        max_length: int = 512,
        chunk_overlap: int = 64,
        batch_size: int = 16,
        device: str | None = None,
        **_: Any,
    ) -> RepresentationOutput:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "BioLORD requires sentence-transformers: pip install 'lab[selection]'."
            ) from error

        resolved = model or (
            BIOLORD_EN if str(language).lower().startswith("en") else BIOLORD_MULTILINGUAL
        )
        encoder = SentenceTransformer(resolved, device=device)
        tokenizer = encoder.tokenizer
        effective_max = min(int(max_length), int(getattr(encoder, "max_seq_length", max_length)))
        document_vectors: list[np.ndarray] = []
        chunk_counts: list[int] = []
        for text in texts:
            token_chunks, token_counts = _token_chunks(
                tokenizer, str(text), effective_max, chunk_overlap
            )
            chunk_texts = [
                tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for ids in token_chunks
            ]
            chunk_counts.append(len(chunk_texts))
            vectors = encoder.encode(
                chunk_texts,
                batch_size=int(batch_size),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            weights = np.asarray(token_counts, dtype=np.float64)
            if float(weights.sum()) <= 0:
                weights = np.ones(len(vectors), dtype=np.float64)
            document = np.average(vectors, axis=0, weights=weights).astype(np.float32)
            document /= max(float(np.linalg.norm(document)), 1e-12)
            document_vectors.append(document)

        return RepresentationOutput(
            list(map(str, doc_ids)),
            np.vstack(document_vectors),
            {
                "representation": "biolord",
                "model": resolved,
                "language": language,
                "effective_max_length": int(effective_max),
                "chunk_overlap_tokens": int(chunk_overlap),
                "chunk_counts": chunk_counts,
                "pooling": "SentenceTransformer chunk embedding -> token-weighted mean -> L2 normalize",
            },
        )


class MedCPTRepresentation(TransformerRepresentation):
    def encode(self, doc_ids: Sequence[str], texts: Sequence[str], **params: Any) -> RepresentationOutput:
        params.setdefault("preset", "medcpt")
        return super().encode(doc_ids, texts, **params)


class ALPSSurprisalRepresentation(RepresentationMethod):
    """ALPS-style sparse, maskless MLM-surprisal vectors averaged over chunks."""

    def encode(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        model: str | None = None,
        language: str = "en",
        max_length: int = 256,
        chunk_overlap: int = 32,
        sample_fraction: float = 0.15,
        batch_size: int = 8,
        device: str | None = None,
        random_state: int = 13,
        **_: Any,
    ) -> RepresentationOutput:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("ALPS representations require torch and transformers; install lab[torch].") from error

        resolved_model = model or (
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            if str(language).lower().startswith("en")
            else "xlm-roberta-base"
        )
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(resolved_model)
        mlm = AutoModelForMaskedLM.from_pretrained(resolved_model).to(resolved_device)
        mlm.eval()
        effective_max = _resolved_model_limit(tokenizer, mlm, int(max_length))
        rng = np.random.default_rng(random_state)
        special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
        documents: list[np.ndarray] = []
        chunk_counts: list[int] = []

        for text in texts:
            chunks, _token_counts = _token_chunks(tokenizer, str(text), effective_max, chunk_overlap)
            chunk_counts.append(len(chunks))
            vectors: list[np.ndarray] = []
            for start in range(0, len(chunks), int(batch_size)):
                batch = chunks[start : start + int(batch_size)]
                width = effective_max
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                input_ids = np.full((len(batch), width), pad_id, dtype=np.int64)
                attention = np.zeros((len(batch), width), dtype=np.int64)
                positions: list[np.ndarray] = []
                for row, ids in enumerate(batch):
                    clipped = ids[:width]
                    input_ids[row, : len(clipped)] = clipped
                    attention[row, : len(clipped)] = 1
                    special = np.asarray(
                        [int(token_id) in special_ids for token_id in clipped],
                        dtype=bool,
                    )
                    valid = np.where(~special)[0]
                    count = max(1, int(round(float(sample_fraction) * len(valid)))) if len(valid) else 0
                    positions.append(
                        rng.choice(valid, size=min(count, len(valid)), replace=False)
                        if count
                        else np.array([], dtype=int)
                    )

                ids_tensor = torch.tensor(input_ids, device=resolved_device)
                attention_tensor = torch.tensor(attention, device=resolved_device)
                with torch.no_grad():
                    logits = mlm(input_ids=ids_tensor, attention_mask=attention_tensor).logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                for row, sampled in enumerate(positions):
                    vector = np.zeros(width, dtype=np.float32)
                    if len(sampled):
                        pos_tensor = torch.tensor(sampled, device=resolved_device, dtype=torch.long)
                        targets = ids_tensor[row, pos_tensor]
                        losses = -log_probs[row, pos_tensor, targets]
                        vector[sampled] = losses.detach().cpu().numpy().astype(np.float32)
                    norm = float(np.linalg.norm(vector))
                    if norm > 0:
                        vector /= norm
                    vectors.append(vector)

            document = np.mean(np.vstack(vectors), axis=0).astype(np.float32)
            document /= max(float(np.linalg.norm(document)), 1e-12)
            documents.append(document)

        return RepresentationOutput(
            list(map(str, doc_ids)),
            np.vstack(documents),
            {
                "representation": "alps",
                "model": resolved_model,
                "language": language,
                "effective_max_length": int(effective_max),
                "chunk_overlap_tokens": int(chunk_overlap),
                "sample_fraction": float(sample_fraction),
                "chunk_counts": chunk_counts,
                "device": resolved_device,
                "random_state": int(random_state),
            },
        )


REPRESENTATIONS: dict[str, type[RepresentationMethod]] = {
    "tfidf": TFIDFRepresentation,
    "transformer": TransformerRepresentation,
    "biolord": BioLORDRepresentation,
    "medcpt": MedCPTRepresentation,
    "alps": ALPSSurprisalRepresentation,
}

REPRESENTATION_INFO: dict[str, RepresentationInfo] = {
    "tfidf": RepresentationInfo(
        "tfidf", "lexical", "Word n-gram TF-IDF document vectors.",
        parameters={
            "language": "Language code; English enables sklearn English stop words.",
            "chunk_size": "Words per chunk (default: 350).",
            "chunk_overlap": "Overlapping words (default: 50).",
            "min_df": "Minimum document frequency (default: 1).",
            "max_df": "Maximum document-frequency proportion (default: 1.0).",
            "ngram_min/ngram_max": "Word n-gram range (default: 1..2).",
            "max_features": "Optional vocabulary cap.",
        },
    ),
    "transformer": RepresentationInfo(
        "transformer", "dense", "Generic Hugging Face Transformer document encoder.",
        requires_model=True,
        parameters={
            "model": "Model identifier or local path (required without a preset).",
            "pooling": "mean or cls (default: mean).",
            "max_length": "Maximum model sequence length requested (default: 512).",
            "chunk_overlap": "Token overlap between chunks (default: 64).",
            "batch_size": "Chunk inference batch size (default: 16).",
            "device": "Optional torch device.",
        },
    ),
    "biolord": RepresentationInfo(
        "biolord", "dense preset", "BioLORD-2023/BioLORD-2023-M Transformer preset.",
        requires_model=True,
        implementation_note="English uses BioLORD-2023; other languages use BioLORD-2023-M unless model is overridden.",
    ),
    "medcpt": RepresentationInfo(
        "medcpt", "dense preset", "MedCPT Article Encoder preset with CLS chunk pooling.",
        requires_model=True,
        implementation_note="Primarily intended for English/PubMed-style text.",
    ),
    "alps": RepresentationInfo(
        "alps", "surprisal", "Sparse MLM-surprisal vectors used by ALPS.",
        requires_model=True,
        parameters={
            "model": "Optional masked language model identifier/path; defaults to PubMedBERT for English and XLM-R for other languages.",
            "max_length": "Surprisal vector/chunk length (default: 256).",
            "sample_fraction": "Fraction of non-special tokens scored (default: 0.15).",
            "chunk_overlap": "Token overlap (default: 32).",
            "batch_size": "MLM inference batch size (default: 8).",
        },
    ),
}


def list_representations() -> list[RepresentationInfo]:
    return [REPRESENTATION_INFO[name] for name in REPRESENTATIONS]


def representation_info(name: str) -> RepresentationInfo:
    try:
        return REPRESENTATION_INFO[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown representation {name!r}. Available: {', '.join(REPRESENTATIONS)}."
        ) from error


def get_representation(name: str) -> RepresentationMethod:
    try:
        return REPRESENTATIONS[name]()
    except KeyError as error:
        raise ValueError(
            f"Unknown representation {name!r}. Available: {', '.join(REPRESENTATIONS)}."
        ) from error
