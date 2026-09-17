"""Synthetic corpora and optional real-corpus locations for the verification scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

SAMPLES_ENV = "NER_LAB_SAMPLES"
DEFAULT_SAMPLES = Path.home() / "bsc" / "NER-API" / "data_samples"

SENTENCES = [
    "El paciente refiere {mention} desde hace tres semanas.",
    "Se observa {mention} en la exploracion fisica realizada hoy.",
    "Antecedentes personales: {mention}, sin otras alteraciones.",
    "Tras el tratamiento persiste {mention} de intensidad moderada.",
]

MENTIONS = ["neumonia bilateral", "fiebre alta", "dolor toracico", "hipertension arterial"]


def samples_root() -> Path | None:
    """Locate the real sample corpora, or None when they are not available here."""
    root = Path(os.environ.get(SAMPLES_ENV, DEFAULT_SAMPLES))

    return root if root.is_dir() else None


def synthetic_documents(n_documents: int = 24, label: str = "DISEASE") -> tuple[dict, pd.DataFrame]:
    """
    Build a deterministic BRAT-shaped corpus: `{doc_id: text}` plus an annotation frame.

    Every document gets between one and three entities, so entity-count
    stratification has something uneven to balance.
    """
    documents: dict[str, str] = {}
    rows = []

    for index in range(n_documents):
        doc_id = f"doc_{index:03d}"
        n_entities = 1 + index % 3
        text = ""

        for entity_index in range(n_entities):
            mention = MENTIONS[(index + entity_index) % len(MENTIONS)]
            template = SENTENCES[(index + entity_index) % len(SENTENCES)]
            start = len(text) + template.index("{mention}")

            text += template.format(mention=mention) + " "
            rows.append(
                {
                    "filename": doc_id,
                    "mark": f"T{entity_index + 1}",
                    "label": label,
                    "start_span": start,
                    "end_span": start + len(mention),
                    "text": mention,
                }
            )

        documents[doc_id] = text.rstrip()

    return documents, pd.DataFrame(rows)


def synthetic_corpus(n_documents: int = 24, label: str = "DISEASE") -> pd.DataFrame:
    """Build a canonical corpus DataFrame directly, bypassing the BRAT readers."""
    from lab.core import build_corpus

    documents, annotations = synthetic_documents(n_documents, label)

    return build_corpus(documents, annotations)


def write_brat(root: Path, documents: dict[str, str], annotations: pd.DataFrame) -> tuple[Path, Path]:
    """Write a synthetic corpus to disk as .txt and .ann directories."""
    txt_dir = root / "txt"
    ann_dir = root / "ann"
    txt_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, text in documents.items():
        (txt_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

    for doc_id, group in annotations.groupby("filename"):
        lines = [
            f"{row.mark}\t{row.label} {row.start_span} {row.end_span}\t{row.text}"
            for row in group.itertuples(index=False)
        ]
        (ann_dir / f"{doc_id}.ann").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return txt_dir, ann_dir


def entities_of(corpus: pd.DataFrame, doc_id: str) -> list[dict]:
    """Read one document's entity list back out of a canonical corpus."""
    row = corpus.loc[corpus["doc_id"] == doc_id].iloc[0]

    return json.loads(row["entities_json"])


def tiny_base_model(root: Path, tokenizer) -> Path:
    """
    Write a randomly initialized miniature BERT backbone to disk and return its path.

    Lets the model and training checks exercise the real `from_pretrained` path
    without downloading a full backbone, and keeps a CPU training run to seconds.
    """
    from transformers import AutoModel, BertConfig

    base_model = root / "tiny-bert"

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )

    AutoModel.from_config(config).save_pretrained(base_model)
    tokenizer.save_pretrained(base_model)

    return base_model
