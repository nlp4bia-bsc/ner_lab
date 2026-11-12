"""What a saved multihead model records about itself, and loading it back."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedConfig, PreTrainedTokenizerBase

from lab.ner.inference.loading import load_state_dict, resolve_device
from lab.ner.multihead.encoding import MultiHeadEncoder, describe_multihead_encoder
from lab.ner.multihead.model import MultiHeadForTokenClassification
from lab.ner.training.trainer import read_model_encoding

ARCHITECTURE = "multihead"


def multihead_model_encoding(
    encoder: MultiHeadEncoder,
    base_model: str,
    dropout: float,
    backbone_config: PreTrainedConfig,
) -> dict[str, Any]:
    """
    What a saved multihead model states about itself, written beside it as `encoding.json`.

    `backbone_config` is the backbone's own configuration, recorded in full, so the
    saved directory rebuilds the model without the base model it was trained from.
    `base_model` is recorded for provenance only.
    """
    return {
        "model": {"base_model": base_model, "architecture": ARCHITECTURE, "dropout": dropout},
        "backbone_config": backbone_config.to_dict(),
        "encoding": describe_multihead_encoder(encoder),
    }


def load_multihead_model(
    model_dir: str | Path, device: str = "auto"
) -> tuple[MultiHeadForTokenClassification, PreTrainedTokenizerBase, dict[str, Any]]:
    """
    Load a saved multihead model, its tokenizer, and the `encoding.json` beside them.

    The backbone is built from the configuration `encoding.json` records, with no
    pretrained weights, and the fine-tuned weights are loaded over the whole model
    strictly. A head catalogue that disagrees with the weights' fails to load.
    """
    model_dir = Path(model_dir).resolve()
    description = read_model_encoding(model_dir)
    architecture = description["model"].get("architecture")

    if architecture != ARCHITECTURE:
        raise ValueError(
            f"{model_dir} holds a {architecture!r} model, not a multihead one; load it with "
            "lab.ner.inference.load_model, or predict with ner.predict_entities."
        )

    model = MultiHeadForTokenClassification(
        base_model=description["model"]["base_model"],
        entity_labels=description["encoding"]["entity_labels"],
        dropout=description["model"]["dropout"],
        backbone_config=description["backbone_config"],
    )
    state = load_state_dict(model_dir)

    if not torch.equal(state["catalogue_digest"], model.catalogue_digest):
        raise ValueError(
            f"The weights in {model_dir} were trained under a different head catalogue than "
            f"its encoding.json lists ({description['encoding']['entity_labels'][:10]}); "
            "the directory mixes files from two runs."
        )

    model.load_state_dict(state, strict=True)
    model.to(resolve_device(device))
    model.eval()

    return model, AutoTokenizer.from_pretrained(model_dir), description
