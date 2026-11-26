"""What a saved multihead model records about itself, and loading it back."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoTokenizer, PreTrainedConfig, PreTrainedTokenizerBase

from lab.ner.inference.loading import load_state_dict, resolve_device
from lab.ner.multihead.crf import MultiHeadCRFForTokenClassification
from lab.ner.multihead.encoding import MultiHeadEncoder, describe_multihead_encoder
from lab.ner.multihead.model import MultiHeadForTokenClassification
from lab.ner.training.trainer import read_model_encoding

MultiHeadArchitecture = Literal["linear", "crf"]

ARCHITECTURES: dict[str, str] = {"linear": "multihead", "crf": "multihead_crf"}
CHAIN_SETTINGS = ("constrain_transitions",)
MODEL_CLASSES: dict[str, type[MultiHeadForTokenClassification]] = {
    "multihead": MultiHeadForTokenClassification,
    "multihead_crf": MultiHeadCRFForTokenClassification,
}


def recorded_architecture(architecture: str) -> str:
    """
    The name `encoding.json` records for a multihead `architecture`: `"linear"` or `"crf"`.

    `"multihead"` for the cross-entropy heads, `"multihead_crf"` for a chain per head,
    one field either way (D160). Any other value raises, before anything is trained.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown multihead architecture {architecture!r}; expected one of "
            f"{tuple(ARCHITECTURES)}."
        )

    return ARCHITECTURES[architecture]


def chain_settings(
    architecture: MultiHeadArchitecture, constrain_transitions: bool | None
) -> dict[str, Any]:
    """
    The settings a multihead `architecture` takes beyond the heads, as model keyword arguments.

    A CRF takes `constrain_transitions`, `True` when left out; the linear heads take
    none, so passing it with them raises rather than being silently ignored.
    """
    if recorded_architecture(architecture) == "multihead_crf":
        return {"constrain_transitions": constrain_transitions is not False}

    if constrain_transitions is not None:
        raise ValueError(
            "constrain_transitions applies to a multihead CRF only; leave it out, or pass "
            "architecture='crf'."
        )

    return {}


def multihead_model_encoding(
    encoder: MultiHeadEncoder,
    base_model: str,
    dropout: float,
    backbone_config: PreTrainedConfig,
    architecture: MultiHeadArchitecture = "linear",
    constrain_transitions: bool | None = None,
) -> dict[str, Any]:
    """
    What a saved multihead model states about itself, written beside it as `encoding.json`.

    `backbone_config` is the backbone's own configuration, recorded in full, so the
    saved directory rebuilds the model without the base model it was trained from.
    `base_model` is recorded for provenance only; `architecture` as
    `recorded_architecture` names it, beside a CRF's `chain_settings`.
    """
    return {
        "model": {
            "base_model": base_model,
            "architecture": recorded_architecture(architecture),
            "dropout": dropout,
            **chain_settings(architecture, constrain_transitions),
        },
        "backbone_config": backbone_config.to_dict(),
        "encoding": describe_multihead_encoder(encoder),
    }


def load_multihead_model(
    model_dir: str | Path, device: str = "auto"
) -> tuple[MultiHeadForTokenClassification, PreTrainedTokenizerBase, dict[str, Any]]:
    """
    Load a saved multihead model, its tokenizer, and the `encoding.json` beside them.

    The class is the one `encoding.json`'s architecture names — the cross-entropy heads
    or a chain per head, with the chain settings it records. The backbone is built from
    the configuration it records, with no pretrained weights, and the fine-tuned weights are loaded over the whole model
    strictly. A head catalogue that disagrees with the weights' fails to load.
    """
    model_dir = Path(model_dir).resolve()
    description = read_model_encoding(model_dir)
    architecture = description["model"].get("architecture")

    if architecture not in MODEL_CLASSES:
        raise ValueError(
            f"{model_dir} holds a {architecture!r} model, not a multihead one; load it with "
            "lab.ner.inference.load_model, or predict with ner.predict_entities."
        )

    model = MODEL_CLASSES[architecture](
        base_model=description["model"]["base_model"],
        entity_labels=description["encoding"]["entity_labels"],
        dropout=description["model"]["dropout"],
        backbone_config=description["backbone_config"],
        **{key: description["model"][key] for key in CHAIN_SETTINGS if key in description["model"]},
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
