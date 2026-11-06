"""Loading a saved model: its weights, its tokenizer, the encoding it was trained under."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lab.ner.models.registry import Architecture, build_model
from lab.ner.training.trainer import read_model_encoding


def load_model(
    model_dir: str | Path,
    device: str = "auto",
    architecture: str | Architecture | None = None,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase, dict[str, Any]]:
    """
    Load a saved model, its tokenizer, and the `encoding.json` beside them.

    A linear model is a standard `from_pretrained` load. A CRF model is not a
    `PreTrainedModel`, so the Trainer saved it as a bare state dict: it is
    rebuilt over the backbone `encoding.json` names, then the fine-tuned weights
    are loaded on top. `architecture` overrides the recorded name, which a model
    trained with a custom architecture callable needs.
    """
    model_dir = Path(model_dir).resolve()
    description = read_model_encoding(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    label2id = description["encoding"]["label2id"]
    id2label = {int(index): label for label, index in label2id.items()}

    recorded = description["model"]
    architecture = recorded["architecture"] if architecture is None else architecture

    if architecture == "multihead":
        _validate_multihead_description(description)
        settings = dict(recorded.get("architecture_kwargs", {}))
        settings["backbone_config"] = description["backbone_config"]
        model = build_model(
            recorded["base_model"], label2id, id2label, architecture="multihead", **settings
        )
        try:
            state = load_state_dict(model_dir)
            if not torch.equal(
                state.get("head_catalogue_digest", torch.empty(0)), model.head_catalogue_digest
            ):
                raise ValueError("Checkpoint head order disagrees with encoding.json.")
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Incompatible multihead checkpoint in {model_dir}: {error}"
            ) from error
    elif architecture == "linear":
        model = build_model(
            base_model=str(model_dir),
            label2id=label2id,
            id2label=id2label,
            architecture=architecture,
            **recorded.get("architecture_kwargs", {}),
        )
    else:
        model = build_model(
            base_model=recorded["base_model"],
            label2id=label2id,
            id2label=id2label,
            architecture=architecture,
            **recorded.get("architecture_kwargs", {}),
        )
        model.load_state_dict(load_state_dict(model_dir))

    model.to(resolve_device(device))
    model.eval()

    return model, tokenizer, description


def load_state_dict(model_dir: str | Path) -> dict[str, Any]:
    """Read the weights the Trainer wrote for a model it could not `save_pretrained`."""
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"
    binary_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    if binary_path.exists():
        return torch.load(binary_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {model_dir}.")


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve `"auto"` against what this machine actually has."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested but CUDA is not available.")

    return torch.device(device)


def _validate_multihead_description(description: dict) -> None:
    from lab.ner.encoding.tagging import validate_entity_labels

    labels = validate_entity_labels(description.get("entity_labels", []))
    expected = {str(index): label for index, label in enumerate(labels)}
    actual = {str(index): label for index, label in description.get("head_to_label", {}).items()}
    settings = description["model"].get("architecture_kwargs", {})
    if (
        description.get("format_version") != 2
        or description.get("num_heads") != len(labels)
        or actual != expected
        or description.get("bio_schema") != {"O": 0, "B": 1, "I": 2}
        or description["encoding"].get("entity_labels") != labels
        or settings.get("entity_labels") != labels
        or "backbone_config" not in description
    ):
        raise ValueError("Incompatible multihead encoding.json: head catalogue or schema mismatch.")
