"""Building a token-classification model from a name or from your own factory."""

from __future__ import annotations

from typing import Any, Callable

from transformers import AutoModelForTokenClassification

from lab.ner.models.bio import normalize_id2label, normalize_label2id

Architecture = Callable[..., Any]

BUILTIN_ARCHITECTURES = ("linear", "crf")


def build_model(
    base_model: str,
    label2id: dict,
    id2label: dict,
    architecture: str | Architecture = "linear",
    **architecture_kwargs: Any,
):
    """
    Build a token-classification model over a pretrained base model.

    `architecture` is `"linear"` (a linear head over the encoder), `"crf"`, or any
    callable taking `(base_model, label2id, id2label, **kwargs)` and returning a
    model whose `forward` accepts `input_ids`/`attention_mask`/`labels` and returns
    an object with `loss` and `logits`.

    Pass `label2id` and `id2label` from the `Encoder`'s vocabulary, so the label
    ids the model is configured with are the ones the rows were encoded against.
    """
    label2id = normalize_label2id(label2id)
    id2label = normalize_id2label(id2label)

    if callable(architecture):
        return architecture(base_model, label2id, id2label, **architecture_kwargs)

    if architecture not in BUILTIN_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; expected one of {BUILTIN_ARCHITECTURES} "
            "or a callable."
        )

    if architecture == "crf":
        from lab.ner.models.crf import CRFForTokenClassification

        return CRFForTokenClassification(base_model, label2id, id2label, **architecture_kwargs)

    return build_linear_model(base_model, label2id, id2label, **architecture_kwargs)


def build_linear_model(base_model: str, label2id: dict, id2label: dict, **kwargs: Any):
    """A pretrained encoder with the stock `AutoModelForTokenClassification` head."""
    return AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        label2id=normalize_label2id(label2id),
        id2label=normalize_id2label(id2label),
        **kwargs,
    )
