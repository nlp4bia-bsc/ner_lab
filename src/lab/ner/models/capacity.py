"""Checking that a built model accepts the sequence length its rows are encoded at."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lab.ner.encoding.rows import special_token_template

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

PREFIX = "[check]"


def check_max_length(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    base_model: str,
    report: bool = True,
) -> None:
    """
    Run one forward pass at `max_length` tokens, and raise if the model cannot take it.

    The input is a full-length window as `build_row` assembles one — the tokenizer's
    special tokens around content — so absolute position embeddings, the RoBERTa offset
    past the padding index, and any architecture's own limit are all exercised. Reading
    the limit off the config would need per-architecture knowledge; running the model
    needs none.

    Call it before the model is moved to an accelerator: on CPU an out-of-range position
    raises a readable error, whereas on CUDA it surfaces as a device-side assert at
    whichever later call happens to synchronize. The pass runs in eval mode without
    gradients, so it draws no random numbers, and the model's mode is restored.

    `report` prints a line saying the check passed, so the run's log shows it was made.
    """
    import torch

    parameter = next(model.parameters(), None)

    if parameter is not None and parameter.device.type != "cpu":
        raise ValueError(
            f"check_max_length runs on CPU, but {base_model} is on {parameter.device}. "
            "Call it before moving the model to a device."
        )

    prefix_ids, suffix_ids = special_token_template(tokenizer)
    content_id = tokenizer.encode("a", add_special_tokens=False)[0]
    content_length = max_length - len(prefix_ids) - len(suffix_ids)
    input_ids = torch.tensor([[*prefix_ids, *[content_id] * content_length, *suffix_ids]])

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    except (IndexError, RuntimeError) as error:
        limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)

        raise ValueError(
            f"{base_model} cannot take sequences of max_length={max_length}: a forward pass "
            f"on CPU failed with {type(error).__name__}: {error}. Its config declares "
            f"max_position_embeddings={limit}. Lower `max_length` (`max_lengths` in a sweep)."
        ) from error
    finally:
        model.train(was_training)

    if report:
        print(
            f"{PREFIX} {base_model}: a forward pass at max_length={max_length} passed",
            flush=True,
        )
