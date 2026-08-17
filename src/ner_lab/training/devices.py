"""How many GPUs a single run is allowed to see."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import TrainingArguments

BANNER = "!" * 78


def require_single_device(
    arguments: TrainingArguments | None = None,
    allow_multi_device: bool = False,
    context: str | None = None,
) -> int:
    """
    Refuse — or loudly permit — a run that can see more than one GPU.

    `per_device_train_batch_size` is per device, so the batch HuggingFace actually
    trains at is `micro x devices x accumulation`. `Trainer` reaches that batch by
    wrapping the model in `nn.DataParallel`, not by taking more steps, so every
    extra visible device both multiplies the batch and divides the step count. A
    fractional `warmup_steps` is a ratio of a schedule that is now shorter, a
    cosine decay is re-sized against it, and epoch-counted early stopping fires
    after the same number of epochs but far fewer optimizer updates. A
    configuration is only the configuration that was tuned once the device count
    matches the one it was tuned on.

    `allow_multi_device` trades that away for throughput: the run proceeds and
    warns with the batch size it will really use. It cannot make the run
    equivalent to a single-device one — the searched hyperparameters stop
    describing it — so the default is to raise.

    `context` appends a caller-specific line to whichever of the two is emitted.

    Returns the number of GPUs the run will train on, which is 0 under `use_cpu`
    or without CUDA, where `Trainer` never reaches the `DataParallel` path.
    """
    import torch

    if (arguments is not None and arguments.use_cpu) or not torch.cuda.is_available():
        return 0

    visible = torch.cuda.device_count()

    if visible <= 1:
        return visible

    message = multi_device_message(visible, arguments, context)

    if not allow_multi_device:
        raise RuntimeError(
            f"{message}\nSet CUDA_VISIBLE_DEVICES to a single device, before torch "
            f"initializes CUDA, to train the configuration that was tuned — or pass "
            f"allow_multi_device=True to accept the batch size above."
        )

    warnings.warn(
        f"\n{BANNER}\n{message}\nProceeding on all {visible} because "
        f"allow_multi_device=True: this run is valid, but it is no longer the "
        f"configuration that was tuned.\n{BANNER}",
        RuntimeWarning,
        stacklevel=2,
    )

    return visible


def effective_train_batch_size(arguments: TrainingArguments, device_count: int) -> int:
    """
    The batch one optimizer step really covers: `micro x devices x accumulation`.

    `per_device_train_batch_size` names only the first factor, so this is the number
    a learning rate was actually tuned against and the number that sets how many
    steps an epoch takes. A `device_count` of 0 — CPU, or no CUDA — still trains one
    batch per step, so it counts as one device.
    """
    return (
        arguments.per_device_train_batch_size
        * max(device_count, 1)
        * arguments.gradient_accumulation_steps
    )


def multi_device_message(
    visible: int,
    arguments: TrainingArguments | None = None,
    context: str | None = None,
) -> str:
    """What `visible` devices will do to the batch size and the schedule, in words."""
    batch = f"per_device_train_batch_size x {visible} devices"

    if arguments is not None:
        batch = (
            f"batch {arguments.per_device_train_batch_size} x {visible} devices x "
            f"{arguments.gradient_accumulation_steps} accumulation = "
            f"{effective_train_batch_size(arguments, visible)}, not the "
            f"{effective_train_batch_size(arguments, 1)} that "
            f"per_device_train_batch_size names"
        )

    lines = [
        f"{visible} GPUs are visible to this run, which is meant to see at most one.",
        f"HuggingFace will wrap the model in nn.DataParallel and train at {batch}.",
        f"Steps per epoch fall {visible}x with it, so a fractional warmup_steps, a "
        f"cosine decay and epoch-counted early stopping all resolve against a "
        f"schedule {visible}x shorter than the one these hyperparameters were chosen on.",
    ]

    if context:
        lines.append(context)

    return "\n".join(lines)
