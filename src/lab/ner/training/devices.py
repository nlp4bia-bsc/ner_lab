"""How many GPUs a single run is allowed to see."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from transformers import TrainingArguments

BANNER = "!" * 78

DevicePolicy = Literal[1, "all"]


def apply_device_policy(
    arguments: TrainingArguments,
    devices: DevicePolicy = 1,
    context: str | None = None,
) -> int:
    """
    Hold a run to one GPU, or loudly hand it every GPU it can see.

    `per_device_train_batch_size` is per device, so the batch HuggingFace actually
    trains at is `micro x devices x accumulation`. `Trainer` reaches that batch by
    wrapping the model in `nn.DataParallel`, not by taking more steps, so every
    extra device both multiplies the batch and divides the step count. A fractional
    `warmup_steps` is a ratio of a schedule that is now shorter, a cosine decay is
    re-sized against it, and epoch-counted early stopping fires after the same
    number of epochs but far fewer optimizer updates. A configuration is only the
    configuration that was tuned once the device count matches the one it was tuned
    on, which is why `1` is the default.

    `1` pins the run to `cuda:0` however many devices are visible, by forcing
    `arguments._n_gpu` — the one flag `Trainer` reads to decide whether to wrap in
    `nn.DataParallel`, and the one HuggingFace itself forces down under model
    parallelism. Reading `arguments.device` first resolves and caches
    `_setup_devices`, so nothing recomputes `_n_gpu` from the visible count
    afterwards. This works whichever devices torch has already seen, so it needs no
    cooperation from `CUDA_VISIBLE_DEVICES` and none from the calling process.

    `"all"` trades the tuned schedule for throughput: the run spreads over every
    visible device and warns with the batch size it will really use. It cannot be
    equivalent to a single-device run — the searched hyperparameters stop describing
    it. Anything between one and all is not offered because `Trainer` wraps with a
    bare `nn.DataParallel(model)`, which replicates over every visible device no
    matter what `_n_gpu` says; narrow the visible set instead.

    `context` appends a caller-specific line to the warning.

    Returns the number of GPUs the run will train on, which is 0 under `use_cpu` or
    without CUDA, where `Trainer` never reaches the `DataParallel` path.
    """
    import torch

    if devices != 1 and devices != "all":
        raise ValueError(
            f"devices must be 1 to pin the run to a single GPU, or 'all' to spread it "
            f"over every visible one, not {devices!r}. Trainer replicates over the "
            f"whole visible set, so a count in between is not something it can honour "
            f"— set CUDA_VISIBLE_DEVICES to the devices you want and pass 'all'."
        )

    if arguments.use_cpu or not torch.cuda.is_available():
        return 0

    visible = torch.cuda.device_count()

    if devices == 1:
        _ = arguments.device
        arguments._n_gpu = min(visible, 1)

        return arguments._n_gpu

    if visible > 1:
        warnings.warn(
            f"\n{BANNER}\n{multi_device_message(visible, arguments, context)}\n"
            f"Proceeding on all {visible} because devices='all': this run is valid, "
            f"but it is no longer the configuration that was tuned.\n{BANNER}",
            RuntimeWarning,
            stacklevel=2,
        )

    return visible


def require_single_device(
    arguments: TrainingArguments | None = None,
    context: str | None = None,
) -> int:
    """
    Refuse a run that can see more than one GPU, rather than adapting to it.

    This is for callers whose one device is an invariant of how they were launched,
    not a preference — an `lab.ner.hpo` trial holds a one-GPU Ray reservation, so a
    second visible device means the reservation was bypassed and the sweep's whole
    accounting is wrong. Pinning would hide that. `apply_device_policy` is the one
    to reach for when the caller is merely choosing.

    `context` appends a caller-specific line to the error.

    Returns the number of GPUs the run will train on, which is 0 under `use_cpu`
    or without CUDA, where `Trainer` never reaches the `DataParallel` path.
    """
    import torch

    if (arguments is not None and arguments.use_cpu) or not torch.cuda.is_available():
        return 0

    visible = torch.cuda.device_count()

    if visible <= 1:
        return visible

    raise RuntimeError(
        f"{multi_device_message(visible, arguments, context)}\nSet CUDA_VISIBLE_DEVICES "
        f"to a single device to train the configuration that was tuned."
    )


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
        f"{visible} GPUs are visible to this run.",
        f"HuggingFace will wrap the model in nn.DataParallel and train at {batch}.",
        f"Steps per epoch fall {visible}x with it, so a fractional warmup_steps, a "
        f"cosine decay and epoch-counted early stopping all resolve against a "
        f"schedule {visible}x shorter than the one these hyperparameters were chosen on.",
    ]

    if context:
        lines.append(context)

    return "\n".join(lines)
