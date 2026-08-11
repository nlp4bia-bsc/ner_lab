"""Per-epoch metric capture and per-run resource accounting."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import pandas as pd
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

if TYPE_CHECKING:
    from datasets import Dataset

_silent_logger = logging.getLogger("ner_lab.training.tracking")
_silent_logger.addHandler(logging.NullHandler())
_silent_logger.propagate = False
_silent_logger.setLevel(logging.CRITICAL + 1)


class EpochMetricsLogger(TrainerCallback):
    """
    Buffer one row per (epoch, split) as the Trainer evaluates.

    With `scope="both"`, an extra pass over the training set runs inside
    `on_evaluate` — the only moment that epoch's weights are still in memory,
    since per-epoch checkpoints are not kept.
    """

    def __init__(
        self,
        trainer: Trainer,
        scope: str = "eval",
        train_dataset: Dataset | None = None,
        train_compute_metrics: Callable[[Any], dict] | None = None,
    ) -> None:
        self.trainer = trainer
        self.scope = scope
        self.train_dataset = train_dataset
        self.train_compute_metrics = train_compute_metrics
        self.rows: list[dict[str, Any]] = []
        self._in_train_pass = False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._in_train_pass:
            return

        self._append("eval", metrics or {}, state.epoch, state.global_step)

        if self.scope == "both" and self.train_dataset is not None:
            original = self.trainer.compute_metrics
            self.trainer.compute_metrics = self.train_compute_metrics
            self._in_train_pass = True

            try:
                train_metrics = self.trainer.evaluate(
                    eval_dataset=cast(Any, self.train_dataset), metric_key_prefix="train"
                )
            finally:
                self.trainer.compute_metrics = original
                self._in_train_pass = False

            self._append("train", train_metrics, state.epoch, state.global_step)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def write(self, path: str | Path) -> Path | None:
        """Flush the buffered rows to parquet, or do nothing if there are none."""
        if not self.rows:
            return None

        path = Path(path)
        self.frame().to_parquet(path, index=False)

        return path

    def _append(self, split: str, metrics: dict, epoch: float | None, step: int) -> None:
        prefix = f"{split}_"
        row: dict[str, Any] = {"split": split, "epoch": epoch, "step": step}

        for key, value in metrics.items():
            row[key[len(prefix):] if key.startswith(prefix) else key] = value

        self.rows.append(row)


def visible_gpu_ids() -> list[int] | None:
    """
    The GPU ids this process actually owns, from `CUDA_VISIBLE_DEVICES`.

    Ray sets that per trial, but NVML-based enumeration ignores it and reports
    every physical GPU — which on a shared node attributes other trials' power
    draw to this one.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if not visible:
        return None

    return [int(gpu_id) for gpu_id in visible.split(",") if gpu_id.strip().isdigit()]


def gpu_hardware_info() -> dict[str, Any]:
    """Static GPU identity for this process's visible devices."""
    import torch

    if not torch.cuda.is_available():
        return {"gpu_count": 0, "gpu_model": None}

    gpu_ids = visible_gpu_ids() or list(range(torch.cuda.device_count()))

    import pynvml

    pynvml.nvmlInit()

    try:
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(gpu_ids[0]))

        if isinstance(name, bytes):
            name = name.decode()

        return {"gpu_count": len(gpu_ids), "gpu_model": name}
    finally:
        pynvml.nvmlShutdown()


class ResourceTracker:
    """
    Wall-clock, energy, emissions and peak VRAM for one block of work.

    Peak VRAM comes from torch's allocator; everything else from codecarbon's
    sampling thread. `tracking_mode="process"` plus explicit `gpu_ids` keep the
    numbers attributed to this process, not the whole node.
    """

    def __init__(self, measure_power_secs: int = 1) -> None:
        import torch
        from codecarbon import EmissionsTracker, OutputMethod
        from codecarbon.output_methods.logger import LoggerOutput

        self._torch = torch
        self._gpu_available = torch.cuda.is_available()

        self._tracker = EmissionsTracker(
            tracking_mode="process",
            gpu_ids=visible_gpu_ids(),
            output_methods=[OutputMethod.LOGGER],
            logging_logger=LoggerOutput(logger=_silent_logger),
            allow_multiple_runs=True,
            log_level="error",
            measure_power_secs=measure_power_secs,
        )

    def start(self) -> None:
        if self._gpu_available:
            self._torch.cuda.reset_peak_memory_stats()

        self._tracker.start()

    def stop(self) -> dict[str, Any]:
        """Stop tracking and return one flat dict. Safe to call after a failure."""
        self._tracker.stop()
        data = self._tracker.final_emissions_data

        stats: dict[str, Any] = {
            "duration_sec": data.duration if data else None,
            "energy_kwh": data.energy_consumed if data else None,
            "emissions_kg_co2": data.emissions if data else None,
            "cpu_utilization_percent": data.cpu_utilization_percent if data else None,
            "gpu_utilization_percent": data.gpu_utilization_percent if data else None,
            "ram_used_gb": data.ram_used_gb if data else None,
            "gpu_peak_vram_allocated_gb": None,
            "gpu_peak_vram_reserved_gb": None,
        }

        if self._gpu_available:
            stats["gpu_peak_vram_allocated_gb"] = self._torch.cuda.max_memory_allocated() / 1024**3
            stats["gpu_peak_vram_reserved_gb"] = self._torch.cuda.max_memory_reserved() / 1024**3

        return stats

    def __enter__(self) -> "ResourceTracker":
        self.start()

        return self

    def __exit__(self, *exception) -> None:
        self.stats = self.stop()
