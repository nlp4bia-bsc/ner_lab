"""One line per trial event while a sweep runs, in place of Ray Tune's tables."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from ray.tune import Callback

from lab.ner.hpo.report import format_duration, format_value

PREFIX = "[hpo]"
EVENT_WIDTH = 5


def format_configuration(configuration: Mapping[str, Any]) -> str:
    """A sampled configuration as `name=value` pairs, in the order a reader scans them."""
    return "  ".join(
        f"{name}={format_value(value)}" for name, value in sorted(configuration.items())
    )


class SweepProgress(Callback):
    """
    Prints a start line and a result line for every trial of one Ray Tune run.

    Ray's own console output is silenced (`TUNE_VERBOSITY`), which leaves a
    multi-hour sweep emitting nothing until it ends — a hung job and a slow one
    look identical. Two lines per trial is the smallest thing that tells them
    apart: the start line proves a trial was picked up and with what, the result
    line closes it with a score and an elapsed time.

    Every line is flushed, so a batch scheduler's block-buffered stdout shows the
    sweep progressing rather than dumping at exit.
    """

    def __init__(
        self,
        metric: str,
        greater_is_better: bool,
        total: int,
        stage: str = "trial",
    ) -> None:
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.total = total
        self.stage = stage
        self.positions: dict[str, int] = {}
        self.starts: dict[str, float] = {}
        self.best: float | None = None

    def on_trial_start(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        self.positions[trial.trial_id] = len(self.positions) + 1
        self.starts[trial.trial_id] = time.monotonic()

        self.announce(trial, "start", format_configuration(trial.config))

    def on_trial_result(
        self, iteration: int, trials: list, trial: Any, result: Mapping[str, Any], **info: Any
    ) -> None:
        self.announce(trial, "done", self.outcome(trial, result))

    def on_trial_error(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        self.announce(trial, "error", f"after {self.elapsed(trial)} — see the raised traceback")

    def elapsed(self, trial: Any) -> str:
        """How long this trial has been running, as of now."""
        started = self.starts.get(trial.trial_id)

        return "?" if started is None else format_duration(time.monotonic() - started)

    def outcome(self, trial: Any, result: Mapping[str, Any]) -> str:
        """The result line's body: what the trial scored, over how many seeds, and the best yet."""
        score = result.get(self.metric)
        spread = result.get(f"{self.metric}_std")
        seeds = result.get("n_seeds")
        parts: list[str] = []

        if result.get("oom"):
            parts.append("OOM — scored worst")
        elif score is not None:
            scored = f"{self.metric} {format_value(score)}"
            parts.append(scored if spread is None else f"{scored} ±{format_value(spread)}")

        if seeds:
            seeds = int(seeds)
            parts.append(f"{seeds} seed{'' if seeds == 1 else 's'}")

        parts.append(self.elapsed(trial))

        if isinstance(score, (int, float)) and math.isfinite(score):
            better = (
                self.best is None
                or (score > self.best if self.greater_is_better else score < self.best)
            )

            if better:
                self.best = float(score)

            parts.append(f"best {format_value(self.best)}")

        return "  ".join(parts)

    def announce(self, trial: Any, event: str, message: str) -> None:
        position = self.positions.get(trial.trial_id, len(self.positions))
        counter = f"{position:>{len(str(self.total))}}/{self.total}"

        print(
            f"{PREFIX} {self.stage} {counter} {event.ljust(EVENT_WIDTH)}  {message}",
            flush=True,
        )
