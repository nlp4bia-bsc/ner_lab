"""A readable summary of a finished sweep: what won, whether to believe it, what it cost."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

EPOCH_METRICS_GLOB = "seed_*/epoch_metrics.parquet"

EDGE_FRACTION = 0.05
TIE_SET_NOTE_FRACTION = 0.5
OOM_NOTE_THRESHOLD = 0.1
EPOCH_CAP_NOTE_THRESHOLD = 0.25

RULE_WIDTH = 66
BODY_WIDTH = 96
SMALL_UNIT_THRESHOLD = 0.01


def format_value(value: Any) -> str:
    """A sampled value as it should read in the winner block."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"

    return f"{value:.4g}"


def format_difference(value: float) -> str:
    """
    A gap or an across-seed spread, kept visible when it is small enough to matter.

    Scores keep a fixed four decimals so two of them line up and compare by eye.
    The differences between them do not: a gap of 3e-05 is the number deciding
    whether a winner is real, and `0.0000` reads as though there were none.
    """
    if value == 0:
        return "0"

    return f"{value:.1e}" if abs(value) < 1e-4 else f"{value:.4f}"


def format_duration(seconds: float) -> str:
    """A duration as the two coarsest units that describe it."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m"

    if minutes:
        return f"{minutes}m{seconds:02d}s"

    return f"{seconds}s"


def format_quantity(value: float, unit: str, scale: float, small_unit: str) -> str:
    """
    A physical total in whichever of its two units keeps significant digits on show.

    A sweep's emissions run to fractions of a kilogram, so a fixed `kg` reading
    rounds most of them to zero; the same number in grams reads at a glance.
    """
    if value == 0:
        return f"0 {unit}"

    if abs(value) < SMALL_UNIT_THRESHOLD:
        return f"{value * scale:.4g} {small_unit}"

    return f"{value:.4g} {unit}"


def trial_outcomes(trials: pd.DataFrame, metric: str) -> dict[str, pd.Series]:
    """Boolean masks splitting the trial table into errored, out-of-memory and completed."""
    errored = (
        trials["error"].notna() if "error" in trials.columns else pd.Series(False, trials.index)
    )
    oom = (
        trials["oom"].fillna(False).astype(bool)
        if "oom" in trials.columns
        else pd.Series(False, trials.index)
    )
    oom = oom & ~errored
    scored = trials[metric].notna() & ~trials[metric].isin([math.inf, -math.inf])

    return {"errored": errored, "oom": oom, "completed": ~errored & ~oom & scored}


def noise_scale(completed: pd.DataFrame, metric: str) -> float | None:
    """
    The typical across-seed spread of a trial score, as the tie set's yardstick.

    The median over trials rather than the winner's own value, which is one draw of
    the same noisy quantity and can be small by luck.
    """
    column = f"{metric}_std"

    if column not in completed.columns:
        return None

    spreads = completed.loc[completed.get("n_seeds", 0) > 1, column].dropna()
    spreads = spreads[spreads > 0]

    return float(spreads.median()) if not spreads.empty else None


def range_position(value: Any, specification: Mapping[str, Any]) -> float | None:
    """Where `value` sits in a dimension's declared range: 0.0 at the low bound, 1.0 at the high."""
    low, high = specification.get("low"), specification.get("high")

    if low is None or high is None or high <= low or not isinstance(value, (int, float)):
        return None

    if specification.get("type") == "loguniform" and low > 0 and value > 0:
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))

    return (value - low) / (high - low)


def edge_annotation(value: Any, specification: Mapping[str, Any]) -> str:
    """A note on a winning value that sits against the boundary of what was searched."""
    position = range_position(value, specification)

    if position is None:
        return ""

    bounds = f"[{format_value(specification['low'])}, {format_value(specification['high'])}]"

    if position >= 1 - EDGE_FRACTION:
        return f"   (top {max(1, round((1 - position) * 100))}% of {bounds})"

    if position <= EDGE_FRACTION:
        return f"   (bottom {max(1, round(position * 100))}% of {bounds})"

    return ""


def split_dimensions(
    search_space: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Search dimensions split into continuous and categorical, dropping pinned ones."""
    continuous: list[str] = []
    categorical: list[str] = []

    for name, specification in search_space.items():
        if specification.get("type") == "choice":
            if len(specification.get("categories", [])) > 1:
                categorical.append(name)
        elif specification.get("low") is not None:
            continuous.append(name)

    return continuous, categorical


def dimension_effects(
    completed: pd.DataFrame,
    metric: str,
    search_space: Mapping[str, Mapping[str, Any]],
    greater_is_better: bool,
) -> list[tuple[str, str]]:
    """
    Per dimension, whether it moved the metric across the trials that finished.

    Continuous dimensions get Spearman's rho — a rank correlation, so a loguniform
    axis needs no linearisation and a single outlying trial cannot carry it.
    Categorical ones get the mean, spread and count behind each category.
    """
    continuous, categorical = split_dimensions(search_space)
    effects: list[tuple[str, str]] = []

    for name in continuous:
        if name not in completed.columns or completed[name].nunique() < 2 or len(completed) < 3:
            continue

        correlation = completed[[name, metric]].corr(method="spearman").iloc[0, 1]

        if pd.notna(correlation):
            effects.append((name, f"rho {correlation:+.2f}"))

    for name in categorical:
        if name not in completed.columns:
            continue

        grouped = (
            completed.groupby(name)[metric]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=not greater_is_better)
        )
        described = " · ".join(
            f"{category} {row['mean']:.4f}"
            + (f" ±{format_difference(row['std'])}" if pd.notna(row["std"]) else "")
            + f" ({int(row['count'])})"
            for category, row in grouped.iterrows()
        )
        effects.append((name, described))

    return effects


def convergence_stats(
    trials: pd.DataFrame, metric: str, epoch_cap: int, greater_is_better: bool
) -> dict[str, Any] | None:
    """
    How far the seed runs behind each trial actually got, read from their epoch metrics.

    A sweep whose runs mostly reach the epoch cap scored them before they converged,
    and so ranked how fast a configuration improves rather than how good it becomes.
    """
    if "trial_path" not in trials.columns:
        return None

    runs = 0
    at_cap = 0
    best_epochs: list[int] = []

    for trial_path in trials["trial_path"].dropna():
        for path in sorted(Path(trial_path).glob(EPOCH_METRICS_GLOB)):
            try:
                frame = pd.read_parquet(path)
            except (OSError, ValueError):
                continue

            if "split" in frame.columns:
                frame = frame[frame["split"] == "eval"]

            if frame.empty:
                continue

            runs += 1
            at_cap += int(len(frame) >= epoch_cap)

            if metric in frame.columns and frame[metric].notna().any():
                values = frame[metric].reset_index(drop=True)
                best = values.idxmax() if greater_is_better else values.idxmin()
                best_epochs.append(int(best) + 1)

    if not runs:
        return None

    return {
        "runs": runs,
        "at_cap": at_cap,
        "median_best_epoch": int(pd.Series(best_epochs).median()) if best_epochs else None,
    }


def cost_lines(trials: pd.DataFrame) -> list[tuple[str, str]]:
    """Totals for what the whole sweep spent, out-of-memory trials included."""
    lines: list[tuple[str, str]] = []

    def total(column: str) -> float | None:
        if column not in trials.columns:
            return None

        values = trials[column].dropna()

        return float(values.sum()) if not values.empty else None

    def peak(column: str) -> float | None:
        if column not in trials.columns:
            return None

        values = trials[column].dropna()

        return float(values.max()) if not values.empty else None

    duration = total("duration_sec")

    if duration is not None:
        lines.append(("wall time", format_duration(duration)))

    energy, emissions = total("energy_kwh"), total("emissions_kg_co2")

    if energy is not None or emissions is not None:
        parts = []

        if energy is not None:
            parts.append(format_quantity(energy, "kWh", 1000, "Wh"))
        if emissions is not None:
            parts.append(format_quantity(emissions, "kg CO2", 1000, "g CO2"))

        lines.append(("energy", " · ".join(parts)))

    allocated, reserved = peak("gpu_peak_vram_allocated_gb"), peak("gpu_peak_vram_reserved_gb")

    if allocated is not None or reserved is not None:
        parts = []

        if allocated is not None:
            parts.append(f"{allocated:.2f} GB allocated")
        if reserved is not None:
            parts.append(f"{reserved:.2f} GB reserved")

        lines.append(("peak VRAM", " · ".join(parts)))

    return lines


def render_block(title: str, rows: list[tuple[str, str]]) -> list[str]:
    """One titled block of aligned label/value rows, wrapped to the body width."""
    if not rows:
        return []

    width = max(len(label) for label, _ in rows)
    lines = [f"\n{title}"]

    for label, value in rows:
        indent = 2 + width + 2
        wrapped = textwrap.wrap(
            value, width=BODY_WIDTH - indent, break_long_words=False, break_on_hyphens=False
        ) or [""]
        lines.append(f"  {label.ljust(width)}  {wrapped[0]}")
        lines.extend(f"{' ' * indent}{line}" for line in wrapped[1:])

    return lines


def summarize_sweep(
    trials: pd.DataFrame,
    metric: str,
    greater_is_better: bool,
    search_space: Mapping[str, Mapping[str, Any]],
    epoch_cap: int,
    convergence: bool = True,
) -> str:
    """
    The end-of-sweep report: the winner, how far it stands above the noise, and the cost.

    Built to answer whether the winner can be trusted and what the next sweep should
    change — hence the tie set, which counts the trials indistinguishable from the
    best at the sweep's own across-seed spread, and the edge annotations, which mark
    a winning value sitting against the boundary of the range it was searched in.
    """
    direction = "higher is better" if greater_is_better else "lower is better"
    header = [f"Sweep summary — {metric} ({direction})", "—" * RULE_WIDTH]

    masks = trial_outcomes(trials, metric)
    completed = trials[masks["completed"]]
    counts = {name: int(mask.sum()) for name, mask in masks.items()}

    header.append(
        f"Trials      {len(trials)} · {counts['completed']} completed · "
        f"{counts['oom']} OOM · {counts['errored']} errored"
    )

    if completed.empty:
        return "\n".join([*header, "\nNo trial completed; nothing to summarise."])

    ranked = completed.sort_values(metric, ascending=not greater_is_better)
    best = ranked.iloc[0]
    notes: list[str] = []

    seeds = best.get("n_seeds")
    spread = best.get(f"{metric}_std")
    identity = " · ".join(
        part
        for part in (
            str(best.get("trial_id")) if best.get("trial_id") else "",
            f"{int(seeds)} seed{'' if seeds == 1 else 's'}" if pd.notna(seeds) else "",
            f"across-seed std {format_difference(spread)}" if pd.notna(spread) else "",
        )
        if part
    )
    header.append(f"Best        {best[metric]:.4f}   {identity}")

    if len(ranked) > 1:
        gap = abs(float(best[metric]) - float(ranked.iloc[1][metric]))
        header.append(
            f"Runner-up   {ranked.iloc[1][metric]:.4f}   gap {format_difference(gap)}"
        )

    noise = noise_scale(completed, metric)

    if noise is not None:
        tied = int((abs(completed[metric] - float(best[metric])) <= noise).sum()) - 1
        header.append(
            f"Tie set     {tied} other trial{'' if tied == 1 else 's'} within one across-seed "
            f"std ({format_difference(noise)}) of the best"
        )

        share = (tied + 1) / len(completed)

        if tied and share >= TIE_SET_NOTE_FRACTION:
            notes.append(
                f"{tied + 1} of the {len(completed)} completed trials are indistinguishable "
                f"at this spread ({share:.0%}) — the ranking among them is noise, not signal"
            )

    winner_rows: list[tuple[str, str]] = []

    for name in search_space:
        if name not in best.index or pd.isna(best[name]):
            continue

        annotation = edge_annotation(best[name], search_space[name])
        winner_rows.append((name, f"{format_value(best[name])}{annotation}"))

        if annotation:
            notes.append(
                f"the winning {name} sits against the edge of its range — "
                "the optimum may lie outside it"
            )

    micro = best.get("per_device_train_batch_size")
    accumulation = best.get("gradient_accumulation_steps")

    if pd.notna(micro) and pd.notna(accumulation):
        winner_rows.append(("batch resolution", f"micro {int(micro)} x accum {int(accumulation)}"))

    sections = render_block("Winner", winner_rows)

    if convergence:
        stats = convergence_stats(trials, metric, epoch_cap, greater_is_better)

        if stats is not None:
            fraction = stats["at_cap"] / stats["runs"]
            rows = [
                ("seed runs", str(stats["runs"])),
                (
                    f"reached the {epoch_cap}-epoch cap",
                    f"{stats['at_cap']}  ({fraction:.0%})",
                ),
            ]

            if stats["median_best_epoch"] is not None:
                rows.append(("median best epoch", str(stats["median_best_epoch"])))

            sections.extend(render_block("Convergence", rows))

            if fraction >= EPOCH_CAP_NOTE_THRESHOLD:
                notes.append(
                    f"{fraction:.0%} of seed runs hit the epoch cap — trials were scored "
                    "before converging, so the sweep partly ranked convergence speed"
                )

    sections.extend(
        render_block(
            f"Dimension effects ({len(completed)} completed trials)",
            dimension_effects(completed, metric, search_space, greater_is_better),
        )
    )
    sections.extend(render_block("Cost", cost_lines(trials)))

    if counts["oom"] and counts["oom"] / len(trials) >= OOM_NOTE_THRESHOLD:
        notes.append(
            f"{counts['oom'] / len(trials):.0%} of trials ran out of memory and scored worst — "
            "the batch-size dimension or max_micro_batch_size is too generous"
        )

    sections.extend(render_block("Notes", [("·", note) for note in notes]))

    return "\n".join([*header, *sections, ""])
