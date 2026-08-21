"""Command-line entry point: `ner-lab run <config.yaml>`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import yaml

from ner_lab import __version__
from ner_lab.data.io import DEFAULT_PARQUET_COMPRESSION
from ner_lab.data.stats import compute_annotation_stats, compute_text_stats
from ner_lab.tasks import TASKS, resolve_task

if TYPE_CHECKING:
    import pandas as pd

OVERRIDES = ("random_state", "output_dir")
STATS_CHOICES = ("none", "text", "both")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(config).__name__}."
        )

    return config


def _write_stats(frame: pd.DataFrame, dataset_root: Path, stem: str) -> None:
    frame.to_json(dataset_root / f"{stem}.json", orient="records", indent=2, force_ascii=False)
    frame.to_parquet(
        dataset_root / f"{stem}.parquet",
        engine="pyarrow",
        compression=DEFAULT_PARQUET_COMPRESSION,
        index=False,
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    name = config.get("task")

    if name is None:
        raise ValueError(f"Config has no 'task' key. Available tasks: {', '.join(sorted(TASKS))}.")

    task = resolve_task(name)
    parameters = {key: value for key, value in config.items() if key != "task"}

    for override in OVERRIDES:
        value = getattr(args, override)

        if value is not None:
            parameters[override] = value

    if name != "prepare_dataset":
        task(**parameters)
        return

    stats = parameters.pop("stats", "none")
    base_model = parameters.pop("base_model", None)
    language = parameters.pop("language", None)
    normalize_labels = parameters.pop("normalize_labels", False)

    if stats not in STATS_CHOICES:
        raise ValueError(f"stats must be one of {STATS_CHOICES}, got {stats!r}.")

    result = task(**parameters)

    if stats == "none":
        return

    if base_model is None or language is None:
        raise ValueError("stats requires both base_model and language in the config.")

    _write_stats(
        compute_text_stats(result.corpus, base_model, language), result.dataset_root, "text_stats"
    )

    if stats == "both":
        _write_stats(
            compute_annotation_stats(result.corpus, base_model, normalize_labels=normalize_labels),
            result.dataset_root,
            "annotation_stats",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ner-lab",
        description="Clinical NER: corpus conversion, splitting, hyperparameter search, training, inference and evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"ner-lab {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    run_parser = subparsers.add_parser("run", help="Run the task described by a YAML config.")
    run_parser.add_argument("config", help="Path to the YAML config.")
    run_parser.add_argument(
        "--random-state",
        "--seed",
        dest="random_state",
        type=int,
        default=None,
        help="Override the config's random_state.",
    )
    run_parser.add_argument("--output-dir", default=None, help="Override the config's output_dir.")
    run_parser.set_defaults(handler=run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    try:
        args.handler(args)
    except (ValueError, FileNotFoundError, TypeError) as error:
        parser.exit(2, f"ner-lab: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
