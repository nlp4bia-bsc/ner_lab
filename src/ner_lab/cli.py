"""Command-line entry point: `ner-lab run <config.yaml>`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml

from ner_lab import __version__
from ner_lab.tasks import TASKS, resolve_task

OVERRIDES = ("random_state", "output_dir")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(config).__name__}."
        )

    return config


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

    task(**parameters)


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
