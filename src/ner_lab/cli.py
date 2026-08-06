"""Command-line entry point: `ner-lab run <config.yaml>`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from ner_lab import __version__

TASKS: dict[str, Callable[..., Any]] = {}

OVERRIDES = ("seed", "output_dir")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(config).__name__}.")

    return config


def resolve_task(name: str | None) -> Callable[..., Any]:
    if name is None:
        raise ValueError(f"Config has no 'task' key. Available tasks: {', '.join(sorted(TASKS)) or 'none'}.")

    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}. Available tasks: {', '.join(sorted(TASKS)) or 'none'}.")

    return TASKS[name]


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    task = resolve_task(config.get("task"))

    parameters = {key: value for key, value in config.items() if key != "task"}

    for name in OVERRIDES:
        value = getattr(args, name)

        if value is not None:
            parameters[name] = value

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
    run_parser.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
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
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"ner-lab: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
