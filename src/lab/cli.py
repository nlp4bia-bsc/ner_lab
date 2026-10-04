"""Command-line entry point: `lab run <config.yaml>` and `lab tasks`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from lab import __version__
from lab.core.tasks import list_tasks, resolve_task

OVERRIDES = ("random_state", "output_dir")


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config, which must be a mapping at the top level."""
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(config).__name__}."
        )

    return config


def run(args: argparse.Namespace) -> None:
    """
    `lab run`: call the config's `task` with the rest of the config as keyword arguments.

    `--random-state` and `--output-dir` override the config's values when given.
    """
    config = read_config(args.config)
    name = config.get("task")

    if name is None:
        raise ValueError("Config has no 'task' key. `lab tasks` lists the available ones.")

    task = resolve_task(name)
    parameters = {key: value for key, value in config.items() if key != "task"}

    for override in OVERRIDES:
        value = getattr(args, override)

        if value is not None:
            parameters[override] = value

    task(**parameters)


def tasks(args: argparse.Namespace) -> None:
    """`lab tasks`: list every task, naming the extra to install for any that is missing."""
    listed = list_tasks()
    width = max(len(name) for name, _ in listed)

    for name, missing in listed:
        namespace = name.partition(".")[0]
        note = f"  needs: pip install lab[{namespace}]" if missing else ""

        print(f"{name:<{width}}{note}")


def build_parser() -> argparse.ArgumentParser:
    """The `lab` argument parser, with the `run` and `tasks` commands."""
    parser = argparse.ArgumentParser(
        prog="lab",
        description="Run one of the library's tasks from a YAML config.",
    )
    parser.add_argument("--version", action="version", version=f"lab {__version__}")

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

    tasks_parser = subparsers.add_parser(
        "tasks", help="List every task, marking those whose extra is not installed."
    )
    tasks_parser.set_defaults(handler=tasks)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse `argv` and run the command, returning the exit code.

    A `ValueError`, `FileNotFoundError` or `TypeError` from the task exits with
    status 2 and its message rather than a traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    try:
        args.handler(args)
    except (ValueError, FileNotFoundError, TypeError) as error:
        parser.exit(2, f"lab: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
