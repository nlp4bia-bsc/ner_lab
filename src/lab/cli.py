"""Command-line entry point: `lab run <config.yaml>` and `lab tasks`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml

from lab import __version__
from lab.core.tasks import list_tasks, resolve_task

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
        raise ValueError("Config has no 'task' key. `lab tasks` lists the available ones.")

    task = resolve_task(name)
    parameters = {key: value for key, value in config.items() if key != "task"}

    for override in OVERRIDES:
        value = getattr(args, override)

        if value is not None:
            parameters[override] = value

    task(**parameters)


def tasks(args: argparse.Namespace) -> None:
    listed = list_tasks()
    width = max(len(name) for name, _ in listed)

    for name, missing in listed:
        namespace = name.partition(".")[0]
        note = f"  needs: pip install lab[{namespace}]" if missing else ""

        print(f"{name:<{width}}{note}")


def build_parser() -> argparse.ArgumentParser:
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
