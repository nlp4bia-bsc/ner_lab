"""Command-line entry point: `lab run <config.yaml>` and `lab tasks`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import click
import typer
import yaml

from lab import __version__
from lab.core.tasks import list_tasks, resolve_task

OVERRIDES = ("random_state", "output_dir")

app = typer.Typer(
    name="lab",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "NLP research tooling for canonical corpora, model workflows and task-independent "
        "document selection. Use [bold]lab COMMAND --help[/bold] for detailed workflow help."
    ),
)
selection_app = typer.Typer(
    name="selection",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "Select documents from a canonical lab Parquet without coupling the selection "
        "algorithm to NER, NEL, summarisation, or another downstream task.\n\n"
        "The canonical input schema is [bold]doc_id | text | entities_json | n_entities[/bold]. "
        "Existing annotations are preserved in selected.parquet but are not used by the "
        "selection algorithms. Previous rounds can be supplied with repeated --selected "
        "options so each new batch excludes documents already chosen."
    ),
)
app.add_typer(selection_app, name="selection")

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(config).__name__}."
        )

    return config

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"lab {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed lab version and exit.",
    ),
) -> None:
    """Top-level lab command."""

@app.command("run")
def run_command(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="YAML task configuration."),
    random_state: Optional[int] = typer.Option(
        None,
        "--random-state",
        "--seed",
        help="Override config.random_state without modifying the YAML file.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Override config.output_dir without modifying the YAML file.",
    ),
) -> None:
    """Run a namespaced library task from the existing YAML configuration contract."""
    config_values = load_config(config)
    name = config_values.get("task")
    if name is None:
        raise ValueError("Config has no 'task' key. `lab tasks` lists the available ones.")
    task = resolve_task(str(name))
    parameters = {key: value for key, value in config_values.items() if key != "task"}
    overrides = {"random_state": random_state, "output_dir": output_dir}
    for key in OVERRIDES:
        value = overrides[key]
        if value is not None:
            parameters[key] = value
    task(**parameters)


@app.command("tasks")
def tasks_command() -> None:
    """List every YAML-addressable task and any missing namespace requirements."""
    listed = list_tasks()
    width = max(len(name) for name, _ in listed)
    for name, missing in listed:
        namespace = name.partition(".")[0]
        note = f"  needs: pip install lab[{namespace}]" if missing else ""
        typer.echo(f"{name:<{width}}{note}")


def _parse_key_values(values: Sequence[str], option_name: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{option_name} expects KEY=VALUE; received {item!r}.")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{option_name} contains an empty key: {item!r}.")
        if key in parsed:
            raise ValueError(f"{option_name} repeats key {key!r}.")
        parsed[key] = yaml.safe_load(raw)
    return parsed

def _yes_no(value: bool) -> str:
    return "yes" if value else "no"

@selection_app.command("methods")
def selection_methods() -> None:
    """List registered selection algorithms and their required inputs."""
    from lab.selection.methods import list_methods

    rows = list_methods()
    headers = ("Method", "Family", "Representation", "Scores", "Probabilities", "Previous")
    values = [
        (
            row.name,
            row.family,
            _yes_no(row.requires_representation),
            _yes_no(row.requires_scores),
            _yes_no(row.requires_probabilities),
            _yes_no(row.supports_previous_selection),
        )
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in values)) for i in range(len(headers))]
    typer.echo("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    typer.echo("  ".join("-" * width for width in widths))
    for row in values:
        typer.echo("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    typer.echo("\nUse `lab selection method NAME` for parameters, notes and references.")


@selection_app.command("method")
def selection_method(
    name: str = typer.Argument(..., help="Registered method name, e.g. typiclust or kcenter."),
) -> None:
    """Explain one method, including scientific status and configurable parameters."""
    from lab.selection.methods import method_info

    info = method_info(name)
    typer.echo(info.name)
    typer.echo("=" * len(info.name))
    typer.echo(f"Family: {info.family}")
    typer.echo(f"Description: {info.description}")
    typer.echo(f"Requires representation: {_yes_no(info.requires_representation)}")
    typer.echo(f"Requires scores: {_yes_no(info.requires_scores)}")
    typer.echo(f"Requires probabilities: {_yes_no(info.requires_probabilities)}")
    typer.echo(f"Supports previous selections: {_yes_no(info.supports_previous_selection)}")
    if info.default_representation:
        typer.echo(f"Default representation: {info.default_representation}")
    if info.implementation_note:
        typer.echo(f"Implementation note: {info.implementation_note}")
    if info.reference:
        typer.echo(f"Reference: {info.reference}")
    if info.parameters:
        typer.echo("Parameters:")
        for key, description in info.parameters.items():
            typer.echo(f"  {key}: {description}")

@selection_app.command("representations")
def selection_representations() -> None:
    """List document representations available to representation-based selectors."""
    from lab.selection.representations import list_representations

    rows = list_representations()
    headers = ("Representation", "Family", "Model")
    values = [(row.name, row.family, _yes_no(row.requires_model)) for row in rows]
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in values)) for i in range(len(headers))]
    typer.echo("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    typer.echo("  ".join("-" * width for width in widths))
    for row in values:
        typer.echo("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    typer.echo("\nUse `lab selection representation NAME` for configuration details.")

@selection_app.command("validate")
def selection_validate(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=False,
        readable=True,
        help=(
            "Canonical lab Parquet to validate. Expected schema: "
            "doc_id | text | entities_json | n_entities."
        ),
    ),
) -> None:
    """Validate a corpus before using it for document selection.

    Validation delegates to lab.core.read_corpus(), so this command applies
    exactly the same canonical corpus contract used by the rest of the library.

    The command checks the canonical schema and corpus invariants, then reports
    the number of documents, annotated/unannotated documents, entities and the
    SHA-256 fingerprint of the input file.

    Example:

        lab selection validate --input documents.parquet
    """
    from lab.selection.api import validate_selection_input

    summary = validate_selection_input(input_path)

    typer.echo(f"path: {summary['path']}")
    typer.echo(f"documents: {summary['n_documents']}")
    typer.echo(f"annotated documents: {summary['n_annotated_documents']}")
    typer.echo(f"unannotated documents: {summary['n_unannotated_documents']}")
    typer.echo(f"entities: {summary['n_entities']}")
    typer.echo(f"columns: {', '.join(summary['columns'])}")
    typer.echo(f"sha256: {summary['sha256']}")

@selection_app.command("representation")
def selection_representation(
    name: str = typer.Argument(..., help="Representation name, e.g. tfidf, biolord, medcpt or alps."),
) -> None:
    """Explain one document representation and its configurable parameters."""
    from lab.selection.representations import representation_info

    info = representation_info(name)
    typer.echo(info.name)
    typer.echo("=" * len(info.name))
    typer.echo(f"Family: {info.family}")
    typer.echo(f"Description: {info.description}")
    typer.echo(f"Requires model: {_yes_no(info.requires_model)}")
    if info.implementation_note:
        typer.echo(f"Implementation note: {info.implementation_note}")
    if info.parameters:
        typer.echo("Parameters:")
        for key, description in info.parameters.items():
            typer.echo(f"  {key}: {description}")


@selection_app.command("represent")
def selection_represent(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True, help="Canonical input Parquet."),
    representation: str = typer.Option(..., "--representation", help="Registered representation name."),
    output_path: Path = typer.Option(..., "--output", help="Output Parquet containing doc_id | embedding."),
    seed: int = typer.Option(13, "--seed", "--random-state", help="Seed for stochastic representation procedures."),
    representation_param: list[str] = typer.Option(
        [],
        "--representation-param",
        metavar="KEY=VALUE",
        help="Representation-specific parameter. Repeat this option for multiple parameters.",
    ),
) -> None:
    """Precompute document representations once so several selection methods can reuse them."""
    from lab.selection.api import build_representations

    path = build_representations(
        input_path,
        output_path,
        representation,
        random_state=seed,
        representation_params=_parse_key_values(representation_param, "--representation-param"),
    )
    typer.echo(path)


@selection_app.command("select")
def selection_select(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True, help="Canonical lab Parquet containing the full document pool."),
    method: str = typer.Option(..., "--method", help="Selection method. See `lab selection methods`."),
    n_select: int = typer.Option(..., "--n", min=1, help="Exact number of NEW documents to select in this round."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for selected.parquet, ranking.parquet, history.parquet and manifest.json."),
    selected: list[Path] = typer.Option(
        [],
        "--selected",
        help=(
            "Previous canonical selected.parquet or selection history/ranking Parquet. "
            "Repeat to union several previous/manual selections."
        ),
    ),
    representation: Optional[str] = typer.Option(
        None,
        "--representation",
        help="Compute this representation from the input corpus for a representation-based method.",
    ),
    representations_path: Optional[Path] = typer.Option(
        None,
        "--representations",
        exists=True,
        dir_okay=False,
        help="Reuse a precomputed doc_id | embedding Parquet instead of recomputing representations.",
    ),
    scores_path: Optional[Path] = typer.Option(
        None,
        "--scores",
        exists=True,
        dir_okay=False,
        help="Task-produced doc_id | score Parquet for uncertainty/model-based methods.",
    ),
    probabilities_path: Optional[Path] = typer.Option(
        None,
        "--probabilities",
        exists=True,
        dir_okay=False,
        help="Task-produced doc_id | probabilities Parquet for methods such as PATRON and DEUCE.",
    ),
    seed: int = typer.Option(13, "--seed", "--random-state", help="Random seed recorded in the run manifest."),
    round_number: Optional[int] = typer.Option(None, "--round", min=1, help="Annotation round. Inferred from previous history when omitted."),
    method_param: list[str] = typer.Option(
        [],
        "--method-param",
        metavar="KEY=VALUE",
        help="Method-specific parameter. Repeat as needed; inspect `lab selection method NAME` first.",
    ),
    representation_param: list[str] = typer.Option(
        [],
        "--representation-param",
        metavar="KEY=VALUE",
        help="Representation-specific parameter. Repeat as needed.",
    ),
) -> None:
    """Select the next N documents while excluding all documents selected previously.

    Examples:

      lab selection select --input documents.parquet --method random --n 25 --output-dir runs/random/round_001

      lab selection select --input documents.parquet --method typiclust --representation tfidf --n 25 --output-dir runs/typiclust/round_001

      lab selection select --input documents.parquet --selected runs/typiclust/round_001/history.parquet --method typiclust --representation tfidf --n 25 --output-dir runs/typiclust/round_002

    For controlled comparisons, precompute embeddings with `lab selection represent`
    and pass the same file through --representations to every compatible method.
    """
    from lab.selection.api import select_documents

    run = select_documents(
        input_path,
        output_dir,
        method,
        n_select,
        selected=selected or None,
        representation=representation,
        representations_path=representations_path,
        scores_path=scores_path,
        probabilities_path=probabilities_path,
        random_state=seed,
        round_number=round_number,
        method_params=_parse_key_values(method_param, "--method-param"),
        representation_params=_parse_key_values(representation_param, "--representation-param"),
    )
    typer.echo(f"round: {run.round_number}")
    typer.echo(f"selected: {len(run.selected_doc_ids)}")
    for rank, doc_id in enumerate(run.selected_doc_ids, start=1):
        typer.echo(f"  {rank:02d}. {doc_id}")
    typer.echo(f"selected parquet: {run.selected_path}")
    typer.echo(f"ranking parquet: {run.ranking_path}")
    typer.echo(f"history parquet: {run.history_path}")
    typer.echo(f"manifest: {run.manifest_path}")


@selection_app.command("history")
def selection_history_command(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="history.parquet produced by a selection run."),
) -> None:
    """Inspect the cumulative set of documents selected across annotation rounds."""
    from lab.selection.api import selection_history

    frame = selection_history(path)
    typer.echo(frame.to_string(index=False))


@selection_app.command("compare")
def selection_compare(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True, help="Canonical input Parquet."),
    method: list[str] = typer.Option(..., "--method", help="Method to run. Repeat at least twice."),
    n_select: int = typer.Option(..., "--n", min=1, help="Documents selected by each method."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Root directory for per-method runs and overlap.parquet."),
    selected: list[Path] = typer.Option([], "--selected", help="Previous selection source; repeat as needed."),
    representation: Optional[str] = typer.Option(None, "--representation", help="Representation computed independently for compatible methods."),
    representations_path: Optional[Path] = typer.Option(None, "--representations", exists=True, dir_okay=False, help="Shared precomputed doc_id | embedding Parquet."),
    scores_path: Optional[Path] = typer.Option(None, "--scores", exists=True, dir_okay=False, help="Shared doc_id | score Parquet."),
    probabilities_path: Optional[Path] = typer.Option(None, "--probabilities", exists=True, dir_okay=False, help="Shared doc_id | probabilities Parquet."),
    seed: int = typer.Option(13, "--seed", "--random-state", help="Shared random seed."),
    method_param: list[str] = typer.Option([], "--method-param", metavar="KEY=VALUE", help="Parameter applied to each compatible method."),
    representation_param: list[str] = typer.Option([], "--representation-param", metavar="KEY=VALUE", help="Shared representation parameter."),
) -> None:
    """Run several selectors on the same pool and write pairwise intersection/Jaccard."""
    from lab.selection.api import compare_methods

    path = compare_methods(
        input_path,
        output_dir,
        method,
        n_select,
        selected=selected or None,
        representation=representation,
        representations_path=representations_path,
        scores_path=scores_path,
        probabilities_path=probabilities_path,
        random_state=seed,
        method_params=_parse_key_values(method_param, "--method-param"),
        representation_params=_parse_key_values(representation_param, "--representation-param"),
    )
    typer.echo(path)



@selection_app.command("consensus")
def selection_consensus(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Canonical lab Parquet containing the full document pool.",
    ),
    method: list[str] = typer.Option(
        ...,
        "--method",
        help="Method contributing to the consensus. Repeat at least twice.",
    ),
    n_select: int = typer.Option(
        ...,
        "--n",
        min=1,
        help="Exact FINAL number of NEW consensus documents to select.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help=(
            "Directory for consensus selected.parquet, ranking.parquet, "
            "history.parquet and manifest.json."
        ),
    ),
    selected: list[Path] = typer.Option(
        [],
        "--selected",
        help=(
            "Previous canonical selected.parquet or selection history/ranking "
            "Parquet. Repeat to union several previous selections."
        ),
    ),
    representation: Optional[str] = typer.Option(
        None,
        "--representation",
        help="Shared representation computed for compatible methods.",
    ),
    representations_path: Optional[Path] = typer.Option(
        None,
        "--representations",
        exists=True,
        dir_okay=False,
        help="Shared precomputed doc_id | embedding Parquet.",
    ),
    scores_path: Optional[Path] = typer.Option(
        None,
        "--scores",
        exists=True,
        dir_okay=False,
        help="Shared doc_id | score Parquet for model-based methods.",
    ),
    probabilities_path: Optional[Path] = typer.Option(
        None,
        "--probabilities",
        exists=True,
        dir_okay=False,
        help=(
            "Shared doc_id | probabilities Parquet for methods such as "
            "PATRON and DEUCE."
        ),
    ),
    seed: int = typer.Option(
        13,
        "--seed",
        "--random-state",
        help="Shared random seed recorded in the run manifest.",
    ),
    round_number: Optional[int] = typer.Option(
        None,
        "--round",
        min=1,
        help="Annotation round. Inferred from previous history when omitted.",
    ),
    method_weight: list[str] = typer.Option(
        [],
        "--method-weight",
        metavar="METHOD=WEIGHT",
        help=(
            "Optional positive vote weight for one participating method. "
            "Repeat as needed; unspecified methods have weight 1."
        ),
    ),
    method_param: list[str] = typer.Option(
        [],
        "--method-param",
        metavar="KEY=VALUE",
        help="Parameter applied to each compatible method.",
    ),
    representation_param: list[str] = typer.Option(
        [],
        "--representation-param",
        metavar="KEY=VALUE",
        help="Shared representation parameter.",
    ),
) -> None:
    """Select one exact batch by majority voting across several methods.

    Every participating selector independently chooses N documents and produces
    one score for every remaining candidate. The final batch is NOT N per method:
    it contains exactly N documents total.

    Primary ordering is weighted majority vote. Ties are resolved with a
    weighted Borda-style fusion of the full candidate rankings, so every method
    contributes information for every document.

    Previous selections supplied through --selected are excluded and are passed
    to conditional methods, enabling iterative active-learning rounds.
    """
    from lab.selection.api import consensus_select_documents

    weights = _parse_key_values(
        method_weight,
        "--method-weight",
    )

    run = consensus_select_documents(
        input_path,
        output_dir,
        method,
        n_select,
        selected=selected or None,
        representation=representation,
        representations_path=representations_path,
        scores_path=scores_path,
        probabilities_path=probabilities_path,
        random_state=seed,
        round_number=round_number,
        method_params=_parse_key_values(
            method_param,
            "--method-param",
        ),
        representation_params=_parse_key_values(
            representation_param,
            "--representation-param",
        ),
        method_weights=weights or None,
    )

    typer.echo(f"round: {run.round_number}")
    typer.echo(f"consensus selected: {len(run.selected_doc_ids)}")
    for rank, doc_id in enumerate(
        run.selected_doc_ids,
        start=1,
    ):
        typer.echo(f"  {rank:02d}. {doc_id}")
    typer.echo(f"selected parquet: {run.selected_path}")
    typer.echo(f"ranking parquet: {run.ranking_path}")
    typer.echo(f"history parquet: {run.history_path}")
    typer.echo(f"manifest: {run.manifest_path}")

def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point preserving the existing integer-return convention."""
    command = typer.main.get_command(app)
    try:
        command.main(
            args=list(argv) if argv is not None else None,
            prog_name="lab",
            standalone_mode=False,
        )
        return 0
    except click.ClickException as error:
        error.show()
        return int(error.exit_code)
    except click.exceptions.Exit as error:
        return int(error.exit_code)
    except (ValueError, FileNotFoundError, TypeError, ImportError) as error:
        typer.echo(f"lab: {error}", err=True)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())