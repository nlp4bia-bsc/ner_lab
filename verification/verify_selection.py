"""Behavioural checks for task-independent document selection.

All checks are deliberately offline.  Selection algorithms are exercised with
synthetic corpora, representations, scores and probabilities so verification
never needs to download a Transformer checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from _harness import Checks, run
from lab.core import read_corpus, write_corpus
from lab.selection.api import (
    build_representations,
    compare_methods,
    consensus_select_documents,
    select_documents,
    selection_history,
    validate_selection_input,
)
from lab.selection.methods import get_method, list_methods
from lab.selection.representations import list_representations


EXPECTED_METHODS = (
    "random",
    "kmedoids",
    "facility-location",
    "kcenter",
    "typiclust",
    "alps",
    "patron",
    "deuce",
    "uncertainty",
    "uncertainty-facility",
)

EXPECTED_COMMANDS = (
    "methods",
    "method",
    "representations",
    "validate",
    "representation",
    "represent",
    "select",
    "history",
    "compare",
    "consensus",
)


def _corpus() -> pd.DataFrame:
    """Synthetic canonical corpus with annotated and unannotated documents."""
    rows = []

    for index in range(12):
        text = (
            f"cardiology heart rhythm patient {index}"
            if index < 4
            else f"neurology brain seizure patient {index}"
            if index < 8
            else f"oncology tumour treatment patient {index}"
        )

        entities = (
            []
            if index % 2
            else [
                {
                    "id": "T1",
                    "start": 0,
                    "end": text.index(" "),
                    "label": "TEST",
                    "text": text[: text.index(" ")],
                }
            ]
        )

        rows.append(
            {
                "doc_id": f"doc_{index:02d}",
                "text": text,
                "entities_json": json.dumps(entities),
                "n_entities": len(entities),
            }
        )

    return pd.DataFrame(rows).astype(
        {
            "doc_id": "string",
            "text": "string",
            "entities_json": "string",
            "n_entities": "int32",
        }
    )


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installed source CLI through Python without a shell."""
    return subprocess.run(
        [sys.executable, "-m", "lab.cli", *arguments],
        capture_output=True,
        text=True,
        check=True,
    )


def _write_scores(corpus: pd.DataFrame, path: Path) -> Path:
    """Write deterministic external uncertainty scores."""
    frame = pd.DataFrame(
        {
            "doc_id": corpus["doc_id"],
            "score": np.linspace(0.0, 1.0, len(corpus)),
        }
    )
    frame.to_parquet(path, index=False)
    return path


def _write_probabilities(corpus: pd.DataFrame, path: Path) -> Path:
    """Write deterministic two-class distributions for PATRON/DEUCE tests."""
    frame = pd.DataFrame(
        {
            "doc_id": corpus["doc_id"],
            "probabilities": [
                [0.5 + index / 100, 0.5 - index / 100]
                for index in range(len(corpus))
            ],
        }
    )
    frame.to_parquet(path, index=False)
    return path


def main() -> int:
    checks = Checks("selection")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        corpus = _corpus()
        corpus_path = write_corpus(
            corpus,
            root / "documents.parquet",
        )

        # ------------------------------------------------------------------
        # Canonical input and validation
        # ------------------------------------------------------------------

        summary = validate_selection_input(corpus_path)

        checks.equal(
            "validation counts documents",
            summary["n_documents"],
            12,
        )
        checks.equal(
            "validation counts annotated documents",
            summary["n_annotated_documents"],
            6,
        )
        checks.equal(
            "validation counts unannotated documents",
            summary["n_unannotated_documents"],
            6,
        )
        checks.equal(
            "validation counts entities",
            summary["n_entities"],
            6,
        )
        checks.equal(
            "validation exposes canonical columns",
            summary["columns"],
            ["doc_id", "text", "entities_json", "n_entities"],
        )
        checks.check(
            "validation records SHA-256",
            len(summary["sha256"]) == 64,
        )

        # ------------------------------------------------------------------
        # Registries
        # ------------------------------------------------------------------

        method_names = tuple(item.name for item in list_methods())

        checks.equal(
            "method registry contains the expected methods",
            method_names,
            EXPECTED_METHODS,
        )

        representation_names = {
            item.name
            for item in list_representations()
        }

        checks.check(
            "representation registry includes tfidf",
            "tfidf" in representation_names,
        )
        checks.check(
            "representation registry includes alps",
            "alps" in representation_names,
        )

        # ------------------------------------------------------------------
        # Random baseline and reproducibility
        # ------------------------------------------------------------------

        first = select_documents(
            corpus_path,
            root / "random1",
            "random",
            3,
            random_state=13,
        )

        second_same = select_documents(
            corpus_path,
            root / "random1_repeat",
            "random",
            3,
            random_state=13,
        )

        checks.equal(
            "random is deterministic for a fixed seed",
            first.selected_doc_ids,
            second_same.selected_doc_ids,
        )

        checks.equal(
            "random returns the exact requested budget",
            len(first.selected_doc_ids),
            3,
        )

        checks.equal(
            "random selection contains unique documents",
            len(set(first.selected_doc_ids)),
            3,
        )

        # ------------------------------------------------------------------
        # Iterative selection / previous rounds
        # ------------------------------------------------------------------

        second = select_documents(
            corpus_path,
            root / "random2",
            "random",
            3,
            selected=[first.history_path],
            random_state=13,
        )

        checks.equal(
            "rounds are disjoint",
            set(first.selected_doc_ids) & set(second.selected_doc_ids),
            set(),
        )

        history = selection_history(second.history_path)

        checks.equal(
            "history accumulates six documents",
            len(history),
            6,
        )

        checks.equal(
            "history contains unique documents",
            history["doc_id"].nunique(),
            6,
        )

        checks.equal(
            "second run is inferred as round two",
            second.round_number,
            2,
        )

        # ------------------------------------------------------------------
        # Canonical corpus preservation
        # ------------------------------------------------------------------

        original = read_corpus(corpus_path).set_index("doc_id")
        selected = read_corpus(first.selected_path).set_index("doc_id")

        preserved = all(
            selected.loc[doc_id, "text"]
            == original.loc[doc_id, "text"]
            and selected.loc[doc_id, "entities_json"]
            == original.loc[doc_id, "entities_json"]
            and int(selected.loc[doc_id, "n_entities"])
            == int(original.loc[doc_id, "n_entities"])
            for doc_id in selected.index
        )

        checks.check(
            "canonical rows and annotations are preserved exactly",
            preserved,
        )

        checks.equal(
            "selected parquet keeps canonical columns only",
            list(read_corpus(first.selected_path).columns),
            ["doc_id", "text", "entities_json", "n_entities"],
        )

        # ------------------------------------------------------------------
        # Representations
        # ------------------------------------------------------------------

        representation_path = build_representations(
            corpus_path,
            root / "tfidf.parquet",
            "tfidf",
            representation_params={
                "max_features": 64,
            },
        )

        representation_frame = pd.read_parquet(
            representation_path
        )

        checks.equal(
            "representation has one row per document",
            len(representation_frame),
            12,
        )

        checks.check(
            "representation artifact contains doc_id",
            "doc_id" in representation_frame.columns,
        )

        checks.check(
            "representation artifact contains embeddings",
            "embedding" in representation_frame.columns,
        )

        checks.equal(
            "representation doc_ids are unique",
            representation_frame["doc_id"].nunique(),
            12,
        )

        # ------------------------------------------------------------------
        # Representation-based selectors
        # ------------------------------------------------------------------

        for method in (
            "kmedoids",
            "facility-location",
            "kcenter",
            "typiclust",
            "alps",
        ):
            result = select_documents(
                corpus_path,
                root / method,
                method,
                3,
                representations_path=representation_path,
                random_state=13,
            )

            checks.equal(
                f"{method} returns exact budget",
                len(result.selected_doc_ids),
                3,
            )

            checks.equal(
                f"{method} returns unique ids",
                len(set(result.selected_doc_ids)),
                3,
            )

        # ALPS above deliberately uses a precomputed synthetic/TF-IDF
        # representation.  This verifies the ALPS selection kernel without
        # downloading an MLM checkpoint.  The actual ALPS representation
        # implementation remains an integration/model-dependent operation.

        # ------------------------------------------------------------------
        # Model-produced scores
        # ------------------------------------------------------------------

        scores_path = _write_scores(
            corpus,
            root / "scores.parquet",
        )

        uncertain = select_documents(
            corpus_path,
            root / "uncertainty",
            "uncertainty",
            2,
            scores_path=scores_path,
        )

        checks.equal(
            "uncertainty selects highest scores",
            uncertain.selected_doc_ids,
            ("doc_11", "doc_10"),
        )

        hybrid = select_documents(
            corpus_path,
            root / "uncertainty_facility",
            "uncertainty-facility",
            3,
            representations_path=representation_path,
            scores_path=scores_path,
            random_state=13,
        )

        checks.equal(
            "uncertainty-facility returns exact budget",
            len(hybrid.selected_doc_ids),
            3,
        )

        checks.equal(
            "uncertainty-facility returns unique ids",
            len(set(hybrid.selected_doc_ids)),
            3,
        )

        # ------------------------------------------------------------------
        # Probability-based selectors
        # ------------------------------------------------------------------

        probabilities_path = _write_probabilities(
            corpus,
            root / "probabilities.parquet",
        )

        for method in (
            "patron",
            "deuce",
        ):
            result = select_documents(
                corpus_path,
                root / method,
                method,
                3,
                representations_path=representation_path,
                probabilities_path=probabilities_path,
                random_state=13,
            )

            checks.equal(
                f"{method} returns exact budget",
                len(result.selected_doc_ids),
                3,
            )

            checks.equal(
                f"{method} returns unique ids",
                len(set(result.selected_doc_ids)),
                3,
            )

        # ------------------------------------------------------------------
        # Direct method API
        # ------------------------------------------------------------------

        random_method = get_method("random")

        direct_result = random_method.select(
            list(corpus["doc_id"].astype(str)),
            3,
            random_state=13,
        )

        checks.equal(
            "direct method API returns exact budget",
            len(direct_result.selected_indices),
            3,
        )

        checks.equal(
            "direct method API returns unique indices",
            len(set(direct_result.selected_indices)),
            3,
        )

        # ------------------------------------------------------------------
        # Comparison workflow
        # ------------------------------------------------------------------

        overlap_path = compare_methods(
            corpus_path,
            root / "comparison",
            (
                "kmedoids",
                "kcenter",
                "typiclust",
            ),
            3,
            representations_path=representation_path,
            random_state=13,
        )

        overlap = pd.read_parquet(overlap_path)

        checks.equal(
            "comparison contains full pairwise matrix",
            len(overlap),
            9,
        )

        checks.check(
            "comparison contains Jaccard values",
            "jaccard" in overlap.columns,
        )

        checks.check(
            "comparison Jaccard is bounded",
            bool(
                overlap["jaccard"]
                .between(0.0, 1.0)
                .all()
            ),
        )


        # ------------------------------------------------------------------
        # Consensus workflow
        # ------------------------------------------------------------------

        consensus = consensus_select_documents(
            corpus_path,
            root / "consensus",
            (
                "kmedoids",
                "kcenter",
                "typiclust",
            ),
            3,
            representations_path=representation_path,
            random_state=13,
        )

        checks.equal(
            "consensus returns one exact shared budget",
            len(consensus.selected_doc_ids),
            3,
        )

        checks.equal(
            "consensus selected ids are unique",
            len(set(consensus.selected_doc_ids)),
            3,
        )

        consensus_ranking = pd.read_parquet(
            consensus.ranking_path
        )

        checks.equal(
            "consensus ranking covers the full candidate pool",
            len(consensus_ranking),
            12,
        )

        checks.equal(
            "consensus has exactly three final selected rows",
            int(consensus_ranking["selected"].sum()),
            3,
        )

        checks.equal(
            "consensus total binary votes equal methods times budget",
            int(consensus_ranking["vote_count"].sum()),
            9,
        )

        for method in (
            "kmedoids",
            "kcenter",
            "typiclust",
        ):
            prefix = method.replace("-", "_")

            checks.equal(
                f"consensus {method} casts exactly three top-batch votes",
                int(
                    consensus_ranking[
                        f"{prefix}__vote"
                    ].sum()
                ),
                3,
            )

            checks.check(
                f"consensus {method} exposes finite scores for every document",
                bool(
                    np.isfinite(
                        consensus_ranking[
                            f"{prefix}__score"
                        ].to_numpy(dtype=float)
                    ).all()
                ),
            )

            checks.equal(
                f"consensus {method} gives a complete unique ranking",
                consensus_ranking[
                    f"{prefix}__rank"
                ].nunique(),
                12,
            )

        consensus_second = consensus_select_documents(
            corpus_path,
            root / "consensus_round_2",
            (
                "kmedoids",
                "kcenter",
                "typiclust",
            ),
            3,
            selected=[consensus.history_path],
            representations_path=representation_path,
            random_state=13,
        )

        checks.equal(
            "consensus active-learning rounds are disjoint",
            set(consensus.selected_doc_ids)
            & set(consensus_second.selected_doc_ids),
            set(),
        )

        consensus_history = selection_history(
            consensus_second.history_path
        )

        checks.equal(
            "consensus history accumulates both rounds",
            len(consensus_history),
            6,
        )

        checks.equal(
            "consensus second run is inferred as round two",
            consensus_second.round_number,
            2,
        )

        # ------------------------------------------------------------------
        # CLI: global selection command catalogue
        # ------------------------------------------------------------------

        selection_help = _run_cli(
            "selection",
            "--help",
        )

        for command in EXPECTED_COMMANDS:
            checks.check(
                f"selection CLI exposes {command}",
                command in selection_help.stdout,
            )

        # ------------------------------------------------------------------
        # CLI: validate
        # ------------------------------------------------------------------

        validate_result = _run_cli(
            "selection",
            "validate",
            "--input",
            str(corpus_path),
        )

        checks.check(
            "selection validate reports document count",
            "documents: 12" in validate_result.stdout,
        )

        checks.check(
            "selection validate reports annotated documents",
            "annotated documents: 6"
            in validate_result.stdout,
        )

        checks.check(
            "selection validate reports unannotated documents",
            "unannotated documents: 6"
            in validate_result.stdout,
        )

        checks.check(
            "selection validate reports entity count",
            "entities: 6"
            in validate_result.stdout,
        )

        checks.check(
            "selection validate reports canonical columns",
            "doc_id" in validate_result.stdout
            and "text" in validate_result.stdout
            and "entities_json" in validate_result.stdout
            and "n_entities" in validate_result.stdout,
        )

        # ------------------------------------------------------------------
        # CLI: select help
        # ------------------------------------------------------------------

        select_help = _run_cli(
            "selection",
            "select",
            "--help",
        )

        checks.check(
            "selection help documents previous selections",
            "--selected" in select_help.stdout,
        )

        checks.check(
            "selection help documents representations",
            "--representations" in select_help.stdout,
        )

        checks.check(
            "selection help documents external scores",
            "--scores" in select_help.stdout,
        )

        checks.check(
            "selection help documents probabilities",
            "--probabilities" in select_help.stdout,
        )

        checks.check(
            "selection help documents method parameters",
            "--method-param" in select_help.stdout,
        )

        # ------------------------------------------------------------------
        # CLI: method/representation discovery
        # ------------------------------------------------------------------

        methods_result = _run_cli(
            "selection",
            "methods",
        )

        for method in EXPECTED_METHODS:
            checks.check(
                f"methods command lists {method}",
                method in methods_result.stdout,
            )

        typiclust_help = _run_cli(
            "selection",
            "method",
            "typiclust",
        )

        checks.check(
            "method command exposes scientific reference",
            "Hacohen" in typiclust_help.stdout,
        )

        representations_result = _run_cli(
            "selection",
            "representations",
        )

        checks.check(
            "representations command lists tfidf",
            "tfidf" in representations_result.stdout,
        )

        # ------------------------------------------------------------------
        # YAML task integration
        # ------------------------------------------------------------------

        tasks_result = _run_cli("tasks")

        checks.check(
            "task registry exposes selection.select_documents",
            "selection.select_documents"
            in tasks_result.stdout,
        )

        checks.check(
            "task registry exposes selection.build_representations",
            "selection.build_representations"
            in tasks_result.stdout,
        )

        checks.check(
            "task registry exposes selection.compare_methods",
            "selection.compare_methods"
            in tasks_result.stdout,
        )

        checks.check(
            "task registry exposes selection.consensus_select_documents",
            "selection.consensus_select_documents"
            in tasks_result.stdout,
        )

    return checks.report()


if __name__ == "__main__":
    run(main)