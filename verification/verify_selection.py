"""Behavioural checks for task-independent document selection.

All checks are deliberately offline. Selection algorithms are exercised with
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
    _partial_borda_ranking,
    build_representations,
    compare_methods,
    consensus_select_documents,
    select_documents,
    selection_history,
    validate_selection_input,
)
from lab.selection.methods import (
    MethodResult,
    get_method,
    group_methods_by_methodology,
    list_methods,
    methods_by_methodology,
)
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


def _representation_matrix(path: Path) -> np.ndarray:
    """Read the synthetic precomputed representation matrix in file order."""
    frame = pd.read_parquet(path)
    return np.vstack(
        [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in frame["embedding"]
        ]
    )


def _probability_matrix(path: Path) -> np.ndarray:
    """Read the synthetic probability matrix in file order."""
    frame = pd.read_parquet(path)
    return np.vstack(
        [
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in frame["probabilities"]
        ]
    )


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
        # Registries and methodology metadata
        # ------------------------------------------------------------------

        method_names = tuple(item.name for item in list_methods())

        checks.equal(
            "method registry contains the expected methods",
            method_names,
            EXPECTED_METHODS,
        )

        checks.equal(
            "geometry methodology group is registered",
            methods_by_methodology("geometry"),
            (
                "kmedoids",
                "facility-location",
                "kcenter",
                "typiclust",
                "alps",
                "patron",
                "deuce",
                "uncertainty-facility",
            ),
        )
        checks.equal(
            "uncertainty methodology group is registered",
            methods_by_methodology("uncertainty"),
            (
                "patron",
                "deuce",
                "uncertainty",
                "uncertainty-facility",
            ),
        )
        checks.equal(
            "surprisal methodology group contains ALPS",
            methods_by_methodology("surprisal"),
            ("alps",),
        )

        grouped = group_methods_by_methodology(
            ("kcenter", "alps", "uncertainty")
        )
        checks.equal(
            "methodology grouping is descriptive and multi-label",
            grouped,
            {
                "geometry": ("kcenter", "alps"),
                "surprisal": ("alps",),
                "uncertainty": ("uncertainty",),
            },
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
        selected_frame = read_corpus(first.selected_path).set_index("doc_id")

        preserved = all(
            selected_frame.loc[doc_id, "text"]
            == original.loc[doc_id, "text"]
            and selected_frame.loc[doc_id, "entities_json"]
            == original.loc[doc_id, "entities_json"]
            and int(selected_frame.loc[doc_id, "n_entities"])
            == int(original.loc[doc_id, "n_entities"])
            for doc_id in selected_frame.index
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

        representation_matrix = _representation_matrix(
            representation_path
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
        # representation. This verifies the ALPS selection kernel without
        # downloading an MLM checkpoint. The actual ALPS representation
        # implementation remains an integration/model-dependent operation.

        # ------------------------------------------------------------------
        # Strict method-parameter contracts
        # ------------------------------------------------------------------

        strict_typiclust = select_documents(
            corpus_path,
            root / "typiclust_params",
            "typiclust",
            3,
            representations_path=representation_path,
            method_params={
                "n_clusters": 4,
                "typicality_knn": 3,
            },
            random_state=13,
        )

        checks.equal(
            "valid method parameters remain supported",
            len(strict_typiclust.selected_doc_ids),
            3,
        )

        typo_rejected = False
        try:
            select_documents(
                corpus_path,
                root / "typiclust_bad_param",
                "typiclust",
                3,
                representations_path=representation_path,
                method_params={
                    "typicality_knnn": 3,
                },
                random_state=13,
            )
        except ValueError as error:
            typo_rejected = (
                "Unknown method parameter" in str(error)
                and "typicality_knnn" in str(error)
            )

        checks.check(
            "single-method parameter typos fail loudly",
            typo_rejected,
        )

        reserved_rejected = False
        try:
            select_documents(
                corpus_path,
                root / "random_bad_param",
                "random",
                2,
                method_params={
                    "random_state": 99,
                },
            )
        except ValueError as error:
            reserved_rejected = (
                "run-level option" in str(error)
            )

        checks.check(
            "run-level seed cannot be hidden in method params",
            reserved_rejected,
        )

        direct_unknown_rejected = False
        try:
            get_method("random").select(
                list(corpus["doc_id"].astype(str)),
                2,
                unknown_parameter=1,
            )
        except ValueError as error:
            direct_unknown_rejected = (
                "Unknown parameter" in str(error)
            )

        checks.check(
            "direct method API rejects unknown kwargs",
            direct_unknown_rejected,
        )

        # ------------------------------------------------------------------
        # Shared SelectionMethod framework kwargs
        # ------------------------------------------------------------------

        # The high-level API intentionally invokes all selectors through the
        # common SelectionMethod contract. A concrete selector may therefore
        # receive generic framework kwargs that it does not use directly.
        #
        # These kwargs are NOT method-specific parameters and must not be
        # rejected by _reject_unknown_params(). In particular, random_state
        # reaches selectors such as kmedoids through **params because it is a
        # run-level option rather than a kmedoids-specific hyperparameter.
        framework_random_allowed = True
        try:
            framework_random = get_method("random").select(
                list(corpus["doc_id"].astype(str)),
                2,
                representations=None,
                selected_representations=None,
                scores=None,
                probabilities=None,
                random_state=13,
            )
        except Exception:
            framework_random_allowed = False
            framework_random = None

        checks.check(
            "random accepts shared framework kwargs",
            framework_random_allowed,
        )
        checks.equal(
            "shared framework kwargs do not change random budget",
            (
                len(framework_random.selected_indices)
                if framework_random is not None
                else -1
            ),
            2,
        )

        framework_kmedoids_allowed = True
        try:
            framework_kmedoids = get_method("kmedoids").select(
                list(corpus["doc_id"].astype(str)),
                2,
                representations=representation_matrix,
                selected_representations=None,
                scores=None,
                probabilities=None,
                random_state=13,
                max_swap_passes=1,
            )
        except Exception:
            framework_kmedoids_allowed = False
            framework_kmedoids = None

        checks.check(
            "kmedoids accepts shared framework kwargs including random_state",
            framework_kmedoids_allowed,
        )
        checks.equal(
            "shared framework kwargs do not change kmedoids budget",
            (
                len(framework_kmedoids.selected_indices)
                if framework_kmedoids is not None
                else -1
            ),
            2,
        )

        # ------------------------------------------------------------------
        # Meaningful within-selected ranking for set/partition selectors
        # ------------------------------------------------------------------

        kmedoids_direct = get_method("kmedoids").select(
            list(corpus["doc_id"].astype(str)),
            3,
            representations=representation_matrix,
            max_swap_passes=2,
        )

        kmedoids_importance = kmedoids_direct.extras[
            "selected_leave_one_out_importance"
        ]
        kmedoids_selected_importance = [
            float(kmedoids_importance[index])
            for index in kmedoids_direct.selected_indices
        ]

        checks.check(
            "kmedoids final selected rank is based on finite leave-one-out importance",
            all(
                np.isfinite(value)
                for value in kmedoids_selected_importance
            ),
        )
        checks.equal(
            "kmedoids final selected order follows leave-one-out importance",
            kmedoids_selected_importance,
            sorted(
                kmedoids_selected_importance,
                reverse=True,
            ),
        )

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
        probability_matrix = _probability_matrix(
            probabilities_path
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

        patron_direct = get_method("patron").select(
            list(corpus["doc_id"].astype(str)),
            3,
            representations=representation_matrix,
            probabilities=probability_matrix,
            random_state=13,
        )

        patron_priority = patron_direct.extras[
            "selected_priority"
        ]
        patron_selected_priority = [
            float(patron_priority[index])
            for index in patron_direct.selected_indices
        ]

        checks.check(
            "PATRON final selected rank has finite method-specific priorities",
            all(
                np.isfinite(value)
                for value in patron_selected_priority
            ),
        )
        checks.equal(
            "PATRON final selected order follows method-specific selected priority",
            patron_selected_priority,
            sorted(
                patron_selected_priority,
                reverse=True,
            ),
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
        # Partial Borda helper invariants
        # ------------------------------------------------------------------

        synthetic_a = MethodResult(
            selected_indices=[2, 0, 1],
            scores=np.asarray(
                [1000.0, -1000.0, 0.0, 99.0],
                dtype=np.float64,
            ),
        )
        synthetic_b = MethodResult(
            selected_indices=[2, 0, 1],
            scores=np.asarray(
                [-5.0, 9999.0, -100.0, -999.0],
                dtype=np.float64,
            ),
        )

        _, ranks_a, points_a, scores_a, votes_a = _partial_borda_ranking(
            synthetic_a,
            4,
            3,
        )
        _, ranks_b, points_b, scores_b, votes_b = _partial_borda_ranking(
            synthetic_b,
            4,
            3,
        )

        checks.check(
            "partial Borda ignores raw method-score magnitudes",
            bool(
                np.array_equal(points_a, points_b)
                and np.array_equal(scores_a, scores_b)
                and np.array_equal(votes_a, votes_b)
                and np.allclose(
                    ranks_a,
                    ranks_b,
                    equal_nan=True,
                )
            ),
        )
        checks.equal(
            "partial Borda assigns N..1 points to actual selected order",
            points_a.tolist(),
            [2.0, 1.0, 3.0, 0.0],
        )
        checks.check(
            "partial Borda leaves unselected candidate rank undefined",
            bool(np.isnan(ranks_a[3])),
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

        scoped_comparison = compare_methods(
            corpus_path,
            root / "comparison_scoped",
            (
                "typiclust",
                "alps",
            ),
            3,
            representations_path=representation_path,
            random_state=13,
            method_params={
                "typiclust.n_clusters": 4,
                "typiclust.typicality_knn": 3,
                "alps.n_clusters": 4,
            },
        )

        checks.check(
            "compare accepts scoped METHOD.PARAM settings",
            scoped_comparison.exists(),
        )

        unscoped_multi_rejected = False
        try:
            compare_methods(
                corpus_path,
                root / "comparison_unscoped",
                (
                    "typiclust",
                    "alps",
                ),
                3,
                representations_path=representation_path,
                method_params={
                    "n_clusters": 4,
                },
            )
        except ValueError as error:
            unscoped_multi_rejected = (
                "METHOD.PARAM" in str(error)
            )

        checks.check(
            "multi-method workflows reject unscoped method params",
            unscoped_multi_rejected,
        )

        # ------------------------------------------------------------------
        # Consensus / equal-weight partial Borda rank aggregation
        # ------------------------------------------------------------------

        consensus_methods = (
            "kmedoids",
            "kcenter",
            "typiclust",
        )

        consensus = consensus_select_documents(
            corpus_path,
            root / "consensus",
            consensus_methods,
            3,
            representations_path=representation_path,
            random_state=13,
            method_params={
                "kmedoids.max_swap_passes": 2,
                "typiclust.n_clusters": 4,
                "typiclust.typicality_knn": 3,
            },
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
            "consensus total Top-N memberships equal methods times budget",
            int(consensus_ranking["vote_count"].sum()),
            9,
        )

        point_columns: list[str] = []

        for method in consensus_methods:
            prefix = method.replace("-", "_")
            point_column = f"{prefix}__rank_points"
            point_columns.append(point_column)

            checks.equal(
                f"consensus {method} contributes exactly three ranked documents",
                int(
                    consensus_ranking[
                        f"{prefix}__rank"
                    ].notna().sum()
                ),
                3,
            )
            checks.equal(
                f"consensus {method} casts exactly three Top-N memberships",
                int(
                    consensus_ranking[
                        f"{prefix}__vote"
                    ].sum()
                ),
                3,
            )

            positive_points = sorted(
                consensus_ranking.loc[
                    consensus_ranking[
                        point_column
                    ] > 0,
                    point_column,
                ].astype(float).tolist()
            )
            checks.equal(
                f"consensus {method} uses truncated Borda points 1..N",
                positive_points,
                [1.0, 2.0, 3.0],
            )
            checks.equal(
                f"consensus {method} leaves unselected documents unranked",
                int(
                    consensus_ranking.loc[
                        ~consensus_ranking[
                            f"{prefix}__vote"
                        ],
                        f"{prefix}__rank",
                    ].notna().sum()
                ),
                0,
            )

        expected_points = (
            consensus_ranking[
                point_columns
            ]
            .sum(axis=1)
            .to_numpy(dtype=float)
        )

        checks.check(
            "aggregate rank points equal the sum of per-method Borda points",
            bool(
                np.allclose(
                    expected_points,
                    consensus_ranking[
                        "rank_aggregation_points"
                    ].to_numpy(dtype=float),
                )
            ),
        )

        expected_score = (
            consensus_ranking[
                "rank_aggregation_points"
            ].to_numpy(dtype=float)
            / float(len(consensus_methods) * 3)
        )

        checks.check(
            "consensus score is normalized partial-Borda score",
            bool(
                np.allclose(
                    expected_score,
                    consensus_ranking[
                        "consensus_score"
                    ].to_numpy(dtype=float),
                )
            ),
        )

        manifest = json.loads(
            consensus.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        checks.check(
            "consensus manifest records partial Borda aggregation",
            "truncated Borda" in manifest["consensus_rule"],
        )
        checks.equal(
            "consensus manifest confirms raw method scores are ignored",
            manifest[
                "raw_method_scores_used_for_consensus"
            ],
            False,
        )
        checks.equal(
            "consensus manifest records equal method weights",
            manifest["method_weights"],
            {
                name: 1.0
                for name in consensus_methods
            },
        )
        checks.check(
            "consensus manifest records methodology groups",
            "geometry" in manifest["methodology_groups"],
        )

        weighted_rejected = False
        try:
            consensus_select_documents(
                corpus_path,
                root / "consensus_weighted_rejected",
                consensus_methods,
                3,
                representations_path=representation_path,
                method_weights={
                    "kmedoids": 2.0,
                },
            )
        except ValueError as error:
            weighted_rejected = (
                "Weighted consensus is no longer supported"
                in str(error)
            )

        checks.check(
            "non-unit method weights are rejected",
            weighted_rejected,
        )

        consensus_second = consensus_select_documents(
            corpus_path,
            root / "consensus_round_2",
            consensus_methods,
            3,
            selected=[consensus.history_path],
            representations_path=representation_path,
            random_state=13,
            method_params={
                "kmedoids.max_swap_passes": 2,
                "typiclust.n_clusters": 4,
                "typiclust.typicality_knn": 3,
            },
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
        # CLI: selection command help
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

        compare_help = _run_cli(
            "selection",
            "compare",
            "--help",
        )
        checks.check(
            "compare CLI documents scoped METHOD.PARAM syntax",
            "METHOD.PARAM" in compare_help.stdout,
        )

        consensus_help = _run_cli(
            "selection",
            "consensus",
            "--help",
        )
        checks.check(
            "consensus CLI describes rank aggregation",
            "rank aggregation"
            in consensus_help.stdout.lower(),
        )
        checks.check(
            "consensus CLI no longer exposes method weighting",
            "--method-weight"
            not in consensus_help.stdout,
        )
        checks.check(
            "consensus CLI documents scoped METHOD.PARAM syntax",
            "METHOD.PARAM" in consensus_help.stdout,
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

        checks.check(
            "methods command exposes methodology metadata",
            "Methodologies"
            in methods_result.stdout,
        )

        typiclust_help = _run_cli(
            "selection",
            "method",
            "typiclust",
        )

        checks.check(
            "method command exposes scientific reference",
            "Hacohen"
            in typiclust_help.stdout,
        )
        checks.check(
            "method command exposes methodology tags",
            "Methodologies:"
            in typiclust_help.stdout,
        )

        representations_result = _run_cli(
            "selection",
            "representations",
        )

        checks.check(
            "representations command lists tfidf",
            "tfidf"
            in representations_result.stdout,
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
