"""Offline behavioral verification for NER error/generalization analysis."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

from _harness import Checks, run
from lab.core import read_spans, write_corpus
from lab.ner.analysis import analyze_evaluation, inspect_analysis, regenerate_report
from lab.ner.analysis.diagnostics import (
    DiagnosticMatchConfig,
    enrich_error_diagnostics,
    evaluate_to_events,
)
from lab.ner.analysis.reporting import error_taxonomy_table, oracle_error_budget


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lab.cli", *arguments],
        capture_output=True,
        text=True,
        check=True,
    )


def _training_corpus() -> pd.DataFrame:
    rows = []
    examples = [
        ("train_1", "alcohol tabaco", [(0, 7, "Alcohol"), (8, 14, "Tobacco")]),
        ("train_2", "bebedor habitual", [(0, 7, "Alcohol")]),
        ("train_3", "fumador activo", [(0, 7, "Tobacco")]),
    ]
    for doc_id, text, entities in examples:
        payload = [
            {
                "id": f"T{i+1}",
                "start": start,
                "end": end,
                "label": label,
                "text": text[start:end],
            }
            for i, (start, end, label) in enumerate(entities)
        ]
        rows.append(
            {
                "doc_id": doc_id,
                "text": text,
                "entities_json": json.dumps(payload, ensure_ascii=False),
                "n_entities": len(payload),
            }
        )
    return pd.DataFrame(rows).astype(
        {"doc_id": "string", "text": "string", "entities_json": "string", "n_entities": "int32"}
    )


def _gold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # SEEN + correct
            ("doc1", "Alcohol", 0, 7, "alcohol"),
            # LEXICALLY_SIMILAR + boundary error: alcohol -> alcoho
            ("doc2", "Alcohol", 0, 6, "alcoho"),
            # NOVEL + missed
            ("doc3", "Alcohol", 0, 10, "etanolismo"),
            # multiclass exact-boundary label error
            ("doc4", "Tobacco", 0, 8, "tabaquismo"),
            # duplicate is intentionally preserved/audited
            ("doc5", "Alcohol", 0, 7, "bebedor"),
            ("doc5", "Alcohol", 0, 7, "bebedor"),
        ],
        columns=["filename", "label", "start_span", "end_span", "text"],
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("doc1", "Alcohol", 0, 7, "alcohol", 0.95),
            ("doc2", "Alcohol", 0, 5, "alcoh", 0.80),
            ("doc4", "Alcohol", 0, 8, "tabaquismo", 0.90),
            ("doc5", "Alcohol", 0, 7, "bebedor", 0.88),
            ("doc6", "Alcohol", 0, 7, "alcohol", 0.70),
        ],
        columns=["filename", "label", "start_span", "end_span", "text", "score"],
    )


def main() -> int:
    checks = Checks("analysis")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        gold = _gold()
        predicted = _predictions()
        training = _training_corpus()

        gold_tsv = root / "gold.tsv"
        pred_parquet = root / "predictions.parquet"
        training_parquet = root / "documents.parquet"
        gold.to_csv(gold_tsv, sep="\t", index=False)
        predicted.to_parquet(pred_parquet, index=False)
        write_corpus(training, training_parquet)

        reread_gold = read_spans(gold_tsv)
        checks.equal("read_spans preserves duplicate rows", int(reread_gold.duplicated().sum()), 1)
        checks.equal("read_spans preserves row count", len(reread_gold), len(gold))

        result = analyze_evaluation(
            pred_parquet,
            gold_tsv,
            training_parquet,
            root / "analysis",
            run_id="verification",
            levenshtein_threshold=0.80,
            bootstrap_samples=50,
            random_state=13,
            verbose=1,
        )

        checks.check("evaluation parquet exists", result.evaluation_path.exists())
        checks.check("manifest exists", result.manifest_path.exists())
        checks.check("multiclass is detected", result.multiclass)
        checks.check("analysis writes many reusable tables", len(result.table_paths) >= 25)
        checks.check("verbose=1 writes SVG figures", len(result.figure_paths) >= 8)
        checks.check("all generated figures are SVG", all(path.suffix == ".svg" for path in result.figure_paths))
        checks.check(
            "publication plot manifest exists",
            (result.output_dir / "figures" / "plot_manifest.json").exists(),
        )
        checks.check(
            "publication captions exist",
            (result.output_dir / "captions.md").exists(),
        )

        generalization = pd.read_parquet(result.table_paths["metrics_by_generalization"])
        checks.equal(
            "generalization classes are explicit and ordered",
            generalization["value"].tolist(),
            ["SEEN", "LEXICALLY_SIMILAR", "NOVEL"],
        )
        checks.check(
            "generalization table marks low-support subgroups",
            "low_support" in generalization.columns,
        )
        profile = pd.read_parquet(result.table_paths["generalization_error_profile"])
        checks.check(
            "generalization error profile is mutually exclusive",
            set(profile["error_type"]).issuperset({"CORRECT", "MISSED", "BOUNDARY_ERROR", "LABEL_ERROR"}),
        )
        checks.check(
            "error-type Pareto table exists",
            result.table_paths["error_type_pareto"].exists(),
        )
        checks.check(
            "dataset-shift summary exists",
            result.table_paths["dataset_shift_summary"].exists(),
        )
        checks.check(
            "boundary-threshold sensitivity table exists",
            result.table_paths["boundary_overlap_sensitivity"].exists(),
        )
        checks.check(
            "without-replacement partition table exists",
            result.table_paths["document_partition_metrics"].exists(),
        )

        events = pd.read_parquet(result.evaluation_path)
        checks.check("boundary error is detected", "BOUNDARY_ERROR" in set(events["error_primary"]))
        checks.check("label error is detected", "LABEL_ERROR" in set(events["error_primary"]))
        checks.check("missed error is detected", "MISSED" in set(events["error_primary"]))
        checks.check("spurious error is detected", "SPURIOUS" in set(events["error_primary"]))

        # ------------------------------------------------------------------
        # Adversarial diagnostic span matching
        # ------------------------------------------------------------------

        def diagnose(gold_rows, pred_rows, *, multiclass=False, threshold=0.60):
            g = pd.DataFrame(gold_rows, columns=["filename", "label", "start_span", "end_span", "text"])
            p = pd.DataFrame(pred_rows, columns=["filename", "label", "start_span", "end_span", "text"])
            tags = sorted(set(g["label"].astype(str)) | set(p["label"].astype(str)))
            strict, _ = evaluate_to_events(g, p, run_id="diagnostic-test", tags=tags)
            return enrich_error_diagnostics(
                strict,
                g,
                p,
                match_config=DiagnosticMatchConfig(overlap_threshold=threshold),
                multiclass=multiclass,
            )

        exact = diagnose(
            [("d", "X", 10, 20, "abcdefghij")],
            [("d", "X", 10, 20, "abcdefghij")],
        )
        checks.equal(
            "exact span and label remains correct",
            set(exact["error_primary"]),
            {"CORRECT"},
        )

        distant = diagnose(
            [("d", "X", 10, 20, "abcdefghij")],
            [("d", "X", 100, 110, "klmnopqrst")],
        )
        checks.equal(
            "distant spans remain missed plus spurious",
            sorted(distant["error_primary"].tolist()),
            ["MISSED", "SPURIOUS"],
        )

        trivial = diagnose(
            [("d", "X", 10, 100, "g" * 90)],
            [("d", "X", 99, 120, "p" * 21)],
        )
        checks.equal(
            "one-character overlap below Dice threshold is not a boundary error",
            sorted(trivial["error_primary"].tolist()),
            ["MISSED", "SPURIOUS"],
        )

        boundary = diagnose(
            [("d", "X", 10, 20, "abcdefghij")],
            [("d", "X", 11, 20, "bcdefghij")],
        )
        checks.equal(
            "plausible high-overlap mismatch is a boundary error",
            set(boundary["error_primary"]),
            {"BOUNDARY_ERROR"},
        )
        checks.check(
            "diagnostic pairing stores normalized Dice overlap",
            float(boundary["span_dice"].dropna().iloc[0]) >= 0.60,
        )

        containment = diagnose(
            [("d", "X", 10, 30, "abcdefghijklmnopqrst")],
            [("d", "X", 15, 30, "fghijklmnopqrst")],
        )
        checks.equal(
            "plausible contained span is a boundary error",
            set(containment["error_primary"]),
            {"BOUNDARY_ERROR"},
        )

        label = diagnose(
            [("d", "Disease", 10, 20, "abcdefghij")],
            [("d", "Symptom", 10, 20, "abcdefghij")],
            multiclass=True,
        )
        checks.equal(
            "label-only mismatch is diagnosed only in multiclass mode",
            set(label["error_primary"]),
            {"LABEL_ERROR"},
        )

        label_disabled = diagnose(
            [("d", "Disease", 10, 20, "abcdefghij")],
            [("d", "Symptom", 10, 20, "abcdefghij")],
            multiclass=False,
        )
        checks.equal(
            "label mismatch is not converted to label error when label diagnostics are disabled",
            sorted(label_disabled["error_primary"].tolist()),
            ["MISSED", "SPURIOUS"],
        )

        boundary_and_label = diagnose(
            [("d", "Disease", 10, 20, "abcdefghij")],
            [("d", "Symptom", 11, 20, "bcdefghij")],
            multiclass=True,
        )
        checks.equal(
            "boundary plus label mismatch is diagnosed in multiclass mode",
            set(boundary_and_label["error_primary"]),
            {"BOUNDARY_AND_LABEL_ERROR"},
        )

        fragmentation = diagnose(
            [("d", "X", 0, 20, "abcdefghijklmnopqrst")],
            [
                ("d", "X", 0, 8, "abcdefgh"),
                ("d", "X", 8, 14, "ijklmn"),
            ],
        )
        taxonomy = error_taxonomy_table(fragmentation)
        boundary_count = int(
            taxonomy.loc[taxonomy["error_type"] == "BOUNDARY_ERROR", "count"].sum()
        )
        checks.equal(
            "fragmentation counts as one boundary diagnostic unit",
            boundary_count,
            1,
        )
        checks.equal(
            "fragmentation links multiple predictions to one gold",
            int(fragmentation.loc[fragmentation["error_pair_role"] == "GOLD", "fragmentation_pred_count"].iloc[0]),
            2,
        )
        fragmentation_oracle = oracle_error_budget(fragmentation, {})
        fragmentation_boundary_oracle = fragmentation_oracle.loc[
            fragmentation_oracle["scenario"] == "boundary_correction"
        ].iloc[0]
        checks.equal(
            "fragmentation oracle correction removes all fragment false positives",
            int(fragmentation_boundary_oracle["fp_removed"]),
            2,
        )
        checks.equal(
            "merging is not analysed in the current profile",
            int(fragmentation["error_merging"].fillna(False).astype(bool).sum()),
            0,
        )

        nested = diagnose(
            [
                ("d", "X", 0, 20, "abcdefghijklmnopqrst"),
                ("d", "X", 5, 15, "fghijklmno"),
            ],
            [
                ("d", "X", 0, 19, "abcdefghijklmnopqrs"),
                ("d", "X", 5, 14, "fghijklmn"),
            ],
        )
        nested_gold = nested.loc[nested["error_pair_role"] == "GOLD"].sort_values("gold_id")
        checks.equal(
            "nested spans receive unique one-to-one diagnostic partners",
            nested_gold["diagnostic_pred_id"].astype(int).tolist(),
            [0, 1],
        )

        merging_ignored = diagnose(
            [
                ("d", "X", 0, 8, "abcdefgh"),
                ("d", "X", 8, 16, "ijklmnop"),
            ],
            [("d", "X", 0, 16, "abcdefghijklmnop")],
        )
        checks.equal(
            "many-gold-to-one-prediction merging flag remains disabled",
            int(merging_ignored["error_merging"].fillna(False).astype(bool).sum()),
            0,
        )

        audit = pd.read_parquet(result.table_paths["gold_annotation_audit"])
        duplicate_flags = audit["annotation_issues"].map(lambda issues: "DUPLICATE_ANNOTATION" in list(issues))
        checks.equal("duplicate gold rows are reported rather than dropped", int(duplicate_flags.sum()), 2)

        confusion = pd.read_parquet(result.table_paths["label_confusion_counts"])
        tobacco_to_alcohol = confusion[
            (confusion["gold_label"] == "Tobacco")
            & (confusion["pred_label"] == "Alcohol")
        ]
        checks.equal("multiclass label confusion is counted", int(tobacco_to_alcohol.iloc[0]["count"]), 1)

        summary = inspect_analysis(root / "analysis")
        checks.equal("inspect restores run id", summary["run_id"], "verification")
        plot_manifest = json.loads(
            (root / "analysis" / "figures" / "plot_manifest.json").read_text(encoding="utf-8")
        )
        checks.equal(
            "plot profile keeps category labels unrotated",
            plot_manifest["design"]["rotated_category_tick_labels"],
            False,
        )
        checks.check(
            "plot manifest records generated and omitted decisions",
            bool(plot_manifest.get("decisions")),
        )
        checks.equal("manifest records duplicate gold count", summary["counts"]["gold_duplicates"], 1)
        checks.equal("manifest records threshold", summary["generalization"]["lexical_similarity_threshold"], 0.8)
        checks.equal("manifest records hybrid lexical mode", summary["generalization"]["similarity_mode"], "hybrid")
        sensitivity = pd.read_parquet(result.table_paths["boundary_overlap_sensitivity"])
        checks.check(
            "default diagnostic sensitivity includes the selected 0.60 threshold",
            bool((pd.to_numeric(sensitivity["diagnostic_overlap_threshold"]) == 0.60).any()),
        )
        partition_summary = pd.read_parquet(result.table_paths["document_partition_summary"])
        checks.check(
            "partition summary records full-test and descriptive partition statistics",
            {"full_test_estimate", "partition_mean", "partition_std", "partition_min", "partition_max"}.issubset(partition_summary.columns),
        )

        regenerated = regenerate_report(root / "analysis", verbose=1)
        checks.check("report regeneration works without rescoring", len(regenerated) >= 6)

        cli_help = _run_cli("ner", "analysis", "--help")
        for command in ("run", "inspect", "report"):
            checks.check(f"analysis CLI exposes {command}", command in cli_help.stdout)

        run_help = _run_cli("ner", "analysis", "run", "--help")
        for option in (
            "--predictions",
            "--gold",
            "--training",
            "--levenshtein-threshold",
            "--lexical-similarity-mode",
            "--diagnostic-overlap-threshold",
            "--partition-size",
            "--bootstrap-samples",
            "--verbose",
        ):
            checks.check(f"analysis run help exposes {option}", option in run_help.stdout)

        tasks = _run_cli("tasks")
        checks.check(
            "task registry exposes ner.analyze_evaluation",
            "ner.analyze_evaluation" in tasks.stdout,
        )

    return checks.report()


if __name__ == "__main__":
    run(main)
