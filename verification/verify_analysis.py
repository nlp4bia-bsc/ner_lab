"""Verify the NER error analysis: strict events, pairing, exposure, subgroups, bootstrap, outputs."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run

LABEL = "DISEASE"
COLUMNS = ["filename", "label", "start_span", "end_span", "text"]


def spans(rows: list[tuple], scored: bool = False) -> pd.DataFrame:
    """A span table from `(filename, start, end, text[, score])` rows, all labelled `LABEL`."""
    columns = ["filename", "start_span", "end_span", "text", *(["score"] if scored else [])]
    frame = pd.DataFrame(rows, columns=columns)
    frame.insert(1, "label", LABEL)

    return frame


def pairing_of(gold: pd.DataFrame, predicted: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
    from lab.ner.analysis.events import strict_events
    from lab.ner.analysis.pairing import pair_errors

    events, _ = strict_events(gold, predicted, LABEL)

    return pair_errors(events, overlap_threshold=threshold)


def types_of(units: pd.DataFrame) -> list[str]:
    return sorted(units["error_type"])


def check_strict_accounting(checks: Checks) -> None:
    from lab.ner.analysis.events import strict_events
    from lab.ner.evaluation.scoring import span_metrics

    gold = spans([("a", 0, 7, "alcohol"), ("a", 10, 16, "tabaco"), ("b", 0, 5, "fiebre"), ("c", 0, 3, "HTA")])
    predicted = spans([("a", 0, 7, "alcohol"), ("a", 10, 14, "taba"), ("b", 20, 25, "tos x"), ("d", 0, 2, "xx")])
    events, metrics = strict_events(gold, predicted, LABEL)
    counts = events["strict_outcome"].value_counts().to_dict()

    checks.equal("metrics equal span_metrics on the same frames", metrics, span_metrics(gold, predicted, tags=[LABEL]))
    checks.equal("TP rows equal strict correct", counts.get("TP", 0), metrics["span_strict_correct"])
    checks.equal("every gold span appears once", sorted(events["gold_id"].dropna().astype(int)), [0, 1, 2, 3])
    checks.equal("every prediction appears once", sorted(events["pred_id"].dropna().astype(int)), [0, 1, 2, 3])
    checks.equal("FN + TP rows equal gold", counts["FN"] + counts["TP"], len(gold))
    checks.equal("FP + TP rows equal predictions", counts["FP"] + counts["TP"], len(predicted))

    duplicated = spans([("a", 0, 7, "alcohol"), ("a", 0, 7, "alcohol")])
    events, _ = strict_events(duplicated, spans([("a", 0, 7, "alcohol")]), LABEL)
    checks.equal("a duplicated gold span is one TP and one FN", sorted(events["strict_outcome"]), ["FN", "TP"])


def check_pairing(checks: Checks) -> None:
    gold = spans([("a", 0, 8, "abcdefgh")])
    uncredited = pairing_of(gold, spans([("a", 0, 4, "abcd"), ("a", 0, 8, "abcdefgh")]))
    checks.check(
        "an exact prediction after an overlapping one is UNCREDITED_EXACT",
        "UNCREDITED_EXACT" in set(uncredited["error_type"]),
    )
    checks.equal(
        "the overlapping one stays spurious",
        types_of(uncredited),
        ["SPURIOUS", "UNCREDITED_EXACT"],
    )

    text = "insuficiencia cardiaca"
    long_pair = pairing_of(spans([("a", 0, 13, text[:13])]), spans([("a", 0, 22, text)]))
    checks.equal("Dice 0.74 pairs as a boundary error", types_of(long_pair), ["BOUNDARY_ERROR"])
    checks.equal("a prediction containing the gold is TOO_LONG", long_pair["boundary_kind"].iloc[0], "TOO_LONG")
    checks.equal("offsets are prediction minus gold", (long_pair["start_offset"].iloc[0], long_pair["end_offset"].iloc[0]), (0, 9))
    checks.equal("running past the end gets the START right", long_pair["matched_edge"].iloc[0], "START")

    early = pairing_of(spans([("a", 3, 12, "cardiaca!")]), spans([("a", 0, 12, "la cardiaca!")]))
    checks.equal("starting early but ending right gets the END right", (early["boundary_kind"].iloc[0], early["matched_edge"].iloc[0]), ("TOO_LONG", "END"))

    shifted = pairing_of(spans([("a", 0, 10, "x" * 10)]), spans([("a", 2, 12, "x" * 10)]))
    checks.equal("a shifted prediction gets NEITHER edge right", (shifted["boundary_kind"].iloc[0], shifted["matched_edge"].iloc[0]), ("SHIFTED_RIGHT", "NEITHER"))

    far = pairing_of(spans([("a", 14, 22, "cardiaca")]), spans([("a", 0, 28, "insuficiencia cardiaca aguda")]))
    checks.equal("Dice 0.44 stays missed and spurious", types_of(far), ["MISSED", "SPURIOUS"])
    checks.equal(
        "a lower threshold pairs it",
        types_of(pairing_of(spans([("a", 14, 22, "cardiaca")]), spans([("a", 0, 28, "insuficiencia cardiaca aguda")]), 0.40)),
        ["BOUNDARY_ERROR"],
    )

    fragmented = pairing_of(spans([("a", 0, 8, "anti-TNF")]), spans([("a", 0, 4, "anti"), ("a", 5, 8, "TNF")]))
    checks.equal("anti + TNF inside anti-TNF is one unit", len(fragmented), 1)
    checks.equal("it is FRAGMENTED", fragmented["boundary_kind"].iloc[0], "FRAGMENTED")
    checks.check("a fragmentation has no matched edge", pd.isna(fragmented["matched_edge"].iloc[0]))
    checks.equal("it holds one gold span and two predictions", (len(fragmented["gold_ids"].iloc[0]), len(fragmented["pred_ids"].iloc[0])), (1, 2))

    merged = pairing_of(
        spans([("a", 0, 14, "dolor toracico"), ("a", 17, 23, "disnea")]),
        spans([("a", 0, 23, "dolor toracico y disnea")]),
    )
    checks.equal("one prediction over two gold spans is one MERGED unit", list(merged["boundary_kind"]), ["MERGED"])

    conflict = pairing_of(
        spans([("a", 0, 10, "x" * 10), ("a", 4, 14, "x" * 10)]),
        spans([("a", 0, 9, "x" * 9), ("a", 5, 14, "x" * 9)]),
    )
    pairs = {(unit.gold_ids[0], unit.pred_ids[0]) for unit in conflict.itertuples() if unit.error_type == "BOUNDARY_ERROR"}
    checks.equal("a 2x2 conflict is assigned to maximise total Dice", pairs, {(0, 0), (1, 1)})


def check_target_label(checks: Checks) -> None:
    from lab.ner.analysis.analyze import read_mentions, read_predicted

    mixed = pd.concat([spans([("a", 0, 3, "HTA")]), spans([("a", 5, 8, "IAM")]).assign(label="OTHER")])
    checks.raises("a prediction of another label raises", ValueError, read_predicted, mixed, LABEL, match="OTHER")
    checks.equal("gold of other labels is left out", len(read_mentions(mixed, LABEL, "reference")), 1)
    checks.raises("no mention of target_label raises", ValueError, read_mentions, mixed, "ABSENT", "training", match="ABSENT")


def check_exposure(checks: Checks) -> None:
    from lab.ner.analysis.exposure import describe_exposure

    training = pd.Series(["HTA", "alcohol", "alcohol", "Alcohol", "tabaco"])
    mentions = pd.Series(["hta", "alcoho", "etanolismo", "HTA", "alcohol"])
    exposure = describe_exposure(mentions, training, similarity_threshold=0.80)

    checks.equal("hta against training HTA is SEEN", exposure["exposure"].iloc[0], "SEEN")
    checks.equal("but not seen verbatim", bool(exposure["seen_verbatim"].iloc[0]), False)
    checks.equal("HTA is seen verbatim", bool(exposure["seen_verbatim"].iloc[3]), True)
    checks.equal("alcoho is LEXICALLY_SIMILAR", exposure["exposure"].iloc[1], "LEXICALLY_SIMILAR")
    checks.check("at Levenshtein 6/7", abs(exposure["similarity"].iloc[1] - 6 / 7) < 1e-9)
    checks.equal("etanolismo is NOVEL", exposure["exposure"].iloc[2], "NOVEL")
    checks.equal("frequency counts normalized forms", int(exposure["training_frequency"].iloc[4]), 3)
    checks.check(
        "SEEN exactly when similarity is 1.0",
        ((exposure["exposure"] == "SEEN") == (exposure["similarity"] == 1.0)).all(),
    )
    checks.equal("SEEN falls in the 1.00 bin", exposure.loc[exposure["exposure"] == "SEEN", "similarity_bin"].unique().tolist(), ["1.00"])


def check_features(checks: Checks) -> None:
    from lab.ner.analysis.features import gold_structure, mention_features

    features = mention_features(pd.Series(["anti-TNF", "EPOC", "β-talasemia", "insuficiencia cardiaca"]))

    checks.equal("anti-TNF counts 3 UAX #29 words", int(features["word_count"].iloc[0]), 3)
    checks.equal("EPOC is acronym-like", bool(features["acronym_like"].iloc[1]), True)
    checks.equal("a hyphen is found", bool(features["has_hyphen"].iloc[0]), True)
    checks.equal("a Greek letter is found", bool(features["has_greek"].iloc[2]), True)
    checks.equal("lowercase casing", features["casing"].iloc[3], "LOWERCASE")

    structure = gold_structure(spans([("a", 0, 10, "x"), ("a", 2, 5, "y"), ("a", 8, 12, "z"), ("a", 20, 25, "w")]))
    checks.equal("nesting is found on both sides", structure["nested"].tolist(), [True, True, False, False])
    checks.equal("overlap is found", structure["overlapping"].tolist(), [True, True, True, False])


def synthetic_run(root: Path, scored: bool = True, **parameters):
    from lab.ner.analysis import analyze_predictions

    generator = np.random.default_rng(3)
    vocabulary = ["fiebre alta", "dolor toracico", "insuficiencia cardiaca", "HTA", "anti-TNF", "asma"]
    novel = ["miocarditis", "β-talasemia", "hta"]
    gold_rows, predicted_rows, training_rows = [], [], []

    for index in range(40):
        training_rows.append((f"t{index}", 0, 6, generator.choice(vocabulary)))

    for index in range(30):
        position = 0

        for _ in range(int(generator.integers(1, 5))):
            mention = str(generator.choice(vocabulary + novel))
            start, end = position, position + len(mention)
            gold_rows.append((f"d{index}", start, end, mention))
            draw = generator.random()
            score = float(np.clip(generator.normal(0.9 if draw < 0.7 else 0.7, 0.08), 0.3, 1.0))

            if draw < 0.65:
                predicted_rows.append((f"d{index}", start, end, mention, score))
            elif draw < 0.8:
                predicted_rows.append((f"d{index}", start, end - 1, mention[:-1], score))
            elif draw < 0.9:
                predicted_rows.append((f"d{index}", end + 1, end + 4, "con", score))

            position = end + 5

    training = spans([(name, 0, len(text), text) for name, _, _, text in training_rows])
    predicted = spans(predicted_rows, scored=True)

    if not scored:
        predicted = predicted.drop(columns="score")

    return analyze_predictions(
        predictions=predicted,
        reference=spans(gold_rows),
        training=training,
        target_label=LABEL,
        output_dir=root,
        bootstrap_samples=300,
        **parameters,
    )


def check_subgroups_and_bootstrap(checks: Checks, root: Path) -> None:
    result = synthetic_run(root / "first")
    again = synthetic_run(root / "again", figures=False)
    gold = result.tables["gold_mentions"]
    subgroups = result.tables["gold_subgroups"]
    exposure = subgroups[subgroups["family"] == "exposure"].set_index("value")

    for value, row in exposure.iterrows():
        members = gold[gold["exposure"] == value]
        checks.check(
            f"{value} recall equals the share of CORRECT gold mentions",
            abs(row["strict_recall"] - (members["outcome"] == "CORRECT").mean()) < 1e-12,
        )

    predicted = result.tables["predicted_mentions"]
    predicted_subgroups = result.tables["predicted_subgroups"]
    seen = predicted_subgroups[(predicted_subgroups["family"] == "exposure") & (predicted_subgroups["value"] == "SEEN")]
    checks.check(
        "SEEN precision equals the share of CORRECT seen predictions",
        abs(seen["strict_precision"].iloc[0] - (predicted.loc[predicted["exposure"] == "SEEN", "outcome"] == "CORRECT").mean()) < 1e-12,
    )
    checks.check("no subgroup carries an F1", not any("f1" in column for column in [*subgroups.columns, *predicted_subgroups.columns]))
    checks.check("gold outcomes sum to support", (subgroups[["correct", "boundary_error", "missed", "uncredited_exact"]].sum(axis=1) == subgroups["support"]).all())

    documents = result.tables["documents"]
    checks.equal("per-document TP sum to the strict total", int(documents["strict_tp"].sum()), result.metrics["span_strict_correct"])
    checks.check(
        "per-document Dice credits sum to the overall Dice recall",
        abs(documents["dice_gold_credit"].sum() / documents["dice_gold"].sum() - result.metrics["dice_recall"]) < 1e-4,
    )
    checks.check(
        "a gold mention's best Dice averages to the overall Dice recall",
        abs(gold["best_dice"].mean() - result.metrics["dice_recall"]) < 1e-4,
    )
    checks.check(
        "SEEN Dice recall is the mean best Dice of its gold mentions",
        abs(exposure.loc["SEEN", "dice_recall"] - gold.loc[gold["exposure"] == "SEEN", "best_dice"].mean()) < 1e-12,
    )

    intervals = result.tables["bootstrap_intervals"]
    checks.frames_equal("the bootstrap is identical at one random_state", intervals, again.tables["bootstrap_intervals"])
    checks.check("every interval holds its estimate", ((intervals["low"] <= intervals["estimate"]) & (intervals["estimate"] <= intervals["high"])).all())
    checks.check(
        "the Dice F1 estimate is the overall Dice F1",
        abs(intervals.set_index("metric").loc["dice_f1", "estimate"] - result.metrics["dice_f1"]) < 1e-4,
    )
    checks.check(
        "the estimate is the overall strict F1",
        abs(intervals.set_index("metric").loc["span_strict_f1", "estimate"] - result.metrics["span_strict_f1"]) < 1e-4,
    )
    checks.check(
        "every subgroup interval holds its recall",
        ((subgroups["strict_recall_low"] <= subgroups["strict_recall"] + 1e-12) & (subgroups["strict_recall"] <= subgroups["strict_recall_high"] + 1e-12)).all(),
    )

    sensitivity = result.tables["overlap_sensitivity"]
    checks.check("boundary units never rise with the threshold", (np.diff(sensitivity["boundary_error"]) <= 0).all())
    checks.equal("strict matches are the same at every threshold", sensitivity["correct"].nunique(), 1)

    budget = result.tables["oracle_error_budget"].set_index("scenario")
    checks.equal("fixing every error gives F1 1.0", budget.loc["fix_all", "oracle_f1"], 1.0)
    checks.check("no scenario lowers F1", (budget["gain"] >= 0).all())


def check_outputs(checks: Checks, root: Path) -> None:
    from lab.cli import main
    from lab.ner.analysis import ANALYSIS_MANIFEST_FILENAME, write_figures
    from lab.ner.inference.io import read_predictions, write_predictions

    result = synthetic_run(root / "outputs")
    manifest = json.loads((root / "outputs" / ANALYSIS_MANIFEST_FILENAME).read_text())

    checks.equal("the manifest's metrics are the result's", manifest["metrics"], result.metrics)
    checks.check("every table is written", all(Path(path).is_file() for path in manifest["outputs"]["tables"].values()))
    checks.frames_equal(
        "a written table equals the returned one",
        pd.read_parquet(manifest["outputs"]["tables"]["gold_subgroups"]),
        result.tables["gold_subgroups"],
    )
    checks.equal("all 14 figures are drawn with a varying score", len(manifest["outputs"]["figures"]), 14)
    summary = result.tables["boundary_summary"]
    checks.equal(
        "boundary_summary counts every boundary error once across kind and edge",
        int(summary["count"].sum()),
        len(result.tables["boundary_errors"]),
    )
    checks.check(
        "and splits one-to-one kinds by their matched edge",
        summary.loc[summary["boundary_kind"] == "TOO_SHORT", "matched_edge"].notna().all(),
    )
    checks.check("the summary names the strict F1", "strict F1" in result.summary())

    unscored = synthetic_run(root / "unscored", scored=False)
    names = {Path(path).stem for path in unscored.paths["figures"]}
    checks.check("without score the confidence figures are skipped", not names & {"confidence_by_outcome", "calibration", "confidence_sweep"})
    checks.check("and so are the confidence tables", "calibration" not in unscored.tables)

    redrawn = write_figures(root / "outputs", figure_format="pdf")
    checks.check("write_figures redraws from the tables, as pdf", len(redrawn) == 14 and all(path.suffix == ".pdf" for path in redrawn))

    surfaces = spans([("a", 0, 2, "NA", 0.9), ("a", 5, 9, "null", 0.8)], scored=True)

    for suffix in (".tsv", ".parquet"):
        path = write_predictions(surfaces, root / f"predictions{suffix}")
        checks.equal(f"read_predictions keeps NA as text ({suffix})", read_predictions(path)["text"].tolist(), ["NA", "null"])

    predictions_path = root / "predictions.parquet"
    reference_path = root / "reference.tsv"
    training_path = root / "training.tsv"
    spans([("a", 0, 2, "NA"), ("a", 5, 9, "nulo")]).to_csv(reference_path, sep="\t", index=False)
    spans([("t", 0, 2, "NA")]).to_csv(training_path, sep="\t", index=False)
    config = root / "analysis.yaml"
    config.write_text(
        f"task: ner.analyze_predictions\npredictions: {predictions_path}\nreference: {reference_path}\n"
        f"training: {training_path}\ntarget_label: {LABEL}\noutput_dir: {root / 'unused'}\n"
        "bootstrap_samples: 50\nfigures: false\n"
    )
    exit_code = main(["run", str(config), "--seed", "5", "--output-dir", str(root / "from_cli")])
    cli_manifest = json.loads((root / "from_cli" / ANALYSIS_MANIFEST_FILENAME).read_text())

    checks.equal("ner.analyze_predictions runs through lab run", exit_code, 0)
    checks.equal("--seed reaches random_state", cli_manifest["parameters"]["random_state"], 5)


def main() -> int:
    checks = Checks("NER error analysis")

    check_strict_accounting(checks)
    check_pairing(checks)
    check_target_label(checks)
    check_exposure(checks)
    check_features(checks)

    with tempfile.TemporaryDirectory() as directory:
        check_subgroups_and_bootstrap(checks, Path(directory))
        check_outputs(checks, Path(directory))

    return checks.report()


if __name__ == "__main__":
    run(main)
