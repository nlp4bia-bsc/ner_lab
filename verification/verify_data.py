"""Verify the data layer: labels, BRAT reading, the canonical schema, splitting, orchestration."""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run
from fixtures import entities_of, samples_root, synthetic_documents, write_brat

from lab.core import (
    LABEL_ALIASES,
    assign_partitions,
    build_corpus,
    count_labels,
    create_split,
    document_fingerprints,
    normalize_annotation_labels,
    normalize_label,
    prepare_dataset,
    read_ann,
    read_annotations,
    read_corpus,
    read_split,
    resolve_conflicts,
    resolve_mismatch,
    split_documents,
    validate_assignments,
    validate_corpus,
    write_corpus,
)
from lab.core.corpus import DOCUMENT_COLUMNS, DOCUMENT_DTYPES
from lab.core.stratification import normalize_ratios, parse_document_label_counts


def verify_labels(checks: Checks) -> None:
    checks.equal("alias reassigns FARMACO", normalize_label("FARMACO"), "MEDICATION")
    checks.equal("alias is accent insensitive", normalize_label("farmacos"), "MEDICATION")
    checks.equal("alias is case insensitive", normalize_label("Enfermedad"), "DISEASE")
    checks.equal("unknown label passes through", normalize_label("GENE"), "GENE")
    checks.equal("unknown label is canonicalized", normalize_label("cell-line"), "CELL_LINE")
    checks.equal("spaces become underscores", normalize_label("cell line"), "CELL_LINE")
    checks.equal("empty label stays empty", normalize_label("   "), "")

    checks.equal(
        "custom aliases replace the table",
        normalize_label("GEN", aliases={"GEN": "GENE"}),
        "GENE",
    )
    checks.equal(
        "custom aliases do not inherit the defaults",
        normalize_label("FARMACO", aliases={"GEN": "GENE"}),
        "FARMACO",
    )

    annotations = pd.DataFrame({"label": ["Enfermedad", "GENE"], "other": [1, 2]})
    normalized = normalize_annotation_labels(annotations)

    checks.equal("frame normalization maps known", list(normalized["label"]), ["DISEASE", "GENE"])
    checks.check("frame normalization copies", list(annotations["label"]) == ["Enfermedad", "GENE"])
    checks.raises(
        "missing label column raises",
        ValueError,
        normalize_annotation_labels,
        pd.DataFrame({"nope": [1]}),
    )

    checks.check("alias table is exported", "FARMACO" in LABEL_ALIASES)


def verify_brat(checks: Checks, workspace: Path) -> None:
    documents, annotations = synthetic_documents(6)
    txt_dir, ann_dir = write_brat(workspace / "brat", documents, annotations)

    from_dir = read_ann(ann_dir)

    checks.equal("ann directory row count", len(from_dir), len(annotations))
    checks.equal(
        "ann offsets round-trip",
        [
            documents[row.filename][row.start_span:row.end_span]
            for row in from_dir.itertuples(index=False)
        ],
        list(from_dir["text"]),
    )

    checks.equal("labels are verbatim by default", set(read_ann(ann_dir)["label"]), {"DISEASE"})

    spanish_annotations = annotations.assign(label="ENFERMEDAD")
    _, spanish_ann_dir = write_brat(workspace / "brat_es", documents, spanish_annotations)

    checks.equal(
        "read_ann keeps source labels",
        set(read_ann(spanish_ann_dir)["label"]),
        {"ENFERMEDAD"},
    )
    checks.equal(
        "read_ann normalizes on request",
        set(read_ann(spanish_ann_dir, normalize_labels=True)["label"]),
        {"DISEASE"},
    )

    tsv_path = workspace / "annotations.tsv"
    annotations[["filename", "label", "start_span", "end_span", "text"]].to_csv(
        tsv_path, sep="\t", index=False
    )

    checks.frames_equal("tsv route matches ann route", read_annotations(tsv_path), from_dir)
    checks.frames_equal("dataframe route matches ann route", read_annotations(annotations), from_dir)

    checks.raises("missing path raises", FileNotFoundError, read_annotations, workspace / "nope")

    unsupported = workspace / "annotations.json"
    unsupported.write_text("[]", encoding="utf-8")

    checks.raises("unsupported suffix raises", ValueError, read_annotations, unsupported)

    discontinuous = ann_dir / "disc.ann"
    (workspace / "brat" / "txt" / "disc.txt").write_text("Dolor agudo pecho", encoding="utf-8")
    discontinuous.write_text("T1\tDISEASE 0 11;12 17\tDolor agudo pecho\n", encoding="utf-8")

    fragments = read_ann(discontinuous)

    checks.equal("discontinuous span becomes two rows", len(fragments), 2)
    checks.equal("discontinuous fragments split", list(fragments["text"]), ["Dolor agudo", "pecho"])


def verify_corpus(checks: Checks, workspace: Path) -> None:
    documents, annotations = synthetic_documents(12)
    corpus = build_corpus(documents, annotations)

    checks.equal("corpus columns", list(corpus.columns), DOCUMENT_COLUMNS)
    checks.equal("one row per document", len(corpus), len(documents))
    checks.equal(
        "dtypes are pinned",
        {column: str(corpus[column].dtype) for column in DOCUMENT_DTYPES},
        {column: str(dtype) for column, dtype in DOCUMENT_DTYPES.items()},
    )
    checks.equal(
        "n_entities matches entities_json",
        list(corpus["n_entities"]),
        [len(json.loads(value)) for value in corpus["entities_json"]],
    )

    entities = entities_of(corpus, "doc_002")
    text = documents["doc_002"]

    checks.check(
        "entity offsets round-trip against text",
        all(text[entity["start"]:entity["end"]] == entity["text"] for entity in entities),
    )
    checks.equal(
        "entity keys are stable",
        sorted(entities[0]),
        ["end", "id", "label", "start", "text"],
    )

    spanish = build_corpus(documents, annotations.assign(label="ENFERMEDAD"))
    checks.equal(
        "build_corpus keeps source labels",
        entities_of(spanish, "doc_000")[0]["label"],
        "ENFERMEDAD",
    )

    normalized = build_corpus(
        documents, annotations.assign(label="ENFERMEDAD"), normalize_labels=True
    )
    checks.equal(
        "build_corpus normalizes on request",
        entities_of(normalized, "doc_000")[0]["label"],
        "DISEASE",
    )

    shifted = annotations.copy()
    shifted.loc[0, "start_span"] = int(shifted.loc[0, "start_span"]) + 1

    checks.raises("shifted offset is caught", ValueError, build_corpus, documents, shifted)

    orphaned = annotations.assign(filename="ghost")
    checks.raises("orphan annotation is caught", ValueError, build_corpus, documents, orphaned)

    verify_mismatch_policies(checks, documents, annotations)
    verify_conflict_policies(checks, documents, annotations)

    corpus_path = workspace / "documents.parquet"
    write_corpus(corpus, corpus_path)

    checks.frames_equal("parquet round-trips", read_corpus(corpus_path), corpus)
    checks.frames_equal("validate_corpus is idempotent", validate_corpus(corpus.copy()), corpus)

    counted = count_labels(corpus)

    checks.equal(
        "count_labels totals every label",
        sum(counted.values()),
        int(corpus["n_entities"].sum()),
    )
    checks.equal(
        "count_labels agrees with the raw json",
        counted,
        dict(
            Counter(
                entity["label"]
                for value in corpus["entities_json"]
                for entity in json.loads(value)
            )
        ),
    )
    checks.equal(
        "a corpus with no entities counts nothing",
        count_labels(corpus.assign(entities_json="[]", n_entities=0)),
        {},
    )

    vocabulary = ["DISEASE", "MEDICATION", "PROCEDURE"]
    mixed = build_corpus(
        documents,
        annotations.assign(
            label=[vocabulary[index % len(vocabulary)] for index in range(len(annotations))]
        ),
    )
    mixed_counts = count_labels(mixed)

    checks.equal("count_labels separates a mixed vocabulary", sorted(mixed_counts), vocabulary)
    checks.equal(
        "and each label keeps its own total",
        sum(mixed_counts.values()),
        int(mixed["n_entities"].sum()),
    )

    inconsistent = corpus.copy()
    inconsistent.loc[0, "n_entities"] = int(inconsistent.loc[0, "n_entities"]) + 1

    checks.raises("a stale n_entities is caught", ValueError, count_labels, inconsistent)

    fingerprints = document_fingerprints(corpus)

    checks.equal("one fingerprint per document", len(fingerprints), len(corpus))
    checks.equal("fingerprints are unique", len(set(fingerprints)), len(corpus))

    edited = corpus.copy()
    edited.loc[0, "text"] = edited.loc[0, "text"] + " extra"

    checks.check(
        "edited text changes its fingerprint",
        document_fingerprints(edited).iloc[0] != fingerprints.iloc[0],
    )
    checks.equal(
        "editing one document leaves the others alone",
        list(document_fingerprints(edited))[1:],
        list(fingerprints)[1:],
    )


def verify_stratification(checks: Checks) -> None:
    checks.equal("ratios normalize to one", sum(normalize_ratios([8, 2])), 1.0)
    checks.equal("normalized ratios keep proportions", normalize_ratios([8, 2]), [0.8, 0.2])
    checks.raises("negative ratio raises", ValueError, normalize_ratios, [1.0, -0.5])
    checks.raises("zero-sum ratios raise", ValueError, normalize_ratios, [0.0, 0.0])

    documents, annotations = synthetic_documents(30)
    corpus = build_corpus(documents, annotations)
    labels, counts = parse_document_label_counts(corpus)

    checks.equal("label axis is discovered", labels, ["DISEASE"])
    checks.equal("count matrix shape", counts.shape, (len(corpus), 1))
    checks.equal(
        "counts total the corpus entities",
        int(counts.sum()),
        int(corpus["n_entities"].sum()),
    )

    assignments = assign_partitions(corpus, ratios=[0.8, 0.2])

    checks.equal("every document is assigned", len(assignments), len(corpus))
    checks.equal("two partitions", sorted(assignments["fold"].unique()), [0, 1])
    checks.equal(
        "partition names default",
        sorted(assignments["partition_name"].unique()),
        ["train", "validation"],
    )

    repeat = assign_partitions(corpus, ratios=[0.8, 0.2])
    checks.frames_equal("same seed gives the same assignment", repeat, assignments)

    different = assign_partitions(corpus, ratios=[0.8, 0.2], random_state=7)
    checks.check(
        "a different seed can move documents",
        not different["fold"].equals(assignments["fold"]),
    )

    kfold = assign_partitions(corpus, ratios=[0.2] * 5, holdout_index=0)

    checks.equal("five partitions", sorted(kfold["fold"].unique()), [0, 1, 2, 3, 4])
    checks.equal(
        "holdout is named test",
        sorted(set(kfold["partition_name"])),
        ["fold_01", "fold_02", "fold_03", "fold_04", "test"],
    )

    checks.raises(
        "out-of-range holdout raises",
        ValueError,
        assign_partitions,
        corpus,
        ratios=[0.5, 0.5],
        holdout_index=5,
    )
    checks.raises(
        "more partitions than documents raises",
        ValueError,
        assign_partitions,
        corpus.head(2),
        ratios=[0.2] * 5,
    )

    validated = validate_assignments(
        documents_df=corpus, assignments_df=assignments, ratios=[0.8, 0.2]
    )
    checks.equal("validation accepts a matching manifest", len(validated), len(corpus))

    checks.raises(
        "validation rejects an edited corpus",
        ValueError,
        validate_assignments,
        documents_df=corpus.assign(text=corpus["text"] + "!"),
        assignments_df=assignments,
        ratios=[0.8, 0.2],
    )
    checks.raises(
        "validation rejects a missing document",
        ValueError,
        validate_assignments,
        documents_df=corpus.head(len(corpus) - 1),
        assignments_df=assignments,
        ratios=[0.8, 0.2],
    )


def verify_split(checks: Checks, workspace: Path) -> None:
    documents, annotations = synthetic_documents(30)
    corpus = build_corpus(documents, annotations)
    assignments = assign_partitions(corpus, ratios=[0.8, 0.2])

    partitions = split_documents(corpus, assignments)

    checks.equal("split names", sorted(partitions), ["train", "validation"])
    checks.equal(
        "partitions total the corpus",
        sum(len(frame) for frame in partitions.values()),
        len(corpus),
    )
    checks.equal(
        "partitions are disjoint",
        len(set(partitions["train"]["doc_id"]) & set(partitions["validation"]["doc_id"])),
        0,
    )

    split_dir = workspace / "split"
    result = create_split(corpus, split_dir, ratios=[0.8, 0.2])

    checks.equal("SplitResult carries every partition path", sorted(result.paths), ["train", "validation"])
    checks.check("partition files exist", all(path.exists() for path in result.paths.values()))
    checks.equal(
        "balance report covers every partition and metric",
        len(result.balance_report),
        2 * result.balance_report["metric"].nunique(),
    )
    checks.equal(
        "balance report metrics",
        sorted(result.balance_report["metric"].unique()),
        ["label:DISEASE", "n_documents", "n_entities"],
    )
    checks.equal(
        "balance report entity counts total the corpus",
        int(result.balance_report.loc[
            result.balance_report["metric"] == "n_entities", "count"
        ].sum()),
        int(corpus["n_entities"].sum()),
    )

    reused = create_split(corpus, split_dir, ratios=[0.8, 0.2], random_state=999)
    checks.frames_equal(
        "an existing split is reused, seed change ignored", reused.assignments, result.assignments
    )

    rebuilt = create_split(
        corpus, split_dir, ratios=[0.8, 0.2], random_state=999, reuse_assignments=False
    )
    checks.check(
        "reuse_assignments=False rebuilds",
        not rebuilt.assignments["fold"].equals(result.assignments["fold"]),
    )

    checks.raises(
        "reuse rejects an edited corpus",
        ValueError,
        create_split,
        corpus.assign(text=corpus["text"] + "!"),
        split_dir,
        ratios=[0.8, 0.2],
    )

    corpus_path = workspace / "corpus_for_split.parquet"
    write_corpus(corpus, corpus_path)

    from_frame = create_split(corpus, workspace / "split_frame", ratios=[0.8, 0.2])
    from_disk = create_split(corpus_path, workspace / "split_disk", ratios=[0.8, 0.2])

    checks.frames_equal(
        "frame input and path input agree", from_disk.assignments, from_frame.assignments
    )

    loaded = read_split(workspace / "split_frame")
    expected = split_documents(corpus, from_frame.assignments)

    checks.equal("read_split finds both partitions", sorted(loaded), ["train", "validation"])
    checks.frames_equal(
        "read_split round-trips train",
        loaded["train"].reset_index(drop=True),
        expected["train"].reset_index(drop=True),
    )
    checks.equal(
        "read_split partitions stay disjoint",
        len(set(loaded["train"]["doc_id"]) & set(loaded["validation"]["doc_id"])),
        0,
    )


def verify_mismatch_policies(
    checks: Checks,
    documents: dict[str, str],
    annotations: pd.DataFrame,
) -> None:
    unannotated = sorted(documents)[-1]
    partial = annotations[annotations["filename"] != unannotated]
    mixed = pd.concat([partial, partial.head(1).assign(filename="ghost")], ignore_index=True)

    checks.raises("mismatch raises by default", ValueError, build_corpus, documents, mixed)

    documents_win = build_corpus(documents, mixed, on_mismatch="documents")

    checks.equal("documents policy keeps every document", len(documents_win), len(documents))
    checks.equal(
        "documents policy leaves the unannotated document empty",
        int(documents_win.loc[documents_win["doc_id"] == unannotated, "n_entities"].iloc[0]),
        0,
    )

    annotations_win = build_corpus(documents, mixed, on_mismatch="annotations")

    checks.equal(
        "annotations policy drops the unannotated document",
        len(annotations_win),
        len(documents) - 1,
    )
    checks.check(
        "annotations policy invents no document for the orphan",
        "ghost" not in set(annotations_win["doc_id"]),
    )

    kept_documents, kept_annotations, report = resolve_mismatch(documents, mixed, "annotations")

    checks.equal("report names the dropped document", report.dropped_documents, [unannotated])
    checks.equal("report names the orphan annotation", report.dropped_annotation_files, ["ghost"])
    checks.equal("kept documents match the report", len(kept_documents), len(documents) - 1)
    checks.equal("kept annotations drop the orphan", len(kept_annotations), len(partial))

    _, _, clean = resolve_mismatch(documents, annotations, "annotations")

    checks.equal("an agreeing corpus drops nothing", clean.dropped_documents, [])
    checks.raises(
        "unknown policy raises", ValueError, resolve_mismatch, documents, mixed, "txt"
    )


def verify_conflict_policies(
    checks: Checks,
    documents: dict[str, str],
    annotations: pd.DataFrame,
) -> None:
    drifted = annotations.copy()
    target = drifted.index[0]
    filename = str(drifted.at[target, "filename"])
    start = int(drifted.at[target, "start_span"]) + 1
    end = int(drifted.at[target, "end_span"]) + 1
    drifted.loc[target, ["start_span", "end_span"]] = [start, end]

    checks.raises("conflict raises by default", ValueError, build_corpus, documents, drifted)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rewritten = build_corpus(documents, drifted, on_conflict="rewrite")

    entity = entities_of(rewritten, filename)[0]

    checks.equal(
        "rewrite keeps the annotated offsets",
        (entity["start"], entity["end"]),
        (start, end),
    )
    checks.equal(
        "rewrite takes the document text",
        entity["text"],
        documents[filename][start:end],
    )
    checks.check("rewrite warns", any("on_conflict" in str(item.message) for item in caught))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_corpus(documents, drifted, on_conflict="drop")

    checks.check("drop warns", any("on_conflict='drop'" in str(item.message) for item in caught))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        kept, resolved, report = resolve_conflicts(documents, drifted, "rewrite")

        checks.equal("report names the rewritten document", report.rewritten_documents, [filename])
        checks.equal("report counts the rewrite", report.n_rewritten_entities, 1)
        checks.equal("rewrite keeps every document", len(kept), len(documents))
        checks.equal(
            "the resolved annotation matches the document",
            str(resolved.at[target, "text"]),
            documents[filename][start:end],
        )

        _, _, clean = resolve_conflicts(documents, annotations, "rewrite")

        checks.equal("an agreeing corpus rewrites nothing", clean.n_rewritten_entities, 0)
        checks.equal("an agreeing corpus names no document", clean.rewritten_documents, [])

        kept, remaining, dropped_report = resolve_conflicts(documents, drifted, "drop")

        checks.equal(
            "report names the dropped document", dropped_report.dropped_documents, [filename]
        )
        checks.equal("drop rewrites nothing", dropped_report.n_rewritten_entities, 0)
        checks.equal("drop removes the document", len(kept), len(documents) - 1)
        checks.check("the dropped document is gone", filename not in kept)
        checks.check(
            "drop takes the document's clean annotations with it",
            not (remaining["filename"].astype(str) == filename).any(),
        )

        dropped_corpus = build_corpus(documents, drifted, on_conflict="drop")

        checks.equal("the dropped corpus omits the document", len(dropped_corpus), len(documents) - 1)

        checks.raises(
            "drop refuses to empty the corpus",
            ValueError,
            resolve_conflicts,
            {filename: documents[filename]},
            drifted[drifted["filename"].astype(str) == filename],
            "drop",
        )

        out_of_range = annotations.copy()
        out_of_range.loc[target, "end_span"] = len(documents[filename]) + 50

        checks.raises(
            "rewrite still rejects out-of-range offsets",
            ValueError,
            build_corpus,
            documents,
            out_of_range,
            on_conflict="rewrite",
        )
    checks.raises(
        "unknown conflict policy raises",
        ValueError,
        resolve_conflicts,
        documents,
        drifted,
        "txt",
    )
    checks.raises(
        "resolve_conflicts rejects an unaligned pair",
        ValueError,
        resolve_conflicts,
        documents,
        annotations.assign(filename="ghost"),
    )


def verify_prepare_dataset(checks: Checks, workspace: Path) -> None:
    documents, annotations = synthetic_documents(30)
    txt_dir, ann_dir = write_brat(workspace / "prepare_source", documents, annotations)

    output_dir = workspace / "prepared"
    prepared = prepare_dataset(
        output_dir=output_dir, documents=txt_dir, annotations=ann_dir, dataset_name="synthetic"
    )

    checks.equal("dataset root is named", prepared.dataset_root.name, "synthetic")
    checks.check("corpus parquet is written", prepared.corpus_path.exists())
    checks.equal("split descriptor", prepared.split_dir.name, "train_val_80_20")
    checks.equal("corpus row count", len(prepared.corpus), len(documents))

    manifest_path = prepared.dataset_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    checks.check("manifest records normalize_labels", "normalize_labels" in manifest)
    checks.equal("manifest records the default", manifest["normalize_labels"], False)

    checks.equal(
        "the manifest names the labels present",
        manifest["n_entities_by_label"],
        count_labels(prepared.corpus),
    )
    checks.equal(
        "and their counts sum to n_entities",
        sum(manifest["n_entities_by_label"].values()),
        manifest["n_entities"],
    )

    checks.equal("manifest records the mismatch policy", manifest["on_mismatch"], "error")
    checks.equal("nothing is dropped when the sources agree", manifest["dropped_documents"], [])
    checks.equal("manifest records the conflict policy", manifest["on_conflict"], "raise")
    checks.equal(
        "nothing is rewritten when the annotations agree",
        manifest["n_rewritten_entities"],
        0,
    )
    checks.equal(
        "nothing is dropped when the annotations agree",
        manifest["dropped_conflicting_documents"],
        [],
    )

    (txt_dir / "unannotated.txt").write_text("Sin anotaciones en este documento.", encoding="utf-8")
    (ann_dir / "ghost.ann").write_text("T1\tDISEASE 0 6\tfiebre\n", encoding="utf-8")

    checks.raises(
        "prepare_dataset raises on a mismatch by default",
        ValueError,
        prepare_dataset,
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        dataset_name="synthetic_mismatch",
    )

    reconciled = prepare_dataset(
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        dataset_name="synthetic_reconciled",
        on_mismatch="annotations",
    )
    reconciled_manifest = json.loads(
        (reconciled.dataset_root / "source_manifest.json").read_text()
    )

    checks.equal("the reconciled corpus keeps only annotated documents", len(reconciled.corpus), len(documents))
    checks.equal(
        "the manifest names the dropped document",
        reconciled_manifest["dropped_documents"],
        ["unannotated"],
    )
    checks.equal(
        "the manifest names the dropped annotation file",
        reconciled_manifest["dropped_annotation_files"],
        ["ghost"],
    )

    (txt_dir / "unannotated.txt").unlink()
    (ann_dir / "ghost.ann").unlink()

    (txt_dir / "drifted.txt").write_text("Fiebre alta y tos.", encoding="utf-8")
    (ann_dir / "drifted.ann").write_text("T1\tDISEASE 1 7\tFiebre\n", encoding="utf-8")

    checks.raises(
        "prepare_dataset raises on a conflict by default",
        ValueError,
        prepare_dataset,
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        dataset_name="synthetic_conflict",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        rewritten = prepare_dataset(
            output_dir=output_dir,
            documents=txt_dir,
            annotations=ann_dir,
            dataset_name="synthetic_rewritten",
            on_conflict="rewrite",
        )

    rewritten_manifest = json.loads(
        (rewritten.dataset_root / "source_manifest.json").read_text()
    )

    checks.equal(
        "the manifest names the rewritten document",
        rewritten_manifest["rewritten_documents"],
        ["drifted"],
    )
    checks.equal(
        "the manifest counts the rewrite", rewritten_manifest["n_rewritten_entities"], 1
    )
    checks.equal(
        "the rewritten corpus stores the document text",
        entities_of(rewritten.corpus, "drifted")[0]["text"],
        "iebre ",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        dropped = prepare_dataset(
            output_dir=output_dir,
            documents=txt_dir,
            annotations=ann_dir,
            dataset_name="synthetic_dropped",
            on_conflict="drop",
        )

    dropped_manifest = json.loads((dropped.dataset_root / "source_manifest.json").read_text())

    checks.equal(
        "the manifest names the dropped conflicting document",
        dropped_manifest["dropped_conflicting_documents"],
        ["drifted"],
    )
    checks.equal(
        "the dropped corpus omits the conflicting document",
        len(dropped.corpus),
        len(documents),
    )
    checks.check(
        "the conflicting document is not in the corpus",
        "drifted" not in set(dropped.corpus["doc_id"].astype(str)),
    )

    (txt_dir / "drifted.txt").unlink()
    (ann_dir / "drifted.ann").unlink()

    kfold = prepare_dataset(
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        dataset_name="synthetic_kfold",
        kfolds=5,
        holdout_fold=0,
    )

    checks.equal("kfold descriptor", kfold.split_dir.name, "kfold_5_holdout_0")
    checks.check("holdout is written", (kfold.split_dir / "test.parquet").exists())
    checks.check("rotatable folds are written", (kfold.split_dir / "fold_01.parquet").exists())

    corpus_only = prepare_dataset(
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        dataset_name="synthetic_nosplit",
        split=False,
    )

    checks.check("split=False writes no split", corpus_only.split_dir is None)
    checks.check("split=False still writes the corpus", corpus_only.corpus_path.exists())

    from_parquet = prepare_dataset(
        output_dir=output_dir,
        source_parquet=prepared.corpus_path,
        dataset_name="from_parquet",
    )

    checks.frames_equal(
        "parquet input reproduces the corpus", from_parquet.corpus, prepared.corpus
    )
    checks.equal(
        "normalize_labels is not recorded for a parquet source",
        json.loads((from_parquet.dataset_root / "source_manifest.json").read_text())[
            "normalize_labels"
        ],
        None,
    )

    checks.raises(
        "no input raises", ValueError, prepare_dataset, output_dir=output_dir
    )
    checks.raises(
        "both inputs raise",
        ValueError,
        prepare_dataset,
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        source_parquet=prepared.corpus_path,
    )
    checks.raises(
        "documents without annotations raises",
        ValueError,
        prepare_dataset,
        output_dir=output_dir,
        documents=txt_dir,
    )
    checks.raises(
        "kfolds below two raises",
        ValueError,
        prepare_dataset,
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        kfolds=1,
    )
    checks.raises(
        "holdout outside the folds raises",
        ValueError,
        prepare_dataset,
        output_dir=output_dir,
        documents=txt_dir,
        annotations=ann_dir,
        kfolds=5,
        holdout_fold=9,
    )


def verify_real_corpus(checks: Checks, workspace: Path) -> None:
    root = samples_root()

    if root is None:
        checks.skip("real corpus conversion", "sample corpora not found")
        return

    source = root / "MultiClinNER-es-train-disease"

    if not (source / "txt").is_dir():
        checks.skip("real corpus conversion", f"{source} has no txt/ directory")
        return

    corpus = build_corpus(source / "txt", source / "ann")

    checks.check("real corpus is non-empty", len(corpus) > 0)
    checks.check("real corpus has entities", int(corpus["n_entities"].sum()) > 0)
    checks.frames_equal("real corpus validates unchanged", validate_corpus(corpus.copy()), corpus)

    labels = {
        entity["label"]
        for value in corpus["entities_json"]
        for entity in json.loads(value)
    }
    checks.check(f"real corpus labels are verbatim ({sorted(labels)})", "ENFERMEDAD" in labels or "DISEASE" in labels)

    result = create_split(corpus, workspace / "real_split", ratios=[0.8, 0.2])
    checks.equal(
        "real split covers every document",
        sum(len(pd.read_parquet(path)) for path in result.paths.values()),
        len(corpus),
    )


def main() -> int:
    checks = Checks("data")

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        verify_labels(checks)
        verify_brat(checks, workspace)
        verify_corpus(checks, workspace)
        verify_stratification(checks)
        verify_split(checks, workspace)
        verify_prepare_dataset(checks, workspace)
        verify_real_corpus(checks, workspace)

    return checks.report()


if __name__ == "__main__":
    run(main)
