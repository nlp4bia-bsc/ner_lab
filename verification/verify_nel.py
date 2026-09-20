"""lab.nel: the ported nlp4bia-linking tests, side-by-side equivalence with it, and `link_entities` end to end."""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd

from _harness import Checks, run

from lab.nel import (
    EntityLinkingPipeline,
    GazetteerEntry,
    MatchCandidate,
    MentionAnnotation,
    build_matcher,
    link_entities,
    reciprocal_rank_fusion,
)
from lab.nel.evaluation import (
    evaluate_candidate_dataframe,
    load_snomed_graph_pickle,
    retrieval_metrics_from_codes,
    snomed_graph_distance_and_direction,
)
from lab.nel.io import read_table, write_table
from lab.nel.matching import MATCHER_REGISTRY, StringMatchMatcher
from lab.nel.retrieval import FaissBiEncoder, MatrixBiEncoder

NLP4BIA_ENV = "NLP4BIA_LINKING_ROOT"
DEFAULT_NLP4BIA = Path.home() / "bsc" / "nlp4bia-linking"

GAZETTEER = pd.DataFrame(
    [
        ("insuficiencia cardiaca", "84114007", "DISEASE"),
        ("fallo cardiaco", "84114007", "DISEASE"),
        ("infarto de miocardio", "22298006", "DISEASE"),
        ("infarto agudo de miocardio", "57054005", "DISEASE"),
        ("insuficiencia renal", "42399005", "DISEASE"),
        ("insuficiencia renal aguda", "14669001", "DISEASE"),
        ("fallo renal", "42399005", "DISEASE"),
        ("diabetes mellitus", "73211009", "DISEASE"),
        ("diabetes mellitus tipo 2", "44054006", "DISEASE"),
        ("hipertension arterial", "38341003", "DISEASE"),
        ("hipertensión arterial", "38341003", "DISEASE"),
        ("neumonia", "233604007", "DISEASE"),
        ("neumonía bacteriana", "53084003", "DISEASE"),
        ("fibrilacion auricular", "49436004", "DISEASE"),
        ("anemia ferropenica", "87522002", "DISEASE"),
        ("anemia", "271737000", "DISEASE"),
        ("cirrosis hepatica", "19943007", "DISEASE"),
        ("hepatitis", "128241005", "DISEASE"),
        ("ecocardiograma", "40701008", "PROCEDURE"),
        ("radiografia de torax", "399208008", "PROCEDURE"),
        ("cateterismo cardiaco", "41976001", "PROCEDURE"),
        ("biopsia hepatica", "86259008", "PROCEDURE"),
        ("colonoscopia", "73761001", "PROCEDURE"),
        ("trasplante renal", "70536003", "PROCEDURE"),
    ],
    columns=["term", "code", "label"],
)

MENTIONS = pd.DataFrame(
    [
        ("d1", "DISEASE", 0, 22, "insuficiencia cardiaca", "84114007"),
        ("d1", "DISEASE", 40, 62, "insuficiencia cardíaca", "84114007"),
        ("d1", "DISEASE", 80, 97, "infarto miocardio", "22298006"),
        ("d2", "DISEASE", 0, 26, "infarto agudo de miocardio", "57054005"),
        ("d2", "DISEASE", 30, 41, "fallo renal", "42399005"),
        ("d2", "DISEASE", 50, 75, "insuficiencia renal aguda", "14669001"),
        ("d3", "DISEASE", 0, 17, "diabetes tipo 2", "44054006"),
        ("d3", "DISEASE", 20, 32, "hipertensión", "38341003"),
        ("d3", "DISEASE", 40, 48, "neumonia", "233604007"),
        ("d4", "DISEASE", 0, 24, "fibrilación auricular", "49436004"),
        ("d4", "DISEASE", 30, 36, "anemia", "271737000"),
        ("d4", "PROCEDURE", 40, 54, "ecocardiograma", "40701008"),
        ("d5", "PROCEDURE", 0, 20, "rx de torax", "399208008"),
        ("d5", "PROCEDURE", 30, 44, "biopsia hepática", "86259008"),
        ("d5", None, 50, 60, "hepatitis", "128241005"),
    ],
    columns=["filename", "label", "start_span", "end_span", "text", "code"],
)


def original_package():
    root = Path(os.environ.get(NLP4BIA_ENV, DEFAULT_NLP4BIA))
    source = root / "src"

    if not (source / "nlp4bia_linking").is_dir():
        return None

    sys.path.insert(0, str(source))

    import nlp4bia_linking

    return nlp4bia_linking


def entries_with(entry_class) -> list:
    return [entry_class(term=row.term, code=row.code, label=row.label) for row in GAZETTEER.itertuples()]


def mentions_with(mention_class) -> list:
    return [
        mention_class(row.filename, row.label, row.start_span, row.end_span, row.text, row.code)
        for row in MENTIONS.itertuples()
    ]


def candidate_tuples(candidates: list) -> list[tuple]:
    return [(c.code, round(float(c.score), 6), c.rank, c.method, c.candidate_term) for c in candidates]


def verify_ported_tests(checks: Checks) -> None:
    gazetteer = [GazetteerEntry("heart failure", "A", "DISEASE"), GazetteerEntry("renal failure", "B", "DISEASE")]
    mention = MentionAnnotation("doc", "DISEASE", 0, 13, "heart failure", None)
    result = EntityLinkingPipeline(StringMatchMatcher(gazetteer=gazetteer)).link_mentions([mention])[0]
    checks.equal("string match pipeline links the exact term", result.predicted_code, "A")
    checks.equal("the top candidate has rank 1", result.candidates[0].rank, 1)

    retriever = MatrixBiEncoder(top_k=2, use_word_ngrams=False)
    retriever.build_index([GazetteerEntry("heart failure", "A"), GazetteerEntry("renal failure", "B")])
    candidates = retriever.search([MentionAnnotation("doc", None, 0, 13, "heart failure", None)])[0]
    checks.equal("matrix retrieval ranks the exact term first", candidates[0].code, "A")

    def candidate(code, score, rank, method):
        return MatchCandidate(None, "doc", "mention", None, code, code, score, method, rank=rank)

    class StaticRetriever:
        def __init__(self, method, candidates):
            self.method = method
            self._candidates = candidates

        def search(self, mentions):
            return [list(self._candidates) for _ in mentions]

    first = StaticRetriever("first", [candidate("A", 0.8, 1, "first"), candidate("B", 0.7, 2, "first")])
    second = StaticRetriever("second", [candidate("B", 0.9, 1, "second"), candidate("A", 0.6, 2, "second")])
    fused = EntityLinkingPipeline(candidate_generators=[first, second], top_k_candidates=2).link_mentions(
        [MentionAnnotation("doc", None, 0, 7, "mention", None)]
    )[0]
    checks.equal("two generators are fused by RRF, ties broken by best rank", fused.predicted_code, "A")
    checks.equal("fused candidates carry the rrf method", [c.method for c in fused.candidates], ["rrf", "rrf"])
    checks.equal("fusion records its sources", fused.candidates[0].metadata["retrieval_sources"], ["first", "second"])

    result = reciprocal_rank_fusion(
        {
            "a": [candidate("A", 0.9, 1, "a"), candidate("B", 0.8, 2, "a")],
            "b": [candidate("B", 0.95, 1, "b"), candidate("A", 0.7, 2, "b")],
        },
        k=60,
    )
    checks.equal("rrf is deterministic on a tie", [item.code for item in result], ["A", "B"])
    checks.equal("rrf keeps source ranks", result[0].metadata["source_ranks"], {"a": 1, "b": 2})
    checks.equal("rrf keeps source scores", result[1].metadata["source_scores"], {"a": 0.8, "b": 0.95})

    metrics = retrieval_metrics_from_codes(["A", "B", "C"], [["A", "A"], ["X", "B"], []], [1, 2])
    checks.equal("recall@1 deduplicates candidates", metrics["recall@1"], 1 / 3)
    checks.equal("recall@2 uses one-based ranks", metrics["recall@2"], 2 / 3)
    checks.equal("mrr", metrics["mrr"], (1 + 0.5 + 0) / 3)
    checks.equal("coverage counts non-empty lists", metrics["coverage"], 2 / 3)

    graph = nx.DiGraph()
    graph.add_edge("PARENT", "CHILD")
    undirected = graph.to_undirected()
    checks.equal("same code is exact", snomed_graph_distance_and_direction(graph, undirected, "PARENT", "PARENT")["direction"], "exact")
    checks.equal("a child prediction is narrow", snomed_graph_distance_and_direction(graph, undirected, "PARENT", "CHILD")["direction"], "narrow")
    checks.equal("a parent prediction is broad", snomed_graph_distance_and_direction(graph, undirected, "CHILD", "PARENT")["direction"], "broad")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hierarchy.pkl"

        with path.open("wb") as handle:
            pickle.dump((graph, undirected), handle)

        loaded_graph, loaded_undirected = load_snomed_graph_pickle(path)
        checks.equal("a (graph, undirected) pickle loads", list(loaded_graph.edges()), [("PARENT", "CHILD")])
        checks.check("the undirected view is kept", loaded_undirected.has_edge("PARENT", "CHILD"))

        expected = pd.DataFrame([{"mention": "heart failure", "code": "A"}])

        for suffix in (".tsv", ".jsonl", ".parquet"):
            table = Path(tmp) / f"records{suffix}"
            write_table(expected, table)
            checks.equal(f"{suffix} round-trips", read_table(table).to_dict("records"), expected.to_dict("records"))

    frame = pd.DataFrame([{"gold_code": "A", "codes": '["A", "B"]'}, {"gold_code": "B", "codes": '["X", "B"]'}])
    metrics = evaluate_candidate_dataframe(frame, [1, 2])
    checks.equal("serialized code lists evaluate: recall@1", metrics["recall@1"], 0.5)
    checks.equal("serialized code lists evaluate: recall@2", metrics["recall@2"], 1.0)
    checks.equal("serialized code lists evaluate: mrr", metrics["mrr"], 0.75)


def verify_equivalence(checks: Checks, original) -> None:
    if original is None:
        checks.skip("equivalence with nlp4bia-linking", f"set {NLP4BIA_ENV} to its checkout")
        return

    from nlp4bia_linking.ensembling import reciprocal_rank_fusion as original_rrf
    from nlp4bia_linking.evaluation import (
        retrieval_metrics_from_codes as original_metrics,
        snomed_graph_distance_and_direction as original_distance,
    )
    from nlp4bia_linking.matching import MATCHER_REGISTRY as ORIGINAL_REGISTRY, build_matcher as original_matcher
    from nlp4bia_linking.preprocessing import normalize_text as original_normalize
    from nlp4bia_linking.retrieval import FaissBiEncoder as OriginalFaiss, MatrixBiEncoder as OriginalMatrix

    from lab.nel.preprocessing import normalize_text

    checks.equal("the same matcher names are registered", sorted(MATCHER_REGISTRY), sorted(ORIGINAL_REGISTRY))

    ours_entries, theirs_entries = entries_with(GazetteerEntry), entries_with(original.GazetteerEntry)
    ours_mentions, theirs_mentions = mentions_with(MentionAnnotation), mentions_with(original.MentionAnnotation)

    for name in sorted(MATCHER_REGISTRY):
        ours = build_matcher(name, gazetteer=ours_entries, top_k=5).fit().predict(ours_mentions)
        theirs = original_matcher(name, gazetteer=theirs_entries, top_k=5).fit().predict(theirs_mentions)
        checks.equal(
            f"{name} matches the original on every mention",
            [candidate_tuples(c) for c in ours],
            [candidate_tuples(c) for c in theirs],
        )

    for label, ours_class, theirs_class, kwargs in (
        ("matrix", MatrixBiEncoder, OriginalMatrix, {"top_k": 5}),
        ("matrix without word n-grams", MatrixBiEncoder, OriginalMatrix, {"top_k": 5, "use_word_ngrams": False}),
        ("matrix with a threshold", MatrixBiEncoder, OriginalMatrix, {"top_k": 5, "threshold": 0.3}),
        ("faiss", FaissBiEncoder, OriginalFaiss, {"top_k": 5, "device": "cpu"}),
    ):
        ours_retriever, theirs_retriever = ours_class(**kwargs), theirs_class(**kwargs)
        ours_retriever.build_index(ours_entries)
        theirs_retriever.build_index(theirs_entries)
        checks.equal(
            f"{label} retrieval matches the original",
            [candidate_tuples(c) for c in ours_retriever.search(ours_mentions)],
            [candidate_tuples(c) for c in theirs_retriever.search(theirs_mentions)],
        )

    def rankings(candidate_class):
        def candidate(code, score, rank, method):
            return candidate_class(None, "doc", "m", None, code, code, score, method, rank=rank)

        return {
            "a": [candidate("A", 0.9, 1, "a"), candidate("B", 0.8, 2, "a"), candidate("C", 0.1, 3, "a")],
            "b": [candidate("B", 0.95, 1, "b"), candidate("A", 0.7, 2, "b"), candidate("D", 0.5, 3, "b")],
            "c": [candidate("D", 0.99, 1, "c")],
        }

    checks.equal(
        "rrf matches the original",
        [asdict(c) for c in reciprocal_rank_fusion(rankings(MatchCandidate), k=60, top_k=3)],
        [asdict(c) for c in original_rrf(rankings(original.MatchCandidate), k=60, top_k=3)],
    )

    gold = ["A", "B", "C", "D"]
    predicted = [["A", "A", "B"], '["X", "B"]', [], "D"]
    checks.equal(
        "retrieval metrics match the original",
        retrieval_metrics_from_codes(gold, predicted, [1, 2, 5]),
        original_metrics(gold, predicted, [1, 2, 5]),
    )

    texts = ["Insuficiencia  Cardíaca", "Infarto-de miocardio.", "  NEUMONÍA ", "ééé"]
    for kwargs in ({}, {"strip_accents": True}, {"normalize_punct": True}, {"lowercase": False, "strip_accents": True, "normalize_punct": True}):
        checks.equal(
            f"normalize_text matches the original with {kwargs or 'defaults'}",
            [normalize_text(t, **kwargs) for t in texts],
            [original_normalize(t, **kwargs) for t in texts],
        )

    graph = nx.DiGraph()
    graph.add_edges_from([("ROOT", "A"), ("A", "B"), ("B", "C"), ("A", "D"), ("ROOT", "E")])
    undirected = graph.to_undirected()
    pairs = [("A", "A"), ("A", "C"), ("C", "A"), ("B", "D"), ("D", "E"), ("A", "ZZZ"), ("A", None)]
    checks.equal(
        "graph distance and direction match the original",
        [snomed_graph_distance_and_direction(graph, undirected, g, p) for g, p in pairs],
        [original_distance(graph, undirected, g, p) for g, p in pairs],
    )

    from lab.nel.brat import load_brat_ann
    from lab.nel.io import load_annotations_tsv, load_gazetteer_tsv
    from nlp4bia_linking.data import (
        load_annotations_tsv as original_annotations,
        load_brat_ann as original_brat,
        load_gazetteer_tsv as original_gazetteer,
    )

    with tempfile.TemporaryDirectory() as tmp:
        annotations = Path(tmp) / "mentions.tsv"
        MENTIONS.assign(is_abbreviation=["yes", "no", "1", "0", "", "true", "false", None, "", "", "", "", "", "", ""]).to_csv(annotations, sep="\t", index=False)
        gazetteer = Path(tmp) / "gazetteer.tsv"
        GAZETTEER.assign(sem_tag="disorder").to_csv(gazetteer, sep="\t", index=False)
        ann = Path(tmp) / "doc.ann"
        ann.write_text(
            "T1\tDISEASE 0 22\tinsuficiencia cardiaca\nN1\tReference T1 SNOMED:84114007\tinsuficiencia cardiaca\n"
            "T2\tPROCEDURE 40 54\tecocardiograma\n",
            encoding="utf-8",
        )
        checks.equal("annotation TSV reader matches the original", [asdict(m) for m in load_annotations_tsv(annotations)], [asdict(m) for m in original_annotations(annotations)])
        checks.equal("gazetteer TSV reader matches the original", [asdict(e) for e in load_gazetteer_tsv(gazetteer)], [asdict(e) for e in original_gazetteer(gazetteer)])
        checks.equal("BRAT reader matches the original", [asdict(m) for m in load_brat_ann(ann)], [asdict(m) for m in original_brat(ann)])


def verify_link_entities(checks: Checks) -> None:
    from lab.core.tasks import resolve_task
    from lab.nel.linking import GOLD_COLUMN, LINKED_COLUMNS

    checks.check("the task name resolves to link_entities", resolve_task("nel.link_entities") is link_entities)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spans_path, gazetteer_path = root / "spans.tsv", root / "gazetteer.tsv"
        MENTIONS.to_csv(spans_path, sep="\t", index=False)
        GAZETTEER.to_csv(gazetteer_path, sep="\t", index=False)

        result = link_entities(spans_path, gazetteer_path, root / "matrix", method="matrix", top_k=5, k_values=(1, 5))

        checks.equal("the input columns are kept, code becoming gold_code, and four columns appended", result.spans.columns.tolist(), [*MENTIONS.columns.drop("code"), GOLD_COLUMN, *LINKED_COLUMNS])
        checks.equal("one row per input span", len(result.spans), len(MENTIONS))
        checks.equal("exact terms link to their code", result.spans["code"].iloc[0], "84114007")
        checks.equal("the label restricts candidates: a procedure never links to a disease code", result.spans.loc[result.spans["label"] == "PROCEDURE", "code"].isin(GAZETTEER.loc[GAZETTEER["label"] == "PROCEDURE", "code"]).all(), True)
        checks.check("every mention got a code", result.spans["code"].notna().all())
        checks.check("recall@5 is high on near-exact mentions", result.metrics["recall@5"] >= 0.9)
        checks.equal("metrics count the scored rows", (result.metrics["n_evaluated_rows"], result.metrics["n_rows_without_gold"]), (len(MENTIONS), 0))
        checks.equal("metrics name the method", result.metrics["method"], "matrix")

        candidates = json.loads(result.spans["candidates_json"].iloc[0])
        checks.equal("candidates_json holds top_k records with rank", ([c["rank"] for c in candidates], candidates[0]["code"]), ([1, 2, 3, 4, 5], "84114007"))
        checks.equal("code_term and code_score come from the top candidate", (result.spans["code_term"].iloc[0], result.spans["code_score"].iloc[0]), (candidates[0]["term"], candidates[0]["score"]))

        checks.equal("three files are written", sorted(p.name for p in (root / "matrix").iterdir()), ["linking_manifest.json", "linking_metrics.json", "predictions.tsv"])
        written = pd.read_csv(result.paths["predictions"], sep="\t", dtype=str)
        checks.equal("predictions.tsv carries the same columns", written.columns.tolist(), result.spans.columns.tolist())
        checks.equal("the manifest records the inputs' sha256", (result.manifest["spans_sha256"] is not None, result.manifest["gazetteer_sha256"] is not None, result.manifest["n_gazetteer_entries"], result.manifest["scored_against_gold"]), (True, True, len(GAZETTEER), True))
        checks.equal("the manifest records the method and top_k", (result.manifest["method"], result.manifest["top_k"], result.manifest["k_values"]), ("matrix", 5, [1, 5]))

        relinked = link_entities(result.spans, GAZETTEER, root / "relinked", method="levenshtein", top_k=3, k_values=(1,))
        checks.equal("a linked table links again: gold_code survives and the code columns are overwritten", relinked.spans.columns.tolist(), result.spans.columns.tolist())
        checks.equal("in-memory inputs leave the manifest paths empty", (relinked.manifest["spans"], relinked.manifest["spans_sha256"]), (None, None))

        unlabelled = link_entities(MENTIONS.drop(columns=["code"]), GAZETTEER, root / "nogold", method="bm25", top_k=3, k_values=(1, 3))
        checks.equal("without gold there are no metrics and no metrics file", (unlabelled.metrics, sorted(p.name for p in (root / "nogold").iterdir())), (None, ["linking_manifest.json", "predictions.tsv"]))
        checks.check("without gold no gold column appears", GOLD_COLUMN not in unlabelled.spans.columns)

        partial = MENTIONS.copy()
        partial.loc[partial.index[:5], "code"] = ""
        partial_result = link_entities(partial, GAZETTEER, root / "partial", method="string_match", top_k=3, k_values=(1,))
        checks.equal("only rows with a gold code are scored", (partial_result.metrics["n_evaluated_rows"], partial_result.metrics["n_rows_without_gold"]), (len(MENTIONS) - 5, 5))

        graph = nx.DiGraph()
        graph.add_edges_from([("84114007", "42399005"), ("22298006", "57054005"), ("73211009", "44054006"), ("42399005", "14669001"), ("271737000", "87522002")])
        hierarchy = root / "hierarchy.pkl"
        with hierarchy.open("wb") as handle:
            pickle.dump(graph, handle)
        graded = link_entities(MENTIONS, GAZETTEER, root / "graded", method="matrix", top_k=5, k_values=(1,), hierarchy=hierarchy)
        checks.check("a hierarchy adds the direction rates", {"exact_rate", "narrow_rate", "broad_rate", "non_rel_rate", "mean_graph_distance"} <= set(graded.metrics))
        checks.equal("the manifest records the hierarchy", graded.manifest["hierarchy"], str(hierarchy.resolve()))

        checks.raises("an unknown method is refused", ValueError, link_entities, MENTIONS, GAZETTEER, root / "x", method="magic", match="Unknown method")
        checks.raises("encoder methods need base_model", ValueError, link_entities, MENTIONS, GAZETTEER, root / "x", method="dense", match="requires base_model")
        checks.raises("base_model is refused elsewhere", ValueError, link_entities, MENTIONS, GAZETTEER, root / "x", method="matrix", base_model="m", match="only applies")
        checks.raises("k_values beyond top_k are refused", ValueError, link_entities, MENTIONS, GAZETTEER, root / "x", top_k=3, k_values=(1, 5), match="k_values")
        checks.raises("a span table without the canonical columns is refused", ValueError, link_entities, MENTIONS.drop(columns=["text"]), GAZETTEER, root / "x", match="missing columns")
        checks.raises("a gazetteer without term/code is refused", ValueError, link_entities, MENTIONS, GAZETTEER.drop(columns=["code"]), root / "x", match="missing columns")

        verify_encoder_paths(checks, root)


def verify_encoder_paths(checks: Checks, root: Path) -> None:
    from fixtures import tiny_base_model

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception as error:
        checks.skip("encoder retrievers and the reranker", f"no cached bert-base-uncased tokenizer ({type(error).__name__})")
        return

    base_model = tiny_base_model(root, tokenizer)

    for method, kwargs in (("transformer_faiss", {"device": "cpu", "batch_size": 8}), ("dense", {"device": "cpu"})):
        result = link_entities(MENTIONS, GAZETTEER, root / method, method=method, base_model=str(base_model), method_kwargs=kwargs, top_k=5, k_values=(1, 5))
        checks.equal(f"{method} links every mention with five candidates", ([len(json.loads(c)) for c in result.spans["candidates_json"]], result.spans["code"].notna().all()), ([5] * len(MENTIONS), True))
        checks.equal(f"{method} records base_model", result.manifest["base_model"], str(base_model))

    reranked = link_entities(MENTIONS, GAZETTEER, root / "reranked", method="matrix", reranker=base_model, reranker_kwargs={"device": "cpu", "batch_size": 8}, top_k=5, k_values=(1, 5))
    candidates = [json.loads(c) for c in reranked.spans["candidates_json"]]
    checks.equal("the reranker rescored every candidate", {c["method"] for row in candidates for c in row}, {"cross_encoder"})
    checks.check("reranked candidates are in descending score order", all([c["score"] for c in row] == sorted((c["score"] for c in row), reverse=True) for row in candidates))
    checks.equal("the metrics name the reranked method", reranked.metrics["method"], "matrix+cross_encoder")
    checks.equal("the manifest records the reranker", reranked.manifest["reranker"], str(base_model.resolve()))


def main() -> int:
    checks = Checks("nel")

    verify_ported_tests(checks)
    verify_equivalence(checks, original_package())
    verify_link_entities(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
