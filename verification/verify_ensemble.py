"""Verify ensembling: word tables, voting and decoding, CRF marginals, fitting and predicting."""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


def table(doc_id: str, tags: list[str | None], probabilities: list[list[float]] | None = None) -> pd.DataFrame:
    """A word table over word ids 0..n-1; a `None` tag leaves that word unlabelled."""
    from lab.ner.ensemble.tables import WORD_TABLE_COLUMNS

    rows = []

    for word_id, tag in enumerate(tags):
        if tag is None:
            continue

        row_probabilities = (
            probabilities[word_id]
            if probabilities is not None
            else [0.8 if candidate == tag else 0.1 for candidate in ("O", "B", "I")]
        )
        rows.append([doc_id, word_id, tag, *row_probabilities])

    return pd.DataFrame(rows, columns=WORD_TABLE_COLUMNS)


def decode(tables, text, weights=None, threshold=0.5, votes="hard"):
    from lab.ner.ensemble.voting import align_votes, ensemble_spans, normalize_weights

    word_votes = align_votes(tables)
    weights = weights or normalize_weights([1.0] * len(tables))

    return ensemble_spans(
        word_votes,
        weights=weights,
        threshold=threshold,
        votes=votes,
        texts={"d": text},
        target_label="DISEASE",
    )


def offsets(spans: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(spans["start_span"].astype(int), spans["end_span"].astype(int)))


def verify_voting(checks: Checks) -> None:
    from lab.ner.ensemble.voting import align_votes, combine_votes, normalize_weights

    text = "w0 w1 w2 w3 w4 w5"
    word_extent = [(index * 3, index * 3 + 2) for index in range(6)]

    aligned = align_votes([table("d", ["O", "B", None]), table("d", [None, "B", "I", "O"])])
    checks.equal("alignment takes the union of labelled words", len(aligned.words), 4)
    checks.equal(
        "and records who labelled what",
        aligned.present.tolist(),
        [[True, False], [True, True], [False, True], [False, True]],
    )

    majority = decode([table("d", ["B", "O"]), table("d", ["B", "B"]), table("d", ["O", "O"])], text)
    checks.equal("two of three members carry a word into an entity", offsets(majority), [word_extent[0]])

    tie = decode([table("d", ["B"]), table("d", ["O"])], text)
    checks.equal("a one-one tie is not a majority", len(tie), 0)

    missing = decode([table("d", ["B", "I"]), table("d", ["B", None])], text)
    checks.equal(
        "a member that did not label a word is left out of its vote",
        offsets(missing),
        [(word_extent[0][0], word_extent[1][1])],
    )

    orphan = decode([table("d", ["O", "I", "I"])], text)
    checks.equal("an orphan I begins an entity", offsets(orphan), [(word_extent[1][0], word_extent[2][1])])

    adjacent = decode([table("d", ["B", "B"])], text)
    checks.equal("a B after an entity word begins a new one", offsets(adjacent), [word_extent[0], word_extent[1]])

    split = decode([table("d", ["B", "B"]), table("d", ["B", "I"])], text)
    checks.equal(
        "a begin/inside tie continues the entity",
        offsets(split),
        [(word_extent[0][0], word_extent[1][1])],
    )

    gap = decode([table("d", ["B", None, "I"])], text)
    checks.equal(
        "a word nobody labelled is passed over, and the span covers it",
        offsets(gap),
        [(word_extent[0][0], word_extent[2][1])],
    )
    checks.equal("span text is the document's", gap["text"].tolist(), ["w0 w1 w2"])

    scored = decode(
        [table("d", ["B", "I"], [[0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])],
        text,
    )
    checks.check(
        "a span's score is the mean probability of the tags its words took",
        abs(float(scored["score"].iloc[0]) - 0.65) < 1e-9,
    )

    soft_tables = [
        table("d", ["O"], [[0.6, 0.4, 0.0]]),
        table("d", ["O"], [[0.6, 0.4, 0.0]]),
        table("d", ["B"], [[0.0, 1.0, 0.0]]),
    ]
    checks.equal("hard votes: one of three says entity, so no entity", len(decode(soft_tables, text)), 0)
    checks.equal(
        "soft votes: mean entity probability 0.6 clears 0.5",
        len(decode(soft_tables, text, votes="soft")),
        1,
    )
    checks.equal(
        "a weight can carry a lone member past the threshold",
        len(decode(soft_tables, text, weights=[0.2, 0.2, 0.6])),
        1,
    )
    checks.equal(
        "and a threshold above the share keeps it out",
        len(decode(soft_tables, text, weights=[0.2, 0.2, 0.6], threshold=0.6)),
        0,
    )

    combined = combine_votes(align_votes(soft_tables), weights=[1 / 3] * 3, votes="soft")
    checks.check(
        "combined tag probabilities stay a distribution",
        np.allclose(combined.probabilities.sum(axis=1), 1.0),
    )

    checks.equal("weights are normalised", normalize_weights([1, 3]), [0.25, 0.75])
    checks.raises("all-zero weights raise", ValueError, normalize_weights, [0, 0])
    checks.raises("negative weights raise", ValueError, normalize_weights, [1, -1])
    checks.raises(
        "the wrong number of weights raises",
        ValueError,
        combine_votes,
        align_votes(soft_tables),
        weights=[0.5, 0.5],
        votes="soft",
    )
    checks.raises(
        "an unknown kind of vote raises",
        ValueError,
        combine_votes,
        align_votes(soft_tables),
        weights=[1 / 3] * 3,
        votes="mean",
    )


def verify_single_member_equivalence(checks: Checks) -> None:
    from fixtures import synthetic_corpus
    from transformers import AutoTokenizer

    from lab.ner.encoding import Encoder
    from lab.ner.ensemble.tables import word_table
    from lab.ner.ensemble.voting import align_votes, ensemble_spans
    from lab.ner.evaluation.spans import softmax
    from lab.ner.inference import decode_spans

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(tokenizer=tokenizer, target_label="DISEASE", language="es", max_length=256)
    corpus = synthetic_corpus(12)
    rows = encoder.encode(corpus)
    texts = dict(zip(corpus["doc_id"], corpus["text"]))

    checks.equal("each document fits one window, so decoding scopes agree", len(rows), len(corpus))

    width = max(len(row) for row in rows["input_ids"])
    logits = np.random.default_rng(7).normal(size=(len(rows), width, 3)).astype(np.float32)

    def compare(name: str, logits: np.ndarray) -> None:
        single = decode_spans(
            rows=rows,
            predictions=logits,
            tokenizer=tokenizer,
            id2label=encoder.id2label,
            texts=texts,
        )
        member = word_table(
            rows=rows,
            probabilities=softmax(logits),
            tag_ids=logits.argmax(axis=-1),
            tokenizer=tokenizer,
            id2label=encoder.id2label,
        )
        ensembled = ensemble_spans(
            align_votes([member]),
            weights=[1.0],
            threshold=0.5,
            votes="hard",
            texts=texts,
            target_label="DISEASE",
        )

        checks.check(f"{name}: there are spans to compare", len(single) > 10)
        checks.frames_equal(
            f"{name}: a one-member hard vote decodes predict_entities' spans",
            ensembled.drop(columns="score"),
            single.drop(columns="score"),
        )

        if name == "no I tags":
            checks.check(
                f"{name}: and their scores",
                np.allclose(ensembled["score"].to_numpy(), single["score"].to_numpy()),
            )

    compare("random tags", logits)

    without_inside = logits.copy()
    without_inside[:, :, encoder.label2id["I-DISEASE"]] = -10.0
    compare("no I tags", without_inside)


def brute_force_marginals(crf, emissions, length: int) -> np.ndarray:
    import torch

    start = crf.start_transitions.detach().numpy()
    end = crf.end_transitions.detach().numpy()
    transitions = crf.transitions.detach().numpy()
    scores = emissions[:length]
    marginals = np.zeros((length, scores.shape[1]))
    total = 0.0

    for path in itertools.product(range(scores.shape[1]), repeat=length):
        score = start[path[0]] + end[path[-1]] + sum(scores[t, tag] for t, tag in enumerate(path))
        score += sum(transitions[path[t - 1], path[t]] for t in range(1, length))
        weight = float(torch.tensor(score, dtype=torch.float64).exp())
        total += weight

        for t, tag in enumerate(path):
            marginals[t, tag] += weight

    return marginals / total


def verify_crf_marginals(checks: Checks) -> None:
    import torch
    from torchcrf import CRF

    from lab.ner.models.crf import CRFForTokenClassification

    torch.manual_seed(0)

    class Stub:
        constrain_transitions = False
        crf = CRF(num_tags=3, batch_first=True)

    for parameter in (Stub.crf.start_transitions, Stub.crf.end_transitions, Stub.crf.transitions):
        torch.nn.init.normal_(parameter)

    emissions = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    log_marginals = CRFForTokenClassification.marginals_from_emissions(Stub(), emissions, mask)

    checks.equal("marginals keep the emissions' shape", tuple(log_marginals.shape), (2, 5, 3))
    checks.equal("and are float32", log_marginals.dtype, torch.float32)

    for row, length in enumerate((5, 3)):
        expected = brute_force_marginals(Stub.crf, emissions[row].numpy(), length)
        actual = log_marginals[row, :length].detach().exp().numpy()

        checks.check(
            f"forward-backward matches enumerating every path (length {length})",
            np.allclose(actual, expected, atol=1e-5),
        )

    half = CRFForTokenClassification.marginals_from_emissions(Stub(), emissions.half(), mask)

    checks.equal("fp16 emissions still give float32 marginals", half.dtype, torch.float32)


def verify_io(checks: Checks) -> None:
    from lab.ner.ensemble.io import (
        describe_member,
        read_ensemble,
        shared_target_label,
        weights_sha256,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "model"
        model_dir.mkdir()

        checks.raises("a directory with no weights has no digest", FileNotFoundError, weights_sha256, model_dir)

        (model_dir / "model.safetensors").write_bytes(b"one")
        first = weights_sha256(model_dir)
        (model_dir / "model.safetensors").write_bytes(b"two")

        checks.check("the digest follows the weights", weights_sha256(model_dir) != first)

        (model_dir / "encoding.json").write_text(
            json.dumps({"model": {}, "encoding": {"target_label": "DISEASE", "strategy": "my_windows"}})
        )
        checks.raises(
            "a member trained with a callable strategy is refused",
            ValueError,
            describe_member,
            model_dir,
            match="callable window strategy",
        )

        checks.raises(
            "members predicting different labels are refused",
            ValueError,
            shared_target_label,
            [
                {"model_dir": "a", "encoding": {"target_label": "DISEASE"}},
                {"model_dir": "b", "encoding": {"target_label": "SYMPTOM"}},
            ],
            match="same target_label",
        )

        checks.raises("a directory with no ensemble.json raises", FileNotFoundError, read_ensemble, root)

        (root / "ensemble.json").write_text(json.dumps({"format_version": 99}))
        checks.raises("an unknown format version raises", ValueError, read_ensemble, root, match="format_version")


def verify_arguments(checks: Checks) -> None:
    from lab.ner.ensemble import fit_ensemble

    with tempfile.TemporaryDirectory() as tmp:
        checks.raises(
            "an unknown method raises",
            ValueError,
            fit_ensemble,
            ["m"],
            method="median",
            output_dir=tmp,
        )
        checks.raises(
            "votes with a method that fixes its own raises",
            ValueError,
            fit_ensemble,
            ["m"],
            method="majority",
            output_dir=tmp,
            votes="soft",
            match="votes only applies",
        )
        checks.raises(
            "tune_threshold outside soft raises",
            ValueError,
            fit_ensemble,
            ["m"],
            method="weighted",
            output_dir=tmp,
            tune_threshold=True,
            match="tune_threshold",
        )
        checks.raises(
            "a tuned ensemble with no validation raises",
            ValueError,
            fit_ensemble,
            ["m"],
            method="weighted",
            output_dir=tmp,
            match="validation=",
        )
        checks.raises(
            "soft with a tuned threshold needs validation too",
            ValueError,
            fit_ensemble,
            ["m"],
            method="soft",
            output_dir=tmp,
            tune_threshold=True,
            match="validation=",
        )


def verify_task_registration(checks: Checks) -> None:
    import lab.ner
    from lab.core.tasks import resolve_task
    from lab.ner.ensemble import fit_ensemble, predict_ensemble

    checks.check("ner.fit_ensemble resolves", resolve_task("ner.fit_ensemble") is fit_ensemble)
    checks.check("ner.predict_ensemble resolves", resolve_task("ner.predict_ensemble") is predict_ensemble)
    checks.check("both are on lab.ner", lab.ner.fit_ensemble is fit_ensemble and lab.ner.predict_ensemble is predict_ensemble)


def verify_end_to_end(checks: Checks) -> None:
    from fixtures import synthetic_corpus
    from verify_inference import train_saved_model

    from lab.core.spans import SCORED_SPAN_COLUMNS
    from lab.ner.ensemble import fit_ensemble, predict_ensemble
    from lab.ner.inference import predict_entities

    shared = tempfile.TemporaryDirectory()
    root = Path(shared.name)

    linear_dir = train_saved_model(root / "linear", "linear")
    crf_dir = train_saved_model(root / "crf", "crf")
    validation = synthetic_corpus(8)
    documents = synthetic_corpus(6)

    single = predict_entities(
        model_dir=linear_dir, documents=documents, output_dir=root / "single", device="cpu", batch_size=4
    )

    checks.check("predict_entities writes parquet predictions too", single.paths["predictions_parquet"].exists())
    checks.frames_equal(
        "which hold the same spans as the TSV",
        pd.read_parquet(single.paths["predictions_parquet"]),
        single.spans[SCORED_SPAN_COLUMNS],
    )

    for name, model_dir in (("linear", linear_dir), ("crf", crf_dir)):
        reference = predict_entities(
            model_dir=model_dir, documents=documents, output_dir=root / f"{name}_single", device="cpu", batch_size=4
        )
        fitted = fit_ensemble([model_dir], method="majority", output_dir=root / f"{name}_alone")
        alone = predict_ensemble(
            ensemble_dir=fitted.output_dir,
            documents=documents,
            output_dir=root / f"{name}_alone_predictions",
            device="cpu",
            batch_size=4,
        )

        checks.equal(
            f"a one-member {name} ensemble predicts what predict_entities does",
            offsets(alone.spans),
            offsets(reference.spans),
        )
        checks.equal(
            f"and the {name} run scores the same strict F1",
            alone.metrics["span_strict_f1"],
            reference.metrics["span_strict_f1"],
        )

    untuned = fit_ensemble([linear_dir, crf_dir], method="soft", output_dir=root / "soft")

    checks.equal("an untuned ensemble needs no validation", untuned.metrics, None)
    checks.equal("its weights are equal", [m["weight"] for m in untuned.ensemble["members"]], [0.5, 0.5])
    checks.equal("its threshold is the default", untuned.ensemble["threshold"], 0.5)
    checks.equal("it records soft votes", untuned.ensemble["votes"], "soft")
    checks.check("and writes no trials", "trials" not in untuned.paths)

    weighted = fit_ensemble(
        [linear_dir, crf_dir],
        method="weighted",
        output_dir=root / "weighted",
        validation=validation,
        n_trials=6,
        random_state=3,
        device="cpu",
        batch_size=4,
    )
    weights = [member["weight"] for member in weighted.ensemble["members"]]

    checks.check("tuned weights sum to one", abs(sum(weights) - 1) < 1e-9)
    checks.check("the threshold is within its bounds", 0.05 <= weighted.ensemble["threshold"] <= 0.95)
    checks.equal("every trial is recorded", len(pd.read_parquet(weighted.paths["trials"])), 6)
    checks.equal("each member is scored alone", len(weighted.metrics["members"]), 2)
    checks.check("and the ensemble beside them", "span_strict_f1" in weighted.metrics["ensemble"])
    checks.equal("the manifest records the search", weighted.manifest["search"]["n_trials"], 6)
    checks.equal("and the validation documents", weighted.manifest["validation"]["n_documents"], 8)
    checks.check("and each member's encoding", all("encoding" in m for m in weighted.manifest["members"]))
    checks.check("the summary names the best member", "best member" in weighted.summary())

    again = fit_ensemble(
        [linear_dir, crf_dir],
        method="weighted",
        output_dir=root / "weighted_again",
        validation=validation,
        n_trials=6,
        random_state=3,
        device="cpu",
        batch_size=4,
    )
    checks.equal("a seeded search is reproducible", again.ensemble["members"], weighted.ensemble["members"])

    tuned_soft = fit_ensemble(
        [linear_dir, crf_dir],
        method="soft",
        output_dir=root / "tuned_soft",
        validation=validation,
        tune_threshold=True,
        n_trials=4,
        device="cpu",
        batch_size=4,
    )
    checks.equal("soft with tune_threshold keeps equal weights", [m["weight"] for m in tuned_soft.ensemble["members"]], [0.5, 0.5])
    checks.equal("and searches the threshold alone", tuned_soft.manifest["search"]["tuned"], ["threshold"])

    checks.raises(
        "an unknown metric raises before searching",
        ValueError,
        fit_ensemble,
        [linear_dir, crf_dir],
        method="weighted",
        output_dir=root / "bad_metric",
        validation=validation,
        metric="span_strict_f2",
        device="cpu",
        match="span_strict_f1",
    )

    predicted = predict_ensemble(
        ensemble_dir=weighted.output_dir,
        documents=documents,
        output_dir=root / "weighted_predictions",
        device="cpu",
        batch_size=4,
    )

    checks.check("ensemble predictions are written as TSV", predicted.paths["predictions"].exists())
    checks.check("and as parquet", predicted.paths["predictions_parquet"].exists())
    checks.check("gold input is scored", predicted.metrics is not None and "char_f1" in predicted.metrics)
    checks.equal("the manifest carries the ensemble", predicted.manifest["ensemble"], weighted.ensemble)
    checks.equal("predictions are the span table plus a score", list(predicted.spans.columns), SCORED_SPAN_COLUMNS)

    moved = root / "linear_copy"
    shutil.copytree(linear_dir, moved)
    copied = fit_ensemble([moved, crf_dir], method="majority", output_dir=root / "copied")
    (moved / "model.safetensors").write_bytes((moved / "model.safetensors").read_bytes() + b"\0")

    checks.raises(
        "a member retrained in place is refused",
        ValueError,
        predict_ensemble,
        ensemble_dir=copied.output_dir,
        documents=documents,
        output_dir=root / "stale",
        match="changed since",
    )

    shared.cleanup()


def main() -> int:
    checks = Checks("ensemble")

    if not torch_available():
        checks.skip("ensemble", "torch is not installed in this environment")

        return checks.report()

    verify_voting(checks)
    verify_single_member_equivalence(checks)
    verify_crf_marginals(checks)
    verify_io(checks)
    verify_arguments(checks)
    verify_task_registration(checks)
    verify_end_to_end(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
