"""Heterogeneous catalogue, coverage, tuning and artifact checks with local Transformers."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from _harness import Checks, run
from fixtures import local_tokenizer, tiny_base_model

from lab.core import build_corpus
from lab.core.spans import span_dataframe
from lab.ner import Encoder, build_model, fit_ensemble, predict_ensemble
from lab.ner.ensemble.entities import (
    VotingParameters,
    align_entities,
    decode_entities,
    score_ensemble,
)
from lab.ner.ensemble.io import read_ensemble
from lab.ner.ensemble.search import search_entity_parameters
from lab.ner.ensemble.spans import combine_spans
from lab.ner.ensemble.tables import MemberPredictions, predict_members
from lab.ner.ensemble.voting import align_votes, normalize_weights
from lab.ner.training import model_encoding, write_model_encoding

TEXT = "dolor abdominal intenso"
LABELS = ["DISEASE", "SYMPTOM", "PROCEDURE"]


def evidence(tags_by_label, supported=None):
    from verify_ensemble import table

    frames, spans = [], []
    for label, tags in tags_by_label.items():
        frame = table("d", tags)
        frame["entity_label"] = label
        frames.append(frame)
        from lab.ner.ensemble.voting import ensemble_spans

        spans.extend(
            ensemble_spans(align_votes([frame]), [1], 0.5, "hard", {"d": TEXT}, label).to_dict(
                "records"
            )
        )
    columns = ["doc_id", "word_id", "tag", "p_O", "p_B", "p_I", "entity_label"]
    words = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return MemberPredictions(
        words, span_dataframe(spans, scored=True), tuple(supported or tags_by_label)
    )


def verify_coverage(checks):
    a = evidence({"DISEASE": ["O", "O", "O"]})
    b = evidence({"SYMPTOM": ["O", "B", "I"]})
    c = evidence(
        {"PROCEDURE": ["O", "O", "B"], "SYMPTOM": ["O", "B", "I"], "DISEASE": ["B", "I", "O"]}
    )
    for name, members, expected in [
        ("A", [a, a, c], [(0, 1, 2), (2,)]),
        ("B", [a, b, c], [(0, 2), (1, 2), (2,)]),
        (
            "C",
            [
                evidence({"DISEASE": ["B"], "SYMPTOM": ["B"]}),
                evidence({"SYMPTOM": ["B"], "PROCEDURE": ["B"]}),
                evidence({"DISEASE": ["B"], "PROCEDURE": ["B"]}),
            ],
            [(0, 2), (0, 1), (1, 2)],
        ),
    ]:
        labels = ["DISEASE", "SYMPTOM"] if name == "A" else LABELS
        aligned = align_entities(members, labels)
        checks.equal(
            f"case {name} eligible columns", [v.member_indices for v in aligned.values()], expected
        )
    aligned = align_entities([a, b, c], LABELS)
    spans = decode_entities(aligned, VotingParameters([1, 1, 1]), "hard", {"d": TEXT})
    checks.check("O on unknown label does not vote", "SYMPTOM" in set(spans.label))
    checks.check("single eligible expert survives", "PROCEDURE" in set(spans.label))
    symptom = spans[spans.label == "SYMPTOM"].iloc[0]
    checks.equal("offsets use shared UAX word ids", (symptom.start_span, symptom.end_span), (6, 23))
    overlap = decode_entities(
        align_entities([c], LABELS), VotingParameters([1]), "hard", {"d": TEXT}
    )
    checks.equal("overlapping types retained", len(overlap), 3)
    reversed_c = MemberPredictions(c.words.iloc[::-1], c.spans, tuple(reversed(c.entity_labels)))
    reordered = decode_entities(
        align_entities([c, reversed_c], LABELS), VotingParameters([1, 1]), "soft", {"d": TEXT}
    )
    checks.frames_equal("catalogue order is not head identity", reordered, overlap)
    gap = evidence({"DISEASE": ["B", None, "I"]})
    aligned_gap = align_entities([gap], ["DISEASE"])
    checks.equal(
        "new default breaks uncovered gaps",
        len(decode_entities(aligned_gap, VotingParameters([1]), "hard", {"d": TEXT})),
        2,
    )
    checks.equal(
        "legacy skip policy still available",
        len(decode_entities(aligned_gap, VotingParameters([1]), "hard", {"d": TEXT}, "skip")),
        1,
    )
    zero = decode_entities(aligned, VotingParameters([0, 0, 0]), "soft", {"d": TEXT})
    checks.equal("zero evidence never fabricates confidence", len(zero), 0)
    checks.equal(
        "threshold one excludes everything",
        len(decode_entities(aligned, VotingParameters([1, 1, 1], 1), "soft", {"d": TEXT})),
        0,
    )
    checks.raises(
        "unsupported entity cannot be discovered from gold",
        ValueError,
        align_entities,
        [a],
        ["UNSEEN"],
    )
    invalid = c.words.copy()
    invalid.loc[0, "p_B"] = np.nan
    checks.raises("NaN probabilities rejected", ValueError, align_votes, [invalid])
    checks.raises("nonfinite weights rejected", ValueError, normalize_weights, [float("inf")])
    specialists = [
        evidence({"DISEASE": ["B"], "SYMPTOM": ["O"]}),
        evidence({"DISEASE": ["O"], "SYMPTOM": ["B"]}),
    ]
    scoped = VotingParameters([1, 1], entity_weights={"DISEASE": [1, 0], "SYMPTOM": [0, 1]})
    specialist_votes = align_entities(specialists, ["DISEASE", "SYMPTOM"])
    checks.equal(
        "independent per-entity weights select different specialists",
        len(decode_entities(specialist_votes, scoped, "hard", {"d": TEXT})),
        2,
    )
    thresholds = VotingParameters(
        [1, 1], entity_weights=scoped.entity_weights, entity_thresholds={"SYMPTOM": 1}
    )
    checks.equal(
        "per-entity threshold does not affect another type",
        set(decode_entities(specialist_votes, thresholds, "hard", {"d": TEXT}).label),
        {"DISEASE"},
    )
    return [a, b, c]


def verify_span_methods(checks):
    a = evidence({"DISEASE": ["B", "I", "O"]})
    b = evidence({"DISEASE": ["O", "B", "I"], "SYMPTOM": ["O", "B", "I"]})
    for method, count in [("span_union", 3), ("span_consensus", 1), ("span_intersection", 1)]:
        spans = combine_spans([a, b], ["DISEASE", "SYMPTOM"], method, {"d": TEXT})
        checks.equal(f"{method} exact limits, type-specific denominator", len(spans), count)
    union = combine_spans([a, b], LABELS, "span_union", {"d": TEXT})
    checks.equal(
        "same-label overlapping alternatives survive union", len(union[union.label == "DISEASE"]), 2
    )
    checks.equal(
        "explicit support one retains alternatives",
        len(combine_spans([a, b], LABELS, "span_consensus", {"d": TEXT}, min_support=1)),
        3,
    )
    absent = evidence({}, supported=["DISEASE"])
    checks.equal(
        "masked members do not veto span consensus",
        len(combine_spans([a, absent], ["DISEASE"], "span_intersection", {"d": TEXT})),
        1,
    )
    duplicate = MemberPredictions(a.words, pd.concat([a.spans, a.spans]), a.entity_labels)
    checks.equal(
        "duplicate mentions do not inflate support",
        len(combine_spans([duplicate, b], ["DISEASE"], "span_intersection", {"d": TEXT})),
        0,
    )


def verify_crf_decisions(checks):
    from types import SimpleNamespace

    from torchcrf import CRF

    from lab.ner.models.crf import CRFForTokenClassification

    torch.manual_seed(17)
    decoder = SimpleNamespace(crf=CRF(3, batch_first=True), constrain_transitions=False)
    found_difference = False
    for _ in range(20):
        emissions = torch.randn(2, 5, 3)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        log_probabilities = CRFForTokenClassification.marginals_from_emissions(
            decoder, emissions, mask
        )
        checks.check(
            "CRF valid-position marginal distributions",
            torch.allclose(
                log_probabilities.exp().sum(-1)[mask], torch.ones(mask.sum()), atol=1e-5
            ),
        )
        checks.check(
            "CRF valid-position finite", torch.isfinite(log_probabilities[mask]).all().item()
        )
        paths = decoder.crf.decode(emissions, mask)
        found_difference |= any(
            path != log_probabilities[index, : len(path)].argmax(-1).tolist()
            for index, path in enumerate(paths)
        )
    checks.check("Viterbi legitimately differs from marginal argmax", found_difference)


def verify_tuning(checks, predictions):
    gold = predictions[-1].spans
    aligned = align_entities(predictions, LABELS)
    metric = score_ensemble(gold, predictions[0].spans, LABELS)
    checks.equal("full-task specialist misses uncovered types", metric["span_strict_macro_f1"], 0.0)
    manual = score_ensemble(gold, gold[gold.label == "DISEASE"], LABELS)
    checks.equal("strict macro from per-label components", manual["span_strict_macro_f1"], 1 / 3)
    checks.equal("strict micro from pooled counts", manual["span_strict_micro_f1"], 0.5)
    args = dict(
        aligned=aligned,
        gold=gold,
        texts={"d": TEXT},
        n_members=3,
        votes="soft",
        tune_weights=True,
        weight_scope="entity",
        threshold_scope="entity",
        n_trials=4,
        random_state=7,
        min_gold_entities=1,
    )
    result = search_entity_parameters(**args)
    again = search_entity_parameters(**args)
    checks.equal("seeded scoped tuning reproducible", result.parameters, again.parameters)
    checks.check(
        "equal-weight baseline cannot be silently degraded", result.score >= result.baseline_score
    )
    for raw in result.trials_df.parameters_json:
        params = json.loads(raw)
        checks.check(
            "singleton label has no weight degree of freedom",
            not any(key.startswith("entity_2_weight") for key in params),
        )
    fallback = search_entity_parameters(**{**args, "min_gold_entities": 100})
    checks.equal("insufficient gold uses deterministic default", fallback.parameters.threshold, 0.5)
    checks.equal("no useless Optuna trials without supervision", len(fallback.trials_df), 0)
    for metric in ["span_strict_macro_f1", "span_strict_micro_f1", "span_strict_f1__SYMPTOM"]:
        chosen = search_entity_parameters(**{**args, "metric": metric, "n_trials": 1})
        checks.check("configurable objective " + metric, np.isfinite(chosen.score))


def corpus_fixture():
    texts, annotations = {}, []
    for index in range(3):
        name = f"validation_{index}"
        texts[name] = "Paciente con dolor abdominal intenso."
        for label, mention in [
            ("DISEASE", "dolor abdominal"),
            ("SYMPTOM", "abdominal intenso"),
            ("PROCEDURE", "intenso"),
        ]:
            start = texts[name].index(mention)
            annotations.append(
                dict(
                    filename=name,
                    label=label,
                    start_span=start,
                    end_span=start + len(mention),
                    text=mention,
                )
            )
    return build_corpus(texts, pd.DataFrame(annotations))


def save_member(root, base, tokenizer, architecture, labels, outside=False):
    directory = root / "best_model"
    directory.mkdir(parents=True)
    multihead = architecture == "multihead"
    encoder = Encoder(
        tokenizer,
        None if multihead else labels[0],
        "es",
        max_length=32,
        entity_labels=list(labels) if multihead else None,
    )
    kwargs = {"entity_labels": labels, "dropout": 0.0} if multihead else ({"crf_version": 1} if architecture == "crf" else {})
    model = build_model(str(base), encoder.label2id, encoder.id2label, architecture, **kwargs)
    with torch.no_grad():
        model.classifier.weight.zero_()
        biases = [
            [8.0, -8.0, -8.0]
            if outside is True or isinstance(outside, set) and label in outside
            else [-8.0, -8.0, 8.0]
            for label in (labels if multihead else labels[:1])
        ]
        model.classifier.bias.copy_(torch.tensor(biases).reshape(-1))
    if architecture == "linear":
        model.save_pretrained(directory)
    else:
        from safetensors.torch import save_file

        save_file(model.state_dict(), str(directory / "model.safetensors"))
    description = model_encoding(encoder, str(base), architecture, kwargs)
    if multihead:
        description["backbone_config"] = model.config.to_dict()
    write_model_encoding(description, directory)
    tokenizer.save_pretrained(directory)
    return directory


def make_checkpoints(root):
    tokenizer = local_tokenizer()
    base = tiny_base_model(root, tokenizer)
    specs = [
        ("linear", ["DISEASE"], False),
        ("crf", ["DISEASE"], False),
        ("linear", ["SYMPTOM"], False),
        ("multihead", ["SYMPTOM", "DISEASE"], False),
        ("multihead", ["PROCEDURE", "SYMPTOM"], False),
        ("multihead", ["DISEASE", "PROCEDURE"], False),
        ("multihead", ["PROCEDURE", "DISEASE", "SYMPTOM"], False),
        ("linear", ["DISEASE"], True),
        ("multihead", ["SYMPTOM"], False),
    ]
    paths = [
        save_member(root / f"member_{index}", base, tokenizer, architecture, labels, outside)
        for index, (architecture, labels, outside) in enumerate(specs)
    ]
    from transformers import BertTokenizer

    words = [
        word
        for word, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
        if word not in ("intens", "##o")
    ]
    words.append("intenso")
    alternate = BertTokenizer(vocab={word: index for index, word in enumerate(words)})
    alternate_base = tiny_base_model(root / "alternate", alternate)
    paths.append(
        save_member(
            root / "reordered",
            alternate_base,
            alternate,
            "multihead",
            ["DISEASE", "SYMPTOM"],
            {"DISEASE"},
        )
    )
    return paths


def verify_checkpoints(checks, root):
    import lab.ner.ensemble.tables as adapters
    from lab.ner.inference.loading import load_model

    paths = make_checkpoints(root)
    corpus = corpus_fixture()
    calls, loads = [], []

    def tracked(path, **kwargs):
        model, tokenizer, description = load_model(path, **kwargs)
        loads.append(str(path))
        backbone = model.backbone if hasattr(model, "backbone") else model.base_model
        backbone.register_forward_hook(lambda *args: calls.append(str(path)))
        return model, tokenizer, description

    hybrid = [paths[0], paths[1], paths[6]]
    with patch.object(adapters, "load_model", side_effect=tracked):
        fitted = fit_ensemble(
            hybrid,
            "weighted",
            root / "fitted",
            validation=corpus,
            weight_scope="entity",
            threshold_scope="entity",
            n_trials=3,
            min_gold_entities=1,
            random_state=13,
            batch_size=2,
            device="cpu",
        )
    checks.equal("one load per member across all methods and Optuna trials", len(loads), 3)
    checks.equal("one Transformer forward per member batch, not per label", len(calls), 6)
    checks.equal("metadata defines ordered union", fitted.ensemble["entity_labels"], sorted(LABELS))
    checks.equal(
        "multientity default is strict macro", fitted.ensemble["metric"], "span_strict_macro_f1"
    )
    checks.equal("format two persisted", read_ensemble(fitted.output_dir), fitted.ensemble)
    artifact_path = fitted.output_dir / "ensemble.json"
    artifact_text = artifact_path.read_text()
    broken = json.loads(artifact_text)
    broken["entity_weights"]["DISEASE"] = [1.0]
    artifact_path.write_text(json.dumps(broken))
    checks.raises(
        "persisted weight dimensions checked",
        ValueError,
        read_ensemble,
        fitted.output_dir,
        match="eligible members",
    )
    broken = json.loads(artifact_text)
    broken["members"][0].pop("settings_sha256")
    artifact_path.write_text(json.dumps(broken))
    checks.raises(
        "format two cannot omit metadata digest",
        ValueError,
        read_ensemble,
        fitted.output_dir,
        match="digests",
    )
    broken = json.loads(artifact_text)
    broken["entity_thresholds"]["DISEASE"] = float("nan")
    artifact_path.write_text(json.dumps(broken))
    checks.raises(
        "invalid persisted threshold rejected",
        ValueError,
        read_ensemble,
        fitted.output_dir,
        match="thresholds",
    )
    artifact_path.write_text(artifact_text)

    checks.equal("all individuals compared on full task", len(fitted.metrics["members"]), 3)
    checks.check("specialist scope distinguished", "specialists" in fitted.metrics)
    checks.equal(
        "global best reported with objective",
        fitted.metrics["best_member_global"]["metric"],
        "span_strict_macro_f1",
    )
    checks.check(
        "weighted scope baselines present",
        "weighted_entity_weights_entity_thresholds" in fitted.metrics["baselines"],
    )
    for name, selected in [
        ("A", [paths[0], paths[1], paths[3]]),
        ("B", [paths[1], paths[2], paths[6]]),
        ("C", [paths[3], paths[4], paths[5]]),
        ("D", [paths[7], paths[8]]),
        ("ORDER", [paths[3], paths[9]]),
    ]:
        fit = fit_ensemble(selected, "majority", root / name)
        output = predict_ensemble(
            fit.output_dir, corpus, root / f"{name}_pred", device="cpu", batch_size=2
        )
        expected = {"SYMPTOM"} if name in ("D", "ORDER") else set(fit.ensemble["entity_labels"])
        checks.equal("real checkpoint case " + name, set(output.spans.label), expected)
        checks.check(
            "canonical span columns",
            set(["filename", "label", "start_span", "end_span", "text", "score"]).issubset(
                output.spans.columns
            ),
        )
    result = predict_ensemble(fitted.output_dir, corpus, root / "pred", device="cpu", batch_size=2)
    again = predict_ensemble(
        fitted.output_dir, corpus, root / "pred_again", device="cpu", batch_size=2
    )
    checks.frames_equal("fitted ensemble roundtrip deterministic", result.spans, again.spans)
    checks.frames_equal(
        "Parquet artifact", pd.read_parquet(result.paths["predictions_parquet"]), result.spans
    )
    tsv = pd.read_csv(result.paths["predictions"], sep="\t", keep_default_na=False)
    checks.equal("TSV identity", tsv.label.tolist(), result.spans.label.tolist())
    checks.check(
        "gold scoring includes per-label counts", "span_strict_fn__SYMPTOM" in result.metrics
    )
    for method in ["soft", "best", "span_union", "span_consensus", "span_intersection"]:
        fit = fit_ensemble(
            hybrid,
            method,
            root / method,
            validation=corpus,
            compare_methods=False,
            min_gold_entities=1,
            device="cpu",
            batch_size=2,
        )
        predicted = predict_ensemble(
            fit.output_dir, corpus, root / f"{method}_pred", device="cpu", batch_size=2
        )
        checks.check("end-to-end method " + method, predicted.metrics is not None)
    from lab.core.spans import spans_from_corpus
    from lab.ner.ensemble.search import search_ensemble

    classic_evidence = predict_members(paths[:2], corpus, device="cpu", use_annotations=True)
    classic_votes = align_entities(classic_evidence, ["DISEASE"])["DISEASE"].votes
    classic_gold = spans_from_corpus(corpus, label="DISEASE")
    texts = dict(zip(corpus.doc_id, corpus.text))
    for method, tune_weights in [("weighted", True), ("soft", False)]:
        historical = search_ensemble(
            classic_votes,
            classic_gold,
            texts,
            "DISEASE",
            "soft",
            tune_weights,
            n_trials=3,
            random_state=19,
        )
        fitted_classic = fit_ensemble(
            paths[:2],
            method,
            root / f"legacy_{method}",
            validation=corpus,
            tune_threshold=not tune_weights,
            n_trials=3,
            random_state=19,
            compare_methods=False,
            device="cpu",
        )
        checks.equal(
            "legacy fixed-seed weights " + method,
            [member["weight"] for member in fitted_classic.ensemble["members"]],
            historical.weights,
        )
        checks.equal(
            "legacy fixed-seed threshold " + method,
            fitted_classic.ensemble["threshold"],
            historical.threshold,
        )
    # A format-1 file must still use the historical decoding and score semantics.
    classic = fit_ensemble(paths[:2], "majority", root / "classic")
    before = predict_ensemble(classic.output_dir, corpus, root / "classic_pred", device="cpu")
    old = {
        key: classic.ensemble[key]
        for key in ["method", "votes", "target_label", "threshold", "members"]
    }
    old["format_version"] = 1
    for member in old["members"]:
        for key in ["settings_sha256", "entity_labels", "architecture"]:
            member.pop(key)
    (classic.output_dir / "ensemble.json").write_text(json.dumps(old))
    restored = predict_ensemble(classic.output_dir, corpus, root / "v1_pred", device="cpu")
    checks.frames_equal("format one predictions and scores unchanged", restored.spans, before.spans)
    checks.equal(
        "format one strict score unchanged",
        restored.metrics["span_strict_f1"],
        before.metrics["span_strict_f1"],
    )
    encoding_path = paths[0] / "encoding.json"
    original = encoding_path.read_text()
    metadata = json.loads(original)
    metadata["encoding"]["max_length"] = 30
    encoding_path.write_text(json.dumps(metadata))
    checks.raises(
        "changed encoding is detected before inference",
        ValueError,
        predict_ensemble,
        fitted.output_dir,
        corpus,
        root / "changed",
        match="changed since",
    )
    encoding_path.write_text(original)
    weight_path = paths[0] / "model.safetensors"
    original = weight_path.read_bytes()
    weight_path.write_bytes(original + b"x")
    checks.raises(
        "changed weights detected",
        ValueError,
        predict_ensemble,
        fitted.output_dir,
        corpus,
        root / "changed_weights",
        match="changed since",
    )
    weight_path.write_bytes(original)
    overlap = root / "overlap" / "best_model"
    shutil.copytree(paths[0], overlap)
    training = root / "train.parquet"
    corpus.to_parquet(training, index=False)
    (overlap.parent / "training_summary.json").write_text(
        json.dumps({"metadata": {"train_parquet": str(training)}})
    )
    checks.raises(
        "known document leakage refused",
        ValueError,
        fit_ensemble,
        [overlap],
        "weighted",
        root / "leaky",
        validation=corpus,
        match="overlaps training",
    )
    renamed = corpus.copy()
    renamed["doc_id"] = "renamed_" + renamed["doc_id"]
    checks.raises(
        "renaming training documents does not hide leakage",
        ValueError,
        fit_ensemble,
        [overlap],
        "weighted",
        root / "renamed_leak",
        validation=renamed,
        match="overlaps training",
    )
    checks.raises(
        "unknown subset labels refused",
        ValueError,
        fit_ensemble,
        paths[:1],
        "soft",
        root / "unknown",
        entity_labels=["UNKNOWN"],
    )
    checks.raises(
        "stacking cannot pretend independent calibration",
        ValueError,
        fit_ensemble,
        paths[:1],
        "stacking",
        root / "stacking",
        validation=corpus,
        match="out-of-fold",
    )
    import yaml

    from lab.cli import main as cli_main

    configs = Path(__file__).resolve().parents[1] / "docs" / "ner" / "configs"
    cases = {
        "classic": paths[:2],
        "multihead": [paths[3], paths[4]],
        "hybrid": hybrid,
        "scoped": hybrid,
        "spans": [paths[1], paths[6]],
    }
    outputs = {}
    for name, selected_paths in cases.items():
        config = yaml.safe_load((configs / f"ensemble-{name}.yaml").read_text())
        config.update(
            model_dirs=[str(path) for path in selected_paths], output_dir=str(root / f"cli_{name}")
        )
        if name == "scoped":
            config.update(validation=str(training), n_trials=2, device="cpu", compare_methods=False)
        config_path = root / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        checks.equal("documented fit YAML " + name, cli_main(["run", str(config_path)]), 0)
        outputs[name] = config["output_dir"]
    inference_config = yaml.safe_load((configs / "ensemble-predict.yaml").read_text())
    inference_config.update(
        ensemble_dir=outputs["hybrid"],
        documents=str(training),
        output_dir=str(root / "cli_prediction"),
        device="cpu",
    )
    inference_path = root / "predict.yaml"
    inference_path.write_text(yaml.safe_dump(inference_config))
    checks.equal("documented predict YAML", cli_main(["run", str(inference_path)]), 0)
    from lab.nel import link_entities

    spans = result.spans
    if len(spans):
        gazetteer = spans[["text", "label"]].drop_duplicates().rename(columns={"text": "term"})
        gazetteer["code"] = [str(index) for index in range(len(gazetteer))]
        linked = link_entities(spans, gazetteer, root / "linked", method="string_match")
        checks.equal("ensemble spans compose with NEL", len(linked.spans), len(spans))
    raw = predict_ensemble(fitted.output_dir, {"raw": TEXT}, root / "raw", device="cpu")
    checks.equal("raw text inference does not need gold", raw.metrics, None)
    hard = fit_ensemble(
        hybrid,
        "weighted",
        root / "hard_weighted",
        validation=corpus,
        votes="hard",
        compare_methods=False,
        n_trials=2,
        device="cpu",
        min_gold_entities=1,
    )
    checks.equal("weighted hard uses persisted Viterbi decisions", hard.ensemble["votes"], "hard")
    return paths, corpus


def verify_all(checks):
    torch.set_num_threads(2)
    predictions = verify_coverage(checks)
    verify_span_methods(checks)
    verify_crf_decisions(checks)
    verify_tuning(checks, predictions)
    with tempfile.TemporaryDirectory() as temporary:
        verify_checkpoints(checks, Path(temporary))


def main():
    checks = Checks("unified ensemble")
    verify_all(checks)
    return checks.report()


if __name__ == "__main__":
    run(main)
