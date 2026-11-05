"""Verify summary evaluation: reading, ROUGE, BERTScore, the judges, judging, scores, the task."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run
from fixtures import synthetic_summaries, tiny_base_model

MULTICLINSUM2_ENV = "MULTICLINSUM2_EVAL_ROOT"
DEFAULT_MULTICLINSUM2 = Path.home() / "bsc" / "multiclinsum2" / "evaluation"
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

ANSWERS = {
    "faithfulness": 'Sure. {"score": 8, "reasoning": "grounded", "flags": [], "confidence": 0.9}',
    "completeness": json.dumps(
        {
            "patient_demographics": 10,
            "clinical_presentation": 8,
            "diagnosis": 6,
            "intervention": 4,
            "outcome": 2,
            "follow_up": 0,
            "reasoning": "partial",
            "flags": ["no follow-up"],
        }
    ),
    "fluency": "no json here at all",
}

PARSE_CASES = {
    "faithfulness": [
        '{"score": 7.5, "reasoning": "ok", "flags": ["a"], "confidence": 0.8}',
        'Here you go:\n{"score": 9, "reasoning": "fine", "confidence": 0.5} and then {"x": 1}',
        '{"score": 6, "reasoning": "eurollm scale", "confidence": 8}',
        '{"score": 6, "reasoning": "x" "broken", "flags": []}',
        '{"score": 12, "reasoning": "out of range"}',
        '{"score": 5, "reasoning": "non-string flags", "flags": [1, 2]}',
        '{"score": "7", "reasoning": "string score"}',
        '{"score": "seven", "reasoning": "unreadable"}',
        '{"reasoning": "no score"}',
        '{"score": 4}',
        '{"score": NaN, "reasoning": "nan"}',
        "nothing at all",
        "",
    ],
    "completeness": [
        ANSWERS["completeness"],
        '{"patient_demographics": 10, "clinical_presentation": 10, "diagnosis": 10, '
        '"intervention": 10, "outcome": 10, "follow_up": 10, "reasoning": "all", "confidence": 3}',
        '{"patient_demographics": 15, "clinical_presentation": 0, "diagnosis": 0, '
        '"intervention": 0, "outcome": 0, "follow_up": 0, "reasoning": "over"}',
        '{"patient_demographics": 5, "reasoning": "missing dimensions"}',
        '{"score": 5, "reasoning": "single score"}',
    ],
}


class CannedJudge:
    """A stand-in judge model: answers each judge with a fixed response, marks LONG prompts cut."""

    description = {"model": "canned", "backend": "canned"}
    max_prompt_tokens = 1000

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls = 0

    def prepare(self, prompt: str) -> tuple[list[int], bool]:
        judge = next(name for name in self.answers if f"evaluating the {name}" in prompt)

        return [list(self.answers).index(judge)], "LONG" in prompt

    def generate(self, prompt_token_ids: list[list[int]]) -> list[str]:
        self.calls += 1
        responses = list(self.answers.values())

        return [responses[token_ids[0]] for token_ids in prompt_token_ids]


def multiclinsum2_root() -> Path | None:
    root = Path(os.environ.get(MULTICLINSUM2_ENV, DEFAULT_MULTICLINSUM2))

    return root if (root / "runner.py").is_file() else None


def verify_reading(checks: Checks, root: Path) -> None:
    from lab.summarization.io import read_summaries

    frame = synthetic_summaries()
    jsonl_path = root / "results.jsonl"
    ed2_rows = frame.rename(columns={"doc_id": "case_id", "text": "full_case"})
    ed2_rows["case_id"] = range(len(ed2_rows))
    ed2_rows.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    frame.to_parquet(root / "summaries.parquet", index=False)

    from_jsonl = read_summaries(jsonl_path)
    checks.equal(
        "a MultiClinSum JSONL is renamed to the summaries columns",
        list(from_jsonl.columns),
        ["doc_id", "language", "text", "reference_summary", "generated_summary"],
    )
    checks.equal("integer case ids are read as strings", from_jsonl["doc_id"].tolist()[:2], ["0", "1"])
    checks.frames_equal("a parquet reads back as the frame", read_summaries(root / "summaries.parquet"), frame)
    checks.equal(
        "a list of files is concatenated",
        len(read_summaries([jsonl_path, root / "summaries.parquet"])),
        2 * len(frame),
    )

    duplicated = pd.concat([frame, frame.iloc[:1]])
    checks.raises("a repeated (doc_id, language) raises", ValueError, read_summaries, duplicated, match="more than once")
    checks.raises(
        "a missing column raises, naming it",
        ValueError,
        read_summaries,
        frame.drop(columns="reference_summary"),
        match="reference_summary",
    )
    checks.raises("a non-text summary raises", ValueError, read_summaries, frame.assign(generated_summary=None), match="non-string")
    checks.raises("an empty table raises", ValueError, read_summaries, frame.iloc[:0], match="empty")
    frame.to_csv(root / "results.csv", index=False)
    checks.raises("an unknown file type raises", ValueError, read_summaries, root / "results.csv", match=".jsonl or .parquet")
    checks.raises("a missing file raises", FileNotFoundError, read_summaries, root / "absent.jsonl")

    failed = frame.copy()
    failed.loc[1, "generated_summary"] = "ERROR: CUDA out of memory"
    checks.raises("a failed generation raises by default", ValueError, read_summaries, failed, match="on_failed='drop'")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dropped = read_summaries(failed, on_failed="drop")

    checks.equal("on_failed='drop' removes the failed row", len(dropped), len(frame) - 1)
    checks.check("and warns with the count", any("Dropped 1 of 9" in str(item.message) for item in caught))
    checks.raises("an unknown policy raises", ValueError, read_summaries, frame, on_failed="skip")


def verify_rouge(checks: Checks) -> None:
    from rouge_score import rouge_scorer

    from lab.summarization.evaluation.rouge import UnicodeTokenizer, score_rouge

    tokenizer = UnicodeTokenizer()
    checks.equal("Greek letters are kept", tokenizer.tokenize("Ασθενής 70 ετών"), ["ασθενής", "70", "ετών"])
    checks.equal(
        "a decomposed accent tokenizes as the composed one",
        tokenizer.tokenize("años"),
        tokenizer.tokenize("años"),
    )

    frame = synthetic_summaries()
    scores = score_rouge(frame["generated_summary"].tolist(), frame["reference_summary"].tolist())
    checks.equal("one row per pair, one column per ROUGE type", scores.shape, (9, 3))
    checks.check("a summary equal to its reference scores 1", (scores.loc[[0, 3, 6]] == 1.0).all().all())
    checks.check("Greek is scored on its words, not its digits", 0.0 < scores.loc[4, "rouge1"] < 1.0)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=False, tokenizer=UnicodeTokenizer())
    expected = scorer.score(frame.loc[1, "reference_summary"], frame.loc[1, "generated_summary"])
    checks.equal(
        "scores are rouge_score's F1, rounded to six decimals",
        scores.loc[1].tolist(),
        [round(expected[name].fmeasure, 6) for name in ("rouge1", "rouge2", "rougeLsum")],
    )
    checks.raises("unequal lengths raise", ValueError, score_rouge, ["a"], [])


def verify_judges(checks: Checks) -> None:
    from lab.summarization.evaluation.judges import (
        CARE_DIMENSIONS,
        build_prompt,
        parse_judgement,
        read_prompt_template,
    )

    prompt = build_prompt("faithfulness", "SOURCE TEXT", "GENERATED", "es")
    checks.check("faithfulness sees the source and the summary", "SOURCE TEXT" in prompt and "GENERATED" in prompt)
    checks.check("and the language", "The texts are in es." in prompt)
    checks.check("no placeholder is left", "{source_text}" not in prompt and "{language}" not in prompt)
    checks.check("fluency does not see the source", "SOURCE TEXT" not in build_prompt("fluency", "SOURCE TEXT", "G", "es"))
    checks.check("prompts end on the JSON template", read_prompt_template("completeness").rstrip().endswith("}"))
    checks.raises("an unknown judge raises", ValueError, build_prompt, "consistency", "t", "g", "es")

    parsed = parse_judgement("faithfulness", PARSE_CASES["faithfulness"][1])
    checks.equal("prose around the JSON is ignored", (parsed["raw_score"], parsed["score"]), (9.0, 0.9))
    checks.equal("a 0-10 confidence is divided by 10", parse_judgement("faithfulness", PARSE_CASES["faithfulness"][2])["confidence"], 0.8)

    regex = parse_judgement("faithfulness", PARSE_CASES["faithfulness"][3])
    checks.equal("broken JSON keeps a findable score", (regex["score"], regex["flags"]), (0.6, ["json_malformed"]))

    for case in PARSE_CASES["faithfulness"][4:6] + PARSE_CASES["faithfulness"][7:]:
        judgement = parse_judgement("faithfulness", case)
        checks.equal(f"unreadable answer scores 0, flagged: {case[:40]!r}", (judgement["score"], judgement["flags"]), (0.0, ["llm_output_malformed"]))

    care = parse_judgement("completeness", ANSWERS["completeness"])
    checks.equal("completeness is the mean of the six dimensions", (care["raw_score"], care["score"]), (5.0, 0.5))
    checks.equal("each dimension is normalised", [care["dimensions"][name] for name in CARE_DIMENSIONS], [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    checks.equal("completeness asks for no confidence", care["confidence"], 0.0)
    checks.equal(
        "a missing dimension is unreadable",
        parse_judgement("completeness", PARSE_CASES["completeness"][3])["flags"],
        ["llm_output_malformed"],
    )
    checks.equal(
        "completeness has no score-only fallback",
        parse_judgement("completeness", PARSE_CASES["completeness"][4])["flags"],
        ["llm_output_malformed"],
    )


def verify_judging_and_scores(checks: Checks) -> None:
    from lab.summarization.evaluation.judging import JUDGEMENT_COLUMNS, judge_summaries
    from lab.summarization.evaluation.scores import (
        count_judgement_issues,
        describe_scores,
        describe_values,
        merge_scores,
    )

    frame = synthetic_summaries()
    frame.loc[2, "text"] = "LONG " + frame.loc[2, "text"]
    judge = CannedJudge(ANSWERS)
    judgements = judge_summaries(frame, judge)

    checks.equal("one row per summary and judge", judgements.shape, (27, len(JUDGEMENT_COLUMNS)))
    checks.equal("all prompts go to the model in one call", judge.calls, 1)
    checks.equal("rows follow the summaries, judges in order", judgements["judge"].tolist()[:3], ["faithfulness", "completeness", "fluency"])
    checks.equal("each answer is read by its own judge", judgements.groupby("judge")["score"].first().to_dict(), {"completeness": 0.5, "faithfulness": 0.8, "fluency": 0.0})
    checks.equal("a cut prompt is marked", int(judgements["truncated"].sum()), 2)
    checks.raises("an unknown judge raises", ValueError, judge_summaries, frame, judge, judges=["consistency"])

    subset = judge_summaries(frame, CannedJudge(ANSWERS), judges=["fluency"])
    checks.equal("judges narrows what is asked", subset["judge"].unique().tolist(), ["fluency"])

    classic = frame[["doc_id", "language"]].assign(rouge1=0.5, rouge2=0.25, rougeLsum=0.5, bertscore=0.75)
    scores = merge_scores(classic, judgements)
    checks.equal(
        "scores hold one column per metric and CARE dimension",
        list(scores.columns)[2:9],
        ["rouge1", "rouge2", "rougeLsum", "bertscore", "faithfulness", "completeness", "fluency"],
    )
    checks.equal("and one row per summary", len(scores), len(frame))
    checks.equal("CARE dimensions are columns", scores["completeness_diagnosis"].unique().tolist(), [0.6])
    checks.frames_equal("without judgements the classic scores stand alone", merge_scores(classic, None), classic)

    metrics = describe_scores(scores)
    checks.equal("statistics are per language", sorted(metrics), ["el", "en", "es"])
    checks.equal("each metric has mean, std, min, max", metrics["es"]["faithfulness"], {"mean": 0.8, "std": 0.0, "min": 0.8, "max": 0.8})
    checks.equal("the deviation is the population one", describe_values([0.0, 1.0])["std"], 0.5)
    checks.equal("no values describe as zeros", describe_values([]), {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0})

    issues = count_judgement_issues(judgements)
    checks.equal("unreadable answers are counted", issues["es"]["fluency"], {"n_judged": 3, "n_malformed": 3, "n_truncated": 0})
    checks.equal("and cut prompts, per judge", issues["es"]["faithfulness"]["n_truncated"], 1)


def verify_judge_model(checks: Checks, root: Path, tokenizer) -> Path:
    from transformers import AutoModelForCausalLM, LlamaConfig

    from lab.summarization.evaluation.judge_model import load_judge_model

    model_dir = root / "tiny-llama"
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.cls_token_id,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(root / "no-template")
    AutoModelForCausalLM.from_config(config).save_pretrained(root / "no-template")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(model_dir)

    judge = load_judge_model(str(model_dir), dtype="float32", device="cpu", max_model_len=64, max_new_tokens=8)
    checks.equal("a prompt keeps at most max_model_len - max_new_tokens tokens", judge.max_prompt_tokens, 56)

    short_ids, short_cut = judge.prepare("a short prompt")
    checks.check("a short prompt is not cut", not short_cut)
    checks.equal("the prompt is the chat template's user turn", short_ids, tokenizer.encode("a short prompt"))

    long_prompt = " ".join(f"word{index}" for index in range(200)) + " end of the instruction"
    long_ids, long_cut = judge.prepare(long_prompt)
    checks.check("a long prompt is cut", long_cut and len(long_ids) == 56)
    checks.equal("from the left, so its end survives", long_ids, tokenizer.encode(long_prompt)[-56:])

    answers = judge.generate([short_ids, long_ids])
    checks.check("generation answers each prompt with text", len(answers) == 2 and all(isinstance(answer, str) for answer in answers))
    checks.equal("greedy decoding is deterministic", judge.generate([short_ids]), answers[:1])
    checks.equal("the description records the loading", (judge.description["backend"], judge.description["device"], judge.description["max_new_tokens"]), ("transformers", "cpu", 8))

    checks.raises("a model without a chat template is refused", ValueError, load_judge_model, str(root / "no-template"), dtype="float32", device="cpu", match="chat template")
    checks.raises("an unknown backend raises", ValueError, load_judge_model, str(model_dir), backend="hf")
    checks.raises("an unknown dtype raises", ValueError, load_judge_model, str(model_dir), dtype="fp16")
    checks.raises("an answer budget past the context raises", ValueError, load_judge_model, str(model_dir), max_model_len=64, max_new_tokens=64)
    checks.raises("vLLM knobs are refused with transformers", ValueError, load_judge_model, str(model_dir), tensor_parallel_size=2, match="backend='vllm'")
    checks.raises("device is refused with vLLM", ValueError, load_judge_model, str(model_dir), backend="vllm", device="cpu", match="device")

    return model_dir


def verify_bertscore(checks: Checks, encoder_dir: Path) -> None:
    import bert_score.utils

    from lab.summarization.evaluation.bertscore import score_bertscore

    frame = synthetic_summaries().query("language == 'es'")
    generated, references = frame["generated_summary"].tolist(), frame["reference_summary"].tolist()
    original = bert_score.utils.sent_encode

    scores = score_bertscore(generated, references, language="es", model=str(encoder_dir), num_layers=2, device="cpu")
    checks.equal("one score per pair", len(scores), 3)
    checks.equal("a summary equal to its reference scores 1", scores[0], 1.0)
    checks.check("scores are in [0, 1]", all(0.0 <= score <= 1.0 for score in scores))
    checks.check("bert_score's sent_encode is restored afterwards", bert_score.utils.sent_encode is original)

    one_at_a_time = score_bertscore(generated, references, language="es", model=str(encoder_dir), num_layers=2, batch_size=1, device="cpu")
    checks.equal("the batch size does not change the scores", one_at_a_time, scores)
    checks.raises("unequal lengths raise", ValueError, score_bertscore, ["a"], [], language="es", model=str(encoder_dir), num_layers=2)


def verify_evaluate(checks: Checks, root: Path, encoder_dir: Path, judge_dir: Path) -> None:
    from lab.cli import main as cli_main
    from lab.summarization import evaluate_summaries, load_judge_model

    frame = synthetic_summaries()
    frame.loc[2, "text"] = "LONG " + frame.loc[2, "text"]
    classic = {"bertscore_model": str(encoder_dir), "bertscore_num_layers": 2, "device": "cpu"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluate_summaries(frame, root / "evaluation", judge_model=CannedJudge(ANSWERS), chunk_size=4, **classic)

    checks.check("a cut prompt warns with the count", any("2 of 27 judge prompts" in str(item.message) for item in caught))
    checks.equal(
        "every output is written",
        sorted(path.name for path in result.paths.values()),
        sorted(["classic_scores.parquet", "judgements.jsonl", "scores.parquet", "evaluation_metrics.json", "evaluation_manifest.json"]),
    )
    checks.equal("judgements are appended chunk by chunk", len(result.paths["judgements"].read_text(encoding="utf-8").splitlines()), 27)
    checks.frames_equal("scores read back as returned", pd.read_parquet(result.paths["scores"]), result.scores)
    checks.equal("scores carry every metric", list(result.scores.columns)[2:9], ["rouge1", "rouge2", "rougeLsum", "bertscore", "faithfulness", "completeness", "fluency"])
    checks.equal("metrics are per language", sorted(result.metrics), ["el", "en", "es"])
    checks.equal("the manifest counts summaries per language", result.manifest["languages"], {"el": 3, "en": 3, "es": 3})
    checks.equal("and unreadable answers per judge", result.manifest["judgement_issues"]["en"]["fluency"]["n_malformed"], 3)
    checks.equal("and records the ROUGE tokenizer", result.manifest["rouge"]["tokenizer"], "unicode-nfkc-v1")
    checks.check("and the judge model", result.manifest["judge_model"] == CannedJudge.description)
    checks.check("summary() prints one line per language", len(result.summary().splitlines()) == 3)
    checks.equal("a summary equal to its reference gets ROUGE and BERTScore 1", result.scores.loc[0, ["rouge1", "bertscore"]].tolist(), [1.0, 1.0])

    classic_only = evaluate_summaries(frame, root / "classic", judges=[], **classic)
    checks.equal("judges=[] scores the classic metrics only", list(classic_only.scores.columns), ["doc_id", "language", "rouge1", "rouge2", "rougeLsum", "bertscore"])
    checks.check("and writes no judgements", "judgements" not in classic_only.paths and classic_only.judgements is None)
    checks.frames_equal("classic scores do not depend on judging", classic_only.scores, result.scores[classic_only.scores.columns])

    checks.raises("judges without a judge model raise", ValueError, evaluate_summaries, frame, root / "x", **classic, match="judges=[] to evaluate")
    checks.raises("a judge model without judges raises", ValueError, evaluate_summaries, frame, root / "x", judge_model="m", judges=[], match="nothing would ask")
    checks.raises("an unknown judge raises", ValueError, evaluate_summaries, frame, root / "x", judge_model="m", judges=["consistency"])
    checks.raises(
        "loading arguments with a loaded model raise",
        ValueError,
        evaluate_summaries,
        frame,
        root / "x",
        judge_model=CannedJudge(ANSWERS),
        max_new_tokens=16,
        match="already loaded",
    )

    by_name = evaluate_summaries(
        frame.iloc[:2], root / "by-name", judge_model=str(judge_dir), judges=["fluency"],
        dtype="float32", max_model_len=128, max_new_tokens=8, **classic,
    )  # fmt: skip
    checks.equal("a judge model given by name is loaded and asked", len(by_name.judgements), 2)
    checks.equal("its loading is recorded", by_name.manifest["judge_model"]["model"], str(judge_dir))

    reused = load_judge_model(str(judge_dir), dtype="float32", device="cpu", max_model_len=128, max_new_tokens=8)
    again = evaluate_summaries(frame.iloc[:2], root / "reused", judge_model=reused, judges=["fluency"], **classic)
    checks.frames_equal("a loaded judge model judges as one loaded by name", again.judgements, by_name.judgements)

    frame.to_json(root / "results.jsonl", orient="records", lines=True, force_ascii=False)
    config_path = root / "evaluate.yaml"
    config_path.write_text(
        "task: summarization.evaluate_summaries\n"
        f"summaries: {root / 'results.jsonl'}\n"
        f"output_dir: {root / 'from-yaml'}\n"
        "judges: []\n"
        f"bertscore_model: {encoder_dir}\n"
        "bertscore_num_layers: 2\n"
        "device: cpu\n",
        encoding="utf-8",
    )
    checks.equal("the task runs from YAML", cli_main(["run", str(config_path)]), 0)
    checks.frames_equal(
        "and scores what the function scores",
        pd.read_parquet(root / "from-yaml" / "scores.parquet"),
        classic_only.scores,
    )
    checks.check("a YAML run records the input's sha256", len(json.loads((root / "from-yaml" / "evaluation_manifest.json").read_text())["summaries"][0]["sha256"]) == 64)


def verify_multiclinsum2_equivalence(checks: Checks, encoder_dir: Path) -> None:
    root = multiclinsum2_root()

    if root is None:
        checks.skip("equivalence with MultiClinSum2's evaluation", f"set {MULTICLINSUM2_ENV} to its evaluation/ checkout")
        return

    sys.path.insert(0, str(root))

    import config as ed2_config
    from metrics.classic.bertscore_metric import compute_bertscore_batch
    from metrics.classic.rouge_metric import compute_rouge
    from metrics.llm_judge.agents.completeness_agent import CompletenessAgent
    from metrics.llm_judge.agents.faithfulness_agent import FaithfulnessAgent
    from metrics.llm_judge.agents.fluency_agent import FluencyAgent
    from metrics.llm_judge.aggregator import _metric_stats
    from schemas import AgentOutput, EvaluationInput

    from lab.summarization.evaluation.bertscore import score_bertscore
    from lab.summarization.evaluation.judges import build_prompt, parse_judgement
    from lab.summarization.evaluation.rouge import score_rouge
    from lab.summarization.evaluation.scores import describe_values

    frame = synthetic_summaries()
    inputs = [
        EvaluationInput(
            sample_id=row.doc_id,
            team_name="team",
            language=row.language,
            source_text=row.text,
            generated_summary=row.generated_summary,
            reference_summary=row.reference_summary,
        )
        for row in frame.itertuples(index=False)
    ]

    ours = score_rouge(frame["generated_summary"].tolist(), frame["reference_summary"].tolist())
    theirs = [compute_rouge(item) for item in inputs]
    checks.equal(
        "ROUGE equals MultiClinSum2's on every pair",
        ours.values.tolist(),
        [[item.rouge1, item.rouge2, item.rougeLsum] for item in theirs],
    )

    agents = {"faithfulness": FaithfulnessAgent, "completeness": CompletenessAgent, "fluency": FluencyAgent}
    agents = {name: object.__new__(agent) for name, agent in agents.items()}

    for name, agent in agents.items():
        checks.check(
            f"the {name} prompt equals MultiClinSum2's",
            all(agent.build_prompt(item) == build_prompt(name, item.source_text, item.generated_summary, item.language) for item in inputs),
        )

    for name in ("faithfulness", "fluency", "completeness"):
        cases = PARSE_CASES["completeness" if name == "completeness" else "faithfulness"]

        for case in cases:
            try:
                expected = agents[name].parse_response(case, inputs[0])
            except Exception:
                expected = AgentOutput(
                    agent_name=name, sample_id="", team_name="", language="", raw_score=0.0,
                    normalized_score=0.0, reasoning="parsing_failed", flags=["llm_output_malformed"], confidence=0.0,
                )  # fmt: skip

            parsed = parse_judgement(name, case)
            detail = expected.completeness_detail
            checks.equal(
                f"{name} reads {case[:30]!r} as MultiClinSum2 does",
                (parsed["raw_score"], parsed["score"], parsed["confidence"], parsed["reasoning"], parsed["flags"], parsed["dimensions"]),
                (
                    expected.raw_score, expected.normalized_score, expected.confidence, expected.reasoning, expected.flags,
                    None if detail is None else {key: value for key, value in detail.model_dump().items() if key != "overall"},
                ),
            )  # fmt: skip

    values = [0.123456789, 0.5, 0.9, 0.0, 1.0]
    checks.equal("statistics equal MultiClinSum2's", describe_values(values), _metric_stats(values).model_dump())

    ed2_config.BERTSCORE_MODEL = str(encoder_dir)
    ed2_config.BERTSCORE_NUM_LAYERS = 2
    ed2_config.BERTSCORE_RESCALE_BASELINE = False
    ed2_config.BERTSCORE_BATCH_SIZE = 4
    generated, references = frame["generated_summary"].tolist(), frame["reference_summary"].tolist()
    ours = score_bertscore(generated, references, language="es", model=str(encoder_dir), num_layers=2, batch_size=4, device="cpu")
    theirs = compute_bertscore_batch(generated, references, language="es")
    checks.check("BERTScore equals MultiClinSum2's", all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(ours, theirs)))

    results = sorted((root.parent / "baseline_causal_llm" / "outputs" / "test_set").glob("*/el/results.jsonl"))

    if not results:
        checks.skip("ROUGE on a real MultiClinSum2 run", "no baseline results.jsonl beside the checkout")
        return

    from lab.summarization.io import read_summaries

    real = read_summaries(results[0]).iloc[:200]
    real_inputs = [
        EvaluationInput(sample_id=row.doc_id, team_name="t", language=row.language, source_text=row.text,
                        generated_summary=row.generated_summary, reference_summary=row.reference_summary)
        for row in real.itertuples(index=False)
    ]  # fmt: skip
    ours = score_rouge(real["generated_summary"].tolist(), real["reference_summary"].tolist())
    theirs = [compute_rouge(item) for item in real_inputs]
    checks.equal(
        f"ROUGE equals MultiClinSum2's on 200 real Greek summaries ({results[0].parent.parent.name})",
        ours.values.tolist(),
        [[item.rouge1, item.rouge2, item.rougeLsum] for item in theirs],
    )


def main() -> int:
    checks = Checks("summarization")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        verify_reading(checks, root)
        verify_rouge(checks)
        verify_judges(checks)
        verify_judging_and_scores(checks)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        encoder_dir = tiny_base_model(root, tokenizer)
        judge_dir = verify_judge_model(checks, root, AutoTokenizer.from_pretrained("bert-base-multilingual-cased"))
        verify_bertscore(checks, encoder_dir)
        verify_evaluate(checks, root, encoder_dir, judge_dir)
        verify_multiclinsum2_equivalence(checks, encoder_dir)

    return checks.report()


if __name__ == "__main__":
    run(main)
