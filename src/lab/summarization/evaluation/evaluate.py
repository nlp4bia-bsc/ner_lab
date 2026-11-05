"""The `summarization.evaluate_summaries` task: generated summaries to scores and statistics."""

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from lab.core.provenance import file_sha256, write_manifest
from lab.summarization.evaluation.bertscore import (
    DEFAULT_BERTSCORE_LAYERS,
    DEFAULT_BERTSCORE_MODEL,
    MAX_SENTENCE_TOKENS,
    score_bertscore,
)
from lab.summarization.evaluation.judge_model import (
    DEFAULT_DTYPE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    Backend,
    JudgeModel,
    load_judge_model,
)
from lab.summarization.evaluation.judges import JUDGES, Judge
from lab.summarization.evaluation.judging import judge_summaries
from lab.summarization.evaluation.rouge import ROUGE_TOKENIZER, ROUGE_TYPES, score_rouge
from lab.summarization.evaluation.scores import (
    count_judgement_issues,
    describe_scores,
    merge_scores,
)
from lab.summarization.io import FailedPolicy, read_summaries

CLASSIC_SCORES_FILENAME = "classic_scores.parquet"
JUDGEMENTS_FILENAME = "judgements.jsonl"
SCORES_FILENAME = "scores.parquet"
METRICS_FILENAME = "evaluation_metrics.json"
EVALUATION_MANIFEST_FILENAME = "evaluation_manifest.json"
DEFAULT_CHUNK_SIZE = 200


@dataclass(frozen=True)
class SummaryEvaluation:
    """What evaluating summaries produced: per-summary scores, judgements and statistics."""

    output_dir: Path
    scores: pd.DataFrame
    judgements: pd.DataFrame | None
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    paths: dict[str, Path]

    def summary(self) -> str:
        """One line per language with the mean of every metric."""
        lines = []

        for language, metrics in self.metrics.items():
            means = ", ".join(
                f"{name} {stats['mean']:.3f}"
                for name, stats in metrics.items()
                if not name.startswith("completeness_")
            )
            lines.append(f"{language} ({self.manifest['languages'][language]} summaries): {means}")

        return "\n".join(lines)


def evaluate_summaries(
    summaries: pd.DataFrame | str | Path | Sequence[str | Path],
    output_dir: str | Path,
    judge_model: str | Path | JudgeModel | None = None,
    judges: Sequence[Judge] = JUDGES,
    backend: Backend = "transformers",
    dtype: str = DEFAULT_DTYPE,
    device: str = "auto",
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
    bertscore_num_layers: int = DEFAULT_BERTSCORE_LAYERS,
    rescale_with_baseline: bool = False,
    bertscore_batch_size: int = 32,
    on_failed: FailedPolicy = "raise",
) -> SummaryEvaluation:
    """
    Score generated summaries against their references and sources, and write the scores.

    `summaries` is a summaries frame, a `.jsonl` or `.parquet` file, or a list of
    them, as `read_summaries` reads them; `on_failed` is passed to it. Languages
    may be mixed: every metric is computed per summary and summarised per
    language.

    Every summary gets ROUGE-1, ROUGE-2 and ROUGE-Lsum F1 against its reference
    (Unicode tokens, no stemming) and BERTScore F1 with `bertscore_model`, then one
    score in [0, 1] from each of `judges`, asked of `judge_model`: faithfulness and
    completeness against the source text, fluency on the summary alone. Every
    metric is its own column; nothing combines them. `judges=[]` evaluates the
    classic metrics only, and needs no `judge_model`.

    `judge_model` is a model name or path, loaded with `backend`, `dtype`,
    `device`, `max_model_len`, `max_new_tokens`, `tensor_parallel_size` and
    `gpu_memory_utilization` as `load_judge_model` describes, or a `JudgeModel`
    already loaded, so one model can judge several runs. Judging runs `chunk_size`
    summaries at a time and appends each chunk to `judgements.jsonl`, so a job
    killed partway keeps what it judged; `classic_scores.parquet` is written
    before judging starts for the same reason.

    `scores.parquet` holds one row per summary, `evaluation_metrics.json` the mean,
    standard deviation, minimum and maximum of each metric per language, and
    `evaluation_manifest.json` the inputs, settings, and per judge how many
    answers were unreadable (scored 0) and how many prompts were cut to fit.
    """
    loaded = not isinstance(judge_model, (str, Path))
    _check_judging(
        judge_model,
        judges,
        loaded,
        {
            "backend": (backend, "transformers"),
            "dtype": (dtype, DEFAULT_DTYPE),
            "max_model_len": (max_model_len, DEFAULT_MAX_MODEL_LEN),
            "max_new_tokens": (max_new_tokens, DEFAULT_MAX_NEW_TOKENS),
            "tensor_parallel_size": (tensor_parallel_size, DEFAULT_TENSOR_PARALLEL_SIZE),
            "gpu_memory_utilization": (gpu_memory_utilization, DEFAULT_GPU_MEMORY_UTILIZATION),
        },
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    summaries_df = read_summaries(summaries, on_failed=on_failed)
    languages = summaries_df["language"].value_counts().sort_index()
    print(
        f"Evaluating {len(summaries_df)} summaries in {len(languages)} language(s): "
        f"{', '.join(languages.index)}",
        flush=True,
    )

    classic_df = _score_classic(
        summaries_df,
        model=bertscore_model,
        num_layers=bertscore_num_layers,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=bertscore_batch_size,
        device=device,
    )
    paths = {"classic_scores": output_dir / CLASSIC_SCORES_FILENAME}
    classic_df.to_parquet(paths["classic_scores"], index=False)

    judgements_df: pd.DataFrame | None = None
    judge_description: dict[str, Any] | None = None

    if judges:
        _release_gpu_memory()

        if not loaded:
            judge_model = load_judge_model(
                model=str(judge_model),
                backend=backend,
                dtype=dtype,
                device=device if backend == "transformers" else "auto",
                max_model_len=max_model_len,
                max_new_tokens=max_new_tokens,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )

        paths["judgements"] = output_dir / JUDGEMENTS_FILENAME
        judgements_df = _judge_in_chunks(
            summaries_df, judge_model, judges, chunk_size, paths["judgements"]
        )
        judge_description = judge_model.description
        _warn_truncated(judgements_df, judge_model)

    scores_df = merge_scores(classic_df, judgements_df)
    paths["scores"] = output_dir / SCORES_FILENAME
    scores_df.to_parquet(paths["scores"], index=False)

    metrics = describe_scores(scores_df)
    paths["metrics"] = write_manifest(metrics, output_dir / METRICS_FILENAME)

    issues = None if judgements_df is None else count_judgement_issues(judgements_df)
    manifest = {
        "summaries": _describe_summaries(summaries),
        "output_dir": str(output_dir),
        "n_summaries": int(len(summaries_df)),
        "languages": {str(language): int(count) for language, count in languages.items()},
        "on_failed": on_failed,
        "rouge": {
            "types": list(ROUGE_TYPES),
            "tokenizer": ROUGE_TOKENIZER,
            "use_stemmer": False,
            "rouge_score": metadata.version("rouge-score"),
        },
        "bertscore": {
            "model": bertscore_model,
            "num_layers": int(bertscore_num_layers),
            "rescale_with_baseline": bool(rescale_with_baseline),
            "batch_size": int(bertscore_batch_size),
            "max_tokens": MAX_SENTENCE_TOKENS,
            "bert_score": metadata.version("bert-score"),
        },
        "judges": list(judges),
        "judge_model": judge_description,
        "chunk_size": int(chunk_size),
        "judgement_issues": issues,
        "duration_sec": round(time.perf_counter() - started, 3),
    }
    paths["evaluation_manifest"] = write_manifest(
        manifest, output_dir / EVALUATION_MANIFEST_FILENAME
    )

    return SummaryEvaluation(
        output_dir=output_dir,
        scores=scores_df,
        judgements=judgements_df,
        metrics=metrics,
        manifest=manifest,
        paths=paths,
    )


def _check_judging(
    judge_model: Any,
    judges: Sequence[str],
    loaded: bool,
    loading: dict[str, tuple[Any, Any]],
) -> None:
    unknown = [judge for judge in judges if judge not in JUDGES]

    if unknown:
        raise ValueError(f"Unknown judges {unknown}. Available judges: {JUDGES}.")

    if judges and judge_model is None:
        raise ValueError(
            f"judges={list(judges)} need a judge_model to ask. Pass judge_model= a model name "
            "or path, or judges=[] to evaluate the classic metrics only."
        )

    if not judges and judge_model is not None:
        raise ValueError("judge_model is given but judges is empty; nothing would ask it.")

    changed = [name for name, (value, default) in loading.items() if value != default]

    if judge_model is not None and loaded and changed:
        raise ValueError(
            f"{changed} configure loading judge_model by name, but it is already loaded. "
            "Pass them to load_judge_model instead."
        )


def _score_classic(summaries_df: pd.DataFrame, **bertscore: Any) -> pd.DataFrame:
    rouge_df = score_rouge(
        summaries_df["generated_summary"].tolist(), summaries_df["reference_summary"].tolist()
    )
    bertscores = pd.Series(0.0, index=summaries_df.index, name="bertscore")

    for language, group in summaries_df.groupby("language", sort=True):
        print(f"  BERTScore: {len(group)} {language} summaries", flush=True)
        bertscores.loc[group.index] = score_bertscore(
            group["generated_summary"].tolist(),
            group["reference_summary"].tolist(),
            language=str(language),
            **bertscore,
        )

    return pd.concat([summaries_df[["doc_id", "language"]], rouge_df, bertscores], axis=1)


def _release_gpu_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _judge_in_chunks(
    summaries_df: pd.DataFrame,
    judge_model: JudgeModel,
    judges: Sequence[Judge],
    chunk_size: int,
    path: Path,
) -> pd.DataFrame:
    chunks = []
    started = time.perf_counter()

    with path.open("w", encoding="utf-8") as file:
        for start in range(0, len(summaries_df), chunk_size):
            chunk_df = judge_summaries(
                summaries_df.iloc[start : start + chunk_size], judge_model, judges=judges
            )
            file.write(chunk_df.to_json(orient="records", lines=True, force_ascii=False))
            file.flush()
            chunks.append(chunk_df)

            done = min(start + chunk_size, len(summaries_df))
            elapsed = time.perf_counter() - started
            print(f"  judged {done}/{len(summaries_df)} summaries in {elapsed:.0f}s", flush=True)

    return pd.concat(chunks, ignore_index=True)


def _warn_truncated(judgements_df: pd.DataFrame, judge_model: JudgeModel) -> None:
    truncated = judgements_df["truncated"]

    if not truncated.any():
        return

    examples = judgements_df.loc[truncated, ["doc_id", "language", "judge"]]

    warnings.warn(
        f"{int(truncated.sum())} of {len(judgements_df)} judge prompts were longer than "
        f"{judge_model.max_prompt_tokens} tokens and were cut from the start, e.g. "
        f"{list(examples.itertuples(index=False, name=None))[:10]}. Raise max_model_len "
        "to keep them whole.",
        stacklevel=3,
    )


def _describe_summaries(summaries: Any) -> list[dict[str, str]] | None:
    if isinstance(summaries, pd.DataFrame):
        return None

    paths = [summaries] if isinstance(summaries, (str, Path)) else list(summaries)

    return [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in paths]
