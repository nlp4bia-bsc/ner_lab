# Evaluate generated summaries

```python
from lab.summarization import evaluate_summaries

result = evaluate_summaries(
    summaries="outputs/qwen2.5-7B-Instruct/es/results.jsonl",
    output_dir="assets/evaluation/qwen_es",
    judge_model="utter-project/EuroLLM-9B-Instruct",
    backend="vllm",
)

result.scores       # DataFrame: doc_id | language | rouge1 | rouge2 | rougeLsum | bertscore
                    #            | faithfulness | completeness | fluency | completeness_<dimension> ...
result.metrics      # {"es": {"rouge1": {"mean", "std", "min", "max"}, ...}}
result.summary()    # "es (1000 summaries): rouge1 0.412, rouge2 0.163, ..., fluency 0.871"
```

| Parameter | Default | Meaning |
|---|---|---|
| `summaries` | *required* | A summaries frame, a `.jsonl` or `.parquet` file, or a list of them, concatenated. See [the table](README.md#the-summaries-table). |
| `output_dir` | *required* | Where the scores are written. |
| `judge_model` | `None` | The judging LLM: a Hub name or local snapshot, loaded with the arguments below, or a `JudgeModel` from `load_judge_model`. Required unless `judges=[]`. |
| `judges` | `["faithfulness", "completeness", "fluency"]` | Which judges to ask. `[]` scores the classic metrics only. |
| `backend` | `"transformers"` | `"transformers"`, one prompt at a time; or `"vllm"`, every prompt of a chunk in one batched call. vLLM is installed separately. |
| `dtype` | `"float16"` | The judge's weights: `"float16"`, `"bfloat16"` or `"float32"`. |
| `device` | `"auto"` | Where BERTScore and a `transformers` judge run. vLLM places its model itself. |
| `max_model_len` | `4096` | Prompt plus answer, in tokens. A prompt keeps at most `max_model_len - max_new_tokens`. |
| `max_new_tokens` | `1024` | The longest answer. |
| `tensor_parallel_size` | `1` | vLLM only: GPUs to shard the judge across. |
| `gpu_memory_utilization` | `0.70` | vLLM only: the share of GPU memory it takes; the rest is for BERTScore. |
| `chunk_size` | `200` | Summaries judged per batch, and per write to `judgements.jsonl`. |
| `bertscore_model` | `"microsoft/mdeberta-v3-base"` | BERTScore's encoder: a Hub name or local snapshot. |
| `bertscore_num_layers` | `12` | The encoder layer BERTScore reads. Always passed, since `bert_score` cannot look up a local snapshot's. |
| `rescale_with_baseline` | `False` | Rescale against `bert_score`'s per-language baseline, which exists only for Hub names it ships one for. |
| `bertscore_batch_size` | `32` | Pairs per BERTScore call. |
| `on_failed` | `"raise"` | A generated summary starting with `ERROR:` marks a failed generation: `"raise"`, or `"drop"` it with a warning. |

**Returns** a `SummaryEvaluation` with `scores`, `judgements`, `metrics`, `manifest` and
`paths`. The judge-loading arguments (`backend` through `gpu_memory_utilization`, except
`device`) apply only when `judge_model` is a name; with a loaded model they raise.

## The classic metrics

**ROUGE-1, ROUGE-2 and ROUGE-Lsum** are F1 against the reference, from `rouge_score`, with no
stemming. ROUGE-Lsum splits on newlines before the longest common subsequence, which suits
multi-sentence summaries. Tokens are NFKC-normalised, lower-cased `\w+` runs
(`unicode-nfkc-v1`), so letters in any alphabet count and composed and decomposed accents
tokenize alike.

MultiClinSum2's official ROUGE was computed with `rouge_score`'s default tokenizer, which
deletes everything outside `[a-z0-9]`: Greek and Russian were scored on their digits alone
and accented words were split. Official ROUGE and this one agree for English only.

**BERTScore** is F1 with mdeberta-v3-base's 12th layer, each text cut at 512 tokens, scored
`bertscore_batch_size` pairs at a time per language. MultiClinSum2 ran without baseline
rescaling. BERTScore is not exactly reproducible across environments (Q23): with mdeberta, a
summary's score moves by up to about 1e-3 with the batch it is padded in, and the
transformers 5 tokenizer reads full-width punctuation as `[UNK]` where MultiClinSum2's read it
as ASCII.

## The judges

Each judge is a prompt template and a reader for its answer. The prompts are MultiClinSum2's,
verbatim, and ask for a JSON object:

| Judge | Reads | Answers |
|---|---|---|
| `faithfulness` | source + summary | one `score` in [0, 10]: is every statement supported by the source? |
| `completeness` | source + summary | six CARE dimensions in [0, 10]: demographics, presentation, diagnosis, intervention, outcome, follow-up; the score is their mean |
| `fluency` | summary only | one `score` in [0, 10]: grammar and readability, in the summary's language |

Scores are divided by 10, to [0, 1]. Decoding is greedy, so one model gives one prompt one
answer. Each prompt is the user turn of the judge model's chat template, so the judge must be
an instruction-tuned model; a prompt longer than `max_model_len - max_new_tokens` is cut from
the start — the source text — so the instruction at its end survives, and the run warns with
how many were cut.

**An unreadable answer scores 0**, flagged `llm_output_malformed`, rather than stopping the
run (D136): one bad generation in thousands should not cost a GPU job. Since a 0 enters the
mean, the manifest counts, per language and judge, how many answers were unreadable. When an
answer's JSON is broken but a `"score": n` can still be found, that score is kept and flagged
`json_malformed`.

Cross-lingual consistency, MultiClinSum2's fourth judge, is not here (D137).

## What it writes

```
<output_dir>/
    classic_scores.parquet      ROUGE and BERTScore per summary, written before judging starts
    judgements.jsonl            one line per summary and judge: scores, reasoning, flags,
                                CARE dimensions, whether the prompt was cut, the full answer;
                                appended after every chunk
    scores.parquet              one row per summary, one column per metric
    evaluation_metrics.json     mean, std, min and max of every metric, per language
    evaluation_manifest.json    inputs and their sha256, settings, the judge model,
                                unreadable answers and cut prompts per judge
```

A job killed during judging keeps `classic_scores.parquet` and every judged chunk. The
standard deviation is the population one, since the evaluation set is fixed.

## Backends

`backend="transformers"` runs anywhere `lab[summarization]` is installed, one prompt at a
time. `backend="vllm"` sends a whole chunk to one engine and is several times faster on a GPU,
but vLLM pins its own `torch`, which need not be the `torch==2.10.0` that `lab[torch]` pins,
so it is not part of the extra (D134). Let vLLM bring its `torch` and add `lab` without
dependencies, as MultiClinSum2's own environment did:

```bash
pip install vllm
pip install --no-deps -e /path/to/ner_lab
pip install pandas pyarrow pyyaml pysbd nltk icu4py nervaluate regex rouge-score bert-score sentencepiece
```

To judge several runs with one model load, load it once:

```python
from lab.summarization import evaluate_summaries, load_judge_model

judge = load_judge_model("utter-project/EuroLLM-9B-Instruct", backend="vllm", tensor_parallel_size=2)

for language in ("es", "en", "el"):
    evaluate_summaries(
        summaries=f"outputs/qwen/{language}/results.jsonl",
        output_dir=f"assets/evaluation/qwen/{language}",
        judge_model=judge,
    )
```

Or put every language in one run: `summaries` takes a list of files.

## From YAML

```yaml
task: summarization.evaluate_summaries
summaries:
  - outputs/qwen2.5-7B-Instruct/es/results.jsonl
  - outputs/qwen2.5-7B-Instruct/en/results.jsonl
output_dir: assets/evaluation/qwen
judge_model: /gpfs/.../models--utter-project--EuroLLM-9B-Instruct/snapshots/f7ae2bc3...
backend: vllm
bertscore_model: /gpfs/.../hf_models/mdeberta-v3-base
```

Classic metrics only:

```yaml
task: summarization.evaluate_summaries
summaries: outputs/qwen2.5-7B-Instruct/es/results.jsonl
output_dir: assets/evaluation/qwen_es_classic
judges: []
```
