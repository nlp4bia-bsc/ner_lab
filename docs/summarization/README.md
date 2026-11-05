# `lab.summarization` — clinical summarization

**Authors**:  Miguel Rodríguez Ortega · miguel.rod.bsc@gmail.com

Evaluates generated summaries of clinical case reports: ROUGE and BERTScore against a
reference summary, and LLM judges that read the summary against its source. Every metric is
reported on its own; nothing combines them. Ported from the MultiClinSum2 shared-task
evaluation (D132). Installed by `lab[summarization]`; the optional vLLM backend is installed
separately (D134).

| Page | Covers |
|---|---|
| [evaluate-summaries.md](evaluate-summaries.md) | `evaluate_summaries`: the summaries table, the classic metrics, the judges and their backends, what it writes |

| CLI task | Function |
|---|---|
| `summarization.evaluate_summaries` | `lab.summarization.evaluate_summaries` |

## The summaries table

One row per summarised document:

```
doc_id | language | text | reference_summary | generated_summary
```

`text` is the source the summary was written from. A document may appear once per language,
so several languages are evaluated in one run. `read_summaries` reads it from a frame, a
`.parquet`, or the `results.jsonl` files the MultiClinSum baselines and `prepare_submission.py`
write, whose `case_id` and `full_case` are renamed to `doc_id` and `text` (D133). The table
has one consumer, so it lives here rather than in `lab.core` (D84).

## Importing

```python
from lab.summarization import (
    evaluate_summaries,   # the task
    load_judge_model,     # load a judge once, to reuse across runs
    read_summaries,       # the summaries table from JSONL, parquet or a frame
)
```

The pieces underneath are in `lab.summarization.evaluation`: `score_rouge`,
`score_bertscore`, `build_prompt`, `parse_judgement`, `judge_summaries`, `merge_scores`,
`describe_scores`. As with `lab.ner`, names on `lab.summarization` resolve on first use, so
`import lab.summarization` loads no torch.
