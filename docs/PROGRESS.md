# Progress

What has actually landed, and the evidence for it. One entry per subsystem, not per session.
Decisions live in [DECISIONS.md](DECISIONS.md); this file records outcomes.

**Verification suite: 348 checks, all passing.** See [`verification/`](../verification/).

| Subsystem | Checks | Equivalence with NER-API |
|---|---|---|
| `data/` | 97 | conversion + split on the real MultiClinNER sample |
| `encoding/` leaves | 79 | — |
| `encoding/` `Encoder` | 45 | exact, 4 tokenizer × strategy combinations |
| `models/` | 33 | — |
| `training/` | 35 | — |
| `evaluation/` | 59 | exact, every metric family |

---

## Packaging — 2026-08-05

NER-API had no `[build-system]`, so it was never installable. Added `pyproject.toml`
(hatchling, `src/` layout, `ner-lab` console script), `.gitignore`, and this document set.

Seven dependencies removed: `dataclasses`/`pathlib`/`typing` (stdlib since 3.4–3.7, but
installed as real redundant files), `nltk`/`pydantic`/`psutil` (zero imports). One latent bug
fixed: `ray` was declared without `[tune]` while the code imports `ray.tune`, `ray.air` and
`ray.tune.search.optuna` — none of which ship in the base wheel. It worked in NER-API's venv
only because something else pulled them in.

---

## `data/` — 2026-08-06

`labels.py`, `brat.py`, `corpus.py`, `io.py`, `stratification.py`, `assignments.py`,
`split.py`, `dataset.py`. Task `prepare_dataset` wired into the CLI.

`BratLoader` became module functions and lost ~150 lines of JSONL handling, dead since the
parquet migration. The canonical corpus schema moved out of the splitter and into
`corpus.py`, inverting a backwards dependency — the splitter had owned the schema every other
stage depends on.

Three places gained a pure path that returns objects instead of only writing files:
`build_corpus`, `split_documents`, and later `Encoder.encode`.

Verified against NER-API on the real `MultiClinNER-es-train-disease` sample (1258 documents,
26296 entities): byte-identical corpora and splits.

**Two bugs found in my own work, both caught by verification:**

- `create_split` accepted a DataFrame without running it through `validate_corpus`, so the
  same corpus serialized as `int32` via a frame and `int8` via a path. Fixed by making
  `validate_corpus` the single normalization point.
- A removability audit found label normalization welded into the BRAT reader with no opt-out.
  Fixed by threading `normalize_labels` through five functions — and later inverted entirely
  by D32.

Also proved the data layer is entity-agnostic, on a `GENE`/`protein`/`cell-line`/`Chemical`
corpus.

---

## `encoding/` — 2026-08-06

`overlaps.py`, `segmentation.py`, `windowing.py`, `tagging.py`, `rows.py`, `encoder.py`.
`Encoder` replaces NER-API's `DataLoader`.

The two `_process_document_greedy` / `_process_document_context` methods collapsed into one
path that detects a core window by the presence of `core_token_start`, rather than by
strategy name. That is what makes a user-supplied strategy a real extension point: a custom
callable that tags core bounds gets flank masking without `Encoder` knowing anything about it.

Entity-type whitelists removed from three places, so any label vocabulary works.

Verified exact against `DataLoader` on 60 real documents:

```
gpt2/greedy  493 rows    bert/greedy  465 rows
gpt2/context 921 rows    bert/context 915 rows
encode(df) == encode_parquet(path) in all four
```

Guards NER-API lacked: `context_tokens` is rejected for the greedy strategy (it only enforced
the converse), and a `max_length` too small for the tokenizer's special tokens raises instead
of silently producing a negative budget.

### The process failure worth remembering

I listed ten "findings" about the port, flagged one for the user, then implemented the rest as
if agreed. Audited on request and disclosed eleven unilateral decisions. Two were reverted
(`context_tokens` and `resolve_entities(policy=)` are required again, as NER-API had them).

Standing rule, now in [DESIGN.md](DESIGN.md): **findings are not decisions. Ask before
implementing one.**

---

## Verification moved into the repo — 2026-08-07

D8 amended: no tests ship in the package, but verification lives in the repo. Everything had
been disposable scratch scripts in `/tmp`, which was cleared between sessions and took all of
it with it.

`verification/` sits outside `src/`, so hatchling never packages it. Everything runs on
synthetic fixtures, so a machine with no corpora gets a full green run; real-data checks skip
loudly rather than silently passing.

One real defect surfaced while writing them: `build_label_vocabulary("  ")` did not raise,
because a whitespace-only string is truthy, so it built `B-  ` while the error message beside
it promised a non-empty label.

---

## `models/` — 2026-08-07

`bio.py`, `crf.py`, `registry.py`.

`BaseTokenClassificationModel` was 661 lines doing model construction, training, prediction,
saving, metrics and `TrainingArguments` behind one class. Only the first of those is in
`models/` now; the rest went to `training/`. Swapping an architecture no longer means
subclassing a trainer — `build_model` takes a name or a plain factory callable.

BIO constraint reasoning is torch-free in `bio.py`, returning nested booleans that `crf.py`
turns into buffers. It is pure combinatorics over label strings and the part most likely to be
silently wrong, so keeping it out of the tensor code makes it directly verifiable.

---

## `training/` — 2026-08-07

`dataset.py`, `arguments.py`, `tracking.py`, `trainer.py`.

`NERTrainingConfig`'s 46 fields, counted honestly: 8 encoding (deleted — `Encoder` owns them),
5 model (deleted — `build_model` owns them), 2 data paths (now DataFrames), 16 optimization,
15 run plumbing. Roughly 16 of the last 31 were `TrainingArguments` fields under different
names, so `train()` accepts a `TrainingArguments` directly and `training_arguments()` supplies
this library's defaults.

`best_config.json` became `training_summary.json`, assembled by a public `build_summary`.
NER-API's `write_best_config` read 46 config fields back out by hand; `TrainingArguments
.to_dict()` does that for free.

Dropped: `metrics_module_path` (imported metrics by string, would break silently on rename),
`filter_supported_kwargs` (introspection papering over config/constructor drift that no longer
exists), and all three `*Config` dataclasses.

The verification environment gained CPU torch here, so `models/` and `training/` are actually
exercised. End-to-end checks train a randomly initialized miniature BERT written to a temp
directory — nothing downloads, and both the linear and CRF paths run in seconds. They confirm
checkpoint cleanup on the no-save path, weights plus tokenizer on the save path, a finite CRF
loss, and the early-stopping guard firing.

---

## `evaluation/` — 2026-08-07

`spans.py`, `scoring.py`, `tokens.py`, `metrics.py`, `multiclinner.py`.

The migration flagged as highest-risk: offset arithmetic where a wrong answer still looks
plausible. Verified exact against NER-API on 40 real documents with random logits — gold
spans, predicted spans, flattened span metrics, token metrics, per-entity token metrics, the
confusion matrix, and the official MultiClinNER scorer all match element for element.

The legacy word-level scoring path (~250 lines) is gone: `Encoder` only emits token-level
rows, so it was unreachable.

`compute_span_metrics_from_logits` split into `evaluate_predictions` (everything) and
`span_metrics` (spans only), so HPO can compute its objective without paying for token metrics
and a confusion matrix on every trial.

D41 discharged: `training_arguments()` now defaults to `span_strict_f1`, and `train()` raises
immediately if a span metric is requested with no `compute_metrics`, naming
`build_compute_metrics` in the message.
