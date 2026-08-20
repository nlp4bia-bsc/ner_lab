# Progress

What has actually landed, and the evidence for it. One entry per subsystem, not per session.
Decisions live in [DECISIONS.md](DECISIONS.md); this file records outcomes.

**Verification suite: 598 checks, all passing.** See [`verification/`](../../verification/).

| Subsystem | Checks | Equivalence with NER-API |
|---|---|---|
| `data/` | 97 | conversion + split on the real MultiClinNER sample |
| `encoding/` leaves | 79 | — |
| `encoding/` `Encoder` | 45 | exact, 4 tokenizer × strategy combinations |
| `models/` | 33 | — |
| `training/` | 35 | — |
| `evaluation/` | 59 | exact, every metric family |
| `training/assessment.py` | 65 | — |
| `hpo/` | 94 | — |
| `inference.py` | 91 | not diffed — see below |

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

---

## Training orchestrator — 2026-08-07

`training/assessment.py`. Task `train_model`, the first new task since `prepare_dataset`.

`02_train_assessment.py` was 347 lines, of which roughly half was argparse and config
plumbing for `best_config.json`. That half is gone: `TrainingArguments` is the config, and
the YAML supplies it as a mapping of overrides (D49). What survives is the part that was
actually load-bearing — read the split's `data_manifest.json`, pick the run shape from its
mode, write a manifest before training so a crash is debuggable, train, aggregate.

Everything the script kept private is now a public function: `fold_rotations`,
`run_directory_name`, `resolve_training_arguments`, `encode_partition`, `best_epoch_metrics`,
`aggregate_metrics`. D26 applied to a second orchestrator.

Three behaviour changes from NER-API, each with a number: the summary files are renamed and
written in both split modes (D51), the return value carries metric tables rather than live
`Trainer`s (D50), and no git commit is recorded (D52).

Two things the script did not do:

- **Encoded rows are cached per fold parquet, not just loaded.** NER-API cached the loaded
  DataFrames, so a 5-fold rotation tokenized every fold four times. Encoding is the expensive
  half; loading a parquet is not.
- **Each fold builds a fresh model and releases the previous one.** The script rebuilt per
  fold too, but nothing dropped the old one — with the results held in a list, five
  BERT-base folds accumulate on the GPU while the next one trains.

Verified on synthetic fixtures with the miniature BERT: both split modes, fold rotation
including the `folds=` narrowing, the fixed holdout never entering a run, manifest contents,
artifact layout, checkpoint cleanup, and `save_model` writing weights. The pure helpers —
rotation selection, best-epoch lookup with `greater_is_better` in both directions, mean/std
aggregation — are checked directly rather than through a training run.

---

## `hpo/` — 2026-08-07

`space.py`, `variants.py`, `trial.py`, `search.py`. Task `search_hyperparameters`.

The stage D29 flagged for top-down migration: 13 of `01_HPO_ner.py`'s functions lived in the
script, and lifting its 691-line shape would have made Ray-script structure the public API.
Instead the signature was agreed first (D54–D57), and the script's functions landed as public
pieces behind it: `build_search_space`/`describe_search_space` (the declarative YAML form and
its inverse), `build_variants`/`encode_variants` (the variant dimension, windowed once per
sweep), `resolve_batch_sizes`, `top_k_epoch_mean`, `run_trial` (the whole trial body,
callable and verifiable without Ray), `winner_configuration`.

What survived intact from NER-API, deliberately: the noise-control scoring (mean over seeds
of each seed's mean-of-top-k epochs, with the measured jitter rationale), OOM-as-outcome
(caught, scored worst) versus everything-else-halts-the-sweep (`fail_fast`), the pre-flight
smoke test, checkpoint-free trials, the variant dimension being one dimension rather than
four, and Optuna resumability via `study_name`/`storage`.

What changed, each with a number: trials call `train()` with pre-encoded rows rather than any
orchestrator (D54), the search space is user-supplied with the old hardcoded ranges demoted
to `DEFAULT_SEARCH_SPACE` (D55), input is a `split_dir` with one validation rotation (D56),
and the winner comes out as a ready-to-run `train_model` YAML block instead of a
`best_config.json` anything reads back (D57). `train()`'s early-stopping guard was relaxed to
fire only when weights are kept (D58) — verified against transformers 5.14.1, where the
callback works without checkpointing and HF itself only warns.

Trial objectives honor `greater_is_better` end to end (Optuna mode, top-k direction, OOM
worst score), where NER-API assumed maximization throughout. Trial evaluation skips token
metrics and the confusion matrix via `build_compute_metrics(include_tokens=False,
include_by_entity=False)` — the split `evaluation/` made for exactly this.

Verified: 94 checks, including a real two-trial Ray Tune + Optuna sweep on the miniature
BERT (smoke test, manifest, trials table, summary), an OOM trial reported rather than
raised, a non-OOM error propagating, and the emitted winner block executed through
`train_model(**block)` unchanged. The YAML path was smoke-tested separately:
`ner-lab run sweep.yaml --seed 5` with a declarative search space, seed reaching the
manifest.

The winner block is written to `winner.yaml` as well as into `hpo_summary.json` (D60), so the
hand-off to `train_model` is one command rather than a hand-extracted JSON subtree retyped as
YAML. Verified by reading the file back through the CLI's own `load_config` and comparing it
to the block.

---

## `inference.py` — 2026-08-09

Task `predict_entities`. The last unmigrated library stage.

`03_infer_model.py` was 653 lines. Most of it was already in this library and did not need
porting: `CRFTrainer.prediction_step` re-encodes Viterbi paths as one-hot logits, so both
architectures reach decoding in one shape; `evaluate_predictions` replaces its four separate
metric calls; `multiclinner` replaces its official-scoring block; `NERTrainingConfig`,
`filter_supported_kwargs` and `get_git_commit` were already gone by D30, D38 and D52.

What survived is the part that was actually load-bearing: load a checkpoint, window the input
the way the model was trained, predict, decode spans with a confidence, score against gold if
there is any.

Four decisions, agreed before any of it was written (D29's rule, applied to the second
script-shaped stage):

- **A saved model describes itself** (D62). `train(model_encoding=...)` writes an
  `encoding.json` beside the weights. Inference takes a `model_dir` and restates nothing — a
  wrong `max_length` silently shifts every window boundary, which is precisely the class of
  transcription failure `winner.yaml` had just been added to remove.
- **`Encoder(require_target_label=False)`** (D63), because unannotated documents legitimately
  contain no entity and the guard that catches a training typo would otherwise reject them.
- **Score and text go into `evaluation/spans.py`** (D64), not into a second BIO walk. Both
  additions are off by default, so the scoring path is unchanged.
- **`predict_entities`, taking a model directory rather than a run directory** (D61), so a
  checkpoint copied anywhere still loads.

The CRF path is the one NER-API never validated on a cluster. It now runs in verification:
`CRFForTokenClassification` is an `nn.Module`, so `Trainer.save_model` writes a bare state
dict with no `config.json` — the wrapper is rebuilt over the backbone `encoding.json` names
and the fine-tuned weights loaded on top.

Verified: 91 checks. The span-decoding arithmetic is checked directly — mean-of-tokens scores,
surface forms sliced from the document, the higher-scoring copy winning when two windows
predict the same span, softmax not overflowing on a large logit. End to end on the miniature
BERT, both architectures: training writes the sidecar, a saved model reloads and predicts,
gold input is scored twice over (this library's metrics and the official scorer), raw text
predicts without gold and is officially scored only when given a `reference`, and `min_score`
filters. The YAML path was smoke-tested separately with `ner-lab run predict.yaml`.

**Two bugs of my own, both found by auditing the finished stage rather than by the checks:**

- A model that predicted no entity at all returned a frame with no `score` column, so
  `min_score` and the prediction writer both raised `KeyError`. `span_dataframe` now takes the
  caller's intent (`scored=`) instead of inferring it from spans that may not exist.
- `device="cpu"` was honoured by `load_model` and then silently undone: a `Trainer` places the
  model itself, so on a GPU machine it moved it straight back. `predict_logits` now sets
  `use_cpu` on the arguments it builds, which is where the decision actually lives. NER-API
  did this by setting `CUDA_VISIBLE_DEVICES=""`, a global environment mutation a library
  should not perform.

**The one thing this stage does not have: equivalence with NER-API.** `data/`, `encoding/` and
`evaluation/` were each diffed element-for-element against the original on real documents
before being allowed to differ. Inference was not, because its script cannot run: it loads a
`best_config.json` written by a `NERTrainingConfig` that no longer exists, so there is no
checkpoint on disk both implementations can read. The pieces it is built from are the ones
already verified exact — span reconstruction, the metric families, the official scorer — and
what is new here is the decoding of score and text, which is checked directly. Worth stating
plainly rather than leaving implied by a table.
