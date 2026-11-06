# Multi-Head integration: validation record

Historical record before unified ensembles (D136–D140). Current ensemble support and checks
are described in [UNIFIED_ENSEMBLE_RESULTS.md](UNIFIED_ENSEMBLE_RESULTS.md).

2026-11-06. The implementation is integrated and locally verified; full acceptance remains
blocked by remote tokenizer access and review of the four requested scientific sources.
No large clinical checkpoint or GPU was used. No new runtime dependency was added to the
repository; its existing NER dependencies, including codecarbon, were installed locally.

## Integration map and public contracts

The path remains `Encoder -> build_model -> train/Trainer -> evaluation -> saved model ->
predict_entities`. `train_model` retains split rotation and artifacts; HPO retains encoded
variants, Ray/Optuna, seeds, epoch selection, OOM handling and `winner.yaml`. The CLI and its
task table have no architecture-specific branch.

- Shared: canonical corpus and span tables; tokenizer/segmentation; windows; first-subword
  and context masks; BIO tagger and span decoder; Dataset; Trainer scheduling/callbacks;
  provenance; prediction TSV/Parquet; character scoring.
- Extended: optional head catalogue on `Encoder`; `[L,K]` label conversion and padding;
  logits-aware reconstruction and metrics; architecture defaults; model persistence/loading;
  HPO variant catalogue; fold-specific discovery and provenance.
- New concerns: vectorized shared-backbone model, per-type overlap/discovery helpers,
  exact multitype set scoring, vector-label collator, architecture-specific argument helpers.

`train_model` and `search_hyperparameters` accept `target_label=None` for `multihead`;
`language` must still be explicitly supplied (a missing value raises before training).
Existing positional argument order is retained. `overlap_policy=None` resolves to strict
intraclass error for Multi-Head and the historical merge/keep-longest for old models.
`Encoder(..., target_label=None, entity_labels=None)` discovers types on its first training
encode. Low-level `build_model` receives `entity_labels=encoder.entity_labels` through
architecture kwargs. Low-level `training_arguments(..., architecture="multihead")` supplies
the architecture default; explicit arguments instances/selection choices are preserved.
Reconstruction/metric helpers accept an optional final `entity_labels` argument and can
read the catalogue from encoded-row attrs. The root `lab.ner` exports, `.pyi` and `tasks.py`
need no new names; existing imported functions carry their extended signatures.

Each head has three categorical classes. The loss averages each head's valid CE, then
nonempty heads; entirely masked batches produce a differentiable zero. Argmax is over the
last dimension of `[B,L,K,3]`. BIO reconstruction and softmax confidence are independent per
head; exact duplicate predictions retain the best score, with no cross-type suppression.

Strict scoring uses identity sets, so exact overlaps remain distinct by label and repeated
predictions cannot inflate TP. Macro includes all trained heads, including zero-support
heads scoring zero; unknown gold types warn, receive metrics and count as micro FN without
adding heads. Gold attrs preserve original boundaries. Fold summaries and manifests record
their actual catalogues; the holdout is never used for discovery. See
[the library guide](../docs/ner/library.md#multi-head-bio) for all metric keys and semantics.

## File inventory

| Files | Responsibility |
|---|---|
| `src/lab/ner/models/multihead.py` (new) | Shared Transformer, vectorized BIO projection, head-balanced CE, catalogue digest, checkpointing delegation |
| `src/lab/ner/models/registry.py` | Register `multihead` without changing linear/CRF factories |
| `src/lab/ner/encoding/multihead.py` (new) | Train-only label discovery and intraclass overlap resolution |
| `src/lab/ner/encoding/encoder.py` | Shared windows/inputs with multidimensional labels; catalogue and original gold metadata |
| `src/lab/ner/encoding/rows.py` | Extract reusable first-subword label projection; avoid copying shared fields per head |
| `src/lab/ner/encoding/tagging.py` | Torch-free ordered catalogue validation |
| `src/lab/ner/encoding/__init__.py` | Export discovery for direct composition |
| `src/lab/ner/training/collation.py` (new) | Vector padding, both padding sides; old collator retained for old models |
| `src/lab/ner/training/multihead.py` (new) | Selection defaults and head-aware factory kwargs |
| `src/lab/ner/training/arguments.py` | Explicit architecture default and loss minimization default |
| `src/lab/ner/training/dataset.py` | Nested integer targets in Dataset conversion |
| `src/lab/ner/training/trainer.py` | Head-order checks, minimal left-padding prediction alignment, self-contained encoding save |
| `src/lab/ner/training/assessment.py` | Per-fold training-only discovery, metrics and catalogue provenance |
| `src/lab/ner/evaluation/multihead.py` (new) | Shared-offset per-head decoding; exact per-type, macro/micro and character scores |
| `src/lab/ner/evaluation/spans.py`, `metrics.py` | Dispatch the existing public entry points; per-head token diagnostics |
| `src/lab/ner/inference/loading.py` | Strict metadata, state dimensions and catalogue-digest validation; standalone reconstruction |
| `src/lab/ner/inference/prediction.py` | Head-aware collator and decoding on the existing batching path |
| `src/lab/ner/inference/predict.py`, `io.py` | Unified multitype spans and gold scoring; optional target type annotation |
| `src/lab/ner/hpo/search.py`, `variants.py`, `trial.py`, `winner.py` | Automatic joint catalogue, macro objective, reused encoded variants and winner compatibility |
| `src/lab/ner/ensemble/io.py`, `tables.py` | Explicit rejection of unsupported Multi-Head members |
| `verification/verify_multihead.py` (new) | Offline functional, learning, integration, legacy round-trip, folds and real HPO checks |
| `verification/benchmark_multihead.py` (new) | Reproducible isolated-process CPU measurements |
| `verification/run_all.py` | Register the new behavioral script |
| `verification/README.md`, `MULTIHEAD_RESULTS.md` (new report) | Commands, results, skips and limits |
| `docs/ner/configs/multihead-{train,predict}.yaml` (new) | Executable examples; tested with local paths substituted |
| `README.md`, `docs/ner/{README,library,train-model,predict-entities,search-hyperparameters,ensemble}.md` | Extend existing user documentation |
| `docs/dev/DECISIONS.md`, `ROADMAP.md` | Append D132–D135 and record scope/remaining validation |

`lab.core`, NEL, dependency declarations and the lockfile are unchanged. D43 remains in the
historical register; D132 explicitly extends it for the opt-in architecture.

## Executed validation

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_multihead.py
.venv-verify/bin/python verification/verify_layering.py
.venv-verify/bin/ruff check src/lab/ner \
  verification/verify_multihead.py verification/benchmark_multihead.py
.venv-verify/bin/ruff format --check src/lab/ner
git diff --check
```

Multi-Head: **116 passed, 0 failed, 1 skipped** (CUDA unavailable). Includes:

- Automatic/deterministic train-only catalogue; K=1, 2 and 4; unannotated documents;
  disjoint, nested, cross-type and same-token entities; intraclass rejection; UAX words;
  real WordPiece continuations and specials; greedy/context windows; vector padding.
- `[B,L,K,3]`; one backbone forward; finite gradients; explicit head-average CE equivalence;
  empty-head exclusion; zero masked loss/gradients; activation-checkpointing backward;
  capacity forward; BERT and DistilBERT (including supplied token types on DistilBERT).
- Joint optimization learns all four overlapping types (strict macro-F1 = 1.0 on synthetic
  training fixtures). This is a learnability check, not evidence of clinical generalization.
- Exact per-type/macro/micro scores, zero-support heads, unknown gold, duplicates, scores,
  original offsets and highest-score deduplication through the existing span helper.
- Actual Trainer training, early stopping before the epoch cap, best checkpoint selection,
  saved-logit equality, standalone loading without the original backbone, explicit malformed
  mapping/reordering/dimension errors, raw-text inference and TSV/Parquet output.
- `linear` and `crf` trained/saved/reloaded on tiny BERT; their logits round-trip, keep 3D
  and retain the old selection default/metadata format. This does not claim every old
  checkpoint or backbone was tested.
- Both documented YAMLs executed after substituting fixture input/output paths; two folds
  with different catalogues and no holdout file; unknown validation-only labels count as FN.
- Real Ray/Optuna smoke plus 2 trials × 2 seeds, macro objective and unchanged `winner.yaml`
  execution through `lab run`. No separate per-label HPO.

Layering: **137 passed, 0 failed**. Explicit imports of Encoder/evaluation also leave torch
unloaded. Ruff and whitespace checks passed.

The full runner was executed as well:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/run_all.py
```

It exits **1**, not success. The latest full-run invocation included the then-current
113-check Multi-Head script, which passed; the final standalone script above adds two
checkpoint-tampering checks and a highest-confidence deduplication check, and passes 116. The existing data script passed 192 checks
with 2 missing-real-data skips; NEL passed 62 with 2 skips (external linking checkout and
uncached BERT tokenizer); layering passed 137. Nine existing scripts failed at their first
uncached BERT/GPT-2 tokenizer lookup: encoding, encoder, models, training, evaluation,
assessment, HPO, inference and ensemble. They were **not** counted as passed. Before offline
execution, an HTTPS request to the actual Hugging Face tokenizer endpoint returned proxy
HTTP 403. Offline mode avoids futile retries; it is not a substitute for those regressions.
Local legacy round-trip checks supplement but do not replace that missing full validation.

Session logs: `/workspace/multihead-check-final.log`,
`/workspace/multihead-regression-final.log`. They are local artifacts, not committed fixtures.

## CPU efficiency measurement

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/benchmark_multihead.py --repeats 1000 \
  --output /workspace/multihead-benchmark.json
```

AMD EPYC 9V45 host, 5 exposed vCPUs, 2 torch threads, x86_64 Linux; torch 2.10.0+cpu.
All scenarios: the same random 2-layer BERT, hidden size 32, batch 8, length 64, five warmups,
1,000 measured batches. Scenarios run in separate processes; four independent models are
resident together. Timings include only model forwards, excluding tokenization, decoding,
Trainer startup and I/O. Windows/sec counts input windows for completion of the scenario;
the one-head baseline produces fewer entity types than the other two.

| Scenario | Backbone forwards | Seconds | Windows/s | Tokens/s | Parameters | Classifier parameters | Peak process RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| One linear head | 1,000 | 0.6594 | 12,132 | 776,478 | 34,211 | 99 | 581.0 |
| Four shared heads | 1,000 | 0.7346 | 10,890 | 696,947 | 34,508 | 396 | 579.6 |
| Four independent models | 4,000 | 2.9586 | 2,704 | 173,052 | 136,844 | 396 | 581.5 |

The extra three heads cost 297 parameters here. In this measurement shared heads take about
4× less forward time than four models, and about 11% more than one head. This is one tiny CPU
benchmark, not a general speedup claim. Full process peaks include Python, dependencies and
model initialization; at this scale they do not establish a meaningful memory saving.
Document tokenization count is measured separately in the functional check: exactly one
`tokenize_document` call per document, regardless of head count. No clinical-model, CUDA,
end-to-end latency or sample-efficiency benchmark was run.

## Scientific references and access blocker

The four supplied URLs were requested through the configured HTTPS proxy; all returned
CONNECT HTTP 403. Their full texts/abstracts were not available for analysis in this session:

| Requested source | Evidence available |
|---|---|
| [ClinicalEncoder26AM](https://arxiv.org/abs/2605.28521) | Title and URL from the request only; paper unreviewed |
| [ClinicalAligner26AM](https://aclanthology.org/2026.smm4h-1.27/) | Title and URL from the request only; paper unreviewed |
| [I-NOWJ](https://doi.org/10.1093/database/baag056) | Title and URL from the request only; paper unreviewed |
| [NOWJ / ToxHabits](https://arxiv.org/abs/2602.09469) | Title and URL from the request only; paper unreviewed |

No mechanism, result or Multi-Head architecture is attributed to these papers. Scientific
review remains outstanding; names alone are insufficient evidence. The following are
engineering choices **in this implementation**, motivated by the requested contracts, not
claimed conclusions of those sources:

- Domain representation is selected through the existing `base_model` interface. No clinical
  adaptation method or backbone was benchmarked here.
- Boundary handling reuses UAX words, original offsets and independent BIO reconstruction;
  orphan-I repair is the existing linear policy, not a new learned structured decoder.
- Joint outputs share the encoder for computational efficiency. Improved few-shot learning
  from task sharing is a hypothesis, not a measured result.
- Existing greedy/context windows supply context; there is no adaptive context selector.
- Exact set scoring and per-head spans preserve cross-type overlap. Intraclass overlap needs
  a different output representation, not a hidden resolution across heads.
- CNN boundaries, CRFs, alignment losses, distillation, dictionaries, knowledge-guided learning
  and ensembles were intentionally not added to the minimum architecture.

The environment draft now includes `huggingface.co`, `arxiv.org`, `aclanthology.org`,
`doi.org`, `academic.oup.com`. Saving the draft did not apply/publish it. To close these
blockers, review/save network settings, retry tokenizer/source access, rerun the nine
regressions and review the papers before claiming full acceptance. Publication remains a
separate user action after validation.

## Remaining limits

Intraclass overlaps are unsupported by one BIO track; explicit lossy policies are documented.
Whole-word supervision cannot exactly express every mid-word boundary. Multi-Head parquet
encoding currently loads a partition in memory. Encoded-row attrs carry original gold;
if these are discarded, score reconstructed BIO gold or provide the original span table to
the explicit scorer. Multi-Head CRF and ensembles are deferred; existing single-target
ensembles reject these checkpoints clearly. GPU, mixed-precision GPU and distributed
execution are unverified here. Training checkpoint resumption was not added: the public
`train` orchestrator does not expose it for old models either.
