# Native sigmoid Multi-Label NER: implementation and measured results

Recorded on 2026-11-06. This report describes the local implementation and executed
checks, including checks blocked by the environment. It does not claim clinical
accuracy improvements or complete execution of the project's verification suite.

The subsequent [acceleration review](ACCELERATION_REVIEW.md) records device-local inference,
replica-policy fixes, memory collection changes and additional verification. The measurements
below describe the original feature benchmark, not GPU qualification.

## Integration and architecture

The integration map was established against the workspace, which already contained
`multihead`, `multihead_crf`, constrained word-level CRF and heterogeneous ensembles.
The existing Encoder, model registry, Trainer, fold orchestration, Ray/Optuna search,
span reconstruction, scoring and ensemble word tables are extended in place.
There is no additional task registration, CLI, corpus reader or training pipeline.

`architecture="multilabel"`, with `target_label=None`, discovers sorted entity names
exclusively from the training partition of each fold. Validation cannot add heads.
Unknown gold types are reported and count as micro false negatives; macro metrics
use the trained catalogue. Trained types with zero support remain in the macro mean.

One shared Transformer produces `[B,L,H]`; one linear projection produces
`[B,L,K,2]` raw logits. Channels are B=0 and I=1. O is implicit, with binary target
`[0,0]`; B and I targets are `[1,0]` and `[0,1]`. This differs from Multi-Head's
categorical `[B,L,K,3]` BIO classifiers. Model classification uses independent
sigmoids, with no categorical softmax over channels or entities.

The shared Encoder tokenizes a document once, reuses offsets, UAX #29 words and
windows, and converts its per-type BIO targets to binary targets once. Only first
eligible subwords receive supervision; special tokens, padding and context receive
`[-100,-100]`. `word_mask` is derived from input geometry, independently of gold.
The specialized collator pads the complete `[K,2]` label block and supports left
padding and inference without labels. Inference drops labels before the forward.

### Losses and transformations

The default loss is stable FP32 BCEWithLogits. `weighted_bce`, binary `focal` and
`asymmetric` are selectable through explicit architecture kwargs. Losses average
valid binary elements within each entity, then average active entities. Negative
examples remain supervised even if a batch contains no positive mentions. Entirely
ignored heads are excluded; entirely ignored batches return differentiable zero.

Training-derived positive weights use only encoded train rows:
`clip((negatives + smoothing)/(positives + smoothing), 1, maximum)`, independently
for each named entity and B/I channel. Defaults are smoothing=1 and maximum=20.
Explicit weights must be finite, positive and aligned by entity name. Incompatible
loss-family options are rejected instead of silently ignored.

`training_temperature` changes logits supplied to the training loss and is separate
from decoding temperature. Decoding supports sigmoid, temperature sigmoid and the
experimental hard sigmoid. Hard sigmoid is restricted to scores/decisions, not
substituted into BCE training. Selecting temperature by span F1 is decoder tuning;
it is not a demonstration of statistical calibration.

### Decoding and overlaps

`threshold_bio` uses `1-(1-p_B)(1-p_I) >= entity_threshold`. An active word begins a
mention after an inactive/missing position, or when `p_B > p_I`; ties continue an
open mention. Decisions are vectorized over entities and valid word positions.

`constrained_viterbi` reuses the common BIO2 chain with fixed zero transitions.
Its local O/B/I scores are stable Bernoulli log-potentials, with an optional entity
bias. START→I and O→I are excluded. This is constrained decoding, not a trained CRF.
Both decoders return `[B,L,K]` tags and reuse the existing per-type span decoder.

Cross-type overlapping, nesting and simultaneous labels are retained. Same-type
overlapping gold is rejected explicitly: a single BIO chain cannot represent it.
Only exact `(document,label,start,end)` duplicates are removed, retaining the
maximum score across windows. There is no global NMS or overlap-based merging.

For confidence and soft ensembles, legal state weights are
`q_O=(1-p_B)(1-p_I)`, `q_B=p_B(1-p_I)`, `q_I=(1-p_B)p_I`, normalized in log space.
This conditions independent Bernoulli outputs on excluding simultaneous B+I;
it is an adapter distribution, not the model's original binary outputs. A span's
score is the mean adapted probability of its selected BIO states. It is neither
the probability of a whole mention nor guaranteed calibrated. Hard ensembles use
the configured decoder's tags. Coverage and normalization remain per entity, so
unsupported labels cannot receive negative votes from specialist members.

## Public API, selection and persistence

- Existing `build_model`, `train`, `train_model`, `predict_logits`,
  `predict_entities`, evaluation, HPO and ensemble entry points remain in use.
- New architecture kwargs describe loss, weighting, transformation and decoder
  separately; full signatures and examples are in `docs/ner/library.md`.
- Low-level `Encoder` adds `label_schema="multilabel"`; orchestration selects it
  automatically for the new architecture. Existing `bio` behavior remains default.
- Evaluation accepts an explicit typed `multilabel_decoding` override. Trainer
  checks agreement between the metric decoder and model decoder.
- Default selection is `span_strict_macro_f1`; explicit micro F1, named-type F1,
  other supported metrics and training `eval_loss` selections are retained.
- Strict metrics use original gold when available, report TP/FP/FN and P/R/F1 by
  type, macro and micro aggregates, characters and binary token diagnostics.
  Unrepresentable known-type gold caused by word/window boundaries is counted and
  warned about rather than removed from the original-gold denominator.
- New model metadata uses encoding format 3, ordered catalogue, head mapping,
  binary channel schema, loss/decoder settings, encoder geometry and standalone
  backbone configuration. Loading validates schema, dimensions, catalogue digest
  and positive-weight buffers. Older formats and architectures keep their routes.
- `predict_entities(..., decoder_path=...)` optionally applies a checkpoint-bound
  decoder artifact. It never fits thresholds during inference.

Native training HPO accepts typed `loss_variants`; family-specific options are
sampled conditionally and routed to architecture kwargs, not TrainingArguments.
`winner.yaml` includes the selected loss and saves the Multi-Label checkpoint.
This save default is specific to the new architecture.

The same `search_hyperparameters` entry point supports decoder-only selection via
`model_dir` and `decoder_search_space`, without `base_models`. It loads/encodes once,
runs one Transformer inference and reuses cached logits for every Optuna trial.
Thresholds and temperatures can be global or named by entity; insufficient unique
gold support preserves a deterministic checkpoint fallback (default minimum 5).
The baseline is retained if no candidate improves it. Temperature search is omitted
for Viterbi: positive temperature alone cannot change its best path and can trade
off with state bias. Checkpoint-fixed temperature/bias settings remain explicit.

Calibration provenance uses existing manifests: proven training overlap is rejected,
and unprovable separation is warned about. Selection reads validation, never test.
Artifacts include trial traces, support, metric, provenance and checkpoint/tokenizer
fingerprint. Modified checkpoints cannot consume an old decoder artifact silently.
Selected settings require revalidation after retraining on changed weights/data.

Ensembles consume checkpoint-persisted decoding settings. An external decoder-path
overlay is currently an individual-inference option; it is not automatically applied
to ensemble members. Existing ensemble formats and public methods are retained.

## Files and responsibilities

New source modules:

| Module under `src/lab/ner/` | Responsibility |
| --- | --- |
| `models/multilabel.py` | Shared-backbone binary model and serialized options |
| `models/binary_loss.py` | Masked losses and train-only positive weights |
| `models/binary_decoding.py` | Typed score transforms, decisions and O/B/I adapter |
| `training/binary.py` | Train/metric configuration binding |
| `evaluation/multilabel.py` | Binary-to-span evaluation and representation diagnostics |
| `inference/binary_settings.py` | Bound decoder artifact loading/fingerprints |
| `hpo/binary.py` | Conditional loss-family search validation |
| `hpo/decoding.py` | Pure cached-logit decoder fitting |
| `hpo/decoder_search.py` | Decoder mode inside native HPO orchestration |

Extended source modules:

| Modules under `src/lab/ner/` | Change |
| --- | --- |
| `models/registry.py`, `models/multihead.py` | Registry and reusable private projection |
| `encoding/encoder.py` | Binary schema and shared structural word mask |
| `training/dataset.py`, `collation.py`, `trainer.py` | Nested targets, padding, raw predictions and saving |
| `training/assessment.py`, `multihead.py` | Fold discovery, weights and selection defaults |
| `evaluation/metrics.py`, `spans.py`, `multihead.py` | Dispatch and shared tag-based reconstruction |
| `inference/loading.py`, `prediction.py`, `predict.py` | Strict load, no-loss inference, decoder override |
| `hpo/search.py`, `trial.py`, `variants.py`, `winner.py` | Native search, profiles and reproducible winners |
| `ensemble/tables.py` | Binary member adapter to existing word-table contract |

No `lab.core`, NEL or CLI task implementation was changed. Documentation updates
cover README, all relevant NER guides, decisions D146–D152 and the roadmap. Seven
YAML examples live in `docs/ner/configs/`: `multilabel-train`, `multilabel-focal`,
`multilabel-asymmetric`, `multilabel-hpo`, `multilabel-decoder-hpo`,
`multilabel-predict` and `ensemble-multilabel`. Their paths are templates for local
assets; verification binds their actual signatures and executes all seven through
the existing CLI with local fixture paths and small budgets.

`verification/verify_multilabel.py` adds functional and mathematical checks;
`verify_models.py` recognizes five built-ins; `run_all.py` registers the new checks.
The existing `benchmark_crf.py` is extended with explicit Multi-Label variants;
its previous default comparison remains available. Raw measurements are committed
in `multilabel_benchmark_cpu.json`.

## Executed verification

Commands used the local `.venv-verify`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and
`HF_HUB_OFFLINE=1`. Ray verification required the environment's socket permission.

```sh
.venv-verify/bin/python verification/verify_multilabel.py
.venv-verify/bin/python verification/run_all.py
.venv-verify/bin/ruff check src verification
.venv-verify/bin/ruff format --check src
git diff --check
```

The final focused Multi-Label run: **348 passed, 0 failed, 1 CUDA skip**. It includes
numeric loss references, finite differences, extreme logits, empty masks, CPU
fp16/bf16, explicit decoder paths, Viterbi exhaustive enumeration, word/subword
geometry, overlaps, duplicate windows, BERT/DistilBERT, all four losses through
Trainer and round-trip, early stopping, folds, actual Ray trials and seeds, cached
decoder fitting, provenance rejection, mixed ensembles, CLI YAML and TSV/Parquet.

The full runner was executed, and exited nonzero. Completed scripts reported:

| Script | Passed | Skipped |
| --- | ---: | ---: |
| layering | 157 | 0 |
| data | 192 | 2 unavailable corpus assets |
| models | 40 | 0 |
| ensemble | 207 | 0 |
| NEL | 62 | 2 unavailable reference assets/encoders |
| multihead | 116 | 1 CUDA |
| chain | 383 | 1 CUDA |
| word CRF | 204 | 1 CUDA |
| multilabel, at full-run snapshot | 338 | 1 CUDA |

Subsequent focused checks increased Multi-Label coverage to 348. Seven legacy
scripts could not complete because their Hugging Face tokenizers were not cached
and outbound model downloads are unavailable: `verify_encoding.py`,
`verify_encoder.py`, `verify_training.py`, `verify_evaluation.py`,
`verify_assessment.py`, `verify_hpo.py`, `verify_inference.py`. These are **blocked
executions, not passes**. Local-fixture checks exercise the corresponding new paths
but do not substitute for claiming those legacy scripts passed.

## Reproducible CPU benchmark

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/benchmark_crf.py \
  --variants linear_argmax multihead_argmax multilabel_bce \
  multilabel_weighted_bce multilabel_focal multilabel_asymmetric multihead_crf \
  --output verification/multilabel_benchmark_cpu.json
```

AMD EPYC 9V45 CPU, 5 available vCPUs, 2 Torch threads, Torch 2.10.0+cpu; no CUDA.
Random BERT H32, two layers, two attention heads; shared initialization, dropout=0,
AdamW lr=.005, weight decay=.01, clipping=1, 40 full-batch steps. Six training and
three validation documents, padded length 16, configured maximum 32; seeds 21/22/23.
Each fresh worker measures 30 repetitions after 3 warmups. All 21 runs recorded
34/34 expected backbone forwards: one per batch, irrespective of entity count.

Means over three seeds:

| Variant | Parameters | Full-task macro/micro F1 | Forward ms | Decoder ms | Train 40 steps s | Tokens/s | Peak RSS MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Linear argmax | 34211 | .2500 / .4000 | .3753 | .0013 | .0978 | 61435 | 596.3 |
| Multi-Head argmax | 34508 | 1 / 1 | .3977 | .0018 | .0980 | 57841 | 597.7 |
| Multi-Head CRF | 34568 | 1 / 1 | .4318 | .2005 | .1487 | 36377 | 599.1 |
| Multi-Label BCE | 34376 | 1 / 1 | .4239 | .0482 | .1082 | 48784 | 597.8 |
| Multi-Label weighted BCE | 34376 | 1 / 1 | .3749 | .0434 | .1006 | 55622 | 597.7 |
| Multi-Label focal | 34376 | .9667 / .9608 | .4699 | .0458 | .1073 | 46622 | 598.4 |
| Multi-Label asymmetric | 34376 | .9500 / .9412 | .4025 | .0446 | .1109 | 51660 | 599.1 |

The full-task denominator contains the four training-derived types. The classic
baseline supports only SYMPTOM; its full-task scores include unsupported types as
false negatives. Its specialist-only metrics are also recorded and are a different
scope. The backbone has 34112 parameters; binary output adds 264 versus 396 for
categorical Multi-Head. No replicated backbone is allocated for extra heads.

Raw records contain precision/recall, per-type F1, throughput, per-seed measurements
and cached comparisons of sigmoid thresholding, temperature sigmoid, hard sigmoid
and constrained Viterbi. Those comparisons reuse emissions without another forward.
Decoder timing covers tag decisions, not span reconstruction or I/O. Training timing
covers optimization steps, not Trainer orchestration. RSS includes Python imports,
model and optimizer; it is not incremental tensor memory or peak VRAM. Throughput
uses non-padding Transformer tokens. Submillisecond CPU timings are noisy.

Repeated synthetic templates are a correctness/compute fixture, not independent
clinical generalization evidence. These measurements establish neither superiority
of any loss nor a universal speed advantage. Large-scale GPU and clinical-corpus
comparisons remain unperformed.

## Research access and remaining limits

The requested PMC, OUP, CVF and arXiv article endpoints were attempted but returned
proxy CONNECT 403 in this environment. Their full text could not be reviewed here.
The W2NER GitHub README was accessible: its word-word NNW/THW representation
addresses nested/discontinuous extraction, a different capacity from binary BIO.
No external implementation was copied. Formula-based loss and decoding choices
are documented as project implementation choices, not attributed as this exact
architecture to inaccessible papers.

Remaining limits include arbitrary overlapping spans within one type, discontinuous
entities, within-word boundaries not representable by word BIO, unavailable GPU and
clinical-backbone validation, and the seven blocked legacy scripts. Channel-specific
thresholds and likelihood-fitted temperature calibration are not implemented.
Decoder-only orchestration reconstructs persisted built-in window strategies;
custom callable strategies need caller-managed encoded rows for pure decoder fitting.
Post-hoc settings require independent validation and revalidation after retraining.
