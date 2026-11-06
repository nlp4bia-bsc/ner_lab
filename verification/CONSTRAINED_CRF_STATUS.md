# Word-level constrained CRF: implementation and executed evidence

The native registry now supports `linear`, `crf`, `multihead`, and `multihead_crf`.
New `crf` uses version 2; historical checkpoints automatically retain the version-1 class.
No second training, inference, ensemble or HPO pipeline was introduced. No dependency added.

## Audit and integration map

The workspace already contained shared-backbone Multi-Head BIO and unified ensembles.
The historical CRF in `models/crf.py` clamps invalid parameters through `.data`, registers
hooks, replaces ignored gold by O and uses Transformer attention as its chain mask. That
class remains solely for explicit legacy construction and historical checkpoint loading;
it was not silently reinterpreted under word-level semantics.

The extension reuses Encoder tokenization/UAX word IDs/windows, the Multi-Head backbone and
projection, canonical spans and exact multitype metrics, the existing Trainer orchestration,
folds, checkpoint ownership, Ray/Optuna trials, and sparse per-label ensemble word tables.
D141–D145 document the new contracts without rewriting earlier decisions.

## Architecture, constraints and loss

`ConstrainedChain` accepts `[B,T,C]` or `[B,T,K,C]`, with independent start `[K,C]`,
transition `[K,C,C]` and end `[K,C]` parameters. Integrated BIO heads use O=0, B=1, I=2.
The pure BIO helper also supplies flat B-X/I-X restrictions to the generic chain core.
For a path, score is start + emissions + successive edges + end. NLL is legal-path
log-partition minus gold score. The same boolean masks apply exact `-inf` to START→I,
O→I and cross-type inside transitions in partition, Viterbi and forward-backward.
Ending on B is legal in BIO2. Safe unreachable-state reductions avoid undefined gradients.

Sensitive DP runs in FP32 with autocast disabled. Batch/head work is vectorized; only the
necessary time recursion/backtrace loops remain. The integrated model emits `[B,L,3]` or
`[B,L,K,3]`, and one Transformer forward produces all heads. `forward` returns emissions,
not pseudo decisions. `StructuredPrediction` holds restored Viterbi tags and optional
log marginals. Marginals are computed only when requested and reuse those emissions.

NLL is summed over sequences per head, divided by its supervised word count, then averaged
with explicit named nonnegative head weights over active heads. Entirely empty batches
return differentiable zero. Auxiliary emission CE defaults to zero; `aux_ce_weight=0.3` is
an optional ablation. `loss_type="cross_entropy"` freezes zero chain parameters and trains
only local CE, providing a constrained BIO decoding baseline without trainable CRF weights.
Only `loss_reduction="word_mean"` and `tag_scheme="BIO2"` are supported.

## Masks, windows and overlapping

New CRFs require structural `crf_word_mask`, independent of Transformer attention and gold
values. It selects eligible first subwords in the core; specials, secondary subwords,
padding and context flanks do not become CRF O states. Indexed batched compaction preserves
original positions and gradients. Decoding restores `-100` at absent positions; absent
marginals have log probability `-inf`, hence probability zero.

New window geometry uses text alone. Classic word CRFs filter to the target type before
overlap resolution. Multi-Head keeps independent tracks and cross-type overlapping spans.
The existing window-local tagger starts a clipped mention with B at a new core. An entity
crossing cores can therefore become fragments; this is an explicit modelling limitation,
not a repair in the CRF loss. Legacy window/mask semantics stay intact for old checkpoints.

Complete supervision is required at eligible words. Missing whole-word gold and illegal
BIO2 fail before training with document/window/type/token/transition diagnostics. The core
reports batch/word/head coordinates for direct calls. Unknown whole words are never removed
to connect independent gold segments. Partial annotation is not implemented.

BIO voting still cannot express arbitrary same-type overlaps. Multi-Head defaults to an
intraclass conflict error; existing explicit lossy encoding policies remain explicit.
Exact span ensembles can preserve different overlapping same-type intervals from sources.
No cross-type NMS or competition is introduced.

## Integration and compatibility

- `build_model(..., architecture="crf")` defaults to version 2;
  `multihead_crf` requires the training-discovered ordered catalogue.
- `train_model` constructs the structural encoder automatically. Low-level callers must add
  `include_word_mask=True` to `Encoder` and retain that field in their rows. **This is a
  deliberate migration requirement for new low-level CRF training**; old construction can
  use `architecture_kwargs={"crf_version": 1}` with classic `crf`.
- `WordCRFTrainer` validates and selects checkpoints from Viterbi metrics. Classic defaults
  remain strict F1; Multi-Head defaults to strict macro-F1. Explicit metrics and loss
  minimization are preserved. Existing early stopping, accumulation and best-model saving
  remain in the shared orchestrator.
- The historical `predict_logits` contract transports Viterbi decisions as compatibility
  logits. `WordMarginalTrainer` transports log marginals and decisions in separate last-axis
  blocks. Inference uses real marginal confidence, never softmax of those pseudo decisions.
- `predict_entities` strips gold for new CRF inference, giving identical spans and scores
  with/without annotations, and returns canonical TSV/Parquet spans with exact identity
  deduplication. Mean selected-tag marginal score is not full mention probability or a
  calibration guarantee.
- All ensemble families consume the same word adapter: legacy/new CRF hard voting uses
  Viterbi; soft voting uses true marginal distributions. All heads come from one member
  inference. Unsupported labels remain absent evidence. Existing format-1 ensembles load.
- Ray/Optuna shares encoded variants with eligibility masks. Supported namespaced dimensions
  `architecture__dropout`, `architecture__aux_ce_weight`,
  `architecture__constrain_transitions`, `architecture__loss_type`,
  `architecture__loss_reduction`, `architecture__tag_scheme` route separately from
  TrainingArguments. Invalid bounds/schemes and incompatible CE/auxiliary families fail
  before trials. Architecture stays fixed per sweep. Winners replay as ordinary
  `architecture_kwargs`; no manual editing is needed.
- New `encoding.json` format 2 records `crf_version=2`, ordered labels/count/mapping, BIO
  schema, structural mask policy, full backbone configuration and exact model options.
  Weights contain learned transitions normally. Loading checks head digests, dimensions,
  restriction buffers and named weights against metadata before inference. Original base
  model files need not exist. Legacy CRF weights and semantics use the original class.

## Verification commands and results

All commands used `.venv-verify` (Python 3.12, PyTorch 2.10.0+cpu, transformers 5.22.2):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_chain.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_word_crf.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_multihead.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_ensemble.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/run_all.py
.venv-verify/bin/ruff check src verification
.venv-verify/bin/ruff format --check src
```

Ray tests require local sockets; the restricted command sandbox was lifted for those test
commands. They still used the managed network policy and local tiny checkpoints.

| Verification | Executed result |
|---|---|
| Exhaustive CRF mathematics | 383 passed; 1 CUDA skip |
| Word-CRF integration, final targeted run | 204 passed; 1 CUDA skip |
| Existing Multi-Head | 116 passed; 1 CUDA skip |
| Unified and classic ensembles | 207 passed |
| Models/factory/capacity | 40 passed |
| Layering / core dependency isolation | 148 passed |
| Corpus/data | 192 passed; 2 missing-corpus skips |
| NEL | 62 passed; 2 skips (reference checkout, uncached external model) |
| Ruff lint and source format | Passed |

The exhaustive oracle enumerates every legal three-state path for lengths 1–5, K=1/2/4,
comparing gold score, partition, NLL, Viterbi, marginals and partition derivatives with
`atol=2e-5`. It also covers large illegal-state emissions, flat cross-type constraints,
padding/empty chains, invalid gold and CPU bf16/fp16 autocast.

Integration checks cover independent-chain equivalence, K=1/2/4, compact/restored gradients,
empty heads, named weights below unit total, unexpected supervision holes, greedy/context
annotation independence, special/subword masks, a backbone with token types (BERT) and one
without (DistilBERT), capacity forward, CPU autocast forward/backward, gradient accumulation,
early stopping, best checkpoint, standalone save/load, metadata tampering, real marginal
scores, TSV/Parquet, all voting/span families and a five-member mixture of linear,
legacy CRF, Multi-Head (permuted heads), new classic CRF and Multi-Head CRF. Real Ray/Optuna
runs two trials × two seeds and replays winner YAML. Fold checks remove the holdout file,
exercise unknown validation labels and aggregate strict macro scores.

`run_all.py` **does not pass overall** in this environment: seven original scripts abort
because `bert-base-uncased` is unavailable in cache and Hugging Face is blocked:
`verify_encoding.py`, `verify_encoder.py`, `verify_training.py`, `verify_evaluation.py`,
`verify_assessment.py`, `verify_hpo.py`, `verify_inference.py`. These are blocked runs,
not successful tests or silently skipped regressions. New offline integration does cover
those paths for the new architecture, but does not substitute for those entire scripts.
The existing model-factory check was made offline with the shared tiny tokenizer and its
architecture tuple updated. Legacy CRF fixture checks explicitly select version 1.
CUDA training/inference/precision and clinical-backbone compatibility are not verified.

## Reproducible CPU ablations

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/benchmark_crf.py \
  --output verification/crf_benchmark_cpu.json
```

AMD EPYC 9V45, five available vCPUs, two PyTorch threads, CPU only. The
[raw JSON](crf_benchmark_cpu.json) contains all 27 isolated process results: nine variants ×
seeds 21/22/23. Random BERT: H=32, two layers, two attention heads. Identical initial backbone,
classifier initialization within each task family, dropout disabled, AdamW lr=0.005,
weight decay=0.01, 40 full-batch steps, six training documents and three validation documents,
max length 32, padded sequence length 16. Timing: three warmups, 30 repeats, validation batch
of three. Each member forward executes its backbone once; emission reuse adds no calls
when computing Viterbi/marginals (34 actual = 30 repeats + 3 warmups + 1 retained forward).

Timing means, three seeds; decoding with marginals includes Viterbi and forward-backward:

| Variant | Train s / 40 steps | Forward ms | Decode ms | Decode + marginals ms | Windows/s | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| `crf_auxiliary` | 0.1616 | 0.4336 | 0.1952 | 0.5529 | 4777 | 600.0 |
| `crf_masked` | 0.1447 | 0.4267 | 0.2042 | 0.5496 | 4759 | 599.1 |
| `crf_unconstrained` | 0.1304 | 0.3562 | 0.1755 | 0.5361 | 5650 | 598.9 |
| `linear_argmax` | 0.0936 | 0.3663 | 0.0018 | 0.0000 | 8172 | 597.9 |
| `linear_constrained` | 0.1079 | 0.3943 | 0.1798 | 0.6152 | 5241 | 599.3 |
| `multihead_argmax` | 0.1003 | 0.3836 | 0.0018 | 0.0000 | 7842 | 597.6 |
| `multihead_constrained` | 0.1043 | 0.3767 | 0.2356 | 0.5764 | 4941 | 599.7 |
| `multihead_crf` | 0.1501 | 0.3918 | 0.1989 | 0.5733 | 5097 | 599.4 |
| `multihead_crf_auxiliary` | 0.1351 | 0.3449 | 0.1785 | 0.5508 | 5733 | 599.9 |

All nine reach strict F1=1 on the repeated synthetic template; seed dispersion of F1 is
zero and decoded illegal-transition count is zero. Classic rows evaluate SYMPTOM only;
Multi-Head rows evaluate four types. These tasks are not comparable as effectiveness
rankings. Repeated text across document IDs is intentional in this mechanical fixture and
**does not establish clinical generalization, independent-text validation or CRF superiority**.

Peak RSS includes Python/native imports, model, optimizer and training allocations in each
separate worker process; it is not incremental tensor memory or VRAM. Raw reports include
parameter counts, per-type precision/recall/F1, throughput and complete timing scopes.
Classic CRF adds 15 parameters; four-head CRF adds 60 (34,568 total versus 34,508 for
Multi-Head linear). Frozen-transition baselines add no trainable CRF parameters. The measured
structured decoder costs more than argmax on this tiny CPU model. Small latency differences
between other variants are noisy and not evidence of superiority. No GPU throughput claim.

## Files and responsibilities

| File or group | Responsibility |
|---|---|
| `models/chain.py` (new) | Common constrained partition, gold score, NLL, Viterbi, marginal DP. |
| `models/packing.py` (new) | Batched structural compaction, restoration and complete-gold checks. |
| `models/word_crf.py` (new) | Shared Transformer composition, typed decisions, optional CE and named weights. |
| `models/registry.py` | New native architecture, default new CRF and explicit legacy version selection. |
| `encoding/encoder.py` | Structural eligibility, annotation-independent new windows, target filtering. |
| `encoding/supervision.py` (new) | Gold validation with document/window/type diagnostics. |
| `training/dataset.py`, `collation.py` | Retain mask column; independently pad attention, labels and eligibility. |
| `training/trainer.py` | Viterbi evaluation, optional marginal transport and exact model metadata saving. |
| `training/assessment.py`, `multihead.py` | New CRF encoding/folds, head discovery, macro defaults and persistence. |
| `evaluation/spans.py`, `multihead.py` | Accept real probability scores independently of categorical decisions. |
| `inference/loading.py` | Standalone version-2 reconstruction/validation and legacy loading. |
| `inference/prediction.py`, `predict.py` | Real marginal confidence and annotation-independent new CRF prediction. |
| `ensemble/tables.py` | Reuse structured decisions/marginals from one member inference. |
| `hpo/search.py`, `trial.py`, `variants.py`, `winner.py` | Eligibility variants, architecture dimensions, early validation and winner replay. |
| `verification/verify_chain.py`, `verify_word_crf.py` (new) | Mathematical and offline end-to-end verification. |
| `verification/benchmark_crf.py`, `crf_benchmark_cpu.json` (new) | Reproducible nine-way CPU ablation and raw evidence. |
| `verification/run_all.py`, `README.md` | Register and document the new verifications. |
| `verify_models.py`, `verify_multihead.py`, `verify_unified_ensemble.py` | Update architecture expectations and explicitly retain legacy CRF fixture coverage. |
| `docs/ner/library.md`, `train-model.md`, `predict-entities.md`, `search-hyperparameters.md`, `ensemble.md`, `README.md` | Architecture, masks, migration, native usage and limits. |
| `docs/dev/DECISIONS.md`, `ROADMAP.md` | D141–D145 and completed/future scope. |
| Seven `docs/ner/configs/*crf*.yaml` templates (new) | Classic/Multi-Head training, auxiliary ablation, both HPO families, prediction and hybrid ensemble. |

All source paths above are under `src/lab/ner` unless otherwise stated. `lab.core`, NEL and
the task registry/CLI logic were not changed. `__init__.py` / stub contracts reuse the same
public functions; selection remains an architecture string plus explicit kwargs.

## Scientific sources and remaining work

All eight requested advanced-CRF/reference endpoints were attempted through the configured
proxy: [Wei et al.](https://aclanthology.org/2021.naacl-main.163/),
[Lester et al.](https://aclanthology.org/2020.findings-emnlp.166/),
[AdaBioBERT](https://aclanthology.org/2025.bionlp-1.6/),
[DTranNER](https://pmc.ncbi.nlm.nih.gov/articles/PMC7014657/),
[AllenNLP](https://docs.allennlp.org/v2.7.0/api/modules/conditional_random_field/),
[pytorch-crf](https://pytorch-crf.readthedocs.io/en/latest/),
[edsnlp](https://aphp.github.io/edsnlp/), and
[pytorch-struct](https://nlp.seas.harvard.edu/pytorch-struct/model.html).
Every CONNECT request received proxy HTTP 403. **No full-text review or paper-specific
implementation claim is made.** The implementation follows the explicit mathematical
formulation in the task and is independently checked by exhaustive enumeration. No external
implementation was copied. Auxiliary CE and constrained-CE decoding are controllable
experiments, not claims that these papers implement this Multi-Head design.

Remaining optional research: BIOES/BILOU, partial-supervision partitions, dynamic transitions,
probability calibration, OOF stacking, compile/kernel optimizations and clinical evaluations.
Dynamic compact width currently requires a scalar host synchronization, and safety checks
also synchronize when used on GPU; GPU optimization is unmeasured. The current CPU benchmark
is not evidence about GPU peak memory, speed, calibration or accuracy on clinical corpora.
