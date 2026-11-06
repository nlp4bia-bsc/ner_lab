# Acceleration and parallel-execution review

Reviewed on 2026-11-06 against `ea8c4f1`, including the previously added Multi-Head,
word CRF, unified ensembles and independent binary Multi-Label functionality.

## Conclusion and evidence boundary

The tensor implementations share a Transformer and vectorize over batches and heads.
They do not create a Transformer per entity. This review found and fixed execution,
memory and accounting issues; the previous implementation was not fully optimized.

The current environment has Torch **2.10.0+cpu**, no CUDA build, no CUDA devices and
no `nvidia-smi`. CPU verification and mock device checks cannot certify GPU execution,
multi-GPU equivalence, VRAM or GPU speed. There is no defensible claim of universal
optimality. Actual CUDA qualification remains outstanding.

## Findings and corrections

| Finding | Correction | Compatibility |
| --- | --- | --- |
| Copying TrainingArguments for a fold reconstructed device state, losing the one-GPU pin | Reapply the explicit device policy after copying each fold; inference argument copies retain the replica count | Restores `devices=1` to its documented behavior |
| Selecting `devices="all"` after pinning the same arguments left the Trainer flag pinned | Set the visible replica count explicitly in both policy branches | Public choices remain 1/all |
| Inference unconditionally requested FP16 on CUDA | Use the existing precision resolver: BF16 when supported, otherwise FP16 | An explicit TrainingArguments instance remains authoritative |
| Multi-Label native inference/ensemble decoding started after the complete raw-logit array reached CPU | `BinaryPredictionTrainer` computes tags and conditional probabilities on the inference device, batchwise, before transfer | Model forward and ordinary predict_logits still return raw `[B,L,K,2]` logits |
| Prediction accumulation could retain corpus-sized outputs on GPU and repeatedly concatenate growing tensors | Default inference offloads every eight batches, collects chunks, then allocates/pads one final host array | Explicit inference arguments keep their accumulation policy; final return remains a dense NumPy array |
| Word compaction and restoration used dynamic `nonzero`, which synchronizes CUDA | Use unique-index scatter/gather with no dynamic nonzero selection | Exact positions, absence values, gradients and checkpoint tensors are retained |
| Gold transition scores launched an operation per word | Gather all observed transitions and reduce once across time | Numerically equivalent NLL, within FP32 tolerance |
| Partition-only loss retained an unused Python alpha-history list | Retain history only for forward-backward | Mathematical distribution unchanged; autograd still retains what it needs |
| The marginal-only API also ran Viterbi | Compact once and compute marginal-only output directly | Structured requests for both outputs still return both |
| VRAM peaks described only the current GPU | Reset/report allocator peaks separately for participating device IDs | Existing primary-GPU fields remain; flat per-device fields are appended |
| `device="cuda:1"` could load on GPU 1 and then run the Trainer on GPU 0 | Reject nonzero indexed devices in Trainer inference; validate CUDA availability/count for indexed loads | Select physical GPUs with CUDA_VISIBLE_DEVICES before process start; standalone load_model still supports indexed CUDA |

Decisions D153–D156 document these deliberate changes without rewriting history.
The modified source modules are `models/chain.py`, `packing.py`, `word_crf.py`;
`training/arguments.py`, `devices.py`, `assessment.py`, `tracking.py`, `trainer.py`;
`inference/loading.py`, `prediction.py`, `predict.py`; `evaluation/multilabel.py`;
and `ensemble/tables.py`. No dependency, core/NEL contract, CLI task or checkpoint
parameter schema was added.

### Binary prediction transport

The internal scored-inference transport has six channels: raw B/I logits, three
adapted O/B/I probabilities and a decoded tag. Existing span reconstruction reads
the supplied decisions and does not decode them again on CPU. Retaining raw logits
allows binary token diagnostics and explicit decoder overrides without another
Transformer pass. This transport is larger than raw logits; bounded collection
limits device accumulation, but final host memory still grows with windows and K.
The ordinary raw-logit API and cached decoder-search inference remain two channels.
Neither categorical softmax nor one-hot Viterbi confidence replaces binary outputs.

## What runs in parallel, and where

| Stage | Execution and remaining limit |
| --- | --- |
| Tokenization, UAX words, windows and offsets | CPU; one shared encoding per document, not per entity |
| Transformer and projection | Device tensors; one backbone invocation per batch/replica |
| Multi-Head CE and binary loss families | Vectorized batch/head reductions; sensitive binary calculations in FP32 under AMP |
| Word CRF loss, Viterbi and marginals | Device tensors, parallel batch/heads; time DP and traceback remain sequential |
| Binary inference/ensemble decisions | Device-local batched threshold or constrained Viterbi and probability adaptation |
| Training validation and decoder-only HPO | Raw logits gathered once; host vectorized decoding/scoring. Not an all-CUDA optimization loop |
| Canonical span reconstruction and scoring | CPU/Pandas/NumPy using original offsets |
| Ensemble members | Sequential loading/inference/release to bound model VRAM |
| Ensemble combination | Sparse coverage-aware CPU/NumPy voting by entity; no unnecessary unsupported Cartesian product |
| Training on one/multiple GPUs | Existing single-process Trainer policy: one GPU by default, explicit DataParallel with devices="all" |
| Training HPO | Ray schedules independent trials, each reserving at most one GPU; seeds of a trial run sequentially |

Multi-head tensor parallelism, DataParallel replication and concurrent Ray trials
are distinct. Enabling every GPU changes effective batch size and optimizer-step
schedule. With DataParallel, replica-local normalized losses are averaged; with
unequal supervision across replicas this is not mathematically identical to a
single globally token-normalized batch. The equivalence check deliberately uses
equally supervised shards. Multi-process DDP/FSDP orchestration, rank-safe file
artifacts and globally normalized uneven-shard objectives are **not qualified**.

Strict finite/gold/mask checks still synchronize when Python must raise an error.
Compact width still uses one scalar extraction per packing call. Eliminating these
would require a different validation/static-shape contract and GPU measurements;
the review does not hide their cost. AMP is the intended mixed-precision path;
wholesale conversion of model/CRF parameters with `.half()` is not qualified here.

## Executed verification

All commands used `.venv-verify/bin/python`, with `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`, and `HF_HUB_OFFLINE=1` for model verification. Ray requires local
sockets; its verification commands ran with that permission.

```sh
python verification/verify_acceleration.py
python verification/run_all.py
python verification/verify_acceleration.py --require-cuda
ruff check src verification
ruff format --check src
git diff --check
```

The focused final acceleration check reports **118 passed, 0 failed, 2 skips**
(CUDA and DataParallel). It exercises all five model architectures on CPU, one
backbone call, gradients, clipping, non-reentrant checkpointing, CPU BF16 binary
losses and CRF, accumulation, both binary decoders, left/right alignment, variable
batch collection, native training/save/load, mask/empty-chain gradients, absence
semantics and marginal normalization. Device policy and per-GPU accounting mocks
verify bookkeeping only, not real GPU behavior.

The CUDA-required command exits **2**, with an explicit missing-CUDA error. That
is an expected qualification refusal, not a passed GPU test. On CUDA hardware the
same script additionally exercises FP16/BF16, native Trainer AMP/gradient scaling,
device-local decoding and round-trip. With two devices it checks all five model
families through DataParallel, one forward per replica and equal-shard equivalence.
Those hardware branches have **not run in this environment**.

The full runner was executed after the changes and reports:

| Script | Passed | Skips |
| --- | ---: | --- |
| layering | 157 | 0 |
| data | 192 | 2 missing corpus assets |
| models | 40 | 0 |
| ensemble | 207 | 0 |
| NEL | 62 | 2 missing reference assets/encoders |
| Multi-Head | 116 | 1 CUDA |
| constrained chain | 383 | 1 CUDA |
| word CRF | 204 | 1 CUDA |
| Multi-Label | 348 | 1 CUDA |
| acceleration, at full-run snapshot | 113 | 2 CUDA/DataParallel |

The final focused run adds inference-copy replica-count and tracker-constructor checks
(118 above).
The full runner still exits nonzero: `verify_encoding.py`, `verify_encoder.py`,
`verify_training.py`, `verify_evaluation.py`, `verify_assessment.py`, `verify_hpo.py`
and `verify_inference.py` require unavailable Hugging Face tokenizers. These seven
scripts are blocked executions, not passes. Native local-fixture coverage is useful
but does not establish completion of the blocked regressions.

## Measured CPU kernel comparison

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python verification/benchmark_acceleration.py \
  --baseline ea8c4f1 --device cpu --repeats 100 \
  --output verification/acceleration_benchmark_cpu.json
```

The benchmark loads the two relevant baseline modules from the specified local Git
revision, uses identical emissions/masks and validates every output against the new
kernel. Fixture: `[8,128,6,3]`, 25% masked positions, one empty batch row, FP32,
two CPU threads, five warmups, 100 repetitions, no gradients or Transformer.
The committed JSON contains exact wall times, numerical-equivalence checks and
profiler operation counts. CPU allocator/VRAM metrics are not fabricated.

| Kernel | Baseline ms/call | Revised ms/call |
| --- | ---: | ---: |
| Compact and restore | 0.2841 | 0.1289 |
| Observed gold score | 1.0092 | 0.1570 |
| NLL | 4.8541 | 3.7461 |
| Viterbi | 2.5051 | 2.4717 |
| Marginals | 8.0705 | 8.0532 |

Compaction/restoration removes both `nonzero` operations (2→0) while retaining one
compact-width scalar extraction. Gold scoring no longer contains a time loop.
Partition, gold, NLL, Viterbi, marginals and restoration match the baseline within
`atol=rtol=2e-5`. The exhaustive chain verification remains green.

These are kernel measurements on a shared cloud CPU, not end-to-end throughput,
clinical accuracy or GPU speed evidence. Output-scored Multi-Label transport trades
more host-transfer bytes for device-local computation; its CUDA crossover is not
measured. Dynamic-programming time dependence and inference host output size remain.

## Required hardware follow-up

Use a CUDA-enabled build satisfying the existing Torch version pin and actual
allocated GPUs. Select devices before Python starts:

```sh
CUDA_VISIBLE_DEVICES=0 python verification/verify_acceleration.py --require-cuda
CUDA_VISIBLE_DEVICES=0,1 python verification/verify_acceleration.py --require-multi-gpu
CUDA_VISIBLE_DEVICES=0 python verification/benchmark_acceleration.py \
  --baseline ea8c4f1 --device cuda --repeats 100 --output /tmp/acceleration-cuda.json
```

The CUDA benchmark synchronizes before/after timing and records incremental peak
allocated VRAM. End-to-end clinical batch/length/K sweeps, multi-GPU scaling,
padding-dependent behavior of particular backbones, output-transfer overhead and
DDP/FSDP need separate hardware experiments. This review certifies the executed
CPU checks and fixes; it does not certify complete or optimal GPU performance.
