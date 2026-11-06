# Unified ensembles: implementation and validation record

2026-11-06. The local branch already contained the committed Multi-Head architecture;
this extension was implemented against that actual code, not public `main`.

## Architecture and integration map

Existing paths: `fit_ensemble -> member inference -> word tables -> WordVotes -> two-stage
BIO decoding -> scoring/artifacts`; `predict_ensemble` loads the persisted choices. The
following were reused: checkpoint loading, Encoder/windows/masks, batched `predict_logits`,
CRF forward-backward, UAX word IDs, single-type `WordVotes`, two-stage decoding, span-table
persistence, Multi-Head exact scoring, character scoring and CLI dispatch.

The shared adapter returns `MemberPredictions(words, spans, entity_labels, timings)`.
Sparse word rows are keyed by document, UAX word and entity, with the member identified by
its list index. Missing rows imply no vote; alignment makes `present` explicit. Each entity
has an `EntityVotes` collection with only eligible member columns. Native head order is
resolved through label strings, never aligned by head index. No whole corpus is copied per
entity. Member inference runs once for all heads, releases the model, and caches predictions
for all methods/trials. Span methods reuse original reconstructed spans and coverage rather
than loading models again. No dense unsupported word/type/member product is built.

Multi-Head CRF does not exist in the inspected implementation. Supported native models are
classic linear, classic CRF and Multi-Head linear, including heterogeneous catalogues.
CRF decisions are Viterbi; soft votes are actual marginal distributions. Scores are not
claimed to be calibrated. The core/NER/NEL dependency boundary and CLI are unchanged.

## Public API, methods and compatibility

The stage entry points remain `fit_ensemble` and `predict_ensemble`. Existing argument order
is retained. New explicit fitting arguments are `entity_labels`, `weight_scope`,
`threshold_scope`, `min_support`, `missing_policy`, `min_gold_entities` and `compare_methods`.
The default metric is `span_strict_f1` for historical classic single-label BIO ensembles,
otherwise `span_strict_macro_f1`; explicit choices are retained.

- `majority`, `soft`, `weighted` work independently by entity. A missing label is not O;
  weights normalize only over eligible and present members. Hard ties do not form a majority.
- `weight_scope="entity"` and `threshold_scope="entity"` provide independent values.
  Scoped search samples n−1 simplex parameters for n eligible members; singletons have no
  weight degree of freedom. Below five unique gold entities by default, weights are uniform
  and the threshold falls back to 0.5 (entity scope) or a fitted global threshold. No useful
  supervised parameter means no trial. New searches keep the equal-weight baseline unless
  beaten. Classic global fitting with an old metric retains the old sampler/selection;
  fixed-seed equality against `search_ensemble` is checked. A singleton weighted method and
  explicit new gap policies/scopes use the new search path.
- `best` picks an expert per type from validation; ties and sparse supervision fall back
  deterministically to the first eligible member.
- `span_union`, `span_consensus`, `span_intersection` use exact identities; consensus defaults
  to strict majority and intersection to all eligible sources. `min_support` is an explicit
  alternative count. Duplicate mentions from one source cannot inflate support. A member
  can oppose a candidate only when it supports the type and covers the candidate's UAX
  words; missing coverage is not negative evidence. Span confidence is the mean source
  score among supporters.
- New BIO ensembles break gaps by default; legacy single-target and format-1 files keep
  their historical skip rule. Two-stage B/I tie/orphan handling is reused.

Cross-label overlaps survive all methods. BIO voting remains one track per type; it cannot
represent arbitrary intraclass overlaps. Span union can retain different overlapping
same-type intervals. Different boundaries are never silently merged and no NMS is applied.

Strict per-type, macro/micro counts and character metrics reuse Multi-Head set scoring;
zero-support catalogue types score zero and remain in macro. Micro is computed from counts,
not averaged F1. With an explicit label subset, evaluation restricts gold to it. Otherwise
unknown gold types warn and count as micro FN without changing the catalogue. Classic
single-label BIO scoring retains historical SemEval/partial semantics; its new exact
per-type/macro/micro keys can differ from its historical aggregate on ambiguous overlaps.

Individual baselines distinguish the full ensemble catalogue from supported-label-only
specialist tasks. Unsupported catalogue types count as FN in the full-task comparison.
Metrics also report the best global member, selected specialists, majority, soft, all four
weight/threshold scope combinations and the three span methods. All reuse cached evidence.
Extra optimized comparisons can be disabled with `compare_methods=False`; fitting-set
scores are not unbiased test estimates or evidence of statistical superiority.

Calibration inspects available training summaries and canonical training parquet paths.
It rejects known document-ID or exact-text overlap when fitting, including renamed copies.
Incomplete/missing/unreadable provenance is recorded as unknown and warns. It does not claim
unknown datasets are leakage-free. Test/holdout is never selected automatically; callers
must supply appropriate validation. Stacking is explicitly rejected because this API does
not establish the required OOF or independent calibration provenance.

New fits write `ensemble.json` version 2 with labels, member support/architectures and paths,
weights/settings digests, global/entity weights and thresholds, gap/consensus policy,
objective, calibration and expert choices. Other artifact names remain unchanged.
Inference does not fit or discover labels from gold. Configuration/tokenizer/weight changes,
incompatible scoped dimensions and invalid thresholds are detected before model inference.
Version 1 remains readable with original predictions and scores. It did not save metadata
fingerprints, so only its recorded weights/target can be checked retroactively; this limit
is explicit rather than fabricated historical integrity evidence.

## File responsibilities

| File | Responsibility |
|---|---|
| `ensemble/tables.py` | One inference adapter, sparse labelled tables, original spans and member timings; preserve classic word-table API |
| `ensemble/entities.py` (new) | Typed scoped vote collections/parameters, eligibility alignment, independent decoding and shared scoring |
| `ensemble/voting.py` | Preserve old math; add explicit gap policy, shared boundaries and finite-value validation |
| `ensemble/spans.py` (new) | Exact span union/consensus/intersection with coverage-aware eligibility |
| `ensemble/combination.py` (new) | One pure dispatcher over cached evidence, used by fitting and inference |
| `ensemble/search.py` | Keep old public search; add supervised scoped search, simplex and baseline/fallback selection |
| `ensemble/assessment.py` (new) | Full-task versus specialist metrics, best experts and comparable cached baselines |
| `ensemble/fitting.py` (new) | Fit the requested method and optimized baseline comparisons without new inference |
| `ensemble/calibration.py` (new) | Training/calibration provenance, source hashes and duplicate-document detection |
| `ensemble/options.py` (new) | Method vocabulary and early validation of incompatible arguments |
| `ensemble/fit.py` | Existing fitting orchestrator extended to all member families and version-2 artifacts |
| `ensemble/predict.py` | Existing prediction orchestrator, version-1 compatibility branch and fixed version-2 replay |
| `ensemble/io.py` | Metadata catalogues, weight/settings fingerprints, version readers and schema checks |
| `ensemble/__init__.py` | Discoverable typed pure helpers alongside existing exports |
| `verification/verify_ensemble.py` | Original suite extended to call heterogeneous checks; local WordPiece fixtures |
| `verification/verify_unified_ensemble.py` (new) | Catalogue/coverage math, hybrid A–F, native checkpoints, fitting, versions, CLI, NEL and provenance checks |
| `verification/benchmark_ensemble.py` (new) | Reproducible isolated-process CPU component timings and memory accounting |
| `verification/fixtures.py` | Shared local WordPiece tokenizer, moved out of the Multi-Head script |
| `verification/verify_inference.py` | Optional tokenizer injection in the saved-model fixture; existing default unchanged |
| `verification/verify_multihead.py` | Reuse shared tokenizer; replace the superseded ensemble-rejection assertion with support discovery |
| `docs/ner/configs/ensemble-*.yaml` (six new files) | Classic, Multi-Head, hybrid, scoped, span and prediction examples, all executed with fixture paths |
| `docs/ner/{ensemble,README,library}.md` | Contracts, examples, methods, metrics, compatibility and limits |
| `docs/dev/{DECISIONS,ROADMAP}.md` | Append D136–D140; preserve historical decisions and distinguish remaining research |
| `verification/{README,UNIFIED_ENSEMBLE_RESULTS}.md` | Reproduction and evidence |
| `verification/MULTIHEAD_RESULTS.md` | Mark previous ensemble limitations as historical, linking this extension |

Runtime paths in this table are relative to `src/lab/ner/`. `lab.core`, `lab.nel`, model
architectures, dependency declarations/lockfile and CLI implementation were not changed.

## Executed checks

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_ensemble.py
.venv-verify/bin/python verification/verify_layering.py
.venv-verify/bin/ruff check src/lab/ner verification/verify_ensemble.py \
  verification/verify_multihead.py verification/verify_inference.py verification/fixtures.py \
  verification/verify_unified_ensemble.py verification/benchmark_ensemble.py
.venv-verify/bin/ruff format --check src/lab/ner/ensemble
git diff --check
```

Latest ensemble run: **207 passed, 0 failed, no skipped checks**. Includes all 76 original
ensemble checks plus the heterogeneous extension. The original checks now use a real local
WordPiece tokenizer, not a remote BERT tokenizer. The miniature BERT still trains through
`train_model`; validation fixtures have separate document IDs and text. No assertion was
removed to make a test pass. The Multi-Head ensemble-rejection assertion was deliberately
replaced because D136 supersedes that limitation.

Checked: majority/soft/weighted legacy behavior, exact seeded legacy fitting, Viterbi and
brute-force marginal equivalence, legitimate marginal-argmax/Viterbi differences, NaNs,
zero weights, missing members, singleton degrees of freedom, thresholds, per-type weights,
manual macro/micro counts, sparse-gold fallback, reproducibility, native head-order and
tokenizer differences, hybrid A–F, raw-text prediction, unknown label rejection, schema and
metadata tampering, format-1 round trips, every supported method, all six YAMLs, TSV/Parquet,
canonical spans composed with NEL, and ID/text leakage detection. The native inference spy
measures exactly three checkpoint loads and six Transformer forwards (three documents,
batch two) across all selected fitting and comparison trials.

Layering: **144 passed, 0 failed**. Ruff, formatting and whitespace checks passed.

The full runner was also executed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/run_all.py
```

It exits **1**, not success. At that invocation the then-current ensemble suite passed
199 checks; eight later strict-artifact/legacy-equivalence checks bring the final standalone
run to 207. Multi-Head passed **116, 0 failed, 1 CUDA skip**; data passed **192 with 2
real-data skips**; NEL passed **62 with 2 skips** (external equivalence checkout and uncached
BERT tokenizer); layering passed 144. Eight older scripts still fail on uncached BERT/GPT-2
tokenizer access: encoding, encoder, models, training, evaluation, assessment, HPO and
inference. The configured runtime still blocks the remote tokenizer destination; offline
mode makes that limitation immediate rather than retrying. These eight scripts were not
counted as passed. Local legacy and Multi-Head training/inference are nevertheless exercised
by the executed ensemble and Multi-Head suites.

Logs are session artifacts at `/workspace/ensemble-final.log` and
`/workspace/ensemble-regressions.log`, not committed test fixtures.

## Measured CPU behavior

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/benchmark_ensemble.py --repeats 50 \
  --output /workspace/ensemble-benchmark.json
```

AMD EPYC 9V45 host, x86_64 Linux, 5 exposed vCPUs, 2 torch threads, torch 2.10.0+cpu;
random 2-layer BERTs with hidden size 32. Every scenario uses 3 documents, batch 2, window
budget 32 and **fixed padding 32**, even across tokenizers. Each scenario runs in a separate
process. Combination/decoding times are means over 50 repeats, without new inference.
Optuna time is for three per-entity weight/threshold trials. The test models intentionally
use controlled logits: this measures engineering cost, not clinical accuracy.

| Scenario | Members / types | Member pipeline total ms | Batched inference ms | Alignment ms | Combination ms | Decoding ms | Optuna ms | Backbone forwards | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classic specialists | 2 / 2 | 369.638 | 20.945 | 8.071 | 0.041 | 2.327 | 45.648 | 4 | 533.3 |
| Two Multi-Head | 2 / 2 | 374.221 | 22.216 | 12.152 | 0.042 | 2.298 | 43.491 | 4 | 533.4 |
| Hybrid linear + CRF + Multi-Head | 3 / 3 | 585.723 | 37.356 | 15.943 | 0.072 | 3.765 | 52.800 | 6 | 537.6 |
| One Multi-Head, one type | 1 / 1 | 191.372 | 12.694 | 6.023 | 0.024 | 1.225 | 40.411 | 2 | 532.6 |
| One Multi-Head, three types | 1 / 3 | 183.089 | 11.943 | 10.869 | 0.061 | 3.646 | 51.709 | 2 | 533.8 |
| One Multi-Head, two types | 1 / 2 | 183.782 | 11.804 | 8.241 | 0.040 | 2.279 | 43.899 | 2 | 532.4 |

Member pipeline total includes checkpoint loading, encoding, Trainer inference, adaptation,
original-span decoding and release/garbage collection. It is not the same as pure forward
latency. Batched inference includes Trainer/padding overhead. Alignment, combination and
final decoding are separate; word adaptation/original member span decoding are measured
together. No quality or statistically significant speed comparison is claimed.

Increasing one member from one to three heads keeps two backbone forwards; aligned arrays
increase from **468 to 1,404 bytes**, with word/decode work scaling by actual supported
heads. Increasing two-type Multi-Head members from one to two doubles forwards (2 to 4)
and aligned arrays (936 to 1,872 bytes). Classic, Multi-Head and hybrid arrays occupy
936/1,872/2,340 bytes: the Multi-Head case has more type/member evidence than two specialists,
so those are not identical information budgets. Process peaks include Python, dependencies
and fixture generation; tiny-model RSS differences do not establish model-memory savings.
There is no CUDA, production-corpus or real-clinical-checkpoint benchmark.

## Examples and remaining limits

All six `docs/ner/configs/ensemble-*.yaml` files were executed through `lab run` with temporary
fixture paths substituted; Python use and every implemented argument are documented in
[ensemble.md](../docs/ner/ensemble.md). Real use requires saved checkpoints/validation at
the example paths; no models or corpora are bundled into the YAMLs.

The required classic/Multi-Head/hybrid BIO and exact-span families are implemented. Remaining
limits are actual ones: eight external-tokenizer-dependent regressions, GPU/production
validation, absent Multi-Head CRF, unsupported stacking without OOF provenance, optional
alternative structured combiners and partial-span clustering. A callable window strategy
still cannot be reconstructed from metadata and is refused as before. Unknown training
provenance cannot establish independence. Cached predictions are in-memory per operation;
there is no cross-request disk cache. Format 1 cannot verify settings digests it never saved.
