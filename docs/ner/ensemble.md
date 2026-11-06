# Ensemble saved models

```python
from lab.ner import fit_ensemble, predict_ensemble

fit = fit_ensemble(
    model_dirs=[
        "assets/runs/DISEASE__crf__roberta/best_model",
        "assets/runs/DISEASE__linear__mdeberta/best_model",
        "assets/runs/DISEASE__linear__modernbert/best_model",
    ],
    method="weighted",
    output_dir="assets/ensembles/disease",
    validation="assets/datasets/disease/validation.parquet",
    random_state=0,
)
fit.summary()   # "weighted ensemble of 3 members, soft votes, threshold 0.412 | validation strict F1 = 0.83 (best member 0.81)"

result = predict_ensemble(
    ensemble_dir="assets/ensembles/disease",
    documents="assets/datasets/disease/test.parquet",
    output_dir="assets/predictions/disease_ensemble",
)
result.spans    # DataFrame: filename | label | start_span | end_span | text | score
```

An ensemble combines independently trained `linear`, `crf`, `multihead` and `multihead_crf` checkpoints,
including members with different label catalogues, backbones, tokenizers and windowings.
Fitting records how their evidence combines; predicting loads those fixed choices.
Multi-Head CRF uses the same unified member adapter. The legacy single-label examples
below remain valid; see the unified methods and format compatibility sections further down.

## `fit_ensemble`

| Parameter | Default | Meaning |
|---|---|---|
| `model_dirs` | *required* | Saved models, each with the `encoding.json` beside its weights. |
| `method` | *required* | `"majority"`, `"soft"` or `"weighted"` — see below. |
| `output_dir` | *required* | Where the fitted ensemble is written. |
| `validation` | `None` | Gold-annotated data to tune on and score against: a corpus (frame or parquet), a directory of `.txt` files, or a `{doc_id: text}` mapping. Required when anything is tuned. |
| `reference` | `None` | The gold for `validation` when it carries none, as for `predict_entities`. |
| `votes` | `None` | `"weighted"` only: `"soft"` (the default) votes with probabilities, `"hard"` with each member's own tag. |
| `tune_threshold` | `False` | `"soft"` only: tune the threshold on `validation` instead of using 0.5. |
| `metric` | `None` | Resolved to `span_strict_f1` for classic single-label BIO voting, otherwise `span_strict_macro_f1`. An explicit metric is preserved. |
| `n_trials` | `200` | Optuna trials when tuning. |
| `random_state` | `None` | Seed for Optuna's TPE sampler. |
| `pattern`, `encoding` | `"*.txt"`, `"utf-8"` | Used when `validation` is a directory. |
| `min_overlap_percentage` | `40.0` | Character overlap a predicted span needs to count as partial. |
| `batch_size`, `device`, `pad_to_multiple_of` | `16`, `"auto"`, `8` | How the members run. |

**Returns** an `EnsembleFit` with `ensemble`, `metrics`, `trials`, `manifest` and `paths`.

Which data to tune on is yours to choose. Nothing checks whether a member trained on the
validation documents; if it did, it will look better than it is and be weighted up.

## `predict_ensemble`

| Parameter | Default | Meaning |
|---|---|---|
| `ensemble_dir` | *required* | The `output_dir` of `fit_ensemble`. |
| `documents` | *required* | As for `predict_entities`. |
| `output_dir` | *required* | Where predictions and any scores are written. |
| `reference` | `None` | Gold to score against, as for `predict_entities`. |
| `min_score` | `0.0` | Drop predicted entities below this score. |
| `min_overlap_percentage` | `40.0` | As above. |
| `pattern`, `encoding`, `batch_size`, `device`, `pad_to_multiple_of` | | As for `predict_entities`. |

**Returns** an `EnsembleInference` with `spans`, `metrics`, `manifest` and `paths`.

Nothing about the combination is restated: the members, weights, threshold and kind of
vote come from `ensemble.json`. A member whose weights changed since the fit — a model
retrained into the same directory — raises before anything runs.

## How members vote

Members vote **word by word**. A word is a Unicode UAX #29 word of the document, the unit
every model is labelled by whatever its tokenizer, so models with different tokenizers,
window lengths and strategies meet on the same words. Each member gives, at each word it
labelled, its own tag and its probability of `O`, `B` and `I`. A linear model's
probabilities are its softmax and its tag their argmax; a CRF's probabilities are its
forward-backward marginals and its tag its Viterbi path.

Votes decode in two stages, per document:

1. **Is the word inside an entity?** Yes when the weighted share of members saying so is
   strictly above the threshold. A hard vote says so when the member's tag is `B` or `I`; a
   soft vote counts `p_B + p_I`. Weights are renormalised over the members that labelled
   the word, so a member that did not label it is no evidence either way.
2. **Does it begin a new entity?** Yes after a word outside any entity, or when the combined
   `B` weight beats the `I` weight; a tie continues the entity.

The output is always a valid BIO sequence. A span runs from its first word's start to its
last word's end, and its score is the mean combined probability of the tags its words took.

| `method` | Votes | Weights | Threshold |
|---|---|---|---|
| `majority` | hard | equal | 0.5 — strictly more than half |
| `soft` | soft | equal | 0.5, or tuned with `tune_threshold` |
| `weighted` | soft, or hard with `votes="hard"` | tuned | tuned |

Tuning samples one weight per member in [0, 1], normalised to sum to one, and the threshold
in [0.05, 0.95], with Optuna's TPE sampler. The members run once over the validation data;
each trial after that only recombines their votes, so trials are cheap.

A one-member ensemble predicts what `predict_entities` predicts, with two exceptions: a CRF
member under soft votes decodes from its marginals rather than its Viterbi path, and an
ensemble decodes per document, so a span can continue across a window boundary that
`predict_entities` would break it at. Windows end at sentence boundaries, so that only
happens inside a sentence too long for one window.

## What it writes

```
<fit output_dir>/
    ensemble.json              method, votes, threshold, and each member's path, weight and weights' digest
    ensemble_manifest.json     the members' encodings, the validation data, the search settings
    ensemble_trials.parquet    one row per trial, when the fit searched
    ensemble_metrics.json      the ensemble and each member alone on validation, when there was any

<predict output_dir>/
    predictions.tsv            the span table plus a score column
    predictions.parquet        the same table
    gold.tsv                   the gold spans scored against, when there were any
    prediction_metrics.json    span and character metrics, when there was gold
    inference_manifest.json    the ensemble and what was predicted on
```

`ensemble_metrics.json` scores each member alone on the same documents, so whether the
ensemble beats its best member is on paper. Token diagnostics are not computed: the members
do not share token rows.

## From YAML

```yaml
task: ner.fit_ensemble
model_dirs:
  - assets/runs/DISEASE__crf__roberta/best_model
  - assets/runs/DISEASE__linear__mdeberta/best_model
method: weighted
validation: assets/datasets/disease/validation.parquet
output_dir: assets/ensembles/disease
n_trials: 200
random_state: 0
```

```yaml
task: ner.predict_ensemble
ensemble_dir: assets/ensembles/disease
documents: assets/datasets/disease/test.parquet
output_dir: assets/predictions/disease_ensemble
```



## Unified prediction contract and coverage

All supported heads adapt to `(doc_id, word_id, entity_label, member index)` with a BIO tag
and `p_O,p_B,p_I`. `MemberPredictions` carries a sparse labelled word table and original
scored spans. Row presence is coverage; `EntityVotes` maps its local columns to eligible
member indices and reuses `WordVotes.present`. There is no dense product of unsupported
word/type/member combinations. Member order is persisted; head order is resolved through
each checkpoint's **label names**, never matched by head index.

The catalogue is the sorted union of member metadata. `entity_labels=[...]` selects an
explicit, nonempty subset; members supporting none of it are omitted. Validation/test gold
never defines the catalogue. A model knowing a type and predicting O casts a negative vote.
A model not knowing it, or not labelling a word because of a mask/window, casts no vote.
Weights normalize per word over eligible **present** members only. Zero total weight gives
zero evidence and no entity, not a confident O distribution.

Each checkpoint loads once per fit/prediction, encodes documents once, and predicts all its
heads in a single batched forward. It is released before loading the next. Fitting caches
all member predictions: changing a type, method, threshold or Optuna trial does not rerun
the Transformers. New heterogeneous inference windows unannotated documents, so gold does
not guide cuts or trigger an intraclass-overlap error; the historical classic single-label
BIO path retains its annotation-aware windowing. Alignment uses UAX #29 word IDs, not token
indices. Token offsets, specials, first-subword supervision and context masks remain the
member encoder's responsibility.

CRF hard decisions are Viterbi. Its soft probabilities are real forward-backward marginals;
one-hot Viterbi values are used solely to transport decisions, never as soft confidence.
Multi-Head linear probabilities are independent softmax distributions per head. None of
these distributions is claimed to be calibrated.

## Methods and explicit options

| Method | Combination and fitting |
|---|---|
| `majority` | Equal weights, hard BIO decisions, strict majority above 0.5, independently per type |
| `soft` | Equal weights, averaged distributions; `p_B+p_I > 0.5`; optional `tune_threshold=True` |
| `weighted` | Optuna weights and thresholds, with `votes="soft"` (default) or `"hard"` |
| `best` | One validation-selected specialist per entity, returning its original decoded spans |
| `span_union` | Union of exact identities, preserving different boundaries even within one type |
| `span_consensus` | Exact agreement from a strict majority of eligible members, or `min_support` |
| `span_intersection` | Exact agreement from every eligible member, or explicit `min_support` |

The two-stage BIO decoder remains unchanged: decide membership, then B versus I. An orphan
I starts a span; B/I ties continue it. Types never compete, so their spans may overlap.
BIO voting still has a single track per type and cannot express arbitrary intraclass
overlaps. The independent span family can retain overlapping same-type candidates.
It never merges intervals because they share characters and applies no NMS.

| New argument | Default | Applies to |
|---|---|---|
| `entity_labels` | `None` | Metadata union or explicit subset, all methods |
| `weight_scope` | `"global"` | `weighted`: `"global"` or `"entity"` |
| `threshold_scope` | `"global"` | `weighted`, or `soft` with `tune_threshold=True`: `"global"` or `"entity"` |
| `min_support` | `None` | Positive integer for `span_consensus`/`span_intersection` |
| `missing_policy` | `None` | BIO voting: `"skip"` or `"break"` |
| `min_gold_entities` | `5` | Minimum unique gold mentions per label for scoped fitting/specialist selection |
| `compare_methods` | `True` | With validation, include four optimized weight/threshold scope baselines |

Unsupported combinations raise before checkpoint inference. `votes` remains exclusive to
`weighted`, and `tune_threshold` exclusive to `soft`; weighted always tunes its threshold.
`best`, weighted, and tuned soft require annotated validation (or `reference`). Untuned
majority, soft and span methods can be saved without validation. `min_score` remains a
post-decoding confidence filter in `predict_ensemble`; it is not a word-membership threshold.
`min_support` is a count, not a probability or a span-confidence filter.

Per-entity weights have one component per member supporting that entity, in persisted
member order. There are no parameters for unsupported members. A singleton has unit weight
and no weight parameter; n eligible weights use n−1 simplex parameters. Per-entity thresholds
are independent. A label below `min_gold_entities` uses uniform weights; its threshold stays
0.5 under entity scope, or uses the fitted global threshold under global scope. Specialist
selection falls back to the first eligible member. No useful supervised parameter means
no Optuna trials. Fallbacks and gold support counts are recorded in the metrics artifact.

Optuna maximizes the selected metric and uses `random_state` for reproducibility. New scoped
searches explicitly evaluate the equal weight/0.5 baseline and retain it unless beaten.
Classic single-target global fitting with an existing metric retains its original TPE
sampling and selection, including fixed-seed weights/thresholds; the uniform baseline is
reported without changing that historical selection. Its pre-existing global fitting does
not apply the new per-entity minimum-support fallback. New scopes and metrics use the scoped
sampler, simplex and fallback rules.
Set `compare_methods=False` to avoid the extra four baseline searches. Default comparisons
use the same cached evidence and trial budget, but their scores are fitting-set scores,
not an unbiased test estimate or statistical evidence that one method is superior.

`missing_policy=None` resolves to `skip` for classic single-label BIO ensembles and `break`
for new heterogeneous/Multi-Head ensembles. `skip` preserves the historical behavior across
words nobody labelled. `break` starts a new span after a gap in word IDs. A word with no
positive-weight evidence ends an entity. Format-1 files always retain their old skip rule.

Span agreement is exact by `(filename,label,start,end)`. Duplicate spans from one member
count once. A member can oppose a candidate only if it supports that label and covers all
UAX words the span overlaps; members with missing coverage do not veto it. Supporting
members are eligible by construction. Scores are the mean original confidence of supporting
members, not support fractions or calibrated probabilities. Union keeps every candidate;
consensus/intersection filter identities without altering their boundaries.

## Metrics, baselines and calibration provenance

New multientity ensembles reuse Multi-Head exact set scoring and character metrics:
`span_strict_{precision,recall,f1,tp,fp,fn}__LABEL`, `span_strict_macro_*`,
`span_strict_micro_*`, plus `char_*`. Duplicate identities count once. Macro averages every
catalogue type, including zero-support types scoring zero. Micro pools TP/FP/FN before
computing F1. Unknown gold labels do not create heads: they warn and count as micro FN,
as in Multi-Head inference. With an explicit `entity_labels` subset, gold is restricted to
that subset. Historical classic single-label BIO voting retains its SemEval score and
partial-overlap interpretation of `min_overlap_percentage`; exact per-type/macro/micro keys
are additionally available. No partial matching is substituted for strict scoring.

`metric` can select strict macro-F1, strict micro-F1, `span_strict_f1__LABEL`, character F1 or
another reported metric to maximize. `ensemble_metrics.json` distinguishes:

- Individual member scores on the **full ensemble task**, where unsupported catalogue
  types are FN, and scores restricted to supported types (`specialists`).
- Best global member under the selected metric and best specialist per entity.
- Majority, soft, all four global/entity weight/threshold combinations, and all three exact
  span families, reusing cached predictions. Additional fitting is controlled by
  `compare_methods`; the requested method's trial trace stays in `ensemble_trials.parquet`.
- The requested ensemble, calibration support, sparse-label fallbacks and baseline selection.

Only `validation` is used to select parameters. Available `training_summary.json` provenance
is checked against calibration document IDs, exact text hashes and recorded training file
hashes. Identical training text is detected even when a document is renamed. Known
training/calibration overlap raises when any fitting is requested, including optimized
baselines. Missing/unreadable training-source evidence warns and records `unknown`; it is
never described as leakage-free. Users still must reserve test/holdout: arbitrary external
paths cannot establish which dataset was intended as test. No random token split is used.
Stacking is not trained: selecting `method="stacking"` explains the need for independently
calibrated or out-of-fold predictions and rejects this unsupported path.

## Persisted inference and compatibility

New fits write `ensemble.json` **format 2**, with ordered entity labels, supported labels and
architecture per member, paths, weight digests and metadata/tokenizer digests, global/scoped
weights and thresholds, decision/missing-coverage policies, consensus support, objective,
specialist choices and calibration metadata. Scoped weight vectors follow the eligible
members' order, which is derived from those persisted catalogues. No checkpoint is copied.
`predict_ensemble` reads these values; it never fits, discovers labels from gold or changes a
threshold for the new dataset. Metadata, tokenizer or weight changes are refused before
inference. Invalid dimensions/thresholds/weights are rejected on read.

Format **1 remains readable without re-fitting**. Its original decoding, confidences and
scoring path are retained. Its weight digest and single target label remain verifiable;
it never recorded full encoder/tokenizer digests, so their historical values cannot be
recovered or verified retroactively. New format-2 digests close that gap.

## Executable examples

The configs assume existing local checkpoints and validation data at their `assets/...`
paths; the verification script substitutes local tiny-model fixtures and executes all six.

```bash
lab run docs/ner/configs/ensemble-classic.yaml
lab run docs/ner/configs/ensemble-multihead.yaml
lab run docs/ner/configs/ensemble-hybrid.yaml
lab run docs/ner/configs/ensemble-scoped.yaml
lab run docs/ner/configs/ensemble-spans.yaml
lab run docs/ner/configs/ensemble-predict.yaml
```

The same public functions work in Python:

```python
from lab.ner import fit_ensemble, predict_ensemble

fitted = fit_ensemble(
    model_dirs=["models/disease-crf/best_model", "models/multihead/best_model"],
    method="weighted", weight_scope="entity", threshold_scope="entity",
    validation="splits/validation.parquet", output_dir="ensembles/hybrid",
    random_state=42,
)
result = predict_ensemble(
    ensemble_dir=fitted.output_dir, documents={"note": "Paciente con dolor abdominal"},
    output_dir="predictions/hybrid",
)
```

See [verification evidence](../../verification/UNIFIED_ENSEMBLE_RESULTS.md) for executed checks,
benchmarks and remaining hardware/access limits. No learned stacking,
partial-span clustering, probability calibration or alternative Viterbi combiner is added.
The existing two-stage decoder already produces coherent BIO spans.


## Word-level constrained CRF members

`crf` version 2 and `multihead_crf` reuse the unified member table. One Transformer inference
per member batch yields all emissions. The adapter compacts eligible words once per structured
operation, restores Viterbi tags for hard votes and forward-backward marginal distributions
for soft votes. It never softmaxes Viterbi compatibility logits. Head identity comes from
metadata, not index coincidence. Missing types remain absent evidence.

Legacy CRFs keep their original marginal/decision adapter and checkpoint semantics. New and
old CRFs can mix with linear and Multi-Head members, using existing scoped weights/thresholds,
exact span methods, metrics and manifests. Marginals are normalized probabilities under the
member CRF, not a claim of calibration. See [hybrid CRF YAML](configs/ensemble-crf.yaml).


## Independent sigmoid Multi-Label members

`multilabel` joins linear, legacy/new CRF, Multi-Head and Multi-Head CRF through the same sparse
word-table adapter. Each checkpoint executes one batched Transformer inference for every label.
Hard tags follow its persisted decoder. Soft votes condition the independent Bernoulli B/I
outputs on the three legal local states, normalizing qO/qB/qI in log space; this is a derived
distribution, not the original model activation. `1-pB-pI` is never used. Unknown types and
masked words remain absent evidence, and head indices are matched by entity name.

A single Multi-Label soft ensemble can differ from its binary threshold decoder; hard votes
retain its categorical decisions. Mean conditioned chosen-state probabilities define span
confidence in both member inference and ensemble adaptation. Different-type overlaps survive;
BIO's intraclass limitation remains. Version-1/2 ensemble artifacts keep their contracts.
See [hybrid YAML](configs/ensemble-multilabel.yaml). Ensembles use checkpoint-persisted settings,
not an external `decoder_path`; save and revalidate the chosen member configuration before fitting.
