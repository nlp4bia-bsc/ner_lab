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

An ensemble combines models that were trained on their own: different backbones,
architectures (linear or CRF) and windowings, as long as they predict the same
`target_label`. Fitting decides how their votes combine and writes that down; predicting
runs the members over new documents and applies it.

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
| `metric` | `"span_strict_f1"` | What tuning maximises: any metric `predict_entities` reports. |
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
