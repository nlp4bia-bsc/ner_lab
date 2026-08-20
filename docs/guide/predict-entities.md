# Predict with a trained model

```python
from ner_lab import predict_entities

result = predict_entities(
    model_dir="assets/runs/DISEASE__crf__.../best_model",
    documents="assets/gold/test.parquet",
    output_dir="assets/predictions",
)

result.spans        # DataFrame: filename | label | start_span | end_span | text | score
result.metrics      # span and token metrics, when the input carried gold
result.official     # MultiClinNER strict + character F1, when there was gold to score against
```

| Parameter | Default | Meaning |
|---|---|---|
| `model_dir` | *required* | A directory of saved weights with an `encoding.json` beside them. |
| `documents` | *required* | A canonical corpus (frame or parquet), a directory of `.txt` files, or a `{doc_id: text}` mapping. |
| `output_dir` | *required* | Where predictions and any scores are written. |
| `reference` | `None` | Gold to score against when `documents` carries none: a corpus parquet or an annotation TSV. |
| `pattern` | `"*.txt"` | Glob used when `documents` is a directory. |
| `encoding` | `"utf-8"` | Text encoding used when `documents` is a directory. |
| `batch_size` | `16` | Windows per forward pass. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, or a specific device string. |
| `min_score` | `0.0` | Drop predicted entities below this mean token probability. |
| `min_overlap_percentage` | `40.0` | Character overlap a predicted span needs against a gold one to count as partial. |
| `include_confusion` | `False` | Add the token confusion matrix to the metrics. |
| `strategy` | `None` | Only needed when the model was trained with a callable window strategy, which `encoding.json` cannot name. |
| `pad_to_multiple_of` | `8` | Pad batches to a multiple of this. |

**Returns** an `InferenceResult` with `spans`, `metrics`, `official`, `manifest` and `paths`.

## The encoding is not restated

Nothing about the windowing is a parameter here. A model saved by
`train_model(save_model=True)` carries an `encoding.json` beside its weights holding the base
model, architecture and the whole windowing configuration, so inference windows its input
exactly as training did and cannot silently disagree with it. That file is also what makes a
CRF model loadable at all: `CRFForTokenClassification` is not a `PreTrainedModel`, so the
Trainer saves it as a bare state dict with no `config.json` to rebuild it from.

Gold entities in the input are scored automatically; raw text has none, so pass `reference=`
to score it officially.

## What it writes

```
<output_dir>/
    predictions.tsv            the official schema plus a score column
    gold.tsv                   the gold set scored against, in the same schema
    prediction_metrics.json    span + token metrics, when the input carried gold
    multiclinner_eval.json     the official strict and character-overlap F1
    inference_manifest.json    the model, its encoding, and what was predicted on
```

A span two overlapping windows both predict is kept once, at its higher score; `min_score`
drops predictions below a mean token probability. A score is the mean softmax probability of
the tokens the span was decoded from.

## From YAML

```yaml
task: predict_entities
model_dir: assets/runs/DISEASE__crf__20260809_120000/best_model
documents: assets/gold/test.parquet
output_dir: assets/predictions
batch_size: 32
min_score: 0.5
```
