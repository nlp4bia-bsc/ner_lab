# Predict with a trained model

```python
from lab.ner import predict_entities

result = predict_entities(
    model_dir="assets/runs/DISEASE__crf__.../best_model",
    documents="assets/gold/test.parquet",
    output_dir="assets/predictions",
)

result.spans        # DataFrame: filename | label | start_span | end_span | text | score
result.metrics      # span, character and token metrics, when there was gold to score against
result.summary()    # "412 entities from 96 windows | strict P/R/F1 = 0.81/0.78/0.79 | char F1 = 0.86"
```

| Parameter | Default | Meaning |
|---|---|---|
| `model_dir` | *required* | A directory of saved weights with an `encoding.json` beside them. |
| `documents` | *required* | A canonical corpus (frame or parquet), a directory of `.txt` files, or a `{doc_id: text}` mapping. |
| `output_dir` | *required* | Where predictions and any scores are written. |
| `reference` | `None` | Gold to score against: a corpus parquet or a span table TSV. Takes precedence over any entities `documents` carries. |
| `pattern` | `"*.txt"` | Glob used when `documents` is a directory. |
| `encoding` | `"utf-8"` | Text encoding used when `documents` is a directory. |
| `batch_size` | `16` | Windows per forward pass. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, or a specific device string. |
| `min_score` | `0.0` | Drop predicted entities below this mean token probability. |
| `min_overlap_percentage` | `40.0` | Character overlap a predicted span needs against a gold one to count as partial. |
| `include_confusion` | `False` | Add the token confusion matrix to the metrics. |
| `strategy` | `None` | Only needed when the model was trained with a callable window strategy, which `encoding.json` cannot name. |
| `pad_to_multiple_of` | `8` | Pad batches to a multiple of this. |

**Returns** an `InferenceResult` with `spans`, `metrics`, `manifest` and `paths`.

## The encoding is not restated

Nothing about the windowing is a parameter here. A model saved by
`train_model(save_model=True)` carries an `encoding.json` beside its weights holding the base
model, architecture and the whole windowing configuration, so inference windows its input
exactly as training did and cannot silently disagree with it. That file is also what makes a
CRF model loadable at all: `CRFForTokenClassification` is not a `PreTrainedModel`, so the
Trainer saves it as a bare state dict with no `config.json` to rebuild it from.

## Scoring

Gold entities in the input are scored automatically; raw text has none, so pass `reference=`
to score it. Span and character metrics score the predicted spans against the gold **as
annotated**, restricted to the model's `target_label` — every entity in the corpus counts,
including ones an overlap policy would have merged or a window would have cut. That is what
makes these the honest inference numbers, and also why `span_strict_f1` here can differ from
training's, which is necessarily computed against the gold the model was shown.

Token diagnostics (`token_*`, `confusion_*`) need the encoded rows to carry labels, so they
are added only when `documents` itself is the annotated corpus; a `reference` on raw text
gets span and character metrics alone.

## What it writes

```
<output_dir>/
    predictions.tsv            the span table plus a score column
    gold.tsv                   the gold spans scored against, when there were any
    prediction_metrics.json    span, character and token metrics, when there was gold
    inference_manifest.json    the model, its encoding, and what was predicted on
```

A span two overlapping windows both predict is kept once, at its higher score; `min_score`
drops predictions below a mean token probability. A score is the mean softmax probability of
the tokens the span was decoded from.

## From YAML

```yaml
task: ner.predict_entities
model_dir: assets/runs/DISEASE__crf__20260809_120000/best_model
documents: assets/gold/test.parquet
output_dir: assets/predictions
batch_size: 32
min_score: 0.5
```
