# Train a model against a split

Takes a split directory produced by [`prepare_dataset`](prepare-dataset.md) and scores one
configuration against it. The split's `data_manifest.json` decides the shape of the run: a
plain train/validation split trains once, a fixed-holdout k-fold split trains once per
rotatable fold and aggregates. The fixed holdout is never read.

```python
from ner_lab import train_model

assessment = train_model(
    split_dir="assets/splits/disease/kfold_5_holdout_0",
    output_dir="assets/runs",
    base_model="PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    target_label="DISEASE",
    language="es",
    architecture="crf",
    training_arguments={"learning_rate": 2e-5, "num_train_epochs": 8},
)

assessment.fold_metrics                    # DataFrame, one row per fold
assessment.aggregate["best_metric"]        # {"mean": ..., "std": ...}
assessment.run_dir                         # where everything was written
```

| Parameter | Default | Meaning |
|---|---|---|
| `split_dir` | *required* | A split directory containing `data_manifest.json`. |
| `output_dir` | *required* | Root for run directories. |
| `base_model` | *required* | Pretrained encoder, by hub name or local path. |
| `target_label` | *required* | The entity type to tag. One label, three classes (D43). |
| `language` | *required* | Sentence-segmentation language for `pysbd`. Never guessed (D33). |
| `architecture` | `"linear"` | `"linear"`, `"crf"`, or your own `(base_model, label2id, id2label, **kwargs) -> model`. |
| `architecture_kwargs` | `None` | Extra keywords for the chosen architecture, e.g. `{"dropout": 0.2}`. |
| `training_arguments` | `None` | A `TrainingArguments`, or a mapping of overrides applied to `ner_lab.training.DEFAULTS`. The YAML path uses the mapping. |
| `max_length` | `256` | Token budget per window, including special tokens. |
| `strategy` | `"greedy"` | Window strategy, or your own callable. |
| `context_tokens` | `None` | Flanking context per window. Only valid with `strategy="context"`. |
| `overlap_policy` | `"merge_same_label_then_keep_longest"` | How overlapping gold spans are resolved at encoding time. |
| `min_sentence_tokens` | `4` | Sentences shorter than this are merged into their neighbour. |
| `early_stopping_patience` | `5` | Epochs without improvement before stopping. `None` disables it. |
| `pad_to_multiple_of` | `8` | Pad batches to a multiple of this. |
| `metrics_scope` | `"eval"` | `"both"` also scores the training set each epoch, at the cost of a second pass. |
| `min_overlap_percentage` | `40.0` | Character overlap a predicted span needs against a gold one to count as partial. Strict and exact scenarios ignore it. |
| `include_confusion` | `False` | Add the token confusion matrix to each evaluation. |
| `save_model` | `False` | Keep weights. Off because this scores a configuration; it does not build a deployment artifact. |
| `track_resources` | `True` | Wall clock, energy, emissions and peak VRAM per run, via codecarbon. |
| `devices` | `1` | Pin the run to one GPU whatever the allocation exposes. `"all"` spreads it over every visible one, warning loudly that the batch is no longer the one that was tuned. |
| `overwrite` | `False` | Replace a run already occupying `output_dir` instead of refusing. |
| `folds` | `None` | Narrow a k-fold run to specific validation folds. Invalid without a fixed holdout. |
| `random_state` | `None` | Overrides `TrainingArguments.seed`. |

Everything after `language` configures the `Encoder`, `build_model` and `train` that this
assembles. Build those yourself and call `ner_lab.training.train` if the assembly is in the
way — the orchestrator applies no policy you cannot reach by
[composing them](library.md).

**Returns** an `AssessmentResult` with `run_dir`, `mode`, `fold_metrics`, `aggregate`,
`summaries`, `manifest` and `paths`. It deliberately does **not** return the `Trainer` or the
model: one per fold would keep every fold's weights resident while the next one trains. Use
`save_model=True`, or call `train` directly.

## What it writes

```
<output_dir>/                  the run directory itself: nothing is derived or nested
    run_manifest.json          written before training starts, so a crash is debuggable
    fold_metrics.parquet       one row per run: selection metric, best epoch, resources
    assessment_summary.json    mean and standard deviation across runs
    fold_01/                   per-fold, or the run directory itself for a plain split
        epoch_metrics.parquet
        training_summary.json
        best_model/            only with save_model=True
    fold_02/
```

`metric_for_best_model` defaults to `span_strict_f1`; the `compute_metrics` that produces it
is built for you from each run's own validation rows.

## From YAML

```yaml
task: train_model
split_dir: assets/splits/disease/kfold_5_holdout_0
output_dir: assets/runs
base_model: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
target_label: DISEASE
language: es
architecture: crf
max_length: 256
training_arguments:
  learning_rate: 2.0e-5
  num_train_epochs: 8
  per_device_train_batch_size: 16
```
