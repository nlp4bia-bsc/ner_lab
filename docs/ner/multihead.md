# Multihead models: one Transformer, a head per entity type

A single-label model (`train_model`) learns one entity type, as `O` / `B-<type>` / `I-<type>`.
A multihead model shares one Transformer between every type and gives each its own
independent O/B/I head, so a stretch of text can be inside a `DISEASE` and a `SYMPTOM` at
once. Each head's labels, windows and decoding are exactly what a single-label model for
that type would have, which keeps the two directly comparable.

It has its own three tasks, the counterparts of the single-label ones:

| CLI task | Function | Counterpart of |
|---|---|---|
| `ner.train_multihead` | `lab.ner.train_multihead` | [`train_model`](train-model.md) |
| `ner.search_multihead` | `lab.ner.search_multihead` | [`search_hyperparameters`](search-hyperparameters.md) |
| `ner.predict_multihead` | `lab.ner.predict_multihead` | [`predict_entities`](predict-entities.md) |

Each takes the same parameters as its counterpart, run the same way and writing the same
files, with the differences below. A multihead model directory loads only through
`predict_multihead`, and a single-label one only through `predict_entities`.

## Train

```yaml
task: ner.train_multihead
split_dir: assets/splits/clinical/kfold_5_holdout_0
output_dir: assets/runs/multihead
base_model: assets/base-model
language: es
entity_labels: [DISEASE, SYMPTOM]   # optional: every annotated type when left out
dropout: 0.1
max_length: 256
training_arguments:
  learning_rate: 3.0e-5
  num_train_epochs: 8
save_model: true
```

| Parameter | Default | Meaning |
|---|---|---|
| `entity_labels` | `None` | The heads, in order. Left out, every type annotated in the split's train and validation partitions, sorted, so every fold trains the same heads. A listed type annotated nowhere there raises. |
| `dropout` | `0.1` | Dropout before the heads. Replaces `architecture` and `architecture_kwargs`. |
| `overlap_policy` | `"merge_same_label_then_keep_longest"` | Resolves overlapping annotations **within each type**, as a single-label model for that type would. Different types are never resolved against each other. |

There is no `target_label`. Entities of a type with no head are left unlabelled, with a
warning that counts them.

The loss is each head's cross-entropy over its labelled positions, averaged over the heads
that have any in the batch, so every type weighs the same however frequent it is. Model
selection defaults to `span_strict_macro_f1` for the same reason; name another metric in
`training_arguments` to change it.

## Scoring

Every head is scored with nervaluate on its own type's spans, exactly as a single-label
model for that type is scored, so spans of different types are never matched against each
other. The metrics carry three views:

| Keys | Meaning |
|---|---|
| `head_<type>_span_strict_f1`, ... | One head, all four scenarios plus its token diagnostics. |
| `span_strict_f1`, ... | Micro: the heads' counts summed, precision, recall and F1 recomputed from the sums. |
| `span_<scenario>_macro_<precision\|recall\|f1>` | Macro: the mean over heads, each type weighing the same. |

Character metrics (`char_*`) score every type together, a character counting as correct
when a gold and a predicted span of the same type both cover it; entity-weighted character metrics (`dice_*`)
score every type together too, a span credited only by spans of its own type. At inference, gold of a
type with no head is left out, as `predict_entities` restricts gold to its `target_label`.

## Search

`search_multihead` sweeps the same search space, variants, seeds and top-k epochs as
`search_hyperparameters`, with `entity_labels` and `dropout` in place of `target_label`,
`architecture` and `architecture_kwargs`. Its objective defaults to `span_strict_macro_f1`,
and its `winner.yaml` is a ready-to-run `ner.train_multihead` config with the head
catalogue written out in full.

## The saved model

`save_model=True` writes the weights, the tokenizer and an `encoding.json` holding the head
catalogue in order, the windowing, the dropout, and the backbone's full configuration
under `backbone_config`. The saved directory therefore rebuilds the model without the base
model it was trained from: the backbone is built from that configuration with no
pretrained weights and the fine-tuned weights are loaded over it. A digest of the head
catalogue travels in the weights, so a directory whose `encoding.json` lists the heads in a
different order than the weights were trained under refuses to load.

## In code

```python
from lab.ner.multihead import (
    MultiHeadEncoder,                  # corpus -> rows labelled [position][head]
    MultiHeadForTokenClassification,   # the model
    build_multihead_compute_metrics,   # train(compute_metrics=...)
    evaluate_multihead_predictions,    # score logits against encoded rows
    score_heads,                       # score span tables per head, micro and macro
    load_multihead_model,
)
```

`train` and `predict_logits` batch multihead rows like any others, padding their
`[position][head]` labels, and `train` takes `build_multihead_compute_metrics` as its
`compute_metrics`, so a multihead model trains and predicts on the same code as a
single-label one.
