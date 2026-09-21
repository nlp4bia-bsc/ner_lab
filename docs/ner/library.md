# The pieces underneath

[`train_model`](train-model.md) is an assembly of four things: an `Encoder`, a `build_model`
call, `training_arguments`, and `train`. Each is public, takes DataFrames and objects rather
than paths, and applies no policy the orchestrator adds behind your back. Use them directly
when the orchestrator's directory layout, manifests or fold handling are in the way.

They have no YAML task for exactly that reason: there is no single path a config file could
point at.

A complete pipeline built from them:

```python
from transformers import AutoTokenizer

from lab.ner import (
    Encoder, build_model, training_arguments, train, build_compute_metrics,
)
from lab.core import read_corpus
from lab.ner.training import model_encoding

corpus = read_corpus("assets/splits/disease/train_val_80_20/train.parquet")
validation = read_corpus("assets/splits/disease/train_val_80_20/validation.parquet")

base_model = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
tokenizer = AutoTokenizer.from_pretrained(base_model)

encoder = Encoder(tokenizer, target_label="DISEASE", language="es", max_length=256)
train_rows = encoder.encode(corpus)
validation_rows = encoder.encode(validation)

model = build_model(base_model, encoder.label2id, encoder.id2label, architecture="crf")

result = train(
    model=model,
    tokenizer=tokenizer,
    train_rows=train_rows,
    validation_rows=validation_rows,
    training_arguments=training_arguments("assets/my_run", learning_rate=2e-5),
    compute_metrics=build_compute_metrics(validation_rows, tokenizer, encoder.id2label),
    save_model=True,
    model_encoding=model_encoding(encoder, base_model, architecture="crf"),
)
```

---

## `Encoder`

Corpus in, one row per window out. Configured once and reused across a split's partitions, so
every partition shares the same `label2id` and token budget.

```python
from lab.ner import Encoder

Encoder(
    tokenizer,
    target_label,
    language,
    max_length=256,
    overlap_policy="merge_same_label_then_keep_longest",
    strategy="greedy",
    context_tokens=None,
    min_sentence_tokens=4,
    documents_per_batch=500,
    require_target_label=True,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `tokenizer` | *required* | A `PreTrainedTokenizerBase`. Its special tokens are subtracted from `max_length` to get the content budget. |
| `target_label` | *required* | The entity type to tag. One label, three classes: `O`, `B-<label>`, `I-<label>`. |
| `language` | *required* | Sentence-segmentation language for `pysbd`. Never guessed. |
| `max_length` | `256` | Token budget per window, *including* special tokens. Raises if the tokenizer's specials leave no room. |
| `overlap_policy` | `"merge_same_label_then_keep_longest"` | How overlapping gold spans are resolved. See below. |
| `strategy` | `"greedy"` | `"greedy"`, `"context"`, or a callable `(sentence_ranges, tokens, entities, text, max_content_length) -> windows`. |
| `context_tokens` | `None` | Flanking context tokens per window. Required with, and only valid with, `strategy="context"`. |
| `min_sentence_tokens` | `4` | Sentences shorter than this are merged into their neighbour. |
| `documents_per_batch` | `500` | Batch size for `encode_parquet`, which streams so a large corpus is never fully loaded. |
| `require_target_label` | `True` | Reject a corpus in which `target_label` never occurs — for training, a typo would otherwise produce silently all-`O` rows. Inference turns it off, since unannotated documents legitimately carry no entity. |

Methods:

| Call | Returns |
|---|---|
| `encode(corpus)` | Encode an in-memory corpus DataFrame into a row-per-window DataFrame. |
| `encode_parquet(path)` | The same, streaming a corpus parquet in `documents_per_batch` chunks. |
| `encode_document(doc_id, text, entities)` | One document's window rows, as a list of dicts. |

Attributes: `label2id`, `id2label`, `vocabulary`, `max_content_length`, plus every constructor
argument.

**Overlap policies.** Entities are stored in the corpus as annotated; resolving them is a
modelling choice made here, so several policies can be compared without regenerating the
corpus.

| Value | Behaviour |
|---|---|
| `"none"` | Leave overlaps alone. |
| `"error"` | Raise on any overlap. |
| `"keep_longest"` | Keep the longest span of each overlapping cluster. |
| `"keep_shortest"` | Keep the shortest. |
| `"merge_same_label_then_keep_longest"` | Merge overlapping spans that share a label into their union first, then keep the longest of what remains. The default. |

`lab.ner.encoding.describe_encoder(encoder)` returns the configuration as plain data, and
`encoder_from_description(...)` rebuilds one from it — that pair is what lets inference window
its input exactly as training did.

---

## `build_model`

```python
from lab.ner import build_model

build_model(base_model, label2id, id2label, architecture="linear", **architecture_kwargs)
```

| Parameter | Default | Meaning |
|---|---|---|
| `base_model` | *required* | Pretrained encoder, by hub name or local path. |
| `label2id` | *required* | Pass the `Encoder`'s, so the model's label ids are the ones the rows were encoded against. |
| `id2label` | *required* | The same, inverted. |
| `architecture` | `"linear"` | `"linear"`, `"crf"`, or a callable `(base_model, label2id, id2label, **kwargs) -> model`. |
| `**architecture_kwargs` | — | Forwarded to the chosen architecture, e.g. `dropout=0.2`. |

`"linear"` is the stock `AutoModelForTokenClassification` head. `"crf"` is
`lab.ner.models.crf.CRFForTokenClassification`, which decodes by Viterbi under BIO transition
constraints. A callable must return a model whose `forward` accepts
`input_ids`/`attention_mask`/`labels` and returns an object with `loss` and `logits`.

---

## `training_arguments`

```python
from lab.ner import training_arguments

training_arguments(output_dir, **overrides) -> TrainingArguments
```

Builds `transformers.TrainingArguments` from `lab.ner.training.DEFAULTS`, overridden by
whatever you pass. Every keyword `TrainingArguments` accepts is accepted here, so this is a
starting point rather than a wrapper — build your own and hand it to `train` if the defaults
are in the way.

The defaults:

| Key | Value | | Key | Value |
|---|---|---|---|---|
| `learning_rate` | `3e-5` | | `save_total_limit` | `1` |
| `num_train_epochs` | `10` | | `load_best_model_at_end` | `True` |
| `per_device_train_batch_size` | `16` | | `metric_for_best_model` | `span_strict_f1` |
| `per_device_eval_batch_size` | `16` | | `greater_is_better` | `True` |
| `gradient_accumulation_steps` | `1` | | `seed` | `42` |
| `weight_decay` | `0.01` | | `optim` | `adamw_torch` |
| `max_grad_norm` | `1.0` | | `report_to` | `none` |
| `lr_scheduler_type` | `linear` | | `dataloader_num_workers` | `0` |
| `logging_strategy` / `eval_strategy` / `save_strategy` | `epoch` | | `ddp_find_unused_parameters` | `False` |

`metric_for_best_model` defaults to `span_strict_f1`, which requires the `compute_metrics`
built by `build_compute_metrics`. Pass `metric_for_best_model="eval_loss"` with
`greater_is_better=False` to train without any span scoring.

**Mixed precision is not in the defaults.** It is resolved per device: bf16 wherever it
exists, fp16 on pre-Ampere cards that lack it, neither without CUDA. bf16 carries fp32's
exponent range, so a gradient never underflows and no loss scaling is needed, and tensor-core
throughput is identical on every device that has both — so there is nothing to trade against.
Naming either `fp16` or `bf16` in the overrides turns the resolution off and takes exactly
what you passed. Whichever way it lands is recorded in the run manifest beside the GPU that
ran it.

---

## `train`

One training run. Everything that decides *what* is trained is already in the objects you
pass; these parameters decide how the run itself behaves.

```python
from lab.ner import train

train(
    model,
    tokenizer,
    train_rows,
    validation_rows,
    training_arguments,
    compute_metrics=None,
    train_compute_metrics=None,
    trainer_class=None,
    early_stopping_patience=5,
    pad_to_multiple_of=8,
    ignore_index=-100,
    metrics_scope="eval",
    save_model=False,
    track_resources=False,
    run_metadata=None,
    model_encoding=None,
) -> TrainingResult
```

| Parameter | Default | Meaning |
|---|---|---|
| `model` | *required* | Anything `build_model` returns, or your own. |
| `tokenizer` | *required* | The tokenizer the rows were encoded with. |
| `train_rows` | *required* | `Encoder` output for the training partition. |
| `validation_rows` | *required* | `Encoder` output for the validation partition. |
| `training_arguments` | *required* | A `TrainingArguments`. Its `output_dir` is where this run writes. |
| `compute_metrics` | `None` | Handed straight to the Trainer. Build one with `build_compute_metrics`. |
| `train_compute_metrics` | `None` | Used for the extra training-set pass under `metrics_scope="both"`. |
| `trainer_class` | `None` | Defaults to `CRFTrainer` for a CRF model, `Trainer` otherwise. |
| `early_stopping_patience` | `5` | Epochs without improvement before stopping. `None` disables it. |
| `pad_to_multiple_of` | `8` | Pad batches to a multiple of this. |
| `ignore_index` | `-100` | Label id excluded from the loss and from scoring. |
| `metrics_scope` | `"eval"` | `"both"` also scores the training set each epoch, at the cost of a second pass. |
| `save_model` | `False` | Keep the weights. Otherwise the selection checkpoints are removed afterwards, including when training raises. |
| `track_resources` | `False` | Wall clock, energy, emissions and peak VRAM, via codecarbon. |
| `run_metadata` | `None` | Extra keys recorded in `training_summary.json`. |
| `model_encoding` | `None` | Written beside saved weights as `encoding.json`. Build it with `lab.ner.training.model_encoding`. |

**Returns** a `TrainingResult` with `trainer`, `model`, `metrics`, `epoch_metrics`, `summary`,
`output_dir`, `resources` and `paths`.

`epoch_metrics.parquet` and `training_summary.json` are always written to
`training_arguments.output_dir`. Without `model_encoding`, a saved directory cannot be loaded
by `lab.ner.inference` — and a CRF model, which the Trainer saves as a bare state dict, cannot
be rebuilt at all. `save_model=True` with early stopping requires
`load_best_model_at_end=True`, or the saved weights would be the last epoch's rather than the
best epoch's; `train` raises rather than letting that through.

---

## `build_compute_metrics` and `evaluate_predictions`

Span metrics, SemEval 2013 Task 9.1 style via `nervaluate`, plus token diagnostics. The two
take the same arguments; `build_compute_metrics` returns a callable for
`train(compute_metrics=...)`, `evaluate_predictions` scores an array you already have.

```python
from lab.ner import build_compute_metrics, evaluate_predictions

build_compute_metrics(
    rows, tokenizer, id2label,
    ignore_index=-100, min_overlap_percentage=40.0,
    include_tokens=True, include_by_entity=True, include_confusion=False,
) -> Callable

evaluate_predictions(rows, predictions, tokenizer, id2label, ...) -> dict
```

| Parameter | Default | Meaning |
|---|---|---|
| `rows` | *required* | The same frame, in the same order, that produced the dataset being scored. |
| `predictions` | *required* (`evaluate_predictions` only) | Logits or predicted label ids. |
| `tokenizer` | *required* | Used to map tokens back to character offsets. |
| `id2label` | *required* | The `Encoder`'s, so ids mean what they meant at encoding time. |
| `ignore_index` | `-100` | Label id excluded from scoring. |
| `min_overlap_percentage` | `40.0` | Character overlap a predicted span needs against a gold one to count in the partial scenario. Strict and exact ignore it. |
| `include_tokens` | `True` | Add token-level precision/recall/F1. |
| `include_by_entity` | `True` | Add per-entity-type token metrics. |
| `include_confusion` | `False` | Add the token confusion matrix. |

Span metrics are the canonical result and are always computed, in four scenarios — `strict`,
`exact`, `partial`, `ent_type` — flattened to keys like `span_strict_f1`, which is what
`metric_for_best_model` defaults to. Character metrics come with them: `char_precision`,
`char_recall`, `char_f1` and the counts `char_correct` / `char_missed` / `char_spurious`,
the character being the unit, so a boundary that is nearly right scores nearly full marks
instead of zero. Both families are `lab.core.score_spans` and `lab.core.score_characters`
applied to the spans reconstructed from the rows; `lab.ner.evaluation.span_metrics` is the
call that joins them, and `token_diagnostics` the call that adds the token half.
