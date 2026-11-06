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
| `language` | *required* | Sentence-segmentation language, an ISO 639-1 code: it picks pysbd, Punkt or ICU. Never guessed. |
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

**Words.** Only a word's first subword carries a label; its other subwords, and any token that
is whitespace throughout (SentencePiece's bare `▁`, a byte-level token for a repeated space),
get `-100`. A word is a Unicode UAX #29 word, as ICU's word break iterator defines it, never
the tokenizer's own pre-tokenization, so every model is labelled on the same words: each
punctuation character is a word, `12.5` and `can't` are one, and so is `Ιστορικό:ΧΚΜ` — an
entity written straight after a colon cannot be labelled. `lab.core.split_into_words(text)`
returns them.

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

## Multi-Head BIO

`architecture="multihead"` extends the single-label contract (D132–D135); `linear`,
`crf`, their three-dimensional outputs and their existing checkpoints keep their semantics.
One `AutoModel` backbone runs once per batch. A dropout and a single
`Linear(hidden_size, 3 * K)` projection implement K independent categorical heads:

| Value | Shape | Meaning |
|---|---|---|
| Hidden states | `[B, L, H]` | Shared Transformer representation |
| Logits | `[B, L, K, 3]` | Independent O/B/I logits for each entity |
| Labels / predictions | `[B, L, K]` | `0=O`, `1=B`, `2=I`; `-100` masks labels |

The catalogue is a sorted list of the labels actually present in training annotations,
without normalization. It never comes from `CANONICAL_LABELS`. `train_model` and HPO discover
it automatically when `target_label=None`; do not supply entity names in YAML.
For direct composition, first encode **train**, then reuse that encoder for validation:

```python
from lab.ner import Encoder, build_model, training_arguments, build_compute_metrics, train
from lab.ner.training import model_encoding

encoder = Encoder(tokenizer, target_label=None, language="es", max_length=256)
train_rows = encoder.encode(train_corpus)  # discovers and fixes entity_labels
validation_rows = encoder.encode(validation_corpus)  # never adds heads
model = build_model(
    base_model, encoder.label2id, encoder.id2label,
    architecture="multihead", entity_labels=encoder.entity_labels, dropout=0.1,
)
result = train(
    model=model, tokenizer=tokenizer, train_rows=train_rows,
    validation_rows=validation_rows,
    training_arguments=training_arguments("assets/run", architecture="multihead"),
    compute_metrics=build_compute_metrics(validation_rows, tokenizer, encoder.id2label),
    save_model=True,
    model_encoding=model_encoding(encoder, base_model, architecture="multihead"),
)
```

`entity_labels` is an optional final argument on `Encoder`, for a catalogue already discovered
from authorized training data. `lab.ner.encoding.discover_entity_labels(train_corpus)` exposes
the discovery step. An empty training catalogue raises. `encode_document` needs a configured
catalogue; it cannot infer all training types from one document. `encode_parquet` in multihead
mode reads the corpus into memory, unlike the streaming single-label path.

### Overlap and masking

Input IDs, offsets, UAX #29 word IDs, attention masks and windows are shared. One row is
written per window, with only its labels gaining an extra axis. First subwords are supervised;
continuations, whitespace-only tokens, special tokens and context flanks are `-100` for every
head. A multihead collator pads label **vectors** to `[B,L,K]`, including left padding;
`MultiHeadTrainer` realigns left-padded predictions before reconstruction.

Cross-type overlap, nesting, and different types on the same token are preserved. The
existing `build_iob2_labels` and `bio_to_spans` operate independently for each head; there is
no cross-type competition, merging or NMS. `overlap_policy=None` resolves to `"error"` for
Multi-Head and to the historical `"merge_same_label_then_keep_longest"` for a single label.
Intraclass overlaps raise with document and spans by default. Explicit `keep_longest`,
`keep_shortest` or `merge_same_label_then_keep_longest` apply **within each label**, warning
when annotations change. `none` is refused because it cannot faithfully represent such gold.

BIO still supervises whole words. Boundaries inside a UAX #29 word or indivisible tokenizer
unit cannot in general be recovered exactly. Intraclass nesting needs a span decoder or
multiple tracks per type, which this architecture does not implement.

### Joint loss and decoding

Cross entropy is vectorized with `ignore_index=-100`. Each head's loss is its mean over
valid positions, then nonempty heads are averaged uniformly. A head with no valid positions
contributes nothing; an entirely masked batch has a differentiable zero loss. There are no
default class weights, head weights or frequency heuristics. One forward and backward train
all heads jointly. Gradient checkpointing delegates to the backbone; batching, accumulation,
early stopping and precision remain Trainer concerns.

Inference takes argmax over the final **class** axis. The common BIO decoder repairs an
orphan I as a new span, independently per head, as it does for `linear`; this is not CRF
Viterbi or constrained-logit decoding. A span score is the mean softmax probability of its
selected BIO tags. Duplicate `(filename,label,start,end)` predictions keep the highest score;
different overlapping spans survive. No Multi-Head CRF or local CNN is included.

### Strict metrics

Multi-Head uses exact identity sets `(filename,label,start,end)`, avoiding nervaluate's greedy
matching order dependence on overlapping spans. Duplicate gold and predictions count once.
The existing character scorer is reused. For Multi-Head only, the metrics are:

- `span_strict_precision__LABEL`, `span_strict_recall__LABEL`, `span_strict_f1__LABEL`,
  and `span_strict_tp__LABEL`, `span_strict_fp__LABEL`, `span_strict_fn__LABEL`.
- `span_strict_macro_precision`, `span_strict_macro_recall`, `span_strict_macro_f1`:
  arithmetic means over **all trained heads**. A head with no gold and no predictions scores
  zero and remains in the denominator.
- `span_strict_micro_*`: counts aggregated globally, then precision/recall/F1 calculated.
  `span_strict_precision`, `span_strict_recall`, `span_strict_f1` are micro aliases.
- Existing `char_*`; optional existing token diagnostics with `__LABEL` suffixes.
- `unknown_gold_entities`: number of annotated entities whose type has no trained head.

Unknown validation types warn, get their own zero-recall metrics and count as micro false
negatives. They do not create heads or enter the trained-head macro denominator. Per-fold
catalogues and unknown counts must therefore accompany macro comparisons. `train_model`
discovers each fold's catalogue from **that fold's train only**; it never reads the holdout.
A type missing from one fold's train is unknown there even if another fold trains it.

Encoded rows retain original gold and the catalogue in `DataFrame.attrs` so validation can
count original offsets and unknown gold rather than silently lose them in BIO projection.
Scoring restricts this gold to the documents still present in the rows. A subset of windows
for a document still scores against that document's full gold, so omitted entities count as missed. If only rows remain, reconstruction scores representable BIO gold;
pass `entity_labels=...` explicitly to scoring helpers after discarding catalogue attrs.
This differs deliberately from the historical single-label training score, which always
reconstructs gold. `min_overlap_percentage` retains its partial-scoring meaning on the
single-label path; it does not tune exact Multi-Head metrics. Partial/type/exact SemEval
scenarios are not reported for Multi-Head.

`train_model` and HPO default to `span_strict_macro_f1` only when no selection metric is
specified. A mapping can select `span_strict_micro_f1`, `span_strict_f1__LABEL` or `eval_loss`
(the latter defaults to minimization). An explicit `TrainingArguments` instance is used as
supplied. Low-level callers can use `training_arguments(..., architecture="multihead")` or
set `metric_for_best_model` themselves; a generic arguments instance cannot reveal intent.

### Checkpoints

A Multi-Head `best_model` contains the tokenizer, all backbone and head weights, and
`encoding.json` format 2: original base model, architectural parameters, backbone config,
ordered `entity_labels`, `num_heads`, `head_to_label`, BIO schema and the full encoder
configuration. The state dict also binds the head order with a catalogue digest. Loading
validates mappings, dimensions and digest and uses strict state-dict loading; it does not
need the original backbone directory or a network download. Old `linear`/`crf` encodings
remain readable. A bare Multi-Head checkpoint without encoding metadata is not loadable;
pass `model_encoding` when saving through `train`.

## Unified ensembles underneath

`lab.ner.ensemble.tables.MemberPredictions` holds sparse per-type word distributions and
original decoded spans from one inference. `align_entities` groups eligible members by
label and reuses `WordVotes`; `decode_entities` applies independent two-stage BIO decoding.
`combine_spans` preserves exact interval identities for union/consensus/intersection.
`search_entity_parameters` fits scoped values on cached predictions with a uniform baseline
and sparse-gold fallback. These helpers live in their concern modules under
`lab.ner.ensemble`; the public stage functions remain `fit_ensemble` and `predict_ensemble`.
See [the guide](ensemble.md) for supported combinations and provenance requirements.
