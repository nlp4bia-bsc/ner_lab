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

encoder = Encoder(
    tokenizer, target_label="DISEASE", language="es", max_length=256,
    include_word_mask=True,
)
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
| `architecture` | `"linear"` | `"linear"`, `"crf"`, `"multihead"`, `"multihead_crf"`, `"multilabel"`, or a callable `(base_model, label2id, id2label, **kwargs) -> model`. |
| `**architecture_kwargs` | — | Forwarded to the chosen architecture, e.g. `dropout=0.2`. |

`"linear"` is the stock `AutoModelForTokenClassification` head. New `"crf"` models use
`lab.ner.models.word_crf.WordCRFForTokenClassification`, which decodes by Viterbi under BIO transition
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
legacy `crf` checkpoints keep their semantics. New CRF training uses the word-level
constrained implementation described below; classic emissions remain three-dimensional.
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
different overlapping spans survive. `multihead_crf` adds independent constrained chains as described below. No local CNN is included.

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


## Word-level constrained CRF (version 2)

`architecture="crf"` now builds `WordCRFForTokenClassification`; `multihead_crf` uses the
same composition with one shared Transformer and independent transition parameters per type.
The Multi-Head catalogue is discovered from training annotations only, separately in each
fold. All existing training, prediction, HPO and ensemble orchestrators are reused.

The common `models.chain.ConstrainedChain` accepts `[B,T,C]` or `[B,T,K,C]`. The model's
emissions retain token positions: `[B,L,3]` or `[B,L,K,3]`. Gold and restored Viterbi tags
are `[B,L]` or `[B,L,K]`; absent decoded positions are `-100`. One projection produces all
heads, and one backbone forward serves the batch.

For a legal path, the score is its start potential, emissions, successive transition
potentials and final potential. NLL is `log_partition - gold_score`, where partition sums
**only legal paths**. BIO2 disallows START→I and O→I; B→I, I→I, B→B and ending in B are
legal. The core also supports flat B-X/I-X vocabularies and prohibits cross-type inside
transitions. The integrated models use three states per head.

Boolean buffers mask the effective potentials with exact `-inf`, identically in partition,
Viterbi and forward-backward. Parameters are never clamped or modified by gradient hooks.
An all-unreachable reduction has an explicit zero-gradient branch. Dynamic programming
runs in FP32 with autocast disabled, even when Transformer emissions use reduced precision.

### Attention, eligibility and gold

Low-level callers must use `Encoder(..., include_word_mask=True)` for new CRFs. Its
`crf_word_mask` selects first subwords in the window core; specials, continuation subwords,
padding and context flanks are excluded. `attention_mask` still exposes all real tokens to
the Transformer. `WordChainCollator` pads these two masks independently, with `-100` labels.
`pack_words` compacts all batch/head chains with indexed tensor operations, and `PackedWords`
restores decisions and distributions to their original token positions. No retokenization
or character-level dense annotations are introduced.

New word-CRF window geometry uses text only, independently of annotations. The existing
window-local IOB2 tagger starts a clipped entity with B at a new supervised core; it does
not repair gold inside the loss. Long entities crossing cores can consequently be split
into fragments, so maximum length/context remain modelling choices. A classic word CRF
resolves only annotations of its own target type before tagging. Legacy windowing and
legacy overlap resolution remain available for version-1 CRFs.

Gold must be completely observed at every eligible word. Training rejects unknown whole
words and illegal BIO2 with document, window, entity, position and transition diagnostics.
The core reports batch/head coordinates when called directly. It never drops an unknown
whole word and joins its neighbours. Partial annotation, independent missing-gold segments
and partially observed partition losses are not implemented. Entirely empty chains have
absent decisions and zero probability mass; inactive heads are omitted from the loss.

### Options and loss

The following explicit `architecture_kwargs` apply to version-2 `crf`/`multihead_crf`:

| Argument | Default | Meaning |
|---|---|---|
| `dropout` | `0.1` | Dropout before the shared emission projection. |
| `constrain_transitions` | `True` | Exact BIO2 restrictions; `False` is an unconstrained ablation. |
| `aux_ce_weight` | `0.0` | Nonnegative auxiliary local emission CrossEntropy weight. |
| `head_weights` | `None` | Finite nonnegative weights keyed by entity name; omitted names weigh 1. |
| `loss_reduction` | `"word_mean"` | Sum sequence NLL per head / supervised words, then weighted mean of active heads. Only this reduction is supported. |
| `loss_type` | `"crf"` | `"cross_entropy"` freezes zero transitions and trains local CE, yielding the constrained-decoding baseline. |
| `tag_scheme` | `"BIO2"` | Only BIO2 is implemented; BIOES/BILOU fail explicitly. |
| `crf_version` | `2` | Set `1` with classic `crf` only to reproduce the historical implementation. |

The objective is `word_normalized_crf_nll + aux_ce_weight * balanced_local_ce`. Empty batches
return a differentiable zero. The CE baseline uses only balanced local CE and no trainable
transition parameters; `aux_ce_weight` must remain zero in that mode.

`WordCRFTrainer` evaluates Viterbi paths, including early stopping and best-checkpoint
selection. Multi-Head defaults to `span_strict_macro_f1`; classic defaults stay unchanged.
An explicit metric, including `eval_loss` with minimization, takes precedence.

### Decisions, probabilities and checkpoints

`forward` returns actual emissions and optional loss. To reuse them, call
`model.structured_from_emissions(emissions, crf_word_mask, attention_mask,
probabilities=True)`. Its typed `StructuredPrediction` carries restored Viterbi `tags` and
optional `log_marginals`. Without probabilities, forward-backward is not computed.
`decode_from_emissions` and `marginals_from_emissions` also require the explicit word mask.

Trainer's historical `predict_logits` interface transports Viterbi decisions as compatibility
logits; these values are **not confidence**. `WordMarginalTrainer` carries log marginals and
Viterbi in separate last-axis blocks. `predict_entities` uses real marginal probabilities
of the Viterbi-selected tags for mean span confidence. This is not a full mention probability
or a claim of calibration. Zero evidence is not converted to confident O.

Version-2 metadata records `crf_version`, the structural mask policy, ordered heads, backbone
configuration and all model options; normal checkpoint tensors hold learned start,
transition and end parameters. Loading is standalone and strict: catalogue digests, shapes,
constraint buffers and named weights must agree with metadata. Old CRF metadata without a
version loads the original class with its historical attention-token semantics and its
existing parameter names. It is not silently migrated to the new word chain.

See [verification and measurements](../../verification/CONSTRAINED_CRF_STATUS.md).


## Independent binary Multi-Label BIO

`architecture="multilabel"` is a separate alternative to categorical `multihead` and
`multihead_crf`. One shared Transformer runs once per batch, followed by dropout and one
`Linear(H,2*K)` projection. The independent binary channels are B=0 and I=1. O has no
trainable logit: it is implicit when neither boundary channel is active.

| Value | Shape | Meaning |
|---|---|---|
| Hidden states | `[B,L,H]` | One shared Transformer representation |
| Raw logits / binary labels | `[B,L,K,2]` | Independent B/I channels per entity |
| Activated scores | `[B,L,K,2]` | Independent sigmoid evidence, not a categorical distribution |
| Decoded BIO | `[B,L,K]` | One O=0/B=1/I=2 track per entity; absent positions -100 |

Gold encodes O as `[0,0]`, B as `[1,0]`, I as `[0,1]`; ignored positions are `[-100,-100]`.
The Encoder's generic BIO vocabulary remains the **decoder** vocabulary; `label_schema`
distinguishes binary training targets. Do not pass categorical gold to the binary model.

```python
from lab.ner import Encoder, build_model, train, training_arguments, build_compute_metrics
from lab.ner.training import model_encoding

encoder = Encoder(tokenizer, None, "es", max_length=256, label_schema="multilabel")
train_rows = encoder.encode(train_corpus)
validation_rows = encoder.encode(validation_corpus)
model = build_model(
    base_model, encoder.label2id, encoder.id2label, architecture="multilabel",
    entity_labels=encoder.entity_labels, loss="focal", focal_gamma=2.0,
)
result = train(
    model, tokenizer, train_rows, validation_rows,
    training_arguments("assets/run", architecture="multilabel"),
    compute_metrics=build_compute_metrics(
        validation_rows, tokenizer, encoder.id2label, multilabel_decoding=model.decoding,
    ),
    save_model=True, model_encoding=model_encoding(encoder, base_model, "multilabel"),
)
```

High-level `train_model` and HPO select this representation automatically. Discovery is sorted
and train-only, independently per fold. Validation/inference never add labels. Unknown types
warn, count as micro FN and stay outside the trained-catalogue macro denominator, as for
Multi-Head. Negative-only heads remain valid training supervision; fully unsupervised heads
are excluded from loss reduction. All-ignored batches return differentiable zero.

### Binary losses

`architecture_kwargs` takes explicit constructor arguments:

| Setting | Default | Semantics |
|---|---|---|
| `loss` | `"bce"` | `bce`, `weighted_bce`, `focal`, `asymmetric` |
| `pos_weight` | `None` | Weighted BCE only: entity name → positive B/I weight pair |
| `pos_weight_policy` | `"none"` | `train` estimates weighted-BCE weights; weighted BCE without explicit weights requests train estimation automatically |
| `weight_smoothing`, `max_pos_weight` | `1.0`, `20.0` | Automatic weight = clip((negative+smoothing)/(positive+smoothing), 1, maximum) per channel, **train only** |
| `focal_gamma`, `focal_alpha` | `2.0`, `None` | Focal factor `(1-p_target)^gamma`; optional alpha for positive and 1-alpha for negative terms |
| `gamma_pos`, `gamma_neg`, `clip` | `0.0`, `4.0`, `0.05` | ASL uses independent positive/negative exponents and clips negative probability `(1-p+clip)` at 1 |
| `training_temperature` | `1.0` | Positive constant scaling logits for training loss only; separate from post-hoc temperature |
| `dropout` | `0.1` | Before the compact output projection |

BCE uses `binary_cross_entropy_with_logits`, not sigmoid followed by BCELoss. Focal gamma=0
and alpha=None equals BCE. ASL with both exponents zero and clip=0 equals BCE; focal
modulating factors remain differentiable. Weighted BCE and ASL mechanisms cannot be combined.
The objectives use FP32 masked element losses, mean valid channels/words per entity, then
mean over supervised entities. Padding, special/subwords and context never contribute.
No-positive entities still supply negatives; they are not dropped from the mean.

For low-level automatic weighted BCE, use
`lab.ner.training.binary.binary_architecture_kwargs(settings, train_rows, entity_labels)`
before `build_model`. An unresolved train-estimation request fails on supervised forward.
`BinaryLoss` is a typed schema for conditional HPO loss families.

### Score transformation and BIO decisions

`BinaryDecoding` in `lab.ner.models.binary_decoding` separates post-training decisions from
loss. `architecture_kwargs` accepts its explicit fields:

| Setting | Default | Semantics |
|---|---|---|
| `activation` | `"sigmoid"` | `sigmoid`, `temperature_sigmoid`, `hard_sigmoid` |
| `temperature` | `1.0` | Positive global scalar or entity-name mapping; temperature_sigmoid only |
| `decoder` | `"threshold_bio"` | Threshold BIO or `constrained_viterbi` |
| `entity_threshold` | `0.5` | Global scalar or entity-name mapping, threshold BIO only |
| `entity_bias` | `0.0` | Global scalar or entity-name mapping, Viterbi only; added to B and I potentials |

Standard scores are independent sigmoid(z). Temperature scaling uses sigmoid(z/T).
Hard sigmoid clamps `(z+3)/6` to [0,1] **for decoding only**; BCE training is unchanged.
Neither hard scores nor temperature selection by span F1 establishes calibration. Temperature
is monotone and can partly duplicate threshold effects; it is not a new neural architecture.

Threshold BIO activates an entity when `1-(1-p_B)*(1-p_I) >= entity_threshold`, inclusively.
An active word starts B after O; it also starts B when `p_B > p_I`. Otherwise it continues I,
with ties continuing an open mention. An isolated I-like prediction starts B. Entities are
vectorized together, with previous eligible-word positions derived cumulatively; there is
no per-entity inference pipeline. Structural `word_mask` comes from input/core geometry,
independently of gold values. It excludes specials, secondary subwords and context. New
binary window geometry, like new CRFs, depends on text rather than annotations.

Viterbi uses the existing constrained BIO2 kernel with zero, nontrainable transition scores
and local log weights:

```
O: logsigmoid(-z_B) + logsigmoid(-z_I)
B: logsigmoid( z_B) + logsigmoid(-z_I)
I: logsigmoid(-z_B) + logsigmoid( z_I)
```

Temperature, when selected, scales z first. `entity_bias` controls entity evidence under
Viterbi; `entity_threshold` does not apply there. This is a constrained decoder, not a
trained CRF. For hard sigmoid, only the legal-state adapter clamps scores to
`[1e-7,1-1e-7]` before logarithms, making saturated B+I deterministic and finite.

### Spans, metrics and ensembles

BIO reconstruction and original offsets are shared with Multi-Head. Spans of different
labels coexist, including nesting and identical intervals. Intraclass conflicts default to
an informative error. Explicit lossy policies remain within each label and warn. A single
track per type cannot represent arbitrary intraclass nested/overlapping spans. Word-internal
boundaries and entities crossing window cores may be unrepresentable. Original gold remains
in span scoring, with `gold_unrepresentable_entities`, `gold_original_entities` and
`gold_reconstructed_entities` exposing the projection loss when corpus annotations exist.
Window-local clipped mentions start with B before model training.

Strict per-type/macro/micro and character metrics reuse exact identity-set scoring. The
selection default is `span_strict_macro_f1`; explicit micro, per-type F1 and loss choices are
preserved. Binary token diagnostics count independent B/I channels at .5 and do not select
the model. Low-level metrics with custom decoding must pass
`multilabel_decoding=model.decoding` to match the model's persisted decisions. Attrs on
encoded rows distinguish schema/catalogue; retain them when composing the pure path.

The ensemble adapter constructs legal local weights
`q_O=(1-p_B)*(1-p_I)`, `q_B=p_B*(1-p_I)`, `q_I=(1-p_B)*p_I`, then normalizes in stable log
space. This conditions independent Bernoulli outputs on absence of simultaneous B+I.
It never uses `1-p_B-p_I`, nor applies a classifier softmax across entities. Hard votes use
the configured decoder's tags; soft votes use this **derived** O/B/I distribution. Thus a
single member's soft-vote decisions can differ from its threshold decoder. Unsupported
labels remain absent evidence. Span score is the mean derived probability of the chosen
BIO states; it is neither sigmoid entity evidence nor a full calibrated mention probability.
Validation and inference use that same confidence definition. Exact duplicates keep the
largest score; distinct overlapping intervals are retained.

### Checkpoints and post-hoc replay

`encoding.json` format 3 records architecture, original backbone, ordered catalogue/count/
head mapping, binary schema, B/I channel indices, exact loss/weights/training-temperature,
activation/decoder/threshold/temperature and encoder/window/overlap settings. The full
backbone config and ordinary state tensors make loading standalone. Catalogue digests,
positive-weight buffers and projection shapes are checked; incompatible loading fails.
Legacy format-1/2 models and ensembles retain their existing paths.

A decoder-only HPO run additionally writes checkpoint-bound `decoding.json`. Apply it
explicitly with `predict_entities(..., decoder_path="sweep/decoding.json")`; inference
checks weights, metadata and tokenizer fingerprints, then replays fixed choices without
fitting. A retrained checkpoint needs refitting/revalidation. Ensembles consume the decoder
stored in each member checkpoint; external decoder overrides are not ensemble member
arguments. To deploy selected choices in an ensemble, train/save a member with those
explicit decoder arguments, revalidate it and fit the ensemble again.

### Execution devices and prediction memory

The shared Transformer, vectorized heads, binary losses, CRF dynamic programming and binary
tensor decoders operate on the model's device. Native Multi-Label entity inference and ensemble
adapters decode each batch there before transferring predictions to CPU. The internal adapter
carries six channels (two raw logits, three derived probabilities and a decoded tag);
ordinary `predict_logits` still returns `[B,L,K,2]` raw logits. This extra transport keeps token
diagnostics without a second Transformer pass. Span tables, text offsets, strict scoring,
ensemble voting and cached decoder-HPO selection remain CPU computations.

Without explicit TrainingArguments, inference follows the training precision policy (BF16
where CUDA supports it, FP16 otherwise, neither on CPU), offloads every eight batches and
concatenates output arrays once on the host. This bounds device prediction accumulation;
the final dense output still scales with corpus windows. Explicit `training_arguments`
control precision and `eval_accumulation_steps`/`eval_do_concat_batches` as before.

`train_model(devices=1)` pins one replica, including each copied fold's arguments;
`devices="all"` opts into single-process DataParallel and changes the effective batch and
schedule. Ray runs independent trials concurrently with one reserved GPU per trial.
Ensemble members run sequentially to bound model memory. This is not a promise that all
stages run concurrently or on CUDA, or that DDP/FSDP orchestration is qualified.

For Trainer inference select physical devices with `CUDA_VISIBLE_DEVICES` before Python
starts, then use `device="cuda"` or `"auto"`. `device="cuda:1"` is rejected by `predict_logits`
instead of being silently replaced by the Trainer's first visible GPU. `load_model` alone
still accepts valid indexed devices. See the
[acceleration audit](../../verification/ACCELERATION_REVIEW.md) and hardware qualification commands.
