# Analyse a model's errors

```python
from lab.ner import analyze_predictions

result = analyze_predictions(
    predictions="assets/predictions/predictions.parquet",
    reference="assets/datasets/disease/test.parquet",
    training="assets/datasets/disease/train.parquet",
    target_label="DISEASE",
    output_dir="assets/analysis/disease",
)

result.metrics                     # the strict, character and entity-weighted character metrics predict_entities reports
result.tables["gold_subgroups"]    # recall by exposure, frequency, length, casing, traits
result.events                      # one row per gold span or prediction the scorer judged
result.summary()                   # "strict F1 = 0.7912 [0.7731, 0.8084] | 41 boundary, 52 missed, 38 spurious, 2 uncredited exact"
```

The strict score says how good a model is; this says how it is wrong and where. Every strict
error is put in a diagnostic unit — a boundary error, a fragmentation or merge, a miss, a
spurious prediction — and every mention is set against the training mentions it could have
learned from, so the question "does the model find mentions it never saw?" has an answer with
an interval on it.

The analysis was first written by Fernando Gallego; this is its port to the library's
conventions, recorded as D127 and D132–D143 in [the decisions register](../dev/DECISIONS.md).

| Parameter | Default | Meaning |
|---|---|---|
| `predictions` | *required* | The span table `predict_entities` or `predict_ensemble` wrote, as `predictions.parquet` or `predictions.tsv`, or the frame itself. A `score` column is kept when present. |
| `reference` | *required* | The gold: a corpus (frame or parquet) or an annotation TSV. |
| `training` | *required* | The mentions the model learned from, in the same forms. Usually the training corpus. |
| `target_label` | *required* | The label analysed. Gold and training are restricted to it; a prediction of another label raises. |
| `output_dir` | *required* | Where everything is written. Created if missing; a rerun overwrites it. |
| `similarity_threshold` | `0.80` | Normalized Levenshtein similarity at which an unseen mention counts as lexically similar. |
| `overlap_threshold` | `0.60` | Character Dice at which a miss and a spurious prediction pair into a boundary error. |
| `sensitivity_thresholds` | `(0.4, 0.5, 0.6, 0.7, 0.8)` | Thresholds the pairing is rerun at, to show how much the taxonomy owes to the one chosen. |
| `min_overlap_percentage` | `40.0` | nervaluate's partial-match overlap, as everywhere else in the library. |
| `bootstrap_samples` | `2000` | Document resamples behind every interval. |
| `confidence` | `0.95` | Width of the percentile intervals. |
| `random_state` | `20260701` | Seeds the resamples. `--random-state` sets it from the command line. |
| `figures` | `True` | Draw the figures. The tables are written either way. |
| `figure_format` | `"svg"` | `"svg"`, `"pdf"` or `"png"`. |

**Returns** an `ErrorAnalysis` with `metrics`, `events`, `tables` (every table by name),
`manifest` and `paths`.

`write_figures(output_dir, figure_format="svg")` draws the figures again from the tables an
analysis wrote, without rescoring — after changing a figure, or for another format.

## What it measures

### Strict events

The analysis runs the same nervaluate scoring `predict_entities` does, once, and reads its
strict accounting off as rows: a strict match is a `TP` row holding both spans, every other
gold span an `FN` row, every other prediction an `FP` row. The rows are checked against
nervaluate's counts, so `result.metrics` is exactly what `predict_entities` reports for the
same predictions and gold. Nothing that follows changes a strict count.

### Diagnostic units

The strict errors of each document are then grouped into units, in this order:

| Unit | What it is |
|---|---|
| `UNCREDITED_EXACT` | A prediction identical to a gold span that the scorer did not credit: nervaluate gives a gold span to the first prediction overlapping it by at least `min_overlap_percentage`, so an exact prediction listed after such an overlap is counted spurious. A scorer artifact, made visible rather than filed as a model error. |
| `BOUNDARY_ERROR`, `FRAGMENTED` | Two or more predictions, each inside one gold span, whose union reaches `overlap_threshold` in character Dice: the model found the entity and split it (`anti` + `TNF` for `anti-TNF`). One unit. |
| `BOUNDARY_ERROR`, `MERGED` | Two or more gold spans each inside one prediction, the mirror case (`dolor torácico y disnea` over `dolor torácico` and `disnea`). One unit. |
| `BOUNDARY_ERROR`, one to one | The rest are paired one to one, maximising total Dice, and a pair at or above `overlap_threshold` is a boundary error: `TOO_LONG` (the prediction contains the gold), `TOO_SHORT` (it lies inside it), `SHIFTED_LEFT` or `SHIFTED_RIGHT`. |
| `MISSED`, `SPURIOUS` | What is still left. |

A one-to-one boundary error also says which edge it got right, as `matched_edge`: `START`
when only the end is wrong (`dolor torácico` predicted as `dolor torácico izquierdo`), `END`
when only the start is (`un dolor torácico`), `NEITHER` when both are; a shifted prediction is
always `NEITHER`. The two answer different questions — trailing modifiers tend to break the
end, determiners and preceding adjectives the start — so they are kept apart from the kind
rather than folded into it.

Character Dice is `2·overlap / (length_gold + length_predicted)`, symmetric and in [0, 1]. The
0.60 default is a diagnostic convention, not a value from the literature, which is why
`overlap_sensitivity` reruns the pairing across a range. A strict match is a `CORRECT` unit.

### Lexical exposure

Mentions are compared in a normalized form: NFKC, casefolded, whitespace collapsed. A mention
is

- `SEEN` when its normalized form occurs among the training mentions of `target_label`,
- `LEXICALLY_SIMILAR` when its nearest training form has normalized Levenshtein similarity,
  `1 − edits / max(len)`, of at least `similarity_threshold`,
- `NOVEL` otherwise.

`SEEN` is exactly a similarity of 1.0. `training_frequency` counts the training mentions with
the same normalized form, and `seen_verbatim` keeps the case-sensitive view for a cased model.

### Subgroups

A subgroup is reported on the side that defines it. Gold mentions are grouped by their own
exposure, similarity bin, training frequency, word count, casing and traits, and each group
reports its support, strict recall (the share `CORRECT`), character recall, entity-weighted
character recall (the mean of its mentions' `entity_char_credit`), and how its other mentions failed.
Predictions are grouped by the same properties of their own surface and report strict,
character and entity-weighted character precision. There is no per-group F1: it would join recall over
one population to precision over another, and a prediction one word too long on a seen
mention would count against seen recall and novel precision at once.

A mention's word count is its UAX #29 words, punctuation included — the words the encoder
labels, so the tagging decisions a strict match needs. Its traits are acronym-like (letters, at
most 12 characters and 3 words, at least 60% uppercase), and whether it contains a digit,
punctuation, parentheses, a hyphen or dash, a slash or a Greek letter. Gold mentions also say
whether they overlap, or nest with, another gold span.

### Intervals

Every rate carries a percentile interval from a document bootstrap: whole documents are
resampled with replacement, keeping each document's gold and predictions together, and the
rate is recomputed from the resampled totals. A thin subgroup shows a wide interval rather
than being cut at an arbitrary support. The overall intervals are in `bootstrap_intervals`;
each subgroup's sit beside its rate as `_low` and `_high`. Per-document scores are in
`documents`, for whether a few documents dominate the result.

### Confidence

When predictions carry a `score` that varies, three more tables say whether to trust it:
`confidence_by_outcome`, `calibration` over equal-count score bins, and `confidence_sweep`,
the strict scores with predictions below each threshold dropped. Each threshold is scored
afresh, since dropping a prediction can change which one nervaluate credits.

## What it writes

```
<output_dir>/
    events.parquet                one row per gold span or prediction: strict outcome, unit, error type
    tables/
        gold_mentions.parquet     one row per gold span: outcome, exposure, features
        predicted_mentions.parquet  one row per prediction, the same
        units.parquet             one row per diagnostic unit: type, kind, matched edge, its gold and predicted ids, Dice, offsets
        gold_subgroups.parquet    recall and outcomes per family and value, with intervals
        predicted_subgroups.parquet  precision and outcomes per family and value, with intervals
        error_taxonomy.parquet    units per type, with shares of all units and of errors
        boundary_errors.parquet   one row per boundary error: kind, matched edge, surfaces, Dice, offsets, flags
        boundary_summary.parquet  boundary errors per kind and matched edge, with median offsets
        overlap_sensitivity.parquet  units per type at each Dice threshold
        oracle_error_budget.parquet  strict F1 with each kind of error fixed
        error_surfaces.parquet    errors per normalized surface and type
        documents.parquet         strict, character and entity-weighted character scores per document
        bootstrap_intervals.parquet  intervals for overall strict, character and entity-weighted character P/R/F1
        confidence_by_outcome.parquet, calibration.parquet, confidence_sweep.parquet   with a varying score
    figures/                      the figures below, as figure_format
    analysis_manifest.json        inputs (path, sha256, rows), counts, parameters, metrics, outputs, duration
```

`gold_subgroups` and `predicted_subgroups` have one row per `family` and `value`, families
being `exposure`, `similarity_bin`, `frequency_bin`, `word_count_bin`, `casing` and each trait
(values `absent`, `present`). The manifest counts duplicate gold and predicted rows, without
removing them.

## Figures

Each figure answers one question. Colours are fixed per outcome across every figure — correct
blue, boundary error orange, missed vermillion, spurious purple, uncredited exact sky — from
the Okabe–Ito palette, checked for colour-blind separation. Text stays text in SVG, so labels
can be edited for a paper.

| Figure | Question | How to read it |
|---|---|---|
| `error_taxonomy` | What kinds of errors does the model make? | Units per type, correct first, errors by count with their share of all errors. |
| `exposure` | Does it find mentions it never saw? | Left, strict recall with its interval per exposure class; right, what became of each class's gold mentions. A drop from seen to novel that the intervals do not overlap is a generalisation gap. |
| `predicted_exposure` | When it predicts a seen or a novel surface, is it right? | Precision per class of the predicted surface, beside those predictions' outcomes. Low novel precision with many spurious predictions points at over-prediction of unfamiliar strings. |
| `mention_properties` | Does recall change with exposure, length or casing? | One panel each for training frequency, similarity, word count and casing; dots with intervals, support under each. |
| `mention_traits` | Which traits make a mention harder? | Per trait, recall without (grey) and with (blue) it. n is the gold mentions with the trait. |
| `boundary_kinds` | Are boundary errors too long, too short, or split — and at which edge? | Boundary errors per kind, one-to-one kinds split by the edge they got wrong: "too long, past the end", "too short, starts late". |
| `boundary_offsets` | Where do predicted edges fall? | Each one-to-one boundary error placed by its start offset (x) against its end offset (y), prediction minus gold in characters, tails folded at ±10; colour is the count, on a log scale. The vertical line holds the errors that got the start right, the horizontal one those that got the end right; the quadrants are too long (top left), too short (bottom right) and shifted. |
| `overlap_sensitivity` | Does the taxonomy depend on the Dice threshold? | Units per type across thresholds, the chosen one marked. Flat lines mean the threshold matters little. |
| `oracle_error_budget` | What would fixing each kind of error gain? | Observed strict F1 to the F1 with that kind fixed, largest gain first. Ceilings, not forecasts; they do not add up. |
| `error_surfaces` | Do a few surfaces cause many errors? | The 15 normalized surfaces most often in an error, stacked by type. |
| `documents` | Is performance uniform across documents? | The share of documents scoring at or below each F1, strict, character and entity-weighted character. A long flat start is a block of documents the model fails on. |
| `confidence_by_outcome` | Do errors carry lower scores? | The score distribution of each outcome. Curves far apart mean the score separates good predictions from bad. |
| `calibration` | Does a score of 0.9 mean 90% right? | The share of strict matches against the mean score, per equal-count bin; the diagonal is perfect calibration. |
| `confidence_sweep` | What does filtering by score cost? | Strict precision, recall and F1, and the share of predictions kept, against the threshold. |

A figure with nothing to show is not drawn: the confidence figures without a varying `score`,
`documents` with fewer than two documents, the boundary figures without a boundary error. The
manifest lists the figures written.

## From YAML

```yaml
task: ner.analyze_predictions
predictions: assets/predictions/predictions.parquet
reference: assets/datasets/disease/test.parquet
training: assets/datasets/disease/train.parquet
target_label: DISEASE
output_dir: assets/analysis/disease
```

```bash
lab run analysis.yaml
```

## References

- Dice, L. R. (1945). Measures of the amount of ecologic association between species.
  *Ecology*, 26(3), 297–302. The overlap coefficient used for pairing.
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and
  reversals. *Soviet Physics Doklady*, 10(8), 707–710. The edit distance behind lexical
  similarity.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
  The percentile intervals.
