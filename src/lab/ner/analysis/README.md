# NER error and generalization analysis

`lab.ner.analysis` extends the repository's existing NER evaluation without replacing it.
The canonical evaluator remains `lab.ner.evaluation.span_metrics`; analysis only adds
subgroup scoring, diagnostic FP/FN association, training-exposure analysis, error tables,
bootstrap intervals and optional publication-oriented SVG figures.

## CLI

```bash
lab ner analysis run \
  --predictions predictions.tsv \
  --gold gold.tsv \
  --training assets/train_documents.parquet \
  --output-dir runs/model_01/analysis \
  --run-id model_01 \
  --levenshtein-threshold 0.80 \
  --bootstrap-samples 2000 \
  --seed 13 \
  --verbose 1
```

Predictions and gold can be `.tsv` or `.parquet` span tables with:

```text
filename | label | start_span | end_span | text
```

Predictions may additionally contain `score` in `[0, 1]`. Rows are **not** deduplicated
when read; duplicate annotations/predictions are reported by the audit instead of being
silently removed.

Training can use the same span-table contract or the canonical repository corpus:

```text
doc_id | text | entities_json | n_entities
```

The latter is converted with `lab.core.spans_from_corpus`, so no analysis-specific corpus
reader is duplicated.

## Generalization classes

The default operational definition is lexical and is persisted in `manifest.json`:

- `SEEN`: the raw mention surface occurs exactly in training.
- `FEW_SHOT`: no exact raw match, but the nearest normalized training mention has
  normalized Levenshtein similarity `>= 0.80` by default.
- `ZERO_SHOT`: no exact raw match and nearest similarity is below the threshold, or
  training contains no candidate mention.

Here `FEW_SHOT` means *lexically similar to training*. It does not claim that the model was
trained in a conventional k-shot learning regime. The artifact also stores exact frequency,
normalized exposure, nearest same-label/other-label mentions, similarity bins and training
frequency bins.

## Error taxonomy

Official STRICT accounting remains TP/FP/FN. Unmatched FP/FN rows are then associated
within each document only for diagnostics. Association never creates evaluation credit.
The primary diagnostic classes are:

- `CORRECT`
- `MISSED`
- `SPURIOUS`
- `LABEL_ERROR`
- `BOUNDARY_ERROR`
- `BOUNDARY_AND_LABEL_ERROR`

Boundary diagnostics include signed and absolute start/end errors, length difference,
intersection, IoU, containment/overlap direction, one-character offsets, whitespace and
simple punctuation differences. Fragmentation, merging, duplicate prediction, nested and
overlapping entity flags are retained independently.

## Multiclass analysis

The run is considered multiclass when the union of gold and predicted labels contains more
than one label. In that case publication output includes exact-boundary label-confusion
figures. Confusion tables are always persisted in Parquet, including an extended table with
`[MISSED]` and `[SPURIOUS]` flows.

## Metrics and tables

The analysis writes:

```text
analysis_dir/
├── evaluation.parquet
├── manifest.json
├── captions.md                      # verbose=1
├── tables/
│   ├── overall_metrics.parquet
│   ├── metrics_by_generalization.parquet
│   ├── gold_recall_by_generalization.parquet
│   ├── metrics_by_similarity.parquet
│   ├── metrics_by_training_frequency.parquet
│   ├── metrics_by_label.parquet
│   ├── metrics_by_label_generalization.parquet
│   ├── metrics_by_entity_length.parquet
│   ├── metrics_by_acronym.parquet
│   ├── metrics_by_overlap.parquet
│   ├── metrics_by_nested.parquet
│   ├── error_taxonomy.parquet
│   ├── boundary_errors.parquet
│   ├── label_confusion_counts.parquet
│   ├── label_confusion_normalized.parquet
│   ├── label_confusion_extended.parquet
│   ├── document_metrics.parquet
│   ├── dataset_shift_labels.parquet
│   ├── oracle_error_budget.parquet
│   ├── confidence_by_error.parquet
│   ├── confidence_threshold_sweep.parquet
│   ├── error_pareto.parquet
│   ├── error_intersections.parquet
│   ├── bootstrap_intervals.parquet
│   └── *_annotation_audit.parquet
└── figures/                         # verbose=1
    └── *.svg
```

Every subgroup P/R/F1 table calls the existing evaluator on the complete selected gold and
prediction populations. It does not estimate subgroup F1 by filtering only correct events.

Document-level bootstrap intervals retain within-document dependence. Oracle tables are
counterfactual STRICT ceilings and must not be interpreted as expected model improvement.

## Figures

`--verbose 1` additionally generates deterministic SVG vectors. The central required plots
are:

- `generalization_strict_prf.svg`
- `generalization_character_prf.svg`
- `generalization_strict_f1_by_label.svg` (multiclass)
- `generalization_character_f1_by_label.svg` (multiclass)
- similarity/frequency curves
- boundary offset histograms
- error-taxonomy and oracle plots
- label confusion heatmaps (multiclass)
- label-distribution shift
- confidence-by-error plot when scores exist

The plotting layer uses an Okabe–Ito color-blind-safe palette, white background, editable
SVG text, no default grid and no top/right Cartesian spines.

## Inspect and regenerate

```bash
lab ner analysis inspect runs/model_01/analysis
lab ner analysis report --analysis-dir runs/model_01/analysis --verbose 1
```

`report` regenerates figures from persisted Parquet tables without rerunning inference or
rescore/re-exposure work.

## Publication-quality visualization profile (v2)

`--verbose 1` now generates an analytical SVG report rather than a generic set of plots. The plotting layer is designed around model-diagnostic questions and uses consistent semantic colors, horizontal category labels, direct value/support annotations, no top/right spines, and no default grid. Category tick labels are never rotated; long labels are wrapped or moved to a horizontal layout.

Key visualizations include:

- precision/recall/F1 by `SEEN`, `FEW_SHOT`, and `ZERO_SHOT`, with gold-support counts/proportions and document-bootstrap F1 confidence intervals when available;
- a gold-conditioned 100% error profile showing *why* each lexical exposure group fails;
- similarity, training-frequency, and entity-length lollipop plots that explicitly flag low-support estimates;
- multiclass label-by-generalization heatmaps and confusion matrices with horizontal text;
- an error taxonomy plus errors-only Pareto view and recurrent-error-surface Pareto;
- signed and absolute boundary diagnostics with interpretable character-offset bins and span-relation summaries;
- an oracle dumbbell plot that compares each independent counterfactual with the observed STRICT F1 without implying that oracle gains are additive;
- 100% stacked training/evaluation label-composition shift with Jensen–Shannon distance;
- confidence distribution/threshold plots only when prediction scores vary enough to make such figures informative;
- per-document F1 distributions and structural-effect plots for acronym-like, overlapping, and nested mentions.

Every plotting decision is written to `figures/plot_manifest.json`. Figures that would be misleading or redundant are deliberately omitted and the reason is recorded in both `plot_manifest.json` and `captions.md`; examples include single-label confusion matrices or confidence plots for almost constant scores.

The plotting code remains downstream of persisted Parquet analysis tables. Therefore `lab ner analysis report --analysis-dir ... --verbose 1` can regenerate the complete SVG report without rescoring the model.
