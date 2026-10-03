# NER error and generalization analysis

`lab.ner.analysis` extends the repository's existing NER evaluation without replacing it.
The canonical evaluator remains `lab.ner.evaluation.span_metrics`; analysis only adds
subgroup scoring, diagnostic FP/FN association, lexical training-exposure analysis,
bootstrap/stability analyses, reusable Parquet tables, and optional publication-oriented
SVG figures.

## CLI

```bash
lab ner analysis run \
  --predictions predictions.tsv \
  --gold gold.tsv \
  --training assets/train_documents.parquet \
  --output-dir runs/model_01/analysis \
  --run-id model_01 \
  --levenshtein-threshold 0.80 \
  --lexical-similarity-mode hybrid \
  --diagnostic-overlap-threshold 0.60 \
  --bootstrap-samples 2000 \
  --partition-size 50 \
  --seed 13 \
  --verbose 1
```

Predictions and gold can be `.tsv` or `.parquet` span tables with:

```text
filename | label | start_span | end_span | text
```

Predictions may additionally contain `score`. Rows are **not** deduplicated when read;
duplicate annotations/predictions are reported by the audit rather than silently removed.

Training can use the same span-table contract or the canonical repository corpus:

```text
doc_id | text | entities_json | n_entities
```

The latter is converted with `lab.core.spans_from_corpus`, so no analysis-specific corpus
reader is duplicated.

## Diagnostic FN↔FP association

Official STRICT TP/FP/FN remains unchanged. Diagnostic pairing is applied **only after**
strictly correct entities have been removed.

One-to-one candidates must:

1. occur in the same document; and
2. reach the configurable normalized character Sørensen–Dice overlap threshold.

The default is:

```text
diagnostic_overlap_threshold = 0.60
```

with:

```text
Dice(G, P) = 2 * intersection(G, P) / (length(G) + length(P))
```

The coefficient is normalized to `[0, 1]` and symmetric in gold/prediction. The `0.60`
cut-off is a **repository diagnostic heuristic**, not a threshold proposed by the original
Dice paper. It is persisted in the manifest and a sensitivity table is generated for
`0.40, 0.50, 0.60, 0.70, 0.80` by default.

Eligible one-to-one candidates are resolved globally with the existing deterministic
Hungarian assignment. Label agreement may help distinguish error type, but it never decides
whether spans are geometrically eligible.

### Label errors

`LABEL_ERROR` and `BOUNDARY_AND_LABEL_ERROR` are only meaningful when the evaluation has
more than one entity label. In a single-label NER run, the diagnostic layer does not create
an artificial label-error category.

### Fragmentation

Fragmentation is treated as a one-gold-to-many-prediction boundary failure. It is resolved
before one-to-one matching and counts as **one boundary diagnostic unit**, not one error per
fragment.

A conservative fragmentation candidate requires:

- at least two unmatched predictions overlapping the same unmatched gold span;
- no individual fragment already qualifying as a valid one-to-one pair; and
- the union of the fragments reaching the same normalized Dice threshold.

The gold-side diagnostic row stores the complete list of fragment prediction IDs plus union
coverage and union Dice. Standard one-to-one boundary-offset figures exclude fragmentation
rows because a single start/end deviation is not a faithful description of a split entity.

Merging is intentionally not studied in the current analysis profile. The legacy schema
field is retained for compatibility but is not populated or included in intersection
analyses.

### Diagnostic overlap reference

- Dice, L. R. (1945). *Measures of the Amount of Ecologic Association Between Species*.
  **Ecology, 26(3), 297–302**. DOI: `10.2307/1932409`.

The paper motivates the normalized symmetric coefficient. Its original domain is ecological
association; applying the coefficient to character-span overlap is a repository adaptation.

## Lexical exposure classes

The canonical classes are now:

- `SEEN`: the raw mention surface occurs exactly in training.
- `LEXICALLY_SIMILAR`: no raw exact match, but configured lexical similarity is at least
  the threshold (default `0.80`).
- `NOVEL`: no raw exact match and lexical similarity is below the threshold, or training
  contains no neighbour.

The previous `FEW_SHOT` / `ZERO_SHOT` terminology is avoided in new outputs because these
classes describe **lexical exposure**, not conventional few-shot/zero-shot learning.
A `generalization_class_legacy` field maps these values back to `SEEN` / `FEW_SHOT` / `ZERO_SHOT`, and backward-compatible boolean columns such as `few_shot_levenshtein` and `zero_shot_levenshtein` are retained.

### Lexical similarity modes

`--lexical-similarity-mode levenshtein` preserves normalized Levenshtein similarity.

The new default:

```text
--lexical-similarity-mode hybrid
```

uses:

```text
combined_similarity = min(normalized_Levenshtein, normalized_Jaro_Winkler)
```

This is deliberately conservative: both character comparators must support a high-similarity
decision before a non-exact mention is assigned to `LEXICALLY_SIMILAR`. Individual
Levenshtein and Jaro–Winkler values are also persisted so the combined rule remains auditable.

Reference for the Jaro comparator:

- Jaro, M. A. (1989). *Advances in Record-Linkage Methodology as Applied to Matching the
  1985 Census of Tampa, Florida*. **Journal of the American Statistical Association,
  84(406), 414–420**. DOI: `10.1080/01621459.1989.10478785`.

The `min(...)` aggregation rule is a repository design choice, not a rule prescribed by the
Jaro paper.

## Error taxonomy

The existing diagnostic taxonomy is preserved:

- `CORRECT`
- `MISSED`
- `SPURIOUS`
- `LABEL_ERROR`
- `BOUNDARY_ERROR`
- `BOUNDARY_AND_LABEL_ERROR`

No redesign of taxonomy units/denominators is performed in this version.

## Generalization and training-frequency metrics

The existing subgroup PRF semantics are preserved in this version. The audit identified
possible alternative conditioning schemes, but these are intentionally **not** changed here.
The only change is the exposure-class terminology (`SEEN`, `LEXICALLY_SIMILAR`, `NOVEL`) and
the optional hybrid lexical comparator.

## Bootstrap

Document-level bootstrap remains the inferential resampling strategy:

```text
sampling unit = complete document
sampling = with replacement
```

Gold and predictions for the sampled document stay paired, preserving within-document
annotation dependence.

## Non-overlapping document-partition stability

A separate descriptive analysis is now produced with:

```text
--partition-size 50
```

Documents are shuffled deterministically from the run seed and divided **without replacement**
into non-overlapping partitions. The analysis reports:

- per-partition STRICT precision / recall / F1;
- per-partition character metrics;
- per-partition gold-conditioned recall for `SEEN`, `LEXICALLY_SIMILAR`, `NOVEL`;
- full-test estimate;
- partition mean / standard deviation / minimum / maximum.

This must not be interpreted as a bootstrap confidence interval. It is a descriptive
stability analysis answering whether the aggregate result is dominated by particular groups
of documents.

Set `--partition-size 0` to disable it.

## Main reusable tables

In addition to the existing outputs, this version adds:

```text
tables/boundary_overlap_sensitivity.parquet
tables/fragmentation_errors.parquet
tables/document_partition_metrics.parquet
tables/document_partition_generalization.parquet
tables/document_partition_summary.parquet
```

`boundary_errors.parquet` additionally stores normalized overlap fields where available:

```text
span_gold_length
span_pred_length
span_gold_coverage
span_pred_coverage
span_dice
span_iou
error_fragmentation
fragmentation_pred_count
fragmentation_gold_coverage
fragmentation_dice
diagnostic_pred_ids
```

## Figures

The existing publication-quality profile is preserved. Relevant behaviour in this version:

- `generalization_strict_prf.svg` and `generalization_character_prf.svg` keep existing PRF
  semantics but display `Seen`, `Lexically similar`, `Novel`;
- training-frequency figures remain available with their existing metric semantics;
- one-to-one start/end/absolute boundary plots exclude fragmentation rows;
- `boundary_relations.svg` remains disabled;
- `similarity_vs_strict_f1.svg` remains disabled;
- `document_partition_stability.svg` visualizes the new without-replacement stability
  analysis when at least two partitions are available;
- single-label confusion plots and uninformative confidence plots remain automatically
  omitted.

Every plotting decision is stored in `figures/plot_manifest.json`.

## Verification

Run:

```bash
python3 verification/verify_analysis.py
python3 verification/verify_layering.py
python3 verification/run_all.py
```

The analysis verification now includes adversarial checks for:

- distant FN/FP spans;
- one-character trivial overlap below the Dice threshold;
- plausible boundary mismatch;
- multiclass label-only mismatch;
- combined boundary + label mismatch;
- one-gold-to-many-prediction fragmentation counted as one boundary diagnostic unit;
- merging remaining disabled in this analysis profile;
- overlap-threshold sensitivity artifact;
- without-replacement partition artifacts.
