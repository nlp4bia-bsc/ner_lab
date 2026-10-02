# Figure captions

Figures are deterministic SVG vectors designed for scientific publication: Okabe–Ito color-blind-safe semantics, horizontal category labels, no top/right spines, no default grid, direct numerical annotation where useful, and explicit support/uncertainty cues.

- **boundary_absolute_delta.svg** — Absolute start/end boundary deviation among boundary errors. Offsets are grouped into interpretable character-distance bins to expose whether failures are predominantly near-misses.
- **dataset_shift_labels.svg** — Training versus evaluation label composition as 100% stacked bars. Segment labels report meaningful shares, and Jensen-Shannon distance is displayed when available.
- **document_performance_distribution.svg** — Distribution of per-document STRICT and character F1. Box summaries and individual-document points expose heterogeneity that aggregate corpus metrics can hide.
- **entity_length_strict_f1.svg** — STRICT F1 by mention word length, exposing whether span complexity increases with longer mentions.
- **error_surface_pareto.svg** — Most recurrent mention surfaces among diagnostic errors. Cumulative proportions reveal whether a small lexical set accounts for a large fraction of failures.
- **error_taxonomy.svg** — Diagnostic event composition. Counts and percentages distinguish correct detections from missed, spurious, boundary and label-related errors without changing official STRICT scoring.
- **error_type_pareto.svg** — Errors-only Pareto view ordered by frequency. Each bar reports the error count, share of all errors and cumulative share, identifying the highest-yield failure modes.
- **generalization_character_prf.svg** — Character-level precision, recall and F1 across lexical exposure groups, with gold-support counts/proportions and document-bootstrap confidence intervals for F1 when available.
- **generalization_error_profile.svg** — Gold-conditioned outcome composition across seen, few-shot and zero-shot mentions. The 100% stacked bars separate correct detections from missed, boundary, label and combined errors.
- **generalization_strict_prf.svg** — STRICT precision, recall and F1 across lexical exposure groups. Gold-support counts and proportions are shown with each group; F1 markers include document-bootstrap confidence intervals when available, and deltas quantify the generalization gap from seen mentions.
- **oracle_error_budget.svg** — Counterfactual STRICT F1 ceilings for independently correcting each diagnosable error family. Dumbbells connect the observed baseline to each oracle score; scenarios are not treated as additive.
- **span_end_delta.svg** — Signed end-boundary deviation (prediction minus gold). Negative values end too early; positive values end too late.
- **span_start_delta.svg** — Signed start-boundary deviation (prediction minus gold). Negative values start too early; positive values start too late. Tail offsets are collapsed only when necessary for legibility.
- **structural_effects_strict_f1.svg** — Difference in STRICT F1 between mentions with and without structural properties (acronym-like, overlapping or nested). Negative values indicate a performance penalty when the property is present.
- **training_frequency_vs_strict_f1.svg** — STRICT F1 by exact training-mention frequency. Support is reported for every bucket and low-support buckets are visually flagged.

## Deliberately omitted figures

- **generalization_strict_f1_by_label.svg** — Single-label evaluation; a label-by-generalization matrix would be redundant.
- **generalization_character_f1_by_label.svg** — Single-label evaluation; a label-by-generalization matrix would be redundant.
- **similarity_vs_strict_f1.svg** — Disabled by design: lexical-similarity subgroup plots were judged redundant and visually weak for this analysis profile.
- **boundary_relations.svg** — Disabled by design: span-relation panels were judged low-value compared with deviation-range summaries.
- **label_confusion_counts.svg** — Single-label evaluation; a label confusion matrix is not informative.
- **label_confusion_normalized.svg** — Single-label evaluation; a label confusion matrix is not informative.
- **label_confusion_extended.svg** — Single-label evaluation; a label confusion matrix is not informative.
- **confidence_by_error.svg** — Prediction scores span only 0.0012; the confidence signal is effectively constant and a plot would be misleading.
- **confidence_threshold_tradeoff.svg** — Confidence scores have negligible variation, so threshold sweeps do not provide a meaningful operating-point analysis.
