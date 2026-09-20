# Link entities to an ontology

```python
from lab.nel import link_entities

result = link_entities(
    spans="assets/predictions/disease_crf/predictions.tsv",
    gazetteer="assets/ontology/snomed_es.tsv",
    output_dir="assets/linking/disease_crf",
    method="matrix",
)

result.spans        # the input span table plus code | code_term | code_score | candidates_json
result.metrics      # recall@k, MRR and coverage, when the input carried gold codes
```

| Parameter | Default | Meaning |
|---|---|---|
| `spans` | *required* | A span table: frame, TSV or parquet with `filename, label, start_span, end_span, text`. A `code` column is taken as gold. |
| `gazetteer` | *required* | A table with `term` and `code`, optionally `label` and `semantic_tag`. Frame, TSV or parquet. |
| `output_dir` | *required* | Where the linked table, any scores and the manifest are written. |
| `method` | `"matrix"` | How candidates are generated. See below. |
| `base_model` | `None` | Encoder for `transformer_faiss` and `dense`, by hub name or local path. Invalid with the other methods. |
| `method_kwargs` | `None` | Keyword arguments for the method's constructor, e.g. `{"threshold": 0.5}` or `{"f_type": "IVFFlatIP"}`. |
| `reranker` | `None` | A cross-encoder model, by hub name or local path, applied to each mention's `top_k` candidates. |
| `reranker_kwargs` | `None` | Keyword arguments for the reranker, e.g. `{"batch_size": 64, "device": "cuda"}`. |
| `top_k` | `25` | Candidates kept per mention. The first one is the linked code. |
| `k_values` | `(1, 5, 25)` | The `k` in `recall@k`. Each must lie between 1 and `top_k`. |
| `hierarchy` | `None` | A pickled NetworkX graph of the ontology, parent to child. Adds exact / narrow / broad / unrelated proportions to the metrics. |

**Returns** a `LinkingResult` with `spans`, `metrics`, `manifest` and `paths`.

## Methods

| `method` | Kind | Needs |
|---|---|---|
| `string_match` | normalized exact match | — |
| `levenshtein`, `jaro_winkler`, `token_set` | edit-distance and token-set similarity | `rapidfuzz` for speed; falls back to `difflib` |
| `tfidf_char` | TF-IDF over character n-grams | scikit-learn |
| `bm25` | BM25 over tokens | — |
| `matrix` | sparse TF-IDF, character plus word n-grams, top-k by `argpartition` | scikit-learn |
| `faiss` | hashed character n-grams in a FAISS flat index | faiss |
| `transformer_faiss` | a transformer's pooled embeddings in a FAISS index | `base_model`, torch, faiss |
| `dense` | a sentence-transformers bi-encoder and a torch matrix product | `base_model`, sentence-transformers |

Every lexical method restricts candidates to gazetteer entries whose `label` equals the
mention's when both are present (`label_aware`, on by default; pass it in `method_kwargs` to
turn it off). Duplicate codes are collapsed to their best-scoring term. The pieces are public
under `lab.nel.matching` and `lab.nel.retrieval`; `lab.nel.EntityLinkingPipeline` is what
this task drives, and it takes several generators at once, fused by reciprocal rank fusion
(`lab.nel.reciprocal_rank_fusion`).

## The span table in and out

The input keeps every column it came with. When it has a `code` column that column is gold:
it is kept as `gold_code`, and rows with a non-empty value are scored. Four columns are
appended, so the result is still a span table and a downstream tool that does not care about
codes ignores them:

| Column | Meaning |
|---|---|
| `code` | The linked concept: the top candidate's code, or empty when nothing was retrieved. |
| `code_term` | The gazetteer term that candidate was matched through. |
| `code_score` | The generator's or the reranker's score for it. |
| `candidates_json` | The `top_k` candidates as a JSON list of `{code, term, score, method, rank, metadata}`. |

Predictions from [`predict_entities`](predict-entities.md) link as they are: the `score` column
they carry is NER's confidence and is preserved untouched beside `code_score`. A table that was
linked before links again: `gold_code` stays gold and the four columns are written afresh.

## What it writes

```
<output_dir>/
    predictions.tsv            the linked span table
    linking_metrics.json       method, recall@k, MRR, coverage, mean candidates — when there was gold
    linking_manifest.json      inputs and their sha256, method, model, top_k, what was linked
```

## From YAML

```yaml
task: nel.link_entities
spans: assets/predictions/disease_crf/predictions.tsv
gazetteer: assets/ontology/snomed_es.tsv
output_dir: assets/linking/disease_crf
method: transformer_faiss
base_model: ICB-UMA/HERBERT-P
method_kwargs:
  f_type: FlatIP
  batch_size: 128
reranker: assets/models/cross_encoder
top_k: 25
k_values: [1, 5, 25]
```
