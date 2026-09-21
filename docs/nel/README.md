# `lab.nel` — entity linking

**Maintainer**: Fernando Gallego · fgalleg1@bsc.es

Links the mentions in a span table to an ontology: candidates by lexical matching, sparse or
dense retrieval, fused by reciprocal rank fusion, optionally reranked by a cross-encoder, and
scored against gold codes when the table carries them. Installed by `lab[nel]`.

| Page | Covers |
|---|---|
| [link-entities.md](link-entities.md) | `link_entities`: linking a span table to an ontology, the candidate methods, reranking, scoring |

| CLI task | Function |
|---|---|
| `nel.link_entities` | `lab.nel.link_entities` |

The input is the span table `lab.core` defines — the one
[`predict_entities`](../ner/predict-entities.md) writes, or gold `.ann` files read into it,
or one from any other system — and the output is the same table with `code` columns
appended.

## Importing

```python
from lab.nel import (
    link_entities,            # the task
    EntityLinkingPipeline,    # what it drives: several generators at once, fused
    build_matcher,            # a candidate generator by name
    reciprocal_rank_fusion,
    LinkingResult,
    MentionAnnotation, GazetteerEntry, MatchCandidate, LinkedEntity, Concept, HierarchyEdge,
)
```

The matchers live under `lab.nel.matching`, the retrievers under `lab.nel.retrieval`, the
metrics and hierarchy scoring under `lab.nel.evaluation`. As with `lab.ner`, names on
`lab.nel` resolve on first use.
