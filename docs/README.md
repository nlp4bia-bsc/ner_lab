# Documentation

Working documents for the migration of `bsc/NER-API` into `ner_lab`. Four files, four jobs.

| File | Answers | Changes |
|---|---|---|
| [DESIGN.md](DESIGN.md) | Why the library is shaped this way. The governing principle, the hard rules, the layering, how we work. | Rarely. |
| [DECISIONS.md](DECISIONS.md) | What was decided and why. Numbered `D1`–`Dn`, superseded entries struck through in place. | Append-only. |
| [ROADMAP.md](ROADMAP.md) | What is done, what is next, what is still open, what we refuse to port by accident. | Every stage. |
| [PROGRESS.md](PROGRESS.md) | What actually landed and the evidence for it. One entry per subsystem. | Every stage. |

Rules of thumb when updating:

- A decision goes in **DECISIONS.md** and nowhere else. Other files link to it by number.
- Superseding a decision means striking the old row and pointing at the new one, never
  deleting it — the reasoning that was wrong is worth keeping.
- **PROGRESS.md** records outcomes and evidence, not narrative. If an entry is growing a
  blow-by-blow account, it belongs in the commit message.
- Anything not yet decided is an open question in **ROADMAP.md**, not an assumption in code.
