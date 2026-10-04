# Code style

How library code is written. The code should read like a careful person wrote it for the
next person to read. These rules describe how `core` and `ner` are already written, and new
code matches them. Code that breaks them is a finding: surface it, don't fix it on sight.

1. **Short functions that do one thing.** A leaf function's name says what it does
   (`build_window`, `split_oversized_sentence`), and its body does only that.
2. **An entry point reads as a list of steps.** A public orchestrator such as `train_model`
   or `create_split` reads top to bottom: validate, resolve the inputs, write the manifest,
   do the work, write the outputs, return a result object. The details live in the
   functions it calls; the sequence stays visible in one place. A function that has to be
   scrolled through to see what it does is a script, not an orchestrator.
3. **A helper exists when it names a concept or is reused**, never just to make a function
   shorter. No one-line wrappers, no indirection layers. A loop that is only used once stays
   inline.
4. **Short modules, one concern each.** A file does one job, the one its module docstring
   names. When it takes on a second job, it becomes a package with one module per job, named
   after what that job is: `ner/training/` is `arguments`, `devices`, `tracking`, `trainer`
   and `assessment`. A module running to several hundred lines is a sign it holds more than
   one concern. Split by concern, not by line count: a short module with one job is fine,
   and a module split only to be shorter breaks rule 3.
5. **Plain functions by default.** A class has to hold configured state that gets reused
   (`Encoder`), or be a subclass a framework requires (`CRFTrainer`, a `TrainerCallback`),
   whose hook overrides need no docstring. Results are frozen dataclasses. No base
   classes, managers or registries unless a decision asks for one.
6. **Names say what a thing is, in full words**: `fold_by_doc_id`, `remaining_budget`, not
   `fbd`, `rb`. DataFrames end in `_df`, paths in `_path` or `_dir`. Filenames and allowed
   choices are UPPER_CASE constants at the top of the module. Files are read and written
   with `read_*`/`write_*`; models are loaded and saved with `load_*`/`save_*`, as in
   `transformers`.
7. **Docstrings are prose.** A module opens with one line naming its concern. A public
   function gets a one-line summary, then paragraphs on behaviour, defaults and why — no
   `Args:`/`Returns:` sections. Private `_` helpers have none.
8. **No explanatory comments.** Comments are only for what the code cannot say (e.g. why a
   dependency with no imports is required). Rationale goes in the register (D20).
9. **Check and fail early, keep the code flat.** Invalid input raises at the top, the trivial
   case returns early, and nesting rarely goes past two levels.
10. **An error says how to fix itself.** It names the offending value, shows a capped sample
    (`[:10]`), and names the argument that changes the behaviour. Anything done to the
    user's data that they did not explicitly ask for warns, with counts.
11. **Keyword arguments when the call is not obvious.** Calls into another module or to a
    public function pass more than a couple of arguments by keyword. A helper called from
    within the same module, with an obvious argument order, may take its arguments by
    position.
12. **Full, modern type hints**: `from __future__ import annotations`, `X | None`,
    `list[dict]`, `Literal` aliases for policies.

As a matter of layout, blocks are separated by a blank line, there is a blank line before
`return`, and lines stay within 100 characters. `ruff format` and `ruff check --fix`, with the
settings in `pyproject.toml`, keep the line length and the import order; the blank lines are
kept by hand. Where the formatter makes a line harder to read, restructure the code — a named
intermediate, a helper — rather than fight the layout.
