"""Selection task table, strings only so importing it stays dependency-light."""

from __future__ import annotations

TASKS: dict[str, tuple[str, str]] = {
    "build_representations": ("lab.selection.api", "build_representations"),
    "compare_methods": ("lab.selection.api", "compare_methods"),
    "consensus_select_documents": ("lab.selection.api", "consensus_select_documents"),
    "select_documents": ("lab.selection.api", "select_documents"),
}

# Optional dependencies are checked by the individual method/representation that
# needs them.  Keeping this empty means the random baseline remains usable from a
# base installation and `lab tasks` does not mark the whole namespace unavailable.
REQUIRES: tuple[str, ...] = ()
