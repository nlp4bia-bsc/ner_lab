"""NEL's task table, strings only so importing it never loads its dependencies."""

from __future__ import annotations

TASKS: dict[str, tuple[str, str]] = {
    "link_entities": ("lab.nel.linking", "link_entities"),
}

REQUIRES = ("sklearn", "networkx")
