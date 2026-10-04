"""Ontology graphs: building them from concepts and edges, reading them from TSV or a pickle."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx

from lab.nel.io import read_concepts_tsv, read_hierarchy_tsv
from lab.nel.schemas import Concept, HierarchyEdge

DIRECTED_GRAPH_KEYS = ("graph", "G", "digraph")
UNDIRECTED_GRAPH_KEYS = ("UG", "undirected_graph")


def build_ontology_graph(
    concepts: Iterable[Concept] | None = None,
    hierarchy_edges: Iterable[HierarchyEdge] | None = None,
) -> tuple[nx.DiGraph, nx.Graph]:
    """
    A directed `parent -> child` graph and its undirected view.

    Edges follow SNOMED's `is_a` direction and carry their `relation`, `is_a` when
    none is given. Concept nodes carry `term`, `label`, `semantic_tag` and `aliases`.
    """
    graph = nx.DiGraph()

    for concept in concepts or []:
        graph.add_node(
            str(concept.code),
            term=concept.term,
            label=concept.label,
            semantic_tag=concept.semantic_tag,
            aliases=concept.aliases,
        )

    for edge in hierarchy_edges or []:
        parent = str(edge.parent_code)
        child = str(edge.child_code)
        graph.add_node(parent)
        graph.add_node(child)
        graph.add_edge(parent, child, relation=edge.relation or "is_a")

    return graph, graph.to_undirected()


def read_ontology_graph(
    concepts_path: str | None = None, hierarchy_path: str | None = None
) -> tuple[nx.DiGraph, nx.Graph]:
    """`build_ontology_graph` over a concepts TSV and a hierarchy TSV, either optional."""
    concepts = read_concepts_tsv(concepts_path) if concepts_path else []
    hierarchy = read_hierarchy_tsv(hierarchy_path) if hierarchy_path else []

    return build_ontology_graph(concepts, hierarchy)


def read_ontology_graph_pickle(path: str | Path) -> tuple[nx.DiGraph, nx.Graph]:
    """
    A pickled NetworkX ontology graph, as a directed graph and its undirected view.

    The pickle holds a graph, a `(graph, undirected_graph)` tuple, or a dict with
    the graph under `graph`, `G` or `digraph` and optionally the undirected view
    under `UG` or `undirected_graph`. A missing undirected view is derived.
    """
    with Path(path).open("rb") as handle:
        stored = pickle.load(handle)

    graph, undirected_graph = _stored_graphs(stored)

    if not isinstance(graph, nx.Graph):
        raise TypeError(
            "Pickle must contain a NetworkX graph, a (graph, UG) tuple or a graph dictionary"
        )

    directed_graph = graph if isinstance(graph, nx.DiGraph) else nx.DiGraph(graph)

    if undirected_graph is None:
        undirected_graph = directed_graph.to_undirected()

    return directed_graph, undirected_graph


def _stored_graphs(stored: Any) -> tuple[Any, Any]:
    if isinstance(stored, tuple) and stored:
        undirected = stored[1] if len(stored) > 1 and isinstance(stored[1], nx.Graph) else None

        return stored[0], undirected

    if isinstance(stored, dict):
        return _first_key(stored, DIRECTED_GRAPH_KEYS), _first_key(stored, UNDIRECTED_GRAPH_KEYS)

    return stored, None


def _first_key(stored: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((stored[key] for key in keys if key in stored), None)
