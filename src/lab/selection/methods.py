"""Document-selection algorithms.

Every selection algorithm lives in this file on purpose.  To add a new method:

1. subclass :class:`SelectionMethod`;
2. implement ``select``;
3. add one entry to ``METHODS`` and ``METHOD_INFO`` at the bottom.

Methods operate on arrays and document IDs only.  They know nothing about Parquet,
NER, NEL, or the CLI.  This keeps selection reusable across downstream tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MethodInfo:
    """Human- and CLI-readable metadata for one selection algorithm."""

    name: str
    family: str
    description: str
    methodologies: tuple[str, ...] = ()
    requires_representation: bool = False
    requires_scores: bool = False
    requires_probabilities: bool = False
    supports_previous_selection: bool = True
    default_representation: str | None = None
    implementation_note: str | None = None
    reference: str | None = None
    parameters: Mapping[str, str] = field(default_factory=dict)


@dataclass
class MethodResult:
    """Result expressed relative to the candidate pool passed to ``select``."""

    selected_indices: list[int]
    scores: np.ndarray | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SelectionMethod(ABC):
    """Base class shared by every task-independent selection method."""

    @abstractmethod
    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
        random_state: int = 13,
        **params: Any,
    ) -> MethodResult:
        """Select ``n`` indices relative to ``candidate_ids``."""


def _validate_budget(n: int, n_candidates: int) -> None:
    if n <= 0:
        raise ValueError("n must be greater than zero.")
    if n_candidates <= 0:
        raise ValueError("No candidate documents remain after exclusions.")
    if n > n_candidates:
        raise ValueError(
            f"Requested {n} documents but only {n_candidates} candidate documents remain."
        )


def _require_representations(value: np.ndarray | None, n_candidates: int) -> np.ndarray:
    if value is None:
        raise ValueError("This method requires document representations.")
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != n_candidates:
        raise ValueError(
            "representations must be a 2D matrix with one row per candidate; "
            f"got {matrix.shape} for {n_candidates} candidates."
        )
    return _l2_normalize(matrix)

_FRAMEWORK_SELECT_KWARGS = frozenset(
    {
        "representations",
        "selected_representations",
        "scores",
        "probabilities",
        "random_state",
    }
)


def _reject_unknown_params(
    method_name: str,
    params: Mapping[str, Any],
) -> None:
    """Reject unknown method-specific kwargs while allowing shared API plumbing.

    The high-level selection API calls every selector through the common
    SelectionMethod interface and may therefore pass generic inputs such as
    representations, scores or probabilities even when a concrete selector
    does not use them.

    Those common interface arguments are not method-specific parameters and
    must not be rejected here. Any remaining keyword is treated as an
    unsupported method parameter and fails loudly.
    """
    if not params:
        return

    unknown = {
        str(key): value
        for key, value in params.items()
        if str(key) not in _FRAMEWORK_SELECT_KWARGS
    }

    if not unknown:
        return

    allowed = sorted(
        method_info(method_name).parameters
    )
    allowed_text = (
        ", ".join(allowed)
        if allowed
        else "none"
    )

    unknown_text = ", ".join(
        sorted(unknown)
    )

    raise ValueError(
        f"Unknown parameter(s) for selection method "
        f"{method_name!r}: {unknown_text}. "
        f"Allowed method-specific parameters: "
        f"{allowed_text}. "
        "Run-level options such as random_state must "
        "be passed through the selection API/CLI "
        "rather than --method-param."
    )


def validate_method_params(
    method_name: str,
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate method-specific parameters against the registered public contract."""
    values = dict(params or {})
    if not values:
        return {}

    if "random_state" in values:
        raise ValueError(
            "random_state is a run-level option; use --seed/--random-state "
            "instead of --method-param random_state=...."
        )

    allowed = set(method_info(method_name).parameters)
    unknown = sorted(set(values) - allowed)
    if unknown:
        allowed_text = ", ".join(sorted(allowed)) if allowed else "none"
        raise ValueError(
            f"Unknown method parameter(s) for {method_name!r}: {unknown}. "
            f"Allowed: {allowed_text}."
        )
    return values

def _require_scores(value: np.ndarray | None, n_candidates: int) -> np.ndarray:
    if value is None:
        raise ValueError("This method requires one external score per candidate document.")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) != n_candidates:
        raise ValueError(f"Expected {n_candidates} scores, received {len(array)}.")
    if not np.isfinite(array).all():
        raise ValueError("External scores contain NaN or infinite values.")
    return array


def _require_probabilities(value: np.ndarray | None, n_candidates: int) -> np.ndarray:
    if value is None:
        raise ValueError("This method requires a probability distribution per candidate.")
    probs = np.asarray(value, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != n_candidates or probs.shape[1] < 2:
        raise ValueError(
            "probabilities must be a 2D matrix with one row per candidate and at least "
            f"two columns; got {probs.shape}."
        )
    if np.any(probs < 0):
        raise ValueError("Probabilities cannot be negative.")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have positive mass.")
    return probs / row_sums


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _cosine_similarity(matrix: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
    left = _l2_normalize(matrix)
    right = left if other is None else _l2_normalize(other)
    return np.clip(left @ right.T, -1.0, 1.0)


def _cosine_distance(matrix: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - _cosine_similarity(matrix), 0.0, 2.0)


def _pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def _entropy(probabilities: np.ndarray, normalized: bool = False) -> np.ndarray:
    probs = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    if normalized and probabilities.shape[1] > 1:
        entropy /= math.log(probabilities.shape[1])
    return entropy


class RandomSelection(SelectionMethod):
    """Seeded random baseline."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        random_state: int = 13,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        _reject_unknown_params("random", params)
        rng = np.random.default_rng(random_state)
        priority = rng.random(len(candidate_ids))
        chosen = np.argsort(-priority, kind="stable")[:n].astype(int).tolist()
        return MethodResult(
            chosen,
            scores=priority,
            diagnostics={"seed": random_state, "criterion": "seeded random priority"},
        )


class KMedoidsSelection(SelectionMethod):
    """Conditional PAM-style k-medoids with previous selections fixed as medoids."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        max_swap_passes: int = 10,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        _reject_unknown_params("kmedoids", params)

        x = _require_representations(representations, len(candidate_ids))
        distance = _cosine_distance(x)

        fixed = None
        fixed_distance = None
        if selected_representations is not None and len(selected_representations):
            fixed = _l2_normalize(
                np.asarray(selected_representations, dtype=np.float32)
            )
            fixed_distance = np.clip(
                1.0 - x @ fixed.T,
                0.0,
                2.0,
            )
            nearest = fixed_distance.min(axis=1)
        else:
            nearest = np.full(
                len(x),
                np.inf,
                dtype=np.float64,
            )

        chosen: list[int] = []
        build_gain = np.full(
            len(x),
            np.nan,
            dtype=np.float64,
        )

        for _rank in range(n):
            current_cost = (
                float(nearest.sum())
                if np.isfinite(nearest).all()
                else math.inf
            )

            best_index = -1
            best_cost = math.inf
            best_nearest = None

            for candidate in range(len(x)):
                if candidate in chosen:
                    continue

                proposal = distance[:, candidate]
                proposal_nearest = (
                    proposal
                    if not np.isfinite(nearest).all()
                    else np.minimum(nearest, proposal)
                )
                cost = float(proposal_nearest.sum())

                if cost < best_cost:
                    best_index = candidate
                    best_cost = cost
                    best_nearest = proposal_nearest

            if best_index < 0 or best_nearest is None:
                break

            chosen.append(best_index)
            build_gain[best_index] = (
                math.nan
                if not math.isfinite(current_cost)
                else current_cost - best_cost
            )
            nearest = best_nearest

        # PAM-style swap refinement. Previously selected documents remain fixed.
        swap_passes = 0
        for swap_passes in range(
            max(0, int(max_swap_passes))
        ):
            current_cost = float(nearest.sum())
            best: tuple[
                int,
                int,
                float,
                np.ndarray,
            ] | None = None

            remaining = [
                index
                for index in range(len(x))
                if index not in chosen
            ]

            for position, _old in enumerate(chosen):
                for new in remaining:
                    proposal = chosen.copy()
                    proposal[position] = new

                    candidate_nearest = distance[
                        :,
                        proposal,
                    ].min(axis=1)

                    if fixed_distance is not None:
                        candidate_nearest = np.minimum(
                            candidate_nearest,
                            fixed_distance.min(axis=1),
                        )

                    cost = float(candidate_nearest.sum())

                    if (
                        cost + 1e-12 < current_cost
                        and (
                            best is None
                            or cost < best[2]
                        )
                    ):
                        best = (
                            position,
                            new,
                            cost,
                            candidate_nearest,
                        )

            if best is None:
                break

            position, new, _cost, nearest = best
            chosen[position] = new

        # IMPORTANT FOR PARTIAL RANK AGGREGATION:
        #
        # After PAM swaps, `chosen` is a FINAL SET of medoids. The list position
        # is no longer a meaningful acquisition rank because a new medoid may
        # have replaced an older medoid in an arbitrary slot.
        #
        # We therefore rank only the FINAL selected medoids by their
        # leave-one-out contribution to the final k-medoids objective:
        #
        #   importance(m) =
        #       objective(without m) - objective(with all final medoids)
        #
        # A larger value means removing that medoid would worsen representation
        # of the pool more strongly, so it receives a better within-selected
        # rank. This DOES NOT change the selected set; it only defines the order
        # used by truncated-Borda consensus.
        final_cost = float(nearest.sum())
        selected_importance = np.full(
            len(x),
            np.nan,
            dtype=np.float64,
        )

        for medoid in chosen:
            other_medoids = [
                index
                for index in chosen
                if index != medoid
            ]

            if other_medoids:
                without = distance[
                    :,
                    other_medoids,
                ].min(axis=1)

                if fixed_distance is not None:
                    without = np.minimum(
                        without,
                        fixed_distance.min(axis=1),
                    )

                selected_importance[medoid] = (
                    float(without.sum())
                    - final_cost
                )

            elif fixed_distance is not None:
                without = fixed_distance.min(axis=1)
                selected_importance[medoid] = (
                    float(without.sum())
                    - final_cost
                )

            else:
                # n == 1 and there are no previous fixed medoids.
                # The selected set contains one element, so its within-selected
                # rank is trivial. Keep a finite diagnostic value.
                selected_importance[medoid] = 0.0

        chosen = sorted(
            chosen,
            key=lambda index: (
                -float(selected_importance[index]),
                int(index),
            ),
        )

        return MethodResult(
            chosen,
            scores=-nearest,
            extras={
                "distance_to_nearest_final_medoid": nearest,
                "build_marginal_cost_reduction": build_gain,
                "selected_leave_one_out_importance": selected_importance,
            },
            diagnostics={
                "objective": float(nearest.sum()),
                "mean_distance_to_nearest_medoid": float(nearest.mean()),
                "swap_passes_attempted": int(
                    swap_passes + 1
                    if max_swap_passes
                    else 0
                ),
                "previous_selections_fixed": bool(
                    fixed is not None
                ),
                "selected_rank_semantics": (
                    "descending leave-one-out contribution "
                    "to the final k-medoids objective"
                ),
            },
        )


class FacilityLocationSelection(SelectionMethod):
    """Greedy conditional facility-location maximisation."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        x = _require_representations(representations, len(candidate_ids))
        similarity = (_cosine_similarity(x) + 1.0) / 2.0

        if selected_representations is not None and len(selected_representations):
            fixed = _l2_normalize(np.asarray(selected_representations, dtype=np.float32))
            current = ((_cosine_similarity(x, fixed) + 1.0) / 2.0).max(axis=1)
        else:
            current = np.zeros(len(x), dtype=np.float64)

        chosen: list[int] = []
        marginal = np.full(len(x), np.nan, dtype=np.float64)
        for _rank in range(n):
            best_index = -1
            best_gain = -math.inf
            best_coverage = None
            for candidate in range(len(x)):
                if candidate in chosen:
                    continue
                coverage = np.maximum(current, similarity[:, candidate])
                gain = float(coverage.sum() - current.sum())
                if gain > best_gain:
                    best_index, best_gain, best_coverage = candidate, gain, coverage
            if best_index < 0 or best_coverage is None:
                break
            chosen.append(best_index)
            marginal[best_index] = best_gain
            current = best_coverage

        return MethodResult(
            chosen,
            scores=current,
            extras={
                "similarity_to_nearest_representative": current,
                "marginal_coverage_gain": marginal,
            },
            diagnostics={
                "facility_objective": float(current.sum()),
                "mean_pool_coverage": float(current.mean()),
            },
        )


class KCenterSelection(SelectionMethod):
    """Greedy farthest-first k-center/core-set selection."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        x = _require_representations(representations, len(candidate_ids))
        distance = _cosine_distance(x)

        if selected_representations is not None and len(selected_representations):
            fixed = _l2_normalize(np.asarray(selected_representations, dtype=np.float32))
            nearest = np.clip(1.0 - x @ fixed.T, 0.0, 2.0).min(axis=1)
            seed_rule = "farthest from previous selections"
        else:
            first = int(np.argmin(distance.sum(axis=1)))
            nearest = distance[:, first].copy()
            chosen = [first]
            selected_distance = np.full(len(x), np.nan, dtype=np.float64)
            selected_distance[first] = float(distance[:, first].mean())
            seed_rule = "global medoid then farthest-first"

        if selected_representations is not None and len(selected_representations):
            chosen = []
            selected_distance = np.full(len(x), np.nan, dtype=np.float64)

        while len(chosen) < n:
            remaining = [index for index in range(len(x)) if index not in chosen]
            index = max(remaining, key=lambda item: float(nearest[item]))
            selected_distance[index] = float(nearest[index])
            chosen.append(index)
            nearest = np.minimum(nearest, distance[:, index])

        return MethodResult(
            chosen[:n],
            scores=nearest,
            extras={
                "distance_to_nearest_final_center": nearest,
                "distance_before_selection": selected_distance,
            },
            diagnostics={
                "final_covering_radius": float(nearest.max()),
                "mean_distance_to_nearest_center": float(nearest.mean()),
                "seed_rule": seed_rule,
            },
        )


def _typicality_scores(embeddings: np.ndarray, labels: np.ndarray, knn: int) -> np.ndarray:
    distance = _cosine_distance(embeddings)
    typicality = np.zeros(len(embeddings), dtype=np.float64)
    for cluster in np.unique(labels):
        ids = np.where(labels == cluster)[0]
        if len(ids) == 1:
            typicality[ids[0]] = 1.0
            continue
        sub = distance[np.ix_(ids, ids)]
        for local_position, global_index in enumerate(ids):
            values = np.delete(sub[local_position], local_position)
            k = min(max(1, int(knn)), len(values))
            mean_knn = float(np.partition(values, k - 1)[:k].mean())
            typicality[global_index] = 1.0 / (mean_knn + 1e-8)
    return typicality


class TypiClustSelection(SelectionMethod):
    """Document-level TypiClust adaptation with previous-selection occupancy."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        n_clusters: int | None = None,
        typicality_knn: int = 20,
        random_state: int = 13,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        candidates = _require_representations(representations, len(candidate_ids))
        fixed = (
            _l2_normalize(np.asarray(selected_representations, dtype=np.float32))
            if selected_representations is not None and len(selected_representations)
            else np.empty((0, candidates.shape[1]), dtype=np.float32)
        )
        all_embeddings = np.vstack([fixed, candidates])
        n_fixed = len(fixed)
        n_clusters = n_clusters or min(len(all_embeddings), max(1, n_fixed + n))
        n_clusters = max(1, min(int(n_clusters), len(all_embeddings)))

        try:
            from sklearn.cluster import KMeans
        except ImportError as error:
            raise ImportError("TypiClust requires scikit-learn: pip install 'lab[selection]'.") from error

        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(all_embeddings)
        typicality = _typicality_scores(all_embeddings, labels, typicality_knn)
        fixed_count: dict[int, int] = {}
        for cluster in labels[:n_fixed]:
            fixed_count[int(cluster)] = fixed_count.get(int(cluster), 0) + 1

        candidate_labels = labels[n_fixed:]
        candidate_typicality = typicality[n_fixed:]
        chosen: list[int] = []
        for _rank in range(n):
            remaining = [index for index in range(len(candidates)) if index not in chosen]
            by_cluster: dict[int, list[int]] = {}
            for index in remaining:
                by_cluster.setdefault(int(candidate_labels[index]), []).append(index)
            cluster = sorted(
                by_cluster,
                key=lambda item: (
                    fixed_count.get(item, 0),
                    -len(by_cluster[item]),
                    -max(candidate_typicality[index] for index in by_cluster[item]),
                    item,
                ),
            )[0]
            index = max(by_cluster[cluster], key=lambda item: float(candidate_typicality[item]))
            chosen.append(index)
            fixed_count[cluster] = fixed_count.get(cluster, 0) + 1

        return MethodResult(
            chosen,
            scores=candidate_typicality,
            extras={"cluster": candidate_labels.astype(np.int32), "typicality": candidate_typicality},
            diagnostics={
                "n_clusters": n_clusters,
                "typicality_knn": int(typicality_knn),
                "previous_cluster_occupancy_used": bool(n_fixed),
            },
        )


class ALPSSelection(SelectionMethod):
    """ALPS clustering in MLM-surprisal space."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        n_clusters: int | None = None,
        random_state: int = 13,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        candidates = _require_representations(representations, len(candidate_ids))
        fixed = (
            _l2_normalize(np.asarray(selected_representations, dtype=np.float32))
            if selected_representations is not None and len(selected_representations)
            else np.empty((0, candidates.shape[1]), dtype=np.float32)
        )
        all_embeddings = np.vstack([fixed, candidates])
        n_fixed = len(fixed)
        n_clusters = n_clusters or min(len(all_embeddings), max(1, n_fixed + n))

        try:
            from sklearn.cluster import KMeans
        except ImportError as error:
            raise ImportError("ALPS clustering requires scikit-learn: pip install 'lab[selection]'.") from error

        model = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=20)
        labels = model.fit_predict(all_embeddings)
        fixed_count: dict[int, int] = {}
        for cluster in labels[:n_fixed]:
            fixed_count[int(cluster)] = fixed_count.get(int(cluster), 0) + 1

        candidate_labels = labels[n_fixed:]
        centroid_distance = np.linalg.norm(
            candidates - model.cluster_centers_[candidate_labels], axis=1
        )
        chosen: list[int] = []
        for _rank in range(n):
            remaining = [index for index in range(len(candidates)) if index not in chosen]
            by_cluster: dict[int, list[int]] = {}
            for index in remaining:
                by_cluster.setdefault(int(candidate_labels[index]), []).append(index)
            cluster = sorted(
                by_cluster,
                key=lambda item: (fixed_count.get(item, 0), -len(by_cluster[item]), item),
            )[0]
            index = min(by_cluster[cluster], key=lambda item: float(centroid_distance[item]))
            chosen.append(index)
            fixed_count[cluster] = fixed_count.get(cluster, 0) + 1

        return MethodResult(
            chosen,
            scores=-centroid_distance,
            extras={
                "cluster": candidate_labels.astype(np.int32),
                "distance_to_surprisal_centroid": centroid_distance,
            },
            diagnostics={"n_clusters": int(n_clusters), "space": "MLM surprisal"},
        )


class UncertaintySelection(SelectionMethod):
    """Select the highest externally supplied document uncertainty scores."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        scores: np.ndarray | None = None,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        uncertainty = _require_scores(scores, len(candidate_ids))
        chosen = np.argsort(-uncertainty, kind="stable")[:n].astype(int).tolist()
        return MethodResult(chosen, scores=uncertainty, diagnostics={"criterion": "descending external score"})


class UncertaintyFacilitySelection(SelectionMethod):
    """Restrict to uncertain documents, then maximise conditional facility location."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        candidate_multiplier: float = 5.0,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        x = _require_representations(representations, len(candidate_ids))
        uncertainty = _require_scores(scores, len(candidate_ids))
        pool_size = min(len(candidate_ids), max(n, int(math.ceil(float(candidate_multiplier) * n))))
        uncertain_indices = np.argsort(-uncertainty, kind="stable")[:pool_size]

        sub_result = FacilityLocationSelection().select(
            [candidate_ids[index] for index in uncertain_indices],
            n,
            representations=x[uncertain_indices],
            selected_representations=selected_representations,
        )
        chosen = [int(uncertain_indices[index]) for index in sub_result.selected_indices]
        in_uncertain_pool = np.zeros(len(candidate_ids), dtype=bool)
        in_uncertain_pool[uncertain_indices] = True
        facility_score = np.full(len(candidate_ids), np.nan, dtype=np.float64)
        if sub_result.scores is not None:
            facility_score[uncertain_indices] = sub_result.scores

        return MethodResult(
            chosen,
            scores=uncertainty,
            extras={
                "in_uncertainty_candidate_pool": in_uncertain_pool,
                "facility_coverage": facility_score,
            },
            diagnostics={
                "candidate_multiplier": float(candidate_multiplier),
                "uncertainty_pool_size": int(pool_size),
                "facility": sub_result.diagnostics,
            },
        )


class PatronSelection(SelectionMethod):
    """Document-level PATRON adaptation using supplied probability distributions."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        selected_representations: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
        k_neighbors: int = 50,
        rho: float = 0.01,
        beta: float = 0.5,
        mu: float = 0.5,
        gamma: float = 0.5,
        refine_rounds: int = 1,
        random_state: int = 13,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        _reject_unknown_params("patron", params)

        x = _require_representations(
            representations,
            len(candidate_ids),
        )
        probs = _require_probabilities(
            probabilities,
            len(candidate_ids),
        )

        uncertainty = _entropy(probs)
        distance = _pairwise_euclidean(x)

        k = min(
            max(1, int(k_neighbors)),
            max(1, len(x) - 1),
        )

        neighbours = np.argsort(
            distance,
            axis=1,
        )[:, : min(k + 1, len(x))]

        neighbour_distance = np.take_along_axis(
            distance,
            neighbours,
            axis=1,
        )

        propagated = np.mean(
            uncertainty[neighbours]
            * np.exp(
                -neighbour_distance
                * float(rho)
            ),
            axis=1,
        )

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances
        except ImportError as error:
            raise ImportError(
                "PATRON requires scikit-learn: "
                "pip install 'lab[selection]'."
            ) from error

        clusterer = KMeans(
            n_clusters=n,
            random_state=random_state,
            n_init=20,
            max_iter=300,
        )
        labels = clusterer.fit_predict(x)

        chosen: list[int] = []

        for cluster in range(n):
            ids = np.where(labels == cluster)[0]
            if not len(ids):
                continue

            centroid_distance = np.linalg.norm(
                x[ids]
                - clusterer.cluster_centers_[cluster],
                axis=1,
            )

            objective = (
                propagated[ids]
                - float(beta) * centroid_distance
            )

            chosen.append(
                int(
                    ids[
                        int(np.argmax(objective))
                    ]
                )
            )

        fixed = (
            _l2_normalize(
                np.asarray(
                    selected_representations,
                    dtype=np.float32,
                )
            )
            if (
                selected_representations is not None
                and len(selected_representations)
            )
            else np.empty(
                (0, x.shape[1]),
                dtype=np.float32,
            )
        )

        for _round in range(
            max(0, int(refine_rounds))
        ):
            anchors = x[chosen]

            if len(fixed):
                anchors = np.vstack(
                    [fixed, anchors]
                )

            refined: list[int] = []

            for cluster in range(n):
                ids = np.where(
                    labels == cluster
                )[0]

                if not len(ids):
                    continue

                local_mean = x[ids].mean(
                    axis=0,
                    keepdims=True,
                )
                local_distance = np.linalg.norm(
                    x[ids] - local_mean,
                    axis=1,
                )

                if len(anchors):
                    near = pairwise_distances(
                        x[ids],
                        anchors,
                        metric="euclidean",
                    ).min(axis=1)
                    near = np.clip(
                        near,
                        0.0,
                        float(mu),
                    )
                else:
                    near = np.zeros(
                        len(ids)
                    )

                objective = (
                    propagated[ids]
                    - float(beta)
                    * local_distance
                    + float(gamma)
                    * near
                )

                refined.append(
                    int(
                        ids[
                            int(
                                np.argmax(
                                    objective
                                )
                            )
                        ]
                    )
                )

            chosen = (
                list(dict.fromkeys(refined))
                or chosen
            )

        remaining = [
            index
            for index in np.argsort(
                -propagated
            )
            if int(index) not in chosen
        ]

        chosen.extend(
            int(index)
            for index in remaining[
                : max(0, n - len(chosen))
            ]
        )
        chosen = chosen[:n]

        centroid_distance = np.linalg.norm(
            x
            - clusterer.cluster_centers_[labels],
            axis=1,
        )

        # IMPORTANT FOR PARTIAL RANK AGGREGATION:
        #
        # PATRON selects/refines one representative per partition, but KMeans
        # cluster labels (0, 1, 2, ...) are arbitrary. Therefore the order in
        # which clusters are iterated is NOT a meaningful scientific ranking.
        #
        # Once the final batch is known, rank only those selected documents
        # using the same three ingredients used by the PATRON adaptation:
        #
        #   propagated uncertainty
        #   - beta * local representativeness distance
        #   + gamma * diversity from the rest of the final batch / previous set
        #
        # This DOES NOT change which documents were selected. It only defines
        # their within-selected order for truncated-Borda consensus.
        selected_priority = np.full(
            len(x),
            np.nan,
            dtype=np.float64,
        )

        for index in chosen:
            cluster = int(labels[index])
            cluster_ids = np.where(
                labels == cluster
            )[0]

            local_mean = x[
                cluster_ids
            ].mean(
                axis=0,
                keepdims=True,
            )

            local_distance = float(
                np.linalg.norm(
                    x[index]
                    - local_mean[0]
                )
            )

            other_selected = [
                item
                for item in chosen
                if item != index
            ]

            anchor_parts: list[np.ndarray] = []

            if len(fixed):
                anchor_parts.append(fixed)

            if other_selected:
                anchor_parts.append(
                    x[other_selected]
                )

            if anchor_parts:
                anchors = np.vstack(
                    anchor_parts
                )
                near = float(
                    pairwise_distances(
                        x[[index]],
                        anchors,
                        metric="euclidean",
                    ).min()
                )
                near = min(
                    near,
                    float(mu),
                )
            else:
                near = 0.0

            selected_priority[index] = (
                float(propagated[index])
                - float(beta)
                * local_distance
                + float(gamma)
                * near
            )

        chosen = sorted(
            chosen,
            key=lambda index: (
                -float(
                    selected_priority[index]
                ),
                int(index),
            ),
        )

        return MethodResult(
            chosen,
            scores=propagated,
            extras={
                "cluster": labels.astype(
                    np.int32
                ),
                "local_uncertainty": uncertainty,
                "propagated_uncertainty": propagated,
                "distance_to_cluster_centroid": centroid_distance,
                "selected_priority": selected_priority,
            },
            diagnostics={
                "k_neighbors": k,
                "rho": float(rho),
                "beta": float(beta),
                "mu": float(mu),
                "gamma": float(gamma),
                "refine_rounds": int(
                    refine_rounds
                ),
                "selected_rank_semantics": (
                    "descending final PATRON-style "
                    "uncertainty/representativeness/diversity priority"
                ),
            },
        )


def _ova_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    winners = np.argmax(probs, axis=1)
    result = np.zeros(len(probs), dtype=np.float64)
    for index, winner in enumerate(winners):
        event = probs[index, winner] * np.prod(1.0 - np.delete(probs[index], winner))
        result[index] = -math.log(max(float(event), 1e-12))
    return result


def _fuzzy_knn_graph(matrix: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.metrics import pairwise_distances
    except ImportError as error:
        raise ImportError("DEUCE requires scikit-learn: pip install 'lab[selection]'.") from error
    distance = pairwise_distances(matrix, matrix, metric=metric)
    weights = np.zeros_like(distance, dtype=np.float64)
    k = min(max(1, int(k)), max(1, len(matrix) - 1))
    for index in range(len(matrix)):
        neighbours = [item for item in np.argsort(distance[index]) if item != index][:k]
        if not neighbours:
            continue
        scale = max(float(np.median(distance[index, neighbours])), 1e-6)
        for neighbour in neighbours:
            weights[index, neighbour] = math.exp(-float(distance[index, neighbour]) / scale)
    weights = weights + weights.T - weights * weights.T
    np.fill_diagonal(weights, 0.0)
    return weights, distance


class DeuceSelection(SelectionMethod):
    """Document-level DEUCE adaptation using text geometry and class probabilities."""

    def select(
        self,
        candidate_ids: Sequence[str],
        n: int,
        *,
        representations: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
        k_neighbors: int = 10,
        dual_bonus: float = 0.5,
        propagation_alpha: float = 1.0,
        min_cluster_size: int = 3,
        **params: Any,
    ) -> MethodResult:
        _validate_budget(n, len(candidate_ids))
        x = _require_representations(representations, len(candidate_ids))
        probs = _require_probabilities(probabilities, len(candidate_ids))
        uncertainty = _ova_uncertainty(probs)
        text_graph, text_distance = _fuzzy_knn_graph(x, k_neighbors, "cosine")
        class_graph, class_distance = _fuzzy_knn_graph(probs, k_neighbors, "euclidean")
        both = (text_graph > 0) & (class_graph > 0)
        graph = np.where(
            both,
            text_graph * class_graph + float(dual_bonus),
            np.where(text_graph > 0, text_graph, class_graph),
        )
        np.fill_diagonal(graph, 0.0)
        row_sum = graph.sum(axis=1)
        propagated = (
            uncertainty
            + float(propagation_alpha)
            * ((graph @ uncertainty) / np.clip(row_sum, 1e-12, None))
        ) / (1.0 + float(propagation_alpha))

        labels, implementation = self._cluster(graph, min_cluster_size)
        clusters = [int(cluster) for cluster in np.unique(labels) if cluster >= 0]
        clusters.sort(key=lambda cluster: -float(propagated[labels == cluster].mean()))
        chosen: list[int] = []
        for cluster in clusters:
            ids = np.where(labels == cluster)[0]
            density = row_sum[ids] / max(float(row_sum.max()), 1e-12)
            objective = propagated[ids] * (1.0 + density)
            chosen.append(int(ids[int(np.argmax(objective))]))
            if len(chosen) >= n:
                break

        text_scale = max(float(text_distance.max()), 1e-12)
        class_scale = max(float(class_distance.max()), 1e-12)
        combined = 0.5 * (text_distance / text_scale) + 0.5 * (class_distance / class_scale)
        while len(chosen) < n:
            remaining = [index for index in range(len(x)) if index not in chosen]
            if chosen:
                diversity = combined[np.ix_(remaining, chosen)].min(axis=1)
            else:
                diversity = np.ones(len(remaining))
            values = propagated[remaining]
            normalized = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
            objective = diversity * (0.5 + 0.5 * normalized)
            chosen.append(int(remaining[int(np.argmax(objective))]))

        return MethodResult(
            chosen,
            scores=propagated,
            extras={
                "cluster": labels.astype(np.int32),
                "ova_uncertainty": uncertainty,
                "propagated_uncertainty": propagated,
                "dual_graph_density": row_sum,
            },
            diagnostics={
                "clustering_implementation": implementation,
                "k_neighbors": int(k_neighbors),
                "dual_bonus": float(dual_bonus),
                "propagation_alpha": float(propagation_alpha),
            },
        )

    @staticmethod
    def _cluster(graph: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray, str]:
        if len(graph) < 3:
            return np.zeros(len(graph), dtype=int), "single_cluster_small_n"
        similarity = graph / max(float(graph.max()), 1e-12)
        distance = 1.0 - similarity
        distance[graph <= 0] = 1.0
        distance = np.minimum(distance, distance.T)
        np.fill_diagonal(distance, 0.0)
        try:
            from sklearn.cluster import HDBSCAN
            labels = HDBSCAN(
                min_cluster_size=max(2, int(min_cluster_size)), metric="precomputed"
            ).fit_predict(distance)
            return labels.astype(int), "sklearn.HDBSCAN"
        except (ImportError, AttributeError):
            try:
                from sklearn.cluster import DBSCAN
            except ImportError as error:
                raise ImportError("DEUCE requires scikit-learn: pip install 'lab[selection]'.") from error
            labels = DBSCAN(
                eps=0.75,
                min_samples=max(2, int(min_cluster_size) // 2),
                metric="precomputed",
            ).fit_predict(distance)
            return labels.astype(int), "DBSCAN_fallback"


METHODS: dict[str, type[SelectionMethod]] = {
    "random": RandomSelection,
    "kmedoids": KMedoidsSelection,
    "facility-location": FacilityLocationSelection,
    "kcenter": KCenterSelection,
    "typiclust": TypiClustSelection,
    "alps": ALPSSelection,
    "patron": PatronSelection,
    "deuce": DeuceSelection,
    "uncertainty": UncertaintySelection,
    "uncertainty-facility": UncertaintyFacilitySelection,
}

METHOD_INFO: dict[str, MethodInfo] = {
    "random": MethodInfo(
        "random",
        "baseline",
        "Seeded random document sampling.",
        methodologies=("baseline",),
    ),
    "kmedoids": MethodInfo(
        "kmedoids",
        "diversity",
        "Conditional PAM-style k-medoids over document representations.",
        methodologies=("geometry",),
        requires_representation=True,
        reference="Kaufman & Rousseeuw, Finding Groups in Data (1990).",
        parameters={
            "max_swap_passes": "Maximum PAM-style refinement passes (default: 10).",
        },
    ),
    "facility-location": MethodInfo(
        "facility-location",
        "representativeness",
        "Greedy conditional facility-location maximisation.",
        methodologies=("geometry",),
        requires_representation=True,
        reference="Wei, Iyer & Bilmes, ICML 2015.",
    ),
    "kcenter": MethodInfo(
        "kcenter",
        "diversity",
        "Greedy farthest-first core-set selection.",
        methodologies=("geometry",),
        requires_representation=True,
        reference="Sener & Savarese, ICLR 2018.",
    ),
    "typiclust": MethodInfo(
        "typiclust",
        "representativeness",
        "Select typical documents from underrepresented clusters.",
        methodologies=("geometry",),
        requires_representation=True,
        implementation_note=(
            "Document-level adaptation; previous selections contribute cluster occupancy."
        ),
        reference="Hacohen, Dekel & Weinshall, ICML 2022.",
        parameters={
            "n_clusters": "Number of clusters; defaults to previous selections + requested budget.",
            "typicality_knn": "Neighbours used to estimate typicality (default: 20).",
        },
    ),
    "alps": MethodInfo(
        "alps",
        "cold-start",
        "Cluster MLM-surprisal representations and sample centroid-near documents.",
        methodologies=("surprisal", "geometry"),
        requires_representation=True,
        default_representation="alps",
        implementation_note=(
            "Uses ALPS surprisal embeddings; previous selections affect cluster occupancy."
        ),
        reference="Yuan, Lin & Boyd-Graber, EMNLP 2020.",
        parameters={
            "n_clusters": "Optional number of surprisal-space clusters.",
        },
    ),
    "patron": MethodInfo(
        "patron",
        "cold-start",
        "PATRON-style uncertainty propagation, partitioning and refinement.",
        methodologies=("uncertainty", "geometry"),
        requires_representation=True,
        requires_probabilities=True,
        implementation_note=(
            "Document-level adaptation; probabilities must be supplied explicitly."
        ),
        reference="Yu et al., ACL 2023.",
        parameters={
            "k_neighbors": "Neighbours for uncertainty propagation (default: 50).",
            "rho": "Distance decay in propagation (default: 0.01).",
            "beta": "Representativeness penalty (default: 0.5).",
            "mu": "Diversity-distance clipping margin (default: 0.5).",
            "gamma": "Diversity reward (default: 0.5).",
            "refine_rounds": "Refinement iterations (default: 1).",
        },
    ),
    "deuce": MethodInfo(
        "deuce",
        "cold-start",
        "DEUCE-style dual text/class graph selection.",
        methodologies=("uncertainty", "geometry"),
        requires_representation=True,
        requires_probabilities=True,
        implementation_note=(
            "Document-level adaptation; probabilities must be supplied explicitly."
        ),
        parameters={
            "k_neighbors": "Neighbours in each graph (default: 10).",
            "dual_bonus": "Extra weight for edges present in both graphs (default: 0.5).",
            "propagation_alpha": "Neighbour uncertainty weight (default: 1.0).",
            "min_cluster_size": "HDBSCAN minimum cluster size (default: 3).",
        },
    ),
    "uncertainty": MethodInfo(
        "uncertainty",
        "model-based",
        "Select highest externally supplied document uncertainty.",
        methodologies=("uncertainty",),
        requires_scores=True,
        implementation_note=(
            "Scores are task-agnostic and supplied as doc_id | score Parquet."
        ),
    ),
    "uncertainty-facility": MethodInfo(
        "uncertainty-facility",
        "hybrid",
        "Uncertainty filtering followed by conditional facility location.",
        methodologies=("uncertainty", "geometry"),
        requires_representation=True,
        requires_scores=True,
        parameters={
            "candidate_multiplier": "Uncertain candidate-pool size as n × multiplier (default: 5).",
        },
    ),
}


def list_methods() -> list[MethodInfo]:
    """Return metadata for all registered methods in registry order."""
    return [METHOD_INFO[name] for name in METHODS]

def list_methodologies() -> tuple[str, ...]:
    """Return every registered methodology tag in stable order."""
    tags: list[str] = []
    for info in list_methods():
        for tag in info.methodologies:
            if tag not in tags:
                tags.append(tag)
    return tuple(tags)


def methods_by_methodology(methodology: str) -> tuple[str, ...]:
    """Return registered method names carrying one methodology tag."""
    methodology = str(methodology).strip()
    if not methodology:
        raise ValueError("methodology must be a non-empty string.")
    return tuple(
        info.name
        for info in list_methods()
        if methodology in info.methodologies
    )


def group_methods_by_methodology(
    methods: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Group selected/all methods by methodology without changing their weights."""
    names = list(METHODS) if methods is None else list(methods)

    unknown = sorted(set(names) - set(METHODS))
    if unknown:
        raise ValueError(
            f"Unknown selection method(s): {unknown}. "
            f"Available: {', '.join(METHODS)}."
        )

    grouped: dict[str, list[str]] = {}
    for name in names:
        for tag in method_info(name).methodologies:
            grouped.setdefault(tag, []).append(name)

    return {
        tag: tuple(values)
        for tag, values in grouped.items()
    }


def method_info(name: str) -> MethodInfo:
    """Return metadata for one registered method."""
    try:
        return METHOD_INFO[name]
    except KeyError as error:
        raise ValueError(f"Unknown selection method {name!r}. Available: {', '.join(METHODS)}.") from error


def get_method(name: str) -> SelectionMethod:
    """Instantiate one registered selection method."""
    try:
        return METHODS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown selection method {name!r}. Available: {', '.join(METHODS)}.") from error
