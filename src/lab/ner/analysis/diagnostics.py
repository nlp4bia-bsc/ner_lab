"""Strict-event reconstruction, boundary diagnostics and NER error taxonomy."""

from __future__ import annotations

import math
import string
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

import pandas as pd
from nervaluate import Evaluator

from lab.core.scoring import group_by_document, normalize_spans, to_nervaluate
from lab.ner.evaluation import span_metrics

STRICT_TP = "TP"
STRICT_FP = "FP"
STRICT_FN = "FN"


@dataclass(frozen=True)
class DiagnosticMatchConfig:
    """Configuration for post-hoc FN↔FP diagnostic association.

    The Sørensen-Dice coefficient is used as a normalized, symmetric span-overlap
    gate.  The default 0.60 is a repository diagnostic heuristic, not a value
    prescribed by Dice (1945).  It is deliberately configurable and is persisted
    in the analysis manifest.

    Reference
    ---------
    Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between
    Species. Ecology, 26(3), 297–302. https://doi.org/10.2307/1932409
    """

    overlap_threshold: float = 0.60

    def __post_init__(self) -> None:
        value = float(self.overlap_threshold)
        if not 0.0 < value <= 1.0:
            raise ValueError("diagnostic overlap_threshold must be in (0, 1].")


def evaluate_to_events(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    run_id: str,
    tags: list[str],
    min_overlap_percentage: float = 40.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Run the repository evaluator and expose its STRICT accounting as rows.

    The returned event table never changes evaluation semantics: exact STRICT
    matches are TP rows; every unmatched gold/prediction is represented as FN/FP.
    Later boundary/label pairing is diagnostic only and does not convert errors
    into partial credit.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must be a non-empty string.")

    metrics = span_metrics(
        gold=gold,
        predicted=predicted,
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )
    correct_predictions = _official_correct_predictions(
        gold,
        predicted,
        tags,
        min_overlap_percentage,
    )
    events = _strict_events(gold, predicted, str(run_id), correct_predictions)
    _assert_official_accounting(events, metrics)
    return events, metrics


def enrich_error_diagnostics(
    events: pd.DataFrame,
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    match_config: DiagnosticMatchConfig | None = None,
    multiclass: bool | None = None,
) -> pd.DataFrame:
    """Pair unmatched FP/FN rows and classify diagnostic error structure.

    Official STRICT accounting is never changed.  The diagnostic layer first
    detects conservative one-gold-to-many-prediction fragmentation groups, then
    applies global one-to-one matching to the remaining unmatched spans.  A
    one-to-one candidate must satisfy a normalized character-overlap gate based
    on Sørensen-Dice overlap.

    Label disagreement participates in the diagnostic taxonomy only when the
    evaluation contains more than one entity label.  In a single-label setup a
    label error is not a meaningful failure mode.
    """
    config = match_config or DiagnosticMatchConfig()
    if multiclass is None:
        labels = set(gold["label"].astype(str)) | set(predicted["label"].astype(str))
        multiclass = len(labels) > 1

    enriched = events.copy()
    defaults: dict[str, Any] = {
        "error_primary": None,
        "error_pair_id": None,
        "error_pair_role": None,
        "diagnostic_gold_id": None,
        "diagnostic_pred_id": None,
        "diagnostic_pred_ids": None,
        "error_label_agreement": None,
        "error_fragmentation": False,
        # Retained for schema compatibility. Merging is intentionally not
        # analysed in this profile.
        "error_merging": False,
        "fragmentation_pred_count": None,
        "fragmentation_pred_starts": None,
        "fragmentation_pred_ends": None,
        "fragmentation_pred_texts": None,
        "fragmentation_pred_labels": None,
        "fragmentation_gold_coverage": None,
        "fragmentation_dice": None,
        "error_duplicate_prediction": False,
        "error_nested_entity": False,
        "error_overlapping_entity": False,
        "span_intersection": None,
        "span_gold_length": None,
        "span_pred_length": None,
        "span_gold_coverage": None,
        "span_pred_coverage": None,
        "span_dice": None,
        "span_iou": None,
        "span_start_delta": None,
        "span_end_delta": None,
        "span_length_delta": None,
        "span_abs_start_delta": None,
        "span_abs_end_delta": None,
        "span_boundary_error": None,
        "span_relation": None,
        "span_one_character_offset": False,
        "span_leading_whitespace": False,
        "span_trailing_whitespace": False,
        "span_punctuation_included": False,
        "span_punctuation_excluded": False,
        "gold_overlaps_other_gold": False,
        "gold_nested": False,
        "gold_contains_other_gold": False,
        "gold_contained_by_other_gold": False,
        "gold_overlap_depth": None,
    }
    for column, value in defaults.items():
        enriched[column] = value

    gold_structure = structural_span_features(gold)
    duplicate_predictions = _duplicate_indices(predicted)
    unmatched_gold = set(
        enriched.loc[enriched["strict_is_fn"], "gold_id"].dropna().astype(int)
    )
    unmatched_pred = set(
        enriched.loc[enriched["strict_is_fp"], "pred_id"].dropna().astype(int)
    )

    for index, row in enriched.iterrows():
        if bool(row.strict_is_tp):
            enriched.at[index, "error_primary"] = "CORRECT"
        elif bool(row.strict_is_fn):
            enriched.at[index, "error_primary"] = "MISSED"
        else:
            enriched.at[index, "error_primary"] = "SPURIOUS"

        if bool(row.gold_exists) and pd.notna(row.gold_id):
            for name, value in gold_structure[int(row.gold_id)].items():
                enriched.at[index, name] = value
        if bool(row.pred_exists) and pd.notna(row.pred_id) and int(row.pred_id) in duplicate_predictions:
            enriched.at[index, "error_duplicate_prediction"] = True

    gold_event = {
        int(row.gold_id): index
        for index, row in enriched.iterrows()
        if bool(row.strict_is_fn) and pd.notna(row.gold_id)
    }
    pred_event = {
        int(row.pred_id): index
        for index, row in enriched.iterrows()
        if bool(row.strict_is_fp) and pd.notna(row.pred_id)
    }

    next_group_id = 0

    # Fragmentation is a one-gold-to-many-prediction diagnostic.  It is resolved
    # before one-to-one assignment and counts as ONE boundary error through the
    # gold-side diagnostic unit.  A candidate group is only used when no single
    # fragment would already pass the one-to-one Dice gate and the union of the
    # fragments passes the same normalized threshold.
    fragmentation_groups = _fragmentation_groups(
        gold,
        predicted,
        unmatched_gold,
        unmatched_pred,
        config,
    )
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for gold_id, pred_ids, details in fragmentation_groups:
        if gold_id in used_gold or any(pred_id in used_pred for pred_id in pred_ids):
            continue
        used_gold.add(gold_id)
        used_pred.update(pred_ids)
        group_id = next_group_id
        next_group_id += 1

        gold_index = gold_event[gold_id]
        enriched.at[gold_index, "error_pair_id"] = group_id
        enriched.at[gold_index, "error_pair_role"] = "GOLD"
        enriched.at[gold_index, "error_primary"] = "BOUNDARY_ERROR"
        enriched.at[gold_index, "diagnostic_gold_id"] = gold_id
        enriched.at[gold_index, "diagnostic_pred_id"] = pred_ids[0]
        enriched.at[gold_index, "diagnostic_pred_ids"] = list(map(int, pred_ids))
        enriched.at[gold_index, "error_fragmentation"] = True
        for name, value in details.items():
            enriched.at[gold_index, name] = value

        for pred_id in pred_ids:
            pred_index = pred_event[pred_id]
            enriched.at[pred_index, "error_pair_id"] = group_id
            enriched.at[pred_index, "error_pair_role"] = "PRED"
            enriched.at[pred_index, "error_primary"] = "BOUNDARY_ERROR"
            enriched.at[pred_index, "diagnostic_gold_id"] = gold_id
            enriched.at[pred_index, "diagnostic_pred_id"] = pred_id
            enriched.at[pred_index, "diagnostic_pred_ids"] = list(map(int, pred_ids))
            enriched.at[pred_index, "error_fragmentation"] = True
            for name, value in details.items():
                enriched.at[pred_index, name] = value

    unmatched_gold -= used_gold
    unmatched_pred -= used_pred

    pairs: list[tuple[int, int]] = []
    documents = sorted(
        set(gold["filename"].astype(str)) | set(predicted["filename"].astype(str))
    )
    for document in documents:
        gold_ids = sorted(
            index for index in unmatched_gold if str(gold.iloc[index].filename) == document
        )
        pred_ids = sorted(
            index for index in unmatched_pred if str(predicted.iloc[index].filename) == document
        )
        pairs.extend(
            _assign(
                gold,
                predicted,
                gold_ids,
                pred_ids,
                config=config,
                multiclass=bool(multiclass),
            )
        )

    for gold_id, pred_id in pairs:
        gold_row = gold.iloc[gold_id]
        pred_row = predicted.iloc[pred_id]
        details = _pair_details(gold_row, pred_row)
        same_boundary = details["span_boundary_error"] is None
        label_agreement = bool(details["error_label_agreement"])

        if same_boundary:
            # Same-boundary mismatches are meaningful only as label errors and
            # therefore only in a genuinely multi-label-type evaluation.
            if not bool(multiclass) or label_agreement:
                continue
            primary = "LABEL_ERROR"
        elif bool(multiclass) and not label_agreement:
            primary = "BOUNDARY_AND_LABEL_ERROR"
        else:
            primary = "BOUNDARY_ERROR"

        group_id = next_group_id
        next_group_id += 1
        for role, event_index in (("GOLD", gold_event[gold_id]), ("PRED", pred_event[pred_id])):
            enriched.at[event_index, "error_pair_id"] = group_id
            enriched.at[event_index, "error_pair_role"] = role
            enriched.at[event_index, "error_primary"] = primary
            enriched.at[event_index, "diagnostic_gold_id"] = gold_id
            enriched.at[event_index, "diagnostic_pred_id"] = pred_id
            enriched.at[event_index, "diagnostic_pred_ids"] = [int(pred_id)]
            for name, value in details.items():
                enriched.at[event_index, name] = value

    enriched["error_type"] = enriched["error_primary"]
    return enriched



def audit_annotations(
    spans: pd.DataFrame,
    documents: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Audit annotation integrity without changing or deduplicating the data."""
    documents = documents or {}
    duplicate_keys = Counter(
        (
            str(row.filename),
            str(row.label),
            int(row.start_span),
            int(row.end_span),
            str(row.text),
        )
        for row in spans.itertuples(index=False)
    )
    labels_by_boundary: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in spans.itertuples(index=False):
        labels_by_boundary[
            (str(row.filename), int(row.start_span), int(row.end_span))
        ].add(str(row.label))

    records: list[dict[str, Any]] = []
    for row in spans.itertuples(index=False):
        doc_id = str(row.filename)
        label = str(row.label)
        start = int(row.start_span)
        end = int(row.end_span)
        surface = str(row.text)
        issues: list[str] = []
        key = (doc_id, label, start, end, surface)

        if duplicate_keys[key] > 1:
            issues.append("DUPLICATE_ANNOTATION")
        if len(labels_by_boundary[(doc_id, start, end)]) > 1:
            issues.append("CONFLICTING_LABELS")
        if surface[:1].isspace():
            issues.append("LEADING_WHITESPACE")
        if surface[-1:].isspace():
            issues.append("TRAILING_WHITESPACE")

        document = documents.get(doc_id)
        if document is None:
            issues.append("DOCUMENT_UNAVAILABLE")
        elif end > len(document):
            issues.append("OUT_OF_RANGE")
        else:
            actual = document[start:end]
            if actual != surface:
                issues.append("SURFACE_MISMATCH")
                if any(
                    document[max(0, start + shift):min(len(document), end + shift)] == surface
                    for shift in (-1, 1)
                ):
                    issues.append("OFF_BY_ONE")
                if actual.replace("\r\n", "\n") == surface.replace("\r\n", "\n"):
                    issues.append("CRLF_LF_MISMATCH")
                if unicodedata.normalize("NFKC", actual) == unicodedata.normalize("NFKC", surface):
                    issues.append("UNICODE_NORMALIZATION_MISMATCH")

        substantive = [issue for issue in issues if issue != "DOCUMENT_UNAVAILABLE"]
        records.append(
            {
                "annotation_valid": not substantive,
                "annotation_issues": sorted(set(issues)),
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=["annotation_valid", "annotation_issues"],
    )


def entity_features(text: str) -> dict[str, Any]:
    """Dependency-light morphology features used for stratified error analysis."""
    text = str(text)
    words = text.split()
    letters = [character for character in text if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters
        else 0.0
    )
    acronym = bool(
        letters
        and len(text) <= 12
        and len(words) <= 3
        and uppercase_ratio >= 0.60
        and not text.islower()
    )
    return {
        "entity_character_length": len(text),
        "entity_whitespace_token_count": len(words),
        "entity_word_length_bin": str(len(words)) if len(words) < 5 else "5+",
        "entity_contains_digit": any(character.isdigit() for character in text),
        "entity_contains_hyphen": any(character in "-‐‑‒–—" for character in text),
        "entity_contains_slash": "/" in text,
        "entity_contains_parentheses": any(character in "()[]{}" for character in text),
        "entity_contains_punctuation": any(character in string.punctuation for character in text),
        "entity_contains_greek": any(
            "GREEK" in unicodedata.name(character, "") for character in text
        ),
        "entity_casing": _casing(text),
        "entity_acronym_like": acronym,
    }


def span_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Entity morphology plus overlap/nesting properties for every input span."""
    structural = structural_span_features(frame)
    rows = []
    for index, row in frame.iterrows():
        values = entity_features(str(row.text))
        values.update(structural[int(index)])
        rows.append(values)
    if rows:
        return pd.DataFrame.from_records(rows, index=frame.index)
    template = entity_features("")
    template.update(
        {
            "gold_overlaps_other_gold": False,
            "gold_nested": False,
            "gold_contains_other_gold": False,
            "gold_contained_by_other_gold": False,
            "gold_overlap_depth": 0,
            "error_nested_entity": False,
            "error_overlapping_entity": False,
        }
    )
    return pd.DataFrame(columns=list(template), index=frame.index)


def structural_span_features(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    by_document = {
        doc_id: group
        for doc_id, group in frame.groupby(frame["filename"].astype(str), sort=False)
    }
    for index, row in frame.iterrows():
        overlaps = []
        contains = False
        contained = False
        for other_index, other in by_document[str(row.filename)].iterrows():
            if int(index) == int(other_index):
                continue
            intersection = _intersection(
                int(row.start_span),
                int(row.end_span),
                int(other.start_span),
                int(other.end_span),
            )
            if intersection:
                overlaps.append(int(other_index))
                contains |= (
                    int(row.start_span) <= int(other.start_span)
                    and int(row.end_span) >= int(other.end_span)
                )
                contained |= (
                    int(other.start_span) <= int(row.start_span)
                    and int(other.end_span) >= int(row.end_span)
                )
        result[int(index)] = {
            "gold_overlaps_other_gold": bool(overlaps),
            "gold_nested": contains or contained,
            "gold_contains_other_gold": contains,
            "gold_contained_by_other_gold": contained,
            "gold_overlap_depth": len(overlaps) + 1,
            "error_nested_entity": contains or contained,
            "error_overlapping_entity": bool(overlaps),
        }
    return result


def _official_correct_predictions(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    tags: list[str],
    min_overlap_percentage: float,
) -> set[int]:
    gold_documents = group_by_document(normalize_spans(gold))
    predicted_documents = group_by_document(normalize_spans(predicted))
    doc_ids = sorted(set(gold_documents) | set(predicted_documents))
    evaluator = Evaluator(
        true=[to_nervaluate(gold_documents.get(doc_id, [])) for doc_id in doc_ids],
        pred=[to_nervaluate(predicted_documents.get(doc_id, [])) for doc_id in doc_ids],
        tags=tags,
        loader="dict",
        min_overlap_percentage=min_overlap_percentage,
    )
    evaluated = evaluator.evaluate()
    strict_indices = evaluated.get("overall_indices", {}).get("strict")
    if strict_indices is None:
        return set()

    by_document: dict[str, list[int]] = defaultdict(list)
    for frame_index, row in enumerate(predicted.itertuples(index=False)):
        by_document[str(row.filename)].append(frame_index)

    return {
        by_document[doc_ids[doc_index]][prediction_index]
        for doc_index, prediction_index in strict_indices.correct_indices
    }


def _strict_events(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    run_id: str,
    correct_predictions: set[int],
) -> pd.DataFrame:
    unused_gold: dict[tuple[str, str, int, int], deque[int]] = defaultdict(deque)
    for index, row in enumerate(gold.itertuples(index=False)):
        unused_gold[_identity(row)].append(index)

    records: list[dict[str, Any]] = []
    matched_gold: set[int] = set()

    for pred_index in sorted(correct_predictions):
        pred_row = predicted.iloc[pred_index]
        candidates = unused_gold[_identity(pred_row)]
        if not candidates:
            raise RuntimeError(
                "The official evaluator marked a non-identical span as STRICT correct."
            )
        gold_index = candidates.popleft()
        matched_gold.add(gold_index)
        records.append(
            _event(
                run_id,
                STRICT_TP,
                gold_index,
                gold.iloc[gold_index],
                pred_index,
                pred_row,
            )
        )

    for gold_index, gold_row in enumerate(gold.itertuples(index=False)):
        if gold_index not in matched_gold:
            records.append(
                _event(run_id, STRICT_FN, gold_index, gold_row, None, None)
            )

    for pred_index, pred_row in enumerate(predicted.itertuples(index=False)):
        if pred_index not in correct_predictions:
            records.append(
                _event(run_id, STRICT_FP, None, None, pred_index, pred_row)
            )

    records.sort(
        key=lambda row: (
            row["document_id"],
            row["gold_start"] if row["gold_exists"] else row["pred_start"],
            row["strict_outcome"],
        )
    )
    for event_id, record in enumerate(records):
        record["event_id"] = event_id
    return pd.DataFrame.from_records(records)


def _row_score(row: Any) -> float | None:
    """Return an optional score from a pandas Series or itertuples row.

    Analysis mixes ``DataFrame.iloc`` (Series) and ``DataFrame.itertuples``
    (namedtuple). ``Series.index`` is iterable, but a namedtuple inherits the
    built-in tuple ``index`` method, so checking ``"score" in row.index`` is
    invalid for namedtuples. This helper supports both row representations and
    keeps unscored span tables valid.
    """
    if row is None:
        return None

    if isinstance(row, pd.Series):
        if "score" not in row.index:
            return None
        value = row["score"]
    else:
        fields = getattr(row, "_fields", ())
        if "score" not in fields:
            return None
        value = getattr(row, "score", None)

    if value is None or pd.isna(value):
        return None

    return float(value)


def _identity(row: Any) -> tuple[str, str, int, int]:
    return (
        str(row.filename),
        str(row.label),
        int(row.start_span),
        int(row.end_span),
    )


def _event(run_id, outcome, gold_index, gold, pred_index, pred) -> dict[str, Any]:
    gold_exists = gold is not None
    pred_exists = pred is not None
    document_id = str(gold.filename if gold_exists else pred.filename)
    return {
        "run_id": run_id,
        "event_id": None,
        "document_id": document_id,
        "gold_id": None if gold_index is None else int(gold_index),
        "gold_exists": gold_exists,
        "gold_start": None if not gold_exists else int(gold.start_span),
        "gold_end": None if not gold_exists else int(gold.end_span),
        "gold_text": None if not gold_exists else str(gold.text),
        "gold_label": None if not gold_exists else str(gold.label),
        "pred_id": None if pred_index is None else int(pred_index),
        "pred_exists": pred_exists,
        "pred_start": None if not pred_exists else int(pred.start_span),
        "pred_end": None if not pred_exists else int(pred.end_span),
        "pred_text": None if not pred_exists else str(pred.text),
        "pred_label": None if not pred_exists else str(pred.label),
        "pred_confidence": _row_score(pred),
        "strict_outcome": outcome,
        "strict_is_tp": outcome == STRICT_TP,
        "strict_is_fp": outcome == STRICT_FP,
        "strict_is_fn": outcome == STRICT_FN,
        "strict_correct": outcome == STRICT_TP,
    }


def _assert_official_accounting(
    events: pd.DataFrame,
    metrics: dict[str, float | int],
) -> None:
    represented = {
        "correct": int(events["strict_is_tp"].sum()),
        "missed": int(events["strict_is_fn"].sum()),
        "spurious": int(events["strict_is_fp"].sum()),
    }
    official = {
        "correct": int(metrics.get("span_strict_correct", 0)),
        "missed": int(metrics.get("span_strict_missed", 0))
        + int(metrics.get("span_strict_incorrect", 0))
        + int(metrics.get("span_strict_partial", 0)),
        "spurious": int(metrics.get("span_strict_spurious", 0))
        + int(metrics.get("span_strict_incorrect", 0))
        + int(metrics.get("span_strict_partial", 0)),
    }
    if represented != official:
        raise RuntimeError(
            "Canonical STRICT events disagree with the official evaluator: "
            f"events={represented}, official={official}."
        )


def _duplicate_indices(frame: pd.DataFrame) -> set[int]:
    keys = ["filename", "label", "start_span", "end_span", "text"]
    mask = frame.duplicated(subset=keys, keep=False)
    return set(frame.index[mask].astype(int))


def _overlap_maps(gold, predicted, gold_ids, pred_ids):
    """Legacy overlap map using any positive character intersection.

    Retained for compatibility with downstream imports. New diagnostic pairing
    uses :func:`_span_overlap_metrics` plus ``DiagnosticMatchConfig`` instead.
    """
    by_gold: dict[int, list[int]] = defaultdict(list)
    by_pred: dict[int, list[int]] = defaultdict(list)
    for gold_id in gold_ids:
        gold_row = gold.iloc[gold_id]
        for pred_id in pred_ids:
            pred_row = predicted.iloc[pred_id]
            if str(gold_row.filename) != str(pred_row.filename):
                continue
            if _intersection(
                int(gold_row.start_span),
                int(gold_row.end_span),
                int(pred_row.start_span),
                int(pred_row.end_span),
            ) > 0:
                by_gold[gold_id].append(pred_id)
                by_pred[pred_id].append(gold_id)
    return by_gold, by_pred



def _assign(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    gold_ids: list[int],
    pred_ids: list[int],
    *,
    config: DiagnosticMatchConfig,
    multiclass: bool,
) -> list[tuple[int, int]]:
    """Globally assign only diagnostically eligible one-to-one span pairs."""
    size = max(len(gold_ids), len(pred_ids))
    if not size:
        return []
    weights = [[0.0] * size for _ in range(size)]
    for i, gold_id in enumerate(gold_ids):
        for j, pred_id in enumerate(pred_ids):
            weights[i][j] = _association_weight(
                gold.iloc[gold_id],
                predicted.iloc[pred_id],
                config=config,
                multiclass=multiclass,
            )
    assignment = _hungarian_max(weights)
    return [
        (gold_ids[i], pred_ids[j])
        for i, j in enumerate(assignment[: len(gold_ids)])
        if j < len(pred_ids) and weights[i][j] > 0
    ]



def _association_weight(
    gold,
    pred,
    *,
    config: DiagnosticMatchConfig,
    multiclass: bool,
) -> float:
    """Priority among already-eligible diagnostic span pairs.

    Eligibility is determined by normalized Sørensen-Dice overlap. The weight
    then favours exact boundaries, stronger overlap and smaller boundary shifts.
    Label agreement is only a tie-break signal in multi-label-type evaluations;
    it never controls eligibility.
    """
    if str(gold.filename) != str(pred.filename):
        return 0.0
    metrics = _span_overlap_metrics(gold, pred)
    if float(metrics["span_dice"]) < float(config.overlap_threshold):
        return 0.0

    same_boundary = (
        int(gold.start_span) == int(pred.start_span)
        and int(gold.end_span) == int(pred.end_span)
    )
    distance = abs(int(pred.start_span) - int(gold.start_span)) + abs(
        int(pred.end_span) - int(gold.end_span)
    )
    confidence_value = _row_score(pred)
    confidence = 0.0 if confidence_value is None else confidence_value
    label_bonus = 1.0 if multiclass and str(gold.label) == str(pred.label) else 0.0

    return (
        1_000_000 * same_boundary
        + 100_000 * float(metrics["span_dice"])
        + 10_000 * float(metrics["span_iou"])
        + 1_000 * label_bonus
        + 100 / (1 + distance)
        + confidence
    )



def _hungarian_max(weights: list[list[float]]) -> list[int]:
    """Deterministic O(n^3) maximum-weight assignment for a square matrix."""
    n = len(weights)
    if not n:
        return []
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    match = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minimum = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = match[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while True:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
            if j0 == 0:
                break

    answer = [0] * n
    for j in range(1, n + 1):
        answer[match[j] - 1] = j - 1
    return answer


def _pair_details(gold, pred) -> dict[str, Any]:
    gold_start = int(gold.start_span)
    gold_end = int(gold.end_span)
    pred_start = int(pred.start_span)
    pred_end = int(pred.end_span)
    start_delta = pred_start - gold_start
    end_delta = pred_end - gold_end
    overlap = _span_overlap_metrics(gold, pred)
    same_boundary = start_delta == 0 and end_delta == 0

    if same_boundary:
        boundary_error = None
        relation = "EXACT_BOUNDARY"
    elif start_delta != 0 and end_delta == 0:
        boundary_error = "LEFT_BOUNDARY_ERROR"
        relation = _relation(gold_start, gold_end, pred_start, pred_end)
    elif start_delta == 0 and end_delta != 0:
        boundary_error = "RIGHT_BOUNDARY_ERROR"
        relation = _relation(gold_start, gold_end, pred_start, pred_end)
    else:
        boundary_error = "BOTH_BOUNDARIES_ERROR"
        relation = _relation(gold_start, gold_end, pred_start, pred_end)

    gold_text = str(gold.text)
    pred_text = str(pred.text)
    punctuation_gold = gold_text.strip(string.punctuation + " ")
    punctuation_pred = pred_text.strip(string.punctuation + " ")

    return {
        "error_label_agreement": str(gold.label) == str(pred.label),
        **overlap,
        "span_start_delta": int(start_delta),
        "span_end_delta": int(end_delta),
        "span_length_delta": int((pred_end - pred_start) - (gold_end - gold_start)),
        "span_abs_start_delta": abs(int(start_delta)),
        "span_abs_end_delta": abs(int(end_delta)),
        "span_boundary_error": boundary_error,
        "span_relation": relation,
        "span_one_character_offset": abs(start_delta) == 1 or abs(end_delta) == 1,
        "span_leading_whitespace": pred_text[:1].isspace(),
        "span_trailing_whitespace": pred_text[-1:].isspace(),
        "span_punctuation_included": (
            punctuation_gold == punctuation_pred and len(pred_text) > len(gold_text)
        ),
        "span_punctuation_excluded": (
            punctuation_gold == punctuation_pred and len(pred_text) < len(gold_text)
        ),
    }



def _span_overlap_metrics(gold, pred) -> dict[str, float | int]:
    gold_start = int(gold.start_span)
    gold_end = int(gold.end_span)
    pred_start = int(pred.start_span)
    pred_end = int(pred.end_span)
    gold_length = max(0, gold_end - gold_start)
    pred_length = max(0, pred_end - pred_start)
    intersection = _intersection(gold_start, gold_end, pred_start, pred_end)
    union = gold_length + pred_length - intersection
    denominator = gold_length + pred_length
    return {
        "span_intersection": int(intersection),
        "span_gold_length": int(gold_length),
        "span_pred_length": int(pred_length),
        "span_gold_coverage": float(intersection / gold_length) if gold_length else 0.0,
        "span_pred_coverage": float(intersection / pred_length) if pred_length else 0.0,
        "span_dice": float(2.0 * intersection / denominator) if denominator else 0.0,
        "span_iou": float(intersection / union) if union else 0.0,
    }


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _fragmentation_group_details(gold_row, pred_rows: list[Any]) -> dict[str, Any]:
    gold_start = int(gold_row.start_span)
    gold_end = int(gold_row.end_span)
    gold_length = max(0, gold_end - gold_start)
    merged = _merge_intervals(
        [(int(row.start_span), int(row.end_span)) for row in pred_rows]
    )
    pred_union_length = sum(end - start for start, end in merged)
    intersection = sum(
        _intersection(gold_start, gold_end, start, end) for start, end in merged
    )
    union = gold_length + pred_union_length - intersection
    denominator = gold_length + pred_union_length
    hull_start = min(start for start, _ in merged)
    hull_end = max(end for _, end in merged)
    start_delta = hull_start - gold_start
    end_delta = hull_end - gold_end
    return {
        "error_label_agreement": all(str(row.label) == str(gold_row.label) for row in pred_rows),
        "fragmentation_pred_count": int(len(pred_rows)),
        "fragmentation_pred_starts": [int(row.start_span) for row in pred_rows],
        "fragmentation_pred_ends": [int(row.end_span) for row in pred_rows],
        "fragmentation_pred_texts": [str(row.text) for row in pred_rows],
        "fragmentation_pred_labels": [str(row.label) for row in pred_rows],
        "fragmentation_gold_coverage": float(intersection / gold_length) if gold_length else 0.0,
        "fragmentation_dice": float(2.0 * intersection / denominator) if denominator else 0.0,
        "span_intersection": int(intersection),
        "span_gold_length": int(gold_length),
        "span_pred_length": int(pred_union_length),
        "span_gold_coverage": float(intersection / gold_length) if gold_length else 0.0,
        "span_pred_coverage": float(intersection / pred_union_length) if pred_union_length else 0.0,
        "span_dice": float(2.0 * intersection / denominator) if denominator else 0.0,
        "span_iou": float(intersection / union) if union else 0.0,
        "span_start_delta": int(start_delta),
        "span_end_delta": int(end_delta),
        "span_length_delta": int((hull_end - hull_start) - gold_length),
        "span_abs_start_delta": abs(int(start_delta)),
        "span_abs_end_delta": abs(int(end_delta)),
        "span_boundary_error": "FRAGMENTATION",
        "span_relation": "FRAGMENTATION",
        "span_one_character_offset": False,
        "span_leading_whitespace": False,
        "span_trailing_whitespace": False,
        "span_punctuation_included": False,
        "span_punctuation_excluded": False,
    }


def _fragmentation_groups(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    gold_ids: set[int],
    pred_ids: set[int],
    config: DiagnosticMatchConfig,
) -> list[tuple[int, list[int], dict[str, Any]]]:
    """Find conservative one-gold-to-many-prediction fragmentation groups.

    A group is considered only when at least two unmatched predictions overlap
    the gold span, no individual fragment already passes the one-to-one Dice
    threshold, and their union passes the same normalized threshold. Candidate
    groups are ordered deterministically by group Dice, gold coverage and span id.
    """
    candidates: list[tuple[float, float, int, list[int], dict[str, Any]]] = []
    for gold_id in sorted(gold_ids):
        gold_row = gold.iloc[gold_id]
        overlapping = [
            pred_id
            for pred_id in sorted(pred_ids)
            if str(predicted.iloc[pred_id].filename) == str(gold_row.filename)
            and _intersection(
                int(gold_row.start_span), int(gold_row.end_span),
                int(predicted.iloc[pred_id].start_span), int(predicted.iloc[pred_id].end_span),
            ) > 0
        ]
        if len(overlapping) < 2:
            continue
        # If one prediction is already a plausible one-to-one pair, prefer the
        # normal Hungarian path and leave extras spurious rather than overcalling
        # fragmentation.
        if any(
            _span_overlap_metrics(gold_row, predicted.iloc[pred_id])["span_dice"]
            >= float(config.overlap_threshold)
            for pred_id in overlapping
        ):
            continue
        rows = [predicted.iloc[pred_id] for pred_id in overlapping]
        details = _fragmentation_group_details(gold_row, rows)
        if float(details["fragmentation_dice"]) < float(config.overlap_threshold):
            continue
        candidates.append(
            (
                float(details["fragmentation_dice"]),
                float(details["fragmentation_gold_coverage"]),
                int(gold_id),
                overlapping,
                details,
            )
        )

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    result: list[tuple[int, list[int], dict[str, Any]]] = []
    used_pred: set[int] = set()
    for _dice, _coverage, gold_id, overlapping, details in candidates:
        available = [pred_id for pred_id in overlapping if pred_id not in used_pred]
        if len(available) < 2:
            continue
        if available != overlapping:
            details = _fragmentation_group_details(
                gold.iloc[gold_id],
                [predicted.iloc[pred_id] for pred_id in available],
            )
            if float(details["fragmentation_dice"]) < float(config.overlap_threshold):
                continue
        result.append((gold_id, available, details))
        used_pred.update(available)
    return result


def _relation(gold_start, gold_end, pred_start, pred_end) -> str:
    if pred_start <= gold_start and pred_end >= gold_end:
        return "PRED_CONTAINS_GOLD"
    if gold_start <= pred_start and gold_end >= pred_end:
        return "GOLD_CONTAINS_PRED"
    if pred_start < gold_start:
        return "LEFT_OVERLAP"
    return "RIGHT_OVERLAP"


def _intersection(first_start, first_end, second_start, second_end) -> int:
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def _casing(text: str) -> str:
    if text.isupper():
        return "UPPERCASE"
    if text.islower():
        return "LOWERCASE"
    if text.istitle():
        return "TITLECASE"
    return "MIXEDCASE" if any(character.isalpha() for character in text) else "UNCASED"
