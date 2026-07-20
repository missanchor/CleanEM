"""Dynamic, plan-driven canonicalization for CleanEM.

The module contains no dataset-specific repair decisions. An LLM planner chooses
column-level conventions using dirty-table profiles. This executor only applies a
small JSON DSL and validates every proposed operation against observed support.
Unsupported or contradictory plans abstain safely.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from cleanem_models import CellRecord, RepairCandidate


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).lower()).strip()


def _shape(value: Any) -> str:
    output: List[str] = []
    previous = None
    for char in _text(value):
        current = "D" if char.isdigit() else "A" if char.isalpha() else "S" if char.isspace() else char
        if current != previous or current not in {"D", "A", "S"}:
            output.append(current)
        previous = current
    return "".join(output) or "<empty>"


def _archetype(metadata: Dict[str, Dict[str, Any]], column: str) -> str:
    semantics = metadata.get(column, {}).get("semantics") or {}
    if isinstance(semantics, dict):
        return str(semantics.get("archetype") or "unknown")
    return str(getattr(semantics, "archetype", "unknown"))


def _surface(value: Any) -> str:
    text = _text(value)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*%", text):
        return "numeric_percent_suffix"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return "plain_numeric"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s+[^\d\s].*", text):
        return "numeric_with_suffix"
    if text and all(char.isalpha() or char.isspace() or char in "-_/'().&" for char in text):
        return "lexical"
    if any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
        return "mixed_alphanumeric"
    return "other"


def _distribution(counter: Counter, total: int, limit: int = 20) -> List[Dict[str, Any]]:
    return [
        {
            "value": value,
            "count": int(count),
            "ratio": round(float(count) / max(total, 1), 4),
        }
        for value, count in counter.most_common(limit)
    ]


def build_canonicalization_profiles(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build distribution summaries used by the LLM planner, without clean data."""
    enum_references: Dict[str, List[str]] = {}
    for column in df.columns:
        if _archetype(metadata, column) != "closed_enum":
            continue
        counts = Counter(_text(value) for value in df[column].tolist() if _text(value))
        enum_references[column] = [value for value, _ in counts.most_common(40)]

    profiles: Dict[str, Dict[str, Any]] = {}
    for column in df.columns:
        raw_values = [_text(value) for value in df[column].tolist() if _text(value)]
        total = len(raw_values)
        counts = Counter(raw_values)
        normalized_counts = Counter(_normalize(value) for value in raw_values)
        surface_counts = Counter(_surface(value) for value in raw_values)
        shape_counts = Counter(_shape(value) for value in raw_values)
        suffix_counts = Counter(
            value.rsplit(None, 1)[-1]
            for value in raw_values
            if len(value.split()) >= 2
        )
        prefix_counts = Counter(
            value.split(None, 1)[0]
            for value in raw_values
            if len(value.split()) >= 2
        )
        # Include frequent and rare observations so the planner can see both
        # the convention and candidate violations.
        frequent = [value for value, _ in counts.most_common(20)]
        rare = [value for value, count in counts.items() if count <= 3][:30]
        metadata_examples = [
            str(item.get("example") or item.get("value"))
            for item in metadata.get(column, {}).get("low_frequency_values") or []
            if item.get("example") is not None or item.get("value") is not None
        ][:30]
        profiles[column] = {
            "column": column,
            "archetype": _archetype(metadata, column),
            "row_count": int(len(df)),
            "non_missing_count": int(total),
            "unique_count": int(len(normalized_counts)),
            "unique_ratio": round(float(len(normalized_counts)) / max(total, 1), 4),
            "canonicalization_need": float(
                (metadata.get(column, {}).get("semantics") or {}).get(
                    "canonicalization_need",
                    0.0,
                )
            ),
            "possible_error_mechanisms": list(
                (metadata.get(column, {}).get("semantics") or {}).get(
                    "possible_error_mechanisms",
                    [],
                )
            ),
            "surface_distribution": _distribution(surface_counts, total),
            "shape_distribution": _distribution(shape_counts, total, limit=15),
            "top_values": _distribution(counts, total, limit=20),
            "top_suffix_tokens": _distribution(suffix_counts, total, limit=20),
            "top_prefix_tokens": _distribution(prefix_counts, total, limit=15),
            "frequent_examples": frequent,
            "rare_examples": list(dict.fromkeys(rare + metadata_examples))[:40],
            "reference_enum_columns": {
                ref_column: values
                for ref_column, values in enum_references.items()
                if ref_column != column
            },
        }
    return profiles


def _safe_regex(pattern: Any) -> Optional[re.Pattern]:
    text = str(pattern or "")
    if not text or len(text) > 500:
        return None
    try:
        return re.compile(text, flags=re.IGNORECASE)
    except re.error:
        return None


def _fullmatch_count(pattern: Optional[re.Pattern], values: Iterable[str]) -> int:
    if pattern is None:
        return 0
    return sum(1 for value in values if pattern.fullmatch(value))


def _candidate_score(candidate: RepairCandidate) -> float:
    return (
        0.60 * float(candidate.detection_confidence)
        + 0.35 * float(candidate.repairability)
        - 0.25 * float(candidate.ambiguity)
        + (0.05 if candidate.candidate_value is not None else 0.0)
    )


class PlanDrivenCanonicalizationAgent:
    """Validate and execute cached LLM plans against the current dirty table."""

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Dict[str, Any]],
        cell_registry: Dict[str, CellRecord],
        plans: Dict[str, Dict[str, Any]],
    ) -> None:
        self.df = df
        self.metadata = metadata
        self.cell_registry = cell_registry
        self.plans = plans
        self.values_by_column = {
            column: [_text(value) for value in df[column].tolist() if _text(value)]
            for column in df.columns
        }
        self.normalized_support = {
            column: Counter(_normalize(value) for value in values)
            for column, values in self.values_by_column.items()
        }
        self.validated_operations: Dict[str, List[Dict[str, Any]]] = {}
        self.validation_diagnostics: Dict[str, List[Dict[str, Any]]] = {}

    def _apply_operation(
        self,
        column: str,
        value: Any,
        operation: Dict[str, Any],
        row_index: int,
    ) -> Optional[Tuple[Any, Dict[str, Any], List[str]]]:
        text = _text(value)
        if not text:
            return None
        op_name = operation["operation"]
        params = operation.get("params") or {}

        if op_name == "regex_replace":
            pattern = _safe_regex(params.get("pattern"))
            if pattern is None or not pattern.search(text):
                return None
            replacement = str(params.get("replacement") or "")
            try:
                candidate = pattern.sub(replacement, text)
            except (re.error, IndexError):
                return None
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if not candidate or candidate == text:
                return None
            return candidate, {column: candidate}, ["validated regex surface transformation"]

        if op_name == "strip_suffix":
            suffix = str(params.get("suffix") or "")
            if not suffix or not text.lower().endswith(suffix.lower()):
                return None
            base = text[: len(text) - len(suffix)].strip()
            if not base:
                return None
            scale = params.get("numeric_scale", 1.0)
            try:
                numeric = float(base) * float(scale)
                candidate: Any = float(f"{numeric:.12g}")
            except (TypeError, ValueError):
                candidate = base
            return candidate, {column: candidate}, [f"stripped planned suffix {suffix!r}"]

        if op_name == "split_trailing_token":
            reference_column = str(params.get("reference_column") or "")
            if reference_column not in self.df.columns or reference_column == column:
                return None
            parts = text.rsplit(None, 1)
            if len(parts) != 2:
                return None
            candidate, trailing = parts[0].strip(), parts[1].strip()
            reference_values = self.normalized_support.get(reference_column, Counter())
            if _normalize(trailing) not in reference_values or not candidate:
                return None
            changed_fields: Dict[str, Any] = {column: candidate}
            reference_value = self.df.iloc[row_index][reference_column]
            if _is_missing(reference_value) or _normalize(reference_value) != _normalize(trailing):
                changed_fields[reference_column] = trailing
            return candidate, changed_fields, [
                f"trailing token is supported by reference column {reference_column!r}"
            ]

        if op_name == "truncate_at_marker":
            markers = params.get("markers") or []
            if isinstance(markers, str):
                markers = [markers]
            positions = [text.find(str(marker)) for marker in markers if str(marker) and text.find(str(marker)) >= 0]
            if not positions:
                return None
            position = min(positions)
            candidate = text[:position].strip(" ?-_/")
            suffix = text[position:]
            for marker in markers:
                suffix = suffix.replace(str(marker), "", 1).lstrip(" ?-_/")
            if not candidate:
                return None
            changed_fields: Dict[str, Any] = {column: candidate}
            reference_column = str(params.get("suffix_reference_column") or "")
            if suffix and reference_column in self.df.columns:
                reference_values = self.normalized_support.get(reference_column, Counter())
                if _normalize(suffix) in reference_values:
                    changed_fields[reference_column] = suffix
            return candidate, changed_fields, ["truncated at planner-proposed boundary marker"]

        if op_name == "normalize_surface":
            mapping = params.get("mapping") or {}
            if not isinstance(mapping, dict) or not mapping:
                return None
            candidate = text
            replaced = False
            for source, target in sorted(mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
                source_text = str(source)
                target_text = str(target)
                if not source_text or source_text.lower() == target_text.lower():
                    continue
                pattern = re.compile(re.escape(source_text), flags=re.IGNORECASE)
                candidate, count = pattern.subn(target_text, candidate)
                replaced = replaced or count > 0
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if not replaced or not candidate or candidate == text:
                return None
            return candidate, {column: candidate}, ["applied planner-proposed literal surface mapping"]

        return None

    def _validate_operation(self, column: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        values = self.values_by_column.get(column, [])
        plan = self.plans.get(column) or {}
        plan_confidence = float(plan.get("confidence") or 0.0)
        operation_confidence = float(operation.get("confidence") or 0.0)
        diagnostic: Dict[str, Any] = {
            "mechanism": operation.get("mechanism"),
            "operation": operation.get("operation"),
            "accepted": False,
            "reason": "",
            "changed_count": 0,
            "changed_ratio": 0.0,
            "candidate_support_ratio": 0.0,
        }
        if plan_confidence < 0.50 or operation_confidence < 0.50:
            diagnostic["reason"] = "low_planner_confidence"
            return diagnostic

        changed: List[Tuple[str, Any]] = []
        for row_index, value in enumerate(self.df[column].tolist()):
            transformed = self._apply_operation(column, value, operation, row_index)
            if transformed is not None:
                changed.append((_text(value), transformed[0]))
        diagnostic["changed_count"] = len(changed)
        diagnostic["changed_ratio"] = float(len(changed) / max(len(values), 1))
        if not changed:
            diagnostic["reason"] = "no_matching_values"
            return diagnostic

        candidate_support = sum(
            self.normalized_support[column].get(_normalize(candidate), 0) > 0
            for _, candidate in changed
        )
        candidate_support_ratio = float(candidate_support / len(changed))
        diagnostic["candidate_support_ratio"] = candidate_support_ratio

        params = operation.get("params") or {}
        raw_target_pattern = str(params.get("target_pattern") or "")
        known_surface_names = {
            "numeric_percent_suffix",
            "plain_numeric",
            "numeric_with_suffix",
            "lexical",
            "mixed_alphanumeric",
            "other",
        }
        target_pattern = (
            None
            if raw_target_pattern in known_surface_names
            else _safe_regex(raw_target_pattern)
        )
        source_pattern = _safe_regex(params.get("pattern"))
        if operation.get("operation") == "strip_suffix":
            suffix = str(params.get("suffix") or "")
            source_count = sum(value.lower().endswith(suffix.lower()) for value in values) if suffix else 0
        else:
            source_count = (
                sum(1 for value in values if source_pattern.search(value))
                if source_pattern is not None else 0
            )
        target_count = (
            sum(1 for value in values if _surface(value) == raw_target_pattern)
            if raw_target_pattern in known_surface_names
            else _fullmatch_count(target_pattern, values)
        )
        diagnostic["source_count"] = int(source_count)
        diagnostic["target_count"] = int(target_count)

        # Generic dominance guard: a plan cannot replace a surface that is more
        # common than its claimed target convention.
        has_target_contract = bool(raw_target_pattern)
        if has_target_contract and target_count <= source_count:
            diagnostic["reason"] = "target_surface_not_dominant"
            return diagnostic

        if has_target_contract:
            if raw_target_pattern in known_surface_names:
                candidate_target_matches = sum(
                    _surface(candidate) == raw_target_pattern for _, candidate in changed
                )
            else:
                candidate_target_matches = sum(
                    bool(target_pattern.fullmatch(str(candidate))) for _, candidate in changed
                ) if target_pattern is not None else 0
            candidate_target_match_ratio = float(candidate_target_matches / len(changed))
            diagnostic["candidate_target_match_ratio"] = candidate_target_match_ratio
            if candidate_target_match_ratio < 0.80:
                diagnostic["reason"] = "candidate_does_not_match_target_contract"
                return diagnostic

        # A broad surface contract (for example numeric_with_suffix) is not
        # sufficient evidence for choosing one concrete token over competing
        # variants. If regex replacement introduces a literal token, that exact
        # token must already outnumber the values that would be rewritten.
        if operation.get("operation") == "regex_replace":
            replacement = str(params.get("replacement") or "")
            literal_replacement = bool(replacement) and not re.search(r"[\\$]", replacement)
            if literal_replacement and any(char.isalpha() for char in replacement):
                literal_target_count = sum(
                    value.rstrip().endswith(replacement) for value in values
                )
                diagnostic["literal_target_count"] = int(literal_target_count)
                if literal_target_count <= len(changed):
                    diagnostic["reason"] = "literal_target_token_not_dominant"
                    return diagnostic

        op_name = operation.get("operation")
        if op_name == "split_trailing_token":
            if candidate_support_ratio < 0.20:
                diagnostic["reason"] = "geo_or_enum_target_lacks_support"
                return diagnostic
        elif not has_target_contract and candidate_support_ratio < 0.20:
            if diagnostic["changed_ratio"] > 0.02 or operation_confidence < 0.85:
                diagnostic["reason"] = "target_convention_lacks_observed_support"
                return diagnostic

        if diagnostic["changed_ratio"] > 0.80 and candidate_support_ratio < 0.50:
            diagnostic["reason"] = "unsafe_mass_rewrite"
            return diagnostic

        diagnostic["accepted"] = True
        diagnostic["reason"] = "validated_against_dirty_distribution"
        return diagnostic

    def validate_plans(self) -> Dict[str, List[Dict[str, Any]]]:
        self.validated_operations = {}
        self.validation_diagnostics = {}
        for column in self.df.columns:
            plan = self.plans.get(column) or {}
            diagnostics: List[Dict[str, Any]] = []
            accepted: List[Dict[str, Any]] = []
            if bool(plan.get("applicable")):
                for operation in plan.get("operations") or []:
                    diagnostic = self._validate_operation(column, operation)
                    diagnostics.append(diagnostic)
                    if diagnostic["accepted"]:
                        accepted.append(operation)
            self.validation_diagnostics[column] = diagnostics
            self.validated_operations[column] = accepted
        return self.validation_diagnostics

    def generate(self) -> Dict[str, List[RepairCandidate]]:
        if not self.validation_diagnostics:
            self.validate_plans()
        candidates: Dict[str, List[RepairCandidate]] = defaultdict(list)
        for cell_key, record in self.cell_registry.items():
            if _is_missing(record.raw_value):
                continue
            plan = self.plans.get(record.column) or {}
            for operation in self.validated_operations.get(record.column, []):
                transformed = self._apply_operation(
                    record.column,
                    record.raw_value,
                    operation,
                    record.row_index,
                )
                if transformed is None:
                    continue
                candidate_value, changed_fields, support = transformed
                operation_confidence = float(operation.get("confidence") or 0.0)
                plan_confidence = float(plan.get("confidence") or 0.0)
                repairability = float(operation.get("repairability") or 0.0)
                ambiguity = float(max(0.0, 1.0 - min(plan_confidence, operation_confidence)))
                candidates[cell_key].append(RepairCandidate(
                    cell_key=cell_key,
                    row_index=record.row_index,
                    column=record.column,
                    mechanism=str(operation.get("mechanism") or "other_canonicalization"),
                    candidate_value=candidate_value,
                    detection_confidence=float(min(plan_confidence, operation_confidence)),
                    repairability=repairability,
                    ambiguity=ambiguity,
                    changed_fields=changed_fields,
                    supporting_evidence=list(operation.get("rationale") or []) + support,
                ))
        for cell_key in list(candidates):
            candidates[cell_key].sort(key=_candidate_score, reverse=True)
        return dict(candidates)


def flatten_repair_candidates(
    repair_candidates: Dict[str, List[RepairCandidate]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell_key, candidates in repair_candidates.items():
        for rank, candidate in enumerate(candidates, start=1):
            rows.append({
                "cell_key": cell_key,
                "row_index": candidate.row_index,
                "column": candidate.column,
                "rank": rank,
                "mechanism": candidate.mechanism,
                "candidate_value": candidate.candidate_value,
                "detection_confidence": candidate.detection_confidence,
                "repairability_score": candidate.repairability,
                "ambiguity": candidate.ambiguity,
                "changed_fields": json.dumps(
                    candidate.changed_fields,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                "supporting_evidence": "; ".join(candidate.supporting_evidence),
            })
    return rows
