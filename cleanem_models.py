from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValueRecord:
    key: str
    column: str
    normalized_value: str
    representative_value: Any
    row_indices: List[int]
    support_count: int
    prior: float = 0.5
    confidence: float = 0.0


@dataclass
class CellRecord:
    key: str
    row_index: int
    column: str
    raw_value: Any
    value_key: str
    posterior: float = 0.5
    confidence: float = 0.0


@dataclass
class EvidenceObservation:
    target_scope: str
    target_key: str
    source_id: str
    family: str
    polarity: str
    strength: float
    hard: bool
    reason_code: str
    metadata: dict = field(default_factory=dict)


@dataclass
class HardEvidenceLabel:
    target_scope: str
    target_key: str
    label: str
    reasons: List[str]


@dataclass
class ColumnSemantics:
    column: str
    archetype: str
    confidence: float
    open_set_score: float
    structure_strength: float
    canonicalization_need: float
    possible_error_mechanisms: List[str]
    rationale: List[str]


@dataclass
class RepairCandidate:
    cell_key: str
    row_index: int
    column: str
    mechanism: str
    candidate_value: Any
    detection_confidence: float
    repairability: float
    ambiguity: float = 0.0
    changed_fields: Dict[str, Any] = field(default_factory=dict)
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class EvidenceContribution:
    """One auditable evidence observation and its LOEO posterior effect."""

    target_scope: str
    source_id: str
    family: str
    polarity: str
    reason_code: str
    strength: float
    hard: bool
    feature_name: str
    feature_weight: float
    signed_feature_value: float
    weighted_logit: float
    posterior_without: float
    posterior_contribution: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationTrace:
    """Unified, serializable explanation record for one table cell."""

    cell_key: str
    row_index: int
    column: str
    raw_value: Any
    archetype: str
    posterior: float
    confidence: float
    decision_certainty: float
    full_logit: float
    threshold: float
    decision: str
    hard_label: str
    active_oracle_label: str
    calibration_fitted: bool
    calibration_reason: str
    primary_reason: str
    supporting_reasons: List[str]
    counter_evidence: List[str]
    evidence: List[EvidenceContribution]
    repair_candidate: Optional[Any] = None
    repair_mechanism: str = ""
    repairability: float = 0.0
    repair_used_in_inference: bool = False
    repair_recommended: bool = False
    natural_language_explanation: str = ""
