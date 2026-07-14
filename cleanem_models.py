from dataclasses import dataclass, field
from typing import Any, Dict, List


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
