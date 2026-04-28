from dataclasses import dataclass, field
from typing import Any, List


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
