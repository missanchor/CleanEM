"""
Pattern-related data types for PatternExplorer and shared pattern specifications.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PatternEntry:
    """A single pattern entry with regex, coverage, and description."""
    regex: str
    coverage: float
    description: str = ""


@dataclass
class PatternSpec:
    """Unified pattern specification structure for a column.

    This is the output of PatternExplorer and is consumed by both
    PatternLegislator (P_dirty) and CleanPatternLegislator (P_clean).
    """
    column: str
    patterns: List[PatternEntry]
    overall_coverage: float
    exploration_rounds: int = 0
    uncovered_examples: List[str] = field(default_factory=list)

    def get_high_quality_patterns(self, threshold: float = 0.05) -> List[PatternEntry]:
        """Get patterns with coverage above threshold."""
        return [p for p in self.patterns if p.coverage >= threshold]

    def get_combined_regex(self) -> str:
        """Build combined regex from all patterns."""
        if not self.patterns:
            return ""
        # Combine with alternation, wrapping each in non-capturing group
        return "|".join(f"(?:{p.regex})" for p in self.patterns)


@dataclass
class ValueAnnotation:
    """Value-level annotation result from LLM."""
    value: str
    normalized_value: str
    label: str  # "MISSING_TOKEN"|"VALID"|"AMBIGUOUS"|"LIKELY_TYPO"|"VALID_RARE"
    count: int = 1


@dataclass
class ExplorationFeedback:
    """Feedback from one exploration round."""
    round_number: int
    current_coverage: float
    patterns: List[PatternEntry]
    uncovered_examples: List[str]
    uncovered_shapes: Dict[str, int] = field(default_factory=dict)
    improvement: float = 0.0


@dataclass
class AnnotationBatch:
    """A batch of values for LLM annotation."""
    column: str
    values: List[Dict[str, Any]]  # [{"value": "...", "count": N}, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
