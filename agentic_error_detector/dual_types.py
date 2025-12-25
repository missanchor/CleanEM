"""
Dual Verification Types for P_clean/P_dirty dual rule system.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd


@dataclass
class DualRule:
    """A pair of rules: clean predicate and dirty predicate."""
    column: str
    agent_name: str
    clean_rule_str: str
    dirty_rule_str: str
    clean_rule_func: callable = None
    dirty_rule_func: callable = None
    round_number: int = 0  # Iteration round (for refinement tracking)


@dataclass
class DualEvaluationResult:
    """Evaluation result for a dual rule pair."""
    column: str
    rule: DualRule
    total_rows: int

    # Four-zone classification
    conflict_count: int
    grey_count: int
    determined_clean_count: int
    determined_dirty_count: int

    # Rates
    conflict_rate: float
    grey_rate: float
    dirty_rate: float  # determined_dirty_count / total_rows
    clean_rate: float  # determined_clean_count / total_rows

    # Samples for debugging
    conflict_samples: List[Dict[str, Any]]
    grey_samples: List[Dict[str, Any]]
    determined_dirty_samples: List[Dict[str, Any]]
    determined_clean_samples: List[Dict[str, Any]]

    # Status
    status: str  # 'accept', 'reject_conflict', 'reject_grey', 'reject_all_dirty', 'reject_all_clean'
    violation_message: str = ""


@dataclass
class RefinementRound:
    """Track one round of rule refinement."""
    round_number: int
    column: str
    initial_rule: DualRule
    refined_rule: DualRule
    initial_metrics: DualEvaluationResult
    final_metrics: DualEvaluationResult
    samples_used: Dict[str, List[Dict[str, Any]]]  # Type -> samples
    refinement_prompt: str
    success: bool


@dataclass
class DualRuleSet:
    """Collection of dual rules for all columns."""
    column_rules: Dict[str, List[DualRule]]  # column -> list of candidate rules
    best_rules: Dict[str, DualRule]  # column -> best rule
    evaluation_results: Dict[str, DualEvaluationResult]  # column -> evaluation
    refinement_history: Dict[str, List[RefinementRound]]  # column -> refinement rounds