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


@dataclass
class RuleSetConflictAnalysis:
    """分析 base_rules 和 clean_base_rules 之间的冲突.

    冲突定义：clean_rules (AND逻辑) = true 且 detection_rules (OR逻辑) = true
    缺口定义：clean_rules (AND逻辑) = false 且 detection_rules (OR逻辑) = false
    """
    column: str
    total_rows: int

    # 冲突样本：clean(AND)=true 且 dirty(OR)=true
    conflict_samples: List[Dict[str, Any]]

    # 缺口样本：clean(AND)=false 且 dirty(OR)=false
    gap_samples: List[Dict[str, Any]]

    # 指标
    conflict_count: int
    conflict_rate: float
    gap_count: int
    gap_rate: float

    # 按冲突频率排序的数据点：(row_index, conflict_count)
    sorted_conflict_points: List[Tuple[int, int]]

    # 规则覆盖统计
    base_coverage_rate: float  # base_rules 覆盖的数据点比例
    clean_coverage_rate: float  # clean_rules 覆盖的数据点比例
    combined_coverage_rate: float  # 任一规则集覆盖的数据点比例


@dataclass
class RuleSetRefinementResult:
    """规则集细化结果."""
    column: str
    refined_base_rules: List[DualRule]
    refined_clean_rules: List[DualRule]
    refinement_history: List[RefinementRound]
    final_conflicts: RuleSetConflictAnalysis
    success: bool
    final_message: str


@dataclass
class RuleSetRefinementContext:
    """规则细化上下文，包含冲突和缺口样本."""
    column: str
    conflict_samples: List[Dict[str, Any]]
    gap_samples: List[Dict[str, Any]]
    priority_conflict_samples: List[Dict[str, Any]]  # 冲突最多的样本
    instruction: str  # 'resolve_conflicts', 'expand_coverage', 'delete_conflicting'
    conflict_frequency: Dict[int, int]  # row_index -> conflict_count