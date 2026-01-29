"""
Dual Verification Types for P_clean/P_dirty dual rule system.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from core.utils import safe_dict, safe_float


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


@dataclass
class RefinementTask:
    """Task specification for rule refinement."""
    column: str
    conflict_type: str  # 'SUBSET', 'INTERSECT', 'SUPERSET', 'DISJOINT'
    
    # Current rules
    current_clean_rule: str
    current_dirty_rule: str
    
    # Problem samples
    conflict_samples: List[Dict[str, Any]]
    grey_samples: List[Dict[str, Any]]
    
    # Refinement strategy
    strategy: str = "default"  # 'clean', 'dirty', 'both', 'fallback'
    
    # Metadata
    metadata: Dict[str, Any] = None


@dataclass
class DisjointnessResult:
    """Result of disjointness validation after refinement."""
    column: str
    is_disjoint: bool
    intersection_count: int
    intersection_samples: List[Dict[str, Any]]
    
    # Metrics
    clean_coverage_rate: float = 0.0
    dirty_coverage_rate: float = 0.0
    gap_rate: float = 0.0  # both false
    
    # Validation status
    validation_status: str = "unknown"  # 'pass', 'fail', 'error'


# ============================================================================
# CleanRule-Level Refinement Types (Phase 1)
# ============================================================================

@dataclass
class CleanRule:
    """Individual clean rule or dirty rule with version tracking."""
    name: str                  # clean rule name (e.g., 'completeness') or agent name (e.g., 'TypoAgent')
    rule_str: str              # Lambda string
    rule_func: callable = None # Compiled lambda
    version: int = 0           # Incremented on modification
    modification_log: List[str] = None  # List of modification reasons

    def __post_init__(self):
        if self.modification_log is None:
            self.modification_log = []


@dataclass
class CleanRuleSet:
    """Collection of clean rules and dirty rules for a column."""
    column: str
    clean_rules: Dict[str, CleanRule]  # clean_rule_name -> CleanRule
    dirty_rules: Dict[str, CleanRule]  # agent_name -> CleanRule

    def to_dual_rule(self, agent_factory=None) -> DualRule:
        """Convert to DualRule for compatibility with existing judge methods.

        Strategy:
        - P_clean: AND of all clean rules (all rules satisfied)
        - P_dirty: OR of all dirty rules (at least one rule detected)

        Args:
            agent_factory: Optional factory for LLM-based rule fixing
        """
        clean_rule_str = self._compose_and_str_safe(self.clean_rules, agent_factory)
        dirty_rule_str = self._compose_or_str_safe(self.dirty_rules, agent_factory)

        # Final compile attempt with error handling
        try:
            clean_func = eval(clean_rule_str, safe_dict) if clean_rule_str else lambda v, r=None: True
        except Exception as e:
            print(f"  ⚠ Failed to compile clean_rule: {e}")
            clean_func = lambda v, r=None: True

        try:
            dirty_func = eval(dirty_rule_str, safe_dict) if dirty_rule_str else lambda v, r=None: False
        except Exception as e:
            print(f"  ⚠ Failed to compile dirty_rule: {e}")
            dirty_func = lambda v, r=None: False

        return DualRule(
            column=self.column,
            agent_name="PillarComposite",
            clean_rule_str=clean_rule_str,
            dirty_rule_str=dirty_rule_str,
            clean_rule_func=clean_func,
            dirty_rule_func=dirty_func
        )

    def _fix_rule_with_llm(self, broken_rule: str, error_msg: str,
                           rule_name: str, agent_factory) -> str:
        """Attempt to fix a broken rule using LLM.

        Args:
            broken_rule: The original broken rule
            error_msg: The error message from compilation
            rule_name: Name of the rule for context
            agent_factory: Factory to create RuleFixerAgent

        Returns:
            Fixed rule string or None if fixing failed
        """
        if not agent_factory:
            return None

        try:
            from agent import RuleFixerAgent
            fixer = RuleFixerAgent(
                base_url=agent_factory.base_url,
                model=agent_factory.model
            )
            context = f"Rule name: {rule_name}, Column: {self.column}"
            fixed = fixer.fix_syntax_error(broken_rule, error_msg, context)
            if fixed:
                print(f"  ✓ Rule '{rule_name}' fixed by LLM")
            return fixed
        except Exception as e:
            print(f"  ⚠ Failed to fix rule '{rule_name}' with LLM: {e}")
            return None

    @staticmethod
    def _compile_rule_safely(rule_str: str):
        """Try to compile a rule string safely.

        Returns:
            tuple: (success: bool, func: callable or None, error: str or None)
        """
        try:
            return (True, eval(rule_str, safe_dict), None)
        except SyntaxError as e:
            return (False, None, f"SyntaxError: {e}")
        except Exception as e:
            return (False, None, f"{type(e).__name__}: {e}")

    @staticmethod
    def _get_arg_count(rule_str: str) -> int:
        """Extract the number of arguments from a lambda string."""
        import re
        # Match "lambda arg1, arg2, ..." or "lambda arg"
        match = re.match(r'lambda\s+([^\(:)]+)', rule_str)
        if not match:
            return 2  # Default to 2 if can't parse
        args_str = match.group(1)
        # Count commas + 1, handling nested structures
        args = [a.strip() for a in args_str.split(',')]
        return len(args)

    def _compose_and_str_safe(self, rules: Dict[str, CleanRule],
                               agent_factory=None) -> str:
        """Generate AND-composed lambda string with self-healing for syntax errors.

        Args:
            rules: Dict of clean rules
            agent_factory: Optional factory for LLM-based rule fixing

        Returns:
            Composed lambda string
        """
        if not rules:
            return "lambda value, row=None: True"

        valid_parts = []
        for name, rule in rules.items():
            # Try to compile the rule
            success, _, error = CleanRuleSet._compile_rule_safely(rule.rule_str)

            if not success:
                print(f"  ⚠ Rule '{name}' has syntax error: {error}")
                # Try to fix with LLM
                if agent_factory:
                    fixed = self._fix_rule_with_llm(rule.rule_str, error, name, agent_factory)
                    if fixed:
                        # Verify the fixed rule
                        fixed_success, _, _ = CleanRuleSet._compile_rule_safely(fixed)
                        if fixed_success:
                            rule.rule_str = fixed
                            print(f"  ✓ Using fixed rule for '{name}'")
                        else:
                            print(f"  ⚠ Fixed rule still invalid, skipping '{name}'")
                            continue
                    else:
                        continue
                else:
                    print(f"  ⚠ No agent factory, skipping '{name}'")
                    continue

            # Get argument count and build part
            arg_count = CleanRuleSet._get_arg_count(rule.rule_str)
            if arg_count == 1:
                valid_parts.append(f"(lambda v, r: ({rule.rule_str})(v))")
            else:
                valid_parts.append(f"({rule.rule_str})")

        if not valid_parts:
            print(f"  ⚠ No valid clean rules, using default True")
            return "lambda value, row=None: True"

        composed = " and ".join([f"({p})(value, row)" for p in valid_parts])
        return f"lambda value, row=None: {composed}"

    def _compose_or_str_safe(self, rules: Dict[str, CleanRule],
                              agent_factory=None) -> str:
        """Generate OR-composed lambda string with self-healing for syntax errors.

        Args:
            rules: Dict of dirty rules
            agent_factory: Optional factory for LLM-based rule fixing

        Returns:
            Composed lambda string
        """
        if not rules:
            return "lambda value, row=None: False"

        valid_parts = []
        for name, rule in rules.items():
            # Try to compile the rule
            success, _, error = CleanRuleSet._compile_rule_safely(rule.rule_str)

            if not success:
                print(f"  ⚠ Agent '{name}' has syntax error: {error}")
                # Try to fix with LLM
                if agent_factory:
                    fixed = self._fix_rule_with_llm(rule.rule_str, error, name, agent_factory)
                    if fixed:
                        # Verify the fixed rule
                        fixed_success, _, _ = CleanRuleSet._compile_rule_safely(fixed)
                        if fixed_success:
                            rule.rule_str = fixed
                            print(f"  ✓ Using fixed rule for '{name}'")
                        else:
                            print(f"  ⚠ Fixed rule still invalid, skipping '{name}'")
                            continue
                    else:
                        continue
                else:
                    print(f"  ⚠ No agent factory, skipping '{name}'")
                    continue

            # Get argument count and build part
            arg_count = CleanRuleSet._get_arg_count(rule.rule_str)
            if arg_count == 1:
                valid_parts.append(f"(lambda v, r: ({rule.rule_str})(v))")
            else:
                valid_parts.append(f"({rule.rule_str})")

        if not valid_parts:
            print(f"  ⚠ No valid dirty rules, using default False")
            return "lambda value, row=None: False"

        composed = " or ".join([f"({p})(value, row)" for p in valid_parts])
        return f"lambda value, row=None: {composed}"


@dataclass
class ConflictRecord:
    """Record of a single conflict instance."""
    row_index: int
    value: Any
    clean_rule_name: str  # Which clean pillar
    dirty_rule_name: str  # Which dirty agent


@dataclass
class GapRecord:
    """Record of a single gap zone instance."""
    row_index: int
    value: Any
