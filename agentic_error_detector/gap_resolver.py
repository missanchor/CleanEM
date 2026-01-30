"""
Gap Resolver for handling gap zones in clean rule-level refinement.
Extends both clean and dirty rules to cover undefined areas.
"""
from typing import Dict, List, Any
from collections import Counter
import pandas as pd
import json

from dual_types import (
    CleanRule, CleanRuleSet, GapRecord
)
from modification_memory import (
    ModificationMemory, get_logger
)


class GapResolver:
    """
    Resolve gap zones where both P_clean=False and P_dirty=False.
    """

    def __init__(self, memory: ModificationMemory, agent_factory=None,
                 violation_threshold: float = 0.5):
        """
        Args:
            memory: ModificationMemory for tracking changes
            agent_factory: AgentFactory for LLM calls
            violation_threshold: Threshold for rejecting rules that would flag too many violations (default 0.5)
        """
        self.memory = memory
        self.factory = agent_factory
        self.violation_threshold = violation_threshold
        self.logger = get_logger()
        self._dual_agent = None  # Lazy initialization for reuse

    def _get_dual_agent(self):
        """Get or create a reusable DualAgent instance."""
        if self._dual_agent is None:
            if self.factory:
                from agent import DualAgent
                self._dual_agent = DualAgent(
                    base_url=self.factory.base_url,
                    model=self.factory.model
                )
        return self._dual_agent
    
    def resolve(self, df: pd.DataFrame, column: str,
                rule_set: CleanRuleSet,
                metadata: Dict[str, Any] = None,
                round_num: int = 1) -> CleanRuleSet:
        """
        Resolve gap zones for a column.

        Args:
            df: DataFrame to analyze
            column: Column name
            rule_set: Current CleanRuleSet
            metadata: Column metadata (type, sample_values, etc.)
            round_num: Current refinement round number

        Returns:
            Refined CleanRuleSet with extended rules
        """
        self.current_column = column
        self.round_num = round_num

        # Store df for validation in extend methods
        self._current_df = df

        # Store initial rules for logging
        clean_rules_before = {k: v.rule_str for k, v in rule_set.clean_rules.items()}
        dirty_rules_before = {k: v.rule_str for k, v in rule_set.dirty_rules.items()}

        # Begin column refinement logging
        self.logger.begin_column_refinement(
            column=column,
            round_num=round_num,
            stage="gap_resolution",
            clean_rules=clean_rules_before,
            dirty_rules=dirty_rules_before
        )

        # Find gap, clean, and dirty samples
        gap_samples, clean_samples, dirty_samples = self._find_gap_values(df, column, rule_set)

        if not gap_samples:
            # End logging without modifications
            self.logger.end_column_refinement(
                clean_rules=clean_rules_before,
                dirty_rules=dirty_rules_before
            )
            return rule_set

        gap_values = list(set([str(s['value']) for s in gap_samples]))
        clean_values = list(set([str(s['value']) for s in clean_samples[:10]]))
        dirty_values = list(set([str(s['value']) for s in dirty_samples[:10]]))

        # Analyze gaps and classify them with positive and negative examples
        analysis = self._analyze_gap(
            df, column, gap_samples, clean_samples, dirty_samples, rule_set, metadata
        )

        num_modifications = 0

        # === Process clean rules: one at a time with immediate validation ===
        extend_clean = analysis.get('extend_clean', {})
        for rule_name in list(extend_clean.keys()):
            extension_info = extend_clean[rule_name]
            if extension_info and rule_name in rule_set.clean_rules:
                old_rule = rule_set.clean_rules[rule_name]
                new_rule = self._extend_rule_single(
                    rule_set.clean_rules[rule_name],
                    extension_info,
                    'clean',
                    gap_values, clean_values, dirty_values
                )
                if new_rule.rule_str != old_rule.rule_str:
                    num_modifications += 1
                    # Re-evaluate gaps after this extension
                    gap_samples, _, _ = self._find_gap_values(df, column, rule_set)
                    gap_values = list(set([str(s['value']) for s in gap_samples]))
                rule_set.clean_rules[rule_name] = new_rule

        # === Process dirty rules: one at a time with immediate validation ===
        extend_dirty = analysis.get('extend_dirty', {})
        for agent_name in list(extend_dirty.keys()):
            extension_info = extend_dirty[agent_name]
            if extension_info:
                if agent_name not in rule_set.dirty_rules:
                    rule_set.dirty_rules[agent_name] = CleanRule(
                        name=agent_name,
                        rule_str="lambda value, row=None: False",
                        rule_func=lambda value, row=None: False,
                        version=0,
                        modification_log=[]
                    )
                old_rule = rule_set.dirty_rules[agent_name]
                new_rule = self._extend_rule_single(
                    rule_set.dirty_rules[agent_name],
                    extension_info,
                    'dirty',
                    gap_values, clean_values, dirty_values
                )
                if new_rule.rule_str != old_rule.rule_str:
                    num_modifications += 1
                    # Re-evaluate gaps after this extension
                    gap_samples, _, _ = self._find_gap_values(df, column, rule_set)
                    gap_values = list(set([str(s['value']) for s in gap_samples]))
                rule_set.dirty_rules[agent_name] = new_rule

        # Store final rules for logging
        clean_rules_after = {k: v.rule_str for k, v in rule_set.clean_rules.items()}
        dirty_rules_after = {k: v.rule_str for k, v in rule_set.dirty_rules.items()}

        # End column refinement logging
        self.logger.end_column_refinement(
            clean_rules=clean_rules_after,
            dirty_rules=dirty_rules_after
        )

        # Log round summary if modifications were made
        if num_modifications > 0:
            self.logger.log_round_summary(
                column=column,
                round_num=round_num,
                conflict_rate=0.0,
                gap_rate=len(gap_samples) / len(df) if len(df) > 0 else 0,
                num_modifications=num_modifications
            )

        return rule_set

    def _find_gap_values(self, df: pd.DataFrame, column: str,
                        rule_set: CleanRuleSet) -> tuple:
        """Find gap, clean, and dirty samples for better LLM context.

        Returns:
            Tuple of (gap_samples, clean_samples, dirty_samples)
            Each sample is a dict: {'row_index': int, 'value': Any, 'row_data': Dict}
        """
        # Compose P_clean and P_dirty
        P_clean = self._compose_and(rule_set.clean_rules)
        P_dirty = self._compose_or(rule_set.dirty_rules)

        gap_samples = []
        clean_samples = []
        dirty_samples = []

        for idx, row in df.iterrows():
            value = row[column]
            row_dict = row.to_dict()

            try:
                is_clean = P_clean(value, row)
                is_dirty = P_dirty(value, row)

                sample = {
                    'row_index': int(idx),
                    'value': value,
                    'row_data': row_dict
                }

                if not is_clean and not is_dirty:
                    gap_samples.append(sample)
                elif is_clean and not is_dirty:
                    clean_samples.append(sample)
                elif not is_clean and is_dirty:
                    dirty_samples.append(sample)
            except Exception:
                # On exception, treat as gap to be conservative
                gap_samples.append({
                    'row_index': int(idx),
                    'value': value,
                    'row_data': row_dict
                })

        return gap_samples, clean_samples, dirty_samples
    
    def _get_top_gaps(self, gap_samples: List[Dict], k: int = 10) -> List[Dict]:
        """Get top-k most frequent gap samples."""
        value_freq = Counter([s['value'] for s in gap_samples])
        top_values = [v for v, _ in value_freq.most_common(k)]
        return [s for s in gap_samples if s['value'] in top_values]
    
    def _analyze_gap(self, df: pd.DataFrame, column: str,
                    gap_samples: List[Dict],
                    clean_samples: List[Dict],
                    dirty_samples: List[Dict],
                    rule_set: CleanRuleSet,
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze gap values and determine if they should be clean or dirty.
        Uses LLM to make classification with positive and negative examples.
        """
        if not gap_samples:
            return {'extend_clean': {}, 'extend_dirty': {}}

        # Get top gap values for the prompt
        top_gaps = self._get_top_gaps(gap_samples, k=10)
        gap_values = [s['value'] for s in top_gaps]

        # If no factory, return empty extensions
        if not self.factory:
            return {'extend_clean': {}, 'extend_dirty': {}}

        # Compute per-rule statistics on GAP samples (clean rules only)
        gap_rule_stats = self._compute_rule_statistics(top_gaps, rule_set.clean_rules)
        gap_stats_str = self._format_statistics_for_prompt(gap_rule_stats)

        prompt = self._build_gap_analysis_prompt(
            column, gap_values, rule_set, metadata,
            top_gaps, clean_samples[:5], dirty_samples[:5],
            gap_stats_str
        )

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            dual_leg = self._get_dual_agent()
            if not dual_leg:
                raise ValueError("No agent available")
            response = dual_leg._call_llm(prompt, max_tokens=400)

            # Set response for logging
            self.logger.set_response(response)

            # Parse JSON response
            classification = json.loads(response)
            return {
                'culprit_rule': classification.get('culprit_rule', ''),
                'culprit_analysis': classification.get('culprit_analysis', ''),
                'extend_clean': classification.get('extend_clean', {}),
                'extend_dirty': classification.get('extend_dirty', {})
            }
        except Exception as e:
            print(f"  ⚠ Gap analysis failed: {e}")
            return {'extend_clean': {}, 'extend_dirty': {}}

    def _compute_rule_statistics(self, samples: List[Dict[str, Any]],
                                  rules: Dict[str, CleanRule]) -> Dict[str, Dict]:
        """Compute false counts for each rule on the sample set."""
        stats = {}
        for rule_name, rule in rules.items():
            if rule.rule_func is None:
                continue
            false_count = sum(1 for s in samples if not self._invoke_rule(rule.rule_func, s['value'], s.get('row_data', {})))
            stats[rule_name] = {
                'false_count': false_count,
                'total_samples': len(samples)
            }
        return stats

    def _format_statistics_for_prompt(self, stats: Dict[str, Dict]) -> str:
        """Format false counts for each rule for prompt display."""
        if not stats:
            return ""
        lines = []
        for rule_name, stat in sorted(stats.items(), key=lambda x: x[1]['false_count'], reverse=True):
            lines.append(f"- {rule_name}: {stat['false_count']}/{stat['total_samples']} false")
        return "\n".join(lines)

    def _build_gap_analysis_prompt(self, column: str, values: List[Any],
                                   rule_set: CleanRuleSet,
                                   metadata: Dict[str, Any] = None,
                                   gap_samples: List[Dict] = None,
                                   clean_samples: List[Dict] = None,
                                   dirty_samples: List[Dict] = None,
                                   gap_rule_stats: str = "") -> str:
        """Build prompt for LLM gap analysis with positive and negative examples."""
        clean_rules_str = "\n".join([
            f"- {name}: {rule.rule_str}"
            for name, rule in rule_set.clean_rules.items()
        ])

        dirty_rules_str = "\n".join([
            f"- {name}: {rule.rule_str}"
            for name, rule in rule_set.dirty_rules.items()
        ])

        # Format metadata for prompt
        meta_str = ""
        if metadata:
            # Format top values with frequencies
            top_values = metadata.get('top_values', {})
            if top_values:
                # top_values is a dict like {'United': 100, 'American': 95, ...}
                top_items = list(top_values.items())[:10]
                top_str = ", ".join([f"'{v}' ({c}次)" for v, c in top_items])
            else:
                top_str = "N/A"

            meta_str = f"""
**Column Metadata:**
- Type: {metadata.get('type', 'unknown')}
- Top 10 Values (with frequency):
  {top_str}
- Unique Count: {metadata.get('unique_count', 'unknown')}
- Non-Null Count: {metadata.get('non_null_count', 'unknown')}
- Null Count: {metadata.get('null_count', 'unknown')}
"""

        # Format sample sections with negative feedback
        clean_section = ""
        if clean_samples:
            clean_examples = "\n".join([
                f"  - Row {s['row_index']}: value='{s['value']}'"
                for s in clean_samples[:5]
            ])
            clean_section = f"""
**Known Clean Values (correctly classified as clean - for reference):**
These values satisfy clean rules and should NOT be flagged as dirty.
{clean_examples}
"""

        dirty_section = ""
        if dirty_samples:
            dirty_examples = "\n".join([
                f"  - Row {s['row_index']}: value='{s['value']}'"
                for s in dirty_samples[:5]
            ])
            dirty_section = f"""
**Known Dirty Values (correctly detected as dirty - for reference):**
These values are correctly flagged by dirty rules.
{dirty_examples}
"""

        gap_section = ""
        if gap_samples:
            gap_examples = "\n".join([
                f"  - Row {s['row_index']}: value='{s['value']}'"
                for s in gap_samples[:10]
            ])
            gap_section = f"""
**Gap Values (need classification - neither clean nor dirty):**
These values failed all clean rules and all dirty rules. Classify them based on patterns above.
{gap_examples}
"""

        # Format statistics section
        stats_section = ""
        if gap_rule_stats:
            stats_section = f"""
**Per-rule False Counts on GAP Values:**
{gap_rule_stats}

(A rule with HIGH false count indicates it may be too strict or missing valid patterns)
"""

        return f"""Analyze these gap values for column '{column}':
{meta_str}
{clean_section}
{dirty_section}
{gap_section}
{stats_section}
**Current Clean Rules:**
{clean_rules_str}

**Current Dirty Rules:**
{dirty_rules_str}

**Modification History:**
{self.memory.to_context()}

**Task:**
1. **Identify the Culprit Rule (REQUIRED):** First, explicitly identify which SINGLE clean rule is the most likely cause of the gap zone using the per-rule false counts above.
   - Look for clean rules with HIGH false count (rule is too strict)

2. Provide clean rule modification suggestions referencing the culprit rule.

3. **Assume GAP values are erroneous by default.** Propose exactly ONE dirty rule extension that will detect the gap values.
   - If existing dirty rules are insufficient, create a NEW dirty rule and let the LLM name it consistently (global naming strategy).
   - Do NOT split gap values into subsets; use one rule that broadly covers the gap values.

**Important Guidelines for Rule Analysis:**
- A Dirty Rule's purpose is to catch ERRORS.
- A Clean Rule's purpose is to match VALID data.
- **Recall Priority**: If a gap value is ambiguous or differs significantly from the majority of clean patterns, prefer classifying it as **dirty**. It is better to flag a suspicious value for review than to let a potential error slip through.

Return ONLY a JSON object:
{{
  "culprit_rule": "rule_name",
  "culprit_analysis": "explanation of why this rule is the culprit based on false counts",
  "classification": "clean" or "dirty" or "mixed",
  "reasoning": "brief explanation based on comparison",
  "extend_clean": {{"clean_rule_name": "description of pattern to include"}},
  "extend_dirty": {{"dirty_rule_name": "description of pattern to include"}}
}}
"""
    
    def _extend_rule_single(self, rule: CleanRule, extension_info: str,
                            side: str, gap_values: List[str],
                            clean_values: List[str], dirty_values: List[str]) -> CleanRule:
        """Extend a single rule to cover with full context.

        This is the 'single combat' approach - generate one rule at a time
        with full context of all sample values.
        """
        if not self.factory:
            return rule

        column = getattr(self, 'current_column', 'unknown')
        round_num = getattr(self, 'round_num', 1)

        # Format sample values for the prompt
        gap_section = ""
        if gap_values:
            gap_examples = "\n".join([f"  - '{v}'" for v in gap_values[:5]])
            gap_section = f"""
**Gap Values (to be covered by this extension):**
{gap_examples}
"""

        clean_section = ""
        if clean_values:
            clean_examples = "\n".join([f"  - '{v}'" for v in clean_values[:5]])
            clean_section = f"""
**Known Clean Values (should keep matching):**
{clean_examples}
"""

        dirty_section = ""
        if dirty_values:
            dirty_examples = "\n".join([f"  - '{v}'" for v in dirty_values[:5]])
            dirty_section = f"""
**Known Dirty Values (should keep detecting):**
{dirty_examples}
"""

        prompt = f"""Extend this {side} rule to cover with full context:

**Current Rule ({rule.name}):**
{rule.rule_str}

**Pattern to Add:**
{extension_info}
{gap_section}
{clean_section}
{dirty_section}
**Modification History:**
{self.memory.to_context() if hasattr(self.memory, 'to_context') else 'N/A'}

**Task:**
Generate a refined lambda function that:
1. STILL covers all existing clean/dirty values
2. ALSO covers the new gap

Return ONLY the lambda function, no explanation.

Example format:
lambda value, row=None: <expression>
"""

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            dual_leg = self._get_dual_agent()
            if not dual_leg:
                raise ValueError("No agent available")

            response = dual_leg._call_llm(prompt, max_tokens=300, temperature=0.2)

            # Set response for logging
            self.logger.set_response(response)

            new_rule_str = self._extract_lambda(response)
            if new_rule_str:
                # Create new rule object (without compiling yet)
                new_rule = CleanRule(
                    name=rule.name,
                    rule_str=new_rule_str,
                    rule_func=None,
                    version=rule.version + 1,
                    modification_log=rule.modification_log + [f"Extended: {extension_info}"]
                )

                # Get df from outer scope - need to pass it or access via resolve()
                df = getattr(self, '_current_df', None)
                if df is not None:
                    # Validate the extended rule
                    is_accepted, violation_rate, reject_reason = self._validate_refined_rule(
                        df, column, rule, new_rule, side
                    )

                    if not is_accepted:
                        # Reject the rule - log and return original
                        self.logger.log_rejection(
                            rule_type=side,
                            rule_name=rule.name,
                            old_rule=rule.rule_str,
                            new_rule=new_rule_str,
                            violation_rate=violation_rate,
                            reason=reject_reason
                        )

                        if hasattr(self.memory, 'add'):
                            self.memory.add(
                                side,
                                rule.name,
                                'rejected',
                                reject_reason,
                                new_rule_str,
                                metrics={"violation_rate": violation_rate},
                                round_num=getattr(self, 'round_num', None)
                            )

                        print(f"  ⚠ {side} rule '{rule.name}' rejected: violation rate {violation_rate*100:.1f}%")
                        return rule  # Return original rule

                # Rule accepted - compile the rule function for proper execution
                new_rule_func = self._compile_rule(new_rule_str, side)

                # Add modification to logging
                self.logger.add_modification(
                    rule_type=side,
                    rule_name=rule.name,
                    old_rule=rule.rule_str,
                    new_rule=new_rule_str,
                    modification_type="extended_single",
                    reason=extension_info
                )

                if hasattr(self.memory, 'add'):
                    df = getattr(self, '_current_df', None)
                    gap_rate = len(gap_values) / len(df) if df is not None and len(df) > 0 else None
                    metrics = {"gap_rate": gap_rate} if gap_rate is not None else None
                    self.memory.add(
                        side,
                        rule.name,
                        'extended_single',
                        extension_info,
                        new_rule_str,
                        metrics=metrics,
                        round_num=getattr(self, 'round_num', None)
                    )

                return CleanRule(
                    name=rule.name,
                    rule_str=new_rule_str,
                    rule_func=new_rule_func,
                    version=rule.version + 1,
                    modification_log=rule.modification_log + [f"Extended: {extension_info}"]
                )
        except Exception as e:
            self.logger.log_error(column, f"extend_single_{side}", str(e))
            print(f"  ⚠ Failed to extend {side} rule (single): {e}")

        return rule

    def _extend_rule(self, rule: CleanRule, extension_info: str,
                    side: str) -> CleanRule:
        """Extend a rule to cover additional patterns (legacy method for compatibility)."""
        if not self.factory:
            return rule

        column = getattr(self, 'current_column', 'unknown')
        round_num = getattr(self, 'round_num', 1)

        prompt = f"""Extend this {side} rule to cover additional patterns:

**Current Rule ({rule.name}):**
{rule.rule_str}

**Extension Required:**
{extension_info}

**Modification History:**
{self.memory.to_context() if hasattr(self.memory, 'to_context') else 'N/A'}

Generate a refined lambda function that ALSO covers the new values.
Return ONLY the lambda function, no explanation.

Example format:
lambda value, row=None: <expression>
"""

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            dual_leg = self._get_dual_agent()
            if not dual_leg:
                raise ValueError("No agent available")

            response = dual_leg._call_llm(prompt, max_tokens=300, temperature=0.2)

            # Set response for logging
            self.logger.set_response(response)

            new_rule_str = self._extract_lambda(response)
            if new_rule_str:
                # Create new rule object (without compiling yet)
                new_rule = CleanRule(
                    name=rule.name,
                    rule_str=new_rule_str,
                    rule_func=None,
                    version=rule.version + 1,
                    modification_log=rule.modification_log + [f"Extended: {extension_info}"]
                )

                # Get df from outer scope - need to pass it or access via resolve()
                df = getattr(self, '_current_df', None)
                if df is not None:
                    # Validate the extended rule
                    is_accepted, violation_rate, reject_reason = self._validate_refined_rule(
                        df, column, rule, new_rule, side
                    )

                    if not is_accepted:
                        # Reject the rule - log and return original
                        self.logger.log_rejection(
                            rule_type=side,
                            rule_name=rule.name,
                            old_rule=rule.rule_str,
                            new_rule=new_rule_str,
                            violation_rate=violation_rate,
                            reason=reject_reason
                        )

                        if hasattr(self.memory, 'add'):
                            self.memory.add(
                                side,
                                rule.name,
                                'rejected',
                                reject_reason,
                                new_rule_str,
                                metrics={"violation_rate": violation_rate},
                                round_num=getattr(self, 'round_num', None)
                            )

                        print(f"  ⚠ {side} rule '{rule.name}' rejected: violation rate {violation_rate*100:.1f}%")
                        return rule  # Return original rule

                # Rule accepted - compile the rule function for proper execution
                new_rule_func = self._compile_rule(new_rule_str, side)

                # Add modification to logging
                self.logger.add_modification(
                    rule_type=side,
                    rule_name=rule.name,
                    old_rule=rule.rule_str,
                    new_rule=new_rule_str,
                    modification_type="extended",
                    reason=extension_info
                )

                if hasattr(self.memory, 'add'):
                    self.memory.add(
                        side,
                        rule.name,
                        'extended',
                        extension_info,
                        new_rule_str,
                        round_num=getattr(self, 'round_num', None)
                    )

                return CleanRule(
                    name=rule.name,
                    rule_str=new_rule_str,
                    rule_func=new_rule_func,
                    version=rule.version + 1,
                    modification_log=rule.modification_log + [f"Extended: {extension_info}"]
                )
        except Exception as e:
            self.logger.log_error(column, f"extend_{side}", str(e))
            print(f"  ⚠ Failed to extend {side} rule: {e}")

        return rule
    
    def _extract_lambda(self, response: str) -> str:
        """Extract lambda expression from LLM response."""
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith('lambda'):
                return line
        return None

    def _compile_rule(self, rule_str: str, side: str = 'clean') -> callable:
        """Compile a rule string to a callable function.

        Uses the same safe_dict as dual_types.py to ensure consistency.
        """
        import re
        import pandas as pd
        import numpy as np

        def safe_float(value):
            try:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                cleaned = str(value).replace(',', '').strip()
                if not cleaned:
                    return None
                return float(cleaned)
            except Exception:
                return None

        safe_dict = {
            "re": re,
            "str": str,
            "bool": bool,
            "pd": pd,
            "np": np,
            "float": float,
            "int": int,
            "len": len,
            "safe_float": safe_float,
        }

        try:
            return eval(rule_str, safe_dict)
        except Exception as e:
            self.logger.log_error(getattr(self, 'current_column', 'unknown'), f"compile_{side}_rule", str(e))
            print(f"  ⚠ Failed to compile {side} rule: {e}")
            return None

    def _compose_and(self, rules: Dict[str, CleanRule]):
        """Compose AND of all clean rules."""
        def P_clean(value, row=None):
            for rule in rules.values():
                if rule.rule_func is None:
                    continue
                try:
                    if not self._invoke_rule(rule.rule_func, value, row):
                        return False
                except Exception:
                    return False
            return True
        return P_clean

    def _compose_or(self, rules: Dict[str, CleanRule]):
        """Compose OR of all dirty rules."""
        def P_dirty(value, row=None):
            for rule in rules.values():
                if rule.rule_func is None:
                    continue
                try:
                    if self._invoke_rule(rule.rule_func, value, row):
                        return True
                except Exception:
                    pass
            return False
        return P_dirty
    
    def _invoke_rule(self, func, value, row):
        """Invoke a rule function safely with proper argument handling.

        Matches the logic in judge.py's _invoke_predicate to ensure consistency.
        """
        if func is None:
            return False

        try:
            code = func.__code__
            argcount = code.co_argcount
            argnames = code.co_varnames[:argcount]

            if argcount == 0:
                return func()

            if argcount == 1:
                target = argnames[0] or ""
                if target.lower() in ("row", "record", "data", "context"):
                    return func(row)
                return func(value)

            return func(value, row)
        except TypeError:
            try:
                return func(value)
            except TypeError:
                return func(row)
        except Exception:
            return False

    def _validate_refined_rule(self, df: pd.DataFrame, column: str,
                               old_rule: 'CleanRule', new_rule: 'CleanRule',
                               side: str, violation_threshold: float = None) -> tuple:
        """
        Validate a refined rule against the full dataset.

        Args:
            df: Full DataFrame to validate against
            column: Column name
            old_rule: Original rule before refinement
            new_rule: Proposed new rule after refinement
            side: 'clean' or 'dirty'

        Returns:
            (is_accepted: bool, violation_rate: float, reason: str)
        """
        if new_rule.rule_str is None:
            return True, 0.0, "accepted"

        # Compile the new rule function if needed
        new_func = new_rule.rule_func
        if new_func is None and new_rule.rule_str:
            try:
                new_func = self._compile_rule(new_rule.rule_str, side)
            except Exception as e:
                return True, 0.0, f"compile_error: {e}"

        if new_func is None:
            return True, 0.0, "accepted"

        # Calculate violation rate
        violation_count = 0
        total_count = 0

        for idx, row in df.iterrows():
            value = row[column]
            total_count += 1
            try:
                if side == 'clean':
                    # Clean rule: violation = rule returns False
                    if not self._invoke_rule(new_func, value, row):
                        violation_count += 1
                else:  # dirty
                    # Dirty rule: violation = rule returns True
                    if self._invoke_rule(new_func, value, row):
                        violation_count += 1
            except Exception:
                # On exception, count as violation for safety
                violation_count += 1

        violation_rate = violation_count / total_count if total_count > 0 else 0
        threshold = violation_threshold if violation_threshold is not None else self.violation_threshold

        # Threshold check: > threshold is too strict
        if violation_rate > threshold:
            reason = f"Would flag {violation_rate*100:.1f}% as violation (>{threshold*100:.0f}% threshold). Original rule flagged {self._get_original_violation_rate(df, column, old_rule, side)*100:.1f}%"
            return False, violation_rate, reason

        return True, violation_rate, "accepted"

    def _get_original_violation_rate(self, df: pd.DataFrame, column: str,
                                     rule: 'CleanRule', side: str) -> float:
        """Get the original rule's violation rate for comparison."""
        if rule.rule_func is None:
            return 0.0

        violation_count = 0
        total_count = 0

        for idx, row in df.iterrows():
            value = row[column]
            total_count += 1
            try:
                if side == 'clean':
                    if not self._invoke_rule(rule.rule_func, value, row):
                        violation_count += 1
                else:
                    if self._invoke_rule(rule.rule_func, value, row):
                        violation_count += 1
            except Exception:
                pass

        return violation_count / total_count if total_count > 0 else 0
