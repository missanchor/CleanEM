"""
Gap Resolver for handling gap zones in pillar-level refinement.
Extends both clean and dirty rules to cover undefined areas.
"""
from typing import Dict, List, Any
from collections import Counter
import pandas as pd
import json

from agentic_error_detector.dual_types import (
    PillarRule, PillarRuleSet, GapRecord
)
from agentic_error_detector.modification_memory import (
    ModificationMemory, get_logger
)


class GapResolver:
    """
    Resolve gap zones where both P_clean=False and P_dirty=False.
    """

    def __init__(self, memory: ModificationMemory, legislator_factory=None):
        """
        Args:
            memory: ModificationMemory for tracking changes
            legislator_factory: LegislatorFactory for LLM calls
        """
        self.memory = memory
        self.factory = legislator_factory
        self.logger = get_logger()
        self._dual_legislator = None  # Lazy initialization for reuse

    def _get_dual_legislator(self):
        """Get or create a reusable DualLegislator instance."""
        if self._dual_legislator is None:
            if self.factory:
                from agentic_error_detector.legislator import DualLegislator
                self._dual_legislator = DualLegislator(
                    base_url=self.factory.base_url,
                    model=self.factory.model
                )
        return self._dual_legislator
    
    def resolve(self, df: pd.DataFrame, column: str,
                pillar_set: PillarRuleSet,
                metadata: Dict[str, Any] = None,
                round_num: int = 1) -> PillarRuleSet:
        """
        Resolve gap zones for a column.

        Args:
            df: DataFrame to analyze
            column: Column name
            pillar_set: Current PillarRuleSet
            metadata: Column metadata (type, sample_values, etc.)
            round_num: Current refinement round number

        Returns:
            Refined PillarRuleSet with extended rules
        """
        self.current_column = column
        self.round_num = round_num

        # Store df for validation in extend methods
        self._current_df = df

        # Store initial rules for logging
        clean_rules_before = {k: v.rule_str for k, v in pillar_set.clean_pillars.items()}
        dirty_rules_before = {k: v.rule_str for k, v in pillar_set.dirty_agents.items()}

        # Begin column refinement logging
        self.logger.begin_column_refinement(
            column=column,
            round_num=round_num,
            stage="gap_resolution",
            clean_rules=clean_rules_before,
            dirty_rules=dirty_rules_before
        )

        # Find gap, clean, and dirty samples
        gap_samples, clean_samples, dirty_samples = self._find_gap_values(df, column, pillar_set)

        if not gap_samples:
            # End logging without modifications
            self.logger.end_column_refinement(
                clean_rules=clean_rules_before,
                dirty_rules=dirty_rules_before
            )
            return pillar_set

        gap_values = list(set([str(s['value']) for s in gap_samples]))
        clean_values = list(set([str(s['value']) for s in clean_samples[:10]]))
        dirty_values = list(set([str(s['value']) for s in dirty_samples[:10]]))

        # Analyze gaps and classify them with positive and negative examples
        analysis = self._analyze_gap(
            df, column, gap_samples, clean_samples, dirty_samples, pillar_set, metadata
        )

        num_modifications = 0

        # === Process clean rules: one at a time with immediate validation ===
        extend_clean = analysis.get('extend_clean', {})
        for pillar_name in list(extend_clean.keys()):
            extension_info = extend_clean[pillar_name]
            if extension_info and pillar_name in pillar_set.clean_pillars:
                old_rule = pillar_set.clean_pillars[pillar_name]
                new_rule = self._extend_rule_single(
                    pillar_set.clean_pillars[pillar_name],
                    extension_info,
                    'clean',
                    gap_values, clean_values, dirty_values
                )
                if new_rule.rule_str != old_rule.rule_str:
                    num_modifications += 1
                    # Re-evaluate gaps after this extension
                    gap_samples, _, _ = self._find_gap_values(df, column, pillar_set)
                    gap_values = list(set([str(s['value']) for s in gap_samples]))
                pillar_set.clean_pillars[pillar_name] = new_rule

        # === Process dirty rules: one at a time with immediate validation ===
        extend_dirty = analysis.get('extend_dirty', {})
        for agent_name in list(extend_dirty.keys()):
            extension_info = extend_dirty[agent_name]
            if extension_info and agent_name in pillar_set.dirty_agents:
                old_rule = pillar_set.dirty_agents[agent_name]
                new_rule = self._extend_rule_single(
                    pillar_set.dirty_agents[agent_name],
                    extension_info,
                    'dirty',
                    gap_values, clean_values, dirty_values
                )
                if new_rule.rule_str != old_rule.rule_str:
                    num_modifications += 1
                    # Re-evaluate gaps after this extension
                    gap_samples, _, _ = self._find_gap_values(df, column, pillar_set)
                    gap_values = list(set([str(s['value']) for s in gap_samples]))
                pillar_set.dirty_agents[agent_name] = new_rule

        # Store final rules for logging
        clean_rules_after = {k: v.rule_str for k, v in pillar_set.clean_pillars.items()}
        dirty_rules_after = {k: v.rule_str for k, v in pillar_set.dirty_agents.items()}

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

        return pillar_set
    
    def _find_gap_values(self, df: pd.DataFrame, column: str,
                        pillar_set: PillarRuleSet) -> tuple:
        """Find gap, clean, and dirty samples for better LLM context.

        Returns:
            Tuple of (gap_samples, clean_samples, dirty_samples)
            Each sample is a dict: {'row_index': int, 'value': Any, 'row_data': Dict}
        """
        # Compose P_clean and P_dirty
        P_clean = self._compose_and(pillar_set.clean_pillars)
        P_dirty = self._compose_or(pillar_set.dirty_agents)

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
                    pillar_set: PillarRuleSet,
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

        prompt = self._build_gap_analysis_prompt(
            column, gap_values, pillar_set, metadata,
            top_gaps, clean_samples[:5], dirty_samples[:5]
        )

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            dual_leg = self._get_dual_legislator()
            if not dual_leg:
                raise ValueError("No legislator available")
            response = dual_leg._call_llm(prompt, max_tokens=400)

            # Set response for logging
            self.logger.set_response(response)

            # Parse JSON response
            classification = json.loads(response)
            return {
                'extend_clean': classification.get('extend_clean', {}),
                'extend_dirty': classification.get('extend_dirty', {})
            }
        except Exception as e:
            print(f"  ⚠ Gap analysis failed: {e}")
            return {'extend_clean': {}, 'extend_dirty': {}}
    
    def _build_gap_analysis_prompt(self, column: str, values: List[Any],
                                   pillar_set: PillarRuleSet,
                                   metadata: Dict[str, Any] = None,
                                   gap_samples: List[Dict] = None,
                                   clean_samples: List[Dict] = None,
                                   dirty_samples: List[Dict] = None) -> str:
        """Build prompt for LLM gap analysis with positive and negative examples."""
        clean_rules_str = "\n".join([
            f"- {name}: {rule.rule_str}"
            for name, rule in pillar_set.clean_pillars.items()
        ])

        dirty_rules_str = "\n".join([
            f"- {name}: {rule.rule_str}"
            for name, rule in pillar_set.dirty_agents.items()
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

        return f"""Analyze these gap values for column '{column}':
{meta_str}
{clean_section}
{dirty_section}
{gap_section}
**Current Clean Rules:**
{clean_rules_str}

**Current Dirty Rules:**
{dirty_rules_str}

**Modification History:**
{self.memory.to_context()}

**Task:**
1. Compare gap values with known clean and dirty values.
2. Determine if gap values should be classified as clean or dirty.
3. Explain which rules need extension to cover the gap values.

**Important Guidelines for Rule Analysis:**
- A Dirty Rule's purpose is to catch ERRORS.
- A Clean Rule's purpose is to match VALID data.

Return ONLY a JSON object:
{{
  "classification": "clean" or "dirty" or "mixed",
  "reasoning": "brief explanation based on pattern comparison, explicitly mentioning separator precision if applicable",
  "extend_clean": {{"pillar_name": "description of pattern to include"}},
  "extend_dirty": {{"agent_name": "description of pattern to include"}}
}}
"""
    
    def _extend_rule_single(self, rule: PillarRule, extension_info: str,
                            side: str, gap_values: List[str],
                            clean_values: List[str], dirty_values: List[str]) -> PillarRule:
        """Extend a single rule to cover ONE specific pattern with full context.

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

        prompt = f"""Extend this {side} rule to cover ONE specific pattern with full context:

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
2. ALSO covers the new gap pattern

Return ONLY the lambda function, no explanation.

Example format:
lambda value, row=None: <expression>
"""

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            dual_leg = self._get_dual_legislator()
            if not dual_leg:
                raise ValueError("No legislator available")

            response = dual_leg._call_llm(prompt, max_tokens=300, temperature=0.2)

            # Set response for logging
            self.logger.set_response(response)

            new_rule_str = self._extract_lambda(response)
            if new_rule_str:
                # Create new rule object (without compiling yet)
                new_rule = PillarRule(
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
                            self.memory.add(side, rule.name, 'rejected',
                                          reject_reason, new_rule_str)

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
                    self.memory.add(side, rule.name, 'extended_single', extension_info, new_rule_str)

                return PillarRule(
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

    def _extend_rule(self, rule: PillarRule, extension_info: str,
                    side: str) -> PillarRule:
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

Generate a refined lambda function that ALSO covers the new pattern.
Return ONLY the lambda function, no explanation.

Example format:
lambda value, row=None: <expression>
"""

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            dual_leg = self._get_dual_legislator()
            if not dual_leg:
                raise ValueError("No legislator available")

            response = dual_leg._call_llm(prompt, max_tokens=300, temperature=0.2)

            # Set response for logging
            self.logger.set_response(response)

            new_rule_str = self._extract_lambda(response)
            if new_rule_str:
                # Create new rule object (without compiling yet)
                new_rule = PillarRule(
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
                            self.memory.add(side, rule.name, 'rejected',
                                          reject_reason, new_rule_str)

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
                    self.memory.add(side, rule.name, 'extended', extension_info, new_rule_str)

                return PillarRule(
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

    def _compose_and(self, pillars: Dict[str, PillarRule]):
        """Compose AND of all pillar rules."""
        def P_clean(value, row=None):
            for rule in pillars.values():
                if rule.rule_func is None:
                    continue
                try:
                    if not self._invoke_rule(rule.rule_func, value, row):
                        return False
                except Exception:
                    return False
            return True
        return P_clean
    
    def _compose_or(self, agents: Dict[str, PillarRule]):
        """Compose OR of all agent rules."""
        def P_dirty(value, row=None):
            for rule in agents.values():
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
                               old_rule: 'PillarRule', new_rule: 'PillarRule',
                               side: str) -> tuple:
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

        # Threshold check: > 50% is too strict
        if violation_rate > 0.5:
            reason = f"Would flag {violation_rate*100:.1f}% as violation (>50% threshold). Original rule flagged {self._get_original_violation_rate(df, column, old_rule, side)*100:.1f}%"
            return False, violation_rate, reason

        return True, violation_rate, "accepted"

    def _get_original_violation_rate(self, df: pd.DataFrame, column: str,
                                     rule: 'PillarRule', side: str) -> float:
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
