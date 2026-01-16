"""
Conflict Resolver for pillar-level rule refinement.
Handles conflicts between clean pillars and dirty agents.
"""
from typing import Dict, List, Any, Tuple
from collections import Counter
import pandas as pd
import json

from agentic_error_detector.dual_types import (
    PillarRule, PillarRuleSet, ConflictRecord
)
from agentic_error_detector.modification_memory import (
    ModificationMemory, get_logger
)


class ConflictResolver:
    """
    Resolve conflicts between clean pillars and dirty agents.
    """

    def __init__(self, memory: ModificationMemory, legislator_factory=None,
                 max_pairwise_conflicts: int = 5):
        """
        Args:
            memory: ModificationMemory for tracking changes
            legislator_factory: LegislatorFactory for LLM calls
            max_pairwise_conflicts: Maximum number of conflict pairs to process per pillar
        """
        self.memory = memory
        self.factory = legislator_factory
        self.max_pairwise_conflicts = max_pairwise_conflicts
        self.logger = get_logger()

    def resolve(self, df: pd.DataFrame, column: str,
                pillar_set: PillarRuleSet,
                metadata: Dict[str, Any] = None,
                round_num: int = 1) -> PillarRuleSet:
        """
        Resolve conflicts for a column.

        Args:
            df: DataFrame to analyze
            column: Column name
            pillar_set: Current PillarRuleSet
            metadata: Column metadata (type, sample_values, etc.)
            round_num: Current refinement round number

        Returns:
            Refined PillarRuleSet
        """
        # Store initial rules for logging
        clean_rules_before = {k: v.rule_str for k, v in pillar_set.clean_pillars.items()}
        dirty_rules_before = {k: v.rule_str for k, v in pillar_set.dirty_agents.items()}

        # Set current column and round for logging
        self.current_column = column
        self.round_num = round_num

        # Store df for validation in refine methods
        self._current_df = df

        # Begin column refinement logging
        self.logger.begin_column_refinement(
            column=column,
            round_num=round_num,
            stage="conflict_resolution",
            clean_rules=clean_rules_before,
            dirty_rules=dirty_rules_before
        )

        num_modifications = 0

        # For each clean pillar, find conflicts with dirty agents
        for pillar_name, clean_rule in pillar_set.clean_pillars.items():
            conflicts, clean_only, dirty_only = self._find_conflicts(
                df, column, clean_rule, pillar_set.dirty_agents
            )

            if not conflicts:
                continue

            # Get unique conflicting agents for this pillar
            conflicting_agents = list(set([c.dirty_rule_name for c in conflicts]))

            if not conflicting_agents:
                continue

            # Get top conflicting values
            top_conflicts = self._get_top_conflicts(conflicts, k=10)
            conflict_values = [str(c.value) for c in top_conflicts]
            clean_only_values = [str(s['value']) for s in clean_only[:5]]
            dirty_only_values = [str(s['value']) for s in dirty_only[:5]]

            # === Phase 1: Pairwise Analysis ===
            pairwise_results = []
            max_pairs = self.max_pairwise_conflicts

            for agent_name in conflicting_agents[:max_pairs]:
                if agent_name not in pillar_set.dirty_agents:
                    continue

                dirty_rule = pillar_set.dirty_agents[agent_name]

                # Get pairwise conflicting values for this specific pair
                pairwise_conflicts = [c for c in top_conflicts if c.dirty_rule_name == agent_name]
                pairwise_conflict_values = list(set([str(c.value) for c in pairwise_conflicts]))

                # Analyze pairwise conflict
                result = self._analyze_pairwise_conflict(
                    column, clean_rule, dirty_rule, pairwise_conflict_values,
                    clean_only_values, dirty_only_values, metadata
                )

                pairwise_results.append({
                    'clean_rule': pillar_name,
                    'dirty_rule': agent_name,
                    **result
                })

            # === Phase 2: Synthesis Decision ===
            if not pairwise_results:
                continue

            synthesis = self._synthesize_decision(
                column, clean_rule, pairwise_results, metadata
            )

            final_decision = synthesis.get('final_decision', 'clean')
            synthesis_reason = synthesis.get('synthesis_reason', '')

            # === Phase 3: Execute Modification ===
            if final_decision == 'clean':
                # Refine clean rule using the synthesis reason
                analysis = {
                    'reason': synthesis_reason,
                    'llm_analysis': synthesis_reason
                }

                new_clean_rule = self._refine_clean_rule(
                    pillar_name, clean_rule, analysis, metadata, dirty_only_values, conflict_values
                )
                if new_clean_rule.rule_str != clean_rule.rule_str:
                    num_modifications += 1
                    # Record pairwise modification
                    if hasattr(self.memory, 'add'):
                        self.memory.add('clean', pillar_name, 'pairwise_refined',
                                      synthesis_reason, new_clean_rule.rule_str)
                pillar_set.clean_pillars[pillar_name] = new_clean_rule
            else:
                # Modify dirty rules (only those suggested by synthesis)
                affected_rules = synthesis.get('affected_rules', [])

                # If no specific rules mentioned, use all that suggested 'dirty'
                if not affected_rules:
                    affected_rules = [r.get('dirty_rule', '') for r in pairwise_results
                                    if r.get('suggested_modify') == 'dirty']

                for agent_name in affected_rules:
                    if agent_name not in pillar_set.dirty_agents:
                        continue

                    # Find the pairwise result for this agent
                    pair_result = next(
                        (r for r in pairwise_results if r.get('dirty_rule') == agent_name),
                        None
                    )

                    if not pair_result:
                        continue

                    dirty_rule = pillar_set.dirty_agents[agent_name]
                    reason = pair_result.get('reason', synthesis_reason)

                    # Collect conflicting values for this specific agent
                    agent_conflicts = [c for c in top_conflicts if c.dirty_rule_name == agent_name]
                    agent_conflict_values = list(set([str(c.value) for c in agent_conflicts]))

                    new_dirty_rule = self._refine_dirty_rule(
                        agent_name, dirty_rule, [reason], metadata,
                        dirty_only_values, agent_conflict_values
                    )

                    if new_dirty_rule.rule_str != dirty_rule.rule_str:
                        num_modifications += 1
                        # Record pairwise modification
                        if hasattr(self.memory, 'add'):
                            self.memory.add('dirty', agent_name, 'pairwise_refined',
                                          reason, new_dirty_rule.rule_str)

                    pillar_set.dirty_agents[agent_name] = new_dirty_rule

        # Store final rules for logging
        clean_rules_after = {k: v.rule_str for k, v in pillar_set.clean_pillars.items()}
        dirty_rules_after = {k: v.rule_str for k, v in pillar_set.dirty_agents.items()}

        # End column refinement logging
        self.logger.end_column_refinement(
            clean_rules=clean_rules_after,
            dirty_rules=dirty_rules_after
        )

        # Log round summary
        if num_modifications > 0:
            # Calculate conflict rate for summary
            conflicts, _, _ = self._find_conflicts(
                df, column,
                list(pillar_set.clean_pillars.values())[0] if pillar_set.clean_pillars else None,
                pillar_set.dirty_agents
            )
            conflict_rate = len(conflicts) / len(df) if len(df) > 0 else 0

            self.logger.log_round_summary(
                column=column,
                round_num=round_num,
                conflict_rate=conflict_rate,
                gap_rate=0.0,
                num_modifications=num_modifications
            )

        return pillar_set
    
    def _find_conflicts(self, df: pd.DataFrame, column: str,
                       clean_rule: PillarRule,
                       dirty_agents: Dict[str, PillarRule]) -> Tuple[List[ConflictRecord], List[Dict], List[Dict]]:
        """Find conflicts, clean-only samples, and dirty-only samples.

        Returns:
            Tuple of (conflicts, clean_only_samples, dirty_only_samples)
            Each sample is a dict: {'row_index': int, 'value': Any}
        """
        conflicts = []
        clean_only = []   # Values where clean=True and dirty=False
        dirty_only = []   # Values where clean=False and dirty=True

        for idx, row in df.iterrows():
            value = row[column]
            sample = {'row_index': int(idx), 'value': value}

            try:
                clean_result = self._invoke_rule(clean_rule.rule_func, value, row)

                # Check which dirty rules pass
                dirty_results = {}
                for agent_name, dirty_rule in dirty_agents.items():
                    if self._invoke_rule(dirty_rule.rule_func, value, row):
                        dirty_results[agent_name] = True

                # Classify the sample
                if clean_result and dirty_results:
                    # Conflict: clean and at least one dirty rule pass
                    for agent_name in dirty_results.keys():
                        conflicts.append(ConflictRecord(
                            row_index=idx,
                            value=value,
                            clean_rule_name=clean_rule.name,
                            dirty_rule_name=agent_name
                        ))
                elif clean_result and not dirty_results:
                    # Clean only
                    clean_only.append(sample)
                elif not clean_result and dirty_results:
                    # Dirty only
                    dirty_only.append(sample)
            except Exception:
                pass

        return conflicts, clean_only, dirty_only
    
    def _get_top_conflicts(self, conflicts: List[ConflictRecord], 
                          k: int = 10) -> List[ConflictRecord]:
        """Get top-k most frequent conflict values."""
        value_freq = Counter([c.value for c in conflicts])
        top_values = [v for v, _ in value_freq.most_common(k)]
        return [c for c in conflicts if c.value in top_values]
    
    def _analyze_conflict(self, df: pd.DataFrame, column: str,
                         top_conflicts: List[ConflictRecord],
                         clean_rule: PillarRule,
                         dirty_agents: Dict[str, PillarRule],
                         metadata: Dict[str, Any] = None,
                         clean_only_samples: List[Dict] = None,
                         dirty_only_samples: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze conflicts and decide which rule to modify.

        Uses LLM to make the decision based on conflict samples with context.
        """
        # Extract unique values and conflicting agents
        conflict_values = list(set([str(c.value) for c in top_conflicts]))
        conflicting_agents = list(set([c.dirty_rule_name for c in top_conflicts]))

        clean_only_values = [str(s['value']) for s in (clean_only_samples or [])]
        dirty_only_values = [str(s['value']) for s in (dirty_only_samples or [])]

        # If no factory, default to modifying clean (conservative)
        if not self.factory:
            return {
                'modify_clean': True,
                'reason': f'Values {conflict_values[:5]} conflict with dirty agents',
                'conflicting_agents': conflicting_agents,
                'llm_analysis': 'No LLM - default decision',
                'decision': 'clean'
            }

        # Use LLM to decide with sample context
        prompt = self._build_conflict_analysis_prompt(
            column, conflict_values, clean_rule, dirty_agents, conflicting_agents,
            metadata, clean_only_samples, dirty_only_samples
        )

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            from agentic_error_detector.legislator import DualLegislator
            dual_leg = DualLegislator(
                self.factory.base_url,
                self.factory.model
            )

            response = dual_leg._call_llm(prompt, max_tokens=300)

            # Set response for logging
            self.logger.set_response(response)

            # Parse JSON response
            decision = json.loads(response)
            modify_clean = decision.get('modify') == 'clean'
            result = {
                'modify_clean': modify_clean,
                'reason': decision.get('reason', 'No reason provided'),
                'conflicting_agents': conflicting_agents,
                'llm_analysis': decision.get('reason', ''),
                'decision': 'clean' if modify_clean else 'dirty'
            }

            return result
        except Exception as e:
            # Log error
            self.logger.log_error(column, "conflict_analysis", str(e))

            # Fallback: modify clean
            return {
                'modify_clean': True,
                'reason': f'LLM analysis failed: {e}',
                'conflicting_agents': conflicting_agents,
                'llm_analysis': f'Error: {e}',
                'decision': 'clean'
            }
    
    def _build_conflict_analysis_prompt(self, column: str, values: List[Any],
                                       clean_rule: PillarRule,
                                       dirty_agents: Dict[str, PillarRule],
                                       conflicting_agents: List[str],
                                       metadata: Dict[str, Any] = None,
                                       clean_only_samples: List[Dict] = None,
                                       dirty_only_samples: List[Dict] = None) -> str:
        """Build prompt for LLM conflict analysis with sample context."""
        dirty_rules_str = "\n".join([
            f"- {name}: {dirty_agents[name].rule_str}"
            for name in conflicting_agents
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
        if clean_only_samples:
            clean_examples = "\n".join([
                f"  - '{s['value']}'"
                for s in clean_only_samples[:5]
            ])
            clean_section = f"""
**Clean-Only Values (correctly clean - should NOT be flagged as dirty):**
These values satisfy the clean rule and should NOT be detected by dirty rules.
{clean_examples}
"""

        dirty_section = ""
        if dirty_only_samples:
            dirty_examples = "\n".join([
                f"  - '{s['value']}'"
                for s in dirty_only_samples[:5]
            ])
            dirty_section = f"""
**Dirty-Only Values (correctly detected as dirty):**
These values are correctly flagged by dirty rules.
{dirty_examples}
"""

        conflict_examples = "\n".join([
            f"  - '{v}'"
            for v in values[:10]
        ])
        conflict_section = f"""
**Conflicting Values (BOTH clean AND dirty - need to resolve):**
These values are incorrectly flagged by both rules.
{conflict_examples}
"""

        return f"""Analyze this data quality rule conflict for column '{column}':
{meta_str}
{clean_section}
{dirty_section}
{conflict_section}
**Clean Rule ({clean_rule.name}):**
{clean_rule.rule_str}

**Conflicting Dirty Rules:**
{dirty_rules_str}

**Modification History:**
{self.memory.to_context()}

**Task:**
1. Compare conflicting values with clean-only and dirty-only samples.
2. Determine whether the clean rule is too broad or the dirty rule is too aggressive.
3. Decide which rule should be modified.

**Important Guidelines for Rule Analysis:**
- A Dirty Rule's purpose is to catch ERRORS. If it matches valid data, it is too aggressive.
- A Clean Rule's purpose is to match VALID data. If it matches errors, it is too broad.

Return ONLY a JSON object:
{{
  "Analysis": "Brief explanation referencing the sample patterns, explicitly mentioning separator precision if applicable",
  "modify": "clean" or "dirty"
}}
"""

    def _build_pairwise_conflict_prompt(self, column: str,
                                        clean_rule: PillarRule,
                                        dirty_rule: PillarRule,
                                        conflict_values: List[str],
                                        clean_only_values: List[str],
                                        dirty_only_values: List[str],
                                        metadata: Dict[str, Any] = None) -> str:
        """Build prompt for pairwise conflict analysis (single clean-dirty pair)."""
        # Format metadata for prompt
        meta_str = ""
        if metadata:
            top_values = metadata.get('top_values', {})
            if top_values:
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

        # Format clean-only values
        clean_section = ""
        if clean_only_values:
            clean_examples = "\n".join([f"  - '{v}'" for v in clean_only_values[:5]])
            clean_section = f"""
**Clean-Only Values (correctly clean - should NOT be flagged as dirty):**
These values satisfy the clean rule and should NOT be detected by this dirty rule.
{clean_examples}
"""

        # Format dirty-only values
        dirty_section = ""
        if dirty_only_values:
            dirty_examples = "\n".join([f"  - '{v}'" for v in dirty_only_values[:5]])
            dirty_section = f"""
**Dirty-Only Values (correctly detected as dirty):**
These values are correctly flagged by this dirty rule.
{dirty_examples}
"""

        # Format conflicting values
        conflict_examples = "\n".join([f"  - '{v}'" for v in conflict_values[:10]])
        conflict_section = f"""
**Conflicting Values (BOTH clean AND dirty - need to resolve):**
These values are incorrectly flagged by both rules.
{conflict_examples}
"""

        return f"""Analyze this SINGLE data quality rule conflict for column '{column}':

You are comparing ONLY these two rules - do NOT consider other rules.

**Clean Rule ({clean_rule.name}):**
{clean_rule.rule_str}

**Dirty Rule ({dirty_rule.name}):**
{dirty_rule.rule_str}
{meta_str}
{clean_section}
{dirty_section}
{conflict_section}
**Modification History:**
{self.memory.to_context()}

**Task:**
Compare the conflicting values with clean-only and dirty-only samples.
Determine whether the clean rule is too broad or this specific dirty rule is too aggressive.

**Guidelines:**
- A Dirty Rule's purpose is to catch ERRORS. If it matches valid data (clean-only values), it is too aggressive.
- A Clean Rule's purpose is to match VALID data. If it matches errors (dirty-only values), it is too broad.

Return ONLY a JSON object:
{{
  "pair_analysis": "Brief explanation of why this specific rule pair conflicts",
  "suggested_modify": "clean" or "dirty",
  "reason": "Why this modification is needed based on the samples"
}}
"""

    def _analyze_pairwise_conflict(self, column: str,
                                   clean_rule: PillarRule,
                                   dirty_rule: PillarRule,
                                   conflict_values: List[str],
                                   clean_only_values: List[str],
                                   dirty_only_values: List[str],
                                   metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a single pairwise conflict and return local decision.

        Returns:
            Dict with 'suggested_modify', 'reason', 'pair_analysis'
        """
        # If no factory, default to modifying clean (conservative)
        if not self.factory:
            return {
                'suggested_modify': 'clean',
                'reason': f'Default decision: values {conflict_values[:3]} conflict',
                'pair_analysis': 'No LLM available'
            }

        prompt = self._build_pairwise_conflict_prompt(
            column, clean_rule, dirty_rule, conflict_values,
            clean_only_values, dirty_only_values, metadata
        )

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            from agentic_error_detector.legislator import DualLegislator
            dual_leg = DualLegislator(
                self.factory.base_url,
                self.factory.model
            )

            response = dual_leg._call_llm(prompt, max_tokens=300)

            # Set response for logging
            self.logger.set_response(response)

            # Parse JSON response
            decision = json.loads(response)
            suggested_modify = decision.get('suggested_modify', 'clean')

            return {
                'suggested_modify': suggested_modify,
                'reason': decision.get('reason', 'No reason provided'),
                'pair_analysis': decision.get('pair_analysis', ''),
                'llm_response': response
            }

        except Exception as e:
            # Log error
            self.logger.log_error(column, "pairwise_conflict_analysis", str(e))

            # Fallback: modify clean
            return {
                'suggested_modify': 'clean',
                'reason': f'LLM analysis failed: {e}',
                'pair_analysis': f'Error: {e}'
            }

    def _build_synthesis_prompt(self, column: str,
                                 clean_rule: PillarRule,
                                 pairwise_results: List[Dict[str, Any]],
                                 metadata: Dict[str, Any] = None) -> str:
        """Build prompt to synthesize all pairwise decisions into a final decision."""
        # Format metadata
        meta_str = ""
        if metadata:
            top_values = metadata.get('top_values', {})
            if top_values:
                top_items = list(top_values.items())[:5]
                top_str = ", ".join([f"'{v}'" for v, _ in top_items])
            else:
                top_str = "N/A"
            meta_str = f"- Top 5 Values: {top_str}"

        # Format pairwise results summary
        results_summary = "\n".join([
            f"  - {r.get('clean_rule', '?')} vs {r.get('dirty_rule', '?')}: "
            f"suggest modify={r.get('suggested_modify', '?')}, reason={r.get('reason', '?')[:50]}..."
            for r in pairwise_results
        ])

        # Count suggestions
        clean_count = sum(1 for r in pairwise_results if r.get('suggested_modify') == 'clean')
        dirty_count = sum(1 for r in pairwise_results if r.get('suggested_modify') == 'dirty')

        return f"""Synthesize the conflict analysis results and make a FINAL DECISION for column '{column}':

**Clean Rule ({clean_rule.name}):**
{clean_rule.rule_str}

**Column Info:**
{meta_str}

**Pairwise Analysis Results ({len(pairwise_results)} pairs analyzed):**
{results_summary}

**Summary:**
- Suggested modify CLEAN: {clean_count} pairs
- Suggested modify DIRTY: {dirty_count} pairs

**Task:**
Based on all pairwise analyses above, make a FINAL decision on which rules to modify.
Consider:
1. If most pairs suggest modifying the same side, follow the majority
2. If opinions are split, consider which side is causing more conflicts
3. If modifying dirty, list which specific dirty rules should be modified

Return ONLY a JSON object:
{{
  "final_decision": "clean" or "dirty",
  "synthesis_reason": "Detailed explanation of the final decision",
  "affected_rules": ["list", "of", "rule", "names", "to", "modify"]  # empty if modifying clean
}}
"""

    def _synthesize_decision(self, column: str,
                             clean_rule: PillarRule,
                             pairwise_results: List[Dict[str, Any]],
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Synthesize all pairwise decisions into a final decision.
        """
        # If only one result, use it directly
        if len(pairwise_results) == 1:
            return {
                'final_decision': pairwise_results[0].get('suggested_modify', 'clean'),
                'synthesis_reason': 'Single pairwise result, using it directly',
                'affected_rules': [pairwise_results[0].get('dirty_rule', '')]
            }

        # If no factory, use voting
        if not self.factory:
            clean_count = sum(1 for r in pairwise_results if r.get('suggested_modify') == 'clean')
            dirty_count = sum(1 for r in pairwise_results if r.get('suggested_modify') == 'dirty')
            final_decision = 'clean' if clean_count >= dirty_count else 'dirty'
            affected_rules = [r.get('dirty_rule', '') for r in pairwise_results
                            if r.get('suggested_modify') == 'dirty']
            return {
                'final_decision': final_decision,
                'synthesis_reason': f'Veting: clean={clean_count}, dirty={dirty_count}',
                'affected_rules': affected_rules
            }

        # Use LLM to synthesize
        prompt = self._build_synthesis_prompt(column, clean_rule, pairwise_results, metadata)

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            from agentic_error_detector.legislator import DualLegislator
            dual_leg = DualLegislator(
                self.factory.base_url,
                self.factory.model
            )

            response = dual_leg._call_llm(prompt, max_tokens=300)

            # Set response for logging
            self.logger.set_response(response)

            # Parse JSON response
            result = json.loads(response)

            return {
                'final_decision': result.get('final_decision', 'clean'),
                'synthesis_reason': result.get('synthesis_reason', ''),
                'affected_rules': result.get('affected_rules', [])
            }

        except Exception as e:
            self.logger.log_error(column, "synthesis_decision", str(e))

            # Fallback: voting
            clean_count = sum(1 for r in pairwise_results if r.get('suggested_modify') == 'clean')
            dirty_count = sum(1 for r in pairwise_results if r.get('suggested_modify') == 'dirty')
            final_decision = 'clean' if clean_count >= dirty_count else 'dirty'
            affected_rules = [r.get('dirty_rule', '') for r in pairwise_results
                            if r.get('suggested_modify') == 'dirty']
            return {
                'final_decision': final_decision,
                'synthesis_reason': f'Voting fallback: clean={clean_count}, dirty={dirty_count}',
                'affected_rules': affected_rules
            }
    
    def _refine_clean_rule(self, pillar_name: str, rule: PillarRule,
                          analysis: Dict[str, Any],
                          metadata: Dict[str, Any] = None,
                          dirty_only_values: List[str] = None,
                          conflicting_values: List[str] = None) -> PillarRule:
        """Refine a clean pillar rule to exclude conflicts."""
        if not self.factory:
            # Return unchanged if no LLM available
            return rule

        column = getattr(self, 'current_column', 'unknown')
        round_num = getattr(self, 'round_num', 1)

        # Format metadata for prompt
        meta_str = ""
        if metadata:
            top_values = metadata.get('top_values', {})
            if top_values:
                top_items = list(top_values.items())[:10]
                top_str = ", ".join([f"'{v}' ({c}次)" for v, c in top_items])
            else:
                top_str = "N/A"

            meta_str = f"""
**Column Metadata:**
- Type: {metadata.get('type', 'unknown')}
- Top 10 Values: {top_str}
- Unique Count: {metadata.get('unique_count', 'unknown')}
- Non-Null Count: {metadata.get('non_null_count', 'unknown')}
"""

        # Format dirty-only values
        dirty_section = ""
        if dirty_only_values:
            dirty_examples = "\n".join([f"  - '{v}'" for v in dirty_only_values[:5]])
            dirty_section = f"""
**Dirty-Only Values (correctly detected as dirty - should keep detecting):**
{dirty_examples}
"""

        # Format conflicting values
        conflict_section = ""
        if conflicting_values:
            conflict_examples = "\n".join([f"  - '{v}'" for v in conflicting_values[:10]])
            conflict_section = f"""
**Conflicting Values (BOTH clean AND dirty - need to resolve):**
These values incorrectly satisfy both clean and dirty rules.
{conflict_examples}
"""

        prompt = f"""Refine this clean data quality rule to be more precise:

**Current Rule ({pillar_name}):**
{rule.rule_str}
{meta_str}
{dirty_section}
{conflict_section}
**Problem:**
{analysis['reason']}

**Modification History:**
{self.memory.to_context() if hasattr(self.memory, 'to_context') else 'N/A'}

**CRITICAL INSTRUCTIONS for Clean Rules:**
1. **Logic Direction**: A Clean Rule MUST return `True` ONLY when the value is VALID/CLEAN. It MUST return `False` for errors.
2. **Regex Precision**: Pay extreme attention to separators. If the clean samples show 'a.m.' or 'p.m.', your regex MUST include the dots: `[ap]\\.m\\.`.
3. **Task**: Generate a refined lambda function that returns `True` for clean-only values but returns `False` for the conflicting values (which are actually dirty).

Return ONLY the lambda function, no explanation.

Example format:
lambda value, row=None: <expression>
"""

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            from agentic_error_detector.legislator import DualLegislator
            dual_leg = DualLegislator(self.factory.base_url, self.factory.model)

            response = dual_leg._call_llm(prompt, max_tokens=300, temperature=0.2)

            # Set response for logging
            self.logger.set_response(response)

            # Extract lambda from response
            new_rule_str = self._extract_lambda(response)
            if new_rule_str:
                # Create new rule object (without compiling yet)
                new_rule = PillarRule(
                    name=pillar_name,
                    rule_str=new_rule_str,
                    rule_func=None,
                    version=rule.version + 1,
                    modification_log=rule.modification_log + [analysis.get('reason', '')]
                )

                # Get df from outer scope - need to pass it or access via resolve()
                df = getattr(self, '_current_df', None)
                if df is not None:
                    # Validate the refined rule
                    is_accepted, violation_rate, reject_reason = self._validate_refined_rule(
                        df, column, rule, new_rule, 'clean'
                    )

                    if not is_accepted:
                        # Reject the rule - log and return original
                        self.logger.log_rejection(
                            rule_type="clean",
                            rule_name=pillar_name,
                            old_rule=rule.rule_str,
                            new_rule=new_rule_str,
                            violation_rate=violation_rate,
                            reason=reject_reason
                        )

                        if hasattr(self.memory, 'add'):
                            self.memory.add('clean', pillar_name, 'rejected',
                                          reject_reason, new_rule_str)

                        print(f"  ⚠ Clean rule '{pillar_name}' rejected: violation rate {violation_rate*100:.1f}%")
                        return rule  # Return original rule

                # Rule accepted - add modification to logging
                self.logger.add_modification(
                    rule_type="clean",
                    rule_name=pillar_name,
                    old_rule=rule.rule_str,
                    new_rule=new_rule_str,
                    modification_type="tightened",
                    reason=analysis.get('reason', analysis.get('llm_analysis', ''))
                )

                if hasattr(self.memory, 'add'):
                    self.memory.add('clean', pillar_name, 'tightened',
                                  analysis.get('reason', ''), new_rule_str)

                return new_rule
        except Exception as e:
            self.logger.log_error(column, "refine_clean", str(e))
            print(f"  ⚠ Failed to refine clean rule: {e}")

        return rule
    
    def _refine_dirty_rule(self, agent_name: str, rule: PillarRule,
                          reasons: List[str],
                          metadata: Dict[str, Any] = None,
                          dirty_only_values: List[str] = None,
                          conflicting_values: List[str] = None) -> PillarRule:
        """Refine a dirty agent rule based on accumulated reasons."""
        if not self.factory:
            return rule

        column = getattr(self, 'current_column', 'unknown')
        round_num = getattr(self, 'round_num', 1)

        combined_reason = '; '.join(reasons[:3])  # Limit to first 3 reasons

        # Format metadata for prompt
        meta_str = ""
        if metadata:
            top_values = metadata.get('top_values', {})
            if top_values:
                top_items = list(top_values.items())[:10]
                top_str = ", ".join([f"'{v}' ({c}次)" for v, c in top_items])
            else:
                top_str = "N/A"

            meta_str = f"""
**Column Metadata:**
- Type: {metadata.get('type', 'unknown')}
- Top 10 Values: {top_str}
- Unique Count: {metadata.get('unique_count', 'unknown')}
"""

        # Format dirty-only values (should keep detecting)
        dirty_section = ""
        if dirty_only_values:
            dirty_examples = "\n".join([f"  - '{v}'" for v in dirty_only_values[:5]])
            dirty_section = f"""
**Dirty-Only Values (correctly detected as dirty - MUST keep detecting):**
{dirty_examples}
"""

        # Format conflicting values (false positives - should stop detecting)
        conflict_section = ""
        if conflicting_values:
            conflict_examples = "\n".join([f"  - '{v}'" for v in conflicting_values[:10]])
            conflict_section = f"""
**Conflicting Values (FALSE POSITIVES - incorrectly detected as dirty):**
These values are clean and should NOT be flagged.
{conflict_examples}
"""

        prompt = f"""Refine this dirty detection rule to avoid false positives:

**Current Rule ({agent_name}):**
{rule.rule_str}
{meta_str}
{dirty_section}
{conflict_section}
**Accumulated Issues:**
{combined_reason}

**Modification History:**
{self.memory.to_context() if hasattr(self.memory, 'to_context') else 'N/A'}

Generate a refined lambda function that excludes the conflicting values while still detecting dirty-only values.
Return ONLY the lambda function, no explanation.

Example format:
lambda value, row=None: <expression>
"""

        # Set prompt for logging
        self.logger.set_prompt(prompt)

        try:
            from agentic_error_detector.legislator import DualLegislator
            dual_leg = DualLegislator(self.factory.base_url, self.factory.model)

            response = dual_leg._call_llm(prompt, max_tokens=300, temperature=0.2)

            # Set response for logging
            self.logger.set_response(response)

            new_rule_str = self._extract_lambda(response)
            if new_rule_str:
                # Create new rule object (without compiling yet)
                new_rule = PillarRule(
                    name=agent_name,
                    rule_str=new_rule_str,
                    rule_func=None,
                    version=rule.version + 1,
                    modification_log=rule.modification_log + [combined_reason]
                )

                # Get df from outer scope - need to pass it or access via resolve()
                df = getattr(self, '_current_df', None)
                if df is not None:
                    # Validate the refined rule
                    is_accepted, violation_rate, reject_reason = self._validate_refined_rule(
                        df, column, rule, new_rule, 'dirty'
                    )

                    if not is_accepted:
                        # Reject the rule - log and return original
                        self.logger.log_rejection(
                            rule_type="dirty",
                            rule_name=agent_name,
                            old_rule=rule.rule_str,
                            new_rule=new_rule_str,
                            violation_rate=violation_rate,
                            reason=reject_reason
                        )

                        if hasattr(self.memory, 'add'):
                            self.memory.add('dirty', agent_name, 'rejected',
                                          reject_reason, new_rule_str)

                        print(f"  ⚠ Dirty rule '{agent_name}' rejected: violation rate {violation_rate*100:.1f}%")
                        return rule  # Return original rule

                # Rule accepted - add modification to logging
                self.logger.add_modification(
                    rule_type="dirty",
                    rule_name=agent_name,
                    old_rule=rule.rule_str,
                    new_rule=new_rule_str,
                    modification_type="relaxed",
                    reason=combined_reason
                )

                if hasattr(self.memory, 'add'):
                    self.memory.add('dirty', agent_name, 'relaxed', combined_reason, new_rule_str)

                return new_rule
        except Exception as e:
            self.logger.log_error(column, "refine_dirty", str(e))
            print(f"  ⚠ Failed to refine dirty rule: {e}")

        return rule
    
    def _extract_lambda(self, response: str) -> str:
        """Extract lambda expression from LLM response."""
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith('lambda'):
                return line
        return None
    
    def _invoke_rule(self, func, value, row):
        """Invoke a rule function safely."""
        if func is None:
            return False
        try:
            code = func.__code__
            if code.co_argcount == 1:
                return func(value)
            return func(value, row)
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
        if new_rule.rule_func is None:
            return True, 0.0, "accepted"

        # Compile the new rule function if needed
        new_func = new_rule.rule_func
        if new_func is None and new_rule.rule_str:
            try:
                new_func = self._compile_rule(new_rule.rule_str, side)
            except Exception as e:
                return True, 0.0, f"compile_error: {e}"

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

    def _compile_rule(self, rule_str: str, side: str) -> callable:
        """Compile a rule string to a callable function."""
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
            return None
