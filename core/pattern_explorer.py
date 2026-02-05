"""
PatternExplorer: Multi-round pattern exploration for column structure discovery.

This module implements the core logic for exploring patterns in a column through
multiple rounds of LLM inference and local validation, aiming to achieve high
coverage of the column's values.
"""
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter

import pandas as pd
from openai import OpenAI

from .pattern_types import PatternSpec, PatternEntry, ExplorationFeedback


# Configuration constants
MAX_EXPLORATION_ROUNDS = 3  # Maximum number of exploration rounds
PATTERN_COVERAGE_THRESHOLD = 0.05  # Minimum coverage to keep a pattern
TARGET_COVERAGE = 0.98  # Target coverage to stop exploration
MIN_IMPROVEMENT_TO_CONTINUE = 0.02  # Minimum improvement to continue
UNCOVERED_SAMPLES_PER_ROUND = 20  # Number of uncovered samples to show LLM


class PatternExplorer:
    """Pattern Explorer: Multi-round iterative pattern discovery."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None):
        """Initialize the PatternExplorer.

        Args:
            base_url: OpenAI-compatible API endpoint
            model: Model name to use (auto-detected if None)
        """
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model or self._get_available_model()

    def _get_available_model(self) -> str:
        """Fetch the available model from the endpoint."""
        try:
            response = self.client.models.list()
            if response.data:
                return response.data[0].id
            return "Qwen/Qwen2.5-0.5B-Instruct"
        except Exception:
            return "Qwen/Qwen2.5-0.5B-Instruct"

    def explore(self, column: str, df: pd.DataFrame,
                metadata: Dict[str, Any]) -> PatternSpec:
        """Execute multi-round pattern exploration.

        Args:
            column: Column name to explore
            df: DataFrame containing the column
            metadata: Column metadata from profiler

        Returns:
            PatternSpec with discovered patterns and coverage
        """
        # Prepare initial input for the first round
        input_data = self._prepare_initial_input(column, df, metadata)

        # Track discovered patterns across rounds
        discovered_patterns: List[PatternEntry] = []
        round_num = 0
        previous_coverage = 0.0

        while round_num < MAX_EXPLORATION_ROUNDS:
            # 1. Generate candidate patterns from LLM
            candidates = self._generate_candidate_patterns(
                input_data, discovered_patterns, round_num
            )

            # 2. Validate candidates locally (compute coverage)
            validated = self._validate_patterns(df, column, candidates)

            if not validated:
                # No valid patterns found, try fallback
                validated = self._fallback_pattern_detection(column, df, metadata)

            # 3. Compute overall coverage
            current_coverage = self._compute_overall_coverage(df, column, validated)

            # 4. Check stop conditions
            if self._should_stop(current_coverage, previous_coverage, round_num):
                break

            # 5. Get uncovered examples for feedback
            uncovered_examples = self._get_uncovered_examples(
                df, column, validated, max_samples=UNCOVERED_SAMPLES_PER_ROUND
            )

            # 6. Build feedback for next round
            feedback = ExplorationFeedback(
                round_number=round_num,
                current_coverage=current_coverage,
                patterns=validated,
                uncovered_examples=uncovered_examples,
                uncovered_shapes=self._analyze_uncovered_shapes(uncovered_examples),
                improvement=current_coverage - previous_coverage
            )

            # 7. Merge patterns (keep best from each round)
            discovered_patterns = self._merge_patterns(discovered_patterns, validated)

            # 8. Prepare input for next round
            input_data = self._prepare_next_round_input(feedback, round_num + 1)
            previous_coverage = current_coverage
            round_num += 1

        # Build and return final PatternSpec
        return self._build_final_pattern_spec(
            column, discovered_patterns, df, round_num
        )

    def _prepare_initial_input(self, column: str, df: pd.DataFrame,
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for the first exploration round."""
        sample_values = df[column].dropna().astype(str).head(20).tolist()

        return {
            'column': column,
            'pattern_analysis': metadata.get('pattern_analysis', ''),
            'shape_distribution': metadata.get('shape_distribution', [])[:8],
            'length_distribution': metadata.get('length_distribution', [])[:8],
            'sample_values': sample_values,
            'regex_candidates': metadata.get('regex_candidates', [])[:3],
            'is_first_round': True
        }

    def _generate_candidate_patterns(
            self, input_data: Dict[str, Any],
            discovered_patterns: List[PatternEntry],
            round_num: int) -> List[Dict[str, Any]]:
        """Generate candidate patterns from LLM."""
        if round_num == 0:
            return self._generate_initial_patterns(input_data)
        else:
            return self._generate_refined_patterns(input_data, discovered_patterns)

    def _generate_initial_patterns(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial candidate patterns."""
        prompt = f"""
你是一个数据模式分析专家。请分析列 '{input_data['column']}' 的结构模式。

列元数据:
- Pattern分析: {input_data['pattern_analysis']}
- Shape分布: {json.dumps(input_data['shape_distribution'], ensure_ascii=False)}
- 长度分布: {json.dumps(input_data['length_distribution'], ensure_ascii=False)}
- 示例值: {json.dumps(input_data['sample_values'][:15], ensure_ascii=False)}

请输出该列可能遵循的1-5个正则表达式模式。每个模式用JSON表示:
- regex: 正则表达式字符串
- description: 模式的自然语言描述

JSON格式:
{{"patterns": [{{"regex": "...", "description": "..."}}]}}
"""
        response = self._call_llm(prompt)
        return self._parse_pattern_response(response)

    def _generate_refined_patterns(
            self, input_data: Dict[str, Any],
            discovered_patterns: List[PatternEntry]) -> List[Dict[str, Any]]:
        """Generate refined patterns based on uncovered examples."""
        uncovered = input_data.get('uncovered_examples', [])
        current_coverage = input_data.get('current_coverage', 0)

        # Format existing patterns for the prompt
        existing = [{"regex": p.regex, "description": p.description}
                    for p in discovered_patterns]

        prompt = f"""
第 {input_data.get('round_number', 1) + 1} 轮 pattern 调整：

当前已发现的模式:
{json.dumps(existing, ensure_ascii=False)}

当前覆盖率: {current_coverage:.2%}

仍未被覆盖的值样例（这些值需要新的pattern来覆盖）:
{json.dumps(uncovered[:15], ensure_ascii=False)}

请根据这些未覆盖的值：
1. 调整现有 regex 以提高覆盖率
2. 新增补充 pattern 覆盖仍未覆盖的结构
3. 移除冗余或低质量的 pattern

输出JSON格式:
{{"patterns": [{{"regex": "...", "description": "..."}}]}}
"""
        response = self._call_llm(prompt)
        return self._parse_pattern_response(response)

    def _validate_patterns(self, df: pd.DataFrame, column: str,
                           candidates: List[Dict[str, Any]]) -> List[PatternEntry]:
        """Validate candidate patterns and compute coverage."""
        validated = []
        series = df[column].dropna().astype(str)

        for candidate in candidates:
            regex = candidate.get('regex', '')
            if not regex:
                continue

            try:
                # Test the regex pattern
                matches = series.str.fullmatch(regex, na=False).sum()
                coverage = matches / len(series) if len(series) > 0 else 0.0

                if coverage >= PATTERN_COVERAGE_THRESHOLD:
                    validated.append(PatternEntry(
                        regex=regex,
                        coverage=coverage,
                        description=candidate.get('description', '')
                    ))
            except re.error:
                continue

        # Sort by coverage descending
        validated.sort(key=lambda p: p.coverage, reverse=True)
        return validated

    def _compute_overall_coverage(self, df: pd.DataFrame, column: str,
                                   patterns: List[PatternEntry]) -> float:
        """Compute the overall coverage of all patterns combined."""
        if not patterns:
            return 0.0

        series = df[column].dropna().astype(str)
        if series.empty:
            return 0.0

        # Build combined regex
        combined_regex = self._build_combined_regex(patterns)
        if not combined_regex:
            return 0.0

        try:
            matched = series.str.fullmatch(combined_regex, na=False).sum()
            return matched / len(series)
        except re.error:
            return 0.0

    def _build_combined_regex(self, patterns: List[PatternEntry]) -> str:
        """Build combined regex from multiple patterns."""
        if not patterns:
            return ""
        # Wrap each pattern in non-capturing group and combine with alternation
        return "|".join(f"(?:{p.regex})" for p in patterns)

    def _should_stop(self, current_coverage: float, previous_coverage: float,
                     round_num: int) -> bool:
        """Check if exploration should stop."""
        # Condition 1: Target coverage reached
        if current_coverage >= TARGET_COVERAGE:
            return True

        # Condition 2: Maximum rounds reached
        if round_num >= MAX_EXPLORATION_ROUNDS - 1:
            return True

        # Condition 3: Improvement is too small
        improvement = current_coverage - previous_coverage
        if round_num >= 1 and improvement < MIN_IMPROVEMENT_TO_CONTINUE:
            return True

        return False

    def _get_uncovered_examples(self, df: pd.DataFrame, column: str,
                                 patterns: List[PatternEntry],
                                 max_samples: int = 20) -> List[str]:
        """Get sample values not covered by any pattern."""
        series = df[column].dropna().astype(str)
        if series.empty:
            return []

        combined_regex = self._build_combined_regex(patterns)

        def is_uncovered(value: str) -> bool:
            if not value or not value.strip():
                return True
            if not combined_regex:
                return True
            try:
                return not bool(re.match(combined_regex, value.strip()))
            except re.error:
                return True

        # Find all uncovered values
        all_values = series.tolist()
        uncovered = [v for v in all_values if is_uncovered(v)]

        if not uncovered:
            return []

        # Sample diverse uncovered values (different shape/length)
        seen_signatures: Set[Tuple[str, int]] = set()
        diverse_samples = []

        for v in uncovered:
            signature = (self._shape_signature(v), len(v))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                diverse_samples.append(v)
            if len(diverse_samples) >= max_samples:
                break

        return diverse_samples

    def _shape_signature(self, value: str) -> str:
        """Get the shape signature of a string value."""
        if not value:
            return "<empty>"

        tokens = []
        for char in str(value).strip():
            if char.isdigit():
                tokens.append("D")
            elif char.isalpha():
                tokens.append("A")
            elif char in "-_/.":
                tokens.append(char)
            elif char.isspace():
                tokens.append("S")
            else:
                tokens.append("?")
        return "".join(tokens) or "<empty>"

    def _analyze_uncovered_shapes(self, uncovered: List[str]) -> Dict[str, int]:
        """Analyze shape distribution of uncovered values."""
        counter = Counter()
        for v in uncovered:
            counter[self._shape_signature(v)] += 1
        return dict(counter)

    def _merge_patterns(self, existing: List[PatternEntry],
                        new: List[PatternEntry]) -> List[PatternEntry]:
        """Merge patterns from different rounds, keeping the best coverage."""
        pattern_map: Dict[str, PatternEntry] = {}

        # Add existing patterns
        for p in existing:
            pattern_map[p.regex] = p

        # Add or update with new patterns
        for p in new:
            if p.regex not in pattern_map:
                pattern_map[p.regex] = p
            elif p.coverage > pattern_map[p.regex].coverage:
                pattern_map[p.regex] = p

        # Return sorted by coverage
        result = list(pattern_map.values())
        result.sort(key=lambda p: p.coverage, reverse=True)
        return result

    def _prepare_next_round_input(self, feedback: ExplorationFeedback,
                                   round_num: int) -> Dict[str, Any]:
        """Prepare input data for the next round."""
        return {
            'column': feedback.patterns[0].regex if feedback.patterns else '',
            'round_number': feedback.round_number,
            'current_coverage': feedback.current_coverage,
            'uncovered_examples': feedback.uncovered_examples,
            'uncovered_shapes': feedback.uncovered_shapes,
            'patterns': [{'regex': p.regex, 'coverage': p.coverage}
                         for p in feedback.patterns],
            'is_first_round': False
        }

    def _build_final_pattern_spec(self, column: str,
                                   patterns: List[PatternEntry],
                                   df: pd.DataFrame,
                                   rounds: int) -> PatternSpec:
        """Build the final PatternSpec."""
        # Recompute coverage with all patterns combined
        combined_coverage = self._compute_overall_coverage(df, column, patterns)

        # Get final uncovered examples
        uncovered = self._get_uncovered_examples(df, column, patterns,
                                                  max_samples=UNCOVERED_SAMPLES_PER_ROUND)

        return PatternSpec(
            column=column,
            patterns=patterns,
            overall_coverage=combined_coverage,
            exploration_rounds=rounds,
            uncovered_examples=uncovered
        )

    def _fallback_pattern_detection(self, column: str, df: pd.DataFrame,
                                     metadata: Dict[str, Any]) -> List[PatternEntry]:
        """Fallback pattern detection using simple heuristics."""
        series = df[column].dropna().astype(str)
        if series.empty:
            return []

        samples = series.head(20).tolist()
        patterns = []

        # Check for digit-only pattern
        if all(v.isdigit() for v in samples if v):
            digit_len = len(samples[0]) if samples else 5
            regex = f"^\\d{{{digit_len}}}$"
            matches = series.str.fullmatch(regex).sum()
            coverage = matches / len(series)

            if coverage > 0:
                patterns.append(PatternEntry(
                    regex=regex,
                    coverage=coverage,
                    description=f"{digit_len}-digit numeric pattern"
                ))

        # Check for mixed alphanumeric
        if any(not v.isdigit() for v in samples if v):
            # Try to infer a general pattern
            regex = self._infer_general_pattern(samples)
            if regex:
                try:
                    matches = series.str.fullmatch(regex).sum()
                    coverage = matches / len(series)
                    if coverage > PATTERN_COVERAGE_THRESHOLD:
                        patterns.append(PatternEntry(
                            regex=regex,
                            coverage=coverage,
                            description="General alphanumeric pattern"
                        ))
                except re.error:
                    pass

        return patterns

    def _infer_general_pattern(self, samples: List[str]) -> str:
        """Infer a general regex pattern from samples."""
        if not samples:
            return ""

        # Simple heuristic: analyze character types
        all_alpha = all(v.isalpha() for v in samples if v)
        all_digit = all(v.isdigit() for v in samples if v)
        has_special = any(re.search(r'[-_./]', v) for v in samples if v)

        if all_digit:
            # Determine common length
            lengths = [len(v) for v in samples if v.isdigit()]
            if lengths:
                common_len = max(set(lengths), key=lengths.count)
                return f"^\\d{{{common_len}}}$"

        if all_alpha:
            lengths = [len(v) for v in samples if v.isalpha()]
            if lengths:
                common_len = max(set(lengths), key=lengths.count)
                return f"^[A-Za-z]{{{common_len}}}$"

        if has_special:
            # Check for separator-based pattern
            if any('-' in v for v in samples):
                return r"^[A-Za-z0-9\-]+$"

        return r"^[A-Za-z0-9\-_]+$"

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data pattern analysis expert. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            return cleaned
        except Exception as e:
            print(f"  [PatternExplorer] LLM call error: {e}")
            return "{}"

    def _parse_pattern_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract patterns."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('patterns', [])

            # Try parsing as array
            data = json.loads(response)
            if isinstance(data, list):
                return data
            return data.get('patterns', [])
        except json.JSONDecodeError:
            print(f"  [PatternExplorer] Failed to parse response: {response[:100]}...")
            return []
