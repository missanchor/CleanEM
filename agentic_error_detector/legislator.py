"""
LLM-based Rule Generators using local vLLM (OpenAI-compatible API).
"""
import json
import re
from typing import Dict, Any, List, Tuple
from openai import OpenAI

DEFAULT_MISSING_TOKENS = ["", "nan", "none", "null", "n/a", "na", "unknown", "empty", "xxxxx"]
DEFAULT_MISSING_TOKEN_STR = str(DEFAULT_MISSING_TOKENS)


class BaseLegislator:
    """Base class for all legislators."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None):
        """Initialize with local vLLM endpoint."""
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model or self._get_available_model()
        self.base_url = base_url

    def _get_available_model(self) -> str:
        """Fetch the available model from vLLM endpoint."""
        try:
            response = self.client.models.list()
            if response.data:
                return response.data[0].id
            return "Qwen/Qwen2.5-0.5B-Instruct"  # fallback
        except Exception as e:
            print(f"  ⚠ Could not fetch available model: {e}. Using fallback model.")
            return "Qwen/Qwen2.5-0.5B-Instruct"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate Python lambda functions as strings. Returns list of candidates."""
        raise NotImplementedError

    def _get_system_prompt(self) -> str:
        """Get the system prompt for this legislator. Override in subclasses."""
        return "You are a data quality expert. Generate Python lambda functions for data validation. Return ONLY the lambda functions, one per line."

    def _call_llm(self, prompt: str, max_tokens: int = 500, system_prompt: str = None) -> str:
        """Call the LLM with a prompt."""
        try:
            system_msg = system_prompt if system_prompt else self._get_system_prompt()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ Error calling LLM: {e}")
            return ""


class MissingLegislator(BaseLegislator):
    """Agent focused on detecting missing/null values."""

    def _get_system_prompt(self) -> str:
        """System prompt for missing value detection."""
        return """You are a data completeness expert specializing in detecting missing, null, and empty values.
Your role is to generate Python lambda functions that identify incomplete or absent data.
You understand various representations of missing data (None, NaN, empty strings, 'N/A', 'null', etc.).
Return ONLY the lambda functions, one per line. Each function should return True for valid (non-missing) values."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to detect missing values."""
        null_count = metadata.get('null_count', 0)
        total_rows = metadata.get('total_rows', 0)
        
        # Missing value check is simple: value should not be null
        rule = f"lambda value: pd.notna(value) if hasattr(pd, 'notna') else (str(value).strip().lower() not in ['', 'nan', 'none', 'empty'])"
        
        if null_count > 0:
            return [rule]  # Only return if there are nulls
        return []


class TypoLegislator(BaseLegislator):
    """Legislator focused on detecting typos in string columns."""

    def _get_system_prompt(self) -> str:
        """System prompt for typo and spelling error detection."""
        return """You are a spelling and data entry expert specializing in detecting typos and spelling errors.
Your role is to generate Python lambda functions that identify misspellings, character substitutions, and data entry errors.
You understand common typo patterns (character swaps, substitutions, extra/missing characters) and frequency-based anomalies.
Return ONLY the lambda functions, one per line. Each function should return False (flagging an error) for suspected typos."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to detect typos based on frequency analysis."""
        top_values = metadata.get('top_values', {})
        frequency_dist = metadata.get('frequency_distribution', {})
        unique_count = metadata.get('unique_count', 0)
        
        if not top_values or unique_count < 2:
            return []

        top_freq_values = list(top_values.keys())[:10]
        
        prompt = f"""
For column '{column}', generate up to 3 Python lambda functions to detect potential typos.

Column metadata:
- Unique values: {unique_count}
- Top 10 most frequent values: {json.dumps(top_freq_values)}

Rules should:
1. Check if a value is rare (appears < 2 times) AND similar to a frequent value
2. Identify character-level anomalies (e.g., 'x' substitution in '{column}')
3. Flag values not matching the majority pattern

Return ONLY lambda functions, one per line. Format:
lambda value: <expression>

Example:
lambda value: str(value).lower().strip() in {json.dumps([v.lower() for v in top_freq_values])} if value else True
"""
        rules_text = self._call_llm(prompt)
        
        if not rules_text or "lambda" not in rules_text.lower():
            return self._fallback_typo_rules(column, metadata)
        
        # Parse lambda functions from response
        rules = [line.strip() for line in rules_text.split('\n') if line.strip().startswith('lambda')]
        return rules if rules else self._fallback_typo_rules(column, metadata)

    def _fallback_typo_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate fallback typo detection rules."""
        top_values = metadata.get('top_values', {})
        top_freq_values = list(top_values.keys())[:5]
        
        rules = []
        # Rule 1: Value must be in frequent values (conservative)
        if top_freq_values:
            safe_values = "', '".join([str(v).replace("'", "\\'") for v in top_freq_values])
            rules.append(f"lambda value: str(value).lower().strip() in ['{safe_values}'] if value else True")
        
        # Rule 2: Check for common typo characters
        rules.append("lambda value: 'x' not in str(value).lower() if value else True")
        
        return rules


class PatternLegislator(BaseLegislator):
    """Agent focused on detecting pattern violations (IDs, codes, etc.)."""

    def _get_system_prompt(self) -> str:
        """System prompt for format and pattern validation."""
        return """You are a data format validation expert specializing in ID codes, phone numbers, ZIP codes, and structured patterns.
Your role is to generate Python lambda functions that validate format compliance and pattern consistency.
You understand regex patterns, fixed-length codes, alphanumeric conventions, and structural constraints.
Return ONLY the lambda functions, one per line. Each function should return True for valid patterns."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to validate patterns."""
        sample_values = metadata.get('sample_values', [])
        pattern_analysis = metadata.get('pattern_analysis', '')
        
        if not sample_values:
            return []

        prompt = f"""
For column '{column}', generate up to 3 Python lambda functions to validate the format.

Column metadata:
- Pattern analysis: {pattern_analysis}
- Sample values: {json.dumps(sample_values[:10])}

Rules should validate:
1. Expected format (digits only, specific length, special characters)
2. Consistent structure across values
3. Valid characters for the field type

Return ONLY lambda functions, one per line. Include both strict and loose variations.

Examples:
lambda value: bool(re.match(r'^\\d{{5}}$', str(value))) if value else True
lambda value: len(str(value).replace('-', '').replace(' ', '')) == 10 if value else True
"""
        rules_text = self._call_llm(prompt)
        
        if not rules_text or "lambda" not in rules_text.lower():
            return self._fallback_pattern_rules(column, metadata)
        
        rules = [line.strip() for line in rules_text.split('\n') if line.strip().startswith('lambda')]
        return rules if rules else self._fallback_pattern_rules(column, metadata)

    def _fallback_pattern_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate fallback pattern validation rules."""
        rules = []
        sample_values = metadata.get('sample_values', [])
        
        if not sample_values:
            return []
        
        # Analyze sample values for patterns
        sample_str = [str(v) for v in sample_values[:10]]
        
        # Check for digit-only pattern
        if all(v.isdigit() for v in sample_str):
            digit_len = len(sample_str[0]) if sample_str else 5
            rules.append(f"lambda value: str(value).isdigit() and len(str(value)) == {digit_len} if value else True")
        
        # Check for mixed alphanumeric
        if any(not v.isdigit() for v in sample_str):
            rules.append("lambda value: len(str(value).strip()) > 0 if value else True")
        
        return rules


class OutlierLegislator(BaseLegislator):
    """Agent focused on detecting numeric outliers."""

    def _get_system_prompt(self) -> str:
        """System prompt for numeric anomaly and outlier detection."""
        return """You are a statistical analyst expert in detecting numeric outliers and anomalies.
Your role is to generate Python lambda functions that identify values outside acceptable ranges or statistical norms.
You understand domain constraints, statistical thresholds (mean, std dev), and reasonable value ranges for different data types.
Return ONLY the lambda functions, one per line. Each function should return True for valid numeric values."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to detect numeric outliers."""
        numeric_count = metadata.get('numeric_count', 0)
        min_val = metadata.get('min')
        max_val = metadata.get('max')
        mean_val = metadata.get('mean')
        
        if numeric_count == 0 or min_val is None or max_val is None:
            return []

        prompt = f"""
For column '{column}', generate up to 3 Python lambda functions to detect numeric outliers.

Column statistics:
- Min: {min_val}
- Max: {max_val}
- Mean: {mean_val}
- Valid numeric values: {numeric_count}

Rules should:
1. Check for reasonable value ranges based on domain knowledge
2. Flag extremely high or low values
3. Validate against typical patterns for the field

Return ONLY lambda functions, one per line.

Examples:
lambda value: float(value) <= 150 if value else True
lambda value: float(value) >= 0 if value else True
"""
        rules_text = self._call_llm(prompt)
        
        if not rules_text or "lambda" not in rules_text.lower():
            return self._fallback_outlier_rules(min_val, max_val)
        
        rules = [line.strip() for line in rules_text.split('\n') if line.strip().startswith('lambda')]
        return rules if rules else self._fallback_outlier_rules(min_val, max_val)

    def _fallback_outlier_rules(self, min_val: float, max_val: float) -> List[str]:
        """Generate fallback outlier detection rules."""
        rules = []
        
        # Rule 1: Max value should be reasonable
        if max_val and max_val > 1000:  # Likely outliers
            rules.append(f"lambda value: float(value) <= {max_val * 0.5} if value else True")
        
        # Rule 2: Min value should be non-negative for typical metrics
        if min_val and min_val < 0:
            rules.append("lambda value: float(value) >= 0 if value else True")
        
        return rules


class LogicLegislator(BaseLegislator):
    """Agent focused on logical consistency checks (cross-column rules)."""

    def _get_system_prompt(self) -> str:
        """System prompt for logical consistency and cross-column validation."""
        return """You are a data integrity expert specializing in cross-column logical consistency and business rule validation.
Your role is to generate Python lambda functions that verify relationships between multiple columns (e.g., temporal ordering, value dependencies).
You understand domain logic, temporal constraints, and referential integrity requirements.
Return ONLY the lambda functions, one per line. Each function accepts a row (dict) and returns True for logically consistent data."""

    def generate_rules(self, row_data: Dict[str, Any], all_metadata: Dict[str, Any]) -> List[str]:
        """Generate rules for logical consistency across columns."""
        # This is a special agent that operates on entire rows, not single columns
        rules = []
        
        # Example: Check for date consistency (if applicable)
        if 'AdmissionDate' in all_metadata and 'DischargeDate' in all_metadata:
            prompt = """
Generate a Python lambda function to check if DischargeDate >= AdmissionDate for hospital records.

Rule should:
1. Handle date parsing
2. Return True if dates are valid and ordered correctly
3. Return False if discharge is before admission

Format:
lambda row: <expression>

Example:
lambda row: pd.to_datetime(row['DischargeDate']) >= pd.to_datetime(row['AdmissionDate']) if row.get('DischargeDate') and row.get('AdmissionDate') else True
"""
            rule_text = self._call_llm(prompt, max_tokens=300)
            if rule_text and "lambda" in rule_text.lower():
                rules.append(rule_text.strip())
        
        return rules


class CleanBaseRuleAgent:
    """Base class for deterministic clean-rule agents (non-LLM)."""

    pillar_name = "base"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        raise NotImplementedError


class CleanCompletenessLegislator(CleanBaseRuleAgent):
    """Ensure values are present (non-missing)."""

    pillar_name = "completeness"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        rule = (
            "lambda value, row=None: ("
            "value is not None and "
            "str(value).strip().lower() not in " + DEFAULT_MISSING_TOKEN_STR +
            ")"
        )
        return [rule]


class CleanAccuracyLegislator(CleanBaseRuleAgent):
    """Ensure values fall into reasonable numeric/text ranges."""

    pillar_name = "accuracy"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        rules = []
        col_type = metadata.get('type', 'text')
        min_val = metadata.get('min')
        max_val = metadata.get('max')

        if col_type == 'numeric' and min_val is not None and max_val is not None:
            tolerance = max(1.0, abs(max_val - min_val) * 0.05 or 1.0)
            lower_bound = min_val - tolerance
            upper_bound = max_val + tolerance
            rules.append(
                "lambda value, row=None: ("
                "safe_float(value) is not None and "
                f"{lower_bound} <= safe_float(value) <= {upper_bound}"
                ")"
            )
        else:
            sample_values = metadata.get('sample_values') or list((metadata.get('top_values') or {}).keys())
            normalized = [len(str(val).strip()) for val in sample_values if val is not None and str(val).strip()]
            if normalized:
                min_len = max(1, min(normalized) - 2)
                max_len = max(normalized) + 4
                rules.append(
                    "lambda value, row=None: ("
                    "value is not None and "
                    f"{min_len} <= len(str(value).strip()) <= {max_len}"
                    ")"
                )

        return rules


class CleanPatternLegislator(CleanBaseRuleAgent):
    """Ensure values respect known patterns (IDs, codes)."""

    pillar_name = "pattern_consistency"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        rules = []
        pattern_hint = (metadata.get('pattern_analysis') or "").lower()
        column_lower = column.lower()

        if 'zip' in column_lower or 'postal' in column_lower or '5-digit numeric' in pattern_hint:
            rules.append("lambda value, row=None: bool(re.match(r'^\\d{5}$', str(value).strip())) if value is not None else False")
        elif '10-digit numeric' in pattern_hint or 'phone' in column_lower:
            rules.append("lambda value, row=None: bool(re.match(r'^\\d{10}$', re.sub(r'\\D', '', str(value)))) if value is not None else False")
        elif 'pattern' in pattern_hint and 'alphanumeric' in pattern_hint:
            rules.append("lambda value, row=None: bool(re.match(r'^[A-Za-z0-9\\-]+$', str(value).strip())) if value is not None else False")
        else:
            sample_values = metadata.get('sample_values') or list((metadata.get('top_values') or {}).keys())
            if sample_values:
                canonical_lengths = {len(str(val).strip()) for val in sample_values if val is not None and str(val).strip()}
                if len(canonical_lengths) == 1:
                    target_len = canonical_lengths.pop()
                    if target_len > 0:
                        rules.append(
                            "lambda value, row=None: ("
                            "value is not None and len(str(value).strip()) == "
                            f"{target_len}"
                            ")"
                        )

        return rules


class CleanRelationshipLegislator(CleanBaseRuleAgent):
    """Ensure intra-row column relationship constraints."""

    pillar_name = "column_relationship"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        constraints = metadata.get('relationship_constraints') or []
        if not constraints:
            return []

        rules = []
        for constraint in constraints:
            constraint_type = constraint.get('type')
            other_column = constraint.get('other_column')

            if not other_column:
                continue

            if constraint_type == 'prefix_match':
                rules.append(
                    "lambda value, row=None: ("
                    "row is None or row.get('" + other_column + "') in [None, ''] or "
                    "(value is not None and "
                    "str(value).strip().lower().startswith(str(row.get('" + other_column + "')).strip().lower()))"
                    ")"
                )
            elif constraint_type == 'contains':
                rules.append(
                    "lambda value, row=None: ("
                    "row is None or row.get('" + other_column + "') in [None, ''] or "
                    "(value is not None and "
                    "str(row.get('" + other_column + "')).strip().lower() in str(value).strip().lower())"
                    ")"
                )
            elif constraint_type == 'stateavg_format':
                rules.append(
                    "lambda value, row=None: ("
                    "row is None or row.get('" + other_column + "') in [None, ''] or "
                    "(value is not None and "
                    "str(value).strip().lower().startswith(str(row.get('" + other_column + "')).strip().lower() + '_'))"
                    ")"
                )
            elif constraint_type == 'zip_prefix':
                rules.append(
                    "lambda value, row=None: ("
                    "row is None or row.get('" + other_column + "') in [None, ''] or "
                    "(value is not None and "
                    "str(row.get('" + other_column + "')).strip()[:3] == str(value).strip()[:3])"
                    ")"
                )

        return rules


class DualLegislator(BaseLegislator):
    """Legislator that generates paired clean/dirty rules for dual verification."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None):
        """Initialize dual legislator."""
        super().__init__(base_url, model)

    def get_p_clean_system_prompt(self) -> str:
        """System prompt for P_clean (clean data confirmation) rule generation."""
        return """You are a data quality expert specializing in identifying CLEAN data.

Your role is to generate Python lambda functions for P_clean(x) that CONFIRM when a value is definitely CLEAN (correct/proper).

CRITICAL REQUIREMENTS (cover every pillar below):
1. Completeness: reject missing/placeholder tokens (None, NaN, empty strings, 'n/a', 'unknown', etc.)
2. Accuracy: ensure numeric ranges, enumerations, and domain-specific thresholds are respected
3. Column relationship constraints: when metadata provides relational expectations, enforce them conservatively
4. Pattern/format consistency: honor regex-like structures, code lengths, and canonical casing
5. P_clean should remain PERMISSIVE overall—only reject values that clearly break the pillars above
6. Must return boolean values (True/False) and handle edge cases gracefully

Return ONLY the lambda function:
lambda value: <expression>

Example for a categorical column 'diagnosis':
lambda value: value is not None and str(value).strip().lower() in ['diabetes', 'hypertension', 'asthma', 'cancer', 'pneumonia', 'flu', 'cold', 'headache', 'fever', 'cough'] if value else False

Example for numeric column 'age':
lambda value: value is not None and str(value).strip() not in ['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown'] and str(value).replace('.', '', 1).replace('-', '', 1).isdigit() and -1000 <= float(value) <= 200

Be generous in what you accept as clean—P_clean's job is to confirm validity without being stricter than necessary.
"""

    def get_p_dirty_system_prompt(self) -> str:
        """System prompt for P_dirty (dirty data detection) rule generation."""
        return """You are a data quality expert specializing in identifying DIRTY data.

Your role is to generate Python lambda functions for P_dirty(x) that CONFIRM when a value is definitely DIRTY (incorrect/problematic).

CRITICAL REQUIREMENTS:
1. P_dirty should be STRICT - only flag values that are clearly problematic
2. Focus on detecting obvious errors: missing values, extreme outliers, clear typos
3. Do NOT flag values that could potentially be valid (be conservative)
4. Must return boolean values (True/False) and handle edge cases gracefully
5. All missing/placeholder values (None, NaN, empty strings, 'n/a', 'unknown', etc.) MUST return True

Return ONLY the lambda function:
lambda value: <expression>

Example for a categorical column 'diagnosis':
lambda value: value is None or str(value).strip().lower() in ['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'xxxxx', 'asdf', '123'] or len(str(value).strip()) > 50

Example for numeric column 'age':
lambda value: value is None or str(value).strip() in ['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown'] or (str(value).replace('.', '', 1).replace('-', '', 1).isdigit() and (float(value) < 0 or float(value) > 150))

Be strict in what you flag as dirty - P_dirty's job is to catch clear errors, not to be permissive.
"""

    def _looks_like_time_column(self, column: str, metadata: Dict[str, Any]) -> bool:
        """Heuristic: detect time-like columns (e.g., flights *_time)."""
        name = (column or "").lower()
        if "time" in name:
            return True

        samples = metadata.get("sample_values") or []
        top_values = list((metadata.get("top_values") or {}).keys())
        candidates = [str(v) for v in (samples[:20] + top_values[:20]) if v is not None]
        if not candidates:
            return False

        time_like = 0
        for v in candidates[:40]:
            s = str(v).strip().lower()
            if ":" in s and ("a.m" in s or "p.m" in s or "am" in s or "pm" in s):
                time_like += 1
            elif re.search(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}", s):
                time_like += 1
        return time_like >= 2

    def _time_p_clean_rule(self) -> str:
        """
        Deterministic P_clean for time strings used in flights dataset.

        Accepts:
        - 'H:MM a.m.' / 'HH:MM p.m.' (with optional dots/casing)
        - 'MM/DD/YYYY H:MM a.m.' (date + time)
        Rejects:
        - missing/placeholder tokens
        """
        missing_tokens = "['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown']"
        return (
            "lambda value: ("
            "value is not None and "
            "str(value).strip().lower() not in " + missing_tokens + " and "
            "("
            "bool(re.match(r'^\\d{1,2}:\\d{2}\\s*(a\\.?m\\.?|p\\.?m\\.?)\\.?$', str(value).strip(), re.IGNORECASE))"
            " or "
            "bool(re.match(r'^\\d{1,2}/\\d{1,2}/\\d{4}\\s+\\d{1,2}:\\d{2}\\s*(a\\.?m\\.?|p\\.?m\\.?)\\.?$', str(value).strip(), re.IGNORECASE))"
            ")"
            ")"
        )

    @staticmethod
    def _complement_dirty_rule(clean_rule_str: str) -> str:
        """
        Build P_dirty as the (safe) logical complement of P_clean to guarantee:
        - No Grey Zone (coverage-by-design)
        - No Conflict (mutual exclusivity)

        Requires Judge.safe_dict to include `safe_not`.
        """
        return f"lambda value, row=None, _p=({clean_rule_str}): safe_not(_p, value, row)"

    def _combine_clean_base_rules(self, clean_base_rules: List[Tuple[str, str]] = None) -> str:
        """Combine deterministic clean base rules (per pillar) into a single predicate."""
        if not clean_base_rules:
            return None

        callables = [rule for _, rule in clean_base_rules if rule]
        if not callables:
            return None

        assignments = ", ".join([f"_pillar_{idx}=({rule})" for idx, rule in enumerate(callables)])
        invocations = " and ".join([f"_pillar_{idx}(value, row)" for idx in range(len(callables))]) or "True"

        return f"lambda value, row=None, {assignments}: ({invocations})"

    def generate_p_clean_rule(self, column: str, metadata: Dict[str, Any],
                              clean_base_rules: List[Tuple[str, str]] = None) -> str:
        """
        Generate P_clean rule for a column (separately from P_dirty).

        Args:
            column: Column name
            metadata: Column metadata

        Returns:
            P_clean lambda function string
        """
        # Deterministic fast-path for time-like columns
        if self._looks_like_time_column(column, metadata):
            return self._time_p_clean_rule()

        combined_clean = self._combine_clean_base_rules(clean_base_rules)
        if combined_clean:
            return combined_clean

        grey_samples = metadata.get("_refine_grey_samples") or []
        conflict_samples = metadata.get("_refine_conflict_samples") or []
        all_dirty_samples = metadata.get("_refine_all_dirty_samples") or []

        col_type = metadata.get('type', 'text')
        sample_values = metadata.get('sample_values', [])
        top_values = metadata.get('top_values', {})
        unique_count = metadata.get('unique_count', 0)
        null_count = metadata.get('null_count', 0)
        pattern_analysis = metadata.get('pattern_analysis')
        relationship_constraints = metadata.get('relationship_constraints')
        accuracy_constraints = metadata.get('accuracy_constraints') or metadata.get('valid_range')
        base_rule_references = metadata.get('_base_rules') or []

        # Build prompt for P_clean
        prompt_parts = [
            f"Generate P_clean predicate for column '{column}'",
            f"\nColumn type: {col_type}",
            f"\nSample values: {sample_values[:10]}",
            f"\nTop values: {list(top_values.keys())[:10]}",
            f"\nUnique count: {unique_count}",
            f"\nNull count: {null_count}",
        ]

        if accuracy_constraints:
            prompt_parts.append(f"\nAccuracy constraints / valid range hints: {accuracy_constraints}")

        if pattern_analysis:
            prompt_parts.append(f"\nPattern analysis: {pattern_analysis}")

        if relationship_constraints:
            prompt_parts.append(f"\nColumn relationship constraints to respect: {relationship_constraints}")

        if base_rule_references:
            prompt_parts.append("\nReference base rules from specialized agents (align clean predicate with their intent):")
            for agent_name, rule_string in base_rule_references[:3]:
                prompt_parts.append(f"- {agent_name}: {rule_string}")

        prompt_parts.append("\nQuality pillars to encode explicitly:")
        prompt_parts.append("1. Completeness – reject standard missing tokens or placeholders.")
        prompt_parts.append("2. Accuracy – enforce domain ranges or enumerations without being overly strict.")
        prompt_parts.append("3. Column constraints – honor provided inter-column rules conservatively.")
        prompt_parts.append("4. Pattern consistency – respect format/regex guidance when supplied.")

        if grey_samples or conflict_samples or all_dirty_samples:
            prompt_parts.append("\n\nRefinement context (problematic examples to handle):")
            if grey_samples:
                prompt_parts.append(f"- Grey samples: {[s.get('value') for s in grey_samples[:10]]}")
            if conflict_samples:
                prompt_parts.append(f"- Conflict samples: {[s.get('value') for s in conflict_samples[:10]]}")
            if all_dirty_samples:
                prompt_parts.append(f"- All-dirty dominant samples: {[s.get('value') for s in all_dirty_samples[:10]]}")

        prompt_parts.append("\n\nReturn ONLY the lambda function:")
        prompt_parts.append("lambda value: <expression>")

        prompt = "\n".join(prompt_parts)
        rules_text = self._call_llm(prompt, max_tokens=400, system_prompt=self.get_p_clean_system_prompt())

        if not rules_text or "lambda" not in rules_text.lower():
            return self._fallback_p_clean_rule(column, metadata)

        # Parse the response
        lines = [line.strip() for line in rules_text.split('\n') if line.strip()]
        clean_rule = None

        def _extract_lambda(line: str) -> str:
            """Extract lambda expression from a line."""
            if not line:
                return None

            if ':' in line and 'lambda' in line.lower():
                _, expr = line.split(':', 1)
                expr = expr.strip()
            else:
                expr = line.strip()

            return expr if expr.lower().startswith('lambda') else None

        for line in lines:
            if line.lower().startswith('lambda'):
                clean_rule = line
                break
            candidate = _extract_lambda(line)
            if candidate:
                clean_rule = candidate
                break

        if not clean_rule:
            return self._fallback_p_clean_rule(column, metadata)

        return clean_rule

    def generate_p_dirty_rule(self, column: str, metadata: Dict[str, Any]) -> str:
        """
        Generate P_dirty rule for a column (separately from P_clean).

        Args:
            column: Column name
            metadata: Column metadata

        Returns:
            P_dirty lambda function string
        """
        col_type = metadata.get('type', 'text')
        sample_values = metadata.get('sample_values', [])
        top_values = metadata.get('top_values', {})
        unique_count = metadata.get('unique_count', 0)
        null_count = metadata.get('null_count', 0)

        # Build prompt for P_dirty
        prompt_parts = [
            f"Generate P_dirty predicate for column '{column}'",
            f"\nColumn type: {col_type}",
            f"\nSample values: {sample_values[:10]}",
            f"\nTop values: {list(top_values.keys())[:10]}",
            f"\nUnique count: {unique_count}",
            f"\nNull count: {null_count}",
        ]

        prompt_parts.append("\n\nReturn ONLY the lambda function:")
        prompt_parts.append("lambda value: <expression>")

        prompt = "\n".join(prompt_parts)
        rules_text = self._call_llm(prompt, max_tokens=400, system_prompt=self.get_p_dirty_system_prompt())

        if not rules_text or "lambda" not in rules_text.lower():
            return self._fallback_p_dirty_rule(column, metadata)

        # Parse the response
        lines = [line.strip() for line in rules_text.split('\n') if line.strip()]
        dirty_rule = None

        def _extract_lambda(line: str) -> str:
            """Extract lambda expression from a line."""
            if not line:
                return None

            if ':' in line and 'lambda' in line.lower():
                _, expr = line.split(':', 1)
                expr = expr.strip()
            else:
                expr = line.strip()

            return expr if expr.lower().startswith('lambda') else None

        for line in lines:
            if line.lower().startswith('lambda'):
                dirty_rule = line
                break
            candidate = _extract_lambda(line)
            if candidate:
                dirty_rule = candidate
                break

        if not dirty_rule:
            return self._fallback_p_dirty_rule(column, metadata)

        return dirty_rule

    def generate_dual_rules(self, column: str, metadata: Dict[str, Any],
                           grey_samples: List[Dict[str, Any]] = None,
                           conflict_samples: List[Dict[str, Any]] = None,
                           all_dirty_samples: List[Dict[str, Any]] = None,
                           base_rules: List[Tuple[str, str]] = None,
                           clean_base_rules: List[Tuple[str, str]] = None) -> List[Tuple[str, str, str]]:
        """
        Generate paired clean/dirty rules for a column.

        Args:
            column: Column name
            metadata: Column metadata
            grey_samples: Samples from grey zone (for refinement)
            conflict_samples: Samples causing conflicts (for refinement)
            all_dirty_samples: Samples where entire column marked dirty (for refinement)
            base_rules: Base error detection rules from specialized agents (Missing, Typo, Outlier, Pattern)
            clean_base_rules: Deterministic clean rules per quality pillar (completeness, accuracy, etc.)

        Returns:
            List of (agent_name, clean_rule_str, dirty_rule_str) tuples
        """
        # If we have base rules from specialized agents, use them to inform dual rule generation
        if base_rules:
            # Extract patterns from base rules to inform dual rule generation
            self._log_base_rule_patterns(column, base_rules)

        # Thread refinement samples into metadata so generate_p_clean_rule can use them
        meta = dict(metadata or {})
        meta["_refine_grey_samples"] = grey_samples or []
        meta["_refine_conflict_samples"] = conflict_samples or []
        meta["_refine_all_dirty_samples"] = all_dirty_samples or []
        meta["_base_rules"] = base_rules or []
        meta["_clean_base_rules"] = clean_base_rules or []

        # Generate P_clean, then derive P_dirty as safe complement (coverage-by-design).
        clean_rule = self.generate_p_clean_rule(column, meta, clean_base_rules=clean_base_rules)
        if not clean_rule:
            clean_rule = self._fallback_p_clean_rule(column, meta)

        dirty_rule = self._complement_dirty_rule(clean_rule)

        # Return as list of single candidate (can generate multiple in future)
        return [(self.__class__.__name__, clean_rule, dirty_rule)]

    def _log_base_rule_patterns(self, column: str, base_rules: List[Tuple[str, str]]):
        """Log and analyze base rule patterns for debugging/informing dual rule generation."""
        if not base_rules:
            return

        agent_types = set([agent_name for agent_name, _ in base_rules])
        print(f"  Info: Base rules from agents: {', '.join(agent_types)}")
        print(f"  Info: {len(base_rules)} base error detection rules available for reference")

    def _fallback_p_clean_rule(self, column: str, metadata: Dict[str, Any]) -> str:
        """Generate fallback P_clean rule (permissive)."""
        col_type = metadata.get('type', 'text')
        sample_values = metadata.get('sample_values', [])
        top_values = metadata.get('top_values', {})

        # P_clean is permissive - accepts a broad range of values
        missing_tokens = "['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown']"

        if col_type == 'numeric':
            # Permissive numeric: accept most numbers within reasonable range
            clean_rule = (
                "lambda value: value is not None and str(value).strip() not in "
                f"{missing_tokens} and str(value).replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).replace('e', '', 1).replace('E', '', 1).isdigit() "
                "and -100000 <= float(value) <= 1000000"
            )
        elif col_type == 'categorical':
            # Permissive categorical: accept any non-empty string that doesn't look like missing value
            clean_rule = (
                "lambda value: value is not None and str(value).strip() not in "
                f"{missing_tokens} and len(str(value).strip()) > 0"
            )
        else:  # text
            # Permissive text: accept any non-empty string
            clean_rule = (
                "lambda value: value is not None and str(value).strip() not in "
                f"{missing_tokens} and len(str(value).strip()) > 0"
            )

        return clean_rule

    def _fallback_p_dirty_rule(self, column: str, metadata: Dict[str, Any]) -> str:
        """Generate fallback P_dirty rule (strict)."""
        col_type = metadata.get('type', 'text')
        sample_values = metadata.get('sample_values', [])
        top_values = metadata.get('top_values', {})

        # P_dirty is strict - only flag obvious problems
        missing_tokens = "['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'xxxxx', 'asdf', 'test123']"

        if col_type == 'numeric':
            # Strict numeric: only flag if clearly invalid format or extreme outlier
            dirty_rule = (
                "lambda value: value is None or str(value).strip() in "
                f"{missing_tokens} or (str(value).replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).replace('e', '', 1).replace('E', '', 1).isdigit() "
                "and (float(value) < -100000 or float(value) > 1000000))"
            )
        elif col_type == 'categorical':
            # Strict categorical: only flag if clearly missing or obviously invalid
            dirty_rule = (
                "lambda value: value is None or str(value).strip() in "
                f"{missing_tokens} or len(str(value).strip()) > 100"
            )
        else:  # text
            # Strict text: only flag if missing or obviously problematic
            dirty_rule = (
                "lambda value: value is None or str(value).strip() in "
                f"{missing_tokens}"
            )

        return dirty_rule

    def _fallback_dual_rules(self, column: str, metadata: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Generate fallback dual rules."""
        col_type = metadata.get('type', 'text')
        sample_values = metadata.get('sample_values', [])
        top_values = metadata.get('top_values', {})

        # Conservative fallback: enforce full coverage with explicit missing handling
        safe_values = list(top_values.keys())[:5] if top_values else []
        missing_tokens = "['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown']"

        if col_type == 'numeric':
            clean_rule = (
                "lambda value: value is not None and str(value).strip() not in "
                f"{missing_tokens} and str(value).replace('.', '', 1).replace('-', '', 1).isdigit() "
                "and -1000000000 <= float(value) <= 1000000000"
            )
            dirty_rule = (
                "lambda value: value is None or str(value).strip() in "
                f"{missing_tokens} or (str(value).replace('.', '', 1).replace('-', '', 1).isdigit() "
                "and (float(value) < -1000000000 or float(value) > 1000000000))"
            )
        elif col_type == 'categorical':
            if safe_values:
                normalized = [str(v).strip().lower() for v in safe_values if str(v).strip()]
                safe_values_list = ", ".join([f"'{val}'" for val in normalized]) or "''"
                clean_rule = (
                    "lambda value: value is not None and str(value).strip() not in "
                    f"{missing_tokens} and str(value).strip().lower() in [{safe_values_list}]"
                )
                dirty_rule = (
                    "lambda value: value is None or str(value).strip() in "
                    f"{missing_tokens} or (str(value).strip().lower() not in [{safe_values_list}])"
                )
            else:
                clean_rule = (
                    "lambda value: value is not None and str(value).strip() not in "
                    f"{missing_tokens}"
                )
                dirty_rule = (
                    "lambda value: value is None or str(value).strip() in "
                    f"{missing_tokens}"
                )
        else:  # text
            clean_rule = (
                "lambda value: value is not None and str(value).strip() not in "
                f"{missing_tokens}"
            )
            dirty_rule = (
                "lambda value: value is None or str(value).strip() in "
                f"{missing_tokens}"
            )

        return [(self.__class__.__name__, clean_rule, dirty_rule)]


class LegislatorFactory:
    """Factory to create appropriate legislators based on column type."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None):
        """Initialize factory with base URL and model."""
        self.base_url = base_url
        self.model = model

    def create_agents(self, column: str, column_type: str) -> List[BaseLegislator]:
        """Create appropriate agents for a column based on its type."""
        agents = []
        
        # All columns can have missing values
        agents.append(MissingLegislator(self.base_url, self.model))
        
        # Type-specific agents
        if column_type == 'categorical':
            agents.append(TypoLegislator(self.base_url, self.model))
        elif column_type == 'pattern':
            agents.append(PatternLegislator(self.base_url, self.model))
        elif column_type == 'numeric':
            agents.append(OutlierLegislator(self.base_url, self.model))
        
        # Fallback: also try typo for text columns
        if column_type == 'text':
            agents.append(TypoLegislator(self.base_url, self.model))
        
        return agents

    def create_clean_agents(self, column: str, column_type: str) -> List[CleanBaseRuleAgent]:
        """Create clean-rule agents that cover completeness/accuracy/relationships/patterns."""
        return [
            CleanCompletenessLegislator(),
            CleanAccuracyLegislator(),
            CleanRelationshipLegislator(),
            CleanPatternLegislator(),
        ]

    def generate_rules_per_column(self, metadata: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate rules for all columns in metadata.
        
        Returns:
            Dict[column_name] = List[(agent_name, rule_string)]
        """
        all_rules = {}

        for column, col_metadata in metadata.items():
            col_type = col_metadata.get('type', 'text')
            agents = self.create_agents(column, col_type)
            
            col_rules = []
            for agent in agents:
                agent_name = agent.__class__.__name__
                print(f"  → {agent_name} for {column}...")
                
                try:
                    rules = agent.generate_rules(column, col_metadata)
                    for rule in rules:
                        col_rules.append((agent_name, rule))
                        print(f"    ✓ Generated rule: {rule}")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
            
            if col_rules:
                all_rules[column] = col_rules
        
        return all_rules

    def generate_clean_rules_per_column(self, metadata: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate clean base rules per column (four quality pillars).

        Returns:
            Dict[column_name] = List[(pillar_name, rule_string)]
        """
        clean_rules = {}

        for column, col_metadata in metadata.items():
            column_type = col_metadata.get('type', 'text')
            agents = self.create_clean_agents(column, column_type)
            column_rules = []

            for agent in agents:
                try:
                    pillar_rules = agent.generate_rules(column, col_metadata)
                    for rule in pillar_rules:
                        column_rules.append((agent.pillar_name, rule))
                except Exception as e:
                    print(f"    ✗ Clean agent {agent.pillar_name} error on {column}: {e}")

            if column_rules:
                clean_rules[column] = column_rules

        return clean_rules

    def generate_cross_column_rules(self, metadata: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Generate cross-column consistency rules.

        Returns:
            List[(agent_name, rule_string)]
        """
        logic_agent = LogicLegislator(self.base_url, self.model)
        rules = logic_agent.generate_rules({}, metadata)
        return [("LogicLegislator", rule) for rule in rules]

    def generate_dual_rules_per_column(self, metadata: Dict[str, Any],
                                      base_rules: Dict[str, List[Tuple[str, str]]] = None,
                                      clean_base_rules: Dict[str, List[Tuple[str, str]]] = None,
                                      refinement_history: Dict[str, List] = None) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Generate dual rules (paired clean/dirty predicates) for all columns.

        Args:
            metadata: Column metadata dictionary
            base_rules: Base error detection rules from specialized agents
                       {column: [(agent_name, rule_string), ...]}
            refinement_history: Optional history of previous refinement attempts
                               {column: [RefinementRound objects]}

        Returns:
            Dict[column_name] = List[(agent_name, clean_rule_str, dirty_rule_str)]
        """
        all_rules = {}
        base_rules = base_rules or {}
        clean_base_rules = clean_base_rules or {}
        refinement_history = refinement_history or {}

        print("\n" + "="*80)
        print("GENERATING DUAL RULES (P_clean/P_dirTY) FOR ALL COLUMNS")
        print("="*80)
        print("Note: Dual rules are informed by base error detection rules from specialized agents")
        print("      (MissingLegislator, TypoLegislator, OutlierLegislator, PatternLegislator)")

        for column, col_metadata in metadata.items():
            print(f"\n{'='*80}")
            print(f"Column: {column}")
            print(f"{'='*80}")

            # Create dual legislator
            dual_legislator = DualLegislator(self.base_url, self.model)

            # Get base rules for this column (if any)
            column_base_rules = base_rules.get(column, [])

            # Get refinement samples if this column was previously refined
            grey_samples = None
            conflict_samples = None
            all_dirty_samples = None

            if column in refinement_history:
                for round_data in refinement_history[column]:
                    if isinstance(round_data, dict) and 'samples_used' in round_data:
                        grey_samples = round_data['samples_used'].get('grey', [])
                        conflict_samples = round_data['samples_used'].get('conflict', [])
                        all_dirty_samples = round_data['samples_used'].get('all_dirty', [])

            # Generate rules
            try:
                col_rules = dual_legislator.generate_dual_rules(
                    column, col_metadata,
                    grey_samples=grey_samples,
                    conflict_samples=conflict_samples,
                    all_dirty_samples=all_dirty_samples,
                    base_rules=column_base_rules,
                    clean_base_rules=clean_base_rules.get(column)
                )

                if col_rules:
                    all_rules[column] = col_rules
                    agent_name, clean_rule, dirty_rule = col_rules[0]
                    print(f"✓ Generated dual rules from {agent_name}:")
                    print(f"  P_clean: {clean_rule}")
                    print(f"  P_dirty: {dirty_rule}")
                else:
                    print(f"✗ No rules generated")

            except Exception as e:
                print(f"✗ Error generating rules: {e}")

        return all_rules