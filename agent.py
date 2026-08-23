"""
LLM-based Rule Generators using local vLLM (OpenAI-compatible API).
"""
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

import pandas as pd

from cleanem_models import ColumnSemantics
from core.pattern_types import PatternSpec
from core.pattern_explorer import PatternExplorer

DEFAULT_MISSING_TOKENS = ["", "nan", "none", "null", "n/a", "na", "unknown", "empty", "xxxxx"]
LLM_MAX_TOKENS = 2048

# Configuration for batch processing and pattern exploration
VALUE_BATCH_SIZE = 30  # Number of unique values per batch for LLM annotation
MAX_BATCHES_PER_COLUMN = 20  # Maximum batches per column to prevent infinite loops
PATTERN_COVERAGE_THRESHOLD = 0.05  # Minimum coverage to keep a pattern
TARGET_COVERAGE = 0.98  # Target coverage to stop pattern exploration
MIN_IMPROVEMENT_TO_CONTINUE = 0.02  # Minimum improvement to continue exploration


def _batch_unique_values(values_with_counts: List[Tuple[str, int]],
                          batch_size: int = VALUE_BATCH_SIZE) -> List[List[Tuple[str, int]]]:
    """Split unique values into batches for LLM processing.

    Args:
        values_with_counts: List of (value, count) tuples, sorted by frequency
        batch_size: Number of values per batch

    Returns:
        List of batches, each batch is a list of (value, count) tuples
    """
    if not values_with_counts:
        return []

    batches = []
    for i in range(0, len(values_with_counts), batch_size):
        batches.append(values_with_counts[i:i + batch_size])
    return batches


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None):
        """Initialize with local vLLM endpoint."""
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model or self._get_available_model()
        self.base_url = base_url
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

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
        """Get the system prompt for this agent. Override in subclasses."""
        return "You are a data quality expert. Generate Python lambda functions for data validation. Return ONLY the lambda functions, one per line."

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    def _call_llm(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS, system_prompt: str = None, temperature: float = 0.1, extra_body: Dict[str, Any] = None) -> str:
        """Call the LLM with a prompt."""
        try:
            system_msg = system_prompt if system_prompt else self._get_system_prompt()
            self.last_prompt = prompt
            self.last_system_prompt = system_msg
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            raw_content = response.choices[0].message.content or ""
            cleaned = self._strip_think_tags(raw_content).strip()
            self.last_response = cleaned
            return cleaned
        except Exception as e:
            print(f"  ⚠ Error calling LLM: {e}")
            self.last_response = f"Error: {e}"
            return ""

    def get_last_prompt_info(self) -> Dict[str, str]:
        """Get the last prompt and response for logging."""
        return {
            'prompt': self.last_prompt or '',
            'system_prompt': self.last_system_prompt or '',
            'response': self.last_response or ''
        }


class ColumnArchetypeAgent(BaseAgent):
    """Infer semantic column types before downstream evidence generation."""

    ALLOWED_ARCHETYPES = {
        "closed_enum",
        "identifier",
        "open_entity",
        "geo_name",
        "numeric_measure",
        "unit_measure",
        "temporal_measure",
        "free_text",
    }

    MECHANISMS_BY_ARCHETYPE = {
        "closed_enum": ["missing", "closed_set_violation", "value_contamination"],
        "identifier": ["missing", "format_violation", "entity_swap"],
        "open_entity": ["missing", "encoding_artifact", "entity_swap"],
        "geo_name": ["missing", "geo_suffix_shift", "value_contamination"],
        "numeric_measure": ["missing", "numeric_outlier", "percent_marker_contamination"],
        "unit_measure": ["missing", "unit_variant", "suffix_pollution"],
        "temporal_measure": [
            "missing",
            "temporal_format_violation",
            "temporal_range_violation",
            "temporal_order_violation",
        ],
        "free_text": ["missing", "encoding_artifact", "token_contamination"],
    }

    def _get_system_prompt(self) -> str:
        return """You classify a table column by semantic archetype.
Return exactly one JSON object and no prose.
Allowed archetypes: closed_enum, identifier, open_entity, geo_name,
numeric_measure, unit_measure, temporal_measure, free_text.
Definitions:
- closed_enum: a small controlled vocabulary, even when labels are long strings.
- identifier: IDs, codes, phone numbers, zip values, or other entity keys.
- open_entity: names of people, organizations, products, or other long-tail entities.
- geo_name: city, town, province, region, or other geographic names; it may be open-set.
- numeric_measure: numeric quantities, rates, scores, and averages.
- unit_measure: values containing both a number and a measurement unit.
- temporal_measure: dates, years, clock times, timestamps, or durations whose
  values have temporal format, range, or ordering semantics.
- free_text: unconstrained natural-language content.
Scores must be numbers between 0 and 1. The supplied distribution statistics
are objective facts. Do not reinterpret unique_ratio as semantic confidence."""

    @staticmethod
    def _clip_score(value: Any, default: float) -> float:
        try:
            return float(min(1.0, max(0.0, float(value))))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _first_distribution_ratio(metadata: Dict[str, Any], key: str) -> float:
        items = metadata.get(key) or []
        if not items:
            return 0.0
        try:
            return float(items[0].get("ratio", 0.0))
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _heuristic_fallback(
        self,
        column: str,
        metadata: Dict[str, Any],
        fallback_reason: str = "heuristic_fallback",
    ) -> ColumnSemantics:
        name = column.strip().lower()
        snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", column.strip()).lower()
        name_tokens = set(filter(None, re.split(r"[^a-z0-9]+", snake_name)))
        col_type = str(metadata.get("type", "text")).lower()
        unique_ratio = self._clip_score(metadata.get("unique_ratio"), 0.0)
        top_coverage = self._clip_score(metadata.get("top_value_coverage"), 0.0)
        shape_coverage = self._clip_score(
            self._first_distribution_ratio(metadata, "shape_distribution"),
            0.0,
        )
        percent_ratio = self._clip_score(metadata.get("percent_marker_ratio"), 0.0)

        identifier_names = {
            "id",
            "provider_number",
            "zip_code",
            "zipcode",
            "postal_code",
            "phone_number",
            "phonenumber",
            "flight",
        }
        temporal_tokens = {
            "time",
            "date",
            "datetime",
            "timestamp",
            "year",
            "duration",
        }
        if (
            bool(name_tokens.intersection(temporal_tokens))
            or snake_name.endswith("_at")
        ):
            archetype = "temporal_measure"
        elif (
            name in identifier_names
            or snake_name in identifier_names
            or name.endswith("_id")
            or name.endswith("identifier")
        ):
            archetype = "identifier"
        elif name_tokens.intersection({"city", "town", "province", "region"}):
            archetype = "geo_name"
        elif (
            snake_name in {"state", "state_code", "country_code"}
            or name_tokens.intersection({"status", "category", "style"})
        ):
            archetype = "closed_enum"
        elif any(token in name for token in ("ounce", "ounces", "weight", "unit", "size")):
            archetype = "unit_measure"
        elif (
            col_type == "numeric"
            or name_tokens.intersection({"abv", "amount", "price", "rate", "score", "average", "avg"})
            or name.endswith("avg")
        ):
            archetype = "numeric_measure"
        elif "name" in name or unique_ratio >= 0.50:
            archetype = "open_entity"
        elif unique_ratio <= 0.10 and top_coverage >= 0.30:
            archetype = "closed_enum"
        else:
            archetype = "free_text"

        structure_strength = max(top_coverage, shape_coverage)
        canonicalization_need = {
            "unit_measure": 0.95,
            "geo_name": 0.80,
            "numeric_measure": max(0.45, percent_ratio),
            "temporal_measure": 0.30,
            "closed_enum": 0.40,
            "identifier": 0.35,
            "open_entity": 0.25,
            "free_text": 0.20,
        }[archetype]

        return ColumnSemantics(
            column=column,
            archetype=archetype,
            confidence=0.65,
            open_set_score=unique_ratio,
            structure_strength=structure_strength,
            canonicalization_need=canonicalization_need,
            possible_error_mechanisms=list(self.MECHANISMS_BY_ARCHETYPE[archetype]),
            rationale=[fallback_reason],
        )

    @staticmethod
    def _extract_json_object(response: str) -> Dict[str, Any]:
        cleaned = (response or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object in archetype response")
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Archetype response must be a JSON object")
        return payload

    def _reconcile_archetype(
        self,
        column: str,
        metadata: Dict[str, Any],
        llm_archetype: str,
        heuristic: ColumnSemantics,
    ) -> Tuple[str, Optional[str]]:
        """Override LLM output only when profiling supplies a strong constraint."""
        snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", column.strip()).lower()
        name_tokens = set(filter(None, re.split(r"[^a-z0-9]+", snake_name)))
        unique_ratio = self._clip_score(metadata.get("unique_ratio"), 0.0)
        top_coverage = self._clip_score(metadata.get("top_value_coverage"), 0.0)
        coarse_type = str(metadata.get("type", "text")).lower()

        strong_name_signal = bool(
            snake_name == "id"
            or snake_name.endswith("_id")
            or snake_name in {
                "provider_number",
                "zip_code",
                "zipcode",
                "postal_code",
                "phone_number",
                "phonenumber",
                "flight",
                "state",
                "state_code",
                "country_code",
            }
            or name_tokens.intersection(
                {"city", "town", "province", "region", "status", "category", "style", "time"}
            )
            or any(token in snake_name for token in ("ounce", "weight", "unit", "size"))
        )
        strong_temporal_signal = bool(
            name_tokens.intersection(
                {"time", "date", "datetime", "timestamp", "year", "duration"}
            )
            or snake_name.endswith("_at")
        )
        strong_numeric_signal = bool(
            coarse_type == "numeric"
            or name_tokens.intersection({"abv", "amount", "price", "rate", "score", "average", "avg"})
            or snake_name.endswith("avg")
        )
        strong_closed_distribution = bool(
            unique_ratio <= 0.10 and top_coverage >= 0.30
        )
        strong_open_distribution = bool(unique_ratio >= 0.50)

        use_heuristic = (
            strong_name_signal
            or strong_temporal_signal
            or strong_numeric_signal
            or strong_closed_distribution
            or (
                strong_open_distribution
                and heuristic.archetype == "open_entity"
                and llm_archetype == "closed_enum"
            )
        )
        if use_heuristic and llm_archetype != heuristic.archetype:
            return (
                heuristic.archetype,
                f"profile_override:{llm_archetype}->{heuristic.archetype}",
            )
        return llm_archetype, None

    def _validated_mechanisms(
        self,
        archetype: str,
        payload: Dict[str, Any],
    ) -> List[str]:
        allowed = set(self.MECHANISMS_BY_ARCHETYPE[archetype])
        proposed = payload.get("possible_error_mechanisms") or []
        if not isinstance(proposed, list):
            proposed = []
        validated = []
        for mechanism in proposed:
            normalized = str(mechanism).strip()
            if normalized in allowed and normalized not in validated:
                validated.append(normalized)
        # Keep the complete stable vocabulary for coverage, while placing
        # mechanisms explicitly proposed by the LLM first.
        defaults = list(self.MECHANISMS_BY_ARCHETYPE[archetype])
        return validated + [item for item in defaults if item not in validated]

    def infer(self, column: str, metadata: Dict[str, Any]) -> ColumnSemantics:
        compact_metadata = {
            key: metadata.get(key)
            for key in (
                "type",
                "non_null_count",
                "null_count",
                "unique_count",
                "unique_ratio",
                "singleton_ratio",
                "top_value_coverage",
                "percent_marker_ratio",
                "sample_values",
                "normalized_top_values",
                "shape_distribution",
                "regex_candidates",
                "top_tokens",
            )
        }
        prompt = f"""Column name: {column}
Column profile:
{json.dumps(compact_metadata, ensure_ascii=False, default=str)}

Return this schema:
{{
  "archetype": "one allowed archetype",
  "confidence": 0.0,
  "open_set_score": 0.0,
  "structure_strength": 0.0,
  "canonicalization_need": 0.0,
  "possible_error_mechanisms": [],
  "rationale": []
}}"""

        response = self._call_llm(prompt, temperature=0.0)
        try:
            payload = self._extract_json_object(response)
            llm_archetype = str(payload.get("archetype", "")).strip()
            if llm_archetype not in self.ALLOWED_ARCHETYPES:
                raise ValueError(f"Unsupported archetype: {llm_archetype!r}")

            heuristic = self._heuristic_fallback(
                column,
                metadata,
                fallback_reason="profile_reference",
            )
            archetype, override_reason = self._reconcile_archetype(
                column,
                metadata,
                llm_archetype,
                heuristic,
            )
            mechanisms = self._validated_mechanisms(archetype, payload)

            rationale = payload.get("rationale") or []
            if isinstance(rationale, str):
                rationale = [rationale]
            if not isinstance(rationale, list):
                rationale = []
            rationale = [str(item) for item in rationale if str(item).strip()]
            rationale.append("scores_from_profile")
            if override_reason:
                rationale.append(override_reason)

            llm_confidence = self._clip_score(payload.get("confidence"), 0.5)
            confidence = (
                min(llm_confidence, 0.80)
                if override_reason else llm_confidence
            )

            # These values are measured or deterministic profile-derived
            # quantities. LLM-proposed scores are intentionally not trusted.
            profile_semantics = self._heuristic_fallback(
                column,
                metadata,
                fallback_reason="profile_scores",
            )
            if profile_semantics.archetype != archetype:
                profile_semantics = ColumnSemantics(
                    column=column,
                    archetype=archetype,
                    confidence=profile_semantics.confidence,
                    open_set_score=profile_semantics.open_set_score,
                    structure_strength=profile_semantics.structure_strength,
                    canonicalization_need={
                        "unit_measure": 0.95,
                        "geo_name": 0.80,
                        "numeric_measure": max(
                            0.45,
                            self._clip_score(metadata.get("percent_marker_ratio"), 0.0),
                        ),
                        "temporal_measure": 0.30,
                        "closed_enum": 0.40,
                        "identifier": 0.35,
                        "open_entity": 0.25,
                        "free_text": 0.20,
                    }[archetype],
                    possible_error_mechanisms=mechanisms,
                    rationale=[],
                )

            return ColumnSemantics(
                column=column,
                archetype=archetype,
                confidence=confidence,
                open_set_score=profile_semantics.open_set_score,
                structure_strength=profile_semantics.structure_strength,
                canonicalization_need=profile_semantics.canonicalization_need,
                possible_error_mechanisms=mechanisms,
                rationale=rationale,
            )
        except Exception as exc:
            return self._heuristic_fallback(
                column,
                metadata,
                fallback_reason=f"heuristic_fallback:{type(exc).__name__}",
            )


class CanonicalizationPlanningAgent(BaseAgent):
    """Plan column-specific canonicalization with a constrained repair DSL."""

    ALLOWED_OPERATIONS = {
        "regex_replace",
        "strip_suffix",
        "split_trailing_token",
        "truncate_at_marker",
        "normalize_surface",
    }
    ALLOWED_MECHANISMS = {
        "unit_variant",
        "suffix_pollution",
        "percent_decimal_mismatch",
        "geo_suffix_pollution",
        "encoding_artifact",
        "token_contamination",
        "case_variant",
        "whitespace_variant",
        "other_canonicalization",
    }

    def _get_system_prompt(self) -> str:
        return """You are a conservative canonicalization planning agent.
Infer column-specific conventions from the supplied profile; do not assume that
units, percent signs, suffixes, casing, or punctuation are errors merely because
they exist. A dominant surface form is presumptively valid. If the profile does
not establish a safer target convention, abstain.

Return exactly one JSON object and no prose. You may use only these operations:
- regex_replace: params {pattern, replacement, target_pattern}; pattern is
  searched within the value and replaced globally
- strip_suffix: params {suffix, numeric_scale, target_pattern}
- split_trailing_token: params {reference_column}
- truncate_at_marker: params {markers, suffix_reference_column}
- normalize_surface: params {mapping}; mapping keys and values are literal strings

target_pattern may be a full regular expression or one measured surface label:
numeric_percent_suffix, plain_numeric, numeric_with_suffix, lexical,
mixed_alphanumeric, other.

Parameter rules:
- For regex_replace, derive both pattern and replacement from measured minority
  and dominant tokens in the supplied profile.
- For strip_suffix, the suffix must be observed and the target_pattern must
  describe the transformed value, not the original value.
- For split_trailing_token and truncate_at_marker, name an actually supplied
  reference column; otherwise do not propose the operation.
- For normalize_surface, mapping entries must be supported by repeated observed
  variants and a clearly dominant target token.

Regexes and mappings must describe generic surfaces, never lists of row-specific
values. Do not emit executable code. Each operation must describe one source
surface and one target convention. Prefer multiple narrow operations over one
broad transformation. Scores are in [0,1]."""

    @staticmethod
    def _clip(value: Any, default: float = 0.0) -> float:
        try:
            return float(min(1.0, max(0.0, float(value))))
        except (TypeError, ValueError):
            return float(default)

    def _abstain(self, column: str, reason: str) -> Dict[str, Any]:
        return {
            "column": column,
            "applicable": False,
            "canonical_surface": "",
            "confidence": 0.0,
            "operations": [],
            "rationale": [reason],
            "risky_assumptions": [],
        }

    def _validate_operation(self, item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        operation = str(item.get("operation") or "").strip()
        mechanism = str(item.get("mechanism") or "").strip()
        if operation not in self.ALLOWED_OPERATIONS:
            return None
        if mechanism not in self.ALLOWED_MECHANISMS:
            # Mechanism names are descriptive evidence labels, not executable
            # behavior. Preserve a structurally valid operation under a neutral
            # label instead of silently discarding it when an upstream agent
            # uses a synonymous or newly coined mechanism name.
            mechanism = "other_canonicalization"
        params = item.get("params") or {}
        if not isinstance(params, dict):
            return None
        # Keep the DSL JSON-only and bounded. The executor performs deeper
        # operation-specific validation before applying any transformation.
        try:
            encoded_params = json.dumps(params, ensure_ascii=False, default=str)
        except Exception:
            return None
        if len(encoded_params) > 4000:
            return None
        rationale = item.get("rationale") or []
        if isinstance(rationale, str):
            rationale = [rationale]
        if not isinstance(rationale, list):
            rationale = []
        return {
            "mechanism": mechanism,
            "operation": operation,
            "params": params,
            "confidence": self._clip(item.get("confidence"), 0.0),
            "repairability": self._clip(item.get("repairability"), 0.0),
            "rationale": [str(value) for value in rationale if str(value).strip()],
        }

    def infer(self, column: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Column: {column}
Archetype: {profile.get('archetype', 'unknown')}
Observed canonicalization profile (computed only from the dirty table):
{json.dumps(profile, ensure_ascii=False, default=str)}

Return this schema:
{{
  "column": {json.dumps(column)},
  "applicable": false,
  "canonical_surface": "description of observed target convention",
  "confidence": 0.0,
  "operations": [
    {{
      "mechanism": "one allowed mechanism",
      "operation": "one allowed operation",
      "params": {{}},
      "confidence": 0.0,
      "repairability": 0.0,
      "rationale": []
    }}
  ],
  "rationale": [],
  "risky_assumptions": []
}}

Critical rules:
1. Never propose changing the dominant observed surface into a minority surface.
2. Percent signs may be canonical (for example a score) or contaminating (for
   example a decimal-rate column); decide from the observed surface ratios.
3. A column classified as unit_measure may still contain no stable unit grammar;
   abstain unless the profile supports a target convention.
4. Do not use clean/reference data; it is not supplied.
5. If uncertain, return applicable=false and operations=[]."""
        response = self._call_llm(prompt, max_tokens=3072, temperature=0.0)
        try:
            payload = ColumnArchetypeAgent._extract_json_object(response)
        except Exception as exc:
            return self._abstain(column, f"invalid_planner_response:{type(exc).__name__}")

        operations = []
        for item in payload.get("operations") or []:
            validated = self._validate_operation(item)
            if validated is not None:
                operations.append(validated)
        applicable = bool(payload.get("applicable")) and bool(operations)
        if not applicable:
            operations = []
        rationale = payload.get("rationale") or []
        if isinstance(rationale, str):
            rationale = [rationale]
        risky = payload.get("risky_assumptions") or []
        if isinstance(risky, str):
            risky = [risky]
        return {
            "column": column,
            "applicable": applicable,
            "canonical_surface": str(payload.get("canonical_surface") or ""),
            "confidence": self._clip(payload.get("confidence"), 0.0),
            "operations": operations,
            "rationale": [str(value) for value in rationale if str(value).strip()],
            "risky_assumptions": [str(value) for value in risky if str(value).strip()],
        }


class RuleReviewerAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return "You are a data quality reviewer. Rewrite or suggest a safer Python lambda rule. Return ONLY one lambda expression."

    @staticmethod
    def _extract_lambda(text: str) -> Optional[str]:
        if not text:
            return None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("lambda"):
                return line
        return None

    def suggest_rule_fix(
        self,
        column: str,
        metadata: Dict[str, Any],
        clean_rule_str: str,
        dirty_rule_str: str,
        stats: Dict[str, Any],
        samples: List[Dict[str, Any]],
    ) -> Optional[str]:
        column_type = metadata.get("type", "unknown")
        top_values = metadata.get("normalized_top_values") or metadata.get("top_values") or []
        prompt = "\n".join([
            "Review the current dirty rule and suggest a safer but less aggressive rewrite.",
            f"Column: {column}",
            f"Type: {column_type}",
            f"Stats: {json.dumps(stats, ensure_ascii=False)}",
            f"Top values: {json.dumps(top_values[:8], ensure_ascii=False)}",
            f"Samples: {json.dumps(samples, ensure_ascii=False)}",
            f"P_clean: {clean_rule_str}",
            f"P_dirty: {dirty_rule_str}",
            "Return only one Python lambda expression."
        ])
        response = self._call_llm(prompt, temperature=0.2)
        return self._extract_lambda(response)


class MissingAgent(BaseAgent):
    """Agent focused on detecting missing/null values with batch annotation."""

    def _get_system_prompt(self) -> str:
        """System prompt for missing value annotation."""
        return """You are a data completeness expert. Given a list of values from a column,
classify each as either MISSING_TOKEN (represents missing/empty data) or VALID (normal data value).
Common missing tokens include: "", "nan", "null", "none", "n/a", "na", "unknown", "empty", "xxxxx", "--", "-", "?"."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to detect missing values using batch annotation."""
        null_count = metadata.get('null_count', 0)
        missing_token_counts = metadata.get('missing_token_counts', {})
        observed_missing = sum(
            count for token, count in missing_token_counts.items()
            if token != "<NA>"
        )

        if null_count == 0 and observed_missing == 0:
            return []

        # Step 1: Annotate all unique values in batches
        annotations = self._annotate_all_values(column, metadata)

        # Step 2: Extract values labeled as MISSING_TOKEN
        missing_tokens = [v for v, label in annotations.items()
                          if label == "MISSING_TOKEN"]

        # Step 3: Build token pool (defaults + dominant + annotated)
        dominant_tokens = metadata.get('dominant_missing_tokens') or []
        token_pool = sorted(set(DEFAULT_MISSING_TOKENS + dominant_tokens + missing_tokens))

        # Step 4: Generate lambda rule
        # Use repr to ensure valid Python literal for list of strings
        token_literal = repr(token_pool)
        rule = (
            "lambda value: ("
            "(pd.isna(value) if hasattr(pd, 'isna') else value is None)"
            " or str(value).strip().lower() in " + token_literal +
            ")"
        )
        return [rule]

    def _annotate_all_values(self, column: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Annotate all unique values in batches using LLM.

        Returns:
            Dict mapping normalized value -> label ("MISSING_TOKEN" or "VALID")
        """
        # Build unique value -> count mapping
        unique_mapping = self._build_unique_value_mapping(metadata)

        if not unique_mapping:
            return {}

        # Sort by frequency (descending) and batch
        sorted_values = sorted(unique_mapping.items(), key=lambda x: -x[1])
        batches = _batch_unique_values(sorted_values, VALUE_BATCH_SIZE)

        # Limit number of batches
        batches = batches[:MAX_BATCHES_PER_COLUMN]

        # Annotate each batch
        all_labels = {}
        for batch in batches:
            batch_labels = self._annotate_batch(column, batch, metadata)
            all_labels.update(batch_labels)

        return all_labels

    def _build_unique_value_mapping(self, metadata: Dict[str, Any]) -> Dict[str, int]:
        """Build mapping of normalized unique values to their counts."""
        mapping = {}

        # Add from missing_token_counts (excluding <NA>)
        missing_counts = metadata.get('missing_token_counts', {})
        for token, count in missing_counts.items():
            if token != "<NA>":
                mapping[token] = count

        # Add from normalized_top_values
        normalized_top = metadata.get('normalized_top_values', [])
        for item in normalized_top:
            val = item.get('value', '')
            count = item.get('count', 1)
            if val and val not in mapping:
                mapping[val] = count

        return mapping

    def _annotate_batch(self, column: str, batch: List[Tuple[str, int]],
                        metadata: Dict[str, Any]) -> Dict[str, str]:
        """Annotate a single batch of values."""
        dominant = metadata.get('dominant_missing_tokens', [])
        null_count = metadata.get('null_count', 0)

        batch_data = [{"value": v, "count": c} for v, c in batch]

        prompt = f"""
For column '{column}', classify each value as MISSING_TOKEN or VALID.

Column info:
- null_count: {null_count}
- known_missing_tokens: {json.dumps(dominant, ensure_ascii=False)}

Batch values to classify:
{json.dumps(batch_data, ensure_ascii=False)}

Rules:
- MISSING_TOKEN: empty string, null, "nan", "none", "n/a", "na", "unknown", etc.
- VALID: any legitimate data value

Output JSON array:
[{{"value": "...", "label": "MISSING_TOKEN"}}, {{"value": "...", "label": "VALID"}}]
"""
        response = self._call_llm(prompt)

        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                items = json.loads(json_match.group())
                return {item.get('value', ''): item.get('label', 'VALID')
                        for item in items if item.get('value')}
        except json.JSONDecodeError:
            pass

        return {}


class TypoAgent(BaseAgent):
    """Agent focused on detecting typos in string columns with batch annotation."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to detect typos using batch annotation."""
        unique_count = metadata.get('unique_count', 0)

        if unique_count < 2:
            return []

        # Step 1: Annotate low-frequency values in batches
        annotations = self._annotate_low_frequency_values(column, metadata)

        # Step 2: Extract values labeled as LIKELY_TYPO
        typo_values = [v for v, label in annotations.items()
                       if label == "LIKELY_TYPO"]

        if not typo_values:
            return []

        # Step 3: Generate typo detection rules
        # Group typos by similarity to generate general rules
        rules = self._generate_typo_rules(typo_values, annotations, metadata)
        return rules

    def _annotate_low_frequency_values(self, column: str,
                                        metadata: Dict[str, Any]) -> Dict[str, str]:
        """Annotate low-frequency values to identify potential typos.

        Returns:
            Dict mapping value -> label ("LIKELY_TYPO" or "VALID_RARE")
        """
        # Get low-frequency unique values (count <= 3)
        low_freq_items = self._get_low_frequency_values(metadata)

        if not low_freq_items:
            return {}

        # Sort by frequency and batch
        sorted_values = sorted(low_freq_items, key=lambda x: -x[1])
        batches = _batch_unique_values(sorted_values, VALUE_BATCH_SIZE)
        batches = batches[:MAX_BATCHES_PER_COLUMN]

        # Annotate each batch
        all_labels = {}
        for batch in batches:
            batch_labels = self._annotate_typo_batch(column, batch, metadata)
            all_labels.update(batch_labels)

        return all_labels

    def _get_low_frequency_values(self, metadata: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Extract low-frequency values from metadata."""
        low_freq = []

        # From low_frequency_values
        low_freq_metadata = metadata.get('low_frequency_values', [])
        for item in low_freq_metadata:
            val = item.get('value', '')
            count = item.get('count', 1)
            if val and count <= 3:
                low_freq.append((val, count))

        # Also include singletons if not already there
        singleton_count = metadata.get('singleton_count', 0)
        normalized_top = metadata.get('normalized_top_values', [])
        for item in normalized_top:
            val = item.get('value', '')
            count = item.get('count', 1)
            if val and count == 1 and val not in [v for v, _ in low_freq]:
                low_freq.append((val, 1))

        return low_freq

    def _annotate_typo_batch(self, column: str, batch: List[Tuple[str, int]],
                              metadata: Dict[str, Any]) -> Dict[str, str]:
        """Annotate a single batch for potential typos."""
        top_values = list(metadata.get('top_values', {}).keys())[:10]
        normalized_top = [item.get('value', '') for item in metadata.get('normalized_top_values', [])[:10]]

        batch_data = [{"value": v, "count": c} for v, c in batch]

        prompt = f"""
For column '{column}', annotate the following low-frequency values as potential typos.

High-frequency reference values: {json.dumps(top_values, ensure_ascii=False)}
Normalized high-frequency values: {json.dumps(normalized_top, ensure_ascii=False)}

Low-frequency values to annotate in this batch:
{json.dumps(batch_data, ensure_ascii=False)}

Annotation rules:
- "LIKELY_TYPO": value resembles a high-frequency one but has obvious spelling issues (character substitution, insertion/deletion, transposition, case error)
- "VALID_RARE": value is legitimate and reasonable, just infrequent

Output JSON array:
[{{"value": "...", "label": "LIKELY_TYPO"}}, {{"value": "...", "label": "VALID_RARE"}}]
"""
        response = self._call_llm(prompt)

        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                items = json.loads(json_match.group())
                return {item.get('value', ''): item.get('label', 'VALID_RARE')
                        for item in items if item.get('value')}
        except json.JSONDecodeError:
            pass

        return {}

    def _generate_typo_rules(self, typo_values: List[str],
                              annotations: Dict[str, str],
                              metadata: Dict[str, Any]) -> List[str]:
        """Generate typo detection rules from annotated values."""
        rules = []

        if not typo_values:
            return rules

        # Group typos by length first
        length_groups: Dict[int, List[str]] = {}
        for v in typo_values:
            length_groups.setdefault(len(v), []).append(v)

        # Generate rules for each length group
        for length, values in length_groups.items():
            if len(values) >= 1:
                # Create a pattern-based rule for this length
                # Try to identify common character patterns
                rule = self._build_typo_rule_for_values(values, length)
                if rule:
                    rules.append(rule)

        # Also add a general rarity rule for very rare values
        if len(typo_values) > 0:
            rare_values_literal = repr(typo_values[:20])
            rule = f"lambda value: str(value).strip().lower() in {rare_values_literal}"
            rules.append(rule)

        return rules[:5]  # Limit to 5 rules

    def _build_typo_rule_for_values(self, values: List[str], target_length: int) -> str:
        """Build a typo detection rule for a group of values with same length."""
        if not values:
            return ""

        # Check if all values have similar structure
        all_digit = all(v.isdigit() for v in values if v)
        all_alpha = all(v.isalpha() for v in values if v)
        
        values_literal = repr(values)

        if all_digit:
            return f"lambda value: (str(value).isdigit() and len(str(value)) == {target_length} and str(value).strip().lower() in {values_literal})"
        elif all_alpha:
            return f"lambda value: (str(value).isalpha() and len(str(value)) == {target_length} and str(value).strip().lower() in {values_literal})"

        return f"lambda value: str(value).strip().lower() in {values_literal}"


class PatternAgent(BaseAgent):
    """Agent focused on detecting pattern violations using PatternExplorer."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to validate patterns using PatternExplorer."""
        # Check if we have a DataFrame (needed for PatternExplorer)
        if not hasattr(self, 'df') or self.df is None:
            return self._fallback_pattern_rules(column, metadata)

        try:
            # Use PatternExplorer to get PatternSpec
            explorer = PatternExplorer(base_url=self.base_url, model=self.model)
            spec = explorer.explore(column, self.df, metadata)

            # Cache spec in metadata for CleanPatternLegislator to reuse
            metadata['_cached_pattern_spec'] = spec

            # Generate P_dirty rule (value does NOT match any pattern)
            rules = self._generate_dirty_rule_from_spec(column, spec)
            return rules
        except Exception as e:
            print(f"  [PatternLegislator] PatternExplorer failed: {e}")
            return self._fallback_pattern_rules(column, metadata)

    def _generate_dirty_rule_from_spec(self, column: str, spec: PatternSpec) -> List[str]:
        """Generate P_dirty rule from PatternSpec."""
        # Filter high-quality patterns
        high_quality = spec.get_high_quality_patterns(PATTERN_COVERAGE_THRESHOLD)

        if not high_quality:
            return []

        # Build combined regex
        regex_patterns = [p.regex for p in high_quality]
        combined_regex = "|".join(f"(?:{r})" for r in regex_patterns)

        # Generate P_dirty rule: value does NOT match any pattern
        rule = (
            f"lambda value: (value is not None and "
            f"not bool(re.match(r'^{combined_regex}$', str(value).strip()))"
            f")"
        )
        return [rule]

    def _fallback_pattern_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate fallback pattern validation rules (existing implementation)."""
        rules = []
        sample_values = metadata.get('sample_values', [])

        if not sample_values:
            return []

        # Analyze sample values for patterns
        sample_str = [str(v) for v in sample_values[:10]]

        # Check for digit-only pattern - flag values that DON'T match
        if all(v.isdigit() for v in sample_str):
            digit_len = len(sample_str[0]) if sample_str else 5
            rules.append(f"lambda value: not (str(value).isdigit() and len(str(value)) == {digit_len}) if value else False")

        # Check for mixed alphanumeric - flag empty values
        if any(not v.isdigit() for v in sample_str):
            rules.append("lambda value: len(str(value).strip()) == 0 if value else True")

        return rules


class OutlierAgent(BaseAgent):
    """Agent focused on detecting numeric outliers."""

    def _get_system_prompt(self) -> str:
        """System prompt for numeric anomaly and outlier detection."""
        return """You are a domain expert and data quality analyst.
Your role is to generate Python lambda functions that identify numeric values that are logically or statistically impossible or highly improbable.
PRIORITIZE business common sense and domain knowledge (e.g., ages shouldn't be 200, prices shouldn't be negative).
Use statistical thresholds (mean, std dev, IQR) ONLY when clear business logic cannot be inferred from the column name and data samples.
Return ONLY the lambda functions, one per line. Each function should return True when a value is an outlier/invalid, False otherwise."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to detect numeric outliers."""
        numeric_count = metadata.get('numeric_count', 0)
        min_val = metadata.get('min')
        max_val = metadata.get('max')
        mean_val = metadata.get('mean')
        std_val = metadata.get('std')
        quantiles = metadata.get('quantiles', {})
        iqr = metadata.get('iqr')
        mad = metadata.get('mad')
        extreme_values = metadata.get('extreme_numeric_values', {})
        non_numeric_count = metadata.get('non_numeric_count', 0)
        non_numeric_examples = metadata.get('non_numeric_examples', [])[:5]
        
        if numeric_count == 0 or min_val is None or max_val is None:
            return []

        prompt = f"""
For column '{column}', generate up to 5 Python lambda functions to detect numeric outliers.

Column statistics:
- Min: {min_val}
- Max: {max_val}
- Mean: {mean_val}
- Valid numeric values: {numeric_count}
- Std Dev: {std_val}
- Quantiles: {json.dumps(quantiles)}
- IQR: {iqr}, MAD: {mad}
- Extreme samples: {json.dumps(extreme_values)}
- Non-numeric count: {non_numeric_count}, examples: {json.dumps(non_numeric_examples)}

Instructions:
1. PRIORITIZE business common sense: If the column name '{column}' implies a known domain (e.g., year, age, percentage), set hard logical bounds.
2. Use statistics ONLY as a fallback: Use mean/std or IQR for generic numeric columns where business meaning is unclear.
3. Avoid redundant rules: Don't just repeat the min/max if they look normal.
4. Each rule should be a standalone lambda function returning True when a value is an OUTLIER/INVALID.

Return ONLY lambda functions, one per line.

Examples:
lambda value: not (0 <= float(value) <= 120) if value else False  # for 'age'
lambda value: float(value) <= 0 if value else False               # for 'price'
lambda value: abs(float(value) - {mean_val}) > 3 * {std_val} if value else False # fallback statistical rule
"""
        rules_text = self._call_llm(prompt)
        
        if not rules_text or "lambda" not in rules_text.lower():
            return self._fallback_outlier_rules(min_val, max_val)
        
        rules = [line.strip() for line in rules_text.split('\n') if line.strip().startswith('lambda')]
        return rules if rules else self._fallback_outlier_rules(min_val, max_val)

    def _fallback_outlier_rules(self, min_val: float, max_val: float) -> List[str]:
        """Generate fallback outlier detection rules."""
        rules = []

        # Rule 1: Flag values that exceed a reasonable threshold
        if max_val and max_val > 1000:  # Likely outliers
            rules.append(f"lambda value: float(value) > {max_val * 0.5} if value else False")

        # Rule 2: Flag negative values for typical metrics
        if min_val and min_val < 0:
            rules.append("lambda value: float(value) < 0 if value else False")

        return rules


class LogicAgent(BaseAgent):
    """Agent focused on logical consistency checks (cross-column rules)."""

    def _get_system_prompt(self) -> str:
        """System prompt for logical consistency and cross-column validation."""
        return """You are a data integrity expert specializing in cross-column logical consistency and business rule validation.
Your role is to generate Python lambda functions that verify relationships between multiple columns (e.g., temporal ordering, value dependencies).
You understand domain logic, temporal constraints, and referential integrity requirements.
Return ONLY the lambda functions, one per line. Each function accepts a row (dict) and returns True for logically consistent data.
Be STRINGENT - only accept values that are definitely ACCURATE and follow the expected domain format. Any deviation should be rejected.
"""


class DirtyExampleAgent(BaseAgent):
    def generate_dirty_examples(
        self,
        column: str,
        metadata: Dict[str, Any],
        clean_seeds: List[Any],
        error_family: str,
        max_examples: int = 50,
    ) -> List[Dict[str, Any]]:
        column_type = metadata.get("type", "text")
        prompt_lines = [
            f"Generate synthetic DIRTY examples for column '{column}'.",
            f"Error family: {error_family}",
            f"Column type: {column_type}",
            f"Clean seed examples: {json.dumps(clean_seeds[:20], ensure_ascii=False)}",
            f"Metadata: {json.dumps({k: v for k, v in metadata.items() if k in ['min','max','mean','std','quantiles','top_values','pattern_analysis','length_distribution','shape_distribution']}, ensure_ascii=False)}",
            "",
            "Return JSON array. Each item must have:",
            "  value: the dirty value",
            "  reason: short text reason",
            "The error type is implicitly given by error_family and should not be repeated.",
            "",
            "Example:",
            '[{"value": "N/A", "reason": "placeholder missing token"}, {"value": "", "reason": "empty string"}]',
        ]
        prompt = "\n".join(prompt_lines)
        response = self._call_llm(prompt, temperature=0.3)
        examples: List[Dict[str, Any]] = []
        try:
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                items = json.loads(match.group())
                if isinstance(items, list):
                    for item in items[:max_examples]:
                        val = item.get("value")
                        reason = item.get("reason") or ""
                        examples.append(
                            {
                                "value": val,
                                "label": 1,
                                "error_family": error_family,
                                "reason": reason,
                            }
                        )
        except Exception:
            examples = []
        if examples:
            return examples
        return self._fallback_dirty_examples(column, metadata, clean_seeds, error_family, max_examples)

    def _fallback_dirty_examples(
        self,
        column: str,
        metadata: Dict[str, Any],
        clean_seeds: List[Any],
        error_family: str,
        max_examples: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        column_type = metadata.get("type", "text")
        seeds = [v for v in clean_seeds if v is not None]
        seeds = seeds[: max_examples] if seeds else seeds
        if error_family == "missing":
            tokens = metadata.get("dominant_missing_tokens") or []
            if not tokens:
                tokens = ["", "nan", "none", "null", "n/a", "na", "unknown"]
            for t in tokens[:max_examples]:
                results.append(
                    {
                        "value": t,
                        "label": 1,
                        "error_family": error_family,
                        "reason": "common missing token",
                    }
                )
        elif error_family == "outlier":
            min_val = metadata.get("min")
            max_val = metadata.get("max")
            mean_val = metadata.get("mean")
            std_val = metadata.get("std") or 1.0
            if column_type == "numeric" and min_val is not None and max_val is not None:
                extreme_low = float(min_val) - 5 * float(std_val or 1.0)
                extreme_high = float(max_val) + 5 * float(std_val or 1.0)
                results.append(
                    {
                        "value": extreme_low,
                        "label": 1,
                        "error_family": error_family,
                        "reason": "extreme low outlier",
                    }
                )
                results.append(
                    {
                        "value": extreme_high,
                        "label": 1,
                        "error_family": error_family,
                        "reason": "extreme high outlier",
                    }
                )
                if mean_val is not None:
                    results.append(
                        {
                            "value": float(mean_val) * 10.0,
                            "label": 1,
                            "error_family": error_family,
                            "reason": "order-of-magnitude error",
                        }
                    )
            else:
                results.append(
                    {
                        "value": "INVALID_OUTLIER",
                        "label": 1,
                        "error_family": error_family,
                        "reason": "synthetic categorical outlier",
                    }
                )
        elif error_family == "pattern":
            pattern_hint = (metadata.get("pattern_analysis") or "").lower()
            base = None
            if seeds:
                base = str(seeds[0])
            if base:
                results.append(
                    {
                        "value": base + "X",
                        "label": 1,
                        "error_family": error_family,
                        "reason": "extra character",
                    }
                )
                results.append(
                    {
                        "value": base[:-1] if len(base) > 1 else "",
                        "label": 1,
                        "error_family": error_family,
                        "reason": "truncated pattern",
                    }
                )
            elif "zip" in pattern_hint or "postal" in pattern_hint:
                results.append(
                    {
                        "value": "12-345",
                        "label": 1,
                        "error_family": error_family,
                        "reason": "wrong separator",
                    }
                )
            else:
                results.append(
                    {
                        "value": "???",
                        "label": 1,
                        "error_family": error_family,
                        "reason": "generic malformed pattern",
                    }
                )
        trimmed: List[Dict[str, Any]] = []
        for ex in results:
            if len(trimmed) >= max_examples:
                break
            trimmed.append(ex)
        return trimmed

    def generate_rules(self, row_data: Dict[str, Any], all_metadata: Dict[str, Any]) -> List[str]:
        """
        Generate rules for logical consistency across columns.

        This method can work in two modes:
        1. If relationship_profiles exist in metadata, use them (backward compatibility)
        2. If no relationship_profiles, intelligently infer relationships from metadata
        """
        rules = []
        seen_constraints = set()
        max_rules = 10

        # Check if we have existing relationship profiles
        has_existing_profiles = any(
            metadata.get('relationship_profiles')
            for metadata in all_metadata.values()
        )

        if has_existing_profiles:
            # Use existing relationship profiles (backward compatibility)
            for column, metadata in all_metadata.items():
                profiles = metadata.get('relationship_profiles', [])
                if not profiles:
                    continue

                for profile in profiles:
                    constraint_key = (column, profile.get('other_column'), profile.get('type'))
                    if constraint_key in seen_constraints:
                        continue

                    prompt = self._build_constraint_prompt(column, profile, all_metadata)
                    rule_text = self._call_llm(prompt)
                    lambda_rule = self._extract_lambda(rule_text)
                    if lambda_rule:
                        rules.append(lambda_rule)
                        seen_constraints.add(constraint_key)

                    if len(rules) >= max_rules:
                        return rules
        else:
            # Intelligently infer relationships from metadata
            inferred_rules = self._infer_relationships_from_metadata(all_metadata)
            rules.extend(inferred_rules)

        if not rules:
            return self._default_temporal_rules(all_metadata)

        return rules[:max_rules]

    def _infer_relationships_from_metadata(self, all_metadata: Dict[str, Any]) -> List[str]:
        """
        Intelligently infer cross-column relationships from metadata using LLM.

        Returns:
            List of lambda functions for cross-column validation
        """
        rules = []

        # Build a summary of all columns
        column_summaries = []
        for column, metadata in all_metadata.items():
            col_type = metadata.get('type', 'unknown')
            sample_values = metadata.get('sample_values', [])
            top_values = metadata.get('top_values', {})
            unique_count = metadata.get('unique_count', 0)

            # Get a few representative values
            if sample_values:
                representative = sample_values[:5]
            elif top_values:
                representative = list(top_values.keys())[:5]
            else:
                representative = []

            column_summaries.append({
                'column': column,
                'type': col_type,
                'unique_count': unique_count,
                'samples': representative
            })

        # Build prompt for LLM to analyze relationships
        prompt_lines = [
            "Analyze the following columns and identify logical relationships between them.",
            "For each relationship you identify, generate a Python lambda function that validates it.",
            "",
            "Columns in the dataset:"
        ]

        for col in column_summaries:
            prompt_lines.append(
                f"- {col['column']} (type: {col['type']}, unique values: {col['unique_count']}, samples: {col['samples']})"
            )

        prompt_lines.extend([
            "",
            "Instructions:",
            "1. Identify meaningful cross-column relationships (e.g., temporal ordering, geographic consistency, value dependencies)",
            "2. For each relationship, generate a lambda function: lambda row: <validation_logic>",
            "3. Return ONLY the lambda functions, one per line",
            "4. Handle None/empty values gracefully",
            "5. Focus on relationships that make domain sense",
            "",
            "Examples of valid relationships:",
            "- If 'AdmissionDate' and 'DischargeDate' exist: check discharge >= admission",
            "- If 'State' and 'StateAvg' exist: check StateAvg starts with State_",
            "- If 'City' and 'CityInState' exist: check City appears in CityInState",
            "",
            "Return only lambda functions:"
        ])

        prompt = "\n".join(prompt_lines)

        # Call LLM to get relationship rules
        response = self._call_llm(prompt)

        if not response or "lambda" not in response.lower():
            return []

        # Extract lambda functions from response
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("lambda"):
                rules.append(line)
            elif "lambda" in line.lower():
                # Extract lambda from the line
                idx = line.lower().index("lambda")
                candidate = line[idx:].strip()
                if candidate.lower().startswith("lambda"):
                    rules.append(candidate)

        return rules

    def _build_constraint_prompt(self, column: str, profile: Dict[str, Any],
                                 metadata_map: Dict[str, Any]) -> str:
        """Build an LLM prompt for a specific cross-column constraint."""
        other_column = profile.get('other_column')
        constraint_type = profile.get('type')
        description = profile.get('description', '')
        violation_rate = profile.get('violation_rate')
        top_cooccurrences = profile.get('top_cooccurrences', [])[:5]

        column_meta = metadata_map.get(column, {})
        other_meta = metadata_map.get(other_column, {})

        prompt_lines = [
            f"Generate a Python lambda row function that enforces the constraint between '{column}' ({column_meta.get('type')})",
            f"and '{other_column}' ({other_meta.get('type')}).",
            f"Constraint type: {constraint_type}",
            f"Description: {description}",
            f"Violation rate observed in data: {violation_rate:.4f}" if violation_rate is not None else "",
            "Top co-occurring value pairs (value ↔ other_value, ratio):"
        ]

        for pair in top_cooccurrences:
            prompt_lines.append(
                f"- {pair.get('value')} ↔ {pair.get('other_value')} ({pair.get('ratio', 0):.2%}, count={pair.get('count')})"
            )

        prompt_lines.append(
            "\nRequirements:\n"
            "1. Return True when the constraint holds or when required values are missing.\n"
            "2. Return False only for confident violations based on the described constraint.\n"
            "3. Handle None/empty strings without raising exceptions."
        )
        prompt_lines.append("\nReturn ONLY the lambda definition (e.g., lambda row: <expression>).")

        return "\n".join([line for line in prompt_lines if line])

    def _extract_lambda(self, response: str) -> str:
        """Extract the first lambda expression from the LLM response."""
        if not response or "lambda" not in response.lower():
            return None

        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("lambda"):
                return line
            if "lambda" in line.lower():
                idx = line.lower().index("lambda")
                candidate = line[idx:]
                if candidate.lower().startswith("lambda"):
                    return candidate
        return None

    def _default_temporal_rules(self, all_metadata: Dict[str, Any]) -> List[str]:
        """Fallback temporal consistency rules (legacy behaviour)."""
        rules = []
        if 'AdmissionDate' in all_metadata and 'DischargeDate' in all_metadata:
            rules.append(
                "lambda row: ("
                "row.get('DischargeDate') is None or row.get('AdmissionDate') is None or ("
                "pd.to_datetime(row['DischargeDate']) >= pd.to_datetime(row['AdmissionDate'])"
                "))"
            )
        return rules


class CleanCompletenessAgent(BaseAgent):
    """Agent specialized in Completeness - ensuring values are present and complete."""

    clean_agent_name = "completeness"

    def _get_system_prompt(self) -> str:
        """System prompt for completeness validation."""
        return """You are a data completeness expert specializing in ensuring data completeness and presence.
Your role is to generate Python lambda functions that confirm when a value is definitely COMPLETE (present and non-missing).
You understand various representations of missing data (None, NaN, empty strings, 'N/A', 'null', etc.) and know that clean data should NOT contain these placeholders.
Be STRINGENT - only accept values that are definitely ACCURATE and follow the expected domain format. Any deviation should be rejected.
Return ONLY the lambda functions, one per line. Each function should return True for complete/present values."""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to ensure data completeness."""
        dominant_tokens = metadata.get('dominant_missing_tokens') or []
        null_count = metadata.get('null_count', 0)
        observed_missing = sum(
            count for token, count in metadata.get('missing_token_counts', {}).items()
            if token != "<NA>"
        )

        if null_count == 0 and observed_missing == 0:
            # Still generate a completeness check rule
            pass

        token_pool = sorted(set(DEFAULT_MISSING_TOKENS + dominant_tokens))
        token_literal = repr(token_pool)

        rule = (
            "lambda value, row=None: ("
            "value is not None and "
            "str(value).strip().lower() not in " + token_literal +
            ")"
        )
        return [rule]


class CleanAccuracyAgent(BaseAgent):
    """Agent specialized in Accuracy - ensuring values fall into reasonable ranges."""

    clean_agent_name = "accuracy"

    def _get_system_prompt(self) -> str:
        """System prompt for accuracy validation."""
        return """You are a data accuracy expert. Generate Python lambda functions that return True for NORMAL/VALID values.
Rules:
- Return True only when value is definitely accurate and reasonable
- Be permissive for edge cases - better to accept borderline values than reject valid data
- Return False for clearly malformed or impossible values
Return ONLY lambda functions: lambda value, row=None: <expression>"""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to ensure data accuracy."""
        col_type = metadata.get('type', 'text')
        min_val = metadata.get('min')
        max_val = metadata.get('max')
        mean_val = metadata.get('mean')
        std_val = metadata.get('std')
        top_values = metadata.get('top_values', {})
        unique_count = metadata.get('unique_count', 0)
        numeric_count = metadata.get('numeric_count', 0)

        # Build top values with frequencies (as percentages)
        if top_values:
            total_count = sum(top_values.values())
            top_with_freq = [(str(k), f"{v/total_count*100:.3f}%") for k, v in list(top_values.items())[:20]]
        else:
            top_with_freq = []
        sample_values = metadata.get('sample_values', [])

        prompt = f"""
For column '{column}', generate up to 5 Python lambda functions to validate data ACCURACY.

Column metadata:
- Type: {col_type}
- Unique count: {unique_count}
- Top categorical values (with %): {json.dumps(top_with_freq, ensure_ascii=False)}
- Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}
- Numeric count: {numeric_count}

IMPORTANT GUIDELINES:
1. For categorical: look for characteristic patterns (value structure, format, typical values) - use frequency to identify dominant patterns
2. For numeric: check reasonable value ranges based on statistics with tolerance
3. For text: check reasonable length and character constraints
4. NOTE: Top categorical values are **not the only valid values**
5. Be PERMISSIVE - accept all values that look reasonably valid for the domain. Do not overfit to the top values provided. Only reject values that are clearly malformed or impossible.

Return ONLY lambda functions, one per line. Format:
lambda value, row=None: <expression>
"""

        rules_text = self._call_llm(prompt)

        if not rules_text or "lambda" not in rules_text.lower():
            # Fallback to deterministic rules
            return self._fallback_accuracy_rules(col_type, min_val, max_val, sample_values, top_values)

        # Parse lambda functions from response
        rules = [line.strip() for line in rules_text.split('\n') if line.strip().startswith('lambda')]
        return rules if rules else self._fallback_accuracy_rules(col_type, min_val, max_val, sample_values, top_values)

    def _fallback_accuracy_rules(self, col_type: str, min_val: float, max_val: float,
                                  sample_values: List[Any], top_values: Dict) -> List[str]:
        """Generate fallback accuracy validation rules."""
        rules = []

        if col_type == 'numeric' and min_val is not None and max_val is not None:
            tolerance = max(1.0, abs(max_val - min_val) * 0.05 or 1.0)
            lower_bound = min_val - tolerance
            upper_bound = max_val + tolerance
            rules.append(
                "lambda value, row=None: ("
                "value is not None and "
                f"{lower_bound} <= safe_float(value) <= {upper_bound}"
                ")"
            )
        else:
            sample_vals = sample_values or list(top_values.keys())
            normalized = [len(str(val).strip()) for val in sample_vals if val is not None and str(val).strip()]
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


class CleanPatternAgent(BaseAgent):
    """Agent specialized in Pattern Consistenc - ensuring values respect known patterns."""

    clean_agent_name = "pattern_consistency"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to ensure pattern consistency using PatternExplorer."""
        # Check if we have a DataFrame (needed for PatternExplorer)
        if not hasattr(self, 'df') or self.df is None:
            return self._generate_rules_via_llm(column, metadata)

        try:
            # Use cached PatternSpec if available (shared from PatternLegislator)
            if '_cached_pattern_spec' in metadata:
                spec = metadata['_cached_pattern_spec']
            else:
                # Use PatternExplorer to get PatternSpec
                explorer = PatternExplorer(base_url=self.base_url, model=self.model)
                spec = explorer.explore(column, self.df, metadata)

            # Generate P_clean rule (value matches known patterns)
            rules = self._generate_clean_rule_from_spec(column, spec)
            if rules:
                return rules
        except Exception as e:
            print(f"  [CleanPatternLegislator] PatternExplorer failed: {e}")

        # Fallback to LLM-based generation
        return self._generate_rules_via_llm(column, metadata)

    def _generate_rules_via_llm(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules via LLM (fallback when PatternExplorer unavailable)."""
        sample_values = metadata.get('sample_values', [])
        pattern_analysis = metadata.get('pattern_analysis', '')
        shape_distribution = metadata.get('shape_distribution', [])[:8]
        length_distribution = metadata.get('length_distribution', [])[:8]
        top_values = metadata.get('top_values', {})

        prompt = f"""
For column '{column}', generate up to 5 Python lambda functions describing NORMAL PATTERNS.

Key Point: You are defining what CORRECT values look like, NOT listing error types.
- Return True for values matching the expected normal pattern
- Return False for abnormal values (no need to enumerate all errors)
- Be permissive - accept all reasonably valid values

Column metadata:
- Pattern analysis: {pattern_analysis}
- Shape distribution: {shape_distribution}
- Length distribution: {length_distribution}
- Top values: {list(top_values.keys())[:10]}

Return ONLY lambda functions, one per line. Format:
lambda value, row=None: <expression>

Example for ZIP code:
lambda value, row=None: bool(re.match(r'^\\d{5}$', str(value).strip())) if value is not None else False

Example for phone:
lambda value, row=None: bool(re.match(r'^\\d{10}$', re.sub(r'\\D', '', str(value)))) if value is not None else False
"""

        rules_text = self._call_llm(prompt)

        if not rules_text or "lambda" not in rules_text.lower():
            return self._fallback_pattern_rules(column, metadata)

        # Parse lambda functions from response
        rules = [line.strip() for line in rules_text.split('\n') if line.strip().startswith('lambda')]
        return rules if rules else self._fallback_pattern_rules(column, metadata)

    def _generate_clean_rule_from_spec(self, column: str, spec: PatternSpec) -> List[str]:
        """Generate P_clean rule from PatternSpec."""
        # Filter high-quality patterns
        high_quality = spec.get_high_quality_patterns(PATTERN_COVERAGE_THRESHOLD)

        if not high_quality:
            return []

        # Build combined regex
        regex_patterns = [p.regex for p in high_quality]
        combined_regex = "|".join(f"(?:{r})" for r in regex_patterns)

        # Generate P_clean rule: value matches expected pattern
        rule = (
            f"lambda value, row=None: (value is not None and "
            f"bool(re.match(r'^{combined_regex}$', str(value).strip()))"
            f")"
        )
        return [rule]

    def _fallback_pattern_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate fallback pattern validation rules."""
        rules = []
        pattern_hint = (metadata.get('pattern_analysis') or "").lower()
        column_lower = column.lower()
        sample_values = metadata.get('sample_values') or list((metadata.get('top_values') or {}).keys())

        if 'zip' in column_lower or 'postal' in column_lower or '5-digit numeric' in pattern_hint:
            rules.append("lambda value, row=None: bool(re.match(r'^\\d{5}$', str(value).strip())) if value is not None else False")
        elif '10-digit numeric' in pattern_hint or 'phone' in column_lower:
            rules.append("lambda value, row=None: bool(re.match(r'^\\d{10}$', re.sub(r'\\D', '', str(value)))) if value is not None else False")
        elif 'pattern' in pattern_hint and 'alphanumeric' in pattern_hint:
            rules.append("lambda value, row=None: bool(re.match(r'^[A-Za-z0-9\\-]+$', str(value).strip())) if value is not None else False")
        else:
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


class CleanRelationshipAgent(BaseAgent):
    """Agent specialized in Column Relationship - ensuring intra-row column relationship constraints."""

    clean_agent_name = "column_relationship"

    def _get_system_prompt(self) -> str:
        """System prompt for column relationship validation."""
        return """You are a relationship validation expert. Generate lambda functions confirming NORMAL cross-column relationships.
Rules:
- Return True when relationships are valid (referential integrity, temporal ordering, etc.)
- You CAN use the 'row' parameter to access other columns (e.g., row.get('country'), row.get('date'))
- Return False for abnormal relationships or when unsure
Return ONLY lambda functions: lambda value, row=None: <expression>"""

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate rules to ensure column relationship consistency."""
        constraints = metadata.get('relationship_constraints') or []
        sample_values = metadata.get('sample_values', [])
        top_values = metadata.get('top_values', {})

        if not constraints:
            # Try to infer relationships from metadata
            prompt = f"""
For column '{column}', generate up to 5 Python lambda functions validating NORMAL COLUMN RELATIONSHIPS.

You CAN use the 'row' parameter to access other columns for conditional validation:
- Example: row.get('country'), row.get('date'), row.get('variant_id')

Look for hints about relationships in the column name or values.
Generate rules that confirm valid inter-column dependencies.

Return ONLY lambda functions, one per line. Format:
lambda value, row=None: <expression>
"""

            rules_text = self._call_llm(prompt)

            if not rules_text or "lambda" not in rules_text.lower():
                return []

            # Parse lambda functions from response
            rules = [line.strip() for line in rules_text.split('\n') if line.strip().startswith('lambda')]
            return rules

        # Use explicit constraints
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


class TableRelationshipAgent(BaseAgent):
    """Propose semantic cross-row functional dependencies for one table."""

    def _get_system_prompt(self) -> str:
        return (
            "You are a data quality expert. Select only semantically meaningful "
            "cross-row functional dependencies from the supplied candidate summaries. "
            "Do not write Python, lambda functions, SQL, or explanations outside JSON."
        )

    def infer_cross_row_fd_candidates(
        self,
        table_summary: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        prompt = "\n".join([
            "A candidate X -> Y means rows sharing X should normally share Y.",
            "Select only relationships that are meaningful in the table domain, not merely correlated.",
            "Choose the entity-to-attribute direction. Do not treat a time, measurement, or other reusable attribute as an identifier merely because it is correlated with a record identifier.",
            "For determinant_granularity, use instance only when X identifies one concrete record, event, or occurrence. Use recurring_entity when X identifies a reusable entity, service, product, customer, template, or other entity that can occur in multiple instances. Use unknown when the table semantics do not establish either.",
            "For dependent_variability, use invariant only when Y is a stable attribute of X. Use instance_varying when Y can normally vary across occurrences of the same X. Use unknown when the semantics do not establish either.",
            "Operational outcomes, observations, measurements, recorded timestamps, and realized values are normally instance_varying. Do not call them invariant merely because the observed table values are mostly consistent; label them invariant only when the schema establishes that they are a stable property of the reusable entity.",
            "For example, a reusable service, product, customer, or template can have many instances. Its event outcome or observation requires an occurrence-level key, whereas a stable specification or classification can be invariant.",
            "An exact functional dependency is unsafe when a recurring_entity determines an instance_varying attribute. Do not propose such a rule; return no rule instead when the granularity is uncertain.",
            "Return exactly one JSON object with this schema:",
            '{"rules":[{"rule_id":"fd_short_name","scope":"cross_row","type":"functional_dependency","determinant_columns":["X"],"dependent_column":"Y","determinant_granularity":"instance|recurring_entity|unknown","dependent_variability":"invariant|instance_varying|unknown","claim":"Rows with the same X normally share Y"}]}',
            "Use only candidate column names supplied below. Return an empty rules array when none is justified.",
            "Candidate summaries:",
            json.dumps(table_summary, ensure_ascii=False),
        ])
        response = self._call_llm(
            prompt,
            max_tokens=512,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        payload: Any = {}
        try:
            candidate = response.strip()
            fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.I)
            if fenced:
                candidate = fenced.group(1).strip()
            object_match = re.search(r"\{[\s\S]*\}", candidate)
            if object_match:
                candidate = object_match.group(0)
            payload = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        rules = payload.get("rules") if isinstance(payload, dict) else []
        if not isinstance(rules, list):
            rules = []
        return [dict(rule) for rule in rules if isinstance(rule, dict)], self.get_last_prompt_info()

class CleanRuleReflectionAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are a data quality expert specializing in refining CLEAN validation rules.
Your job is to slightly rewrite an existing clean rule so that:
- It returns True for clearly CLEAN/normal values
- It returns False for clearly DIRTY/abnormal values
You are given the current rule and examples where it misbehaves."""

    @staticmethod
    def _extract_lambda(text: str) -> Optional[str]:
        if not text:
            return None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("lambda"):
                return line
        return None

    def refine_clean_rule(
        self,
        column: str,
        metadata: Dict[str, Any],
        family: str,
        rule_str: str,
        clean_mis_examples: List[Dict[str, Any]],
        dirty_mis_examples: List[Dict[str, Any]],
    ) -> Optional[str]:
        column_type = metadata.get("type", "unknown")
        top_values = metadata.get("normalized_top_values") or metadata.get("top_values") or []
        prompt_parts = [
            f"Column: {column}",
            f"Column type: {column_type}",
            f"Error family: {family}",
            f"Current clean rule (returns True for normal/clean values): {rule_str}",
            "Clean examples that SHOULD return True but currently return False:",
            json.dumps(clean_mis_examples[:10], ensure_ascii=False),
            "Dirty examples that SHOULD return False but currently return True:",
            json.dumps(dirty_mis_examples[:10], ensure_ascii=False),
            "Representative frequent values in this column:",
            json.dumps(top_values[:8], ensure_ascii=False),
            "Task:",
            "Rewrite the clean rule as a single Python lambda expression.",
            "Requirements:",
            "- Keep the semantics: True = value is clean/normal, False = value is abnormal.",
            "- Make the rule less aggressive on clean examples and more strict on dirty ones.",
            "- Handle None/NaN and empty strings safely.",
            "Return ONLY one lambda in the exact format:",
            "lambda value, row=None: <boolean expression>",
        ]
        prompt = "\n".join(prompt_parts)
        response = self._call_llm(prompt, temperature=0.2)
        return self._extract_lambda(response)


class DualAgent(BaseAgent):
    """Agent that generates paired clean/dirty rules for dual verification."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None):
        """Initialize dual agent."""
        super().__init__(base_url, model)

    def get_p_clean_system_prompt(self) -> str:
        """System prompt for P_clean (clean data confirmation) rule generation."""
        return """You are a data quality expert specializing in identifying CLEAN data.

Your role is to generate Python lambda functions for P_clean(x) that CONFIRM when a value is definitely CLEAN (correct/proper).

CRITICAL REQUIREMENTS (cover every agent below):
1. Completeness: reject missing/placeholder tokens (None, NaN, empty strings, 'n/a', 'unknown', etc.)
2. Accuracy: ensure numeric ranges, enumerations, and domain-specific thresholds are respected
3. Column relationship constraints: when metadata provides relational expectations, enforce them conservatively
4. Pattern/format consistency: honor regex-like structures, code lengths, and canonical casing
5. P_clean/P_dirty should remain strict overall—only reject values that clearly break the rule OR are highly unlikely to be valid
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
lambda value: value is None or str(value).strip() in ['', 'nan', 'none', 'null', 'n/a', 'na', 'unknown'] or (str(value).replace('.', '',1).replace('-', '', 1).isdigit() and (float(value) < 0 or float(value) > 150))

Be strict in what you flag as dirty - P_dirty's job is to catch clear errors, not to be permissive.
"""

    def get_p_clean_disjointness_prompt(self) -> str:
        """System prompt for P_clean refinement with disjointness constraint (low temperature)."""
        return """You are a data quality expert specializing in refining CLEAN data predicates.

Your role is to refine P_clean(x) to be DISJOINT from P_dirty(x).

CRITICAL DISJOINTNESS REQUIREMENT:
- P_clean(x) and P_dirty(x) MUST NEVER both be True for the same value
- You must EXCLUDE values that P_dirty would flag as dirty

SAMPLING FEEDBACK:
- Review conflict samples (values where both P_clean and P_dirty were True)
- These values MUST return False in your refined P_clean
- Review gap samples (values where both P_clean and P_dirty were False)
- Consider if these should return True in your refined P_clean

Return ONLY the lambda function:
lambda value: <expression>

Example refinement:
If conflict samples show '12345' triggered both rules:
OLD: lambda value: value is not None and str(value).isdigit()
NEW: lambda value: value is not None and str(value).isdigit() and len(str(value)) == 5

Be PRECISE and CONTRACT-ONLY - minimize creative interpretation to avoid new conflicts.
"""

    def get_p_dirty_disjointness_prompt(self) -> str:
        """System prompt for P_dirty refinement with disjointness constraint (low temperature)."""
        return """You are a data quality expert specializing in refining DIRTY data predicates.

Your role is to refine P_dirty(x) to be DISJOINT from P_clean(x).

CRITICAL DISJOINTNESS REQUIREMENT:
- P_clean(x) and P_dirty(x) MUST NEVER both be True for the same value
- You must EXCLUDE values that P_clean would flag as clean

SAMPLING FEEDBACK:
- Review conflict samples (values where both P_clean and P_dirty were True)
- These values MUST return False in your refined P_dirty (let P_dirty handle them)
- Review gap samples (values where both P_clean and P_dirty were False)
- Consider if these should return True in your refined P_dirty

Return ONLY the lambda function:
lambda value: <expression>

Example refinement:
If conflict samples show valid codes like 'NYC' triggered both rules:
OLD: lambda value: value is None or str(value).lower() in ['n/a', 'unknown']
NEW: lambda value: value is None or str(value).lower() in ['n/a', 'unknown', 'xxxxx', 'asdf']

Be PRECISE and CONTRACT-ONLY - minimize creative interpretation to avoid new conflicts.
"""

    def _combine_clean_base_rules(self, clean_base_rules: List[Tuple[str, str]] = None) -> str:
        """Combine deterministic clean base rules into a single predicate."""
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
        
        # Check if disjointness mode is active
        disjointness_mode = metadata.get('disjointness_mode', False)
        
        if disjointness_mode:
            # Use disjointness prompt with lower temperature for precision
            rules_text = self._call_llm(
                prompt,
                max_tokens=LLM_MAX_TOKENS,
                system_prompt=self.get_p_clean_disjointness_prompt(),
                temperature=0.2,
            )
        else:
            # Use standard prompt with default temperature
            rules_text = self._call_llm(
                prompt,
                max_tokens=LLM_MAX_TOKENS,
                system_prompt=self.get_p_clean_system_prompt(),
            )

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
            f"Generate DIRTY DETECTION predicate for column '{column}'",
            f"\nGOAL: Capture ERROR patterns, NOT describe normal data",
            f"\nColumn type: {col_type}",
            f"\nSample values: {sample_values[:10]}",
            f"\nTop values: {list(top_values.keys())[:10]}",
        ]

        # Add refinement context if available
        if '_refine_dirty_only_samples' in metadata:
            dirty_samples = metadata['_refine_dirty_only_samples'][:5]
            prompt_parts.append(f"\nConfirmed Errors (must detect): {dirty_samples}")

        if '_refine_conflict_samples' in metadata:
            conflict_samples = metadata['_refine_conflict_samples'][:10]
            prompt_parts.append(f"\nFalse Positives (must NOT detect): {conflict_samples}")
            prompt_parts.append("\nHint: Look for conditional patterns (country, date ranges, etc.) to separate these groups")

        prompt_parts.append("\n\nReturn ONLY the lambda function:")
        prompt_parts.append("lambda value, row=None: <expression>")

        prompt = "\n".join(prompt_parts)
        
        # Check if disjointness mode is active
        disjointness_mode = metadata.get('disjointness_mode', False)
        
        if disjointness_mode:
            # Use disjointness prompt with lower temperature for precision
            rules_text = self._call_llm(
                prompt,
                max_tokens=LLM_MAX_TOKENS,
                system_prompt=self.get_p_dirty_disjointness_prompt(),
                temperature=0.2,
            )
        else:
            # Use standard prompt with default temperature
            rules_text = self._call_llm(
                prompt,
                max_tokens=LLM_MAX_TOKENS,
                system_prompt=self.get_p_dirty_system_prompt(),
            )

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

        # Validate syntax
        if not self._validate_lambda_syntax(dirty_rule):
            print(f"  ⚠ Invalid P_dirty rule syntax, using fallback")
            return self._fallback_p_dirty_rule(column, metadata)

        return dirty_rule

    def _validate_lambda_syntax(self, rule_str: str) -> bool:
        """Validate that a lambda rule string has correct syntax."""
        try:
            # Try to compile the rule
            compile(rule_str, '<string>', 'eval')
            return True
        except:
            return False

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

        # Generate P_clean and P_dirty independently (no complement rule)
        clean_rule = self.generate_p_clean_rule(column, meta, clean_base_rules=clean_base_rules)
        if not clean_rule:
            clean_rule = self._fallback_p_clean_rule(column, meta)

        # Generate independent dirty rule (not complement of clean)
        dirty_rule = self.generate_p_dirty_rule(column, meta)

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
            # For abv-like columns (alcohol by volume), be very permissive
            if 'abv' in column.lower() or any('.' in str(v) and len(str(v)) <= 4 for v in sample_values[:10]):
                # ABV: accept any numeric value (with or without %), including values > 1
                # This is very permissive to reduce gap zone
                clean_rule = (
                    "lambda value: value is not None and str(value).strip() not in "
                    f"{missing_tokens} and (str(value).replace('%', '').replace('.', '', 1).replace('-', '', 1).isdigit() or "
                    "(isinstance(value, (int, float)) and 0 <= value <= 100))"
                )
            else:
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
            # For abv-like columns, flag values with % symbol or invalid format
            if 'abv' in column.lower() or any('.' in str(v) and len(str(v)) <= 4 for v in sample_values[:10]):
                # ABV: flag values with % symbol or clearly invalid
                dirty_rule = (
                    "lambda value: value is None or str(value).strip() in "
                    f"{missing_tokens} or ('%' in str(value)) or (str(value).replace('.', '', 1).replace('%', '', 1).isdigit() "
                    "and (float(str(value).strip('%')) < 0 or float(str(value).strip('%')) > 100))"
                )
            else:
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


class AgentFactory:
    """Factory to create appropriate agents based on column type."""
    def infer_table_relationship_candidates(
        self,
        table_summary: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Make one table-level LLM call for cross-row FD proposals."""
        agent = TableRelationshipAgent(self.base_url, self.model)
        return agent.infer_cross_row_fd_candidates(table_summary)


    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = None, max_workers: int = 1):
        """Initialize factory with base URL and model."""
        self.base_url = base_url
        self.model = model
        self.max_workers = max_workers or 1

    def create_rule_reviewer(self) -> RuleReviewerAgent:
        return RuleReviewerAgent(self.base_url, self.model)

    def infer_column_semantics(
        self,
        metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, ColumnSemantics]:
        """Infer and validate one semantic archetype for every column."""
        agent = ColumnArchetypeAgent(self.base_url, self.model)
        return {
            column: agent.infer(column, col_metadata)
            for column, col_metadata in metadata.items()
        }

    def infer_canonicalization_plans(
        self,
        profiles: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Generate one conservative, column-specific repair plan per profile."""
        items = list(profiles.items())

        def infer_one(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
            column, profile = item
            agent = CanonicalizationPlanningAgent(self.base_url, self.model)
            return column, agent.infer(column, profile)

        if self.max_workers <= 1 or len(items) <= 1:
            return dict(infer_one(item) for item in items)
        plans: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for column, plan in executor.map(infer_one, items):
                plans[column] = plan
        return plans

    def create_agents(self, column: str, column_type: str) -> List[BaseAgent]:
        """Create appropriate agents for a column based on its type."""
        agents = []
        
        # All columns can have missing values
        agents.append(MissingAgent(self.base_url, self.model))
        agents.append(PatternAgent(self.base_url, self.model))

        # Type-specific agents
        if column_type == 'categorical':
            agents.append(TypoAgent(self.base_url, self.model))
        elif column_type == 'numeric':
            agents.append(OutlierAgent(self.base_url, self.model))

        # Fallback: also try typo for text columns
        if column_type == 'text':
            agents.append(TypoAgent(self.base_url, self.model))
        
        return agents

    def create_clean_agents(self, column: str, column_type: str,
                            col_metadata: Dict[str, Any] = None) -> List[BaseAgent]:
        """Create clean-rule agents that cover completeness/accuracy/relationships/patterns."""
        agents: List[BaseAgent] = [
            CleanCompletenessAgent(self.base_url, self.model),
            CleanAccuracyAgent(self.base_url, self.model),
            CleanPatternAgent(self.base_url, self.model),
        ]
        metadata = col_metadata or {}
        if metadata.get('relationship_constraints') or metadata.get('relationship_profiles'):
            agents.append(CleanRelationshipAgent(self.base_url, self.model))
        return agents

    def _generate_rules_for_single_column(self, column: str, col_metadata: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]], Dict[str, Dict[str, str]], List[str]]:
        col_type = col_metadata.get('type', 'text')
        agents = self.create_agents(column, col_type)
        col_rules: List[Tuple[str, str]] = []
        col_prompts: Dict[str, Dict[str, str]] = {}
        log_lines: List[str] = []
        for agent in agents:
            agent_name = agent.__class__.__name__
            log_lines.append(f"  → {agent_name} for {column}...")
            try:
                rules = agent.generate_rules(column, col_metadata)
                for rule in rules:
                    col_rules.append((agent_name, rule))
                    log_lines.append(f"    ✓ Generated rule: {rule}")
                prompt_info = agent.get_last_prompt_info()
                if prompt_info['prompt']:
                    col_prompts[agent_name] = prompt_info
            except Exception as e:
                log_lines.append(f"    ✗ Error: {e}")
        return column, col_rules, col_prompts, log_lines

    def generate_rules_per_column(self, metadata: Dict[str, Any]) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Dict[str, Dict[str, str]]]]:
        """
        Generate rules for all columns in metadata.

        Returns:
            Tuple of:
            - Dict[column_name] = List[(agent_name, rule_string)]
            - Dict[column_name] = Dict[agent_name] = {'prompt': ..., 'response': ...}
        """
        all_rules: Dict[str, List[Tuple[str, str]]] = {}
        all_prompts: Dict[str, Dict[str, Dict[str, str]]] = {}
        items = list(metadata.items())
        max_workers = self.max_workers if hasattr(self, "max_workers") else 1
        if max_workers <= 1 or len(items) <= 1:
            for column, col_metadata in items:
                _, col_rules, col_prompts, log_lines = self._generate_rules_for_single_column(column, col_metadata)
                for line in log_lines:
                    print(line)
                if col_rules:
                    all_rules[column] = col_rules
                if col_prompts:
                    all_prompts[column] = col_prompts
            return all_rules, all_prompts
        results: Dict[str, Tuple[List[Tuple[str, str]], Dict[str, Dict[str, str]], List[str]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for column, col_metadata in items:
                futures.append(executor.submit(self._generate_rules_for_single_column, column, col_metadata))
            for future in futures:
                column, col_rules, col_prompts, log_lines = future.result()
                results[column] = (col_rules, col_prompts, log_lines)
        for column, _ in items:
            if column not in results:
                continue
            col_rules, col_prompts, log_lines = results[column]
            for line in log_lines:
                print(line)
            if col_rules:
                all_rules[column] = col_rules
            if col_prompts:
                all_prompts[column] = col_prompts
        return all_rules, all_prompts

    def _generate_clean_rules_for_single_column(self, column: str, col_metadata: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]], Dict[str, Dict[str, str]], List[str]]:
        column_type = col_metadata.get('type', 'text')
        agents = self.create_clean_agents(column, column_type, col_metadata)
        column_rules: List[Tuple[str, str]] = []
        column_prompts: Dict[str, Dict[str, str]] = {}
        log_lines: List[str] = []
        for agent in agents:
            try:
                clean_agent_rules = agent.generate_rules(column, col_metadata)
                for rule in clean_agent_rules:
                    column_rules.append((agent.clean_agent_name, rule))
                prompt_info = agent.get_last_prompt_info()
                if prompt_info['prompt']:
                    column_prompts[agent.clean_agent_name] = prompt_info
            except Exception as e:
                log_lines.append(f"    ✗ Clean agent {agent.clean_agent_name} error on {column}: {e}")
        return column, column_rules, column_prompts, log_lines

    def generate_clean_rules_per_column(self, metadata: Dict[str, Any]) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Dict[str, Dict[str, str]]]]:
        """
        Generate clean base rules per column (four quality pillars).

        Returns:
            Tuple of:
            - Dict[column_name] = List[(clean_agent_name, rule_string)]
            - Dict[column_name] = Dict[clean_agent_name] = {'prompt': ..., 'response': ...}
        """
        clean_rules: Dict[str, List[Tuple[str, str]]] = {}
        clean_prompts: Dict[str, Dict[str, Dict[str, str]]] = {}
        items = list(metadata.items())
        max_workers = self.max_workers if hasattr(self, "max_workers") else 1
        if max_workers <= 1 or len(items) <= 1:
            for column, col_metadata in items:
                _, column_rules, column_prompts, log_lines = self._generate_clean_rules_for_single_column(column, col_metadata)
                for line in log_lines:
                    print(line)
                if column_rules:
                    clean_rules[column] = column_rules
                if column_prompts:
                    clean_prompts[column] = column_prompts
            return clean_rules, clean_prompts
        results: Dict[str, Tuple[List[Tuple[str, str]], Dict[str, Dict[str, str]], List[str]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for column, col_metadata in items:
                futures.append(executor.submit(self._generate_clean_rules_for_single_column, column, col_metadata))
            for future in futures:
                column, column_rules, column_prompts, log_lines = future.result()
                results[column] = (column_rules, column_prompts, log_lines)
        for column, _ in items:
            if column not in results:
                continue
            column_rules, column_prompts, log_lines = results[column]
            for line in log_lines:
                print(line)
            if column_rules:
                clean_rules[column] = column_rules
            if column_prompts:
                clean_prompts[column] = column_prompts
        return clean_rules, clean_prompts

    def generate_cross_column_rules(self, metadata: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Generate cross-column consistency rules.

        Returns:
            List[(agent_name, rule_string)]
        """
        return []

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
        print("      (MissingAgent, TypoAgent, OutlierAgent, PatternAgent)")

        for column, col_metadata in metadata.items():
            print(f"\n{'='*80}")
            print(f"Column: {column}")
            print(f"{'='*80}")

            # Create dual agent
            dual_agent = DualAgent(self.base_url, self.model)

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
                col_rules = dual_agent.generate_dual_rules(
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

    def generate_p_clean_predicates_per_column(self, metadata: Dict[str, Any],
                                              clean_base_rules: Dict[str, List[Tuple[str, str]]] = None,
                                              base_rules: Dict[str, List[Tuple[str, str]]] = None,
                                              refinement_context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate independent P_clean predicates for all columns (not as a complement of P_dirty).

        Args:
            metadata: Column metadata dictionary
            clean_base_rules: Pre-generated clean rules per quality pillar
            base_rules: Base error detection rules for reference
            refinement_context: Optional refinement samples (conflict/gap)

        Returns:
            Dict[column] = P_clean rule string
        """
        print("\n" + "="*80)
        print("GENERATING P_CLEAN PREDICATES (INDEPENDENT)")
        print("="*80)

        p_clean_rules = {}
        dual_agent = DualAgent(self.base_url, self.model)

        for column, col_metadata in metadata.items():
            print(f"\n{'='*80}")
            print(f"Column: {column}")
            print(f"{'='*80}")

            try:
                # Prepare metadata with refinement context if provided
                meta = dict(col_metadata or {})
                if refinement_context and column in refinement_context:
                    context = refinement_context[column]
                    meta["_refine_grey_samples"] = context.get('gap_samples', [])
                    meta["_refine_conflict_samples"] = context.get('conflict_samples', [])
                    meta["_base_rules"] = base_rules.get(column, []) if base_rules else []

                # Generate P_clean using DualAgent
                clean_rule = dual_agent.generate_p_clean_rule(
                    column,
                    meta,
                    clean_base_rules=clean_base_rules.get(column, []) if clean_base_rules else None
                )

                if clean_rule:
                    p_clean_rules[column] = clean_rule
                    print(f"✓ Generated P_clean:")
                    print(f"  {clean_rule}")
                else:
                    print(f"✗ Failed to generate P_clean")

            except Exception as e:
                print(f"✗ Error generating P_clean: {e}")

        return p_clean_rules

    def generate_p_dirty_predicates_per_column(self, metadata: Dict[str, Any],
                                              base_rules: Dict[str, List[Tuple[str, str]]] = None,
                                              refinement_context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate independent P_dirty predicates for all columns.

        Args:
            metadata: Column metadata dictionary
            base_rules: Base error detection rules for reference
            refinement_context: Optional refinement samples (conflict/gap)

        Returns:
            Dict[column] = P_dirty rule string
        """
        print("\n" + "="*80)
        print("GENERATING P_DIRTY PREDICATES (INDEPENDENT)")
        print("="*80)

        p_dirty_rules = {}
        dual_agent = DualAgent(self.base_url, self.model)

        for column, col_metadata in metadata.items():
            print(f"\n{'='*80}")
            print(f"Column: {column}")
            print(f"{'='*80}")

            try:
                # Prepare metadata with refinement context if provided
                meta = dict(col_metadata or {})
                if refinement_context and column in refinement_context:
                    context = refinement_context[column]
                    meta["_refine_gap_samples"] = context.get('gap_samples', [])
                    meta["_refine_conflict_samples"] = context.get('conflict_samples', [])
                    meta["_base_rules"] = base_rules.get(column, []) if base_rules else []

                # Generate P_dirty using DualAgent
                dirty_rule = dual_agent.generate_p_dirty_rule(column, meta)

                if dirty_rule:
                    p_dirty_rules[column] = dirty_rule
                    print(f"✓ Generated P_dirty:")
                    print(f"  {dirty_rule}")
                else:
                    print(f"✗ Failed to generate P_dirty")

            except Exception as e:
                print(f"✗ Error generating P_dirty: {e}")

        return p_dirty_rules

    def pair_clean_dirty(self, p_clean_map: Dict[str, str],
                        p_dirty_map: Dict[str, str]) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Pair up independent P_clean and P_dirty rules into dual rule format.

        Args:
            p_clean_map: Dict[column] = P_clean rule string
            p_dirty_map: Dict[column] = P_dirty rule string

        Returns:
            Dict[column] = List[(agent_name, clean_rule_str, dirty_rule_str)]
        """
        print("\n" + "="*80)
        print("PAIRING P_CLEAN AND P_DIRTY RULES")
        print("="*80)

        paired_rules = {}
        all_columns = set(p_clean_map.keys()) | set(p_dirty_map.keys())

        for column in all_columns:
            clean_rule = p_clean_map.get(column)
            dirty_rule = p_dirty_map.get(column)

            if clean_rule and dirty_rule:
                paired_rules[column] = [("DualAgent", clean_rule, dirty_rule)]
                print(f"\n✓ {column}: paired successfully")
            else:
                if not clean_rule:
                    print(f"\n✗ {column}: missing P_clean")
                if not dirty_rule:
                    print(f"\n✗ {column}: missing P_dirty")

        return paired_rules


class RuleFixerAgent(BaseAgent):
    """Agent specialized in fixing syntax errors in lambda rules."""

    def _get_system_prompt(self) -> str:
        """System prompt for fixing syntax errors in rules."""
        return """You are a Python syntax expert specializing in fixing broken lambda functions.
Your task is to fix syntax errors in lambda rules while preserving their semantic meaning.
Common issues to fix:
- Unterminated strings (missing closing quote)
- Unbalanced parentheses
- Invalid escape sequences
- Incorrect indentation
- Reserved keyword conflicts

Return ONLY the fixed lambda function, no explanation.
"""

    def fix_syntax_error(self, broken_rule: str, error_msg: str,
                         context: str = None) -> str:
        """Fix a broken lambda rule given the syntax error.

        Args:
            broken_rule: The original broken rule string
            error_msg: The syntax error message
            context: Optional context about the rule (column name, purpose)

        Returns:
            The fixed lambda function string
        """
        prompt = f"""Fix this broken Python lambda rule:

**Original Rule:**
{broken_rule}

**Syntax Error:**
{error_msg}

{f"**Context:** {context}" if context else ""}

**Requirements:**
1. Fix the syntax error while preserving the rule's semantic meaning
2. Handle any special characters or strings properly
3. Return ONLY the corrected lambda function, no explanation
4. The lambda should have format: lambda value, row=None: <expression>

Examples of fixes:
- If string is unterminated: add the missing closing quote
- If parentheses are unbalanced: add missing )
- If invalid escape: use raw strings or proper escaping

        Return ONLY the fixed lambda:
"""

        response = self._call_llm(prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.1)

        # Extract the lambda from response
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith('lambda'):
                return line

        # If no lambda found, return None
        return None
