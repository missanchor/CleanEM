"""
Pandas-based profiler for generating metadata from hospital data.
"""
import re
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DEFAULT_MISSING_TOKENS = [
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "empty",
    "xxxxx",
    "--",
    "n.a",
    "not available",
]
DEFAULT_MISSING_TOKEN_SET = set(DEFAULT_MISSING_TOKENS)

MAX_TOP_VALUES = 20
MAX_LOW_FREQ_VALUES = 20
MAX_LENGTH_ENTRIES = 20
MAX_SHAPE_ENTRIES = 20
MAX_REGEX_CANDIDATES = 3
MAX_COOCCURRENCES = 15
MAX_VIOLATION_SAMPLES = 10


class PandasProfiler:
    """Generate metadata for columns in a DataFrame."""

    def __init__(self, data_path: str, enable_pattern_detection: bool = False):
        """
        Load and initialize profiler with data.

        Args:
            data_path: Path to the CSV file
            enable_pattern_detection: If True, enable pattern column detection.
                                    If False (default), only use numeric/categorical/text.
        """
        self.df = pd.read_csv(data_path)
        self.data_path = data_path
        self.metadata = {}
        self._clean_and_analyze(enable_pattern_detection)

    def _clean_and_analyze(self, enable_pattern_detection: bool = False):
        """
        Convert 'empty' strings to np.nan for consistent analysis.

        Args:
            enable_pattern_detection: If True, enable pattern column detection.
        """
        # Replace 'empty' strings with NaN
        self.df = self.df.replace('empty', np.nan)
        self._generate_metadata(enable_pattern_detection)

    def _generate_metadata(self, enable_pattern_detection: bool = False):
        """
        Generate comprehensive metadata for each column.

        Args:
            enable_pattern_detection: If True, enable pattern column detection.
                                    If False (default), only use numeric/categorical/text.
        """
        for column in self.df.columns:
            col_type = self._determine_column_type(column, enable_pattern_detection, self.data_path)
            metadata = {
                'type': col_type,
                'total_rows': len(self.df),
                'non_null_count': self.df[column].notna().sum(),
                'null_count': self.df[column].isna().sum(),
            }

            metadata.update(self._analyze_missing_tokens(column))

            if col_type == 'categorical':
                metadata.update(self._analyze_categorical(column))
                metadata.update(self._analyze_string_properties(column))
            elif col_type == 'pattern':
                metadata.update(self._analyze_pattern(column))
                metadata.update(self._analyze_string_properties(column))
            elif col_type == 'numeric':
                metadata.update(self._analyze_numeric(column))
            elif col_type == 'text':
                metadata.update(self._analyze_text(column))
                metadata.update(self._analyze_string_properties(column))

            self.metadata[column] = metadata

        self._infer_relationship_constraints()
        self._attach_relationship_profiles()

    def _determine_column_type(self, column: str, enable_pattern_detection: bool = False, csv_path: str = None) -> str:
        """
        Determine the type of column based on heuristics or config file.

        Args:
            column: Column name
            enable_pattern_detection: If False, only return numeric/categorical/text (default).
                                    If True, may also return 'pattern' for structured values.
            csv_path: Path to CSV file (used for config lookup)
        """
        # Try to load from config first
        if csv_path:
            config_types = self._load_col_type_from_config(csv_path)
            if config_types and column in config_types:
                return config_types[column]

        series = self.df[column].dropna()
        if series.empty:
            return 'text'

        # Check uniqueness ratio
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

        # Heuristics - simplified to only 3 types by default
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'

        # Calculate the proportion of numeric characters across the column
        sample_vals = series.head(50).astype(str).tolist()
        avg_numeric_ratio = self._calculate_numeric_character_ratio(sample_vals)

        # If average numeric character ratio is high, classify as numeric
        if avg_numeric_ratio >= 0.5:
            if enable_pattern_detection:
                # Pattern columns usually have higher uniqueness or fixed structures
                if unique_ratio > 0.5 or any(re.search(r'[-/()]', v) for v in sample_vals):
                    return 'pattern'
            return 'numeric'

        if unique_ratio < 0.2:
            return 'categorical'

        return 'text'

    def _load_col_type_from_config(self, csv_path: str) -> Dict[str, str]:
        """
        Load column types from config file.

        Lookup order:
        1. <csv_path>.config.json (e.g., data/hospital_error-01.csv.config.json)
        2. data/<dataset_name>_config.json (e.g., data/hospital_config.json)

        Args:
            csv_path: Path to the CSV file

        Returns:
            Dict[column_name] -> type, or empty dict if not found
        """
        import os
        import json

        # Try <csv_path>.config.json first
        config_path = csv_path + ".config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'col_type' in config:
                        return config['col_type']
            except Exception:
                pass

        # Try data/<dataset_name>_config.json
        # Extract dataset name from csv_path
        base_name = os.path.basename(csv_path)
        # Remove common suffixes - check longer suffixes first
        for suffix in ['_error-01', '_clean_missing', '_clean_mixed', '_clean_pattern', '_clean_rule', '_clean_outliers', '_clean_typos', '_clean', '_error']:
            if base_name == suffix + '.csv' or base_name.startswith(suffix + '.'):
                # Full match like hospital_clean.csv or hospital_clean.csv.something
                dataset_name = base_name[len(suffix):].lstrip('.').rsplit('.', 1)[0] if '.' in base_name[len(suffix):] else ''
                break
            elif base_name.endswith(suffix + '.csv'):
                # Normal case: hospital_clean.csv -> hospital
                dataset_name = base_name[:-len(suffix) - 4]  # -4 for .csv
                break
        else:
            # Remove .csv extension
            dataset_name = base_name[:-4] if base_name.endswith('.csv') else base_name

        config_path = os.path.join(os.path.dirname(csv_path), f"{dataset_name}_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'col_type' in config:
                        return config['col_type']
            except Exception:
                pass

        return {}

    def _calculate_numeric_character_ratio(self, values: List[str]) -> float:
        """
        Calculate the average proportion of numeric characters across all values.

        Args:
            values: List of string values to analyze

        Returns:
            Average proportion of numeric characters (0.0 to 1.0)
        """
        if not values:
            return 0.0

        total_chars = 0
        total_numeric_chars = 0

        for value in values:
            if not value or value.lower() in ['nan', 'none', 'null', '']:
                continue

            str_value = str(value).strip()
            if not str_value:
                continue

            total_chars += len(str_value)
            # Count numeric characters (0-9)
            numeric_chars = sum(1 for char in str_value if char.isdigit())
            total_numeric_chars += numeric_chars

        if total_chars == 0:
            return 0.0

        return total_numeric_chars / total_chars

    def _normalize_string(self, value: Any) -> str:
        """Normalize value for stable comparisons."""
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        normalized = str(value).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _analyze_missing_tokens(self, column: str) -> Dict[str, Any]:
        """Capture common missing token occurrences."""
        series = self.df[column]
        missing_counter = Counter()

        for value in series:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                missing_counter["<NA>"] += 1
                continue
            normalized = self._normalize_string(value)
            if normalized in DEFAULT_MISSING_TOKEN_SET:
                missing_counter[normalized] += 1

        dominant_tokens = [
            token for token, _ in missing_counter.most_common(10)
            if token != "<NA>"
        ]

        return {
            'missing_token_counts': {k: int(v) for k, v in missing_counter.items()},
            'dominant_missing_tokens': dominant_tokens
        }

    def _analyze_string_properties(self, column: str) -> Dict[str, Any]:
        """Generate rich distributions for string-like columns."""
        series = self.df[column].dropna()
        if series.empty:
            return {}

        str_series = series.astype(str)
        normalized_series = str_series.apply(self._normalize_string)
        total = len(normalized_series)
        if total == 0:
            return {}

        counts = normalized_series.value_counts()
        singleton_count = int((counts == 1).sum())
        top_values = counts.head(MAX_TOP_VALUES)

        examples = {}
        for raw, normalized in zip(str_series, normalized_series):
            if normalized and normalized not in examples:
                examples[normalized] = raw.strip()

        normalized_top_values = []
        for value, count in top_values.items():
            normalized_top_values.append({
                'value': value,
                'count': int(count),
                'ratio': round(float(count) / total, 3),
                'example': examples.get(value, value)
            })

        low_freq_candidates = counts[counts <= 3]
        low_freq_candidates = low_freq_candidates.sort_values(ascending=False).head(MAX_LOW_FREQ_VALUES)
        low_frequency_values = []
        for value, count in low_freq_candidates.items():
            low_frequency_values.append({
                'value': value,
                'count': int(count),
                'example': examples.get(value, value)
            })

        lengths = normalized_series.apply(len)
        length_counts = lengths.value_counts().sort_index()
        length_distribution = []
        for length, count in length_counts.head(MAX_LENGTH_ENTRIES).items():
            length_distribution.append({
                'length': int(length),
                'count': int(count),
                'ratio': round(float(count) / total, 3)
            })

        shape_distribution = self._shape_distribution(str_series)
        regex_candidates = self._generate_regex_candidates(str_series)

        return {
            'normalized_top_values': normalized_top_values,
            'singleton_count': singleton_count,
            'low_frequency_values': low_frequency_values,
            'length_distribution': length_distribution,
            'shape_distribution': shape_distribution,
            'regex_candidates': regex_candidates
        }

    def _shape_distribution(self, values: pd.Series) -> List[Dict[str, Any]]:
        """Summarize structural signatures of string values."""
        signatures = values.dropna().astype(str).apply(self._shape_signature)
        if signatures.empty:
            return []

        counts = signatures.value_counts().head(MAX_SHAPE_ENTRIES)
        total = counts.sum()
        distribution = []
        for signature, count in counts.items():
            distribution.append({
                'shape': signature,
                'count': int(count),
                'ratio': round(float(count) / total, 3)
            })
        return distribution

    def _shape_signature(self, value: str) -> str:
        """Convert a string to a coarse structural signature."""
        if value is None:
            return "<empty>"
        tokens = []
        for char in str(value).strip():
            if char.isdigit():
                tokens.append("D")
            elif char.isalpha():
                tokens.append("A")
            elif char in "-_/":
                tokens.append(char)
            elif char.isspace():
                tokens.append("S")
            else:
                tokens.append("?")
        return "".join(tokens) or "<empty>"

    def _generate_regex_candidates(self, values: pd.Series) -> List[Dict[str, Any]]:
        """Propose regex patterns that cover majority patterns."""
        cleaned = values.dropna().astype(str).str.strip()
        if cleaned.empty:
            return []

        candidate_values = cleaned.value_counts().head(10).index.tolist()
        candidates = []

        for value in candidate_values:
            pattern = self._build_regex_pattern(value)
            try:
                matches = cleaned.str.fullmatch(pattern).sum()
            except re.error:
                continue

            match_rate = matches / len(cleaned)
            if match_rate >= 0.5:
                candidates.append({
                    'pattern': pattern,
                    'match_rate': float(match_rate)
                })

            if len(candidates) >= MAX_REGEX_CANDIDATES:
                break

        return candidates

    def _build_regex_pattern(self, value: str) -> str:
        """Create a regex pattern from a representative value."""
        tokens = []

        def flush(current_token: str, count: int):
            if not current_token or count <= 0:
                return
            if current_token in (r"\d", r"[A-Za-z]", r"\s") and count > 1:
                tokens.append(f"{current_token}{{{count}}}")
            else:
                tokens.extend([current_token] * count)

        previous = None
        count = 0
        for char in str(value):
            if char.isdigit():
                token = r"\d"
            elif char.isalpha():
                token = r"[A-Za-z]"
            elif char.isspace():
                token = r"\s"
            else:
                token = re.escape(char)

            if token == previous:
                count += 1
            else:
                flush(previous, count)
                previous = token
                count = 1

        flush(previous, count)
        return "^" + "".join(tokens) + "$"

    def _extract_numeric_values(self, column: str) -> Dict[str, Any]:
        """Extract numeric values and track parsing stats."""
        series = self.df[column]
        numeric_values = []
        non_numeric_examples = []

        for value in series.dropna():
            str_val = str(value).strip().replace(',', '')
            match = re.search(r'(-?\d+(?:\.\d+)?)', str_val)
            if match:
                try:
                    numeric_values.append(float(match.group(1)))
                except ValueError:
                    continue
            else:
                if len(non_numeric_examples) < 10:
                    non_numeric_examples.append(str_val)

        return {
            'numeric_values': numeric_values,
            'non_numeric_examples': non_numeric_examples,
            'non_numeric_count': int(series.notna().sum() - len(numeric_values))
        }

    def _compute_numeric_distribution(self, column: str, numeric_values: List[float]) -> Dict[str, Any]:
        """Compute extended numeric statistics for a column."""
        if not numeric_values:
            return {}

        arr = np.array(numeric_values)
        quantiles = {
            'p01': float(np.percentile(arr, 1)),
            'p05': float(np.percentile(arr, 5)),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }

        median = quantiles['p50']
        mad = float(np.median(np.abs(arr - median)))
        std = float(np.std(arr))
        iqr = quantiles['p75'] - quantiles['p25']

        sorted_values = np.sort(arr)
        low_extremes = [float(v) for v in sorted_values[:3]]
        high_extremes = [float(v) for v in sorted_values[-3:]]

        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': std,
            'quantiles': quantiles,
            'iqr': float(iqr),
            'mad': mad,
            'extreme_numeric_values': {
                'low': low_extremes,
                'high': high_extremes
            }
        }

    def _analyze_categorical(self, column: str) -> Dict[str, Any]:
        """Analyze categorical column for unique values and frequencies."""
        non_null_values = self.df[column].dropna().astype(str)
        value_counts = non_null_values.value_counts()
        unique_count = len(value_counts)

        sample_size = min(20, len(non_null_values))
        sample_values = non_null_values.sample(n=sample_size, random_state=42).tolist()

        return {
            'unique_count': unique_count,
            'top_values': value_counts.head(10).to_dict(),
            'frequency_distribution': value_counts.to_dict(),
            'sample_values': sample_values,
            'pattern_analysis': self._guess_pattern(sample_values),
        }

    def _analyze_pattern(self, column: str) -> Dict[str, Any]:
        """Analyze pattern-based column (IDs, codes, etc.)."""
        non_null_values = self.df[column].dropna().astype(str)
        unique_count = non_null_values.nunique()
        sample_values = non_null_values.head(20).tolist()

        return {
            'unique_count': unique_count,
            'sample_values': sample_values,
            'pattern_analysis': self._guess_pattern(non_null_values.tolist()),
        }

    def _analyze_numeric(self, column: str) -> Dict[str, Any]:
        """Analyze numeric column."""
        extraction = self._extract_numeric_values(column)
        numeric_values = extraction.pop('numeric_values')

        metadata = {
            'numeric_count': len(numeric_values),
            'non_numeric_count': extraction.get('non_numeric_count', 0),
            'non_numeric_examples': extraction.get('non_numeric_examples', [])
        }

        if numeric_values:
            metadata.update(self._compute_numeric_distribution(column, numeric_values))
        else:
            metadata['all_non_numeric'] = self.df[column].dropna().head(10).tolist()

        # Add pattern-related metadata for PatternLegislator
        non_null_values = self.df[column].dropna()
        sample_values = non_null_values.head(20).astype(str).tolist()
        metadata.update({
            'sample_values': sample_values,
            'pattern_analysis': self._guess_pattern(sample_values),
        })

        # Add string properties for PatternLegislator
        str_series = non_null_values.astype(str)
        metadata.update(self._analyze_string_properties_for_numeric(column, str_series))

        return metadata

    def _analyze_string_properties_for_numeric(self, column: str, str_series: pd.Series) -> Dict[str, Any]:
        """Generate string properties for numeric columns (for PatternLegislator)."""
        if str_series.empty:
            return {}

        normalized_series = str_series.apply(self._normalize_string)
        total = len(normalized_series)
        if total == 0:
            return {}

        counts = normalized_series.value_counts()
        singleton_count = int((counts == 1).sum())
        top_values = counts.head(MAX_TOP_VALUES)

        examples = {}
        for raw, normalized in zip(str_series, normalized_series):
            if normalized and normalized not in examples:
                examples[normalized] = raw.strip()

        normalized_top_values = []
        for value, count in top_values.items():
            normalized_top_values.append({
                'value': value,
                'count': int(count),
                'ratio': round(float(count) / total, 3),
                'example': examples.get(value, value)
            })

        # For numeric columns, we might not have low frequency values in the same way
        # but let's include this for completeness
        low_freq_candidates = counts[counts <= 3]
        low_freq_candidates = low_freq_candidates.sort_values(ascending=False).head(MAX_LOW_FREQ_VALUES)
        low_frequency_values = []
        for value, count in low_freq_candidates.items():
            low_frequency_values.append({
                'value': value,
                'count': int(count),
                'example': examples.get(value, value)
            })

        # Length distribution is important for numeric "ID" columns
        lengths = str_series.apply(lambda x: len(str(x).strip()))
        length_counts = lengths.value_counts().sort_index()
        length_distribution = []
        for length, count in length_counts.head(MAX_LENGTH_ENTRIES).items():
            length_distribution.append({
                'length': int(length),
                'count': int(count),
                'ratio': round(float(count) / total, 3)
            })

        # Shape distribution for numeric columns
        shape_distribution = self._shape_distribution(str_series)
        regex_candidates = self._generate_regex_candidates(str_series)

        return {
            'normalized_top_values': normalized_top_values,
            'singleton_count': singleton_count,
            'low_frequency_values': low_frequency_values,
            'length_distribution': length_distribution,
            'shape_distribution': shape_distribution,
            'regex_candidates': regex_candidates
        }

    def _analyze_text(self, column: str) -> Dict[str, Any]:
        """Analyze text column."""
        non_null_values = self.df[column].dropna().astype(str)
        value_counts = non_null_values.value_counts()
        sample_values = non_null_values.head(20).tolist()

        return {
            'unique_count': len(value_counts),
            'top_values': value_counts.head(5).to_dict(),
            'sample_values': sample_values,
            'pattern_analysis': self._guess_pattern(sample_values),
        }

    def _guess_pattern(self, values: List[str]) -> str:
        """Guess the pattern of values."""
        if not values:
            return "No pattern detected"

        # Analyze the values to guess pattern
        if all(len(v) == 5 and v.isdigit() for v in values[:10]):
            return "5-digit numeric (likely ZIP code)"
        elif all(v.isdigit() and len(v) == 10 for v in values[:10]):
            return "10-digit numeric (likely phone number)"
        elif all(v.replace('x', '').isdigit() and len(v) == 5 for v in values[:10]):
            return "5-digit with possible errors (ProviderNumber with 'x' errors)"
        elif all(v.replace('.', '').isdigit() and len(v) >= 3 and len(v) <= 5 for v in values[:10]):
            # Check if this looks like an ID/code (numeric but with varying lengths)
            return "Numeric ID/code with fixed length pattern"
        elif all(v.isdigit() for v in values[:10]):
            # All numeric - check if it looks like an ID (high uniqueness)
            lengths = [len(v) for v in values[:20]]
            if len(set(lengths)) <= 2:  # Mostly uniform length
                return f"Numeric ID/code with {max(set(lengths), key=lengths.count)}-digit pattern"
            else:
                return "Numeric identifier with mixed length"
        else:
            return "Pattern not clearly identifiable"

    def _infer_relationship_constraints(self):
        """
        Infer column relationship hints based on data content and patterns.

        NOTE: This method provides minimal backward-compatible hints.
        For intelligent relationship inference, use LogicAgent directly.

        The LogicAgent in agent.py can analyze column semantics
        and data patterns to intelligently infer cross-column relationships
        without relying on these heuristic rules.
        """
        # Only infer very obvious relationships based on column names
        # This is kept for backward compatibility
        # For full intelligence, LogicAgent should be used directly
        column_lookup = {col.lower(): col for col in self.df.columns}
        state_col = column_lookup.get('state')
        city_col = column_lookup.get('city')
        zipcode_col = column_lookup.get('zipcode') or column_lookup.get('zip')

        for column, metadata in self.metadata.items():
            constraints = []
            column_lower = column.lower()

            # Very basic constraints based on common patterns
            if state_col and column != state_col and 'state' in column_lower:
                constraints.append({
                    'type': 'prefix_match',
                    'other_column': state_col,
                    'description': f"{column} should start with State column value"
                })

            if city_col and column != city_col and 'city' in column_lower:
                constraints.append({
                    'type': 'contains',
                    'other_column': city_col,
                    'description': f"{column} should contain City column value"
                })

            if state_col and 'stateavg' in column_lower:
                constraints.append({
                    'type': 'stateavg_format',
                    'other_column': state_col,
                    'description': "Stateavg should use <state>_<metric> format"
                })

            if zipcode_col and column != zipcode_col and 'zip' in column_lower:
                constraints.append({
                    'type': 'zip_prefix',
                    'other_column': zipcode_col,
                    'description': f"{column} should match first 3 digits of ZipCode"
                })

            # Note: These are minimal hints for backward compatibility.
            # The main intelligent inference is done by LogicAgent.
            if constraints:
                metadata['relationship_constraints'] = constraints

    def _attach_relationship_profiles(self):
        """Attach distribution summaries for cross-column constraints."""
        for column, metadata in self.metadata.items():
            constraints = metadata.get('relationship_constraints', [])
            if not constraints:
                continue

            profiles = []
            for constraint in constraints:
                other_column = constraint.get('other_column')
                constraint_type = constraint.get('type')
                if not other_column or other_column not in self.df.columns:
                    continue

                profile = self._summarize_relationship_constraint(
                    column,
                    other_column,
                    constraint_type
                )
                if profile:
                    profile['description'] = constraint.get('description')
                    profiles.append(profile)

            if profiles:
                metadata['relationship_profiles'] = profiles

    def _summarize_relationship_constraint(self, column: str, other_column: str,
                                           constraint_type: str) -> Dict[str, Any]:
        """Summarize adherence statistics for a constraint."""
        col_series = self.df[column]
        other_series = self.df[other_column]

        applicable_mask = (
            col_series.notna()
            & other_series.notna()
            & (col_series.astype(str).str.strip() != "")
            & (other_series.astype(str).str.strip() != "")
        )

        applicable_indices = col_series[applicable_mask].index
        if applicable_indices.empty:
            return {}

        validation_results = []
        for idx in applicable_indices:
            is_valid = self._relationship_predicate(
                col_series.at[idx],
                other_series.at[idx],
                constraint_type
            )
            validation_results.append(bool(is_valid))

        valid_series = pd.Series(True, index=self.df.index)
        valid_series.loc[applicable_indices] = validation_results
        violation_mask = applicable_mask & (~valid_series)

        applicable_count = int(applicable_mask.sum())
        violation_count = int(violation_mask.sum())

        violation_samples = []
        violation_indices = self.df.index[violation_mask]
        for idx in violation_indices[:MAX_VIOLATION_SAMPLES]:
            violation_samples.append({
                'row_index': int(idx),
                column: self.df.at[idx, column],
                other_column: self.df.at[idx, other_column]
            })

        cooccurrence_df = self.df.loc[applicable_mask, [column, other_column]]
        if cooccurrence_df.empty:
            top_cooccurrences = []
        else:
            co_counts = (
                cooccurrence_df
                .groupby([column, other_column])
                .size()
                .reset_index(name='count')
                .sort_values('count', ascending=False)
            )
            total_pairs = int(co_counts['count'].sum())
            top_cooccurrences = []
            for _, row in co_counts.head(MAX_COOCCURRENCES).iterrows():
                top_cooccurrences.append({
                    'value': row[column],
                    'other_value': row[other_column],
                    'count': int(row['count']),
                    'ratio': round(float(row['count']) / total_pairs, 3) if total_pairs else 0.0
                })

        return {
            'type': constraint_type,
            'other_column': other_column,
            'applicable_count': applicable_count,
            'violation_rate': violation_count / applicable_count if applicable_count else 0.0,
            'violation_samples': violation_samples,
            'top_cooccurrences': top_cooccurrences
        }

    def _relationship_predicate(self, value: Any, other_value: Any, constraint_type: str) -> bool:
        """Evaluate whether a value pair satisfies the constraint."""
        if value is None or other_value is None:
            return True

        value_str = str(value).strip()
        other_str = str(other_value).strip()
        if not value_str or not other_str:
            return True

        constraint_type = (constraint_type or "").lower()

        if constraint_type == 'prefix_match':
            return value_str.lower().startswith(other_str.lower())
        if constraint_type == 'contains':
            return other_str.lower() in value_str.lower()
        if constraint_type == 'stateavg_format':
            return value_str.lower().startswith(f"{other_str.lower()}_")
        if constraint_type == 'zip_prefix':
            return value_str[:3] == other_str[:3]

        return True

    def get_metadata(self, column: str = None) -> Dict[str, Any]:
        """Get metadata for a specific column or all columns."""
        if column:
            return self.metadata.get(column, {})
        return self.metadata

    def print_summary(self):
        """Print a summary of all columns."""
        print("=== COLUMN METADATA SUMMARY ===\n")
        for column, metadata in self.metadata.items():
            print(f"Column: {column}")
            print(f"  Type: {metadata['type']}")
            print(f"  Non-null: {metadata['non_null_count']}/{metadata['total_rows']}")
            print(f"  Unique values: {metadata.get('unique_count', 'N/A')}")

            if metadata['type'] == 'categorical':
                print(f"  Top values: {list(metadata['top_values'].keys())[:5]}")
            elif metadata['type'] == 'pattern':
                print(f"  Pattern: {metadata.get('pattern_analysis', 'N/A')}")
                print(f"  Samples: {metadata.get('sample_values', [])[:3]}")
            elif metadata['type'] == 'numeric':
                if metadata.get('numeric_count', 0) > 0:
                    print(f"  Range: {metadata['min']:.2f} - {metadata['max']:.2f} (mean: {metadata['mean']:.2f})")
                else:
                    print(f"  No numeric values found")

            print()