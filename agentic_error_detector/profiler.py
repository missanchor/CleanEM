"""
Pandas-based profiler for generating metadata from hospital data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class PandasProfiler:
    """Generate metadata for columns in a DataFrame."""

    def __init__(self, data_path: str):
        """Load and initialize profiler with data."""
        self.df = pd.read_csv(data_path)
        self.metadata = {}
        self._clean_and_analyze()

    def _clean_and_analyze(self):
        """Convert 'empty' strings to np.nan for consistent analysis."""
        # Replace 'empty' strings with NaN
        self.df = self.df.replace('empty', np.nan)
        self._generate_metadata()

    def _generate_metadata(self):
        """Generate comprehensive metadata for each column."""
        for column in self.df.columns:
            col_type = self._determine_column_type(column)
            metadata = {
                'type': col_type,
                'total_rows': len(self.df),
                'non_null_count': self.df[column].notna().sum(),
                'null_count': self.df[column].isna().sum(),
            }

            if col_type == 'categorical':
                metadata.update(self._analyze_categorical(column))
            elif col_type == 'pattern':
                metadata.update(self._analyze_pattern(column))
            elif col_type == 'numeric':
                metadata.update(self._analyze_numeric(column))
            elif col_type == 'text':
                metadata.update(self._analyze_text(column))

            self.metadata[column] = metadata

        self._infer_relationship_constraints()

    def _determine_column_type(self, column: str) -> str:
        """Determine the type of column based on heuristics."""
        series = self.df[column].dropna()
        if series.empty:
            return 'text'

        # Check for numeric-like strings (containing at least one digit)
        import re
        sample_vals = series.head(50).astype(str).tolist()
        has_digits = any(re.search(r'\d', v) for v in sample_vals)
        
        # Check uniqueness ratio
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        
        # Heuristics
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        if has_digits:
            # Pattern columns usually have higher uniqueness or fixed structures
            if unique_ratio > 0.5 or any(re.search(r'[-/()]', v) for v in sample_vals):
                return 'pattern'
            return 'numeric' # Could be numeric values stored as strings (e.g. "97%")
            
        if unique_ratio < 0.2:
            return 'categorical'
        
        return 'text'

    def _analyze_categorical(self, column: str) -> Dict[str, Any]:
        """Analyze categorical column for unique values and frequencies."""
        value_counts = self.df[column].value_counts()
        unique_count = len(value_counts)

        return {
            'unique_count': unique_count,
            'top_values': value_counts.head(10).to_dict(),
            'frequency_distribution': value_counts.to_dict(),
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
        # Try to convert to numeric, handling cases like "97%", "3x patients", etc.
        numeric_values = []
        for val in self.df[column].dropna():
            str_val = str(val).strip()
            # Extract numeric part
            import re
            numeric_match = re.search(r'(\d+(?:\.\d+)?)', str_val)
            if numeric_match:
                numeric_values.append(float(numeric_match.group(1)))

        if numeric_values:
            return {
                'min': min(numeric_values),
                'max': max(numeric_values),
                'mean': sum(numeric_values) / len(numeric_values),
                'numeric_count': len(numeric_values),
                'non_numeric_examples': [v for v in self.df[column].dropna().head(10).tolist() if not re.search(r'(\d+(?:\.\d+)?)', str(v))]
            }
        else:
            return {
                'numeric_count': 0,
                'all_non_numeric': self.df[column].dropna().head(10).tolist()
            }

    def _analyze_text(self, column: str) -> Dict[str, Any]:
        """Analyze text column."""
        value_counts = self.df[column].value_counts()
        return {
            'unique_count': len(value_counts),
            'top_values': value_counts.head(5).to_dict(),
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
        else:
            return "Pattern not clearly identifiable"

    def _infer_relationship_constraints(self):
        """Infer simple column relationship hints (state-prefix, city mentions, etc.)."""
        column_lookup = {col.lower(): col for col in self.df.columns}
        state_col = column_lookup.get('state')
        city_col = column_lookup.get('city')
        zipcode_col = column_lookup.get('zipcode') or column_lookup.get('zip')

        for column, metadata in self.metadata.items():
            constraints = []
            column_lower = column.lower()

            if state_col and column != state_col and 'state' in column_lower:
                constraints.append({
                    'type': 'prefix_match',
                    'other_column': state_col,
                    'description': f"{column} 应以 State 列的值开头"
                })

            if city_col and column != city_col and 'city' in column_lower:
                constraints.append({
                    'type': 'contains',
                    'other_column': city_col,
                    'description': f"{column} 应包含 City 列的值"
                })

            if state_col and 'stateavg' in column_lower:
                constraints.append({
                    'type': 'stateavg_format',
                    'other_column': state_col,
                    'description': "Stateavg 应采用 <state>_<metric> 格式"
                })

            if zipcode_col and column != zipcode_col and 'zip' in column_lower:
                constraints.append({
                    'type': 'zip_prefix',
                    'other_column': zipcode_col,
                    'description': f"{column} 应与 ZipCode 列前 3 位保持一致"
                })

            if constraints:
                metadata['relationship_constraints'] = constraints

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