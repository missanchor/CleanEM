"""
DisjointnessValidator for validating dual rule constraints.

This module provides validation functionality to ensure P_clean and P_dirty rules
are properly disjoint (no intersections) and have acceptable coverage.
"""
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DisjointnessResult:
    """Result of disjointness validation."""
    column: str
    is_disjoint: bool
    intersection_count: int
    intersection_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    clean_coverage_rate: float = 0.0
    dirty_coverage_rate: float = 0.0
    gap_rate: float = 0.0  # both false
    
    # Validation status
    validation_status: str = "unknown"  # 'pass', 'fail', 'error'


@dataclass
class ValidationResult:
    """Aggregated validation result across all columns."""
    total_columns: int
    passed_columns: int
    failed_columns: int
    error_columns: int
    results: Dict[str, DisjointnessResult] = field(default_factory=dict)
    
    # Summary
    overall_pass: bool = False
    total_intersections: int = 0
    total_gaps: int = 0
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        status = "PASS" if self.overall_pass else "FAIL"
        return (
            f"Validation Status: {status}\n"
            f"Total Columns: {self.total_columns}\n"
            f"Passed: {self.passed_columns}, Failed: {self.failed_columns}, Errors: {self.error_columns}\n"
            f"Total Intersections: {self.total_intersections}\n"
            f"Total Gaps: {self.total_gaps}"
        )


class DisjointnessValidator:
    """
    Validator for checking disjointness constraints between P_clean and P_dirty rules.
    
    Constraints:
    1. P_clean(x) AND P_dirty(x) must NEVER both be True (disjointness)
    2. Acceptable grey zone (gap): P_clean(x) = False AND P_dirty(x) = False
    """
    
    def __init__(self, gap_tolerance: float = 0.0):
        """
        Initialize validator.
        
        Args:
            gap_tolerance: Maximum acceptable gap rate (both false)
        """
        self.gap_tolerance = gap_tolerance
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame, column: str,
                 clean_func, dirty_func) -> DisjointnessResult:
        """
        Validate disjointness for a single column.
        
        Args:
            df: DataFrame containing the data
            column: Column name to validate
            clean_func: P_clean predicate function
            dirty_func: P_dirty predicate function
        
        Returns:
            DisjointnessResult with detailed validation information
        """
        try:
            # Apply predicates
            clean_mask = []
            dirty_mask = []
            intersection_samples = []
            
            for idx, row in df.iterrows():
                value = row[column]
                try:
                    is_clean = bool(self._invoke_predicate(clean_func, value, row))
                    is_dirty = bool(self._invoke_predicate(dirty_func, value, row))
                except Exception:
                    is_clean = False
                    is_dirty = False
                
                clean_mask.append(is_clean)
                dirty_mask.append(is_dirty)
                
                if is_clean and is_dirty:
                    intersection_samples.append({
                        'row_index': int(idx),
                        'value': value,
                        'column': column
                    })
            
            # Calculate metrics
            import numpy as np
            clean_mask = np.array(clean_mask)
            dirty_mask = np.array(dirty_mask)
            
            intersection_count = len(intersection_samples)
            clean_coverage_rate = np.mean(clean_mask)
            dirty_coverage_rate = np.mean(dirty_mask)
            gap_mask = (~clean_mask) & (~dirty_mask)
            gap_rate = np.mean(gap_mask)
            
            # Determine status
            is_disjoint = intersection_count == 0
            gap_acceptable = gap_rate <= self.gap_tolerance
            
            if is_disjoint and gap_acceptable:
                validation_status = 'pass'
            elif not is_disjoint:
                validation_status = 'fail'
            else:
                validation_status = 'fail'  # gap too large
            
            return DisjointnessResult(
                column=column,
                is_disjoint=is_disjoint,
                intersection_count=intersection_count,
                intersection_samples=intersection_samples[:10],  # Limit samples
                clean_coverage_rate=clean_coverage_rate,
                dirty_coverage_rate=dirty_coverage_rate,
                gap_rate=gap_rate,
                validation_status=validation_status
            )
        except Exception as e:
            return DisjointnessResult(
                column=column,
                is_disjoint=False,
                intersection_count=-1,
                validation_status='error'
            )
    
    def validate_batch(self, df: pd.DataFrame,
                      best_rules: Dict[str, Any]) -> ValidationResult:
        """
        Validate disjointness for all columns.
        
        Args:
            df: DataFrame containing the data
            best_rules: Dictionary of {column: DualRule}
        
        Returns:
            ValidationResult with aggregated information
        """
        results = {}
        total_intersections = 0
        total_gaps = 0
        passed_count = 0
        failed_count = 0
        error_count = 0
        
        for column, rule in best_rules.items():
            result = self.validate(
                df, column,
                rule.clean_rule_func,
                rule.dirty_rule_func
            )
            results[column] = result
            total_intersections += result.intersection_count
            total_gaps += int(result.gap_rate * len(df))
            
            if result.validation_status == 'pass':
                passed_count += 1
            elif result.validation_status == 'fail':
                failed_count += 1
            else:
                error_count += 1
        
        # Determine overall status
        overall_pass = (failed_count == 0 and error_count == 0)
        
        return ValidationResult(
            total_columns=len(best_rules),
            passed_columns=passed_count,
            failed_columns=failed_count,
            error_columns=error_count,
            results=results,
            overall_pass=overall_pass,
            total_intersections=total_intersections,
            total_gaps=total_gaps
        )
    
    def report_violations(self, validation_result: ValidationResult) -> str:
        """
        Generate a detailed violation report.
        
        Args:
            validation_result: ValidationResult from validate_batch()
        
        Returns:
            Formatted string with violation details
        """
        lines = []
        lines.append("="*80)
        lines.append("DISJOINTNESS VALIDATION REPORT")
        lines.append("="*80)
        lines.append("")
        
        # Summary
        lines.append(f"Overall Status: {'PASS' if validation_result.overall_pass else 'FAIL'}")
        lines.append(f"Total Columns: {validation_result.total_columns}")
        lines.append(f"Passed: {validation_result.passed_columns}, Failed: {validation_result.failed_columns}, Errors: {validation_result.error_columns}")
        lines.append(f"Total Intersections: {validation_result.total_intersections}")
        lines.append(f"Total Gaps: {validation_result.total_gaps}")
        lines.append("")
        
        # Column-wise details
        lines.append("="*80)
        lines.append("COLUMN-WISE DETAILS")
        lines.append("="*80)
        
        for column, result in sorted(validation_result.results.items()):
            status_symbol = "✓" if result.validation_status == 'pass' else "✗"
            lines.append(f"\n{status_symbol} {column}:")
            lines.append(f"  Status: {result.validation_status.upper()}")
            lines.append(f"  Disjoint: {result.is_disjoint}")
            lines.append(f"  Intersections: {result.intersection_count}")
            lines.append(f"  Clean Coverage: {result.clean_coverage_rate:.4f}")
            lines.append(f"  Dirty Coverage: {result.dirty_coverage_rate:.4f}")
            lines.append(f"  Gap Rate: {result.gap_rate:.4f}")
            
            if result.intersection_samples:
                lines.append(f"\n  Intersection Samples:")
                for i, sample in enumerate(result.intersection_samples[:5], 1):
                    lines.append(f"    {i}. Row {sample['row_index']}: {sample['value']}")
        
        return "\n".join(lines)
    
    def save_results(self, validation_result: ValidationResult, output_path: str):
        """
        Save validation results to JSON file.
        
        Args:
            validation_result: ValidationResult from validate_batch()
            output_path: Path to save JSON file
        """
        import os
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        data = {
            'summary': {
                'overall_pass': validation_result.overall_pass,
                'total_columns': validation_result.total_columns,
                'passed_columns': validation_result.passed_columns,
                'failed_columns': validation_result.failed_columns,
                'error_columns': validation_result.error_columns,
                'total_intersections': validation_result.total_intersections,
                'total_gaps': validation_result.total_gaps
            },
            'results': {}
        }
        
        for column, result in validation_result.results.items():
            data['results'][column] = {
                'is_disjoint': result.is_disjoint,
                'intersection_count': result.intersection_count,
                'intersection_samples': result.intersection_samples,
                'clean_coverage_rate': result.clean_coverage_rate,
                'dirty_coverage_rate': result.dirty_coverage_rate,
                'gap_rate': result.gap_rate,
                'validation_status': result.validation_status
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n✓ Validation results saved to {output_path}")
    
    def _invoke_predicate(self, func, value, row):
        """Invoke a predicate that may accept (value), (row), or (value, row)."""
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
    
    def get_validation_summary(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Get a summary dictionary of validation results.
        
        Args:
            validation_result: ValidationResult from validate_batch()
        
        Returns:
            Dictionary with summary metrics
        """
        return {
            'overall_pass': validation_result.overall_pass,
            'total_columns': validation_result.total_columns,
            'passed_columns': validation_result.passed_columns,
            'failed_columns': validation_result.failed_columns,
            'error_columns': validation_result.error_columns,
            'pass_rate': validation_result.passed_columns / validation_result.total_columns if validation_result.total_columns > 0 else 0.0,
            'total_intersections': validation_result.total_intersections,
            'total_gaps': validation_result.total_gaps,
            'results': {
                column: {
                    'status': result.validation_status,
                    'is_disjoint': result.is_disjoint,
                    'intersection_count': result.intersection_count,
                    'clean_coverage_rate': result.clean_coverage_rate,
                    'dirty_coverage_rate': result.dirty_coverage_rate,
                    'gap_rate': result.gap_rate
                }
                for column, result in validation_result.results.items()
            }
        }
