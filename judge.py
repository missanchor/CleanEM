"""
Judge with VR (Violation Rate) based selection logic and Dual-Verification (P_clean/P_dirty).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
import re
from dual_types import DualRule, DualEvaluationResult, RefinementRound
from core.utils import safe_dict, safe_not
try:
    from dateutil.parser import parse  # type: ignore
except Exception:
    parse = None


class Judge:
    """
    Judge class that evaluates rules based on Violation Rate (VR).

    Philosophy: "Errors are rare deviations from a strong rule."
    VR = count(Rule == False) / total_rows

    Criteria:
    - Accept: 0 <= VR < threshold (e.g., 0.05)
    - VR = 0: Rule is trivial (everyone follows it)
    - VR > threshold: Rule is too strict or wrong
    - Small Non-Zero VR: Indicates Anomalies (Errors)
    """

    def __init__(self, threshold: float = 0.1, violation_threshold: float = 0.5):
        """Initialize Judge with VR threshold.

        Args:
            threshold: VR threshold for rule evaluation
            violation_threshold: Threshold for rejecting rules that would flag too many violations in refinement
        """
        self.threshold = threshold
        self.violation_threshold = violation_threshold
        self.evaluation_results = {}
        self.default_dirty_rate_prior = 0.2

    def evaluate_rules(self, df: pd.DataFrame, rules: Dict[str, list],
                        rule_type: str = "dirty") -> Dict[str, list]:
        """
        Evaluate all rules and return results with VR analysis.

        Args:
            df: DataFrame to evaluate
            rules: Dictionary of {column: [(agent_name, rule_string), ...]}
            rule_type: "dirty" for error detection rules, "clean" for clean rules.
                        For dirty: True = violation (error detected)
                        For clean: False = violation (value doesn't meet clean criteria)

        Returns:
            Dictionary with evaluation results and VR analysis per column
        """
        results = {}

        for column, candidate_rules in rules.items():
            print(f"\n{'='*80}")
            rule_type_label = "Clean Rules" if rule_type == "clean" else "Candidate Rules"
            print(f"Evaluating {len(candidate_rules)} {rule_type_label} for column: {column}")
            print(f"{'='*80}")
            
            col_results = []
            
            for agent_name, rule_string in candidate_rules:
                print(f"\n  Agent: {agent_name}")
                print(f"  Rule: {rule_string[:100]}...")

                # Compile the lambda function
                try:
                    rule_func = eval(rule_string, safe_dict)

                    # Apply rule to each row/value
                    violations = []
                    for idx, row in df.iterrows():
                        try:
                            value = row[column]
                            predicate_result = self._invoke_predicate(rule_func, value, row)
                            # For dirty rules: True = violation (error detected)
                            # For clean rules: False = violation (value doesn't meet clean criteria)
                            is_violation = predicate_result if rule_type == "dirty" else not predicate_result
                            if is_violation:
                                violations.append({
                                    'row_index': idx,
                                    'value': value,
                                    'column': column
                                })
                        except Exception as e:
                            # If rule fails on a value, count as violation for both types
                            violations.append({
                                'row_index': idx,
                                'value': row[column],
                                'column': column,
                                'error': str(e)
                            })

                    total_rows = len(df)
                    violation_count = len(violations)
                    vr = violation_count / total_rows if total_rows > 0 else 0

                    result = {
                        'agent': agent_name,
                        'rule_string': rule_string,
                        'rule_function': rule_func,
                        'total_rows': total_rows,
                        'violation_count': violation_count,
                        'violation_rate': vr,
                        'violations': violations,
                        'status': self._evaluate_vr(vr)
                    }
                    col_results.append(result)

                    print(f"    Violations: {violation_count}/{total_rows} (VR: {vr:.4f})")
                    print(f"    Status: {self._evaluate_vr(vr)}")

                except Exception as e:
                    print(f"    Error evaluating rule: {e}")
                    col_results.append({
                        'agent': agent_name,
                        'rule_string': rule_string,
                        'error': str(e),
                        'status': 'error',
                        'violation_count': 0,
                        'violation_rate': 0
                    })
            
            results[column] = col_results

        self.evaluation_results = results
        return results

    def _evaluate_vr(self, vr: float) -> str:
        """Evaluate the violation rate."""
        if vr > self.threshold:
            return "reject_too_strict"
        elif 0 <= vr <= self.threshold:
            return "accept_anomaly"
        else:
            return "unknown"

    def get_accepted_rules(self, results: Dict[str, list]) -> Dict[str, list]:
        """
        Get all accepted rules for each column based on VR.

        Strategy: Return all rules with "accept_anomaly" status (0 <= VR <= threshold).
        This allows using multiple rules per column for more comprehensive error detection.
        """
        accepted_rules = {}

        for column, candidate_results in results.items():
            print(f"\nCollecting accepted rules for column: {column}")
            print(f"  {len(candidate_results)} candidates evaluated")

            # Filter for accepted rules
            accepted = [r for r in candidate_results if r['status'] == 'accept_anomaly']

            if not accepted:
                print(f"  ✗ No rules passed VR criteria")
                continue

            # Sort by VR (smallest first) but keep all accepted rules
            accepted.sort(key=lambda x: x['violation_rate'])
            accepted_rules[column] = accepted

            print(f"  ✓ Selected {len(accepted)} rule(s):")
            for rule in accepted:
                print(f"    - {rule['agent']}: VR={rule['violation_rate']:.4f}")
                print(f"      Rule: {rule['rule_string']}...")

        return accepted_rules

    def get_detected_errors(self, dirty_rules: Dict[str, list],
                             clean_rules: Dict[str, list] = None) -> List[Dict[str, Any]]:
        """
        Extract all detected errors using combined AND/OR logic.

        Logic:
        - Clean Rules (AND): All clean rules must be satisfied for a value to be considered clean
        - Dirty Rules (OR): Violating any dirty rule marks a value as potentially dirty
        - Error = (NOT all clean rules satisfied) AND (at least one dirty rule violated)

        Args:
            dirty_rules: Dictionary with {column: [rule_result1, rule_result2, ...]} (OR logic)
            clean_rules: Dictionary with {column: [rule_result1, rule_result2, ...]} (AND logic).
                        If None, falls back to original OR-only logic.

        Returns:
            List of detected errors (deduplicated by row_index and column)
        """
        # Backward compatibility: if clean_rules is None, use original OR-only logic
        if clean_rules is None:
            all_errors = []
            for column, rule_results in dirty_rules.items():
                for result in rule_results:
                    for violation in result['violations']:
                        error_info = {
                            'row_index': violation['row_index'],
                            'column': column,
                            'value': violation['value'],
                            'violated_rule': result['rule_string'],
                            'violated_agent': result['agent'],
                            'violation_rate': result['violation_rate'],
                            'detection_type': 'dirty_only'
                        }
                        all_errors.append(error_info)

            # Deduplicate errors by (row_index, column)
            seen_errors = {}
            for error in all_errors:
                key = (error['row_index'], error['column'])
                if key not in seen_errors:
                    seen_errors[key] = error
                else:
                    if error['violation_rate'] < seen_errors[key]['violation_rate']:
                        seen_errors[key] = error

            unique_errors = list(seen_errors.values())
            unique_errors.sort(key=lambda x: x['row_index'])
            return unique_errors

        # New combined AND/OR logic
        detected_errors = []

        # Get all columns from both rule sets
        all_columns = set(dirty_rules.keys()) | set(clean_rules.keys())

        # Get sample row to understand DataFrame structure
        # We'll need to re-evaluate rules on actual data
        # Since we don't have access to the original DataFrame here,
        # we use the stored evaluation results to build masks

        # For each column, build clean and dirty masks
        for column in all_columns:
            dirty_results = dirty_rules.get(column, [])
            clean_results = clean_rules.get(column, [])

            if not dirty_results and not clean_results:
                continue

            # Build clean_mask: True if value satisfies ALL clean rules (AND)
            # A value is clean only if ALL clean rules return True for it
            # We need to check each value against all clean rules
            # clean_mask[row_idx] = all(clean_rule_i(row_idx) == True for i in all clean rules)

            # Build dirty_mask: True if value violates ANY dirty rule (OR)
            # dirty_mask[row_idx] = any(dirty_rule_i(row_idx) == True for i in all dirty rules)

            # First, collect all violations and clean hits per row
            # From the stored results, we can reconstruct this

            # For clean rules: collect rows that satisfy EACH clean rule
            # A row is in clean_hits for rule_i if it's NOT in rule_i's violations
            # A row is fully clean (clean_mask=True) only if it's in ALL clean_hits

            # For dirty rules: collect rows that violate EACH dirty rule
            # A row is in dirty_hits for rule_i if it IS in rule_i's violations
            # A row is dirty (dirty_mask=True) if it's in ANY dirty_hits

            # Get total rows from first rule result
            total_rows = dirty_results[0]['total_rows'] if dirty_results else \
                        (clean_results[0]['total_rows'] if clean_results else 0)

            # Build clean_mask: row is clean only if it passes ALL clean rules
            clean_mask = [True] * total_rows
            for result in clean_results:
                for violation in result['violations']:
                    idx = violation['row_index']
                    if idx < total_rows:
                        clean_mask[idx] = False

            # Build dirty_mask: row is dirty if it violates ANY dirty rule
            dirty_mask = [False] * total_rows
            for result in dirty_results:
                for violation in result['violations']:
                    idx = violation['row_index']
                    if idx < total_rows:
                        dirty_mask[idx] = True

            # Error = (NOT clean) AND (dirty) = clean_mask=False AND dirty_mask=True
            for idx in range(total_rows):
                if not clean_mask[idx] and dirty_mask[idx]:
                    # Find the dirty rule that caught this error
                    violated_rule = None
                    violated_agent = None
                    min_vr = float('inf')
                    for result in dirty_results:
                        # Check if this row is in the violations
                        for violation in result['violations']:
                            if violation['row_index'] == idx:
                                if result['violation_rate'] < min_vr:
                                    min_vr = result['violation_rate']
                                    violated_rule = result['rule_string']
                                    violated_agent = result['agent']
                                break

                    # Get the value (from first dirty result's violations)
                    value = None
                    for result in dirty_results:
                        for violation in result['violations']:
                            if violation['row_index'] == idx:
                                value = violation['value']
                                break
                        if value is not None:
                            break

                    detected_errors.append({
                        'row_index': idx,
                        'column': column,
                        'value': value,
                        'violated_rule': violated_rule,
                        'violated_agent': violated_agent,
                        'violation_rate': min_vr if min_vr != float('inf') else 0,
                        'detection_type': 'combined_AND_OR'
                    })

        # Sort by row index
        detected_errors.sort(key=lambda x: x['row_index'])
        return detected_errors

    def print_summary(self, accepted_rules: Dict[str, list], rule_type: str = "dirty"):
        """
        Print a summary of the evaluation.

        Args:
            accepted_rules: Dictionary with {column: [rule_result1, ...]}
            rule_type: Type of rules being evaluated ("dirty" for error rules, "clean" for clean rules)
        """
        print("\n" + "="*80)
        rule_type_label = "Clean Rules" if rule_type == "clean" else "Rule Evaluation Results"
        print(f"JUDGE SUMMARY - {rule_type_label}")
        print("="*80)

        for column, results in accepted_rules.items():
            print(f"\nColumn: {column}")
            print(f"  Number of accepted rules: {len(results)}")
            for i, result in enumerate(results, 1):
                print(f"\n  Rule {i}:")
                print(f"    Name: {result['agent']}")
                print(f"    Status: {result['status']}")
                print(f"    Violation Rate: {result['violation_rate']:.4f}")
                print(f"    Violations: {result['violation_count']}/{result['total_rows']}")
                print(f"    Rule: {result['rule_string'][:100]}...")

    def print_detected_errors(self, errors: List[Dict[str, Any]]):
        """Print detected errors in a formatted table."""
        if not errors:
            print("\n✓ No errors detected!")
            return

        print("\n" + "="*80)
        print("DETECTED ERRORS")
        print("="*80)

        # Group by column for better presentation
        by_column = {}
        for error in errors:
            col = error['column']
            if col not in by_column:
                by_column[col] = []
            by_column[col].append(error)

        for column, col_errors in by_column.items():
            print(f"\n{column} ({len(col_errors)} errors):")
            print("-" * 80)
            for error in col_errors:
                print(f"  Row {error['row_index']}: '{error['value']}' - Rule: {error['violated_rule'][:60]}...")

    def evaluate_with_ground_truth(self, dirty_df: pd.DataFrame, clean_df: pd.DataFrame,
                                    detected_errors: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate detected errors against ground truth (clean data).

        Args:
            dirty_df: DataFrame with errors
            clean_df: Clean DataFrame (ground truth)
            detected_errors: List of detected errors

        Returns:
            Dictionary with per-column and overall metrics (precision, recall, f1)
        """
        # Create ground truth error set
        ground_truth_errors = set()

        # Ensure clean_df has the same columns as dirty_df (best-effort intersection)
        common_columns = [col for col in dirty_df.columns if col in clean_df.columns]
        if len(common_columns) != len(dirty_df.columns):
            missing_cols = set(dirty_df.columns) - set(clean_df.columns)
            if missing_cols:
                print(f"Warning: clean_df is missing columns: {missing_cols}")

        for idx in range(len(dirty_df)):
            dirty_row = dirty_df.iloc[idx]
            clean_row = clean_df.iloc[idx]
            for col in common_columns:
                if col not in clean_row.index:
                    continue
                dirty_val = dirty_row[col]
                clean_val = clean_row[col]
                if str(dirty_val).strip() != str(clean_val).strip():
                    ground_truth_errors.add((idx, col))

        # Create detected error set
        detected_error_set = set()
        for error in detected_errors:
            detected_error_set.add((error['row_index'], error['column']))

        # Calculate overall metrics
        tp = len(detected_error_set & ground_truth_errors)
        fp = len(detected_error_set - ground_truth_errors)
        fn = len(ground_truth_errors - detected_error_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate per-column metrics
        per_column_metrics = {}
        columns = dirty_df.columns

        for col in columns:
            col_gt_errors = {(idx, c) for idx, c in ground_truth_errors if c == col}
            col_detected_errors = {(idx, c) for idx, c in detected_error_set if c == col}

            col_tp = len(col_gt_errors & col_detected_errors)
            col_fp = len(col_detected_errors - col_gt_errors)
            col_fn = len(col_gt_errors - col_detected_errors)

            col_precision = col_tp / (col_tp + col_fp) if (col_tp + col_fp) > 0 else 0
            col_recall = col_tp / (col_tp + col_fn) if (col_tp + col_fn) > 0 else 0
            col_f1 = 2 * (col_precision * col_recall) / (col_precision + col_recall) if (col_precision + col_recall) > 0 else 0

            per_column_metrics[col] = {
                'precision': col_precision,
                'recall': col_recall,
                'f1': col_f1,
                'true_positives': col_tp,
                'false_positives': col_fp,
                'false_negatives': col_fn,
                'total_ground_truth_errors': len(col_gt_errors),
                'total_detected_errors': len(col_detected_errors)
            }

        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'total_ground_truth_errors': len(ground_truth_errors),
                'total_detected_errors': len(detected_error_set)
            },
            'per_column': per_column_metrics
        }

    def analyze_per_column(self, dirty_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze errors per column in detail.

        Args:
            dirty_df: DataFrame with errors
            clean_df: Clean DataFrame (ground truth)

        Returns:
            Dictionary with per-column analysis
        """
        print("\n" + "="*80)
        print("DETAILED PER-COLUMN ERROR ANALYSIS")
        print("="*80)

        per_column_analysis = {}

        for col in dirty_df.columns:
            print(f"\n{col}:")
            print("-" * 80)

            # Count errors in this column
            col_errors = []
            for idx in range(len(dirty_df)):
                dirty_val = dirty_df.iloc[idx][col]
                clean_val = clean_df.iloc[idx][col]
                if str(dirty_val).strip() != str(clean_val).strip():
                    col_errors.append({
                        'row_index': idx,
                        'dirty_value': dirty_val,
                        'clean_value': clean_val,
                        'error_type': self._classify_error(str(dirty_val), str(clean_val))
                    })

            if col_errors:
                print(f"  Total Errors: {len(col_errors)}")
                print(f"  Error Rate: {len(col_errors)/len(dirty_df)*100:.2f}%")

                # Analyze error types
                error_types = {}
                for error in col_errors:
                    error_type = error['error_type']
                    if error_type not in error_types:
                        error_types[error_type] = []
                    error_types[error_type].append(error)

                print(f"\n  Error Types:")
                for error_type, errors in error_types.items():
                    print(f"    - {error_type}: {len(errors)} errors")

                # Show examples
                print(f"\n  Examples (first 5):")
                for i, error in enumerate(col_errors[:5]):
                    print(f"    {i+1}. Row {error['row_index']}: '{error['dirty_value']}' -> '{error['clean_value']}' ({error['error_type']})")

                per_column_analysis[col] = {
                    'total_errors': len(col_errors),
                    'error_rate': len(col_errors)/len(dirty_df),
                    'error_types': {k: len(v) for k, v in error_types.items()},
                    'examples': col_errors[:5]
                }
            else:
                print(f"  ✓ No errors found!")
                per_column_analysis[col] = {
                    'total_errors': 0,
                    'error_rate': 0,
                    'error_types': {},
                    'examples': []
                }

        return per_column_analysis

    def _classify_error(self, dirty_val: str, clean_val: str) -> str:
        """Classify the type of error."""
        dirty_val = str(dirty_val).strip()
        clean_val = str(clean_val).strip()

        # Character substitution (x in wrong position)
        if 'x' in dirty_val.lower() and all(c.isdigit() or c.lower() == 'x' for c in dirty_val):
            return "character_substitution"

        # Typo (character insertion/deletion/substitution)
        if len(dirty_val) == len(clean_val):
            diff_count = sum(1 for a, b in zip(dirty_val, clean_val) if a != b)
            if diff_count > 0:
                return f"typo_{diff_count}_char_diff"
        elif abs(len(dirty_val) - len(clean_val)) <= 2:
            return "length_mismatch"
        else:
            return "major_difference"

        return "other"

    def print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Print a comprehensive evaluation summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)

        overall = evaluation_results['overall']
        per_column = evaluation_results['per_column']

        print(f"\n{'='*80}")
        print(f"OVERALL PERFORMANCE")
        print(f"{'='*80}")
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall: {overall['recall']:.4f}")
        print(f"F1 Score: {overall['f1']:.4f}")
        print(f"\nTotal Ground Truth Errors: {overall['total_ground_truth_errors']}")
        print(f"Total Detected Errors: {overall['total_detected_errors']}")
        print(f"True Positives: {overall['true_positives']}")
        print(f"False Positives: {overall['false_positives']}")
        print(f"False Negatives: {overall['false_negatives']}")

        # Best and worst performing columns
        print(f"\n{'='*80}")
        print(f"PER-COLUMN PERFORMANCE RANKING")
        print(f"{'='*80}")

        # Sort columns by F1 score
        sorted_cols = sorted(per_column.items(), key=lambda x: x[1]['f1'], reverse=True)

        for i, (col, metrics) in enumerate(sorted_cols, 1):
            print(f"\n{i}. {col}")
            print(f"   F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"   GT Errors: {metrics['total_ground_truth_errors']} | Detected: {metrics['total_detected_errors']}")

    # ============================================================================
    # DUAL VERIFICATION METHODS (P_clean/P_dirty)
    # ============================================================================

    def _estimate_dirty_rate_prior(self, metadata: Optional[Dict[str, Any]]) -> float:
        if not metadata:
            return float(self.default_dirty_rate_prior)

        total_rows = int(metadata.get("total_rows") or 0)
        missing_counts = metadata.get("missing_token_counts") or {}
        missing_total = sum(int(v) for v in missing_counts.values())
        missing_ratio = (missing_total / total_rows) if total_rows > 0 else 0.0

        col_type = metadata.get("type")
        if col_type == "text":
            base = 0.2
        elif col_type == "categorical":
            base = 0.15
        elif col_type == "pattern":
            base = 0.12
        else:
            base = 0.1

        non_null = metadata.get("non_null_count")
        if non_null is None:
            null_count = int(metadata.get("null_count") or 0)
            non_null = max(total_rows - null_count, 0)
        unique_count = metadata.get("unique_count")
        unique_ratio = None
        if unique_count is not None and non_null:
            unique_ratio = float(unique_count) / float(non_null)

        prior = max(base, min(0.3, float(missing_ratio) + 0.05))
        if unique_ratio is not None and unique_ratio > 0.9:
            prior = min(prior, 0.08)
        if missing_ratio > 0.4:
            prior = max(prior, min(0.5, float(missing_ratio) + 0.05))

        return float(max(0.02, min(0.5, prior)))

    def evaluate_dual_rules(
        self,
        df: pd.DataFrame,
        dual_rules: Dict[str, List[Tuple[str, str, str]]],
        grey_tolerance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[DualEvaluationResult]]:
        """
        Evaluate dual rules (P_clean/P_dirty pairs) and classify into four zones.

        Args:
            df: DataFrame to evaluate
            dual_rules: Dict[column] = List[(agent_name, clean_rule_str, dirty_rule_str)]
            grey_tolerance: Maximum acceptable grey zone rate

        Returns:
            Dict[column] = List[DualEvaluationResult]
        """
        print("\n" + "="*80)
        print("EVALUATING DUAL RULES (P_clean/P_dirty)")
        print("="*80)

        results = {}

        def call_predicate(pred, value, row=None):
            return self._invoke_predicate(pred, value, row)

        for column, candidate_rule_tuples in dual_rules.items():
            print(f"\n{'='*80}")
            print(f"Evaluating dual rules for column: {column}")
            print(f"{'='*80}")

            col_results = []
            column_metadata = metadata.get(column, {}) if metadata else None
            dirty_rate_prior = self._estimate_dirty_rate_prior(column_metadata) if metadata else None

            for agent_name, clean_rule_str, dirty_rule_str in candidate_rule_tuples:
                print(f"\n  Agent: {agent_name}")
                print(f"  P_clean: {clean_rule_str}")
                print(f"  P_dirty: {dirty_rule_str}")

                try:
                    # Compile lambda functions
                    clean_func = eval(clean_rule_str, safe_dict)
                    dirty_func = eval(dirty_rule_str, safe_dict)

                    # Classify each value into four zones
                    conflict_count = 0
                    grey_count = 0
                    determined_clean_count = 0
                    determined_dirty_count = 0

                    conflict_samples = []
                    grey_samples = []
                    determined_clean_samples = []
                    determined_dirty_samples = []
                    clean_flags: List[bool] = []
                    dirty_flags: List[bool] = []

                    # Sample collection for debugging
                    from collections import Counter
                    value_counts = Counter()

                    for idx, row in df.iterrows():
                        value = row[column]
                        value_counts[value] += 1

                        try:
                            is_clean = bool(self._invoke_predicate(clean_func, value, row))
                            is_dirty = bool(self._invoke_predicate(dirty_func, value, row))
                        except Exception:
                            # Predicate execution error -> grey zone
                            is_clean = False
                            is_dirty = False
                        clean_flags.append(is_clean)
                        dirty_flags.append(is_dirty)

                        # Four-zone classification
                        if is_clean and is_dirty:
                            # Conflict: both predicates true
                            conflict_count += 1
                            if len(conflict_samples) < 10:
                                conflict_samples.append({
                                    'row_index': idx,
                                    'value': value,
                                    'count': value_counts[value]
                                })
                        elif is_clean and not is_dirty:
                            # Determined clean
                            determined_clean_count += 1
                            if len(determined_clean_samples) < 10:
                                determined_clean_samples.append({
                                    'row_index': idx,
                                    'value': value,
                                    'count': value_counts[value]
                                })
                        elif not is_clean and is_dirty:
                            # Determined dirty
                            determined_dirty_count += 1
                            if len(determined_dirty_samples) < 10:
                                determined_dirty_samples.append({
                                    'row_index': idx,
                                    'value': value,
                                    'count': value_counts[value]
                                })
                        else:
                            # Grey zone: both false
                            grey_count += 1
                            if len(grey_samples) < 10:
                                grey_samples.append({
                                    'row_index': idx,
                                    'value': value,
                                    'count': value_counts[value]
                                })

                    total_rows = len(df)
                    conflict_rate = conflict_count / total_rows if total_rows > 0 else 0
                    grey_rate = grey_count / total_rows if total_rows > 0 else 0
                    dirty_rate = determined_dirty_count / total_rows if total_rows > 0 else 0
                    clean_rate = determined_clean_count / total_rows if total_rows > 0 else 0

                    # Evaluate against constraints
                    status, violation_message = self._evaluate_dual_constraints(
                        conflict_rate,
                        grey_rate,
                        dirty_rate,
                        grey_tolerance,
                        dirty_rate_prior=dirty_rate_prior,
                    )

                    result = DualEvaluationResult(
                        column=column,
                        rule=DualRule(
                            column=column,
                            agent_name=agent_name,
                            clean_rule_str=clean_rule_str,
                            dirty_rule_str=dirty_rule_str,
                            clean_rule_func=clean_func,
                            dirty_rule_func=dirty_func
                        ),
                        total_rows=total_rows,
                        conflict_count=conflict_count,
                        grey_count=grey_count,
                        determined_clean_count=determined_clean_count,
                        determined_dirty_count=determined_dirty_count,
                        conflict_rate=conflict_rate,
                        grey_rate=grey_rate,
                        dirty_rate=dirty_rate,
                        clean_rate=clean_rate,
                        conflict_samples=conflict_samples,
                        grey_samples=grey_samples,
                        determined_clean_samples=determined_clean_samples,
                        determined_dirty_samples=determined_dirty_samples,
                        status=status,
                        violation_message=violation_message,
                        dirty_rate_prior=dirty_rate_prior,
                    )

                    col_results.append(result)

                    # Print evaluation summary
                    print(f"\n    Classification:")
                    print(f"      Conflict (both true): {conflict_count} ({conflict_rate:.4f})")
                    print(f"      Gap (both false): {grey_count} ({grey_rate:.4f})")
                    print(f"      Determined Clean: {determined_clean_count} ({clean_rate:.4f})")
                    print(f"      Determined Dirty: {determined_dirty_count} ({dirty_rate:.4f})")
                    print(f"    Status: {status}")
                    if violation_message:
                        print(f"    Violation: {violation_message}")

                except Exception as e:
                    print(f"    ✗ Error evaluating rules: {e}")
                    # Create error result
                    error_result = DualEvaluationResult(
                        column=column,
                        rule=DualRule(
                            column=column,
                            agent_name=agent_name,
                            clean_rule_str=clean_rule_str,
                            dirty_rule_str=dirty_rule_str
                        ),
                        total_rows=len(df),
                        conflict_count=0,
                        grey_count=len(df),
                        determined_clean_count=0,
                        determined_dirty_count=0,
                        conflict_rate=0,
                        grey_rate=1.0,
                        dirty_rate=0,
                        clean_rate=0,
                        conflict_samples=[],
                        grey_samples=[],
                        determined_clean_samples=[],
                        determined_dirty_samples=[],
                        status='reject_error',
                        violation_message=str(e)
                    )
                    col_results.append(error_result)

            results[column] = col_results

        return results

    def _evaluate_dual_constraints(
        self,
        conflict_rate: float,
        grey_rate: float,
        dirty_rate: float,
        grey_tolerance: float,
        dirty_rate_prior: Optional[float] = None,
    ) -> Tuple[str, str]:
        """
        Evaluate if dual rule pair meets hard constraints.

        Constraints:
        1. conflict_rate == 0 (no conflicts: P_clean and P_dirty both true)
        2. grey_rate <= grey_tolerance (no gaps: at least one of P_clean/P_dirty true)
        3. dirty_rate < 1.0 (NOT all dirty)
        4. dirty_rate >= 0 (always true)

        Returns:
            Tuple of (status, message)
        """
        if dirty_rate == 1.0:
            return 'reject_all_dirty', f"Entire column marked dirty (dirty_rate={dirty_rate:.4f})"

        if conflict_rate > 0:
            return 'reject_conflict', f"Conflict zone exists: P_clean AND P_dirty both true (conflict_rate={conflict_rate:.4f})"

        if grey_rate > grey_tolerance:
            return 'reject_gap', f"Gap zone too large: P_clean AND P_dirty both false (gap_rate={grey_rate:.4f} > {grey_tolerance:.4f})"

        if dirty_rate_prior is not None:
            prior_limit = min(0.95, float(dirty_rate_prior) * 3.0)
            if dirty_rate > prior_limit:
                return 'reject_dirty_prior', f"Dirty rate exceeds prior (dirty_rate={dirty_rate:.4f} > prior={prior_limit:.4f})"

        if dirty_rate == 0:
            return 'accept_all_clean', f"Column is clean (dirty_rate={dirty_rate:.4f})"

        message = f"Valid dual rule (dirty_rate={dirty_rate:.4f})"
        return 'accept', message

    def _select_refinement_candidate(self, results: List[DualEvaluationResult]) -> DualEvaluationResult:
        """Pick the result that most urgently needs refinement."""
        def score(result: DualEvaluationResult) -> float:
            base = result.conflict_rate + result.grey_rate
            if result.status == 'reject_all_dirty':
                base += 10
            elif result.status == 'reject_conflict':
                base += 5
            elif result.status == 'reject_grey':
                base += 2
            return base

        return max(results, key=lambda r: (score(r), r.dirty_rate))

    def _generate_repair_candidates(self, column: str, clean_rule_str: str, dirty_rule_str: str,
                                   conflict_samples: List[Dict[str, Any]],
                                   gap_samples: List[Dict[str, Any]],
                                   df: pd.DataFrame, col_metadata: Dict[str, Any],
                                   factory,
                                   evaluation_result: Optional[DualEvaluationResult] = None) -> Dict[str, Tuple[str, str]]:
        """
        Generate multiple candidate repairs for a problematic column.

        Returns:
            Dict[candidate_id] = (new_clean_rule_str, new_dirty_rule_str)
        """
        candidates = {}

        # Candidate 1: Modify P_clean for conflicts
        if conflict_samples:
            try:
                new_clean = factory.generate_p_clean_predicates_per_column(
                    {column: col_metadata},
                    refinement_context={
                        column: {'conflict_samples': conflict_samples, 'gap_samples': []}
                    }
                )
                if column in new_clean and new_clean[column]:
                    candidates['repair_clean_vs_conflict'] = (new_clean[column], dirty_rule_str)
                    print(f"      Generated: repair_clean_vs_conflict")
            except Exception as e:
                print(f"      Failed to generate repair_clean_vs_conflict: {e}")

        # Candidate 2: Modify P_dirty for conflicts
        if conflict_samples:
            try:
                new_dirty = factory.generate_p_dirty_predicates_per_column(
                    {column: col_metadata},
                    refinement_context={
                        column: {'conflict_samples': conflict_samples, 'gap_samples': []}
                    }
                )
                if column in new_dirty and new_dirty[column]:
                    candidates['repair_dirty_vs_conflict'] = (clean_rule_str, new_dirty[column])
                    print(f"      Generated: repair_dirty_vs_conflict")
            except Exception as e:
                print(f"      Failed to generate repair_dirty_vs_conflict: {e}")

        # Candidate 3: Expand P_clean for gaps
        if gap_samples:
            try:
                new_clean = factory.generate_p_clean_predicates_per_column(
                    {column: col_metadata},
                    refinement_context={
                        column: {'gap_samples': gap_samples, 'conflict_samples': []}
                    }
                )
                if column in new_clean and new_clean[column]:
                    candidates['expand_clean_for_gaps'] = (new_clean[column], dirty_rule_str)
                    print(f"      Generated: expand_clean_for_gaps")
            except Exception as e:
                print(f"      Failed to generate expand_clean_for_gaps: {e}")

        # Candidate 4: Expand P_dirty for gaps
        if gap_samples:
            try:
                new_dirty = factory.generate_p_dirty_predicates_per_column(
                    {column: col_metadata},
                    refinement_context={
                        column: {'gap_samples': gap_samples, 'conflict_samples': []}
                    }
                )
                if column in new_dirty and new_dirty[column]:
                    candidates['expand_dirty_for_gaps'] = (clean_rule_str, new_dirty[column])
                    print(f"      Generated: expand_dirty_for_gaps")
            except Exception as e:
                print(f"      Failed to generate expand_dirty_for_gaps: {e}")

        return candidates

    def _score_candidates(self, candidate_results: Dict[str, DualEvaluationResult],
                         grey_tolerance: float,
                         dirty_rate_prior: Optional[float] = None) -> str:
        """
        Score candidates and pick the best based on conflict/grey/dirty rates.
        """
        if not candidate_results:
            return None

        # Filter by hard constraints
        valid = {
            cand_id: result for cand_id, result in candidate_results.items()
            if result.conflict_rate == 0 and result.grey_rate <= grey_tolerance
        }

        if not valid:
            # If no candidate meets constraints, pick the one closest to meeting them
            print(f"      ⚠ No candidate fully meets constraints; picking least-bad option")
            best_id = min(candidate_results.keys(),
                         key=lambda cid: (
                             candidate_results[cid].conflict_rate,
                             candidate_results[cid].grey_rate - grey_tolerance,
                             candidate_results[cid].dirty_rate
                         ))
            return best_id

        def candidate_score(result: DualEvaluationResult) -> float:
            score = float(result.dirty_rate)
            if dirty_rate_prior is not None:
                prior_limit = min(0.95, float(dirty_rate_prior) * 3.0)
                if result.dirty_rate > prior_limit:
                    score += (result.dirty_rate - prior_limit) * 2.0
            return score

        best_id = min(valid.keys(), key=lambda cid: candidate_score(valid[cid]))
        return best_id

    def _analyze_conflict_topology(self, df: pd.DataFrame, column: str,
                                   clean_func, dirty_func) -> Dict[str, Any]:
        from core.deducer import analyze_conflict_topology
        import numpy as np

        clean_mask = []
        dirty_mask = []
        for _, row in df.iterrows():
            value = row[column]
            try:
                clean_mask.append(bool(self._invoke_predicate(clean_func, value, row)))
                dirty_mask.append(bool(self._invoke_predicate(dirty_func, value, row)))
            except Exception:
                clean_mask.append(False)
                dirty_mask.append(False)

        topology = analyze_conflict_topology(
            df,
            column,
            np.array(clean_mask),
            np.array(dirty_mask)
        )

        return {
            'column': topology.column,
            'conflict_type': topology.conflict_type.value,
            'clean_set_size': topology.clean_set_size,
            'dirty_set_size': topology.dirty_set_size,
            'intersection_size': topology.intersection_size,
            'union_size': topology.union_size,
            'iou': topology.iou,
            'clean_in_intersection_ratio': topology.clean_in_intersection_ratio,
            'dirty_in_intersection_ratio': topology.dirty_in_intersection_ratio,
            'intersection_samples': topology.intersection_samples,
            'clean_only_samples': topology.clean_only_samples,
            'dirty_only_samples': topology.dirty_only_samples,
            'total_rows': topology.total_rows
        }

    def _generate_refinement_task(self, df: pd.DataFrame, column: str,
                                  evaluation_result: DualEvaluationResult,
                                  metadata: Dict[str, Any],
                                  em_scores: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        from core.deducer import analyze_conflict_topology, generate_refinement_task
        import numpy as np

        clean_rule_str = evaluation_result.rule.clean_rule_str
        dirty_rule_str = evaluation_result.rule.dirty_rule_str

        clean_func = evaluation_result.rule.clean_rule_func
        dirty_func = evaluation_result.rule.dirty_rule_func
        if clean_func is None:
            clean_func = eval(clean_rule_str, safe_dict)
        if dirty_func is None:
            dirty_func = eval(dirty_rule_str, safe_dict)

        clean_mask = []
        dirty_mask = []
        row_indices = []
        for idx, row in df.iterrows():
            value = row[column]
            row_indices.append(int(idx))
            try:
                clean_mask.append(bool(self._invoke_predicate(clean_func, value, row)))
                dirty_mask.append(bool(self._invoke_predicate(dirty_func, value, row)))
            except Exception:
                clean_mask.append(False)
                dirty_mask.append(False)

        topology = analyze_conflict_topology(
            df,
            column,
            np.array(clean_mask),
            np.array(dirty_mask)
        )
        task = generate_refinement_task(
            topology,
            clean_rule_str,
            dirty_rule_str,
            metadata
        )

        conflict_samples = evaluation_result.conflict_samples
        grey_samples = evaluation_result.grey_samples

        return {
            'column': task.column,
            'conflict_type': task.conflict_type.value,
            'current_clean_rule': task.current_clean_rule,
            'current_dirty_rule': task.current_dirty_rule,
            'conflict_samples': conflict_samples,
            'grey_samples': grey_samples,
            'strategy': task.strategy,
            'metadata': task.metadata
        }

    def refine_dual_rules(self, df: pd.DataFrame, metadata: Dict[str, Any],
                          initial_rules: Dict[str, List[Tuple[str, str, str]]],
                          max_rounds: int = 3, grey_tolerance: float = 0.0,
                          factory=None) -> Tuple[Dict[str, DualRule], Dict[str, List[Dict[str, Any]]]]:
        current_rules = dict(initial_rules)
        refinement_history: Dict[str, List[Dict[str, Any]]] = {column: [] for column in current_rules}
        best_rules: Dict[str, DualRule] = {}

        for round_num in range(1, max_rounds + 1):
            evaluation_results = self.evaluate_dual_rules(
                df,
                current_rules,
                grey_tolerance=grey_tolerance,
                metadata=metadata,
            )

            for column, results in evaluation_results.items():
                if not results:
                    continue
                result = results[0]
                best_rules[column] = result.rule
                refinement_history[column].append({
                    'round': round_num,
                    'status': result.status,
                    'conflict_rate': result.conflict_rate,
                    'grey_rate': result.grey_rate,
                    'dirty_rate': result.dirty_rate,
                    'dirty_rate_prior': result.dirty_rate_prior
                })

            if not factory:
                break

            next_rules: Dict[str, List[Tuple[str, str, str]]] = {}
            for column, results in evaluation_results.items():
                if not results:
                    continue
                result = results[0]
                if result.status in ("accept", "accept_all_clean"):
                    next_rules[column] = [
                        (result.rule.agent_name, result.rule.clean_rule_str, result.rule.dirty_rule_str)
                    ]
                    continue

                candidates = self._generate_repair_candidates(
                    column,
                    result.rule.clean_rule_str,
                    result.rule.dirty_rule_str,
                    result.conflict_samples,
                    result.grey_samples,
                    df,
                    metadata.get(column, {}),
                    factory,
                    evaluation_result=result,
                )
                if candidates:
                    candidate_ids = list(candidates.keys())
                    candidate_rules = [
                        (result.rule.agent_name, candidates[cid][0], candidates[cid][1])
                        for cid in candidate_ids
                    ]
                    candidate_payload = {column: candidate_rules}
                    candidate_eval = self.evaluate_dual_rules(
                        df,
                        candidate_payload,
                        grey_tolerance=grey_tolerance,
                        metadata=metadata,
                    )
                    candidate_results: Dict[str, DualEvaluationResult] = {}
                    if column in candidate_eval:
                        for idx, cand_result in enumerate(candidate_eval[column]):
                            if idx < len(candidate_ids):
                                candidate_results[candidate_ids[idx]] = cand_result
                    best_id = self._score_candidates(
                        candidate_results,
                        grey_tolerance,
                        dirty_rate_prior=result.dirty_rate_prior,
                    )
                    if best_id and best_id in candidate_results:
                        picked = candidate_results[best_id]
                        current_penalty = result.conflict_rate + result.grey_rate
                        picked_penalty = picked.conflict_rate + picked.grey_rate
                        if picked.status in ("accept", "accept_all_clean") or picked_penalty <= current_penalty:
                            next_rules[column] = [
                                (picked.rule.agent_name, picked.rule.clean_rule_str, picked.rule.dirty_rule_str)
                            ]
                            continue

                task = self._generate_refinement_task(
                    df,
                    column,
                    result,
                    metadata.get(column, {}),
                    em_scores=None,
                )
                updated = self._execute_refinement_with_fallback(
                    df,
                    column,
                    task,
                    factory,
                    max_attempts=2
                )
                if updated:
                    next_rules[column] = [(result.rule.agent_name, updated[0], updated[1])]
                else:
                    next_rules[column] = [
                        (result.rule.agent_name, result.rule.clean_rule_str, result.rule.dirty_rule_str)
                    ]

            current_rules = next_rules

        return best_rules, refinement_history

    def get_detected_dirty_values(self, best_rules: Dict[str, DualRule], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get all values flagged as DIRTY by the best dual rules.

        A value is identified as an error only when:
        - P_clean(value) = False AND P_dirty(value) = True (Determined Dirty zone)

        Args:
            best_rules: Dict[column] = DualRule
            df: DataFrame to check

        Returns:
            List of detected dirty values with metadata
        """
        detected_errors = []

        for column, rule in best_rules.items():
            for idx, row in df.iterrows():
                value = row[column]

                try:
                    is_clean = self._invoke_predicate(rule.clean_rule_func, value, row)
                    is_dirty = self._invoke_predicate(rule.dirty_rule_func, value, row)
                    # Only flag as error when P_clean=False AND P_dirty=True
                    if not is_clean and is_dirty:
                        detected_errors.append({
                            'row_index': idx,
                            'column': column,
                            'value': value,
                            'agent': rule.agent_name,
                            'clean_rule': rule.clean_rule_str,
                            'dirty_rule': rule.dirty_rule_str,
                            'round': rule.round_number
                        })
                except Exception:
                    # Skip on error
                    pass

        return detected_errors

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

    def print_dual_summary(self, best_rules: Dict[str, DualRule],
                          evaluation_results: Dict[str, List[DualEvaluationResult]]):
        """Print summary of dual rule evaluation."""
        print("\n" + "="*80)
        print("DUAL VERIFICATION SUMMARY")
        print("="*80)

        for column, rule in best_rules.items():
            if column in evaluation_results:
                result = evaluation_results[column][0]  # Best result

                print(f"\n{column}:")
                print(f"  Agent: {rule.agent_name}")
                print(f"  Status: {result.status}")
                print(f"  Dirty Rate: {result.dirty_rate:.4f} ({result.determined_dirty_count}/{result.total_rows})")
                print(f"  Clean Rate: {result.clean_rate:.4f} ({result.determined_clean_count}/{result.total_rows})")
                print(f"  Gap Rate (both false): {result.grey_rate:.4f} ({result.grey_count}/{result.total_rows})")
                print(f"  Conflict Rate (both true): {result.conflict_rate:.4f} ({result.conflict_count}/{result.total_rows})")
                if result.dirty_rate_prior is not None:
                    print(f"  Dirty Rate Prior: {result.dirty_rate_prior:.4f}")
                print(f"\n  P_clean: {rule.clean_rule_str}")
                print(f"  P_dirty: {rule.dirty_rule_str}")

    def save_dual_results(self, best_rules: Dict[str, DualRule],
                         evaluation_results: Dict[str, List[DualEvaluationResult]],
                         refinement_history: Dict[str, List],
                         detected_errors: List[Dict[str, Any]],
                         coverage_gaps: List[str] = None,
                          output_dir: str = "results/agentic_error_detector"):
        """Save dual rule results to files."""
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)
        coverage_gaps = coverage_gaps or []

        # Save best rules
        rules_data = {}
        for column, rule in best_rules.items():
            rules_data[column] = {
                'agent': rule.agent_name,
                'clean_rule': rule.clean_rule_str,
                'dirty_rule': rule.dirty_rule_str,
                'round': rule.round_number
            }

        with open(f"{output_dir}/dual_rules.json", "w") as f:
            json.dump(rules_data, f, indent=2)

        # Save evaluation metrics
        eval_data = {}
        for column, results in evaluation_results.items():
            eval_data[column] = []
            for result in results:
                eval_data[column].append({
                    'agent': result.rule.agent_name,
                    'status': result.status,
                    'dirty_rate': result.dirty_rate,
                    'clean_rate': result.clean_rate,
                    'grey_rate': result.grey_rate,
                    'conflict_rate': result.conflict_rate,
                    'dirty_rate_prior': result.dirty_rate_prior,
                    'violation_message': result.violation_message,
                    'counts': {
                        'conflict': result.conflict_count,
                        'grey': result.grey_count,
                        'determined_clean': result.determined_clean_count,
                        'determined_dirty': result.determined_dirty_count,
                        'total': result.total_rows
                    }
                })

        with open(f"{output_dir}/dual_evaluation.json", "w") as f:
            json.dump(eval_data, f, indent=2)

        # Save refinement history
        history_data = {}
        for column, rounds in refinement_history.items():
            history_data[column] = rounds

        with open(f"{output_dir}/refinement_history.json", "w") as f:
            json.dump(history_data, f, indent=2, default=str)

        # Save detected dirty values
        with open(f"{output_dir}/detected_dirty_values.json", "w") as f:
            json.dump(detected_errors, f, indent=2, default=str)

        with open(f"{output_dir}/coverage_gaps.json", "w") as f:
            json.dump(coverage_gaps, f, indent=2)

        print(f"\n✓ Dual verification results saved to {output_dir}/")
        print(f"  - dual_rules.json")
        print(f"  - dual_evaluation.json")
        print(f"  - refinement_history.json")
        print(f"  - detected_dirty_values.json")
        print(f"  - coverage_gaps.json")

    def _select_intersection_samples(self, df: pd.DataFrame, column: str,
                                     clean_func, dirty_func, max_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Select representative samples from the intersection (conflict) zone.

        Args:
            df: DataFrame containing the data
            column: Column name
            clean_func: P_clean predicate function
            dirty_func: P_dirty predicate function
            max_samples: Maximum number of samples to return

        Returns:
            List of sample dictionaries with row_index, value
        """
        from core.deducer import select_intersection_samples

        # Build masks
        clean_mask = []
        dirty_mask = []
        for idx, row in df.iterrows():
            value = row[column]
            try:
                clean_mask.append(bool(self._invoke_predicate(clean_func, value, row)))
                dirty_mask.append(bool(self._invoke_predicate(dirty_func, value, row)))
            except Exception:
                clean_mask.append(False)
                dirty_mask.append(False)

        import numpy as np
        clean_mask = np.array(clean_mask)
        dirty_mask = np.array(dirty_mask)

        # Select samples
        samples = select_intersection_samples(df, column, clean_mask, dirty_mask, max_samples)

        # Add values from DataFrame
        for sample in samples:
            idx = sample['row_index']
            sample['value'] = df.iloc[idx][column]

        return samples

    def _validate_disjointness(self, df: pd.DataFrame, column: str,
                               clean_func, dirty_func) -> Dict[str, Any]:
        """
        Validate that P_clean and P_dirty are disjoint (no intersections).

        Args:
            df: DataFrame containing the data
            column: Column name
            clean_func: P_clean predicate function
            dirty_func: P_dirty predicate function

        Returns:
            Dictionary with validation results
        """
        from core.deducer import validate_disjointness

        result = validate_disjointness(df, column, clean_func, dirty_func)

        return {
            'column': result.column,
            'is_disjoint': result.is_disjoint,
            'intersection_count': result.intersection_count,
            'intersection_samples': result.intersection_samples,
            'clean_coverage_rate': result.clean_coverage_rate,
            'dirty_coverage_rate': result.dirty_coverage_rate,
            'gap_rate': result.gap_rate,
            'validation_status': result.validation_status
        }

    def _execute_refinement_with_fallback(self, df: pd.DataFrame, column: str,
                                         task: Dict[str, Any],
                                         factory, max_attempts: int = 3) -> Optional[Tuple[str, str]]:
        """
        Execute refinement with fallback strategy: clean -> dirty -> rollback.

        Args:
            df: DataFrame containing the data
            column: Column name
            task: Refinement task dictionary
            factory: AgentFactory instance for LLM generation
            max_attempts: Maximum number of refinement attempts

        Returns:
            Tuple of (new_clean_rule, new_dirty_rule) or None if all attempts fail
        """
        print(f"\n  Executing refinement with fallback (max {max_attempts} attempts)")

        current_clean = task['current_clean_rule']
        current_dirty = task['current_dirty_rule']
        strategy = task.get('strategy')
        if not strategy:
            conflict_type = task.get('conflict_type')
            if hasattr(conflict_type, "value"):
                conflict_type = conflict_type.value
            if conflict_type == 'SUBSET':
                strategy = 'clean'
            elif conflict_type == 'SUPERSET':
                strategy = 'dirty'
            elif conflict_type == 'INTERSECT':
                strategy = 'both'
            else:
                strategy = 'fallback'

        for attempt in range(1, max_attempts + 1):
            print(f"    Attempt {attempt}/{max_attempts} (strategy: {strategy})")

            try:
                updated_rule = False
                # Generate refined rule based on strategy
                if strategy == 'clean' or (attempt == 1 and strategy in ['both', 'fallback']):
                    new_clean = factory.generate_p_clean_predicates_per_column(
                        {column: task['metadata']},
                        refinement_context={
                            column: {
                                'conflict_samples': task['conflict_samples'],
                                'gap_samples': task['grey_samples'],
                                'disjointness_mode': True
                            }
                        }
                    )
                    if column in new_clean and new_clean[column]:
                        current_clean = new_clean[column]
                        updated_rule = True
                        print(f"      Generated new P_clean rule")
                    else:
                        print(f"      Failed to generate P_clean rule")

                elif strategy == 'dirty' or (attempt == 2 and strategy == 'both'):
                    new_dirty = factory.generate_p_dirty_predicates_per_column(
                        {column: task['metadata']},
                        refinement_context={
                            column: {
                                'conflict_samples': task['conflict_samples'],
                                'gap_samples': task['grey_samples'],
                                'disjointness_mode': True
                            }
                        }
                    )
                    if column in new_dirty and new_dirty[column]:
                        current_dirty = new_dirty[column]
                        updated_rule = True
                        print(f"      Generated new P_dirty rule")
                    else:
                        print(f"      Failed to generate P_dirty rule")

                # Compile and validate
                clean_func = eval(current_clean, safe_dict)
                dirty_func = eval(current_dirty, safe_dict)

                # Validate disjointness
                validation = self._validate_disjointness(df, column, clean_func, dirty_func)

                if validation['is_disjoint']:
                    print(f"      ✓ Disjointness validation passed!")
                    if updated_rule or attempt == max_attempts:
                        return (current_clean, current_dirty)
                    continue
                else:
                    print(f"      ✗ Disjointness validation failed: {validation['intersection_count']} intersections")

                    # Try alternative strategy
                    if strategy == 'clean' and attempt == 1:
                        strategy = 'dirty'
                        print(f"      Switching strategy to: {strategy}")
                    elif strategy == 'dirty' and attempt == 1:
                        strategy = 'both'
                        print(f"      Switching strategy to: {strategy}")
                    elif strategy in ['both', 'fallback'] and attempt < max_attempts:
                        # Continue with current strategy for more attempts
                        pass

            except Exception as e:
                print(f"      ✗ Attempt {attempt} failed: {e}")

        # All attempts failed, return None for rollback
        print(f"    ✗ All refinement attempts failed, rolling back")
        return None

    # =========================================================================================
    # CLEAN RULE-LEVEL REFINEMENT (Phase 2)
    # =========================================================================================

    def refine_clean_rules(self, df: pd.DataFrame,
                           column: str,
                           rule_set: 'CleanRuleSet',
                           factory=None,
                           max_rounds: int = 10,
                           conflict_tolerance: float = 0.01,
                           grey_tolerance: float = 0.01,
                           metadata: Dict[str, Any] = None,
                            output_dir: str = "results/agentic_error_detector",
                           logger=None,
                           console: Optional[Callable[[str], None]] = None) -> Tuple['CleanRuleSet', Dict]:
        """
        Refine clean rule-level rules iteratively to reduce conflicts and gaps.

        Args:
            df: DataFrame to evaluate
            column: Column name
            rule_set: Current CleanRuleSet
            factory: AgentFactory for LLM calls
            max_rounds: Maximum refinement rounds
            conflict_tolerance: Maximum acceptable conflict rate
            grey_tolerance: Maximum acceptable grey zone rate
            metadata: Column metadata (type, sample_values, etc.)
            output_dir: Output directory for logs
            logger: Optional RefinementLogger instance (if provided, will not create new one)

        Returns:
            Tuple of (refined CleanRuleSet, refinement history)
        """
        from modification_memory import ModificationMemory, start_logger, stop_logger, get_logger
        from conflict_resolver import ConflictResolver
        from gap_resolver import GapResolver
        from dual_types import CleanRuleSet
        from copy import deepcopy

        console_fn: Callable[[str], None]
        if console is None:
            console_fn = print
        else:
            console_fn = console

        console_fn("\n" + "=" * 80)
        console_fn(f"CLEAN RULE-LEVEL REFINEMENT: {column}")
        console_fn("=" * 80)

        # Use provided logger or create a new one (for backward compatibility)
        own_logger = None
        if logger is None:
            own_logger = start_logger(output_dir)
            logger = own_logger

        memory = ModificationMemory()
        history = {'rounds': [], 'column': column}

        # Compile rule functions if not already compiled
        rule_set = self._compile_rule_functions(rule_set)

        for round_num in range(1, max_rounds + 1):
            console_fn(f"\n--- Round {round_num}/{max_rounds} ---")

            # Backup current rules
            backup = deepcopy(rule_set)

            # Check current status BEFORE attempting refinement
            dual_rule = rule_set.to_dual_rule(agent_factory=factory)
            conflict_rate = self._count_conflicts(df, column, dual_rule) / len(df)
            grey_rate = self._count_grey_zone(df, column, dual_rule) / len(df)

            console_fn(f"  Current status - Conflict rate: {conflict_rate:.4f}, Grey rate: {grey_rate:.4f}")

            # Check termination - both conditions must be satisfied
            if conflict_rate <= conflict_tolerance and grey_rate <= grey_tolerance:
                console_fn(f"  ✓ Converged (conflict_rate={conflict_rate:.4f} <= {conflict_tolerance}, grey_rate={grey_rate:.4f} <= {grey_tolerance})")
                # Record the final state
                if hasattr(memory, 'add_round_summary'):
                    memory.add_round_summary(round_num, conflict_rate, grey_rate)
                history['rounds'].append({
                    'round': round_num,
                    'conflict_rate': conflict_rate,
                    'grey_rate': grey_rate,
                    'modifications': memory.to_context() if hasattr(memory, 'to_context') else str(memory)
                })
                break

            console_fn("  Resolving gaps...")
            gap_resolver = GapResolver(memory, factory, violation_threshold=self.violation_threshold)
            rule_set = gap_resolver.resolve(
                df,
                column,
                rule_set,
                metadata=metadata,
                round_num=round_num,
                em_scores=None,
            )

            console_fn("  Resolving conflicts...")
            conflict_resolver = ConflictResolver(memory, factory, violation_threshold=self.violation_threshold)
            rule_set = conflict_resolver.resolve(
                df,
                column,
                rule_set,
                metadata=metadata,
                round_num=round_num,
                em_scores=None,
            )

            # Recompile functions after refinement
            rule_set = self._compile_rule_functions(rule_set)

            # Check status after refinement
            dual_rule = rule_set.to_dual_rule(agent_factory=factory)
            conflict_rate = self._count_conflicts(df, column, dual_rule) / len(df)
            grey_rate = self._count_grey_zone(df, column, dual_rule) / len(df)

            console_fn(f"  After refinement - Conflict rate: {conflict_rate:.4f}, Grey rate: {grey_rate:.4f}")

            if hasattr(memory, 'add_round_summary'):
                memory.add_round_summary(round_num, conflict_rate, grey_rate)

            history['rounds'].append({
                'round': round_num,
                'conflict_rate': conflict_rate,
                'grey_rate': grey_rate,
                'modifications': memory.to_context() if hasattr(memory, 'to_context') else str(memory)
            })

            # Check termination after refinement
            if conflict_rate <= conflict_tolerance and grey_rate <= grey_tolerance:
                console_fn(f"  ✓ Converged after refinement (conflict_rate={conflict_rate:.4f} <= {conflict_tolerance}, grey_rate={grey_rate:.4f} <= {grey_tolerance})")
                break

            # Check if making progress
            if round_num > 1 and (conflict_rate > history['rounds'][-2]['conflict_rate'] or
                                  grey_rate > history['rounds'][-2]['grey_rate']):
                console_fn(f"  ⚠ No improvement (conflict: {conflict_rate:.4f} > {history['rounds'][-2]['conflict_rate']:.4f} or grey: {grey_rate:.4f} > {history['rounds'][-2]['grey_rate']:.4f}), rolling back")
                rule_set = backup
                break

            # Check version limits
            max_version = max(
                [r.version for r in rule_set.clean_rules.values()] +
                [r.version for r in rule_set.dirty_rules.values()]
            )
            if max_version >= max_rounds:
                console_fn(f"  ⚠ Max version reached ({max_version}), stopping (conflict={conflict_rate:.4f}, grey={grey_rate:.4f})")
                break

        # Stop logging with summary (only if we created our own logger)
        final_summary = {
            'column': column,
            'total_rounds': len(history['rounds']),
            'final_conflict_rate': history['rounds'][-1]['conflict_rate'] if history['rounds'] else 1.0,
            'final_grey_rate': history['rounds'][-1]['grey_rate'] if history['rounds'] else 1.0,
        }
        if own_logger is not None:
            stop_logger(final_summary)

        history['final_memory'] = memory.to_context() if hasattr(memory, 'to_context') else str(memory)
        history['log_file'] = logger.path if hasattr(logger, 'path') else None

        return rule_set, history

    def _compile_rule_functions(self, rule_set: 'CleanRuleSet') -> 'CleanRuleSet':
        """Compile rule functions from strings."""
        for rule in rule_set.clean_rules.values():
            if rule.rule_func is None and rule.rule_str:
                try:
                    rule.rule_func = eval(rule.rule_str, safe_dict)
                except Exception as e:
                    print(f"  ⚠ Failed to compile {rule.name}: {e}")

        for rule in rule_set.dirty_rules.values():
            if rule.rule_func is None and rule.rule_str:
                try:
                    rule.rule_func = eval(rule.rule_str, safe_dict)
                except Exception as e:
                    print(f"  ⚠ Failed to compile {rule.name}: {e}")

        return rule_set
    
    def _count_conflicts(self, df: pd.DataFrame, column: str, dual_rule: 'DualRule') -> int:
        """Count conflicts (P_clean=True AND P_dirty=True)."""
        conflict_count = 0
        for idx, row in df.iterrows():
            value = row[column]
            try:
                is_clean = self._invoke_predicate(dual_rule.clean_rule_func, value, row)
                is_dirty = self._invoke_predicate(dual_rule.dirty_rule_func, value, row)
                if is_clean and is_dirty:
                    conflict_count += 1
            except Exception:
                pass
        return conflict_count

    def _count_grey_zone(self, df: pd.DataFrame, column: str, dual_rule: 'DualRule') -> int:
        """Count grey zone (NOT is_clean AND NOT is_dirty)."""
        grey_count = 0
        for idx, row in df.iterrows():
            value = row[column]
            try:
                is_clean = self._invoke_predicate(dual_rule.clean_rule_func, value, row)
                is_dirty = self._invoke_predicate(dual_rule.dirty_rule_func, value, row)
                if not is_clean and not is_dirty:
                    grey_count += 1
            except Exception:
                pass
        return grey_count


if __name__ == "__main__":
    pass
