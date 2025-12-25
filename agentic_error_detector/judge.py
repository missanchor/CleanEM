"""
Judge with VR (Violation Rate) based selection logic and Dual-Verification (P_clean/P_dirty).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from agentic_error_detector.dual_types import DualRule, DualEvaluationResult, RefinementRound


class Judge:
    """
    Judge class that evaluates rules based on Violation Rate (VR).

    Philosophy: "Errors are rare deviations from a strong rule."
    VR = count(Rule == False) / total_rows

    Criteria:
    - Accept: 0 < VR < threshold (e.g., 0.05)
    - VR = 0: Rule is trivial (everyone follows it)
    - VR > threshold: Rule is too strict or wrong
    - Small Non-Zero VR: Indicates Anomalies (Errors)
    """

    def __init__(self, threshold: float = 0.05):
        """Initialize Judge with VR threshold."""
        self.threshold = threshold
        self.evaluation_results = {}

    def evaluate_rules(self, df: pd.DataFrame, rules: Dict[str, list]) -> Dict[str, list]:
        """
        Evaluate all rules and return results with VR analysis.

        Args:
            df: DataFrame to evaluate
            rules: Dictionary of {column: [(agent_name, rule_string), ...]}

        Returns:
            Dictionary with evaluation results and VR analysis per column
        """
        results = {}

        for column, candidate_rules in rules.items():
            print(f"\n{'='*80}")
            print(f"Evaluating {len(candidate_rules)} candidate rules for column: {column}")
            print(f"{'='*80}")
            
            col_results = []
            
            for agent_name, rule_string in candidate_rules:
                print(f"\n  Agent: {agent_name}")
                print(f"  Rule: {rule_string[:100]}...")

                # Compile the lambda function
                try:
                    # Safely evaluate the lambda with required imports
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
                    rule_func = eval(rule_string, safe_dict)

                    # Apply rule to each row/value
                    violations = []
                    for idx, row in df.iterrows():
                        try:
                            value = row[column]
                            is_valid = self._invoke_predicate(rule_func, value, row)
                            if not is_valid:
                                violations.append({
                                    'row_index': idx,
                                    'value': value,
                                    'column': column
                                })
                        except Exception as e:
                            # If rule fails on a value, count as violation
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
        if vr == 0:
            return "reject_trivial"
        elif vr > self.threshold:
            return "reject_too_strict"
        elif 0 < vr <= self.threshold:
            return "accept_anomaly"
        else:
            return "unknown"

    def get_accepted_rules(self, results: Dict[str, list]) -> Dict[str, list]:
        """
        Get all accepted rules for each column based on VR.

        Strategy: Return all rules with "accept_anomaly" status (0 < VR <= threshold).
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

    def get_detected_errors(self, accepted_rules: Dict[str, list]) -> List[Dict[str, Any]]:
        """
        Extract all detected errors from accepted rules.

        Args:
            accepted_rules: Dictionary with {column: [rule_result1, rule_result2, ...]}

        Returns:
            List of detected errors (deduplicated by row_index and column)
        """
        all_errors = []

        for column, rule_results in accepted_rules.items():
            for result in rule_results:
                for violation in result['violations']:
                    error_info = {
                        'row_index': violation['row_index'],
                        'column': column,
                        'value': violation['value'],
                        'violated_rule': result['rule_string'],
                        'violated_agent': result['agent'],
                        'violation_rate': result['violation_rate']
                    }
                    all_errors.append(error_info)

        # Deduplicate errors by (row_index, column)
        # Keep the error with the lowest violation rate (most restrictive rule)
        seen_errors = {}
        for error in all_errors:
            key = (error['row_index'], error['column'])
            if key not in seen_errors:
                seen_errors[key] = error
            else:
                # Keep the one with lower violation rate (more precise)
                if error['violation_rate'] < seen_errors[key]['violation_rate']:
                    seen_errors[key] = error

        # Convert back to list and sort by row index
        unique_errors = list(seen_errors.values())
        unique_errors.sort(key=lambda x: x['row_index'])
        return unique_errors

    def print_summary(self, accepted_rules: Dict[str, list]):
        """Print a summary of the evaluation."""
        print("\n" + "="*80)
        print("JUDGE SUMMARY - Rule Evaluation Results")
        print("="*80)

        for column, results in accepted_rules.items():
            print(f"\nColumn: {column}")
            print(f"  Number of accepted rules: {len(results)}")
            for i, result in enumerate(results, 1):
                print(f"\n  Rule {i}:")
                print(f"    Agent: {result['agent']}")
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
        print("\n" + "="*80)
        print("EVALUATION AGAINST GROUND TRUTH")
        print("="*80)

        # Create ground truth error set
        ground_truth_errors = set()
        for idx in range(len(dirty_df)):
            for col in dirty_df.columns:
                dirty_val = dirty_df.iloc[idx][col]
                clean_val = clean_df.iloc[idx][col]
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

        # Print results
        print(f"\nOverall Metrics:")
        print(f"  True Positives (TP): {tp}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        print(f"\nTotal Ground Truth Errors: {len(ground_truth_errors)}")
        print(f"Total Detected Errors: {len(detected_error_set)}")

        # Print per-column summary
        print("\n" + "="*80)
        print("PER-COLUMN EVALUATION RESULTS")
        print("="*80)

        for col, metrics in per_column_metrics.items():
            print(f"\n{col}:")
            print(f"  Ground Truth Errors: {metrics['total_ground_truth_errors']}")
            print(f"  Detected Errors: {metrics['total_detected_errors']}")
            print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")

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

    def evaluate_dual_rules(self, df: pd.DataFrame, dual_rules: Dict[str, List[Tuple[str, str, str]]],
                           grey_tolerance: float = 0.0) -> Dict[str, List[DualEvaluationResult]]:
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

        def call_predicate(pred, value, row=None):
            return self._invoke_predicate(pred, value, row)

        def safe_not(pred, value, row=None) -> bool:
            """Return logical NOT of pred(value,row); on any error treat as dirty (True)."""
            try:
                return not bool(call_predicate(pred, value, row))
            except Exception:
                return True

        safe_dict = {
            "re": re,
            "str": str,
            "bool": bool,
            "pd": pd,
            "np": np,
            "float": float,
            "int": int,
            "len": len,
            "safe_not": safe_not,
            "safe_float": safe_float,
        }

        for column, candidate_rule_tuples in dual_rules.items():
            print(f"\n{'='*80}")
            print(f"Evaluating dual rules for column: {column}")
            print(f"{'='*80}")

            col_results = []

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
                        conflict_rate, grey_rate, dirty_rate, grey_tolerance
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
                        violation_message=violation_message
                    )

                    col_results.append(result)

                    # Print evaluation summary
                    print(f"\n    Classification:")
                    print(f"      Conflict: {conflict_count} ({conflict_rate:.4f})")
                    print(f"      Grey Zone: {grey_count} ({grey_rate:.4f})")
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

    def _evaluate_dual_constraints(self, conflict_rate: float, grey_rate: float,
                                   dirty_rate: float, grey_tolerance: float) -> Tuple[str, str]:
        """
        Evaluate if dual rule pair meets hard constraints.

        Constraints:
        1. conflict_rate == 0 (no conflicts)
        2. grey_rate <= grey_tolerance
        3. dirty_rate < 1.0 (NOT all dirty)
        4. dirty_rate >= 0 (always true)

        Returns:
            Tuple of (status, message)
        """
        if dirty_rate == 1.0:
            return 'reject_all_dirty', f"Entire column marked dirty (dirty_rate={dirty_rate:.4f})"

        if conflict_rate > 0:
            return 'reject_conflict', f"Predicates not mutually exclusive (conflict_rate={conflict_rate:.4f})"

        if grey_rate > grey_tolerance:
            return 'reject_grey', f"Too many uncertain values (grey_rate={grey_rate:.4f} > {grey_tolerance:.4f})"

        if dirty_rate == 0:
            return 'accept_all_clean', f"Column is clean (dirty_rate={dirty_rate:.4f})"

        return 'accept', f"Valid dual rule (dirty_rate={dirty_rate:.4f})"

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

    def select_best_dual_rules(self, evaluation_results: Dict[str, List[DualEvaluationResult]]) -> Dict[str, DualRule]:
        """
        Select the best dual rule for each column based on constraints and dirty rate.

        Strategy:
        1. Filter rules that meet hard constraints
        2. Among valid rules, select one with minimum dirty_rate
        3. If dirty_rate == 0, accept (allows all-clean columns)
        4. If no valid rules, return None for that column
        """
        print("\n" + "="*80)
        print("SELECTING BEST DUAL RULES")
        print("="*80)

        best_rules = {}

        for column, results in evaluation_results.items():
            print(f"\n{column}:")

            # Filter acceptable rules
            acceptable = [r for r in results if r.status in ['accept', 'accept_all_clean']]

            if not acceptable:
                print(f"  ✗ No acceptable rules found")
                # Show why each was rejected
                for r in results:
                    print(f"    - {r.status}: {r.violation_message}")
                continue

            # Sort by dirty_rate (ascending - prefer fewer dirty values)
            acceptable.sort(key=lambda x: x.dirty_rate)

            best_result = acceptable[0]
            best_rules[column] = best_result.rule

            print(f"  ✓ Selected rule:")
            print(f"    Agent: {best_result.rule.agent_name}")
            print(f"    Dirty Rate: {best_result.dirty_rate:.4f}")
            print(f"    Clean Rate: {best_result.clean_rate:.4f}")
            print(f"    Grey Rate: {best_result.grey_rate:.4f}")
            print(f"    Conflict Rate: {best_result.conflict_rate:.4f}")

        return best_rules

    def refine_dual_rules(self, df: pd.DataFrame,
                         metadata: Dict[str, Any],
                         dual_rules: Dict[str, List[Tuple[str, str, str]]],
                         max_rounds: int = 3,
                         grey_tolerance: float = 0.0) -> Tuple[Dict[str, DualRule], Dict[str, List[RefinementRound]]]:
        """
        Iteratively refine dual rules to eliminate grey zones and conflicts.

        Args:
            df: DataFrame to evaluate
            metadata: Column metadata (for regenerating rules)
            dual_rules: Initial dual rules
            max_rounds: Maximum refinement rounds
            grey_tolerance: Maximum acceptable grey zone rate

        Returns:
            Tuple of (best_rules, refinement_history)
        """
        print("\n" + "="*80)
        print("REFINING DUAL RULES - ITERATIVE IMPROVEMENT")
        print("="*80)

        current_rules = {col: list(rules) for col, rules in dual_rules.items()}
        refinement_history = {column: [] for column in current_rules.keys()}
        round_number = 0

        while round_number < max_rounds:
            round_number += 1
            print(f"\n{'='*80}")
            print(f"ROUND {round_number}/{max_rounds}")
            print(f"{'='*80}")

            # Evaluate current rules
            evaluation_results = self.evaluate_dual_rules(df, current_rules, grey_tolerance)

            # Check if all columns have acceptable rules
            all_acceptable = True
            needs_refinement = {}

            for column, results in evaluation_results.items():
                acceptable = [r for r in results if r.status in ['accept', 'accept_all_clean']]

                if not acceptable:
                    all_acceptable = False

                    # Collect problem samples
                    best_result = self._select_refinement_candidate(results)

                    needs_refinement[column] = {
                        'grey': best_result.grey_samples[:5],
                        'conflict': best_result.conflict_samples[:5],
                        'all_dirty': [s for s in best_result.determined_dirty_samples if s.get('count', 0) > len(df) * 0.5]
                    }

                    print(f"\n  {column}: Needs refinement")
                    print(f"    Grey samples: {len(needs_refinement[column]['grey'])}")
                    print(f"    Conflict samples: {len(needs_refinement[column]['conflict'])}")
                    print(f"    All-dirty samples: {len(needs_refinement[column]['all_dirty'])}")

            if all_acceptable:
                print("\n✓ All columns have acceptable rules!")
                break

            # Generate refined rules
            print(f"\n{'='*80}")
            print(f"REFINING RULES FOR PROBLEMATIC COLUMNS")
            print(f"{'='*80}")

            from agentic_error_detector.legislator import LegislatorFactory
            factory = LegislatorFactory()

            # Prepare refinement history for factory
            for column, samples in needs_refinement.items():
                refinement_history[column].append({
                    'round': round_number,
                    'samples_used': samples
                })

            # Generate new rules with refinement samples
            subset_metadata = {col: metadata.get(col, {}) for col in needs_refinement.keys()}
            refined_rules = factory.generate_dual_rules_per_column(
                subset_metadata,
                refinement_history
            )

            # Update current rules
            for column in needs_refinement.keys():
                if column in refined_rules:
                    current_rules[column] = refined_rules[column]
                    print(f"✓ Refined rules generated for {column}")

        # Final evaluation
        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION AFTER {round_number} ROUNDS")
        print(f"{'='*80}")

        final_evaluation = self.evaluate_dual_rules(df, current_rules, grey_tolerance)
        best_rules = self.select_best_dual_rules(final_evaluation)

        return best_rules, refinement_history

    def get_detected_dirty_values(self, best_rules: Dict[str, DualRule], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get all values flagged as DIRTY by the best dual rules.

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
                    is_dirty = self._invoke_predicate(rule.dirty_rule_func, value, row)
                    if is_dirty:
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
                print(f"  Grey Rate: {result.grey_rate:.4f} ({result.grey_count}/{result.total_rows})")
                print(f"  Conflict Rate: {result.conflict_rate:.4f} ({result.conflict_count}/{result.total_rows})")
                print(f"\n  P_clean: {rule.clean_rule_str}")
                print(f"  P_dirty: {rule.dirty_rule_str}")

    def save_dual_results(self, best_rules: Dict[str, DualRule],
                         evaluation_results: Dict[str, List[DualEvaluationResult]],
                         refinement_history: Dict[str, List],
                         detected_errors: List[Dict[str, Any]],
                         coverage_gaps: List[str] = None,
                         output_dir: str = "agentic_error_detector/results"):
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