"""
Command-line entry point for the agentic error detector.

Supports a dual verification pipeline with clean rule-level refinement.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from datetime import datetime

import pandas as pd

from judge import Judge
from agent import AgentFactory
from profiler import PandasProfiler
from validator import DisjointnessValidator
from core.utils import safe_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Error Detector CLI (dual verification by default)."
    )
    parser.add_argument(
        "--dirty_csv",
        default="data/flights_error-01.csv",
        help="Path to the dirty/error-prone CSV that needs inspection."
    )
    parser.add_argument(
        "--clean_csv",
        default="data/flights_clean.csv",
        help="Optional clean CSV for evaluation against ground truth."
    )
    parser.add_argument(
        "--output_dir",
        default="agentic_error_detector/results",
        help="Directory for serialized outputs."
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=10,
        help="Max refinement rounds for dual verification."
    )
    parser.add_argument(
        "--grey_tolerance",
        type=float,
        default=0.1,
        help="Allowed grey-zone rate when evaluating dual rules."
    )
    parser.add_argument(
        "--vr_threshold",
        type=float,
        default=0.5,
        help="Violation-rate threshold for legacy VR mode."
    )
    parser.add_argument(
        "--skip_initial_clean",
        action="store_true",
        help="Skip initial P_clean generation, let refinement generate both."
    )
    parser.add_argument(
        "--skip_initial_dirty",
        action="store_true",
        help="Skip initial P_dirty generation, let refinement generate both."
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint for rule-generating LLMs."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the Legislator factory."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum parallel workers for per-column LLM calls."
    )
    return parser.parse_args()


def run_clean_rule_refinement(df, metadata, base_rules, clean_rules, factory, judge, args,
                              clean_prompts=None, dirty_prompts=None, clean_df=None):
    """
    Run clean rule-level refinement for all columns.

    Args:
        df: DataFrame
        metadata: Column metadata
        base_rules: Dict[column] -> List[(agent, rule_str)]
        clean_rules: Dict[column] -> List[(clean_rule, rule_str)]
        factory: AgentFactory
        judge: Judge instance
        args: CLI arguments
        clean_prompts: Dict[column] -> Dict[clean_rule] -> {'prompt': ..., 'response': ...}
        dirty_prompts: Dict[column] -> Dict[agent] -> {'prompt': ..., 'response': ...}

    Returns:
        Tuple of (best_rules, refinement_history)
    """
    if clean_prompts is None:
        clean_prompts = {}
    if dirty_prompts is None:
        dirty_prompts = {}
    from dual_types import CleanRule, CleanRuleSet
    from modification_memory import start_logger, stop_logger
    import re
    import pandas as pd
    import numpy as np
    import os
    try:
        from dateutil.parser import parse  # type: ignore
    except Exception:
        parse = None

    best_rules: Dict[str, Any] = {}
    all_history: Dict[str, Any] = {}
    clean_rule_sets: Dict[str, Any] = {}
    initial_rules: Dict[str, Any] = {}

    # Extract dataset name from dirty_csv path
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]

    # Start a single logger for all columns
    print(f"\n{'='*80}")
    print(f"Starting unified refinement log for dataset: {dataset_name}")
    print(f"{'='*80}")
    logger = start_logger(args.output_dir, dataset_name)

    # Log initial rule generation
    logger.log_initial_rules(clean_rules, base_rules, clean_prompts, dirty_prompts)

    for column in metadata.keys():
        clean_rules_dict = {}
        for rule_name, rule_str in clean_rules.get(column, []):
            try:
                rule_func = eval(rule_str, safe_dict)
                clean_rules_dict[rule_name] = CleanRule(
                    name=rule_name,
                    rule_str=rule_str,
                    rule_func=rule_func
                )
            except Exception as e:
                print(f"  ⚠ Failed to compile clean rule {rule_name}: {e}")

        dirty_rules = {}
        for agent_name, rule_str in base_rules.get(column, []):
            try:
                rule_func = eval(rule_str, safe_dict)
                dirty_rules[agent_name] = CleanRule(
                    name=agent_name,
                    rule_str=rule_str,
                    rule_func=rule_func
                )
            except Exception as e:
                print(f"  ⚠ Failed to compile dirty rule {agent_name}: {e}")

        if not clean_rules_dict and not dirty_rules:
            continue

        rule_set = CleanRuleSet(
            column=column,
            clean_rules=clean_rules_dict,
            dirty_rules=dirty_rules
        )
        clean_rule_sets[column] = rule_set

    for column, rule_set in clean_rule_sets.items():
        dual_rule = rule_set.to_dual_rule(agent_factory=factory)
        initial_rules[column] = dual_rule

    if clean_df is not None and initial_rules:
        print("\n" + "="*80)
        print("INITIAL CLEAN RULE PERFORMANCE (before refinement)")
        print("="*80)
        initial_detected_errors = judge.get_detected_dirty_values(initial_rules, df)
        initial_metrics_summary = judge.evaluate_with_ground_truth(
            df,
            clean_df,
            initial_detected_errors
        )
        judge.print_evaluation_summary(initial_metrics_summary)

    def _make_console_buffer():
        lines: List[str] = []

        def console(msg: str = "") -> None:
            lines.append(msg)

        return console, lines

    def _refine_single_column(column: str, rule_set: Any):
        console, log_lines = _make_console_buffer()
        col_metadata = metadata.get(column, {})
        refined_set, history = judge.refine_clean_rules(
            df,
            column,
            rule_set,
            factory=factory,
            max_rounds=args.max_rounds,
            conflict_tolerance=0.01,
            metadata=col_metadata,
            output_dir=args.output_dir,
            logger=logger,
            console=console,
        )
        dual_rule = refined_set.to_dual_rule(agent_factory=factory)
        return column, dual_rule, history, log_lines

    max_workers = max(1, int(getattr(args, "max_workers", 1) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_column = {
            executor.submit(_refine_single_column, column, rule_set): column
            for column, rule_set in clean_rule_sets.items()
        }
        for future in as_completed(future_to_column):
            column, dual_rule, history, log_lines = future.result()
            print("\n" + "=" * 80)
            print(f"Processing column: {column}")
            print("=" * 80)
            for line in log_lines:
                print(line)
            best_rules[column] = dual_rule
            all_history[column] = history

    # Stop the shared logger
    final_summary = {
        'dataset': dataset_name,
        'total_columns': len(metadata.keys()),
        'processed_columns': list(best_rules.keys()),
    }
    stop_logger(final_summary)
    print(f"\n{'='*80}")
    print(f"Refinement log saved for dataset: {dataset_name}")
    print(f"{'='*80}")

    return best_rules, all_history


def run_dual_mode(args: argparse.Namespace) -> None:
    # Create console log file with the same timestamp format
    log_dir = os.path.join(args.output_dir, "refinement_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]
    console_log_path = os.path.join(log_dir, f"{timestamp}_{dataset_name}_running_log.log")

    # Custom stdout that writes to both terminal and file
    class DualOutput:
        def __init__(self, filepath, dataset_name):
            self.filepath = filepath
            self.file = open(filepath, 'w', encoding='utf-8')
            self.write_header(dataset_name)

        def write_header(self, dataset_name):
            self.file.write("=" * 60 + "\n")
            self.file.write("CONSOLE OUTPUT LOG\n")
            self.file.write("=" * 60 + "\n")
            self.file.write(f"Dataset: {dataset_name}\n")
            self.file.write(f"Started: {datetime.now().isoformat()}\n")
            self.file.write("=" * 60 + "\n\n")
            self.file.flush()

        def write(self, text):
            import sys
            sys.__stdout__.write(text)
            ts = datetime.now().strftime("%H:%M:%S")
            self.file.write(f"[{ts}] {text}")
            self.file.flush()

        def flush(self):
            import sys
            sys.__stdout__.flush()
            self.file.flush()

        def close(self):
            self.file.write("\n" + "=" * 60 + "\n")
            self.file.write(f"Finished: {datetime.now().isoformat()}\n")
            self.file.write("=" * 60 + "\n")
            self.file.close()

    dual_output = DualOutput(console_log_path, dataset_name)

    # Redirect stdout to both terminal and file
    import sys
    old_stdout = sys.stdout
    sys.stdout = dual_output

    try:
        print(f"[1/7] Profiling dirty dataset: {args.dirty_csv}")
        profiler = PandasProfiler(args.dirty_csv)
        metadata = profiler.get_metadata()

        factory = AgentFactory(base_url=args.base_url, model=args.model, max_workers=args.max_workers)

        print("[2/7] Generating base agent rules (Missing/Typo/Pattern/Outlier)")
        base_rules, dirty_prompts = factory.generate_rules_per_column(metadata)

        print("[3/7] Generating clean rules (Completeness/Accuracy/Pattern/Relationship)")
        clean_rules, clean_prompts = factory.generate_clean_rules_per_column(metadata)

        judge = Judge(threshold=args.vr_threshold)

        print("[6.5/7] Evaluating base rules independently (standalone rule-based detection)")
        print("="*80)
        print("STANDALONE RULE-BASED ERROR DETECTION")
        print("="*80)
        base_evaluation_results = judge.evaluate_rules(profiler.df, base_rules)
        accepted_base_rules = judge.get_accepted_rules(base_evaluation_results)
        judge.print_summary(accepted_base_rules)
        # judge.print_detected_errors(base_detected_errors)

        # Evaluate standalone clean rules (Completeness/Accuracy/Pattern/Relationship)
        print("\n" + "="*80)
        print("STANDALONE CLEAN RULES EVALUATION")
        print("="*80)
        clean_evaluation_results = judge.evaluate_rules(profiler.df, clean_rules, rule_type="clean")
        accepted_clean_rules = judge.get_accepted_rules(clean_evaluation_results)
        judge.print_summary(accepted_clean_rules, rule_type="clean")

        # Combined error detection using AND/OR logic
        # Clean Rule (AND): All clean rules must be satisfied
        # Dirty Rule (OR): Violating any dirty rule marks as potentially dirty
        # Error = (NOT all clean rules satisfied) AND (at least one dirty rule violated)
        print("\n" + "="*80)
        print("COMBINED ERROR DETECTION (AND/OR LOGIC)")
        print("="*80)
        base_detected_errors = judge.get_detected_errors(
            accepted_base_rules,      # dirty rules (OR logic)
            accepted_clean_rules      # clean rules (AND logic)
        )
        print(f"Detected {len(base_detected_errors)} errors using combined AND/OR logic")

        # Evaluate standalone base rules against ground truth if clean CSV is provided
        base_metrics_summary = None
        clean_df = None
        if args.clean_csv:
            print("\n" + "="*80)
            print("STANDALONE BASE RULES - GROUND TRUTH EVALUATION")
            print("="*80)
            clean_df = pd.read_csv(args.clean_csv)
            base_metrics_summary = judge.evaluate_with_ground_truth(
                profiler.df,
                clean_df,
                base_detected_errors
            )
            judge.print_evaluation_summary(base_metrics_summary)

        print("\n[7/7] Clean Rule-Level Refinement Mode")
        print("="*80)
        best_rules, refinement_history = run_clean_rule_refinement(
            profiler.df,
            metadata,
            {
                column: [(r['agent'], r['rule_string']) for r in accepted_base_rules.get(column, [])]
                for column in accepted_base_rules.keys()
            },
            {
                column: [(r['agent'], r['rule_string']) for r in accepted_clean_rules.get(column, [])]
                for column in accepted_clean_rules.keys()
            },
            factory,
            judge,
            args,
            clean_prompts,
            dirty_prompts,
            clean_df
        )

        if not best_rules:
            print("✗ No acceptable dual rules were produced. See refinement logs for details.")
            return

        print("\n[7.5/7] Validating disjointness of refined rules")
        print("="*80)
        print("DISJOINTNESS VALIDATION")
        print("="*80)
        validator = DisjointnessValidator(gap_tolerance=args.grey_tolerance)
        validation_result = validator.validate_batch(profiler.df, best_rules)
        print(validator.report_violations(validation_result))

        coverage_gaps = sorted(set(metadata.keys()) - set(best_rules.keys()))

        print("[8/8] Evaluating final dual rules on the dataset")
        evaluation_payload = _materialize_rule_payload(best_rules)
        evaluation_results = judge.evaluate_dual_rules(
            profiler.df,
            evaluation_payload,
            grey_tolerance=args.grey_tolerance
        )
        detected_dirty_values = judge.get_detected_dirty_values(best_rules, profiler.df)
        judge.print_dual_summary(best_rules, evaluation_results)

        metrics_summary = None
        if args.clean_csv:
            print(f"[9/9] Comparing dual rules against ground truth: {args.clean_csv}")
            if clean_df is None:
                clean_df = pd.read_csv(args.clean_csv)
            metrics_summary = judge.evaluate_with_ground_truth(
                profiler.df,
                clean_df,
                detected_dirty_values
            )
            judge.print_evaluation_summary(metrics_summary)

        os.makedirs(args.output_dir, exist_ok=True)
        judge.save_dual_results(
            best_rules,
            evaluation_results,
            refinement_history,
            detected_dirty_values,
            coverage_gaps,
            output_dir=args.output_dir
        )

        # Save base rules evaluation results
        print("\n" + "="*80)
        print("SAVING STANDALONE BASE RULES RESULTS")
        print("="*80)

        # Save base rules evaluation
        with open(os.path.join(args.output_dir, "base_rules_evaluation.json"), "w") as f:
            json.dump(_serialize_vr_rules(accepted_base_rules), f, indent=2)
        print(f"  - base_rules_evaluation.json")

        # Save base rules detected errors
        with open(os.path.join(args.output_dir, "base_rules_detected_errors.json"), "w") as f:
            json.dump(base_detected_errors, f, indent=2, default=str)
        print(f"  - base_rules_detected_errors.json")

        # Save base rules ground truth metrics if available
        if base_metrics_summary:
            base_metrics_path = os.path.join(args.output_dir, "base_rules_ground_truth_metrics.json")
            with open(base_metrics_path, "w") as f:
                json.dump(base_metrics_summary, f, indent=2)
            print(f"  - base_rules_ground_truth_metrics.json")

        if metrics_summary:
            metrics_path = os.path.join(args.output_dir, "dual_ground_truth_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_summary, f, indent=2)
            print(f"✓ Dual ground-truth metrics saved to {metrics_path}")

        # Save disjointness validation results
        validation_path = os.path.join(args.output_dir, "disjointness_validation.json")
        validator.save_results(validation_result, validation_path)

        print("\n✓ Dual verification complete.")

    finally:
        # Restore stdout and close the log file
        sys.stdout = old_stdout
        dual_output.close()


def _materialize_rule_payload(best_rules) -> Dict[str, List[tuple]]:
    """Convert DualRule map into judge.evaluate_dual_rules input."""
    payload: Dict[str, List[tuple]] = {}
    for column, rule in best_rules.items():
        payload[column] = [(rule.agent_name, rule.clean_rule_str, rule.dirty_rule_str)]
    return payload


def _serialize_vr_rules(accepted_rules: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Strip non-serializable fields (e.g., lambda objects) from VR results."""
    serializable: Dict[str, List[dict]] = {}
    for column, rules in accepted_rules.items():
        trimmed: List[dict] = []
        for result in rules:
            trimmed.append({
                'agent': result.get('agent'),
                'status': result.get('status'),
                'violation_rate': result.get('violation_rate'),
                'violation_count': result.get('violation_count'),
                'total_rows': result.get('total_rows'),
                'rule_string': result.get('rule_string'),
            })
        serializable[column] = trimmed
    return serializable


def main() -> None:
    args = parse_args()
    run_dual_mode(args)


if __name__ == "__main__":
    main()
