"""
Command-line entry point for the agentic error detector.

Supports:
- Dual verification pipeline (P_clean / P_dirty) with iterative refinement.
- Legacy VR (violation-rate) pipeline for quick comparisons.
"""
import argparse
import json
import os
from typing import Dict, List, Any

import pandas as pd

from agentic_error_detector.judge import Judge
from agentic_error_detector.legislator import LegislatorFactory
from agentic_error_detector.profiler import PandasProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Error Detector CLI (dual verification by default)."
    )
    parser.add_argument(
        "--dirty_csv",
        default="data/beers_error-01.csv",
        help="Path to the dirty/error-prone CSV that needs inspection."
    )
    parser.add_argument(
        "--clean_csv",
        default="data/beers_clean.csv",
        help="Optional clean CSV for evaluation against ground truth."
    )
    parser.add_argument(
        "--mode",
        choices=["dual", "vr"],
        default="dual",
        help="Which detection strategy to run (default: dual)."
    )
    parser.add_argument(
        "--output_dir",
        default="agentic_error_detector/results",
        help="Directory for serialized outputs."
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
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
        default=0.05,
        help="Violation-rate threshold for legacy VR mode."
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
    return parser.parse_args()


def run_dual_mode(args: argparse.Namespace) -> None:
    print(f"[1/7] Profiling dirty dataset: {args.dirty_csv}")
    profiler = PandasProfiler(args.dirty_csv)
    metadata = profiler.get_metadata()

    factory = LegislatorFactory(base_url=args.base_url, model=args.model)

    print("[2/7] Generating base agent rules (Missing/Typo/Pattern/Outlier)")
    base_rules = factory.generate_rules_per_column(metadata)

    print("[3/7] Generating clean pillar rules (Completeness/Accuracy/Pattern/Relationship)")
    clean_base_rules = factory.generate_clean_rules_per_column(metadata)

    print("[4/7] Generating independent P_clean predicates per column")
    p_clean_rules = factory.generate_p_clean_predicates_per_column(
        metadata,
        clean_base_rules=clean_base_rules,
        base_rules=base_rules
    )

    print("[5/7] Generating independent P_dirty predicates per column")
    p_dirty_rules = factory.generate_p_dirty_predicates_per_column(
        metadata,
        base_rules=base_rules
    )

    print("[6/7] Assembling dual rule pairs (P_clean / P_dirty)")
    dual_rules = factory.pair_clean_dirty(p_clean_rules, p_dirty_rules)

    judge = Judge(threshold=args.vr_threshold)

    print("[6.5/7] Evaluating base rules independently (standalone rule-based detection)")
    print("="*80)
    print("STANDALONE RULE-BASED ERROR DETECTION")
    print("="*80)
    base_evaluation_results = judge.evaluate_rules(profiler.df, base_rules)
    accepted_base_rules = judge.get_accepted_rules(base_evaluation_results)
    base_detected_errors = judge.get_detected_errors(accepted_base_rules)
    judge.print_summary(accepted_base_rules)
    judge.print_detected_errors(base_detected_errors)

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

    print("\n[7/7] Refining dual rules to resolve conflicts and gaps")
    best_rules, refinement_history = judge.refine_dual_rules(
        profiler.df,
        metadata,
        dual_rules,
        max_rounds=args.max_rounds,
        grey_tolerance=args.grey_tolerance,
        factory=factory
    )

    if not best_rules:
        print("✗ No acceptable dual rules were produced. See refinement logs for details.")
        return

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

    print("\n✓ Dual verification complete.")


def run_vr_mode(args: argparse.Namespace) -> None:
    print(f"[1/4] Profiling dirty dataset: {args.dirty_csv}")
    profiler = PandasProfiler(args.dirty_csv)
    metadata = profiler.get_metadata()

    factory = LegislatorFactory(base_url=args.base_url, model=args.model)

    print("[2/4] Generating base agent rules (VR mode)")
    rules = factory.generate_rules_per_column(metadata)

    judge = Judge(threshold=args.vr_threshold)
    print("[3/4] Evaluating rules via violation rate")
    evaluation_results = judge.evaluate_rules(profiler.df, rules)
    accepted_rules = judge.get_accepted_rules(evaluation_results)
    detected_errors = judge.get_detected_errors(accepted_rules)
    judge.print_summary(accepted_rules)
    judge.print_detected_errors(detected_errors)

    metrics_summary = None
    if args.clean_csv:
        print("[4/4] Comparing accepted detections against ground truth")
        clean_df = pd.read_csv(args.clean_csv)
        metrics_summary = judge.evaluate_with_ground_truth(
            profiler.df,
            clean_df,
            detected_errors
        )

    vr_dir = os.path.join(args.output_dir, "vr_mode")
    os.makedirs(vr_dir, exist_ok=True)
    with open(os.path.join(vr_dir, "accepted_rules.json"), "w") as f:
        json.dump(_serialize_vr_rules(accepted_rules), f, indent=2)
    with open(os.path.join(vr_dir, "detected_errors.json"), "w") as f:
        json.dump(detected_errors, f, indent=2, default=str)
    if metrics_summary:
        with open(os.path.join(vr_dir, "ground_truth_metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"✓ Ground-truth metrics saved to {os.path.join(vr_dir, 'ground_truth_metrics.json')}")

    print("✓ VR mode complete.")


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
    if args.mode == "dual":
        run_dual_mode(args)
    else:
        run_vr_mode(args)


if __name__ == "__main__":
    main()

