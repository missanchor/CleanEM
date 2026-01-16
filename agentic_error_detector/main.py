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
from agentic_error_detector.validator import DisjointnessValidator


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
        "--refinement_mode",
        choices=["dual", "pillar"],
        default="pillar",
        help="Refinement strategy: 'dual' (original) or 'pillar' (pillar-level)."
    )
    return parser.parse_args()


def run_pillar_refinement(df, metadata, base_rules, clean_base_rules, factory, judge, args,
                          clean_prompts=None, dirty_prompts=None, clean_df=None):
    """
    Run pillar-level refinement for all columns.

    Args:
        df: DataFrame
        metadata: Column metadata
        base_rules: Dict[column] -> List[(agent, rule_str)]
        clean_base_rules: Dict[column] -> List[(pillar, rule_str)]
        factory: LegislatorFactory
        judge: Judge instance
        args: CLI arguments
        clean_prompts: Dict[column] -> Dict[pillar] -> {'prompt': ..., 'response': ...}
        dirty_prompts: Dict[column] -> Dict[agent] -> {'prompt': ..., 'response': ...}

    Returns:
        Tuple of (best_rules, refinement_history)
    """
    if clean_prompts is None:
        clean_prompts = {}
    if dirty_prompts is None:
        dirty_prompts = {}
    from agentic_error_detector.dual_types import PillarRule, PillarRuleSet
    from agentic_error_detector.modification_memory import start_logger, stop_logger
    import re
    import pandas as pd
    import numpy as np
    import os
    try:
        from dateutil.parser import parse  # type: ignore
    except Exception:
        parse = None

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
        "parse": parse,
    }

    best_rules = {}
    all_history = {}
    pillar_sets = {}
    initial_rules = {}

    # Extract dataset name from dirty_csv path
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]

    # Start a single logger for all columns
    print(f"\n{'='*80}")
    print(f"Starting unified refinement log for dataset: {dataset_name}")
    print(f"{'='*80}")
    logger = start_logger(args.output_dir, dataset_name)

    # Log initial rule generation
    logger.log_initial_rules(clean_base_rules, base_rules, clean_prompts, dirty_prompts)

    for column in metadata.keys():
        clean_pillars = {}
        for pillar_name, rule_str in clean_base_rules.get(column, []):
            try:
                rule_func = eval(rule_str, safe_dict)
                clean_pillars[pillar_name] = PillarRule(
                    name=pillar_name,
                    rule_str=rule_str,
                    rule_func=rule_func
                )
            except Exception as e:
                print(f"  ⚠ Failed to compile clean rule {pillar_name}: {e}")

        dirty_agents = {}
        for agent_name, rule_str in base_rules.get(column, []):
            try:
                rule_func = eval(rule_str, safe_dict)
                dirty_agents[agent_name] = PillarRule(
                    name=agent_name,
                    rule_str=rule_str,
                    rule_func=rule_func
                )
            except Exception as e:
                print(f"  ⚠ Failed to compile dirty rule {agent_name}: {e}")

        if not clean_pillars and not dirty_agents:
            continue

        pillar_set = PillarRuleSet(
            column=column,
            clean_pillars=clean_pillars,
            dirty_agents=dirty_agents
        )
        pillar_sets[column] = pillar_set

    for column, pillar_set in pillar_sets.items():
        dual_rule = pillar_set.to_dual_rule(legislator_factory=factory)
        initial_rules[column] = dual_rule

    if clean_df is not None and initial_rules:
        print("\n" + "="*80)
        print("INITIAL PILLAR PERFORMANCE (before refinement)")
        print("="*80)
        initial_detected_errors = judge.get_detected_dirty_values(initial_rules, df)
        initial_metrics_summary = judge.evaluate_with_ground_truth(
            df,
            clean_df,
            initial_detected_errors
        )
        judge.print_evaluation_summary(initial_metrics_summary)

    for column, pillar_set in pillar_sets.items():
        print(f"\n{'='*80}")
        print(f"Processing column: {column}")
        print(f"{'='*80}")

        col_metadata = metadata.get(column, {})
        refined_set, history = judge.refine_pillar_rules(
            df,
            column,
            pillar_set,
            factory=factory,
            max_rounds=args.max_rounds,
            conflict_tolerance=0.01,
            metadata=col_metadata,
            output_dir=args.output_dir,
            logger=logger
        )

        dual_rule = refined_set.to_dual_rule(legislator_factory=factory)
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
    print(f"[1/7] Profiling dirty dataset: {args.dirty_csv}")
    profiler = PandasProfiler(args.dirty_csv)
    metadata = profiler.get_metadata()

    factory = LegislatorFactory(base_url=args.base_url, model=args.model)

    print("[2/7] Generating base agent rules (Missing/Typo/Pattern/Outlier)")
    base_rules, dirty_prompts = factory.generate_rules_per_column(metadata)

    print("[3/7] Generating clean pillar rules (Completeness/Accuracy/Pattern/Relationship)")
    clean_base_rules, clean_prompts = factory.generate_clean_rules_per_column(metadata)

    # Only generate P_clean/P_dirty for dual mode
    if args.refinement_mode == "dual":
        # OPTIMIZATION: Skip initial P_clean if requested
        if args.skip_initial_clean:
            print("[4/7] Skipping initial P_clean generation (will be refined)")
            p_clean_rules = {}
            for column in metadata.keys():
                p_clean_rules[column] = "lambda value, row=None: value is not None"  # Placeholder
        else:
            print("[4/7] Generating independent P_clean predicates per column")
            p_clean_rules = factory.generate_p_clean_predicates_per_column(
                metadata,
                clean_base_rules=clean_base_rules,
                base_rules=base_rules
            )

        # Generate P_dirty predicates
        if args.skip_initial_dirty:
            print("[5/7] Skipping initial P_dirty generation (will be refined)")
            p_dirty_rules = {}
            for column in metadata.keys():
                p_dirty_rules[column] = "lambda value, row=None: False"  # Placeholder, all false
        else:
            print("[5/7] Generating independent P_dirty predicates per column")
            p_dirty_rules = factory.generate_p_dirty_predicates_per_column(
                metadata,
                base_rules=base_rules
            )

        print("[6/7] Assembling dual rule pairs (P_clean / P_dirty)")
        dual_rules = factory.pair_clean_dirty(p_clean_rules, p_dirty_rules)
    else:
        print("[4-6/7] Skipping P_clean/P_dirty generation (using pillar mode)")
        dual_rules = {}  # Not used in pillar mode

    judge = Judge(threshold=args.vr_threshold)

    print("[6.5/7] Evaluating base rules independently (standalone rule-based detection)")
    print("="*80)
    print("STANDALONE RULE-BASED ERROR DETECTION")
    print("="*80)
    base_evaluation_results = judge.evaluate_rules(profiler.df, base_rules)
    accepted_base_rules = judge.get_accepted_rules(base_evaluation_results)
    base_detected_errors = judge.get_detected_errors(accepted_base_rules)
    judge.print_summary(accepted_base_rules)
    # judge.print_detected_errors(base_detected_errors)

    # Evaluate standalone clean pillar rules (Completeness/Accuracy/Pattern/Relationship)
    print("\n" + "="*80)
    print("STANDALONE CLEAN PILLAR RULES EVALUATION")
    print("="*80)
    clean_evaluation_results = judge.evaluate_rules(profiler.df, clean_base_rules)
    accepted_clean_rules = judge.get_accepted_rules(clean_evaluation_results)
    judge.print_summary(accepted_clean_rules, pillar_type="clean")

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

    # Branch based on refinement mode
    if args.refinement_mode == "pillar":
        print("\n[7/7] Pillar-Level Refinement Mode")
        print("="*80)
        best_rules, refinement_history = run_pillar_refinement(
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
    else:
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


def run_vr_mode(args: argparse.Namespace) -> None:
    print(f"[1/4] Profiling dirty dataset: {args.dirty_csv}")
    profiler = PandasProfiler(args.dirty_csv)
    metadata = profiler.get_metadata()

    factory = LegislatorFactory(base_url=args.base_url, model=args.model)

    print("[2/4] Generating base agent rules (VR mode)")
    rules, _ = factory.generate_rules_per_column(metadata)

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
