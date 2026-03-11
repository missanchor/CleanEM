"""
Command-line entry point for the agentic error detector.

Supports a dual verification pipeline with clean rule-level refinement.
"""
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from judge import Judge
from agent import AgentFactory, DirtyExampleAgent, CleanRuleReflectionAgent
from profiler import PandasProfiler
from validator import DisjointnessValidator
from core.utils import safe_dict


# CleanEM Logger
cleanem_logger = logging.getLogger("CleanEM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Error Detector CLI with dual and clean+EM modes."
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
        default="results/agentic_error_detector",
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
        default=0.01,
        help="Allowed grey-zone rate when evaluating dual rules."
    )
    parser.add_argument(
        "--vr_threshold",
        type=float,
        default=0.6,
        help="Violation-rate threshold for legacy VR mode."
    )
    parser.add_argument(
        "--mode",
        choices=["dual", "clean_em"],
        default="clean_em",
        help="Pipeline mode: 'dual' uses P_clean/P_dirty, 'clean_em' uses clean rules + EM."
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
    parser.add_argument(
        "--clean_seed_percent",
        type=float,
        default=0.1,
        help="Top percentile per column used as high-confidence clean seeds (0-1)."
    )
    parser.add_argument(
        "--synthetic_per_family",
        type=int,
        default=50,
        help="Synthetic dirty examples per family per column."
    )
    parser.add_argument(
        "--em_max_iters",
        type=int,
        default=10,
        help="Maximum EM iterations per column/family."
    )
    parser.add_argument(
        "--em_prior_dirty",
        type=float,
        default=0.05,
        help="Initial prior dirty rate per family."
    )
    parser.add_argument(
        "--calib_min_clean_pass",
        type=float,
        default=0.8,
        help="Min required pass-rate on clean seeds to keep a rule (0-1)."
    )
    parser.add_argument(
        "--calib_max_dirty_pass",
        type=float,
        default=0.3,
        help="Max allowed pass-rate on synthetic dirty to keep a rule (0-1)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-K highest scoring cells to display."
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Threshold on S_total to treat a cell as error in clean_em mode."
    )
    return parser.parse_args()


def _invoke_rule(rule_func, value, row) -> bool:
    try:
        return bool(rule_func(value, row))
    except TypeError:
        try:
            return bool(rule_func(value))
        except Exception:
            return False
    except Exception:
        return False


def _build_clean_rule_pool(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    factory: AgentFactory,
) -> Dict[str, List[Dict[str, Any]]]:
    clean_rules, _ = factory.generate_clean_rules_per_column(metadata)
    family_map = {
        "completeness": "missing",
        "accuracy": "outlier",
        "pattern_consistency": "pattern",
    }
    pool: Dict[str, List[Dict[str, Any]]] = {}
    for column, rules in clean_rules.items():
        col_pool: List[Dict[str, Any]] = []
        for idx, (agent_name, rule_str) in enumerate(rules):
            family = family_map.get(agent_name)
            if not family:
                continue
            try:
                rule_func = eval(rule_str, safe_dict)
            except Exception:
                continue
            col_pool.append(
                {
                    "agent": agent_name,
                    "family": family,
                    "rule_name": f"{agent_name}_{idx}",
                    "rule_str": rule_str,
                    "rule_func": rule_func,
                }
            )
        if col_pool:
            pool[column] = col_pool
    return pool


def _select_clean_seeds_for_column(
    df: pd.DataFrame,
    column: str,
    rules: List[Dict[str, Any]],
    seed_percent: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_rows = len(df)
    if n_rows == 0 or not rules:
        return np.array([], dtype=int), np.zeros((0, len(rules)), dtype=int)
    outputs = np.zeros((n_rows, len(rules)), dtype=int)
    for j, rule in enumerate(rules):
        func = rule["rule_func"]
        for i, (_, row) in enumerate(df.iterrows()):
            value = row[column]
            outputs[i, j] = int(_invoke_rule(func, value, row))
    scores = outputs.mean(axis=1)
    k = max(1, int(max(seed_percent, 0.0) * n_rows))
    k = min(k, n_rows)
    top_indices = np.argsort(-scores)[:k]
    return top_indices.astype(int), outputs


def _run_em_for_family(
    clean_z: np.ndarray,
    dirty_z: np.ndarray,
    unlabeled_z: np.ndarray,
    max_iters: int,
    prior_dirty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rules = unlabeled_z.shape[1] if unlabeled_z.size > 0 else clean_z.shape[1]
    eps = 1e-3
    if n_rules == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    alpha = np.full(n_rules, 0.9)
    beta = np.full(n_rules, 0.3)
    if clean_z.size > 0:
        alpha = np.clip(clean_z.mean(axis=0), eps, 1 - eps)
    if dirty_z.size > 0:
        beta = np.clip(dirty_z.mean(axis=0), eps, 1 - eps)
    pi1 = min(max(prior_dirty, eps), 1 - eps)
    pi0 = 1.0 - pi1
    n_unlabeled = unlabeled_z.shape[0]
    gamma = np.full(n_unlabeled, pi1)
    for _ in range(max_iters):
        if n_unlabeled > 0:
            log_p0 = np.log(pi0)
            log_p1 = np.log(pi1)
            for r in range(n_rules):
                zr = unlabeled_z[:, r]
                p_z1_y0 = alpha[r]
                p_z0_y0 = 1.0 - alpha[r]
                p_z1_y1 = beta[r]
                p_z0_y1 = 1.0 - beta[r]
                log_p0 += zr * np.log(p_z1_y0 + eps) + (1 - zr) * np.log(p_z0_y0 + eps)
                log_p1 += zr * np.log(p_z1_y1 + eps) + (1 - zr) * np.log(p_z0_y1 + eps)
            max_log = np.maximum(log_p0, log_p1)
            p0 = np.exp(log_p0 - max_log)
            p1 = np.exp(log_p1 - max_log)
            denom = p0 + p1 + eps
            gamma = p1 / denom
        y0_clean = clean_z.shape[0]
        y1_dirty = dirty_z.shape[0]
        y0_unlabeled = float(n_unlabeled) - gamma.sum()
        y1_unlabeled = gamma.sum()
        y0_total = y0_clean + y0_unlabeled
        y1_total = y1_dirty + y1_unlabeled
        if y0_total <= 0 or y1_total <= 0:
            break
        num_alpha = np.zeros(n_rules)
        num_beta = np.zeros(n_rules)
        if clean_z.size > 0:
            num_alpha += clean_z.sum(axis=0)
        if dirty_z.size > 0:
            num_beta += dirty_z.sum(axis=0)
        if n_unlabeled > 0:
            for r in range(n_rules):
                zr = unlabeled_z[:, r]
                num_alpha[r] += float(((1.0 - gamma) * zr).sum())
                num_beta[r] += float((gamma * zr).sum())
        alpha = np.clip(num_alpha / y0_total, eps, 1 - eps)
        beta = np.clip(num_beta / y1_total, eps, 1 - eps)
        total_points = y0_total + y1_total
        pi1 = min(max(y1_total / total_points, eps), 1 - eps)
        pi0 = 1.0 - pi1
    return gamma, alpha, beta


def _setup_cleanem_logger(output_dir: str, dataset_name: str) -> logging.Logger:
    """Setup logger for clean_em mode with both console and file handlers."""
    logger = logging.getLogger("CleanEM")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler - INFO level (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - DEBUG level (detailed format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{timestamp}_{dataset_name}_cleanem.log")
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def run_clean_em_mode(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]
    
    # Setup logger
    logger = _setup_cleanem_logger(args.output_dir, dataset_name)
    logger.info("=" * 60)
    logger.info("CleanEM Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"[1/4] Profiling dirty dataset: {args.dirty_csv}")
    logger.info(f"Configuration:")
    logger.info(f"  - clean_seed_percent: {args.clean_seed_percent}")
    logger.info(f"  - synthetic_per_family: {args.synthetic_per_family}")
    logger.info(f"  - em_max_iters: {args.em_max_iters}")
    logger.info(f"  - em_prior_dirty: {args.em_prior_dirty}")
    logger.info(f"  - score_threshold: {args.score_threshold}")
    logger.info(f"  - top_k: {args.top_k}")
    
    profiler = PandasProfiler(args.dirty_csv)
    df = profiler.df
    metadata = profiler.get_metadata()
    
    logger.info(f"Dataset profile complete:")
    logger.info(f"  - Total rows: {len(df)}")
    logger.info(f"  - Total columns: {len(metadata)}")
    logger.info(f"  - Columns: {list(metadata.keys())}")
    
    # Log per-column metadata summary
    logger.debug("Per-column metadata summary:")
    for col, meta in metadata.items():
        col_type = meta.get('type', 'unknown')
        null_count = meta.get('null_count', 0)
        unique_count = meta.get('unique_count', 0)
        logger.debug(f"  - {col}: type={col_type}, nulls={null_count}, unique={unique_count}")
    
    logger.info("[2/4] Generating clean rules and building rule pool")
    factory = AgentFactory(base_url=args.base_url, model=args.model, max_workers=args.max_workers)
    rule_pool = _build_clean_rule_pool(df, metadata, factory)
    
    if not rule_pool:
        logger.error("No clean rules generated; exiting clean_em mode.")
        print("No clean rules generated; exiting clean_em mode.")
        return
    
    # Log rule pool statistics
    total_rules = sum(len(rules) for rules in rule_pool.values())
    logger.info(f"Rule pool built successfully:")
    logger.info(f"  - Total columns with rules: {len(rule_pool)}")
    logger.info(f"  - Total rules: {total_rules}")
    
    family_counts = {"missing": 0, "outlier": 0, "pattern": 0}
    for column, rules in rule_pool.items():
        col_family_counts = {"missing": 0, "outlier": 0, "pattern": 0}
        for rule in rules:
            fam = rule.get("family")
            if fam in col_family_counts:
                col_family_counts[fam] += 1
                family_counts[fam] += 1
        logger.info(f"  - Column '{column}': {len(rules)} rules "
                    f"(missing={col_family_counts['missing']}, "
                    f"outlier={col_family_counts['outlier']}, "
                    f"pattern={col_family_counts['pattern']})")
        
        # Log detailed rule information for each column
        for rule in rules:
            rule_name = rule.get("rule_name", "unknown")
            rule_family = rule.get("family", "unknown")
            rule_str = rule.get("rule_str", "")
            logger.info(f"    - Rule '{rule_name}' [{rule_family}]: {rule_str}")
    
    logger.info(f"  - Rule family distribution: "
                f"missing={family_counts['missing']}, "
                f"outlier={family_counts['outlier']}, "
                f"pattern={family_counts['pattern']}")
    
    logger.info("[3/4] Running clean-rule EM calibration")
    dirty_agent = DirtyExampleAgent(base_url=args.base_url, model=args.model)
    reflection_agent = CleanRuleReflectionAgent(base_url=args.base_url, model=args.model)
    all_scores: Dict[Tuple[int, str], Dict[str, float]] = {}
    
    # Track statistics per column and family
    em_stats = {}
    
    for col_idx, (column, rules) in enumerate(rule_pool.items(), 1):
        logger.info(f"Processing column {col_idx}/{len(rule_pool)}: '{column}'")
        logger.debug(f"  - Number of rules: {len(rules)}")
        
        seed_indices, outputs = _select_clean_seeds_for_column(
            df,
            column,
            rules,
            args.clean_seed_percent,
        )
        
        if outputs.size == 0:
            logger.warning(f"  - No rule outputs for column '{column}', skipping")
            continue
        
        n_seeds = len(seed_indices)
        n_unlabeled = len(df) - n_seeds
        logger.info(f"  - Clean seeds selected: {n_seeds} ({args.clean_seed_percent*100:.1f}% of {len(df)} rows)")
        logger.debug(f"  - Seed indices (first 10): {seed_indices[:10].tolist()}")
        
        families = ["missing", "outlier", "pattern"]
        family_rule_indices: Dict[str, List[int]] = {f: [] for f in families}
        for idx, rule in enumerate(rules):
            fam = rule["family"]
            if fam in family_rule_indices:
                family_rule_indices[fam].append(idx)
        
        em_stats[column] = {}
        
        for family in families:
            cols = family_rule_indices[family]
            if not cols:
                logger.info(f"  - Family '{family}': no rules, skipping")
                continue
            
            logger.info(f"  - Family '{family}': {len(cols)} rules")
            
            # Log specific rules used for this family
            for idx in cols:
                rule = rules[idx]
                rule_name = rule.get("rule_name", "unknown")
                rule_str = rule.get("rule_str", "")
                logger.info(f"    - Using rule '{rule_name}': {rule_str}")
            
            family_z = outputs[:, cols]
            seed_mask = np.zeros(len(df), dtype=bool)
            seed_mask[seed_indices] = True
            clean_z = family_z[seed_mask]
            unlabeled_z = family_z[~seed_mask]
            
            logger.debug(f"    - Clean Z shape: {clean_z.shape}, Unlabeled Z shape: {unlabeled_z.shape}")
            
            seeds_values = [df.iloc[i][column] for i in seed_indices]
            logger.debug(f"    - Generating dirty examples for family '{family}'")
            
            dirty_examples = dirty_agent.generate_dirty_examples(
                column,
                metadata.get(column, {}),
                seeds_values,
                family,
                max_examples=args.synthetic_per_family,
            )
            
            n_dirty = len(dirty_examples)
            logger.info(f"    - Generated {n_dirty} dirty examples for family '{family}'")
            logger.debug(f"    - Dirty example values (first 5): {[ex['value'] for ex in dirty_examples[:5]]}")
            
            dirty_vals = [ex["value"] for ex in dirty_examples]
            dirty_z_list: List[List[int]] = []
            for val in dirty_vals:
                row_like = {column: val}
                row_series = pd.Series(row_like)
                row_outputs: List[int] = []
                for idx in cols:
                    func = rules[idx]["rule_func"]
                    row_outputs.append(int(_invoke_rule(func, val, row_series)))
                dirty_z_list.append(row_outputs)
            
            dirty_z = np.array(dirty_z_list, dtype=int) if dirty_z_list else np.zeros((0, len(cols)), dtype=int)
            logger.debug(f"    - Dirty Z shape: {dirty_z.shape}")
            
            if clean_z.size > 0 and dirty_z.size > 0:
                clean_pass = clean_z.mean(axis=0)
                dirty_pass = dirty_z.mean(axis=0)
                bad_clean_threshold = 0.6
                bad_dirty_threshold = 0.6
                improve_margin = 0.02
                max_reflections = 3
                candidates: List[Tuple[int, int, float, float]] = []
                for local_idx, rule_idx in enumerate(cols):
                    cp = float(clean_pass[local_idx])
                    dp = float(dirty_pass[local_idx])
                    if cp < bad_clean_threshold or dp > bad_dirty_threshold:
                        candidates.append((local_idx, rule_idx, cp, dp))
                reflections_done = 0
                for local_idx, rule_idx, cp, dp in candidates:
                    if reflections_done >= max_reflections:
                        break
                    rule = rules[rule_idx]
                    rule_name = rule.get("rule_name", "unknown")
                    rule_str = rule.get("rule_str", "")
                    clean_mis_indices = np.where(clean_z[:, local_idx] == 0)[0]
                    dirty_mis_indices = np.where(dirty_z[:, local_idx] == 1)[0]
                    clean_mis_examples: List[Dict[str, Any]] = []
                    for idx_seed in clean_mis_indices[:10]:
                        if idx_seed < len(seeds_values):
                            clean_mis_examples.append({"value": seeds_values[idx_seed]})
                    dirty_mis_examples: List[Dict[str, Any]] = []
                    for idx_dirty in dirty_mis_indices[:10]:
                        if idx_dirty < len(dirty_examples):
                            ex = dirty_examples[idx_dirty]
                            dirty_mis_examples.append(
                                {"value": ex.get("value"), "reason": ex.get("reason", "")}
                            )
                    if not clean_mis_examples and not dirty_mis_examples:
                        continue
                    logger.info(
                        f"    - Reflecting rule '{rule_name}' before calibration "
                        f"(clean_pass={cp:.3f}, dirty_pass={dp:.3f})"
                    )
                    new_rule_str = reflection_agent.refine_clean_rule(
                        column,
                        metadata.get(column, {}),
                        family,
                        rule_str,
                        clean_mis_examples,
                        dirty_mis_examples,
                    )
                    if not new_rule_str or new_rule_str == rule_str:
                        continue
                    try:
                        new_rule_func = eval(new_rule_str, safe_dict)
                    except Exception:
                        continue
                    n_rows = len(df)
                    new_all = np.zeros(n_rows, dtype=int)
                    for row_idx_df, (_, row_df) in enumerate(df.iterrows()):
                        value_df = row_df[column]
                        new_all[row_idx_df] = int(_invoke_rule(new_rule_func, value_df, row_df))
                    new_clean_col = new_all[seed_mask]
                    new_unlabeled_col = new_all[~seed_mask]
                    new_dirty_col = np.zeros(len(dirty_vals), dtype=int)
                    for idx_dirty_val, val in enumerate(dirty_vals):
                        row_like = {column: val}
                        row_series = pd.Series(row_like)
                        new_dirty_col[idx_dirty_val] = int(_invoke_rule(new_rule_func, val, row_series))
                    new_cp = float(new_clean_col.mean()) if new_clean_col.size > 0 else 0.0
                    new_dp = float(new_dirty_col.mean()) if new_dirty_col.size > 0 else 1.0
                    old_loss = (1.0 - cp) + dp
                    new_loss = (1.0 - new_cp) + new_dp
                    if new_loss <= old_loss - improve_margin and new_cp >= args.calib_min_clean_pass:
                        clean_z[:, local_idx] = new_clean_col
                        unlabeled_z[:, local_idx] = new_unlabeled_col
                        dirty_z[:, local_idx] = new_dirty_col
                        rules[rule_idx]["rule_str"] = new_rule_str
                        rules[rule_idx]["rule_func"] = new_rule_func
                        clean_pass[local_idx] = new_cp
                        dirty_pass[local_idx] = new_dp
                        reflections_done += 1
                        logger.info(
                            f"    - Refined rule '{rule_name}': "
                            f"clean_pass {cp:.3f}->{new_cp:.3f}, dirty_pass {dp:.3f}->{new_dp:.3f}"
                        )
                keep_mask = (clean_pass >= args.calib_min_clean_pass) & (
                    dirty_pass <= args.calib_max_dirty_pass
                )
                for local_idx, rule_idx in enumerate(cols):
                    rule = rules[rule_idx]
                    rule_name = rule.get("rule_name", "unknown")
                    cp = float(clean_pass[local_idx])
                    dp = float(dirty_pass[local_idx])
                    decision = "KEEP" if keep_mask[local_idx] else "DROP"
                    logger.info(
                        f"    - Calib rule '{rule_name}': "
                        f"clean_pass={cp:.3f}, dirty_pass={dp:.3f}, decision={decision}"
                    )
                if not keep_mask.any():
                    logger.info(
                        f"    - All rules rejected by calibration for family '{family}', skipping EM"
                    )
                    continue
                kept_local_indices = np.where(keep_mask)[0]
                clean_z = clean_z[:, kept_local_indices]
                unlabeled_z = unlabeled_z[:, kept_local_indices]
                dirty_z = dirty_z[:, kept_local_indices]
                cols = [cols[i] for i in kept_local_indices]
                logger.info(
                    f"    - {keep_mask.sum()}/{len(keep_mask)} rules kept after calibration"
                )
            
            logger.debug(f"    - Running EM algorithm (max_iters={args.em_max_iters})")
            gamma, alpha, beta = _run_em_for_family(
                clean_z,
                dirty_z,
                unlabeled_z,
                args.em_max_iters,
                args.em_prior_dirty,
            )
            
            # Log EM results
            if len(gamma) > 0:
                gamma_mean = float(gamma.mean())
                gamma_std = float(gamma.std())
                gamma_min = float(gamma.min())
                gamma_max = float(gamma.max())
                high_scores = int((gamma > 0.5).sum())
                logger.info(f"    - EM complete: gamma_mean={gamma_mean:.4f}, "
                           f"gamma_std={gamma_std:.4f}, range=[{gamma_min:.4f}, {gamma_max:.4f}]")
                logger.info(f"    - High scores (>0.5): {high_scores}/{len(gamma)}")
                logger.debug(f"    - Alpha (rule precision on clean): {alpha}")
                logger.debug(f"    - Beta (rule recall on dirty): {beta}")
                
                em_stats[column][family] = {
                    "n_rules": len(cols),
                    "n_clean": clean_z.shape[0],
                    "n_dirty": dirty_z.shape[0],
                    "n_unlabeled": unlabeled_z.shape[0],
                    "gamma_mean": gamma_mean,
                    "gamma_std": gamma_std,
                    "high_scores": high_scores
                }
            
            unlabeled_indices = np.where(~seed_mask)[0]
            family_key = {
                "missing": "S_missing",
                "outlier": "S_outlier",
                "pattern": "S_pattern",
            }[family]
            
            scores_assigned = 0
            for local_idx, row_idx in enumerate(unlabeled_indices):
                key = (int(row_idx), column)
                if key not in all_scores:
                    all_scores[key] = {"S_missing": 0.0, "S_outlier": 0.0, "S_pattern": 0.0}
                all_scores[key][family_key] = float(gamma[local_idx]) if local_idx < len(gamma) else 0.0
                scores_assigned += 1
            
            logger.debug(f"    - Assigned scores to {scores_assigned} cells for family '{family}'")
    
    logger.info("EM calibration complete for all columns")
    
    # Build results
    results: List[Dict[str, Any]] = []
    for (row_idx, column), scores in all_scores.items():
        s_missing = scores.get("S_missing", 0.0)
        s_outlier = scores.get("S_outlier", 0.0)
        s_pattern = scores.get("S_pattern", 0.0)
        s_total = (s_missing + s_outlier + s_pattern) / 3.0
        results.append(
            {
                "row_index": row_idx,
                "column": column,
                "value": df.iloc[row_idx][column],
                "S_missing": s_missing,
                "S_outlier": s_outlier,
                "S_pattern": s_pattern,
                "S_total": s_total,
            }
        )
    
    results.sort(key=lambda x: x["S_total"], reverse=True)
    
    # Log results statistics
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total cells scored: {len(results)}")
    
    if results:
        all_totals = [r["S_total"] for r in results]
        logger.info(f"S_total distribution:")
        logger.info(f"  - Mean: {sum(all_totals)/len(all_totals):.4f}")
        logger.info(f"  - Min: {min(all_totals):.4f}")
        logger.info(f"  - Max: {max(all_totals):.4f}")
        
        above_threshold = [s for s in all_totals if s >= args.score_threshold]
        logger.info(f"  - Above threshold ({args.score_threshold}): {len(above_threshold)}/{len(all_totals)} "
                   f"({len(above_threshold)/len(all_totals)*100:.1f}%)")
        
        # Per-family statistics
        for family_key in ["S_missing", "S_outlier", "S_pattern"]:
            scores = [r[family_key] for r in results]
            above = sum(1 for s in scores if s >= args.score_threshold)
            logger.info(f"  - {family_key}: mean={sum(scores)/len(scores):.4f}, "
                       f"above_threshold={above}/{len(scores)}")
    
    top_k = min(args.top_k, len(results))
    logger.info(f"\nTop {top_k} highest scoring cells by S_total:")
    print(f"\nTop {top_k} highest scoring cells by S_total:")
    
    for item in results[:top_k]:
        log_msg = (f"Row {item['row_index']}, Column '{item['column']}': "
                  f"value={item['value']!r}, "
                  f"S_missing={item['S_missing']:.4f}, "
                  f"S_outlier={item['S_outlier']:.4f}, "
                  f"S_pattern={item['S_pattern']:.4f}, "
                  f"S_total={item['S_total']:.4f}")
        logger.info(f"  {log_msg}")
        print(log_msg)
    
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.output_dir,
            f"{timestamp}_{dataset_name}_clean_em_scores.csv",
        )
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"\nSaved all cell scores to: {output_path}")
        print(f"\nSaved all cell scores to: {output_path}")
        
        if args.clean_csv and os.path.exists(args.clean_csv):
            logger.info("\nEvaluating clean_em scores against ground truth:")
            print("\nEvaluating clean_em scores against ground truth:")
            clean_df = pd.read_csv(args.clean_csv)
            judge = Judge()
            detected_errors = [
                {"row_index": r["row_index"], "column": r["column"]}
                for r in results
                if r["S_total"] >= args.score_threshold
            ]
            logger.info(f"Detected {len(detected_errors)} cells with S_total >= {args.score_threshold}")
            print(f"Detected {len(detected_errors)} cells with S_total >= {args.score_threshold}")
            metrics = judge.evaluate_with_ground_truth(df, clean_df, detected_errors)
            judge.print_evaluation_summary(metrics)
            
            # Log metrics
            if hasattr(metrics, 'get'):
                logger.info("Evaluation Metrics:")
                for key, value in metrics.items() if isinstance(metrics, dict) else []:
                    logger.info(f"  - {key}: {value}")
    
    logger.info("\n[4/4] clean_em pipeline complete.")
    print("\n[4/4] clean_em pipeline complete.")
    logger.info("=" * 60)


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
    print(f"\nRefinement log started: {dataset_name}")
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
        print("\nInitial performance (before refinement):")
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
            print(f"\nProcessing column: {column}")
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
    print(f"\nRefinement log saved: {dataset_name}")

    return best_rules, all_history


def run_dual_mode(args: argparse.Namespace) -> None:
    # Create console log file with the same timestamp format
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]
    console_log_path = os.path.join(args.output_dir, f"{timestamp}_{dataset_name}_running_log.log")

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

        judge = Judge(threshold=args.vr_threshold, violation_threshold=args.vr_threshold)

        print("[4/7] Standalone rule-based error detection")
        base_evaluation_results = judge.evaluate_rules(profiler.df, base_rules)
        accepted_base_rules = judge.get_accepted_rules(base_evaluation_results)
        judge.print_summary(accepted_base_rules)
        # judge.print_detected_errors(base_detected_errors)

        # Evaluate standalone clean rules (Completeness/Accuracy/Pattern/Relationship)
        print("\nStandalone clean rules evaluation:")
        clean_evaluation_results = judge.evaluate_rules(profiler.df, clean_rules, rule_type="clean")
        accepted_clean_rules = judge.get_accepted_rules(clean_evaluation_results)
        judge.print_summary(accepted_clean_rules, rule_type="clean")

        # Combined error detection using AND/OR logic
        # Clean Rule (AND): All clean rules must be satisfied
        # Dirty Rule (OR): Violating any dirty rule marks as potentially dirty
        # Error = (NOT all clean rules satisfied) AND (at least one dirty rule violated)
        print("\nCombined AND/OR error detection:")
        base_detected_errors = judge.get_detected_errors(
            accepted_base_rules,      # dirty rules (OR logic)
            accepted_clean_rules      # clean rules (AND logic)
        )
        print(f"Detected {len(base_detected_errors)} errors using combined AND/OR logic")

        # Evaluate standalone base rules against ground truth if clean CSV is provided
        base_metrics_summary = None
        clean_df = None
        if args.clean_csv:
            print("\nGround truth evaluation (base rules):")
            clean_df = pd.read_csv(args.clean_csv)
            base_metrics_summary = judge.evaluate_with_ground_truth(
                profiler.df,
                clean_df,
                base_detected_errors
            )
            judge.print_evaluation_summary(base_metrics_summary)

        print("\n[5/7] Clean rule-level refinement")
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

        print("\n[6/7] Validating disjointness of refined rules")
        validator = DisjointnessValidator(gap_tolerance=args.grey_tolerance)
        validation_result = validator.validate_batch(profiler.df, best_rules)
        print(validator.report_violations(validation_result))

        coverage_gaps = sorted(set(metadata.keys()) - set(best_rules.keys()))

        print("[7/7] Evaluating final dual rules on the dataset")
        evaluation_payload = _materialize_rule_payload(best_rules)
        evaluation_results = judge.evaluate_dual_rules(
            profiler.df,
            evaluation_payload,
            grey_tolerance=args.grey_tolerance,
            metadata=metadata
        )
        detected_dirty_values = judge.get_detected_dirty_values(best_rules, profiler.df)
        judge.print_dual_summary(best_rules, evaluation_results)

        refined_metrics_summary = None
        # Evaluate refined dual rules against ground truth if clean CSV is provided
        if args.clean_csv and clean_df is not None:
            print("\nGround truth evaluation (refined rules):")
            refined_metrics_summary = judge.evaluate_with_ground_truth(
                profiler.df,
                clean_df,
                detected_dirty_values
            )
            judge.print_evaluation_summary(refined_metrics_summary)

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


def main() -> None:
    args = parse_args()
    if args.mode == "dual":
        run_dual_mode(args)
    else:
        run_clean_em_mode(args)


if __name__ == "__main__":
    main()
