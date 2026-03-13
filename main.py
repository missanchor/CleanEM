"""
Command-line entry point for the agentic error detector.

Supports a dual verification pipeline with clean rule-level refinement.
"""
import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from judge import Judge
from agent import AgentFactory, DirtyExampleAgent, CleanRuleReflectionAgent
from profiler import PandasProfiler, DEFAULT_MISSING_TOKEN_SET
from validator import DisjointnessValidator
from core.utils import safe_dict


# CleanEM Logger
cleanem_logger = logging.getLogger("CleanEM")

FAMILIES = ["missing", "outlier", "pattern"]
FAMILY_SCORE_KEYS = {
    "missing": "S_missing",
    "outlier": "S_outlier",
    "pattern": "S_pattern",
}
ANCHOR_HARD_CLEAN = "hard_clean"
ANCHOR_HARD_DIRTY = "hard_dirty"
ANCHOR_UNLABELED = "unlabeled"
ANCHOR_ABSTAIN = "abstain"


@dataclass
class GroupRecord:
    normalized_signature: str
    row_indices: List[int]
    weight: int
    rule_outputs_by_family: Dict[str, np.ndarray]
    anchor_state: str
    anchor_source: str
    representative_value: Any


@dataclass
class FamilyRunStatus:
    status: str
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Error Detector CLI with dual and clean+EM modes."
    )
    parser.add_argument(
        "--dirty_csv",
        default="data/hospital_error-01.csv",
        help="Path to the dirty/error-prone CSV that needs inspection."
    )
    parser.add_argument(
        "--clean_csv",
        default="data/hospital_clean.csv",
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


def _compute_rule_outputs_for_column(
    df: pd.DataFrame,
    column: str,
    rules: List[Dict[str, Any]],
) -> np.ndarray:
    n_rows = len(df)
    if n_rows == 0 or not rules:
        return np.zeros((0, len(rules)), dtype=int)
    outputs = np.zeros((n_rows, len(rules)), dtype=int)
    for j, rule in enumerate(rules):
        func = rule["rule_func"]
        for i, (_, row) in enumerate(df.iterrows()):
            value = row[column]
            outputs[i, j] = int(_invoke_rule(func, value, row))
    return outputs


def _normalize_signature(value: Any) -> str:
    if value is None:
        return "<na>"
    if isinstance(value, float) and np.isnan(value):
        return "<na>"
    normalized = str(value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized if normalized else "<empty>"


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    normalized = str(value).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _shape_signature(value: Any) -> str:
    if value is None:
        return "<empty>"
    tokens: List[str] = []
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
            tokens.append(char)
    return "".join(tokens) if tokens else "<empty>"


def _is_missing_like(value: Any, metadata: Dict[str, Any]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    normalized = _normalize_token(value)
    dominant_tokens = set(metadata.get("dominant_missing_tokens") or [])
    return normalized in DEFAULT_MISSING_TOKEN_SET or normalized in dominant_tokens


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            numeric = float(value)
        except Exception:
            return None
        return None if np.isnan(numeric) else numeric
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        numeric = float(text)
    except Exception:
        return None
    return None if np.isnan(numeric) else numeric


def _build_signature_weights(df: pd.DataFrame, column: str) -> np.ndarray:
    signatures = [_normalize_signature(value) for value in df[column].tolist()]
    counts: Dict[str, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    return np.array([counts[signature] for signature in signatures], dtype=int)


def _build_anchor_groups(
    df: pd.DataFrame,
    column: str,
    metadata: Dict[str, Any],
    family: str,
) -> Tuple[List[GroupRecord], FamilyRunStatus]:
    groups_by_signature: Dict[str, GroupRecord] = {}
    for row_idx, value in enumerate(df[column].tolist()):
        signature = _normalize_signature(value)
        record = groups_by_signature.get(signature)
        if record is None:
            record = GroupRecord(
                normalized_signature=signature,
                row_indices=[],
                weight=0,
                rule_outputs_by_family={},
                anchor_state=ANCHOR_UNLABELED,
                anchor_source="unlabeled",
                representative_value=value,
            )
            groups_by_signature[signature] = record
        record.row_indices.append(row_idx)
        record.weight += 1

    groups = list(groups_by_signature.values())
    if not groups:
        return [], FamilyRunStatus(status="abstained", reason="empty_column")

    if family == "missing":
        for record in groups:
            if _is_missing_like(record.representative_value, metadata):
                record.anchor_state = ANCHOR_HARD_DIRTY
                record.anchor_source = "real_missing_token"
            else:
                record.anchor_state = ANCHOR_HARD_CLEAN
                record.anchor_source = "real_non_missing"
        return groups, FamilyRunStatus(status="ran")

    if family == "outlier":
        if metadata.get("type") != "numeric":
            for record in groups:
                record.anchor_state = ANCHOR_ABSTAIN
                record.anchor_source = "categorical_outlier_disabled"
            return groups, FamilyRunStatus(status="abstained", reason="categorical_outlier_disabled")
        quantiles = metadata.get("quantiles") or {}
        p25 = quantiles.get("p25")
        p75 = quantiles.get("p75")
        p05 = quantiles.get("p05")
        p95 = quantiles.get("p95")
        if p25 is None or p75 is None:
            for record in groups:
                record.anchor_state = ANCHOR_ABSTAIN
                record.anchor_source = "missing_numeric_stats"
            return groups, FamilyRunStatus(status="abstained", reason="missing_numeric_stats")
        iqr = float(metadata.get("iqr") or 0.0)
        inner_low = float(p25)
        inner_high = float(p75)
        if inner_low == inner_high:
            inner_low = float(p05 if p05 is not None else p25)
            inner_high = float(p95 if p95 is not None else p75)
        if iqr > 0:
            outer_low = float(p25) - 6.0 * iqr
            outer_high = float(p75) + 6.0 * iqr
        else:
            outer_low = float(p05 if p05 is not None else p25)
            outer_high = float(p95 if p95 is not None else p75)
        for record in groups:
            value = record.representative_value
            if _is_missing_like(value, metadata):
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "missing_reserved_for_missing_family"
                continue
            numeric = _coerce_numeric(value)
            if numeric is None:
                record.anchor_state = ANCHOR_HARD_DIRTY
                record.anchor_source = "non_numeric_value"
            elif inner_low <= numeric <= inner_high:
                record.anchor_state = ANCHOR_HARD_CLEAN
                record.anchor_source = "central_numeric_band"
            elif numeric < outer_low or numeric > outer_high:
                record.anchor_state = ANCHOR_HARD_DIRTY
                record.anchor_source = "extreme_numeric_value"
            else:
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "numeric_borderline"
        return groups, FamilyRunStatus(status="ran")

    if family == "pattern":
        length_distribution = sorted(
            metadata.get("length_distribution") or [],
            key=lambda item: item.get("count", 0),
            reverse=True,
        )
        shape_distribution = sorted(
            metadata.get("shape_distribution") or [],
            key=lambda item: item.get("count", 0),
            reverse=True,
        )
        top_lengths = [int(item["length"]) for item in length_distribution[:2] if item.get("ratio", 0.0) >= 0.15]
        top_shapes = [str(item["shape"]) for item in shape_distribution[:2] if item.get("ratio", 0.0) >= 0.15]
        length_coverage = float(sum(item.get("ratio", 0.0) for item in length_distribution[:2]))
        shape_coverage = float(sum(item.get("ratio", 0.0) for item in shape_distribution[:2]))
        if length_coverage < 0.75 and shape_coverage < 0.75:
            for record in groups:
                record.anchor_state = ANCHOR_ABSTAIN
                record.anchor_source = "weak_structure_evidence"
            return groups, FamilyRunStatus(status="abstained", reason="weak_structure_evidence")
        for record in groups:
            value = record.representative_value
            if _is_missing_like(value, metadata):
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "missing_reserved_for_missing_family"
                continue
            text = str(value).strip()
            length_ok = True if not top_lengths else len(text) in top_lengths
            shape_ok = True if not top_shapes else _shape_signature(text) in top_shapes
            if length_ok and shape_ok:
                record.anchor_state = ANCHOR_HARD_CLEAN
                record.anchor_source = "stable_structure"
            elif not length_ok or not shape_ok:
                record.anchor_state = ANCHOR_HARD_DIRTY
                record.anchor_source = "structure_violation"
            else:
                record.anchor_state = ANCHOR_UNLABELED
                record.anchor_source = "weak_structure_match"
        return groups, FamilyRunStatus(status="ran")

    for record in groups:
        record.anchor_state = ANCHOR_ABSTAIN
        record.anchor_source = "unsupported_family"
    return groups, FamilyRunStatus(status="abstained", reason="unsupported_family")


def _build_anchor_state_arrays(n_rows: int, anchor_groups: List[GroupRecord]) -> Tuple[np.ndarray, np.ndarray]:
    anchor_states = np.full(n_rows, ANCHOR_UNLABELED, dtype=object)
    anchor_sources = np.full(n_rows, "unlabeled", dtype=object)
    for record in anchor_groups:
        for row_idx in record.row_indices:
            anchor_states[row_idx] = record.anchor_state
            anchor_sources[row_idx] = record.anchor_source
    return anchor_states, anchor_sources


def _build_em_groups(
    anchor_groups: List[GroupRecord],
    family: str,
    family_outputs: np.ndarray,
    df: pd.DataFrame,
    column: str,
) -> List[GroupRecord]:
    em_groups: Dict[Tuple[str, Tuple[int, ...], str, str], GroupRecord] = {}
    for anchor_group in anchor_groups:
        if anchor_group.anchor_state == ANCHOR_ABSTAIN:
            continue
        for row_idx in anchor_group.row_indices:
            obs = tuple(int(v) for v in family_outputs[row_idx].tolist()) if family_outputs.size else tuple()
            key = (
                anchor_group.normalized_signature,
                obs,
                anchor_group.anchor_state,
                anchor_group.anchor_source,
            )
            if key not in em_groups:
                em_groups[key] = GroupRecord(
                    normalized_signature=anchor_group.normalized_signature,
                    row_indices=[],
                    weight=0,
                    rule_outputs_by_family={family: np.array(obs, dtype=int)},
                    anchor_state=anchor_group.anchor_state,
                    anchor_source=anchor_group.anchor_source,
                    representative_value=df.iloc[row_idx][column],
                )
            em_groups[key].row_indices.append(int(row_idx))
            em_groups[key].weight += 1
    return list(em_groups.values())


def _weighted_binary_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0:
        return float("nan")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float("nan")
    return float(np.dot(values.astype(float), weights.astype(float)) / total_weight)


def _select_representative_values(anchor_groups: List[GroupRecord], limit: int = 20) -> List[Any]:
    clean_groups = [record for record in anchor_groups if record.anchor_state == ANCHOR_HARD_CLEAN]
    clean_groups.sort(key=lambda record: record.weight, reverse=True)
    selected: List[Any] = []
    seen_signatures = set()
    for record in clean_groups:
        if record.normalized_signature in seen_signatures:
            continue
        seen_signatures.add(record.normalized_signature)
        selected.append(record.representative_value)
        if len(selected) >= limit:
            break
    return selected


def _run_weighted_em_for_family(
    group_z: np.ndarray,
    group_weights: np.ndarray,
    anchor_states: np.ndarray,
    max_iters: int,
    prior_dirty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_groups = group_z.shape[0]
    n_rules = group_z.shape[1] if group_z.ndim == 2 else 0
    eps = 1e-3
    if n_groups == 0 or n_rules == 0:
        return np.zeros(n_groups), np.zeros(n_rules), np.zeros(n_rules), float("nan")

    clean_mask = anchor_states == ANCHOR_HARD_CLEAN
    dirty_mask = anchor_states == ANCHOR_HARD_DIRTY
    unlabeled_mask = anchor_states == ANCHOR_UNLABELED

    alpha = np.full(n_rules, 0.9)
    beta = np.full(n_rules, 0.3)
    if clean_mask.any():
        alpha = np.clip(
            (group_z[clean_mask] * group_weights[clean_mask, None]).sum(axis=0) / max(group_weights[clean_mask].sum(), eps),
            eps,
            1 - eps,
        )
    if dirty_mask.any():
        beta = np.clip(
            (group_z[dirty_mask] * group_weights[dirty_mask, None]).sum(axis=0) / max(group_weights[dirty_mask].sum(), eps),
            eps,
            1 - eps,
        )

    pi1 = min(max(prior_dirty, eps), 1 - eps)
    gamma = np.zeros(n_groups, dtype=float)
    gamma[dirty_mask] = 1.0
    gamma[clean_mask] = 0.0
    gamma[unlabeled_mask] = pi1

    for _ in range(max_iters):
        if unlabeled_mask.any():
            z_u = group_z[unlabeled_mask]
            log_p0 = np.full(z_u.shape[0], np.log(1.0 - pi1))
            log_p1 = np.full(z_u.shape[0], np.log(pi1))
            for rule_idx in range(n_rules):
                zr = z_u[:, rule_idx]
                log_p0 += zr * np.log(alpha[rule_idx] + eps) + (1 - zr) * np.log(1.0 - alpha[rule_idx] + eps)
                log_p1 += zr * np.log(beta[rule_idx] + eps) + (1 - zr) * np.log(1.0 - beta[rule_idx] + eps)
            max_log = np.maximum(log_p0, log_p1)
            p0 = np.exp(log_p0 - max_log)
            p1 = np.exp(log_p1 - max_log)
            gamma[unlabeled_mask] = p1 / (p0 + p1 + eps)

        gamma[clean_mask] = 0.0
        gamma[dirty_mask] = 1.0

        y1_total = float((group_weights * gamma).sum())
        y0_total = float((group_weights * (1.0 - gamma)).sum())
        if y0_total <= 0 or y1_total <= 0:
            break

        weighted_clean = (group_weights * (1.0 - gamma))[:, None]
        weighted_dirty = (group_weights * gamma)[:, None]
        alpha = np.clip(weighted_clean.T.dot(group_z).ravel() / y0_total, eps, 1 - eps)
        beta = np.clip(weighted_dirty.T.dot(group_z).ravel() / y1_total, eps, 1 - eps)
        total_weight = y0_total + y1_total
        pi1 = min(max(y1_total / max(total_weight, eps), eps), 1 - eps)

    return gamma, alpha, beta, pi1


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
    logger.info(f"  - clean_seed_percent (legacy/unused): {args.clean_seed_percent}")
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
    n_rows = len(df)
    column_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    column_statuses: Dict[str, Dict[str, FamilyRunStatus]] = {}
    column_anchor_sources: Dict[str, Dict[str, np.ndarray]] = {}
    column_signature_weights: Dict[str, np.ndarray] = {}
    em_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for col_idx, (column, rules) in enumerate(rule_pool.items(), 1):
        logger.info(f"Processing column {col_idx}/{len(rule_pool)}: '{column}'")
        logger.debug(f"  - Number of rules: {len(rules)}")

        outputs = _compute_rule_outputs_for_column(df, column, rules)
        if outputs.size == 0:
            logger.warning(f"  - No rule outputs for column '{column}', skipping")
            continue

        column_outputs[column] = {
            "S_missing": np.full(n_rows, np.nan, dtype=float),
            "S_outlier": np.full(n_rows, np.nan, dtype=float),
            "S_pattern": np.full(n_rows, np.nan, dtype=float),
        }
        column_anchor_sources[column] = {
            "missing": np.full(n_rows, "unlabeled", dtype=object),
            "outlier": np.full(n_rows, "abstained", dtype=object),
            "pattern": np.full(n_rows, "abstained", dtype=object),
        }
        column_signature_weights[column] = _build_signature_weights(df, column)

        family_rule_indices: Dict[str, List[int]] = {f: [] for f in FAMILIES}
        for idx, rule in enumerate(rules):
            fam = rule["family"]
            if fam in family_rule_indices:
                family_rule_indices[fam].append(idx)

        column_statuses[column] = {
            family: FamilyRunStatus(status="abstained", reason="family_not_processed")
            for family in FAMILIES
        }
        em_stats[column] = {}

        for family in FAMILIES:
            cols = family_rule_indices[family]
            if not cols:
                logger.info(f"  - Family '{family}': no rules, abstained")
                column_statuses[column][family] = FamilyRunStatus(status="abstained", reason="no_rules")
                column_anchor_sources[column][family][:] = "abstained:no_rules"
                continue

            logger.info(f"  - Family '{family}': {len(cols)} rules")
            for idx in cols:
                rule = rules[idx]
                rule_name = rule.get("rule_name", "unknown")
                rule_str = rule.get("rule_str", "")
                logger.info(f"    - Using rule '{rule_name}': {rule_str}")

            family_z = outputs[:, cols]
            anchor_groups, family_status = _build_anchor_groups(df, column, metadata.get(column, {}), family)
            column_statuses[column][family] = family_status
            anchor_states_by_row, anchor_sources_by_row = _build_anchor_state_arrays(n_rows, anchor_groups)
            column_anchor_sources[column][family] = anchor_sources_by_row.copy()

            hard_clean_groups = [record for record in anchor_groups if record.anchor_state == ANCHOR_HARD_CLEAN]
            hard_dirty_groups = [record for record in anchor_groups if record.anchor_state == ANCHOR_HARD_DIRTY]
            weighted_clean_rows = int(sum(record.weight for record in hard_clean_groups))
            weighted_dirty_rows = int(sum(record.weight for record in hard_dirty_groups))
            source_counts = {}
            for record in anchor_groups:
                source_counts[record.anchor_source] = source_counts.get(record.anchor_source, 0) + record.weight

            logger.info(
                f"    - Anchor summary: groups={len(anchor_groups)}, "
                f"hard_clean_groups={len(hard_clean_groups)} ({weighted_clean_rows} rows), "
                f"hard_dirty_groups={len(hard_dirty_groups)} ({weighted_dirty_rows} rows)"
            )
            logger.info(f"    - Anchor sources: {source_counts}")

            if family_status.status == "abstained":
                logger.info(f"    - Family '{family}' abstained: {family_status.reason}")
                continue

            hard_clean_values = _select_representative_values(anchor_groups)
            logger.debug(f"    - Generating synthetic critique examples for family '{family}'")
            synthetic_critique_examples = dirty_agent.generate_dirty_examples(
                column,
                metadata.get(column, {}),
                hard_clean_values,
                family,
                max_examples=args.synthetic_per_family,
            )

            n_synth = len(synthetic_critique_examples)
            logger.info(f"    - Generated {n_synth} synthetic critique examples for family '{family}'")
            logger.debug(
                f"    - Synthetic critique values (first 5): {[ex['value'] for ex in synthetic_critique_examples[:5]]}"
            )

            synth_vals = [ex["value"] for ex in synthetic_critique_examples]
            synth_z_list: List[List[int]] = []
            for val in synth_vals:
                row_like = {column: val}
                row_series = pd.Series(row_like)
                row_outputs: List[int] = []
                for idx in cols:
                    func = rules[idx]["rule_func"]
                    row_outputs.append(int(_invoke_rule(func, val, row_series)))
                synth_z_list.append(row_outputs)

            synth_z = np.array(synth_z_list, dtype=int) if synth_z_list else np.zeros((0, len(cols)), dtype=int)

            clean_mask = anchor_states_by_row == ANCHOR_HARD_CLEAN
            dirty_mask = anchor_states_by_row == ANCHOR_HARD_DIRTY
            real_dirty_group_count = len(hard_dirty_groups)
            real_dirty_row_count = weighted_dirty_rows
            min_dirty_rows_for_real = max(5, int(np.ceil(0.002 * max(n_rows, 1))))

            clean_pass_real = np.array([
                _weighted_binary_mean(family_z[clean_mask, local_idx], np.ones(int(clean_mask.sum()), dtype=float))
                if clean_mask.any() else float("nan")
                for local_idx in range(len(cols))
            ])
            dirty_pass_real = np.array([
                _weighted_binary_mean(family_z[dirty_mask, local_idx], np.ones(int(dirty_mask.sum()), dtype=float))
                if dirty_mask.any() else float("nan")
                for local_idx in range(len(cols))
            ])
            dirty_pass_synth = np.array([
                float(synth_z[:, local_idx].mean()) if synth_z.shape[0] > 0 else float("nan")
                for local_idx in range(len(cols))
            ])

            improve_margin = 0.02
            max_reflections = 3
            candidates: List[Tuple[int, int, float, float]] = []
            effective_dirty_pass = np.copy(dirty_pass_real)
            use_real_dirty = real_dirty_group_count >= 2 and real_dirty_row_count >= min_dirty_rows_for_real
            if not use_real_dirty:
                for local_idx in range(len(cols)):
                    candidates_for_dirty = [v for v in [dirty_pass_real[local_idx], dirty_pass_synth[local_idx]] if not np.isnan(v)]
                    effective_dirty_pass[local_idx] = max(candidates_for_dirty) if candidates_for_dirty else float("nan")

            for local_idx, rule_idx in enumerate(cols):
                cp = float(clean_pass_real[local_idx]) if not np.isnan(clean_pass_real[local_idx]) else 0.0
                dp = float(effective_dirty_pass[local_idx]) if not np.isnan(effective_dirty_pass[local_idx]) else 1.0
                if cp < args.calib_min_clean_pass or dp > args.calib_max_dirty_pass:
                    candidates.append((local_idx, rule_idx, cp, dp))

            reflections_done = 0
            for local_idx, rule_idx, cp, dp in candidates:
                if reflections_done >= max_reflections:
                    break
                rule = rules[rule_idx]
                rule_name = rule.get("rule_name", "unknown")
                rule_str = rule.get("rule_str", "")

                clean_mis_examples: List[Dict[str, Any]] = []
                for record in sorted(hard_clean_groups, key=lambda item: item.weight, reverse=True):
                    representative_idx = record.row_indices[0]
                    if family_z[representative_idx, local_idx] == 0:
                        clean_mis_examples.append({"value": record.representative_value, "source": "real_hard_clean"})
                    if len(clean_mis_examples) >= 10:
                        break

                dirty_mis_examples: List[Dict[str, Any]] = []
                for record in sorted(hard_dirty_groups, key=lambda item: item.weight, reverse=True):
                    representative_idx = record.row_indices[0]
                    if family_z[representative_idx, local_idx] == 1:
                        dirty_mis_examples.append({"value": record.representative_value, "source": "real_hard_dirty"})
                    if len(dirty_mis_examples) >= 10:
                        break

                if len(dirty_mis_examples) < 10:
                    for synth_idx, example in enumerate(synthetic_critique_examples[:10]):
                        if synth_z.shape[0] > synth_idx and synth_z[synth_idx, local_idx] == 1:
                            dirty_mis_examples.append({
                                "value": example.get("value"),
                                "reason": example.get("reason", ""),
                                "source": "synthetic_dirty",
                            })
                        if len(dirty_mis_examples) >= 10:
                            break

                if not clean_mis_examples and not dirty_mis_examples:
                    continue

                logger.info(
                    f"    - Reflecting rule '{rule_name}' before calibration "
                    f"(clean_pass_real={cp:.3f}, dirty_pass_eval={dp:.3f})"
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

                new_all = np.zeros(n_rows, dtype=int)
                for row_idx_df, (_, row_df) in enumerate(df.iterrows()):
                    value_df = row_df[column]
                    new_all[row_idx_df] = int(_invoke_rule(new_rule_func, value_df, row_df))

                new_synth_col = np.zeros(len(synth_vals), dtype=int)
                for synth_idx, val in enumerate(synth_vals):
                    row_like = {column: val}
                    row_series = pd.Series(row_like)
                    new_synth_col[synth_idx] = int(_invoke_rule(new_rule_func, val, row_series))

                new_cp = float(new_all[clean_mask].mean()) if clean_mask.any() else float("nan")
                new_dp_real = float(new_all[dirty_mask].mean()) if dirty_mask.any() else float("nan")
                new_dp_synth = float(new_synth_col.mean()) if new_synth_col.size > 0 else float("nan")
                old_dirty = dp
                new_dirty = new_dp_real if use_real_dirty and not np.isnan(new_dp_real) else max(
                    [v for v in [new_dp_real, new_dp_synth] if not np.isnan(v)] or [1.0]
                )
                old_loss = (1.0 - cp) + old_dirty
                new_loss = (1.0 - (new_cp if not np.isnan(new_cp) else 0.0)) + new_dirty
                if new_loss <= old_loss - improve_margin and (np.isnan(new_cp) or new_cp >= args.calib_min_clean_pass):
                    family_z[:, local_idx] = new_all
                    if synth_z.shape[0] > 0:
                        synth_z[:, local_idx] = new_synth_col
                    rules[rule_idx]["rule_str"] = new_rule_str
                    rules[rule_idx]["rule_func"] = new_rule_func
                    clean_pass_real[local_idx] = new_cp
                    dirty_pass_real[local_idx] = new_dp_real
                    dirty_pass_synth[local_idx] = new_dp_synth
                    if use_real_dirty:
                        effective_dirty_pass[local_idx] = new_dp_real
                    else:
                        candidates_for_dirty = [v for v in [new_dp_real, new_dp_synth] if not np.isnan(v)]
                        effective_dirty_pass[local_idx] = max(candidates_for_dirty) if candidates_for_dirty else float("nan")
                    reflections_done += 1
                    logger.info(
                        f"    - Refined rule '{rule_name}': "
                        f"clean_pass_real {cp:.3f}->{clean_pass_real[local_idx]:.3f}, "
                        f"dirty_pass_eval {dp:.3f}->{effective_dirty_pass[local_idx]:.3f}"
                    )

            keep_mask = np.zeros(len(cols), dtype=bool)
            for local_idx, rule_idx in enumerate(cols):
                cp = float(clean_pass_real[local_idx]) if not np.isnan(clean_pass_real[local_idx]) else 0.0
                dp_real = float(dirty_pass_real[local_idx]) if not np.isnan(dirty_pass_real[local_idx]) else float("nan")
                dp_synth = float(dirty_pass_synth[local_idx]) if not np.isnan(dirty_pass_synth[local_idx]) else float("nan")
                if use_real_dirty:
                    dirty_eval = dp_real
                    dirty_evidence = "real"
                else:
                    dirty_candidates = [v for v in [dp_real, dp_synth] if not np.isnan(v)]
                    dirty_eval = max(dirty_candidates) if dirty_candidates else 1.0
                    dirty_evidence = "synthetic_assist"
                keep_mask[local_idx] = cp >= args.calib_min_clean_pass and dirty_eval <= args.calib_max_dirty_pass
                decision = "KEEP" if keep_mask[local_idx] else "DROP"
                logger.info(
                    f"    - Calib rule '{rules[rule_idx].get('rule_name', 'unknown')}': "
                    f"clean_pass_real={cp:.3f}, dirty_pass_real={dp_real if not np.isnan(dp_real) else float('nan'):.3f}, "
                    f"dirty_pass_synth={dp_synth if not np.isnan(dp_synth) else float('nan'):.3f}, "
                    f"dirty_evidence={dirty_evidence}, decision={decision}"
                )

            if not keep_mask.any():
                logger.info(f"    - All rules rejected by calibration for family '{family}', abstained")
                column_statuses[column][family] = FamilyRunStatus(status="abstained", reason="all_rules_rejected")
                column_anchor_sources[column][family][:] = "abstained:all_rules_rejected"
                continue

            kept_local_indices = np.where(keep_mask)[0]
            kept_cols = [cols[i] for i in kept_local_indices]
            family_z = family_z[:, kept_local_indices]
            logger.info(f"    - {keep_mask.sum()}/{len(keep_mask)} rules kept after calibration")

            min_clean_rows = max(20, int(np.ceil(0.01 * max(n_rows, 1))))
            min_dirty_rows = max(5, int(np.ceil(0.002 * max(n_rows, 1))))
            if len(hard_clean_groups) < 3 or len(hard_dirty_groups) < 2 or weighted_clean_rows < min_clean_rows or weighted_dirty_rows < min_dirty_rows:
                reason = "insufficient_real_anchors"
                logger.info(f"    - Family '{family}' abstained: {reason}")
                column_statuses[column][family] = FamilyRunStatus(status="abstained", reason=reason)
                column_anchor_sources[column][family][:] = anchor_sources_by_row.copy()
                continue

            em_groups = _build_em_groups(anchor_groups, family, family_z, df, column)
            if not em_groups:
                reason = "empty_em_groups"
                logger.info(f"    - Family '{family}' abstained: {reason}")
                column_statuses[column][family] = FamilyRunStatus(status="abstained", reason=reason)
                continue

            group_z = np.vstack([record.rule_outputs_by_family[family] for record in em_groups])
            group_weights = np.array([record.weight for record in em_groups], dtype=float)
            group_anchor_states = np.array([record.anchor_state for record in em_groups], dtype=object)

            logger.debug(f"    - Running weighted real-only EM (max_iters={args.em_max_iters})")
            gamma, alpha, beta, pi1 = _run_weighted_em_for_family(
                group_z,
                group_weights,
                group_anchor_states,
                args.em_max_iters,
                args.em_prior_dirty,
            )

            weighted_high_scores = int(sum(record.weight for record, score in zip(em_groups, gamma) if score > 0.5))
            gamma_mean = _weighted_binary_mean(gamma, group_weights)
            gamma_min = float(np.min(gamma)) if gamma.size > 0 else float("nan")
            gamma_max = float(np.max(gamma)) if gamma.size > 0 else float("nan")
            logger.info(
                f"    - EM complete: gamma_mean={gamma_mean:.4f}, pi={pi1:.4f}, range=[{gamma_min:.4f}, {gamma_max:.4f}]"
            )
            logger.info(f"    - Weighted high scores (>0.5): {weighted_high_scores}/{int(group_weights.sum())}")
            logger.debug(f"    - Alpha: {alpha}")
            logger.debug(f"    - Beta: {beta}")

            em_stats[column][family] = {
                "n_rules": len(kept_cols),
                "n_groups": len(em_groups),
                "weighted_clean_rows": weighted_clean_rows,
                "weighted_dirty_rows": weighted_dirty_rows,
                "gamma_mean": gamma_mean,
                "pi": pi1,
                "weighted_high_scores": weighted_high_scores,
            }
            column_statuses[column][family] = FamilyRunStatus(status="ran")

            family_key = FAMILY_SCORE_KEYS[family]
            for record, score in zip(em_groups, gamma):
                for row_idx in record.row_indices:
                    column_outputs[column][family_key][row_idx] = float(score)
                    column_anchor_sources[column][family][row_idx] = record.anchor_source
    
    logger.info("EM calibration complete for all columns")
    
    results: List[Dict[str, Any]] = []
    for column, family_scores in column_outputs.items():
        statuses = column_statuses[column]
        signature_weights = column_signature_weights[column]
        for row_idx in range(n_rows):
            s_missing = family_scores["S_missing"][row_idx]
            s_outlier = family_scores["S_outlier"][row_idx]
            s_pattern = family_scores["S_pattern"][row_idx]
            valid_scores = [score for score in [s_missing, s_outlier, s_pattern] if not pd.isna(score)]
            s_total = float(np.mean(valid_scores)) if valid_scores else float("nan")
            anchor_source_summary = ";".join(
                f"{family}={column_anchor_sources[column][family][row_idx]}"
                for family in FAMILIES
            )
            results.append(
                {
                    "row_index": row_idx,
                    "column": column,
                    "value": df.iloc[row_idx][column],
                    "group_weight": int(signature_weights[row_idx]),
                    "anchor_source_summary": anchor_source_summary,
                    "missing_status": statuses["missing"].status,
                    "missing_reason": statuses["missing"].reason,
                    "outlier_status": statuses["outlier"].status,
                    "outlier_reason": statuses["outlier"].reason,
                    "pattern_status": statuses["pattern"].status,
                    "pattern_reason": statuses["pattern"].reason,
                    "S_missing": s_missing,
                    "S_outlier": s_outlier,
                    "S_pattern": s_pattern,
                    "S_total": s_total,
                }
            )

    results.sort(
        key=lambda item: (
            pd.notna(item["S_total"]),
            float(item["S_total"]) if pd.notna(item["S_total"]) else -1.0,
        ),
        reverse=True,
    )

    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total cells scored: {len(results)}")

    if results:
        all_totals = [r["S_total"] for r in results if pd.notna(r["S_total"])]
        logger.info(f"S_total distribution:")
        if all_totals:
            logger.info(f"  - Mean: {sum(all_totals)/len(all_totals):.4f}")
            logger.info(f"  - Min: {min(all_totals):.4f}")
            logger.info(f"  - Max: {max(all_totals):.4f}")
            above_threshold = [s for s in all_totals if s >= args.score_threshold]
            logger.info(
                f"  - Above threshold ({args.score_threshold}): {len(above_threshold)}/{len(all_totals)} "
                f"({len(above_threshold)/len(all_totals)*100:.1f}%)"
            )
        else:
            logger.info("  - No active family scores available")

        for family_key in ["S_missing", "S_outlier", "S_pattern"]:
            scores = [r[family_key] for r in results if pd.notna(r[family_key])]
            above = sum(1 for s in scores if s >= args.score_threshold)
            if scores:
                logger.info(
                    f"  - {family_key}: mean={sum(scores)/len(scores):.4f}, "
                    f"above_threshold={above}/{len(scores)}"
                )
            else:
                logger.info(f"  - {family_key}: no active scores")

    top_k = min(args.top_k, len(results))
    logger.info(f"\nTop {top_k} highest scoring cells by S_total:")
    print(f"\nTop {top_k} highest scoring cells by S_total:")

    for item in results[:top_k]:
        log_msg = (f"Row {item['row_index']}, Column '{item['column']}': "
                  f"value={item['value']!r}, "
                  f"group_weight={item['group_weight']}, "
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
                if pd.notna(r["S_total"]) and r["S_total"] >= args.score_threshold
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
