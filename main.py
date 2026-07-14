"""
Command-line entry point for the agentic error detector.

Supports a dual verification pipeline with clean rule-level refinement.
"""
import argparse
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from judge import Judge
from agent import AgentFactory, DirtyExampleAgent, CleanRuleReflectionAgent
from profiler import PandasProfiler, DEFAULT_MISSING_TOKEN_SET
from validator import DisjointnessValidator
from core.utils import safe_dict
from cleanem_inference import (
    build_cell_evidence_matrix,
    estimate_calibrated_posteriors,
    select_active_queries,
)
from cleanem_models import CellRecord, EvidenceObservation, HardEvidenceLabel, ValueRecord


# CleanEM Logger
cleanem_logger = logging.getLogger("CleanEM")

FAMILIES = ["missing", "outlier", "pattern", "relationship", "prototype_lexical"]
ARCHETYPE_EVIDENCE_GATES = {
    "open_entity": {
        "rarity_high": 0.10,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.30,
    },
    "identifier": {
        "rarity_high": 0.00,
        "prototype_far": 0.00,
        "pattern_mismatch": 1.00,
    },
    "closed_enum": {
        "rarity_high": 1.00,
        "prototype_far": 1.00,
        "pattern_mismatch": 0.80,
    },
    "unit_measure": {
        "rarity_high": 0.20,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.60,
    },
    "numeric_measure": {
        "rarity_high": 0.50,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.80,
    },
    "geo_name": {
        "rarity_high": 0.10,
        "prototype_far": 0.20,
        "pattern_mismatch": 0.30,
    },
    "free_text": {
        "rarity_high": 0.05,
        "prototype_far": 0.10,
        "pattern_mismatch": 0.20,
    },
}
FAMILY_SCORE_KEYS = {
    "missing": "S_missing",
    "outlier": "S_outlier",
    "pattern": "S_pattern",
    "relationship": "S_relationship",
    "prototype_lexical": "S_prototype_lexical",
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
    dirty_prior: float = 0.5
    anchor_weight: float = 0.0
    support_count: int = 0


@dataclass
class FamilyRunStatus:
    status: str
    reason: str = ""
    coverage_rows: int = 0
    coverage_groups: int = 0
    reliability: float = float("nan")
    mode: str = "abstained"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    score_key: str
    uses_rules: bool = True
    allow_synthetic: bool = True


FAMILY_REGISTRY: Dict[str, FamilySpec] = {
    family: FamilySpec(
        name=family,
        score_key=FAMILY_SCORE_KEYS[family],
        uses_rules=family != "prototype_lexical",
        allow_synthetic=family in {"missing", "outlier", "pattern"},
    )
    for family in FAMILIES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Error Detector CLI with dual and clean+EM modes."
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
        default="http://127.0.0.1:8000/v1",
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
        help="Legacy/unused in clean_em mode after evidence-fusion rewrite."
    )
    parser.add_argument(
        "--em_prior_dirty",
        type=float,
        default=0.05,
        help="Base prior dirty rate used for value-prior initialization."
    )
    parser.add_argument(
        "--calib_min_clean_pass",
        type=float,
        default=0.8,
        help="Legacy/unused in clean_em mode after evidence-fusion rewrite."
    )
    parser.add_argument(
        "--calib_max_dirty_pass",
        type=float,
        default=0.3,
        help="Legacy/unused in clean_em mode after evidence-fusion rewrite."
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
        help="Threshold on cell_posterior to treat a cell as error in clean_em mode."
    )
    parser.add_argument(
        "--active_label_budget",
        type=int,
        default=10,
        help="Max oracle cell labels used to calibrate clean_em evidence sources."
    )
    parser.add_argument(
        "--disable_evidence_gating",
        action="store_true",
        help="Disable archetype-conditioned evidence scaling for ablation runs."
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
        "column_relationship": "relationship",
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
    text = str(value).strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.startswith("$"):
        text = text[1:].strip()
    text = text.replace(",", "")
    if not text:
        return None
    if negative:
        text = f"-{text}"
    try:
        numeric = float(text)
    except Exception:
        return None
    return None if np.isnan(numeric) else numeric


def _matching_regex_rate(value: Any, regex_candidates: List[Dict[str, Any]]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    best_rate = 0.0
    for candidate in regex_candidates or []:
        pattern = str(candidate.get("pattern", ""))
        if not pattern:
            continue
        try:
            if re.fullmatch(pattern, text):
                best_rate = max(best_rate, float(candidate.get("match_rate", 0.0)))
        except re.error:
            continue
    return float(best_rate)


def _best_regex_contract_rate(metadata: Dict[str, Any]) -> float:
    regex_candidates = metadata.get("regex_candidates") or []
    if not regex_candidates:
        return 0.0
    return float(max(float(candidate.get("match_rate", 0.0)) for candidate in regex_candidates))


def _has_trusted_numeric_contract(metadata: Dict[str, Any]) -> bool:
    if metadata.get("type") != "numeric":
        return False
    numeric_count = int(metadata.get("numeric_count") or 0)
    non_numeric_count = int(metadata.get("non_numeric_count") or 0)
    total = numeric_count + non_numeric_count
    numeric_ratio = float(numeric_count / total) if total > 0 else 0.0
    regex_rate = _best_regex_contract_rate(metadata)
    return bool(numeric_ratio >= 0.80 or regex_rate >= 0.60)


def _coerce_numeric_with_metadata(value: Any, metadata: Dict[str, Any]) -> Optional[float]:
    numeric = _coerce_numeric(value)
    if numeric is not None:
        return numeric

    regex_rate = _matching_regex_rate(value, metadata.get("regex_candidates") or [])
    if regex_rate <= 0.0:
        return None

    text = str(value).strip().replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        numeric = float(match.group(1))
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


def _clip_probability(value: float, lower: float = 1e-3, upper: float = 1 - 1e-3) -> float:
    return float(min(max(value, lower), upper))


def _format_float(value: float) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"


def _set_group_anchor(
    record: GroupRecord,
    state: str,
    source: str,
    dirty_prior: float,
    anchor_weight: float,
) -> None:
    record.anchor_state = state
    record.anchor_source = source
    record.dirty_prior = _clip_probability(dirty_prior)
    record.anchor_weight = float(np.clip(anchor_weight, 0.0, 1.0))
    record.support_count = record.weight


def _build_value_groups(df: pd.DataFrame, column: str) -> List[GroupRecord]:
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
    return list(groups_by_signature.values())


def _segment_signature(value: str) -> str:
    if value == "":
        return "<empty>"
    if value.isspace():
        return f"S{len(value)}"
    if all(char.isdigit() for char in value):
        return f"D{len(value)}"
    if all(char.isalpha() for char in value):
        return f"A{len(value)}"
    if all(char.isalnum() for char in value):
        return f"M{len(value)}"
    return f"X{len(value)}"


def _grammar_signature(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "<empty>"
    parts = re.split(r'([\-_/.:\s]+)', text)
    signature_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r'[\-_/.:\s]+', part):
            delimiter = part[0]
            if delimiter.isspace():
                delimiter = ' '
            signature_parts.append(f"<{delimiter}:{len(part)}>" )
        else:
            signature_parts.append(_segment_signature(part))
    return "|".join(signature_parts) if signature_parts else "<empty>"


def _relationship_predicate(value: Any, other_value: Any, constraint_type: str) -> bool:
    if value is None or other_value is None:
        return True
    value_str = str(value).strip()
    other_str = str(other_value).strip()
    if not value_str or not other_str:
        return True
    constraint_type = (constraint_type or "").lower()
    if constraint_type == "prefix_match":
        return value_str.lower().startswith(other_str.lower())
    if constraint_type == "contains":
        return other_str.lower() in value_str.lower()
    if constraint_type == "stateavg_format":
        return value_str.lower().startswith(f"{other_str.lower()}_")
    if constraint_type == "zip_prefix":
        return value_str[:3] == other_str[:3]
    return True


def _build_soft_anchor_groups(
    df: pd.DataFrame,
    column: str,
    metadata: Dict[str, Any],
    family: str,
    prior_dirty: float,
) -> Tuple[List[GroupRecord], FamilyRunStatus]:
    groups = _build_value_groups(df, column)
    if not groups:
        return [], FamilyRunStatus(status="abstained", reason="empty_column")

    status = FamilyRunStatus(status="ran", mode="ran")

    if family == "missing":
        for record in groups:
            if _is_missing_like(record.representative_value, metadata):
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, "real_missing_token", 0.99, 1.0)
            else:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "real_non_missing", 0.01, 1.0)
        return groups, status

    if family == "outlier":
        if metadata.get("type") != "numeric":
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "categorical_outlier_disabled", prior_dirty, 0.0)
            return groups, FamilyRunStatus(status="abstained", reason="categorical_outlier_disabled")

        quantiles = metadata.get("quantiles") or {}
        p25 = quantiles.get("p25")
        p75 = quantiles.get("p75")
        p05 = quantiles.get("p05")
        p95 = quantiles.get("p95")
        if p25 is None or p75 is None:
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "missing_numeric_stats", prior_dirty, 0.0)
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
                _set_group_anchor(record, ANCHOR_UNLABELED, "missing_reserved_for_missing_family", prior_dirty, 0.15)
                continue
            numeric = _coerce_numeric(value)
            if numeric is None:
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, "non_numeric_value", 0.99, 1.0)
            elif inner_low <= numeric <= inner_high:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "central_numeric_band", 0.02, 0.95)
            elif numeric < outer_low or numeric > outer_high:
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, "extreme_numeric_value", 0.98, 0.95)
            else:
                offset = 0.5
                if outer_high > inner_high and numeric > inner_high:
                    offset = min(0.85, 0.35 + (numeric - inner_high) / max(outer_high - inner_high, 1e-6))
                elif inner_low > outer_low and numeric < inner_low:
                    offset = min(0.85, 0.35 + (inner_low - numeric) / max(inner_low - outer_low, 1e-6))
                _set_group_anchor(record, ANCHOR_UNLABELED, "numeric_borderline", offset, 0.4)
        return groups, status

    if family == "pattern":
        grammar_variants = metadata.get("grammar_variants") or []
        accepted_variants = [
            str(item.get("signature"))
            for item in grammar_variants
            if float(item.get("ratio", 0.0)) >= 0.05
        ]
        grammar_coverage = float(sum(
            float(item.get("ratio", 0.0))
            for item in grammar_variants
            if float(item.get("ratio", 0.0)) >= 0.05
        ))
        shape_distribution = metadata.get("shape_distribution") or []
        accepted_shapes = {
            str(item.get("shape"))
            for item in shape_distribution
            if float(item.get("ratio", 0.0)) >= 0.05
        }
        regex_candidates = [
            str(item.get("pattern"))
            for item in metadata.get("regex_candidates") or []
            if float(item.get("match_rate", 0.0)) >= 0.35
        ]

        if not accepted_variants and not accepted_shapes and not regex_candidates:
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "weak_structure_evidence", prior_dirty, 0.0)
            return groups, FamilyRunStatus(status="abstained", reason="weak_structure_evidence")

        if grammar_coverage < 0.35 and len(regex_candidates) == 0:
            status.reason = "weak_structure_evidence"
            status.mode = "provisional"

        for record in groups:
            value = record.representative_value
            if _is_missing_like(value, metadata):
                _set_group_anchor(record, ANCHOR_UNLABELED, "missing_reserved_for_missing_family", prior_dirty, 0.15)
                continue

            text = str(value).strip()
            grammar_sig = _grammar_signature(text)
            grammar_match = grammar_sig in accepted_variants if accepted_variants else False
            shape_match = _shape_signature(text) in accepted_shapes if accepted_shapes else False
            regex_match = False
            for pattern in regex_candidates:
                try:
                    if re.fullmatch(pattern, text):
                        regex_match = True
                        break
                except re.error:
                    continue

            if grammar_match:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "stable_grammar", 0.02, 0.95)
            elif regex_match:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "regex_consistent", 0.08, 0.75)
            elif shape_match:
                _set_group_anchor(record, ANCHOR_UNLABELED, "coarse_shape_match", 0.35, 0.35)
            else:
                source = "grammar_violation" if accepted_variants or regex_candidates else "structure_violation"
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, source, 0.95, 0.85)
        return groups, status

    if family == "relationship":
        profiles = metadata.get("relationship_profiles") or []
        strong_profiles = [
            profile for profile in profiles
            if float(profile.get("violation_rate", 1.0)) <= 0.25
            and int(profile.get("applicable_count", 0)) >= max(5, int(0.02 * max(len(df), 1)))
        ]
        if not strong_profiles:
            for record in groups:
                _set_group_anchor(record, ANCHOR_ABSTAIN, "weak_relationship_evidence", prior_dirty, 0.0)
            return groups, FamilyRunStatus(status="abstained", reason="weak_relationship_evidence")

        active_groups = 0
        for record in groups:
            applicable = 0
            valid = 0
            for row_idx in record.row_indices:
                row = df.iloc[row_idx]
                for profile in strong_profiles:
                    other_column = profile.get("other_column")
                    if not other_column or other_column not in df.columns:
                        continue
                    current_value = row[column]
                    other_value = row[other_column]
                    if pd.isna(current_value) or pd.isna(other_value):
                        continue
                    if str(current_value).strip() == "" or str(other_value).strip() == "":
                        continue
                    applicable += 1
                    valid += int(_relationship_predicate(current_value, other_value, profile.get("type", "")))
            if applicable == 0:
                _set_group_anchor(record, ANCHOR_UNLABELED, "no_applicable_relationship", prior_dirty, 0.0)
                continue
            active_groups += 1
            satisfaction = valid / max(applicable, 1)
            weight = min(1.0, applicable / max(len(strong_profiles), 1))
            if satisfaction >= 0.95:
                _set_group_anchor(record, ANCHOR_HARD_CLEAN, "relationship_consistent", 0.02, weight)
            elif satisfaction <= 0.2:
                _set_group_anchor(record, ANCHOR_HARD_DIRTY, "relationship_violation", 0.98, weight)
            else:
                _set_group_anchor(record, ANCHOR_UNLABELED, "relationship_mixed", 1.0 - satisfaction, min(0.6, weight))

        if active_groups == 0:
            return groups, FamilyRunStatus(status="abstained", reason="no_applicable_relationship")
        if active_groups < max(3, int(0.1 * len(groups))):
            return groups, FamilyRunStatus(status="ran", reason="sparse_relationship_support", mode="provisional")
        return groups, status

    for record in groups:
        _set_group_anchor(record, ANCHOR_ABSTAIN, "unsupported_family", prior_dirty, 0.0)
    return groups, FamilyRunStatus(status="abstained", reason="unsupported_family")


def _build_anchor_arrays_v2(
    n_rows: int,
    anchor_groups: List[GroupRecord],
    prior_dirty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_states = np.full(n_rows, ANCHOR_UNLABELED, dtype=object)
    anchor_sources = np.full(n_rows, "unlabeled", dtype=object)
    anchor_priors = np.full(n_rows, float(prior_dirty), dtype=float)
    anchor_weights = np.zeros(n_rows, dtype=float)
    for record in anchor_groups:
        for row_idx in record.row_indices:
            anchor_states[row_idx] = record.anchor_state
            anchor_sources[row_idx] = record.anchor_source
            anchor_priors[row_idx] = record.dirty_prior
            anchor_weights[row_idx] = record.anchor_weight
    return anchor_states, anchor_sources, anchor_priors, anchor_weights


def _build_em_groups_v2(
    anchor_groups: List[GroupRecord],
    family: str,
    family_outputs: np.ndarray,
    df: pd.DataFrame,
    column: str,
) -> List[GroupRecord]:
    em_groups: Dict[Tuple[str, Tuple[int, ...], str, str, float, float], GroupRecord] = {}
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
                round(float(anchor_group.dirty_prior), 3),
                round(float(anchor_group.anchor_weight), 3),
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
                    dirty_prior=anchor_group.dirty_prior,
                    anchor_weight=anchor_group.anchor_weight,
                    support_count=anchor_group.support_count,
                )
            em_groups[key].row_indices.append(int(row_idx))
            em_groups[key].weight += 1
    return list(em_groups.values())


def _select_representative_values_v2(anchor_groups: List[GroupRecord], limit: int = 20) -> List[Any]:
    clean_groups = [
        record for record in anchor_groups
        if record.anchor_weight >= 0.5 and record.dirty_prior <= 0.25
    ]
    clean_groups.sort(key=lambda record: record.weight * max(record.anchor_weight, 1e-3), reverse=True)
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


def _weighted_pass_rate(outputs: np.ndarray, weights: np.ndarray) -> float:
    if outputs.size == 0 or weights.size == 0:
        return float("nan")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float("nan")
    return float(np.dot(outputs.astype(float), weights.astype(float)) / total_weight)


def _compute_family_reliability(
    rule_quality: float,
    coverage_rows: int,
    coverage_groups: int,
    total_rows: int,
    mode: str,
) -> float:
    coverage_ratio = coverage_rows / max(total_rows, 1)
    group_ratio = min(1.0, coverage_groups / max(3.0, np.sqrt(max(total_rows, 1))))
    reliability = 0.45 * max(rule_quality, 0.0) + 0.35 * coverage_ratio + 0.20 * group_ratio
    if mode == "provisional":
        reliability *= 0.65
    return float(np.clip(reliability, 0.05, 0.99))


def _compute_family_weight(status: FamilyRunStatus, total_rows: int) -> float:
    if status.status == "abstained" or pd.isna(status.reliability):
        return 0.0
    coverage_confidence = min(1.0, status.coverage_rows / max(1, int(0.1 * max(total_rows, 1))))
    mode_factor = 1.0 if status.mode == "ran" else 0.6
    return float(np.clip(status.reliability * max(0.25, coverage_confidence) * mode_factor, 0.0, 1.0))


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current
    return previous[-1]


def _char_ngrams(value: str, n: int = 3) -> set:
    text = value.lower()
    if len(text) < n:
        return {text} if text else set()
    return {text[idx: idx + n] for idx in range(len(text) - n + 1)}


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(union))


def _prototype_similarity(left: str, right: str) -> Tuple[float, float, float, int]:
    if not left or not right:
        return 0.0, 0.0, 0.0, max(len(left), len(right))
    edit_distance = _levenshtein_distance(left.lower(), right.lower())
    edit_similarity = 1.0 - (edit_distance / max(len(left), len(right), 1))
    left_ngrams = _char_ngrams(left)
    right_ngrams = _char_ngrams(right)
    ngram_similarity = float(len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)) if left_ngrams and right_ngrams else 0.0
    token_similarity = _token_jaccard(left, right)
    return edit_similarity, ngram_similarity, token_similarity, edit_distance


def _score_prototype_lexical_family(
    df: pd.DataFrame,
    column: str,
    metadata: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, FamilyRunStatus]:
    n_rows = len(df)
    scores = np.full(n_rows, np.nan, dtype=float)
    sources = np.full(n_rows, "prototype_unavailable", dtype=object)

    if metadata.get("type") not in {"categorical", "pattern", "text"}:
        return scores, sources, FamilyRunStatus(status="abstained", reason="prototype_not_applicable")

    groups = _build_value_groups(df, column)
    valid_groups = [group for group in groups if not _is_missing_like(group.representative_value, metadata)]
    if not valid_groups:
        return scores, sources, FamilyRunStatus(status="abstained", reason="empty_column")

    prototype_threshold = max(2, int(np.ceil(0.01 * max(n_rows, 1))))
    prototypes = [
        group for group in valid_groups
        if group.weight >= prototype_threshold or (
            group.weight >= 2 and (group.weight / max(n_rows, 1)) >= 0.02
        )
    ]
    prototypes.sort(key=lambda record: record.weight, reverse=True)
    prototypes = prototypes[:25]
    prototype_coverage = sum(record.weight for record in prototypes) / max(n_rows, 1)

    if len(prototypes) < 2 or prototype_coverage < 0.12:
        return scores, sources, FamilyRunStatus(status="abstained", reason="insufficient_prototype_support")

    for group in valid_groups:
        group_text = str(group.representative_value).strip()
        if group.normalized_signature in {prototype.normalized_signature for prototype in prototypes}:
            group_score = 0.0
            source = "prototype_bank"
        else:
            best_prototype = None
            best_score = 0.0
            best_distance = 0
            for prototype in prototypes:
                if prototype.normalized_signature == group.normalized_signature:
                    continue
                edit_similarity, ngram_similarity, token_similarity, edit_distance = _prototype_similarity(
                    group_text,
                    str(prototype.representative_value).strip(),
                )
                similarity = 0.55 * edit_similarity + 0.25 * ngram_similarity + 0.20 * token_similarity
                if similarity > best_score:
                    best_score = similarity
                    best_prototype = prototype
                    best_distance = edit_distance

            if best_prototype is None or best_score < 0.55:
                group_score = 0.0
                source = "no_close_prototype"
            else:
                support_factor = min(1.0, best_prototype.weight / max(3.0, 0.05 * max(n_rows, 1)))
                rarity_factor = 1.0 - min(0.85, group.weight / max(best_prototype.weight, 1))
                group_score = best_score * support_factor * max(rarity_factor, 0.15)
                if best_distance <= 2 and abs(len(group_text) - len(str(best_prototype.representative_value).strip())) <= 2:
                    group_score += 0.10
                if _shape_signature(group_text) == _shape_signature(str(best_prototype.representative_value).strip()):
                    group_score += 0.05
                group_score = float(np.clip(group_score, 0.0, 1.0))
                source = f"closest_prototype:{best_prototype.normalized_signature}"

        for row_idx in group.row_indices:
            scores[row_idx] = group_score
            sources[row_idx] = source

    coverage_rows = int(np.sum(pd.notna(scores)))
    coverage_groups = len(valid_groups)
    mode = "ran" if prototype_coverage >= 0.25 else "provisional"
    reliability = float(np.clip(0.45 * prototype_coverage + 0.25 * min(1.0, len(prototypes) / 6.0) + 0.30, 0.10, 0.90))
    if mode == "provisional":
        reliability *= 0.7
    return scores, sources, FamilyRunStatus(
        status="ran",
        reason="",
        coverage_rows=coverage_rows,
        coverage_groups=coverage_groups,
        reliability=reliability,
        mode=mode,
    )


def _run_weighted_em_for_family_v2(
    group_z: np.ndarray,
    group_weights: np.ndarray,
    dirty_priors: np.ndarray,
    anchor_weights: np.ndarray,
    max_iters: int,
    prior_dirty: float,
    rule_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_groups = group_z.shape[0]
    n_rules = group_z.shape[1] if group_z.ndim == 2 else 0
    eps = 1e-3
    if n_groups == 0 or n_rules == 0:
        return np.zeros(n_groups), np.zeros(n_rules), np.zeros(n_rules), float("nan")

    dirty_priors = np.clip(dirty_priors.astype(float), eps, 1 - eps)
    anchor_weights = np.clip(anchor_weights.astype(float), 0.0, 1.0)
    if rule_weights is None or rule_weights.size == 0:
        rule_weights = np.ones(n_rules, dtype=float)
    else:
        rule_weights = np.clip(rule_weights.astype(float), 0.1, 1.0)

    pi1 = _clip_probability(float(prior_dirty))
    gamma = anchor_weights * dirty_priors + (1.0 - anchor_weights) * pi1
    gamma = np.clip(gamma, eps, 1 - eps)

    alpha = np.full(n_rules, 0.9, dtype=float)
    beta = np.full(n_rules, 0.3, dtype=float)

    for _ in range(max_iters):
        clean_weights = group_weights * (1.0 - gamma)
        dirty_weights = group_weights * gamma
        clean_total = float(clean_weights.sum())
        dirty_total = float(dirty_weights.sum())
        if clean_total <= 0 or dirty_total <= 0:
            break

        alpha = np.clip(clean_weights.T.dot(group_z).ravel() / clean_total, eps, 1 - eps)
        beta = np.clip(dirty_weights.T.dot(group_z).ravel() / dirty_total, eps, 1 - eps)

        log_p0 = np.full(n_groups, np.log(1.0 - pi1), dtype=float)
        log_p1 = np.full(n_groups, np.log(pi1), dtype=float)
        for rule_idx in range(n_rules):
            zr = group_z[:, rule_idx]
            weight = rule_weights[rule_idx]
            log_p0 += weight * (zr * np.log(alpha[rule_idx] + eps) + (1 - zr) * np.log(1.0 - alpha[rule_idx] + eps))
            log_p1 += weight * (zr * np.log(beta[rule_idx] + eps) + (1 - zr) * np.log(1.0 - beta[rule_idx] + eps))

        max_log = np.maximum(log_p0, log_p1)
        p0 = np.exp(log_p0 - max_log)
        p1 = np.exp(log_p1 - max_log)
        posterior = p1 / (p0 + p1 + eps)
        gamma = anchor_weights * dirty_priors + (1.0 - anchor_weights) * posterior
        gamma = np.clip(gamma, eps, 1 - eps)
        pi1 = _clip_probability(float(np.dot(group_weights, gamma) / max(group_weights.sum(), eps)))

    return gamma, alpha, beta, pi1


def _record_observation(
    observation_map: Dict[str, List[EvidenceObservation]],
    observation: EvidenceObservation,
) -> None:
    target_list = observation_map.setdefault(observation.target_key, [])
    for idx, existing in enumerate(target_list):
        if (
            existing.source_id == observation.source_id
            and existing.polarity == observation.polarity
            and existing.hard == observation.hard
            and existing.reason_code == observation.reason_code
        ):
            if observation.strength > existing.strength:
                target_list[idx] = observation
            return
    target_list.append(observation)


def _column_archetype(column_metadata: Dict[str, Any]) -> str:
    semantics = column_metadata.get("semantics") or {}
    if isinstance(semantics, dict):
        return str(semantics.get("archetype") or "unknown")
    return str(getattr(semantics, "archetype", "unknown"))


def _gate_observation(
    observation: EvidenceObservation,
    column: str,
    metadata: Dict[str, Dict[str, Any]],
) -> Tuple[EvidenceObservation, str]:
    """Apply a fixed semantic gate while retaining full audit provenance."""
    observation.metadata = dict(observation.metadata or {})
    if observation.metadata.get("gate_applied"):
        return observation, "already_gated"

    archetype = _column_archetype(metadata.get(column, {}))
    raw_strength = float(observation.strength)

    if observation.hard:
        scale = 1.0
        gate_reason = "hard_evidence_preserved"
        outcome = "hard_preserved"
    else:
        policy = ARCHETYPE_EVIDENCE_GATES.get(archetype, {})
        scale = float(policy.get(observation.source_id, 1.0))
        gate_reason = (
            "archetype_policy"
            if observation.source_id in policy
            else "source_default"
        )
        if scale <= 0.0:
            outcome = "disabled"
        elif scale < 1.0:
            outcome = "attenuated"
        else:
            outcome = "unchanged"

    observation.metadata.update({
        "gate_applied": True,
        "gate_archetype": archetype,
        "raw_strength": raw_strength,
        "gate_scale": scale,
        "gate_reason": gate_reason,
    })
    observation.strength = float(np.clip(raw_strength * scale, 0.0, 1.0))
    return observation, outcome


def _apply_evidence_gates(
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    metadata: Dict[str, Dict[str, Any]],
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
) -> Dict[str, Any]:
    summary = Counter()
    policy_hits = Counter()

    for value_key, observations in value_observations.items():
        record = value_registry.get(value_key)
        if record is None:
            continue
        archetype = _column_archetype(metadata.get(record.column, {}))
        for observation in observations:
            _, outcome = _gate_observation(observation, record.column, metadata)
            summary[outcome] += 1
            scale = float(observation.metadata.get("gate_scale", 1.0))
            if scale < 1.0:
                policy_hits[(archetype, observation.source_id, scale)] += 1

    for cell_key, observations in cell_observations.items():
        record = cell_registry.get(cell_key)
        if record is None:
            continue
        archetype = _column_archetype(metadata.get(record.column, {}))
        for observation in observations:
            _, outcome = _gate_observation(observation, record.column, metadata)
            summary[outcome] += 1
            scale = float(observation.metadata.get("gate_scale", 1.0))
            if scale < 1.0:
                policy_hits[(archetype, observation.source_id, scale)] += 1

    return {
        "counts": dict(summary),
        "policy_hits": [
            {
                "archetype": archetype,
                "source_id": source_id,
                "scale": scale,
                "count": count,
            }
            for (archetype, source_id, scale), count in sorted(policy_hits.items())
        ],
    }


def _matches_regex_candidates(value: Any, regex_candidates: List[Dict[str, Any]]) -> bool:
    return _matching_regex_rate(value, regex_candidates) > 0.0


def _grammar_distance_score(text: str, metadata: Dict[str, Any]) -> float:
    grammar_variants = metadata.get("grammar_variants") or []
    accepted = {
        str(item.get("signature"))
        for item in grammar_variants
        if float(item.get("ratio", 0.0)) >= 0.05
    }
    if not accepted:
        return 0.0
    signature = _grammar_signature(text)
    return 0.0 if signature in accepted else 1.0


def _shape_distance_score(text: str, metadata: Dict[str, Any]) -> float:
    shape_distribution = metadata.get("shape_distribution") or []
    accepted = {
        str(item.get("shape"))
        for item in shape_distribution
        if float(item.get("ratio", 0.0)) >= 0.05
    }
    if not accepted:
        return 0.0
    shape = _shape_signature(text)
    return 0.0 if shape in accepted else 1.0


def _row_family_pass_ratio(
    outputs: np.ndarray,
    family_indices: Dict[str, List[int]],
    family: str,
    row_idx: int,
) -> Optional[float]:
    indices = family_indices.get(family, [])
    if outputs.size == 0 or not indices:
        return None
    return float(np.mean(outputs[row_idx, indices]))


def _prototype_distance_evidence(text: str, metadata: Dict[str, Any]) -> Tuple[float, float]:
    prototype_candidates = (metadata.get("prototype_candidates") or {}).get("candidates") or []
    if not text or not prototype_candidates:
        return 0.0, 0.0

    best_similarity = 0.0
    for candidate in prototype_candidates:
        candidate_value = str(candidate.get("value") or candidate.get("example") or "").strip()
        if not candidate_value:
            continue
        edit_similarity, ngram_similarity, token_similarity, _ = _prototype_similarity(text, candidate_value)
        similarity = 0.55 * edit_similarity + 0.25 * ngram_similarity + 0.20 * token_similarity
        best_similarity = max(best_similarity, similarity)

    if best_similarity >= 0.88:
        return min(1.0, 0.4 + 0.6 * best_similarity), 0.0
    if best_similarity <= 0.35:
        return 0.0, min(1.0, 0.5 + (0.35 - best_similarity))
    return 0.0, 0.0


def _get_closed_set_legal_values(metadata: Dict[str, Any]) -> set:
    for key in ("allowed_values", "valid_values", "enum_values"):
        values = metadata.get(key)
        if values and isinstance(values, (list, tuple, set)):
            return {_normalize_signature(value) for value in values}
    return set()


def _numeric_rarity_strength(numeric: float, metadata: Dict[str, Any]) -> float:
    quantiles = metadata.get("quantiles") or {}
    p05 = quantiles.get("p05")
    p95 = quantiles.get("p95")
    p01 = quantiles.get("p01")
    p99 = quantiles.get("p99")
    if p05 is None or p95 is None:
        return 0.0

    if p01 is not None and numeric < float(p01):
        return 1.0
    if p99 is not None and numeric > float(p99):
        return 1.0
    if numeric < float(p05):
        denom = max(float(p05) - float(p01 or p05), 1e-6)
        return float(min(1.0, 0.4 + (float(p05) - numeric) / denom))
    if numeric > float(p95):
        denom = max(float(p99 or p95) - float(p95), 1e-6)
        return float(min(1.0, 0.4 + (numeric - float(p95)) / denom))
    return 0.0


def _low_frequency_strength(value_record: ValueRecord, metadata: Dict[str, Any]) -> float:
    normalized = value_record.normalized_value
    low_frequency_values = {
        _normalize_signature(item.get("value"))
        for item in metadata.get("low_frequency_values") or []
        if item.get("value") is not None
    }
    if normalized in low_frequency_values:
        return 0.85
    if value_record.support_count <= 1:
        return 0.80
    if value_record.support_count == 2:
        return 0.65
    if value_record.support_count == 3:
        return 0.50
    return 0.0


def _entropy_from_counts(counts: Counter) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = float(count) / total
        if probability > 0:
            entropy -= probability * np.log(probability)
    return float(entropy)


def _context_predictiveness(
    target_signatures: List[str],
    context_signatures: List[str],
) -> float:
    target_counts = Counter(target_signatures)
    target_entropy = _entropy_from_counts(target_counts)
    if target_entropy <= 1e-12:
        return 0.0

    grouped_counts: Dict[str, Counter] = defaultdict(Counter)
    for target_sig, context_sig in zip(target_signatures, context_signatures):
        if context_sig in {"<na>", "<empty>"}:
            continue
        grouped_counts[context_sig][target_sig] += 1

    total = float(len(target_signatures))
    conditional_entropy = 0.0
    covered = 0.0
    for counts in grouped_counts.values():
        group_total = float(sum(counts.values()))
        if group_total <= 0:
            continue
        covered += group_total
        conditional_entropy += (group_total / total) * _entropy_from_counts(counts)

    if covered <= 0:
        return 0.0
    return float(np.clip(1.0 - conditional_entropy / target_entropy, 0.0, 1.0))


def _add_contextual_consensus_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> None:
    """Add generic row-context evidence learned from conditional concentration.

    A context column is useful only to the extent that it reduces uncertainty
    about the target column across the table. This keeps the signal generic:
    it can represent entity consensus, functional dependencies, or repeated
    source agreement without hard-coding any dataset-specific relationship.
    """

    n_rows = len(df)
    if n_rows <= 1 or len(df.columns) <= 1:
        return

    signatures_by_column = {
        column: [_normalize_signature(value) for value in df[column].tolist()]
        for column in df.columns
    }

    for target_column in df.columns:
        target_signatures = signatures_by_column[target_column]
        context_profiles: List[Tuple[str, float, Dict[str, Counter]]] = []

        for context_column in df.columns:
            if context_column == target_column:
                continue
            context_signatures = signatures_by_column[context_column]
            predictiveness = _context_predictiveness(target_signatures, context_signatures)
            if predictiveness <= 0.0:
                continue

            grouped_counts: Dict[str, Counter] = defaultdict(Counter)
            for target_sig, context_sig in zip(target_signatures, context_signatures):
                if context_sig in {"<na>", "<empty>"}:
                    continue
                grouped_counts[context_sig][target_sig] += 1

            context_profiles.append((context_column, predictiveness, grouped_counts))

        if not context_profiles:
            continue

        col_meta = metadata.get(target_column, {})
        for row_idx, target_sig in enumerate(target_signatures):
            value = df.iloc[row_idx][target_column]
            if _is_missing_like(value, col_meta):
                continue

            best_disagreement = 0.0
            best_agreement = 0.0
            best_disagreement_context = ""
            best_agreement_context = ""

            for context_column, predictiveness, grouped_counts in context_profiles:
                context_sig = signatures_by_column[context_column][row_idx]
                counts = grouped_counts.get(context_sig)
                if not counts:
                    continue
                group_total = float(sum(counts.values()))
                if group_total <= 1:
                    continue

                support = float(counts.get(target_sig, 0))
                conditional_probability = support / group_total
                group_information = np.log1p(group_total - 1.0) / np.log1p(max(float(n_rows), 2.0))
                strength_base = float(np.clip(predictiveness * group_information, 0.0, 1.0))
                disagreement = strength_base * (1.0 - conditional_probability)
                agreement = strength_base * conditional_probability

                if disagreement > best_disagreement:
                    best_disagreement = disagreement
                    best_disagreement_context = context_column
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_agreement_context = context_column

            cell_key = f"{row_idx}::{target_column}"
            if best_disagreement > 0.0:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="contextual_disagreement",
                    family="contextual",
                    polarity="dirty",
                    strength=best_disagreement,
                    hard=False,
                    reason_code="conditional_value_disagreement",
                    metadata={"context_column": best_disagreement_context},
                ))
            if best_agreement > 0.0:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="contextual_agreement",
                    family="contextual",
                    polarity="clean",
                    strength=best_agreement,
                    hard=False,
                    reason_code="conditional_value_agreement",
                    metadata={"context_column": best_agreement_context},
                ))


def _build_value_registry(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, ValueRecord]:
    registry: Dict[str, ValueRecord] = {}
    for column in df.columns:
        groups: Dict[str, ValueRecord] = {}
        for row_idx, value in enumerate(df[column].tolist()):
            normalized = _normalize_signature(value)
            value_key = f"{column}::{normalized}"
            record = groups.get(value_key)
            if record is None:
                record = ValueRecord(
                    key=value_key,
                    column=column,
                    normalized_value=normalized,
                    representative_value=value,
                    row_indices=[],
                    support_count=0,
                )
                groups[value_key] = record
            record.row_indices.append(row_idx)
            record.support_count += 1
        registry.update(groups)
    return registry


def _build_cell_registry(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    value_registry: Dict[str, ValueRecord],
) -> Dict[str, CellRecord]:
    registry: Dict[str, CellRecord] = {}
    for row_idx in range(len(df)):
        for column in df.columns:
            value = df.iloc[row_idx][column]
            normalized = _normalize_signature(value)
            value_key = f"{column}::{normalized}"
            cell_key = f"{row_idx}::{column}"
            registry[cell_key] = CellRecord(
                key=cell_key,
                row_index=row_idx,
                column=column,
                raw_value=value,
                value_key=value_key,
            )
    return registry


def _extract_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    rule_pool: Dict[str, List[Dict[str, Any]]],
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    enable_evidence_gating: bool = True,
) -> Tuple[Dict[str, List[EvidenceObservation]], Dict[str, List[EvidenceObservation]]]:
    value_observations: Dict[str, List[EvidenceObservation]] = {}
    cell_observations: Dict[str, List[EvidenceObservation]] = {}

    outputs_by_column: Dict[str, np.ndarray] = {}
    family_indices_by_column: Dict[str, Dict[str, List[int]]] = {}
    for column in df.columns:
        rules = rule_pool.get(column, [])
        outputs = _compute_rule_outputs_for_column(df, column, rules) if rules else np.zeros((len(df), 0), dtype=int)
        outputs_by_column[column] = outputs
        family_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, rule in enumerate(rules):
            family_indices[rule.get("family", "")].append(idx)
        family_indices_by_column[column] = dict(family_indices)

    for value_key, record in value_registry.items():
        col_meta = metadata.get(record.column, {})
        value = record.representative_value
        normalized = record.normalized_value
        missing_like = _is_missing_like(value, col_meta)
        closed_set_legal_values = _get_closed_set_legal_values(col_meta)
        representative_row_idx = record.row_indices[0]
        regex_match = _matches_regex_candidates(value, col_meta.get("regex_candidates") or [])
        regex_match_rate = _matching_regex_rate(value, col_meta.get("regex_candidates") or [])
        pattern_ratio = _row_family_pass_ratio(
            outputs_by_column.get(record.column, np.zeros((0, 0), dtype=int)),
            family_indices_by_column.get(record.column, {}),
            "pattern",
            representative_row_idx,
        )

        if missing_like:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="missing_token",
                family="missing",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="missing_token",
            ))
            continue

        if closed_set_legal_values and normalized not in closed_set_legal_values:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="closed_set_invalid",
                family="outlier",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="closed_set_invalid",
            ))
            continue

        if col_meta.get("type") == "numeric":
            numeric = _coerce_numeric_with_metadata(value, col_meta)
            hard_numeric_failure = _has_trusted_numeric_contract(col_meta)
            if numeric is None:
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="parse_failure",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_parse_failure",
                ))
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="type_conflict",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_type_conflict",
                ))
                continue

            if regex_match_rate > 0:
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="regex_pass",
                    family="pattern",
                    polarity="clean",
                    strength=min(1.0, max(0.6, regex_match_rate)),
                    hard=False,
                    reason_code="regex_pass",
                ))
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="parser_success",
                family="outlier",
                polarity="clean",
                strength=1.0,
                hard=False,
                reason_code="numeric_parse_success",
            ))
            rarity_strength = _numeric_rarity_strength(numeric, col_meta)
            if rarity_strength > 0:
                _record_observation(value_observations, EvidenceObservation(
                    target_scope="value",
                    target_key=value_key,
                    source_id="rarity_high",
                    family="outlier",
                    polarity="dirty",
                    strength=rarity_strength,
                    hard=False,
                    reason_code="numeric_rarity_high",
                ))
            continue

        if regex_match:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="regex_pass",
                family="pattern",
                polarity="clean",
                strength=1.0,
                hard=False,
                reason_code="regex_pass",
            ))
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="parser_success",
                family="pattern",
                polarity="clean",
                strength=0.8,
                hard=False,
                reason_code="pattern_parser_success",
            ))
        elif pattern_ratio is not None and pattern_ratio >= 0.75:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="parser_success",
                family="pattern",
                polarity="clean",
                strength=pattern_ratio,
                hard=False,
                reason_code="pattern_rule_support",
            ))

        low_frequency_strength = _low_frequency_strength(record, col_meta)
        if low_frequency_strength > 0:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="rarity_high",
                family="pattern",
                polarity="dirty",
                strength=low_frequency_strength,
                hard=False,
                reason_code="low_frequency_value",
            ))

        text = str(value).strip()
        prototype_close, prototype_far = _prototype_distance_evidence(text, col_meta)
        if prototype_close > 0:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="prototype_close",
                family="prototype_lexical",
                polarity="clean",
                strength=prototype_close,
                hard=False,
                reason_code="prototype_close",
            ))
        if prototype_far > 0:
            _record_observation(value_observations, EvidenceObservation(
                target_scope="value",
                target_key=value_key,
                source_id="prototype_far",
                family="prototype_lexical",
                polarity="dirty",
                strength=prototype_far,
                hard=False,
                reason_code="prototype_far",
            ))

    for cell_key, record in cell_registry.items():
        col_meta = metadata.get(record.column, {})
        value = record.raw_value
        normalized = _normalize_signature(value)
        row = df.iloc[record.row_index]
        missing_like = _is_missing_like(value, col_meta)
        closed_set_legal_values = _get_closed_set_legal_values(col_meta)
        regex_match = _matches_regex_candidates(value, col_meta.get("regex_candidates") or [])
        regex_match_rate = _matching_regex_rate(value, col_meta.get("regex_candidates") or [])
        pattern_ratio = _row_family_pass_ratio(
            outputs_by_column.get(record.column, np.zeros((0, 0), dtype=int)),
            family_indices_by_column.get(record.column, {}),
            "pattern",
            record.row_index,
        )

        emitted_hard_dirty = False
        parse_success = False
        principal_parser_success = False
        type_conflict = False
        closed_set_valid = True
        relationship_applicable = 0
        strong_relationship_violation = 0
        weak_relationship_violation = 0
        relationship_satisfied = 0

        if missing_like:
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="missing_token",
                family="missing",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="missing_token",
            ))
            continue

        if closed_set_legal_values:
            closed_set_valid = normalized in closed_set_legal_values
            if not closed_set_valid:
                emitted_hard_dirty = True
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="closed_set_invalid",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=True,
                    reason_code="closed_set_invalid",
                ))

        if col_meta.get("type") == "numeric":
            numeric = _coerce_numeric_with_metadata(value, col_meta)
            hard_numeric_failure = _has_trusted_numeric_contract(col_meta)
            if numeric is None:
                emitted_hard_dirty = hard_numeric_failure
                type_conflict = True
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="cell_parse_failure",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_parse_failure",
                ))
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="cell_type_conflict",
                    family="outlier",
                    polarity="dirty",
                    strength=1.0,
                    hard=hard_numeric_failure,
                    reason_code="numeric_type_conflict",
                ))
            else:
                parse_success = True
                principal_parser_success = True
                if regex_match_rate > 0:
                    _record_observation(cell_observations, EvidenceObservation(
                        target_scope="cell",
                        target_key=cell_key,
                        source_id="pattern_match",
                        family="pattern",
                        polarity="clean",
                        strength=min(1.0, max(0.6, regex_match_rate)),
                        hard=False,
                        reason_code="regex_match",
                    ))
                elif _best_regex_contract_rate(col_meta) >= 0.60:
                    _record_observation(cell_observations, EvidenceObservation(
                        target_scope="cell",
                        target_key=cell_key,
                        source_id="pattern_mismatch",
                        family="pattern",
                        polarity="dirty",
                        strength=0.6,
                        hard=False,
                        reason_code="format_contract_mismatch",
                    ))
        else:
            text = str(value).strip()
            grammar_distance = _grammar_distance_score(text, col_meta)
            shape_distance = _shape_distance_score(text, col_meta)
            mismatch_candidates = [grammar_distance, shape_distance]
            if pattern_ratio is not None:
                mismatch_candidates.append(max(0.0, 1.0 - pattern_ratio))
            mismatch_strength = max(mismatch_candidates) if mismatch_candidates else 0.0
            parse_success = regex_match or (pattern_ratio is not None and pattern_ratio >= 0.75)

            if regex_match:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="pattern_match",
                    family="pattern",
                    polarity="clean",
                    strength=1.0,
                    hard=False,
                    reason_code="regex_match",
                ))
            elif pattern_ratio is not None and pattern_ratio >= 0.75:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="pattern_match",
                    family="pattern",
                    polarity="clean",
                    strength=pattern_ratio,
                    hard=False,
                    reason_code="pattern_rule_match",
                ))

            if not regex_match and mismatch_strength >= 0.55:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="pattern_mismatch",
                    family="pattern",
                    polarity="dirty",
                    strength=mismatch_strength,
                    hard=False,
                    reason_code="pattern_mismatch",
                ))

        for profile in col_meta.get("relationship_profiles") or []:
            other_column = profile.get("other_column")
            if not other_column or other_column not in df.columns:
                continue
            other_value = row[other_column]
            if _is_missing_like(value, col_meta) or _is_missing_like(other_value, metadata.get(other_column, {})):
                continue
            if str(value).strip() == "" or str(other_value).strip() == "":
                continue
            relationship_applicable += 1
            is_valid = _relationship_predicate(value, other_value, profile.get("type", ""))
            if is_valid:
                relationship_satisfied += 1
            else:
                if float(profile.get("violation_rate", 1.0)) <= 0.10:
                    strong_relationship_violation += 1
                else:
                    weak_relationship_violation += 1

        if strong_relationship_violation > 0:
            emitted_hard_dirty = True
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="strong_relationship_violation",
                family="relationship",
                polarity="dirty",
                strength=1.0,
                hard=True,
                reason_code="strong_relationship_violation",
            ))
        elif weak_relationship_violation > 0:
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="weak_relationship_violation",
                family="relationship",
                polarity="dirty",
                strength=min(1.0, 0.35 + weak_relationship_violation / max(relationship_applicable, 1)),
                hard=False,
                reason_code="weak_relationship_violation",
            ))
        elif relationship_applicable > 0:
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="relationship_satisfied",
                family="relationship",
                polarity="clean",
                strength=min(1.0, 0.4 + relationship_satisfied / max(relationship_applicable, 1)),
                hard=False,
                reason_code="relationship_satisfied",
            ))

        if relationship_applicable >= 2:
            if relationship_satisfied == relationship_applicable:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="local_consistency_pass",
                    family="relationship",
                    polarity="clean",
                    strength=0.75,
                    hard=False,
                    reason_code="local_consistency_pass",
                ))
            elif (strong_relationship_violation + weak_relationship_violation) / max(relationship_applicable, 1) >= 0.5:
                _record_observation(cell_observations, EvidenceObservation(
                    target_scope="cell",
                    target_key=cell_key,
                    source_id="local_consistency_fail",
                    family="relationship",
                    polarity="dirty",
                    strength=0.75,
                    hard=False,
                    reason_code="local_consistency_fail",
                ))

        if (
            not emitted_hard_dirty
            and principal_parser_success
            and not type_conflict
            and closed_set_valid
            and (relationship_applicable == 0 or strong_relationship_violation == 0 and weak_relationship_violation == 0)
        ):
            _record_observation(cell_observations, EvidenceObservation(
                target_scope="cell",
                target_key=cell_key,
                source_id="cell_parser_success",
                family="outlier",
                polarity="clean",
                strength=1.0,
                hard=False,
                reason_code="parser_contract_satisfied",
            ))

    _add_contextual_consensus_evidence(df, metadata, cell_observations)

    if enable_evidence_gating:
        gating_summary = _apply_evidence_gates(
            value_observations,
            cell_observations,
            metadata,
            value_registry,
            cell_registry,
        )
        cleanem_logger.info(f"evidence_gating_counts: {gating_summary['counts']}")
        for item in gating_summary["policy_hits"]:
            cleanem_logger.info(
                "  - Evidence gate: "
                f"archetype={item['archetype']}, source={item['source_id']}, "
                f"scale={item['scale']:.2f}, count={item['count']}"
            )
    else:
        cleanem_logger.info("evidence_gating: disabled")

    return value_observations, cell_observations


def _build_hard_evidence(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> Dict[str, Dict[str, HardEvidenceLabel]]:
    hard_labels = {"value": {}, "cell": {}}

    for target_key, observations in value_observations.items():
        dirty_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "dirty"})
        clean_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "clean"})
        if dirty_reasons:
            hard_labels["value"][target_key] = HardEvidenceLabel(
                target_scope="value",
                target_key=target_key,
                label="dirty",
                reasons=dirty_reasons,
            )
        elif clean_reasons:
            hard_labels["value"][target_key] = HardEvidenceLabel(
                target_scope="value",
                target_key=target_key,
                label="clean",
                reasons=clean_reasons,
            )

    for target_key, observations in cell_observations.items():
        dirty_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "dirty"})
        clean_reasons = sorted({obs.reason_code for obs in observations if obs.hard and obs.polarity == "clean"})
        if dirty_reasons:
            hard_labels["cell"][target_key] = HardEvidenceLabel(
                target_scope="cell",
                target_key=target_key,
                label="dirty",
                reasons=dirty_reasons,
            )
        elif clean_reasons:
            hard_labels["cell"][target_key] = HardEvidenceLabel(
                target_scope="cell",
                target_key=target_key,
                label="clean",
                reasons=clean_reasons,
            )

    return hard_labels


def _oracle_label_from_clean(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    cell_record: CellRecord,
) -> Optional[int]:
    if cell_record.row_index >= len(clean_df) or cell_record.column not in clean_df.columns:
        return None
    dirty_val = dirty_df.iloc[cell_record.row_index][cell_record.column]
    clean_val = clean_df.iloc[cell_record.row_index][cell_record.column]
    return int(str(dirty_val).strip() != str(clean_val).strip())


def _query_active_oracle_labels(
    dirty_df: pd.DataFrame,
    clean_df: Optional[pd.DataFrame],
    cell_registry: Dict[str, CellRecord],
    selected_cell_keys: List[str],
) -> Tuple[Dict[str, int], List[Dict[str, object]]]:
    if clean_df is None:
        return {}, []

    labels: Dict[str, int] = {}
    queried_cells: List[Dict[str, object]] = []
    for cell_key in selected_cell_keys:
        record = cell_registry.get(cell_key)
        if record is None:
            continue
        label = _oracle_label_from_clean(dirty_df, clean_df, record)
        if label is None:
            continue
        labels[cell_key] = label
        queried_cells.append(
            {
                "cell_key": cell_key,
                "row_index": int(record.row_index),
                "column": record.column,
                "value": record.raw_value,
                "label": "dirty" if label else "clean",
            }
        )
    return labels, queried_cells


def _summarize_observations(observations: List[EvidenceObservation], limit: int = 6) -> str:
    if not observations:
        return ""
    parts = []
    for obs in sorted(observations, key=lambda item: (item.hard, item.strength), reverse=True)[:limit]:
        hard_marker = "hard" if obs.hard else "soft"
        parts.append(f"{obs.source_id}:{obs.polarity}:{obs.strength:.2f}:{hard_marker}")
    return ";".join(parts)


def _summarize_hard_label(label: Optional[HardEvidenceLabel]) -> str:
    if not label:
        return ""
    return ";".join(label.reasons)


def _evidence_source_coverage(
    observation_map: Dict[str, List[EvidenceObservation]],
) -> Dict[str, int]:
    coverage: Dict[str, set] = defaultdict(set)
    for target_key, observations in observation_map.items():
        for obs in observations:
            coverage[obs.source_id].add(target_key)
    return {source_id: len(targets) for source_id, targets in sorted(coverage.items())}


def _export_cleanem_results(
    df: pd.DataFrame,
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    value_priors: Dict[str, Dict[str, float]],
    cell_posteriors: Dict[str, Dict[str, float]],
    hard_labels: Dict[str, Dict[str, HardEvidenceLabel]],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    active_labels: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    active_labels = active_labels or {}
    for cell_key, cell_record in cell_registry.items():
        value_record = value_registry[cell_record.value_key]
        prior_bundle = value_priors.get(cell_record.value_key, {"prior": 0.5, "confidence": 0.0})
        posterior_bundle = cell_posteriors.get(cell_key, {"posterior": 0.5, "confidence": 0.0})
        cell_label = hard_labels["cell"].get(cell_key)
        active_label = active_labels.get(cell_key)

        results.append({
            "row_index": cell_record.row_index,
            "column": cell_record.column,
            "value": cell_record.raw_value,
            "value_key": cell_record.value_key,
            "value_prior": prior_bundle["prior"],
            "value_prior_confidence": prior_bundle["confidence"],
            "cell_posterior": posterior_bundle["posterior"],
            "cell_confidence": posterior_bundle["confidence"],
            "hard_label": cell_label.label if cell_label else "",
            "hard_reason_summary": _summarize_hard_label(cell_label),
            "queried_for_calibration": cell_key in active_labels,
            "active_oracle_label": "dirty" if active_label == 1 else "clean" if active_label == 0 else "",
            "value_evidence_summary": _summarize_observations(value_observations.get(cell_record.value_key, [])),
            "cell_evidence_summary": _summarize_observations(cell_observations.get(cell_key, [])),
        })

    results.sort(key=lambda item: (item["cell_posterior"], item["value_prior"]), reverse=True)
    return results


def run_clean_em_mode(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dirty_csv))[0]

    logger = _setup_cleanem_logger(args.output_dir, dataset_name)
    logger.info("=" * 60)
    logger.info("CleanEM Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"[1/4] Profiling dirty dataset: {args.dirty_csv}")
    logger.info("Configuration:")
    logger.info(f"  - clean_seed_percent (legacy/unused): {args.clean_seed_percent}")
    logger.info(f"  - synthetic_per_family: {args.synthetic_per_family}")
    logger.info(f"  - em_max_iters (legacy/unused): {args.em_max_iters}")
    logger.info(f"  - em_prior_dirty (base prior): {args.em_prior_dirty}")
    logger.info(f"  - calib_min_clean_pass (legacy/unused): {args.calib_min_clean_pass}")
    logger.info(f"  - calib_max_dirty_pass (legacy/unused): {args.calib_max_dirty_pass}")
    logger.info(f"  - score_threshold (cell_posterior): {args.score_threshold}")
    logger.info(f"  - active_label_budget: {args.active_label_budget}")
    logger.info(f"  - evidence_gating: {not args.disable_evidence_gating}")
    logger.info(f"  - top_k: {args.top_k}")

    profiler = PandasProfiler(args.dirty_csv)
    df = profiler.df
    metadata = profiler.get_metadata()
    n_rows = len(df)

    logger.info("Dataset profile complete:")
    logger.info(f"  - Total rows: {n_rows}")
    logger.info(f"  - Total columns: {len(metadata)}")
    logger.info(f"  - Columns: {list(metadata.keys())}")
    logger.debug("Per-column metadata summary:")
    for col, meta in metadata.items():
        logger.debug(
            f"  - {col}: type={meta.get('type', 'unknown')}, nulls={meta.get('null_count', 0)}, "
            f"unique={meta.get('unique_count', 0)}, families={meta.get('available_families', [])}"
        )

    logger.info("[2/4] Inferring column semantics and generating evidence sources")
    factory = AgentFactory(base_url=args.base_url, model=args.model, max_workers=args.max_workers)
    column_semantics = factory.infer_column_semantics(metadata)
    for column, semantics in column_semantics.items():
        metadata[column]["semantics"] = asdict(semantics)
        logger.info(
            f"  - Column semantics '{column}': archetype={semantics.archetype}, "
            f"confidence={semantics.confidence:.3f}, "
            f"open_set={semantics.open_set_score:.3f}, "
            f"structure={semantics.structure_strength:.3f}, "
            f"canonicalization_need={semantics.canonicalization_need:.3f}, "
            f"mechanisms={semantics.possible_error_mechanisms}"
        )

    rule_pool = _build_clean_rule_pool(df, metadata, factory)

    total_rules = sum(len(rules) for rules in rule_pool.values())
    logger.info("Rule pool built successfully:")
    logger.info(f"  - Total columns with rules: {len(rule_pool)}")
    logger.info(f"  - Total rules: {total_rules}")

    family_counts = {family: 0 for family in FAMILIES if family != "prototype_lexical"}
    for column in metadata.keys():
        rules = rule_pool.get(column, [])
        per_family = {family: 0 for family in family_counts}
        for rule in rules:
            fam = rule.get("family")
            if fam in per_family:
                per_family[fam] += 1
                family_counts[fam] += 1
        logger.info(
            f"  - Column '{column}': {len(rules)} rules "
            f"(missing={per_family.get('missing', 0)}, outlier={per_family.get('outlier', 0)}, "
            f"pattern={per_family.get('pattern', 0)}, relationship={per_family.get('relationship', 0)})"
        )
        for rule in rules:
            logger.info(
                f"    - Rule '{rule.get('rule_name', 'unknown')}' [{rule.get('family', 'unknown')}]: {rule.get('rule_str', '')}"
            )
    logger.info(
        "  - Rule family distribution: "
        + ", ".join(f"{family}={count}" for family, count in family_counts.items())
    )

    logger.info("[3/4] Running active-learning calibrated evidence inference")
    value_registry = _build_value_registry(df, metadata)
    cell_registry = _build_cell_registry(df, metadata, value_registry)
    logger.info(f"Value registry size: {len(value_registry)}")
    logger.info(f"Cell registry size: {len(cell_registry)}")

    value_observations, cell_observations = _extract_evidence(
        df,
        metadata,
        rule_pool,
        value_registry,
        cell_registry,
        enable_evidence_gating=not args.disable_evidence_gating,
    )
    hard_labels = _build_hard_evidence(df, metadata, value_observations, cell_observations)

    evidence_matrix = build_cell_evidence_matrix(
        cell_registry,
        value_observations,
        cell_observations,
    )
    logger.info(
        f"Evidence matrix built: cells={len(evidence_matrix.cell_keys)}, "
        f"features={len(evidence_matrix.feature_names)}"
    )

    clean_df_for_oracle: Optional[pd.DataFrame] = None
    if args.clean_csv and os.path.exists(args.clean_csv):
        clean_df_for_oracle = pd.read_csv(args.clean_csv)
    else:
        logger.info("Active learning oracle unavailable: clean_csv not found")

    selected_cell_keys: List[str] = []
    if clean_df_for_oracle is not None and args.active_label_budget > 0:
        selected_cell_keys = select_active_queries(evidence_matrix, args.active_label_budget)
    active_labels, queried_cells = _query_active_oracle_labels(
        df,
        clean_df_for_oracle,
        cell_registry,
        selected_cell_keys,
    )
    value_priors, cell_posteriors, calibration_trace = estimate_calibrated_posteriors(
        value_registry,
        cell_registry,
        evidence_matrix,
        selected_cell_keys,
        active_labels,
        args.em_prior_dirty,
        queried_cells=queried_cells,
    )
    results = _export_cleanem_results(
        df,
        value_registry,
        cell_registry,
        value_priors,
        cell_posteriors,
        hard_labels,
        value_observations,
        cell_observations,
        active_labels=active_labels,
    )

    for value_key, record in value_registry.items():
        bundle = value_priors.get(value_key, {"prior": 0.5, "confidence": 0.0})
        record.prior = bundle["prior"]
        record.confidence = bundle["confidence"]
    for cell_key, record in cell_registry.items():
        bundle = cell_posteriors.get(cell_key, {"posterior": 0.5, "confidence": 0.0})
        record.posterior = bundle["posterior"]
        record.confidence = bundle["confidence"]

    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"total_cells_exported: {len(results)}")
    logger.info(f"value_records: {len(value_registry)}")
    logger.info(f"cell_records: {len(cell_registry)}")
    logger.info(
        f"hard_evidence_counts: "
        f"value_dirty={sum(1 for label in hard_labels['value'].values() if label.label == 'dirty')}, "
        f"value_clean={sum(1 for label in hard_labels['value'].values() if label.label == 'clean')}, "
        f"cell_dirty={sum(1 for label in hard_labels['cell'].values() if label.label == 'dirty')}, "
        f"cell_clean={sum(1 for label in hard_labels['cell'].values() if label.label == 'clean')}"
    )
    label_counts = Counter(active_labels.values())
    logger.info(
        f"active_calibration: fitted={calibration_trace.fitted}, reason={calibration_trace.reason}, "
        f"queried={len(active_labels)}/{args.active_label_budget}, "
        f"dirty_labels={label_counts.get(1, 0)}, clean_labels={label_counts.get(0, 0)}, "
        f"features={calibration_trace.feature_count}"
    )
    for query in calibration_trace.queried_cells:
        logger.info(
            "  - Active query: "
            f"row={query.get('row_index')}, column={query.get('column')}, "
            f"value={query.get('value')!r}, label={query.get('label')}"
        )
    if calibration_trace.feature_weights:
        top_weights = sorted(
            calibration_trace.feature_weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:20]
        logger.info("learned_evidence_weights:")
        for feature_name, weight in top_weights:
            logger.info(f"  - {feature_name}: {weight:.4f}")
    logger.info(f"value_evidence_source_coverage: {_evidence_source_coverage(value_observations)}")
    logger.info(f"cell_evidence_source_coverage: {_evidence_source_coverage(cell_observations)}")

    value_prior_values = [item["value_prior"] for item in results if pd.notna(item["value_prior"])]
    cell_posterior_values = [item["cell_posterior"] for item in results if pd.notna(item["cell_posterior"])]

    logger.info("value_prior distribution:")
    if value_prior_values:
        logger.info(f"  - Mean: {sum(value_prior_values)/len(value_prior_values):.4f}")
        logger.info(f"  - Min: {min(value_prior_values):.4f}")
        logger.info(f"  - Max: {max(value_prior_values):.4f}")
    else:
        logger.info("  - No value priors")

    logger.info("cell_posterior distribution:")
    if cell_posterior_values:
        logger.info(f"  - Mean: {sum(cell_posterior_values)/len(cell_posterior_values):.4f}")
        logger.info(f"  - Min: {min(cell_posterior_values):.4f}")
        logger.info(f"  - Max: {max(cell_posterior_values):.4f}")
        above_threshold = [score for score in cell_posterior_values if score >= args.score_threshold]
        logger.info(
            f"  - Above threshold ({args.score_threshold}): {len(above_threshold)}/{len(cell_posterior_values)} "
            f"({len(above_threshold)/len(cell_posterior_values)*100:.1f}%)"
        )
    else:
        logger.info("  - No cell posteriors")

    top_k = min(args.top_k, len(results))
    logger.info(f"\nTop {top_k} highest scoring cells by cell_posterior:")
    for item in results[:top_k]:
        log_msg = (
            f"Row {item['row_index']}, Column '{item['column']}': value={item['value']!r}, "
            f"value_prior={item['value_prior']:.4f}, cell_posterior={item['cell_posterior']:.4f}, "
            f"hard_label={item['hard_label'] or 'none'}"
        )
        logger.info(f"  {log_msg}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{timestamp}_{dataset_name}_clean_em_scores.csv")
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"\nSaved all cell scores to: {output_path}")

    if args.clean_csv and os.path.exists(args.clean_csv):
        logger.info("\nEvaluating clean_em scores against ground truth:")
        clean_df = clean_df_for_oracle if clean_df_for_oracle is not None else pd.read_csv(args.clean_csv)
        judge = Judge()
        detected_errors = [
            {"row_index": item["row_index"], "column": item["column"]}
            for item in results
            if pd.notna(item["cell_posterior"]) and item["cell_posterior"] >= args.score_threshold
        ]
        logger.info(f"Detected {len(detected_errors)} cells with cell_posterior >= {args.score_threshold}")
        metrics = judge.evaluate_with_ground_truth(df, clean_df, detected_errors)
        judge.print_evaluation_summary(metrics)
        if hasattr(metrics, "get"):
            logger.info("Evaluation Metrics:")
            for key, value in metrics.items() if isinstance(metrics, dict) else []:
                logger.info(f"  - {key}: {value}")

    logger.info("\n[4/4] clean_em pipeline complete.")
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
