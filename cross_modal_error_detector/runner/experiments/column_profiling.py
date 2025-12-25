"""
Column profiling and corruption utilities for experiments.
"""
from __future__ import annotations

import random
import re
import string
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .data_utils import _safe_str, _try_parse_float
from .types import _ColumnProfile


def _infer_dominant_string_pattern(values: List[str]) -> Optional[str]:
    """
    Very lightweight pattern inference for "col pattern violation" style errors.
    We only try a few common patterns; if one covers most non-empty values, we treat it as dominant.

    Args:
        values: List of string values

    Returns:
        Dominant pattern regex string or None
    """
    candidates = [
        r"^\d{4}-\d{2}-\d{2}$",  # date
        r"^\d{2}/\d{2}/\d{4}$",  # date alt
        r"^[A-Za-z]+$",  # pure letters
        r"^[0-9]+$",  # pure digits
        r"^[A-Za-z0-9\-\_]+$",  # codes
        r"^[A-Za-z0-9\s\-\_]+$",  # relaxed codes
    ]
    non_empty = [v for v in values if v.strip() != ""]
    if len(non_empty) < 30:
        return None
    best_pat = None
    best_cov = 0.0
    for pat in candidates:
        try:
            rgx = re.compile(pat)
        except re.error:
            continue
        cov = sum(1 for v in non_empty if rgx.match(v) is not None) / float(len(non_empty))
        if cov > best_cov:
            best_cov = cov
            best_pat = pat
    if best_cov >= 0.85:
        return best_pat
    return None


def _build_column_profiles(
    clean_rows: List[List[Any]],
    column_names: Optional[List[str]],
) -> List[_ColumnProfile]:
    """
    Build column profiles from clean rows.

    Args:
        clean_rows: List of clean rows
        column_names: Optional list of column names

    Returns:
        List of column profiles
    """
    if not clean_rows:
        return []
    num_cols = len(clean_rows[0])
    profiles: List[_ColumnProfile] = []
    for col_idx in tqdm(range(num_cols), desc="构建列分析", ncols=100):
        name = (
            column_names[col_idx]
            if column_names and col_idx < len(column_names)
            else f"col_{col_idx}"
        )
        raw_vals = [r[col_idx] if col_idx < len(r) else None for r in clean_rows]
        parsed = [p for p in (_try_parse_float(v) for v in raw_vals) if p is not None]
        numeric_ratio = len(parsed) / max(1, len(raw_vals))
        if numeric_ratio >= 0.8:
            arr = np.array(parsed, dtype=np.float64)
            med = float(np.median(arr)) if arr.size else 0.0
            mad = float(np.median(np.abs(arr - med))) if arr.size else 0.0
            # Avoid zero MAD (all equal) which would explode robust z-score
            mad = mad if mad > 1e-9 else 1e-9
            profiles.append(
                _ColumnProfile(
                    name=name, kind="numeric", top_values=tuple(), median=med, mad=mad
                )
            )
        else:
            strs = [_safe_str(v).strip() for v in raw_vals]
            # Top frequent values to support "typo near frequent token" heuristics
            freq: Dict[str, int] = {}
            for s in strs:
                if s == "":
                    continue
                freq[s] = freq.get(s, 0) + 1
            top = tuple(
                [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:50]]
            )
            dom_pat = _infer_dominant_string_pattern(strs)
            profiles.append(
                _ColumnProfile(
                    name=name, kind="string", top_values=top, dominant_pattern=dom_pat
                )
            )

    # Infer simple cross-column constraints (best key column per target column) on string columns.
    # This is a lightweight proxy for "semantic constraints between columns".
    for target_col in range(num_cols):
        p = profiles[target_col]
        if p.kind != "string":
            continue
        best_key = None
        best_score = 0.0
        best_allowed: Optional[Dict[str, Tuple[str, ...]]] = None
        for key_col in range(num_cols):
            if key_col == target_col:
                continue
            # Build key -> set(values) map
            tmp: Dict[str, set] = {}
            cnt = 0
            for r in clean_rows:
                if key_col >= len(r) or target_col >= len(r):
                    continue
                k = _safe_str(r[key_col]).strip()
                v = _safe_str(r[target_col]).strip()
                if k == "" or v == "":
                    continue
                cnt += 1
                tmp.setdefault(k, set()).add(v)
            if cnt < 50:
                continue
            single = sum(1 for vs in tmp.values() if len(vs) == 1)
            score = single / max(1, len(tmp))
            if score > best_score:
                # freeze as tuples for more stable printing/debugging
                allowed = {k: tuple(sorted(vs)) for k, vs in tmp.items()}
                best_score = score
                best_key = key_col
                best_allowed = allowed
        if best_key is not None and best_score >= 0.90 and best_allowed is not None:
            profiles[target_col] = _ColumnProfile(
                name=p.name,
                kind=p.kind,
                top_values=p.top_values,
                median=p.median,
                mad=p.mad,
                dominant_pattern=p.dominant_pattern,
                constraint_key_col=int(best_key),
                allowed_by_key=best_allowed,
            )
    return profiles


def _constraint_violation_score(
    col_idx: int,
    row: List[Any],
    profiles: List[_ColumnProfile],
) -> float:
    """
    Compute constraint violation score for a cell.

    Args:
        col_idx: Column index
        row: Row data
        profiles: List of column profiles

    Returns:
        Constraint violation score (0.0 to 1.0)
    """
    if col_idx < 0 or col_idx >= len(profiles):
        return 0.0
    p = profiles[col_idx]
    if p.kind != "string" or p.constraint_key_col is None or not p.allowed_by_key:
        return 0.0
    kcol = int(p.constraint_key_col)
    if kcol < 0 or kcol >= len(row):
        return 0.0
    key = _safe_str(row[kcol]).strip()
    val = _safe_str(row[col_idx]).strip()
    if key == "" or val == "":
        return 0.0
    allowed = p.allowed_by_key.get(key)
    if not allowed:
        return 0.0
    return 0.8 if val not in set(allowed) else 0.0


def _make_typo(s: str, rng: random.Random) -> str:
    """
    Generate a typo in a string.

    Args:
        s: Input string
        rng: Random number generator

    Returns:
        String with a typo
    """
    if not s:
        return s
    if len(s) == 1:
        return s + rng.choice(string.ascii_lowercase)
    ops = ["delete", "swap", "replace", "insert"]
    op = rng.choice(ops)
    i = rng.randrange(0, len(s))
    if op == "delete":
        return s[:i] + s[i + 1 :]
    if op == "swap" and len(s) >= 2:
        j = min(len(s) - 1, i + 1)
        if i == j:
            return s
        lst = list(s)
        lst[i], lst[j] = lst[j], lst[i]
        return "".join(lst)
    if op == "replace":
        ch = rng.choice(string.ascii_lowercase + string.digits)
        return s[:i] + ch + s[i + 1 :]
    # insert
    ch = rng.choice(string.ascii_lowercase + string.digits)
    return s[:i] + ch + s[i:]


def _corrupt_value(
    value: Any,
    profile: _ColumnProfile,
    corruption_type: str,
    rng: random.Random,
) -> Any:
    """
    Create a single-cell corruption to simulate:
      - syntax: missing / typo / pattern violation
      - semantics: outlier-like (numeric) / swapped-value (string)

    Args:
        value: Cell value
        profile: Column profile
        corruption_type: Type of corruption
        rng: Random number generator

    Returns:
        Corrupted value
    """
    if corruption_type == "missing":
        return ""

    if profile.kind == "numeric":
        x = _try_parse_float(value)
        if x is None:
            # Missing-like for numeric if unparsable
            return "" if corruption_type in {"missing", "outlier"} else value
        if corruption_type == "outlier":
            # Robust outlier: shift by many MADs
            shift = (10.0 + 5.0 * rng.random()) * float(profile.mad or 1.0)
            return x + (shift if rng.random() < 0.5 else -shift)
        # Fallback numeric corruption: small random perturbation
        return x + rng.uniform(-1.0, 1.0) * float(profile.mad or 1.0)

    # string-like
    s = _safe_str(value).strip()
    if corruption_type == "typo":
        return _make_typo(s, rng)
    if corruption_type == "pattern_violation":
        # Make it violate a dominant pattern if known, otherwise inject odd symbols.
        if profile.dominant_pattern in {
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{2}/\d{2}/\d{4}$",
        }:
            return "9999-99-99"
        if "-" in s:
            return s.replace("-", "_", 1)
        if "/" in s:
            return s.replace("/", "-", 1)
        return s + "!!"
    if corruption_type == "swap_with_frequent":
        # semantic-ish: swap into a frequent token (wrong but plausible)
        if profile.top_values:
            cand = rng.choice(profile.top_values)
            if cand != s:
                return cand
        return _make_typo(s, rng)
    # default: typo-ish
    return _make_typo(s, rng)


def _compute_rule_score(value: Any, profile: _ColumnProfile) -> float:
    """
    A cheap score in [0,1] indicating likelihood of a *syntax-style* error.
    Used mainly for candidate generation / debugging; not required for training.

    Args:
        value: Cell value
        profile: Column profile

    Returns:
        Rule-based error score
    """
    s = _safe_str(value).strip()
    if s == "" or s.lower() in {"na", "n/a", "null", "none", "nan"}:
        return 1.0
    if profile.kind == "numeric":
        x = _try_parse_float(value)
        if x is None:
            return 1.0
        # robust z-score
        med = float(profile.median or 0.0)
        mad = float(profile.mad or 1.0)
        rz = abs((x - med) / mad)
        # squash: >8 MAD is very likely an outlier
        return float(min(1.0, rz / 8.0))
    # pattern violation
    if profile.dominant_pattern:
        try:
            if re.match(profile.dominant_pattern, s) is None:
                return 0.9
        except re.error:
            pass
    # typo heuristic: close to a frequent token but not equal
    if profile.top_values and s not in profile.top_values:
        best = 0.0
        for tok in profile.top_values[:20]:
            best = max(best, SequenceMatcher(a=s, b=tok).ratio())
        if best >= 0.85:
            return 0.7
    return 0.0


def _corrupt_row_cell(
    row: List[Any],
    col_idx: int,
    profiles: List[_ColumnProfile],
    corruption_type: str,
    rng: random.Random,
) -> Any:
    """
    Cell corruption that may depend on other columns (e.g., constraint violation).

    Args:
        row: Row data
        col_idx: Column index
        profiles: List of column profiles
        corruption_type: Type of corruption
        rng: Random number generator

    Returns:
        Corrupted cell value
    """
    if col_idx < 0 or col_idx >= len(profiles):
        return row[col_idx]
    profile = profiles[col_idx]
    if corruption_type in {"constraint_violation", "fd_violation"} and profile.allowed_by_key and profile.constraint_key_col is not None:
        kcol = int(profile.constraint_key_col)
        key = _safe_str(row[kcol]).strip() if 0 <= kcol < len(row) else ""
        cur = _safe_str(row[col_idx]).strip()
        if key != "" and cur != "":
            allowed_cur = set(profile.allowed_by_key.get(key, ()))
            # pick a value that is allowed under some other key but not under current key
            candidates: List[str] = []
            for other_k, vals in profile.allowed_by_key.items():
                if other_k == key:
                    continue
                for v in vals:
                    if v not in allowed_cur:
                        candidates.append(v)
            if candidates:
                return rng.choice(candidates)
    return _corrupt_value(row[col_idx], profile, corruption_type, rng)
