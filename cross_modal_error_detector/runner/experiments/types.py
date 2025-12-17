"""
Shared type definitions for experiments module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class _ColumnProfile:
    """
    Column profile for analyzing tabular data characteristics.

    This dataclass stores statistical and structural information about a column
    that is used for error detection and data corruption strategies.
    """
    name: str
    kind: str  # "numeric" | "string"
    top_values: Tuple[str, ...]
    # For numeric
    median: Optional[float] = None
    mad: Optional[float] = None
    # For pattern-ish strings
    dominant_pattern: Optional[str] = None  # regex
    # For simple cross-column semantic constraints (approx functional dependency)
    constraint_key_col: Optional[int] = None
    allowed_by_key: Optional[Dict[str, Tuple[str, ...]]] = None
