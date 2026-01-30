"""
Conflict Topology Analysis for Dual Rule Refinement.

This module provides set-theoretic analysis of conflicts between P_clean and P_dirty rules.
It classifies conflicts into types (SUBSET, INTERSECT, SUPERSET, DISJOINT) and calculates
intersection-over-union (IoU) metrics to guide refinement strategies.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set, Optional
import pandas as pd
import numpy as np


class ConflictType(Enum):
    """Classification of conflict topology between two rule sets."""
    SUBSET = "SUBSET"
    INTERSECT = "INTERSECT"
    SUPERSET = "SUPERSET"
    DISJOINT = "DISJOINT"


@dataclass
class ConflictTopology:
    """Topology analysis result for a conflict between two rule sets."""
    column: str
    conflict_type: ConflictType
    
    # Set sizes
    clean_set_size: int
    dirty_set_size: int
    intersection_size: int
    union_size: int
    
    # Metrics
    iou: float  # Intersection over Union
    clean_in_intersection_ratio: float  # intersection / clean_set
    dirty_in_intersection_ratio: float  # intersection / dirty_set
    
    # Sample indices
    intersection_samples: List[int] = field(default_factory=list)
    clean_only_samples: List[int] = field(default_factory=list)
    dirty_only_samples: List[int] = field(default_factory=list)
    
    # Metadata
    total_rows: int = 0


@dataclass
class RefinementTask:
    """Task specification for rule refinement."""
    column: str
    conflict_type: ConflictType
    
    # Current rules
    current_clean_rule: str
    current_dirty_rule: str
    
    # Problem samples
    conflict_samples: List[Dict[str, Any]] = field(default_factory=list)
    grey_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Refinement strategy
    strategy: str = "default"  # 'clean', 'dirty', 'both', 'fallback'
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DisjointnessResult:
    """Result of disjointness validation after refinement."""
    column: str
    is_disjoint: bool
    intersection_count: int
    intersection_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    clean_coverage_rate: float = 0.0
    dirty_coverage_rate: float = 0.0
    gap_rate: float = 0.0  # both false
    
    # Validation status
    validation_status: str = "unknown"  # 'pass', 'fail', 'error'


def classify_conflict_type(clean_set_size: int, dirty_set_size: int, 
                          intersection_size: int) -> ConflictType:
    """
    Classify the conflict type based on set relationships.
    
    Args:
        clean_set_size: Size of clean set (P_clean = True)
        dirty_set_size: Size of dirty set (P_dirty = True)
        intersection_size: Size of intersection (both True)
    
    Returns:
        ConflictType enum value
    """
    if clean_set_size == 0 or dirty_set_size == 0:
        return ConflictType.DISJOINT
    
    if intersection_size == 0:
        return ConflictType.DISJOINT
    
    clean_ratio = intersection_size / clean_set_size if clean_set_size > 0 else 0
    dirty_ratio = intersection_size / dirty_set_size if dirty_set_size > 0 else 0
    
    threshold = 0.8  # 80% overlap threshold
    
    if clean_ratio >= threshold and dirty_ratio >= threshold:
        return ConflictType.SUBSET
    elif clean_ratio >= threshold:
        return ConflictType.SUPERSET
    elif dirty_ratio >= threshold:
        return ConflictType.SUBSET
    else:
        return ConflictType.INTERSECT


def calculate_iou(clean_set_size: int, dirty_set_size: int, 
                  intersection_size: int) -> float:
    """
    Calculate Intersection over Union (IoU) between clean and dirty sets.
    
    Args:
        clean_set_size: Size of clean set
        dirty_set_size: Size of dirty set
        intersection_size: Size of intersection
    
    Returns:
        IoU value in [0, 1]
    """
    union_size = clean_set_size + dirty_set_size - intersection_size
    if union_size == 0:
        return 0.0
    return intersection_size / union_size


def analyze_conflict_topology(df: pd.DataFrame, column: str,
                               clean_mask: np.ndarray, dirty_mask: np.ndarray,
                               max_samples: int = 10) -> ConflictTopology:
    """
    Analyze the topology of conflicts between P_clean and P_dirty rules.
    
    Args:
        df: DataFrame containing the data
        column: Column name being analyzed
        clean_mask: Boolean array where True indicates P_clean = True
        dirty_mask: Boolean array where True indicates P_dirty = True
        max_samples: Maximum number of samples to collect
    
    Returns:
        ConflictTopology object with detailed analysis
    """
    # Ensure masks are numpy arrays
    clean_mask = np.asarray(clean_mask)
    dirty_mask = np.asarray(dirty_mask)
    
    # Calculate set sizes
    clean_set_size = int(np.sum(clean_mask))
    dirty_set_size = int(np.sum(dirty_mask))
    intersection_mask = clean_mask & dirty_mask
    intersection_size = int(np.sum(intersection_mask))
    union_size = int(np.sum(clean_mask | dirty_mask))
    
    # Calculate metrics
    iou = calculate_iou(clean_set_size, dirty_set_size, intersection_size)
    clean_in_intersection_ratio = intersection_size / clean_set_size if clean_set_size > 0 else 0.0
    dirty_in_intersection_ratio = intersection_size / dirty_set_size if dirty_set_size > 0 else 0.0
    
    # Classify conflict type
    conflict_type = classify_conflict_type(clean_set_size, dirty_set_size, intersection_size)
    
    # Collect samples
    intersection_indices = np.where(intersection_mask)[0]
    clean_only_mask = clean_mask & ~dirty_mask
    clean_only_indices = np.where(clean_only_mask)[0]
    dirty_only_mask = dirty_mask & ~clean_mask
    dirty_only_indices = np.where(dirty_only_mask)[0]
    
    intersection_samples = list(intersection_indices[:max_samples])
    clean_only_samples = list(clean_only_indices[:max_samples])
    dirty_only_samples = list(dirty_only_indices[:max_samples])
    
    return ConflictTopology(
        column=column,
        conflict_type=conflict_type,
        clean_set_size=clean_set_size,
        dirty_set_size=dirty_set_size,
        intersection_size=intersection_size,
        union_size=union_size,
        iou=iou,
        clean_in_intersection_ratio=clean_in_intersection_ratio,
        dirty_in_intersection_ratio=dirty_in_intersection_ratio,
        intersection_samples=intersection_samples,
        clean_only_samples=clean_only_samples,
        dirty_only_samples=dirty_only_samples,
        total_rows=len(df)
    )


def select_intersection_samples(df: pd.DataFrame, column: str,
                               clean_mask: np.ndarray, dirty_mask: np.ndarray,
                               max_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Select representative samples from the intersection (conflict) zone.
    
    Args:
        df: DataFrame containing the data
        column: Column name
        clean_mask: Boolean array for P_clean
        dirty_mask: Boolean array for P_dirty
        max_samples: Maximum number of samples to return
    
    Returns:
        List of sample dictionaries with row_index, value, and count
    """
    intersection_mask = clean_mask & dirty_mask
    intersection_indices = np.where(intersection_mask)[0]
    
    samples = []
    for idx in intersection_indices[:max_samples]:
        value = df.iloc[idx][column]
        samples.append({
            'row_index': int(idx),
            'value': value,
            'column': column
        })
    
    return samples


def select_gap_samples(df: pd.DataFrame, column: str,
                       clean_mask: np.ndarray, dirty_mask: np.ndarray,
                       max_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Select representative samples from the gap zone (both false).
    
    Args:
        df: DataFrame containing the data
        column: Column name
        clean_mask: Boolean array for P_clean
        dirty_mask: Boolean array for P_dirty
        max_samples: Maximum number of samples to return
    
    Returns:
        List of sample dictionaries with row_index, value, and count
    """
    gap_mask = ~clean_mask & ~dirty_mask
    gap_indices = np.where(gap_mask)[0]
    
    samples = []
    for idx in gap_indices[:max_samples]:
        value = df.iloc[idx][column]
        samples.append({
            'row_index': int(idx),
            'value': value,
            'column': column
        })
    
    return samples


def validate_disjointness(df: pd.DataFrame, column: str,
                         clean_func: callable, dirty_func: callable) -> DisjointnessResult:
    """
    Validate that P_clean and P_dirty are disjoint (no intersections).
    
    Args:
        df: DataFrame containing the data
        column: Column name
        clean_func: P_clean predicate function
        dirty_func: P_dirty predicate function
    
    Returns:
        DisjointnessResult object with validation details
    """
    # Apply predicates
    clean_mask = []
    dirty_mask = []
    intersection_samples = []
    
    for idx, row in df.iterrows():
        value = row[column]
        try:
            is_clean = bool(clean_func(value, row))
            is_dirty = bool(dirty_func(value, row))
        except Exception:
            is_clean = False
            is_dirty = False
        
        clean_mask.append(is_clean)
        dirty_mask.append(is_dirty)
        
        if is_clean and is_dirty:
            intersection_samples.append({
                'row_index': int(idx),
                'value': value,
                'column': column
            })
    
    clean_mask = np.array(clean_mask)
    dirty_mask = np.array(dirty_mask)
    
    # Calculate metrics
    intersection_count = len(intersection_samples)
    clean_coverage_rate = np.mean(clean_mask)
    dirty_coverage_rate = np.mean(dirty_mask)
    gap_mask = ~clean_mask & ~dirty_mask
    gap_rate = np.mean(gap_mask)
    
    # Determine status
    is_disjoint = intersection_count == 0
    validation_status = 'pass' if is_disjoint else 'fail'
    
    return DisjointnessResult(
        column=column,
        is_disjoint=is_disjoint,
        intersection_count=intersection_count,
        intersection_samples=intersection_samples[:10],  # Limit samples
        clean_coverage_rate=clean_coverage_rate,
        dirty_coverage_rate=dirty_coverage_rate,
        gap_rate=gap_rate,
        validation_status=validation_status
    )


def generate_refinement_task(
    topology: ConflictTopology,
    current_clean_rule: str,
    current_dirty_rule: str,
    metadata: Dict[str, Any] = None
) -> RefinementTask:
    if topology.conflict_type == ConflictType.SUBSET:
        strategy = "clean"
    elif topology.conflict_type == ConflictType.SUPERSET:
        strategy = "dirty"
    elif topology.conflict_type == ConflictType.INTERSECT:
        strategy = "both"
    else:
        strategy = "fallback"

    conflict_samples = list(topology.intersection_samples) if topology.intersection_samples else []

    return RefinementTask(
        column=topology.column,
        conflict_type=topology.conflict_type,
        current_clean_rule=current_clean_rule,
        current_dirty_rule=current_dirty_rule,
        conflict_samples=conflict_samples,
        grey_samples=[],
        strategy=strategy,
        metadata=metadata or {}
    )
