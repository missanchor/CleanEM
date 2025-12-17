"""
Data loading and processing utilities for experiments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ...datasets import CleanDirtyEvaluationDataset
from ...training import collate_fn_corruption
from ..configuration import _resolve_path_like
from ..data_loading import load_csv_dataset


def _resolve_num_workers(exp_cfg: Dict[str, Any]) -> int:
    """
    Resolve the number of worker processes for data loaders.

    Args:
        exp_cfg: Experiment configuration dictionary

    Returns:
        Number of worker processes to use
    """
    override = exp_cfg.get("num_workers")
    if override is not None:
        return max(1, int(override))
    cpu_cnt = os.cpu_count() or 1
    auto_workers = max(4, cpu_cnt // 2)
    return auto_workers


def _derive_loader_runtime(exp_cfg: Dict[str, Any]) -> Tuple[int, bool, bool]:
    """
    Derive runtime configuration for data loaders.

    Args:
        exp_cfg: Experiment configuration dictionary

    Returns:
        Tuple of (num_workers, pin_memory, persistent_workers)
    """
    num_workers = _resolve_num_workers(exp_cfg)
    pin_memory = bool(exp_cfg.get("pin_memory", True))
    persistent_workers = bool(exp_cfg.get("persistent_workers", True))
    return num_workers, pin_memory, persistent_workers


def _attach_processor_params(
    optimizer: torch.optim.Optimizer,
    tabular_processor: nn.Module,
) -> None:
    """
    Add tabular processor parameters to optimizer if they require gradients.

    Args:
        optimizer: PyTorch optimizer
        tabular_processor: Tabular processor module
    """
    if tabular_processor is None:
        return
    processor_params = [p for p in tabular_processor.parameters() if p.requires_grad]
    if processor_params:
        optimizer.add_param_group({"params": processor_params})


def _prepare_eval_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    """
    Prepare a DataLoader for evaluation.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers

    Returns:
        Configured DataLoader for evaluation
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_corruption,
        num_workers=0,  # Set to 0 to avoid multiprocessing serialization issues
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def _build_eval_dataset_from_config(
    exp_cfg: Dict[str, Any],
    tabular_processor,
    text_processor,
    *,
    config_dir: Path,
    project_root: Path,
) -> Tuple[Optional[CleanDirtyEvaluationDataset], Optional[str]]:
    """
    Build evaluation dataset from configuration.

    Args:
        exp_cfg: Experiment configuration
        tabular_processor: Tabular processor
        text_processor: Text processor
        config_dir: Configuration directory
        project_root: Project root directory

    Returns:
        Tuple of (eval_dataset, dataset_name)
    """
    eval_cfg = exp_cfg.get("evaluation")
    if not eval_cfg:
        return None, None

    clean_path_value = eval_cfg.get("clean_data_path")
    dirty_path_value = eval_cfg.get("dirty_data_path")
    if not clean_path_value or not dirty_path_value:
        raise ValueError("evaluation 配置需要同时提供 clean_data_path 和 dirty_data_path。")

    clean_path = Path(_resolve_path_like(clean_path_value, config_dir, project_root))
    dirty_path = Path(_resolve_path_like(dirty_path_value, config_dir, project_root))
    max_rows = eval_cfg.get("max_rows")
    dataset_name = eval_cfg.get("dataset_name")

    clean_rows, _, _, clean_column_names = load_csv_dataset(
        clean_path, dataset_name=dataset_name, max_rows=max_rows
    )
    dirty_rows, dirty_texts, dirty_dataset_name, dirty_column_names = load_csv_dataset(
        dirty_path,
        dataset_name=dataset_name,
        max_rows=max_rows,
    )

    # Use column names from dirty data (or clean if dirty doesn't have them)
    column_names = dirty_column_names or clean_column_names

    eval_dataset = CleanDirtyEvaluationDataset(
        clean_rows=clean_rows,
        dirty_rows=dirty_rows,
        text_descriptions=dirty_texts,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
        column_names=column_names,
    )

    reported_name = dataset_name or dirty_dataset_name
    return eval_dataset, reported_name


def _safe_str(x: Any) -> str:
    """
    Safely convert a value to string.

    Args:
        x: Value to convert

    Returns:
        String representation or empty string on error
    """
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _try_parse_float(x: Any) -> Optional[float]:
    """
    Try to parse a value as float.

    Args:
        x: Value to parse

    Returns:
        Float value or None if parsing fails
    """
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = _safe_str(x).strip()
    if s == "":
        return None
    # Common missing markers
    if s.lower() in {"na", "n/a", "null", "none", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None
