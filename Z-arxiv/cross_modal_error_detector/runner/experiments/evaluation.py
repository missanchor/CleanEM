"""
Evaluation utilities for experiments.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ...datasets import CorruptionBasedDataset
from .data_utils import _prepare_eval_loader


def _compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute classification metrics from predictions and targets.

    Args:
        preds: Predicted labels
        targets: True labels

    Returns:
        Dictionary of classification metrics
    """
    preds = preds.int()
    targets = targets.int()

    tp = int(((preds == 1) & (targets == 1)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    overall_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    dirty_accuracy = tp / (tp + fn) if (tp + fn) else 0.0
    clean_accuracy = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": dirty_accuracy,
        "dirty_accuracy": dirty_accuracy,
        "clean_accuracy": clean_accuracy,
        "overall_accuracy": overall_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _evaluate_corruption_model(
    model,
    dataset: CorruptionBasedDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: str,
    device_map: Optional[Dict[str, str]],
    threshold: float = 0.5,
    _prepare_eval_loader_fn=_prepare_eval_loader,
    _compute_metrics_fn=_compute_classification_metrics,
) -> Dict[str, float]:
    """
    Evaluate a corruption-based model.

    Args:
        model: Cross-modal error detector model
        dataset: Corruption-based dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        device: Device to use
        device_map: Device mapping for model components
        threshold: Decision threshold
        _prepare_eval_loader_fn: Function to prepare eval loader (injected for testing)
        _compute_metrics_fn: Function to compute metrics (injected for testing)

    Returns:
        Dictionary of evaluation metrics
    """
    eval_loader = _prepare_eval_loader_fn(
        dataset,
        batch_size,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    dirty_preds: List[torch.Tensor] = []
    dirty_targets: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        # Import here to avoid circular dependency
        from ...runtime import _move_batch_to_device, _resolve_runtime_device
        from ...training import build_tabular_inputs_from_rows

        for row_samples, text_inputs, labels in eval_loader:
            tabular_device = _resolve_runtime_device(device, device_map, "tabular_encoder")
            text_device = _resolve_runtime_device(device, device_map, "text_encoder")
            label_device = _resolve_runtime_device(device, device_map, "detection_head")

            tabular_processor = dataset.tabular_processor.to(tabular_device)
            tabular_inputs = build_tabular_inputs_from_rows(
                row_samples,
                tabular_processor,
                getattr(dataset, "column_names", None),
            )
            text_inputs = _move_batch_to_device(text_inputs, text_device)
            labels = labels.to(label_device)

            logits = model(tabular_inputs, text_inputs).squeeze(-1)
            probs = torch.sigmoid(logits)
            clean_predictions = probs > threshold

            pred_dirty = (~clean_predictions).int()
            target_dirty = (labels < 0.5).int()

            dirty_preds.append(pred_dirty.flatten().cpu())
            dirty_targets.append(target_dirty.flatten().cpu())

    if not dirty_preds:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "dirty_accuracy": 0.0,
            "clean_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "dirty_support": 0,
            "clean_support": 0,
            "total_support": 0,
        }

    preds_tensor = torch.cat(dirty_preds)
    targets_tensor = torch.cat(dirty_targets)

    metrics = _compute_metrics_fn(preds_tensor, targets_tensor)
    metrics["dirty_support"] = int(targets_tensor.sum().item())
    metrics["clean_support"] = int((targets_tensor == 0).sum().item())
    metrics["total_support"] = int(targets_tensor.numel())
    return metrics


def _evaluate_threshold_with_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    _compute_metrics_fn=_compute_classification_metrics,
) -> Dict[str, float]:
    """
    Evaluate threshold with given scores and targets.

    Args:
        scores: Prediction scores
        targets: True labels
        threshold: Decision threshold
        _compute_metrics_fn: Function to compute metrics (injected for testing)

    Returns:
        Dictionary of evaluation metrics
    """
    preds_dirty = (scores < threshold).int()
    metrics = _compute_metrics_fn(preds_dirty, targets)
    metrics["threshold"] = threshold
    return metrics
