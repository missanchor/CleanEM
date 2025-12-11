from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..datasets import (
    CleanDirtyEvaluationDataset,
    PerColumnBinaryDataset,
    ContrastiveDataset,
    CorruptionBasedDataset,
)
from ..model import CrossModalErrorDetector
from ..training import (
    collate_fn_contrastive,
    collate_fn_contrastive_cell_level,
    collate_fn_corruption,
    compute_embedding_similarity,
    train_step_contrastive_pretrain,
    train_step_corruption,
)
from .components import (
    DETECTION_HEAD_REGISTRY,
    ENCODER_REGISTRY,
    FUSION_REGISTRY,
    build_component,
    build_optimizer,
)
from .configuration import _instantiate_processors, _resolve_path_like
from .data_loading import load_csv_dataset, load_data_from_config
from .runtime import _move_batch_to_device, _resolve_runtime_device
from ..utils.device import resolve_runtime_device


def _resolve_num_workers(exp_cfg: Dict[str, Any]) -> int:
    override = exp_cfg.get("num_workers")
    if override is not None:
        return max(1, int(override))
    cpu_cnt = os.cpu_count() or 1
    auto_workers = max(4, cpu_cnt // 2)
    return auto_workers


def _derive_loader_runtime(exp_cfg: Dict[str, Any]) -> Tuple[int, bool, bool]:
    num_workers = _resolve_num_workers(exp_cfg)
    pin_memory = bool(exp_cfg.get("pin_memory", True))
    persistent_workers = bool(exp_cfg.get("persistent_workers", True))
    return num_workers, pin_memory, persistent_workers


def _prepare_eval_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_corruption,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def _build_eval_dataset_from_config(
    exp_cfg: Dict[str, Any],
    tabular_processor,
    text_processor,
    *,
    config_dir: Path,
    project_root: Path,
) -> Tuple[Optional[CleanDirtyEvaluationDataset], Optional[str]]:
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

    clean_rows, _, _ = load_csv_dataset(clean_path, dataset_name=dataset_name, max_rows=max_rows)
    dirty_rows, dirty_texts, dirty_dataset_name = load_csv_dataset(
        dirty_path,
        dataset_name=dataset_name,
        max_rows=max_rows,
    )

    eval_dataset = CleanDirtyEvaluationDataset(
        clean_rows=clean_rows,
        dirty_rows=dirty_rows,
        text_descriptions=dirty_texts,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
    )

    reported_name = dataset_name or dirty_dataset_name
    return eval_dataset, reported_name


def _compute_classification_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
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
    model: CrossModalErrorDetector,
    dataset: CorruptionBasedDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: str,
    device_map: Optional[Dict[str, str]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    eval_loader = _prepare_eval_loader(
        dataset,
        batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    dirty_preds: List[torch.Tensor] = []
    dirty_targets: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for tabular_inputs, text_inputs, labels in eval_loader:
            tabular_device = _resolve_runtime_device(device, device_map, "tabular_encoder")
            text_device = _resolve_runtime_device(device, device_map, "text_encoder")
            label_device = _resolve_runtime_device(device, device_map, "detection_head")

            tabular_inputs = _move_batch_to_device(tabular_inputs, tabular_device)
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

    metrics = _compute_classification_metrics(preds_tensor, targets_tensor)
    metrics["dirty_support"] = int(targets_tensor.sum().item())
    metrics["clean_support"] = int((targets_tensor == 0).sum().item())
    metrics["total_support"] = int(targets_tensor.numel())
    return metrics


def _evaluate_threshold_with_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    preds_dirty = (scores < threshold).int()
    metrics = _compute_classification_metrics(preds_dirty, targets)
    metrics["threshold"] = threshold
    return metrics


def _search_best_score_threshold(scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Find the threshold on raw logits that maximizes F1 for detecting dirty pairs.
    """

    if scores.numel() == 0:
        return {
            "threshold": 0.0,
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
        }

    sorted_scores = torch.sort(torch.unique(scores)).values
    if sorted_scores.numel() == 1:
        candidates = [
            float(sorted_scores.item() - 1e-6),
            float(sorted_scores.item() + 1e-6),
        ]
    else:
        midpoints = (sorted_scores[:-1] + sorted_scores[1:]) / 2
        candidates = [float(sorted_scores[0] - 1e-6)]
        candidates += [float(x) for x in midpoints]
        candidates.append(float(sorted_scores[-1] + 1e-6))

    best_metrics = None
    best_f1 = -1.0
    for threshold in candidates:
        metrics = _evaluate_threshold_with_scores(scores, targets, threshold)
        if (
            metrics["f1"] > best_f1
            or (
                metrics["f1"] == best_f1
                and metrics["precision"] > (best_metrics or {}).get("precision", -1.0)
            )
        ):
            best_metrics = metrics
            best_f1 = metrics["f1"]

    return best_metrics or _evaluate_threshold_with_scores(scores, targets, 0.0)


def _evaluate_pairwise_detection(
    model: CrossModalErrorDetector,
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: str,
    device_map: Optional[Dict[str, str]],
    default_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluate pair-level matching accuracy by turning contrastive scores into binary predictions.
    """

    eval_loader = _prepare_eval_loader(
        dataset,
        batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    score_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for tabular_inputs, text_inputs, labels in eval_loader:
            tabular_device = _resolve_runtime_device(device, device_map, "tabular_encoder")
            text_device = _resolve_runtime_device(device, device_map, "text_encoder")
            head_device = _resolve_runtime_device(device, device_map, "detection_head")

            tabular_inputs = _move_batch_to_device(tabular_inputs, tabular_device)
            text_inputs = _move_batch_to_device(text_inputs, text_device)
            labels = labels.to(head_device)

            logits = model(tabular_inputs, text_inputs).squeeze(-1)
            score_list.append(logits.flatten().detach().cpu())
            target_list.append((labels.flatten() > 0.5).int().cpu())

    if not score_list:
        return {
            "pair_precision": 0.0,
            "pair_recall": 0.0,
            "pair_f1": 0.0,
            "pair_accuracy": 0.0,
            "pair_dirty_accuracy": 0.0,
            "pair_clean_accuracy": 0.0,
            "pair_overall_accuracy": 0.0,
            "pair_best_threshold": 0.0,
            "pair_default_threshold": default_threshold or 0.0,
            "pair_precision_default": 0.0,
            "pair_recall_default": 0.0,
            "pair_f1_default": 0.0,
            "pair_dirty_support": 0,
            "pair_clean_support": 0,
            "pair_total_support": 0,
            "pair_clean_score_mean": 0.0,
            "pair_dirty_score_mean": 0.0,
        }

    scores = torch.cat(score_list)
    targets = torch.cat(target_list)

    best_metrics = _search_best_score_threshold(scores, targets)
    default_threshold = 0.0 if default_threshold is None else default_threshold
    default_metrics = _evaluate_threshold_with_scores(scores, targets, default_threshold)

    clean_mask = targets == 0
    dirty_mask = targets == 1
    clean_scores_mean = float(scores[clean_mask].mean().item()) if clean_mask.any() else 0.0
    dirty_scores_mean = float(scores[dirty_mask].mean().item()) if dirty_mask.any() else 0.0

    metrics = {
        "pair_precision": best_metrics["precision"],
        "pair_recall": best_metrics["recall"],
        "pair_f1": best_metrics["f1"],
        "pair_accuracy": best_metrics["accuracy"],
        "pair_dirty_accuracy": best_metrics["dirty_accuracy"],
        "pair_clean_accuracy": best_metrics["clean_accuracy"],
        "pair_overall_accuracy": best_metrics["overall_accuracy"],
        "pair_best_threshold": best_metrics["threshold"],
        "pair_default_threshold": default_threshold,
        "pair_precision_default": default_metrics["precision"],
        "pair_recall_default": default_metrics["recall"],
        "pair_f1_default": default_metrics["f1"],
        "pair_dirty_support": int(targets.sum().item()),
        "pair_clean_support": int((targets == 0).sum().item()),
        "pair_total_support": int(targets.numel()),
        "pair_clean_score_mean": clean_scores_mean,
        "pair_dirty_score_mean": dirty_scores_mean,
    }
    return metrics


from tqdm import tqdm


def _precompute_text_embeddings(
    text_encoder: torch.nn.Module,
    text_descriptions: List[str],
    text_processor,
    device: str,
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """
    Precompute text embeddings for all descriptions.
    """
    print("正在预计算文本 Embeddings...")
    text_encoder.eval()
    target_device = torch.device(device)
    text_encoder.to(target_device)

    cached_embeddings = {}

    # Process in batches
    num_samples = len(text_descriptions)
    for i in tqdm(range(0, num_samples, batch_size), desc="Caching Embeddings", ncols=100):
        batch_texts = text_descriptions[i : i + batch_size]
        indices = list(range(i, min(i + batch_size, num_samples)))

        # Process batch
        batch_inputs = [text_processor.process(t) for t in batch_texts]

        # Collate manually
        input_ids = torch.stack([x["input_ids"] for x in batch_inputs]).to(target_device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch_inputs]).to(target_device)

        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            embeddings = text_encoder(text_inputs)

        # Store individually
        for idx, emb in zip(indices, embeddings):
            cached_embeddings[idx] = emb.cpu().detach()

    print(f"✓ 已缓存 {len(cached_embeddings)} 条文本 Embeddings")
    return cached_embeddings


def run_corruption_experiment(
    exp_cfg: Dict[str, Any],
    device: str,
    device_map: Optional[Dict[str, str]],
    *,
    seed: int,
    config_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("实验：Corruption-based 训练（破坏-重建）")
    print("=" * 80)

    clean_rows, text_descriptions, dataset_name = load_data_from_config(
        exp_cfg,
        default_seed=seed,
        config_dir=config_dir,
        project_root=project_root,
    )

    num_samples = len(clean_rows)
    corruption_prob = exp_cfg.get("corruption_prob", 0.3)
    batch_size = exp_cfg.get("batch_size", 8)
    num_epochs = exp_cfg.get("num_epochs", 10)

    tabular_processor, text_processor = _instantiate_processors(exp_cfg, config_dir, project_root)

    tabular_encoder = build_component(exp_cfg["tabular_encoder"], ENCODER_REGISTRY, config_dir, project_root)
    text_encoder = build_component(exp_cfg["text_encoder"], ENCODER_REGISTRY, config_dir, project_root)
    fusion_module = build_component(exp_cfg["fusion_module"], FUSION_REGISTRY, config_dir, project_root)
    detection_head = build_component(
        exp_cfg["detection_head"],
        DETECTION_HEAD_REGISTRY,
        config_dir,
        project_root,
    )

    # Precompute embeddings if using a frozen text encoder
    cached_embeddings = None
    text_encoder_params = exp_cfg.get("text_encoder", {}).get("params", {})
    is_frozen = text_encoder_params.get("freeze_pretrained", True) or text_encoder_params.get("freeze_base_model", True)

    if is_frozen and hasattr(text_encoder, "forward"):  # Ensure it's a model
        cache_device = device
        if device_map and "text_encoder" in device_map:
            cache_device = resolve_runtime_device(device_map["text_encoder"], fallback_device=device)

        cached_embeddings = _precompute_text_embeddings(
            text_encoder,
            text_descriptions,
            text_processor,
            device=cache_device,
            batch_size=batch_size,
        )

    model = CrossModalErrorDetector(
        tabular_encoder=tabular_encoder,
        text_encoder=text_encoder,
        fusion_module=fusion_module,
        detection_head=detection_head,
        device_map=device_map,
        default_device=device,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 模型参数量: {num_params:,}")

    optimizer = build_optimizer(model, exp_cfg.get("optimizer", {}))

    train_dataset = CorruptionBasedDataset(
        clean_rows=clean_rows,
        text_descriptions=text_descriptions,
        corruption_prob=corruption_prob,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
        cached_text_embeddings=cached_embeddings,
    )
    num_workers, pin_memory, persistent_workers = _derive_loader_runtime(exp_cfg)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_corruption,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    eval_dataset, eval_dataset_name = _build_eval_dataset_from_config(
        exp_cfg,
        tabular_processor,
        text_processor,
        config_dir=config_dir,
        project_root=project_root,
    )

    train_losses: List[float] = []
    for epoch in range(num_epochs):
        epoch_losses: List[float] = []
        for batch in dataloader:
            loss = train_step_corruption(
                model,
                batch,
                optimizer,
                device,
                device_map=device_map,
            )
            epoch_losses.append(loss)
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_losses.append(avg_loss)
        print(f"  Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    dataset_for_eval = eval_dataset or train_dataset
    reported_dataset_name = eval_dataset_name or dataset_name

    eval_metrics = _evaluate_corruption_model(
        model,
        dataset_for_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        device=device,
        device_map=device_map,
    )
    print(
        "\n  ✓ 脏值 Precision / Recall / F1: "
        f"{eval_metrics['precision']:.2%} / {eval_metrics['recall']:.2%} / {eval_metrics['f1']:.2%}"
    )
    print(f"  ✓ Dirty Accuracy: {eval_metrics['dirty_accuracy']:.2%}")
    print(f"  ✓ Overall Accuracy: {eval_metrics['overall_accuracy']:.2%}")

    return {
        "train_losses": train_losses,
        "accuracy": eval_metrics["accuracy"],
        "dirty_accuracy": eval_metrics["dirty_accuracy"],
        "overall_accuracy": eval_metrics["overall_accuracy"],
        "metrics": eval_metrics,
        "num_params": num_params,
        "dataset": reported_dataset_name,
        "num_samples": num_samples,
    }


def compute_error_mask(
    clean_rows: List[List[Any]],
    dirty_rows: List[List[Any]],
) -> List[List[int]]:
    """
    Compute error mask by comparing clean and dirty rows after string conversion.

    Args:
        clean_rows: List of clean rows
        dirty_rows: List of dirty rows

    Returns:
        List of masks where 1 indicates an error (cells differ), 0 indicates no error
    """
    error_masks = []
    for clean_row, dirty_row in zip(clean_rows, dirty_rows):
        mask = [
            1 if str(clean_cell) != str(dirty_cell) else 0
            for clean_cell, dirty_cell in zip(clean_row, dirty_row)
        ]
        error_masks.append(mask)
    return error_masks


def _search_best_score_threshold(scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Find the threshold on raw logits that maximizes F1 for detecting dirty pairs.
    """

    if scores.numel() == 0:
        return {
            "threshold": 0.0,
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
        }

    sorted_scores = torch.sort(torch.unique(scores)).values
    if sorted_scores.numel() == 1:
        candidates = [
            float(sorted_scores.item() - 1e-6),
            float(sorted_scores.item() + 1e-6),
        ]
    else:
        midpoints = (sorted_scores[:-1] + sorted_scores[1:]) / 2
        candidates = [float(sorted_scores[0] - 1e-6)]
        candidates += [float(x) for x in midpoints]
        candidates.append(float(sorted_scores[-1] + 1e-6))

    best_metrics = None
    best_f1 = -1.0
    for threshold in candidates:
        metrics = _evaluate_threshold_with_scores(scores, targets, threshold)
        if (
            metrics["f1"] > best_f1
            or (
                metrics["f1"] == best_f1
                and metrics["precision"] > (best_metrics or {}).get("precision", -1.0)
            )
        ):
            best_metrics = metrics
            best_f1 = metrics["f1"]

    return best_metrics or _evaluate_threshold_with_scores(scores, targets, 0.0)


def evaluate_pretrain_stage_and_search_threshold(
    tabular_encoder,
    text_encoder,
    eval_dataset: CleanDirtyEvaluationDataset,
    device: str,
    device_map: Optional[Dict[str, str]],
) -> float:
    """
    Stage A evaluation: Compute embedding similarity and search for best threshold.

    This evaluates how well the pretrained encoders can distinguish between
    matching and non-matching tabular-text pairs by computing cosine similarity
    and finding the optimal threshold.

    Returns:
        best_threshold: The similarity threshold that maximizes F1 score
    """
    all_scores = []
    all_labels = []

    for clean_row, dirty_row, text in zip(
        eval_dataset.clean_rows,
        eval_dataset.dirty_rows,
        eval_dataset.text_descriptions
    ):
        # Process clean row (positive sample)
        clean_tabular = eval_dataset.tabular_processor.process(clean_row, row_idx=0)
        clean_text = eval_dataset.text_processor.process(text)
        clean_score = compute_embedding_similarity(
            tabular_encoder, text_encoder, clean_tabular, clean_text,
            device=device, device_map=device_map
        ).item()
        all_scores.append(clean_score)
        all_labels.append(1)  # clean matches text (positive)

        # Process dirty row (negative sample)
        dirty_tabular = eval_dataset.tabular_processor.process(dirty_row, row_idx=0)
        dirty_score = compute_embedding_similarity(
            tabular_encoder, text_encoder, dirty_tabular, clean_text,
            device=device, device_map=device_map
        ).item()
        all_scores.append(dirty_score)
        all_labels.append(0)  # dirty doesn't match text (negative)

    # Search for best threshold
    scores_tensor = torch.tensor(all_scores)
    labels_tensor = torch.tensor(all_labels)
    best_threshold = _search_best_score_threshold(scores_tensor, labels_tensor)["threshold"]

    return best_threshold


def evaluate_two_stage_model(
    model: CrossModalErrorDetector,
    eval_dataset: CleanDirtyEvaluationDataset,
    device: str,
    device_map: Optional[Dict[str, str]],
) -> Dict[str, float]:
    """
    Stage B evaluation: Compute precision/recall/f1 for error detection.

    This evaluates how well the trained model detects errors in dirty data.
    Only considers cells that were actually erroneous (as determined by error mask).
    """
    # Compute error mask
    error_masks = compute_error_mask(
        eval_dataset.clean_rows,
        eval_dataset.dirty_rows
    )

    # Predict on dirty data using binary classification
    predictions = []
    for dirty_row, text in zip(eval_dataset.dirty_rows, eval_dataset.text_descriptions):
        tabular_inputs = eval_dataset.tabular_processor.process(dirty_row, row_idx=0)
        text_inputs = eval_dataset.text_processor.process(text)

        # Add batch dimension for single-row evaluation
        tabular_inputs = {k: v.unsqueeze(0) for k, v in tabular_inputs.items()}
        text_inputs = {k: v.unsqueeze(0) for k, v in text_inputs.items()}

        with torch.no_grad():
            logits = model(tabular_inputs, text_inputs)  # [1, seq_len, 1]
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)  # [1, seq_len, 1]
            # Model predicts match probability - lower score means more likely to be an error
            # So we predict error if prob < 0.5 (i.e., mismatch)
            pred = (probs < 0.5).int().squeeze(0).squeeze(-1).cpu().numpy()  # [seq_len]
            predictions.append(pred)

    # Compute metrics only on erroneous cells
    error_precisions, error_recalls, error_f1s = [], [], []

    for pred_row, error_mask in zip(predictions, error_masks):
        # Find positions with errors
        error_positions = [i for i, is_error in enumerate(error_mask) if is_error == 1]

        if not error_positions:
            continue

        # Extract predictions and true labels for error positions
        pred_errors = pred_row[error_positions]
        true_errors = [1] * len(error_positions)

        # Compute metrics for this row
        metrics = _compute_classification_metrics(
            torch.tensor(pred_errors),
            torch.tensor(true_errors)
        )

        error_precisions.append(metrics["precision"])
        error_recalls.append(metrics["recall"])
        error_f1s.append(metrics["f1"])

    # Compute average metrics
    avg_precision = np.mean(error_precisions) if error_precisions else 0.0
    avg_recall = np.mean(error_recalls) if error_recalls else 0.0
    avg_f1 = np.mean(error_f1s) if error_f1s else 0.0

    return {
        "error_precision": avg_precision,
        "error_recall": avg_recall,
        "error_f1": avg_f1,
        "num_error_rows": len([m for m in error_masks if sum(m) > 0]),
    }


def evaluate_per_column_model(
    tabular_encoder: nn.Module,
    text_encoder: nn.Module,
    fusion_module: nn.Module,
    column_mlps: nn.ModuleList,
    eval_dataset: CleanDirtyEvaluationDataset,
    device: str,
    device_map: Optional[Dict[str, str]],
) -> Dict[str, float]:
    """
    Evaluate per-column MLPs for error detection.

    Each column has its own MLP that predicts whether the cell value matches the text.
    """
    # Set to eval mode
    tabular_encoder.eval()
    text_encoder.eval()
    fusion_module.eval()
    column_mlps.eval()

    # Compute error mask
    error_masks = compute_error_mask(
        eval_dataset.clean_rows,
        eval_dataset.dirty_rows
    )

    num_cols = len(eval_dataset.dirty_rows[0]) if eval_dataset.dirty_rows else 0

    # Predict on dirty data using per-column MLPs
    all_predictions = []
    all_labels = []

    for row_idx, (dirty_row, text) in enumerate(zip(eval_dataset.dirty_rows, eval_dataset.text_descriptions)):
        tabular_inputs = eval_dataset.tabular_processor.process(dirty_row, row_idx=0)
        text_inputs = eval_dataset.text_processor.process(text)

        # Add batch dimension (filter out non-tensor values like 'raw_text')
        tabular_inputs = {k: v.unsqueeze(0).to(device) for k, v in tabular_inputs.items() if isinstance(v, torch.Tensor)}
        text_inputs = {k: v.unsqueeze(0).to(device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            H_table = tabular_encoder(tabular_inputs)
            H_text = text_encoder(text_inputs)
            H_fused = fusion_module(H_table, H_text)

            # Get predictions from each column's MLP
            row_preds = []
            for col_idx in range(min(num_cols, H_fused.size(1))):
                col_embedding = H_fused[:, col_idx, :].unsqueeze(1)
                logits = column_mlps[col_idx](col_embedding)
                prob = torch.sigmoid(logits.squeeze())
                # Predict error if match probability < 0.5
                pred = 1 if prob.item() < 0.5 else 0
                row_preds.append(pred)

            all_predictions.append(row_preds)
            all_labels.append(error_masks[row_idx])

    # Compute metrics
    tp, fp, fn = 0, 0, 0
    for preds, labels in zip(all_predictions, all_labels):
        for pred, label in zip(preds, labels):
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"阶段B评估结果 - Error Precision: {precision:.4f}, "
          f"Error Recall: {recall:.4f}, Error F1: {f1:.4f}")

    return {
        "error_precision": precision,
        "error_recall": recall,
        "error_f1": f1,
        "num_error_rows": len([m for m in error_masks if sum(m) > 0]),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def run_contrastive_two_stage_experiment(
    exp_cfg: Dict[str, Any],
    device: str,
    device_map: Optional[Dict[str, str]],
    *,
    seed: int,
    config_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    """
    Two-stage contrastive training experiment.

    Stage A: Pretrain tabular and text encoders using direct embedding similarity.
    Stage B: Freeze encoders, train fusion + MLP for binary classification.

    Returns:
        Dict containing training losses and evaluation metrics
    """
    print("\n" + "=" * 80)
    print("实验：Contrastive 两阶段训练（预训练 + 二分类）")
    print("=" * 80)

    # Load data
    clean_rows, text_descriptions, dataset_name = load_data_from_config(
        exp_cfg,
        default_seed=seed,
        config_dir=config_dir,
        project_root=project_root,
    )

    stage_a_cfg = exp_cfg.get("stage_a", {})
    stage_b_cfg = exp_cfg.get("stage_b", {})

    # Instantiate processors
    tabular_processor, text_processor = _instantiate_processors(exp_cfg, config_dir, project_root)

    # Resolve and fix runtime device once to avoid device mismatch
    runtime_device = resolve_runtime_device(device)
    print(f"\n使用设备: {runtime_device}")

    # ============ Stage A: Pretraining (Direct Embedding Similarity) ============
    print("\n阶段A：预训练tabular encoder和text encoder（直接embedding相似度）...")

    # Build encoders for Stage A
    tabular_encoder = build_component(
        stage_a_cfg["tabular_encoder"],
        ENCODER_REGISTRY,
        config_dir,
        project_root,
    )
    text_encoder = build_component(
        stage_a_cfg["text_encoder"],
        ENCODER_REGISTRY,
        config_dir,
        project_root,
    )
    # ============ Stage A: 预计算文本embeddings缓存 ============
    # 检查是否需要缓存（仅当text_encoder被冻结时）
    cached_text_embeddings = None
    text_encoder_params = stage_a_cfg.get("text_encoder", {}).get("params", {})
    is_frozen = text_encoder_params.get("freeze_pretrained", True) or text_encoder_params.get("freeze_base_model", True)

    if is_frozen and hasattr(text_encoder, "forward"):
        # 解析缓存设备
        cache_device = device
        if device_map and "text_encoder" in device_map:
            cache_device = resolve_runtime_device(device_map["text_encoder"], fallback_device=device)

        # 预计算并缓存文本embeddings
        cached_text_embeddings = _precompute_text_embeddings(
            text_encoder,
            text_descriptions,
            text_processor,
            device=cache_device,
            batch_size=stage_a_cfg.get("batch_size", 16),
        )
        print(f"✓ Stage A已启用文本embedding缓存，共缓存 {len(cached_text_embeddings)} 条")
    else:
        print("⚠ Stage A未启用文本embedding缓存（text_encoder未冻结）")

    # Build pretraining dataset
    pretrain_dataset = ContrastiveDataset(
        clean_rows,
        text_descriptions,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
        cached_text_embeddings=cached_text_embeddings,
    )

    batch_size_a = stage_a_cfg.get("batch_size", 16)
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size_a,
        shuffle=True,
        collate_fn=collate_fn_contrastive,
        num_workers=_resolve_num_workers(exp_cfg),
        pin_memory=bool(exp_cfg.get("pin_memory", True)),
        persistent_workers=bool(exp_cfg.get("persistent_workers", True)),
    )
    # Build optimizer for Stage A (only encoders)
    # Use nn.Module to wrap encoders for optimizer
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, tabular_encoder, text_encoder):
            super().__init__()
            self.tabular_encoder = tabular_encoder
            self.text_encoder = text_encoder

    encoder_wrapper = EncoderWrapper(tabular_encoder, text_encoder)
    optimizer_a = build_optimizer(
        encoder_wrapper,
        stage_a_cfg.get("optimizer", {"type": "Adam", "params": {"lr": 1e-4}})
    )

    # Move encoders to device before Stage A training
    tabular_encoder = tabular_encoder.to(runtime_device)
    text_encoder = text_encoder.to(runtime_device)

    # Train Stage A
    num_epochs_a = stage_a_cfg.get("num_epochs", 30)
    temperature = stage_a_cfg.get("temperature", 0.07)
    pretrain_losses = []

    for epoch in range(num_epochs_a):
        epoch_losses = []
        for batch in tqdm(pretrain_dataloader, desc=f"Epoch {epoch+1}/{num_epochs_a} Training", ncols=100):
            loss = train_step_contrastive_pretrain(
                tabular_encoder,
                text_encoder,
                batch,
                optimizer_a,
                device=device,
                temperature=temperature,
                device_map=device_map,
            )
            epoch_losses.append(loss)
        avg_loss = np.mean(epoch_losses)
        pretrain_losses.append(avg_loss)
        print(f"    Epoch {epoch+1}/{num_epochs_a}, Loss: {avg_loss:.4f}")

    # Load eval dataset for Stage A threshold search
    eval_dataset = _build_eval_dataset(exp_cfg, config_dir, project_root)
    if eval_dataset is not None:
        # Use the same processors as training to ensure dimension consistency
        eval_dataset.tabular_processor = tabular_processor
        eval_dataset.text_processor = text_processor
    # ============ Stage B: Per-Column Binary Classification (Frozen Encoders) ============
    print("\n阶段B：逐列二分类训练（冻结编码器，每列独立MLP）...")

    # Get number of columns
    num_cols = len(clean_rows[0]) if clean_rows else 0

    # Build shared fusion module
    fusion_module = build_component(
        stage_b_cfg["fusion_module"],
        FUSION_REGISTRY,
        config_dir,
        project_root,
    )

    # Create per-column MLP detection heads
    detection_head_cfg = stage_b_cfg["detection_head"]
    column_mlps = nn.ModuleList([
        build_component(detection_head_cfg, DETECTION_HEAD_REGISTRY, config_dir, project_root)
        for _ in range(num_cols)
    ])

    # Move components to device
    fusion_module = fusion_module.to(runtime_device)
    column_mlps = column_mlps.to(runtime_device)

    # Freeze encoders
    tabular_encoder.eval()
    text_encoder.eval()
    for param in tabular_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    batch_size_b = stage_b_cfg.get("batch_size", 32)
    num_epochs_b = stage_b_cfg.get("num_epochs", 20)
    binary_losses = []

    # Build optimizer for all column MLPs and fusion
    optimizer_b = build_optimizer(
        nn.Sequential(fusion_module, column_mlps),
        stage_b_cfg.get("optimizer", {"type": "Adam", "params": {"lr": 1e-3}})
    )

    # ============ 简化的缓存策略 ============
    # 分别缓存每行的 row embedding 和 text embedding
    # 总缓存量: 2N (N = num_rows)，而不是 N × M × (1 + neg_ratio)
    print("\n预计算并缓存编码器输出...")

    num_rows = len(clean_rows)
    cached_row_embeddings = {}  # row_idx -> [seq_len, d_model]
    cached_text_embeddings = {}  # row_idx -> [seq_len, d_model] or [d_model]

    # Process rows in batches
    cache_batch_size = 64
    for batch_start in tqdm(range(0, num_rows, cache_batch_size), ncols=120, desc="缓存编码器输出"):
        batch_end = min(batch_start + cache_batch_size, num_rows)

        # Prepare batch data
        batch_tabular_inputs = []
        batch_text_inputs = []
        for row_idx in range(batch_start, batch_end):
            tabular_inputs = tabular_processor.process(clean_rows[row_idx], row_idx=0)
            text_inputs = text_processor.process(text_descriptions[row_idx])
            batch_tabular_inputs.append(tabular_inputs)
            batch_text_inputs.append(text_inputs)

        # Stack to create batch tensors
        if "cell_embeddings" in batch_tabular_inputs[0]:
            tabular_batch = {
                "cell_embeddings": torch.stack([x["cell_embeddings"] for x in batch_tabular_inputs]),
                "row_indices": torch.stack([x["row_indices"] for x in batch_tabular_inputs]),
                "col_indices": torch.stack([x["col_indices"] for x in batch_tabular_inputs]),
            }
        else:
            tabular_batch = batch_tabular_inputs[0]

        if "cached_embedding" in batch_text_inputs[0]:
            text_batch = {"cached_embedding": torch.stack([x["cached_embedding"] for x in batch_text_inputs])}
        else:
            text_batch = {
                "input_ids": torch.stack([x["input_ids"] for x in batch_text_inputs]),
                "attention_mask": torch.stack([x["attention_mask"] for x in batch_text_inputs]),
            }

        # Move to device and forward
        tabular_batch = {k: v.to(runtime_device) for k, v in tabular_batch.items()}
        text_batch = {k: v.to(runtime_device) for k, v in text_batch.items()}

        with torch.no_grad():
            H_table = tabular_encoder(tabular_batch)  # [batch, seq_len, d_model]
            if "cached_embedding" in text_batch:
                H_text = text_batch["cached_embedding"]
            else:
                H_text = text_encoder(text_batch)  # [batch, text_seq_len, d_model]

        # Cache embeddings by row index
        for i, row_idx in enumerate(range(batch_start, batch_end)):
            cached_row_embeddings[row_idx] = H_table[i].cpu()  # [seq_len, d_model]
            cached_text_embeddings[row_idx] = H_text[i].cpu()  # [text_seq_len, d_model]

    print(f"缓存完成！共缓存 {num_rows} 行的 row/text embeddings")

    # ============ 重建 Dataset 使用缓存 ============
    # 构建样本列表：(row_idx, text_idx, col_idx, label)
    # 正样本: row_i + text_i
    # 负样本: row_i + text_j (j != i)
    samples_by_column = {col_idx: [] for col_idx in range(num_cols)}
    negative_ratio = stage_b_cfg.get("negative_ratio", 1.0)

    for col_idx in range(num_cols):
        # 正样本
        for row_idx in range(num_rows):
            samples_by_column[col_idx].append((row_idx, row_idx, col_idx, 1))  # (row_idx, text_idx, col_idx, label)

        # 负样本 (random text mismatch)
        num_negatives = int(num_rows * negative_ratio)
        for _ in range(num_negatives):
            row_idx = random.randint(0, num_rows - 1)
            text_idx = random.choice([i for i in range(num_rows) if i != row_idx])
            samples_by_column[col_idx].append((row_idx, text_idx, col_idx, 0))

    # Train Stage B - iterate through columns
    for epoch in range(num_epochs_b):
        epoch_losses = []

        for col_idx in range(num_cols):
            # Get samples for this column from new structure
            col_samples = samples_by_column[col_idx]
            if not col_samples:
                continue

            # Shuffle samples each epoch
            random.shuffle(col_samples)

            # Simple batching using cached embeddings
            for batch_start in tqdm(range(0, len(col_samples), batch_size_b), ncols=100, desc=f"Epoch {epoch+1}/{num_epochs_b} Col {col_idx+1}/{num_cols}"):
                batch_end = min(batch_start + batch_size_b, len(col_samples))
                batch_samples = col_samples[batch_start:batch_end]

                # Get cached embeddings by row_idx/text_idx
                batch_tabular_embeds = []
                batch_text_embeds = []
                batch_labels = []

                for row_idx, text_idx, _, label in batch_samples:
                    batch_tabular_embeds.append(cached_row_embeddings[row_idx])
                    batch_text_embeds.append(cached_text_embeddings[text_idx])
                    batch_labels.append(torch.tensor([label], dtype=torch.float32))

                # Stack tensors
                batch_tabular_embeds = torch.stack(batch_tabular_embeds)
                batch_text_embeds = torch.stack(batch_text_embeds)
                batch_labels = torch.stack(batch_labels)

                # Move to device
                batch_tabular_embeds = batch_tabular_embeds.to(runtime_device)
                batch_text_embeds = batch_text_embeds.to(runtime_device)
                batch_labels = batch_labels.to(runtime_device)

                # Forward pass through fusion module (only this needs training)
                H_fused = fusion_module(batch_tabular_embeds, batch_text_embeds)  # [batch, seq_len, d_model]

                # Extract fused embedding for this column and pass through column MLP
                col_embedding = H_fused[:, col_idx, :]  # [batch, d_model]
                col_embedding = col_embedding.unsqueeze(1)  # [batch, 1, d_model]
                logits = column_mlps[col_idx](col_embedding)  # [batch, 1, 1]
                logits = logits.squeeze(-1).squeeze(-1)  # [batch]

                # Compute loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch_labels.squeeze(-1)  # [batch] to match logits shape
                )

                optimizer_b.zero_grad()
                loss.backward()
                optimizer_b.step()

                epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        binary_losses.append(avg_loss)
        print(f"阶段B Epoch {epoch+1}/{num_epochs_b}, Loss: {avg_loss:.4f}")

    # ============ Evaluation ============
    if eval_dataset is not None:
        print("\n阶段B评估：针对错误数据计算precision/recall/f1...")
        eval_results = evaluate_per_column_model(
            tabular_encoder, text_encoder, fusion_module, column_mlps,
            eval_dataset, runtime_device, device_map
        )
    else:
        eval_results = {}
        print("未找到评估数据集，跳过阶段B评估")

    return {
        "stage_a_losses": pretrain_losses,
        "stage_b_losses": binary_losses,
        "dataset": dataset_name,
        **eval_results,
    }


def _build_eval_dataset(exp_cfg: Dict[str, Any], config_dir: Path, project_root: Path) -> Optional[CleanDirtyEvaluationDataset]:
    """Helper to build evaluation dataset from config."""
    eval_cfg = exp_cfg.get("evaluation")
    if not eval_cfg:
        # Check if using mock data - if so, generate eval data from mock
        mock_cfg = exp_cfg.get("mock_data")
        if mock_cfg is not None:
            print("\n检测到mock数据，正在生成评估数据集...")
            from .data_loading import MockDataGenerator
            generator = MockDataGenerator(seed=mock_cfg.get("seed", 42))
            dataset_type = mock_cfg.get("type", "employee")
            num_samples = min(mock_cfg.get("num_samples", 50), 20)  # Use fewer samples for eval

            clean_rows, text_descriptions = generator.generate(dataset_type, num_samples)
            # Generate dirty rows by corrupting some cells
            dirty_rows = []
            for row in clean_rows:
                dirty_row = list(row)
                # Randomly corrupt 1-2 cells
                num_cols = len(dirty_row)
                num_corruptions = np.random.randint(1, min(3, num_cols))
                corrupt_indices = np.random.choice(num_cols, num_corruptions, replace=False)

                for idx in corrupt_indices:
                    if isinstance(dirty_row[idx], (int, float)):
                        # Corrupt numeric value
                        if isinstance(dirty_row[idx], int):
                            dirty_row[idx] = int(dirty_row[idx] * np.random.uniform(0.5, 2.0))
                        else:
                            dirty_row[idx] = dirty_row[idx] * np.random.uniform(0.5, 2.0)
                    else:
                        # Corrupt string value
                        dirty_row[idx] = f"CORRUPTED_{dirty_row[idx]}"

                dirty_rows.append(dirty_row)

            return CleanDirtyEvaluationDataset(
                clean_rows=clean_rows,
                dirty_rows=dirty_rows,
                text_descriptions=text_descriptions,
            )
        return None

    try:
        clean_data_path = _resolve_path_like(eval_cfg["clean_data_path"], config_dir, project_root)
        dirty_data_path = _resolve_path_like(eval_cfg["dirty_data_path"], config_dir, project_root)

        clean_rows, clean_text_descriptions, _ = load_csv_dataset(clean_data_path)
        dirty_rows, dirty_text_descriptions, _ = load_csv_dataset(dirty_data_path)

        # Use text descriptions from clean data
        text_descriptions = clean_text_descriptions

        return CleanDirtyEvaluationDataset(
            clean_rows=clean_rows,
            dirty_rows=dirty_rows,
            text_descriptions=text_descriptions,
        )
    except Exception as e:
        print(f"警告: 无法构建评估数据集: {e}")
        return None


__all__ = [
    "run_corruption_experiment",
    "run_contrastive_two_stage_experiment",
]

