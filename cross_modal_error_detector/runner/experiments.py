from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..datasets import CleanDirtyEvaluationDataset, ContrastiveDataset, CorruptionBasedDataset
from ..model import CrossModalErrorDetector
from ..training import (
    collate_fn_contrastive,
    collate_fn_corruption,
    train_step_contrastive,
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


def _prepare_eval_loader(
    dataset: Dataset,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_corruption,
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
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
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
    device: str,
    device_map: Optional[Dict[str, str]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    eval_loader = _prepare_eval_loader(dataset, batch_size)

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

    train_dataset = CorruptionBasedDataset(
        clean_rows=clean_rows,
        text_descriptions=text_descriptions,
        corruption_prob=corruption_prob,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_corruption,
    )

    eval_dataset, eval_dataset_name = _build_eval_dataset_from_config(
        exp_cfg,
        tabular_processor,
        text_processor,
        config_dir=config_dir,
        project_root=project_root,
    )
    dataset_for_eval = eval_dataset or train_dataset
    reported_dataset_name = eval_dataset_name or dataset_name

    tabular_encoder = build_component(exp_cfg["tabular_encoder"], ENCODER_REGISTRY, config_dir, project_root)
    text_encoder = build_component(exp_cfg["text_encoder"], ENCODER_REGISTRY, config_dir, project_root)
    fusion_module = build_component(exp_cfg["fusion_module"], FUSION_REGISTRY, config_dir, project_root)
    detection_head = build_component(
        exp_cfg["detection_head"],
        DETECTION_HEAD_REGISTRY,
        config_dir,
        project_root,
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

    eval_metrics = _evaluate_corruption_model(
        model,
        dataset_for_eval,
        batch_size=batch_size,
        device=device,
        device_map=device_map,
    )
    print(
        "\n  ✓ 脏值 Precision / Recall / F1: "
        f"{eval_metrics['precision']:.2%} / {eval_metrics['recall']:.2%} / {eval_metrics['f1']:.2%}"
    )
    print(f"  ✓ Overall Accuracy: {eval_metrics['accuracy']:.2%}")

    return {
        "train_losses": train_losses,
        "accuracy": eval_metrics["accuracy"],
        "metrics": eval_metrics,
        "num_params": num_params,
        "dataset": reported_dataset_name,
        "num_samples": num_samples,
    }


def run_contrastive_experiment(
    exp_cfg: Dict[str, Any],
    device: str,
    device_map: Optional[Dict[str, str]],
    *,
    seed: int,
    config_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("实验：Contrastive 训练（对比学习）")
    print("=" * 80)

    clean_rows, text_descriptions, dataset_name = load_data_from_config(
        exp_cfg,
        default_seed=seed,
        config_dir=config_dir,
        project_root=project_root,
    )

    batch_size = exp_cfg.get("batch_size", 8)
    num_epochs = exp_cfg.get("num_epochs", 10)
    temperature = exp_cfg.get("temperature", 0.07)

    tabular_processor, text_processor = _instantiate_processors(exp_cfg, config_dir, project_root)

    dataset = ContrastiveDataset(
        clean_rows=clean_rows,
        text_descriptions=text_descriptions,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_contrastive,
    )

    tabular_encoder = build_component(exp_cfg["tabular_encoder"], ENCODER_REGISTRY, config_dir, project_root)
    text_encoder = build_component(exp_cfg["text_encoder"], ENCODER_REGISTRY, config_dir, project_root)
    fusion_module = build_component(exp_cfg["fusion_module"], FUSION_REGISTRY, config_dir, project_root)
    detection_head = build_component(
        exp_cfg["detection_head"],
        DETECTION_HEAD_REGISTRY,
        config_dir,
        project_root,
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

    train_losses: List[float] = []
    for epoch in range(num_epochs):
        epoch_losses: List[float] = []
        for batch in dataloader:
            loss = train_step_contrastive(
                model,
                batch,
                optimizer,
                device,
                temperature=temperature,
                device_map=device_map,
            )
            epoch_losses.append(loss)
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_losses.append(avg_loss)
        print(f"  Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    model.eval()
    accuracy = 0.0
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        tabular_inputs, text_inputs = sample_batch
        batch_size_eval = tabular_inputs["cell_embeddings"].shape[0]
        score_device = _resolve_runtime_device(device, device_map, "detection_head")
        scores_matrix = torch.zeros(batch_size_eval, batch_size_eval, device=score_device)

        for i in range(batch_size_eval):
            tab_i = {
                "cell_embeddings": tabular_inputs["cell_embeddings"][i : i + 1],
                "row_indices": tabular_inputs["row_indices"][i : i + 1],
                "col_indices": tabular_inputs["col_indices"][i : i + 1],
            }
            for j in range(batch_size_eval):
                txt_j = {
                    "input_ids": text_inputs["input_ids"][j : j + 1],
                    "attention_mask": text_inputs["attention_mask"][j : j + 1],
                }
                score = model(tab_i, txt_j)
                scores_matrix[i, j] = score.squeeze().to(score_device)

        scores_matrix_cpu = scores_matrix.detach().cpu()
        diagonal_scores = torch.diagonal(scores_matrix_cpu)
        correct = sum(
            bool((scores_matrix_cpu[i, i] == scores_matrix_cpu[i].max()).item())
            for i in range(batch_size_eval)
        )
        accuracy = correct / batch_size_eval if batch_size_eval else 0.0

    print(f"\n  ✓ 匹配准确率: {accuracy:.2%}")
    print(f"  ✓ 正样本平均得分: {diagonal_scores.mean().item():.4f}")

    return {
        "train_losses": train_losses,
        "accuracy": accuracy,
        "num_params": num_params,
        "dataset": dataset_name,
    }


def run_ablation_experiment(
    exp_cfg: Dict[str, Any],
    device: str,
    device_map: Optional[Dict[str, str]],
    *,
    seed: int,
    config_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("实验：消融实验（Ablation Study）")
    print("=" * 80)

    clean_rows, text_descriptions, dataset_name = load_data_from_config(
        exp_cfg,
        default_seed=seed,
        config_dir=config_dir,
        project_root=project_root,
    )

    corruption_prob = exp_cfg.get("corruption_prob", 0.3)
    batch_size = exp_cfg.get("batch_size", 8)
    num_epochs = exp_cfg.get("num_epochs", 5)

    variants = exp_cfg.get("variants", [])
    if not variants:
        print("  ⚠️ 未提供任何变体配置，跳过消融实验。")
        return {"dataset": dataset_name, "results": {}}

    tabular_processor, text_processor = _instantiate_processors(exp_cfg, config_dir, project_root)

    train_dataset = CorruptionBasedDataset(
        clean_rows=clean_rows,
        text_descriptions=text_descriptions,
        corruption_prob=corruption_prob,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_corruption,
    )

    eval_dataset, eval_dataset_name = _build_eval_dataset_from_config(
        exp_cfg,
        tabular_processor,
        text_processor,
        config_dir=config_dir,
        project_root=project_root,
    )
    dataset_for_eval = eval_dataset or train_dataset

    results: Dict[str, Dict[str, Any]] = {}
    text_encoder_cfg = exp_cfg["text_encoder"]
    detection_head_cfg = exp_cfg["detection_head"]
    optimizer_cfg = exp_cfg.get("optimizer", {})

    for variant in variants:
        name = variant.get("name", "Unnamed Variant")
        print("\n" + "-" * 80)
        print(f"测试配置: {name}")
        print("-" * 80)

        tabular_encoder = build_component(
            variant["tabular_encoder"],
            ENCODER_REGISTRY,
            config_dir,
            project_root,
        )
        fusion_module = build_component(
            variant["fusion_module"],
            FUSION_REGISTRY,
            config_dir,
            project_root,
        )
        text_encoder = build_component(
            text_encoder_cfg,
            ENCODER_REGISTRY,
            config_dir,
            project_root,
        )
        detection_head = build_component(
            detection_head_cfg,
            DETECTION_HEAD_REGISTRY,
            config_dir,
            project_root,
        )

        model = CrossModalErrorDetector(
            tabular_encoder=tabular_encoder,
            text_encoder=text_encoder,
            fusion_module=fusion_module,
            detection_head=detection_head,
            device_map=device_map,
            default_device=device,
        )

        optimizer = build_optimizer(model, optimizer_cfg)

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

        eval_metrics = _evaluate_corruption_model(
            model,
            dataset_for_eval,
            batch_size=batch_size,
            device=device,
            device_map=device_map,
        )

        results[name] = {
            "train_losses": train_losses,
            "accuracy": eval_metrics["accuracy"],
            "metrics": eval_metrics,
        }
        print(
            "  ✓ 脏值 Precision / Recall / F1: "
            f"{eval_metrics['precision']:.2%} / {eval_metrics['recall']:.2%} / {eval_metrics['f1']:.2%}"
        )
        print(f"  ✓ Overall Accuracy: {eval_metrics['accuracy']:.2%}")

    reported_dataset_name = eval_dataset_name or dataset_name

    return {
        "dataset": reported_dataset_name,
        "results": results,
    }


__all__ = [
    "run_corruption_experiment",
    "run_contrastive_experiment",
    "run_ablation_experiment",
]


