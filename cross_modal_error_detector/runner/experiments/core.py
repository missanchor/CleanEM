"""
Core experiment functions for cross-modal error detection.
"""
from __future__ import annotations

import os
import random
import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import all the new modules
from .column_profiling import (
    _build_column_profiles,
    _constraint_violation_score,
    _corrupt_row_cell,
    _corrupt_value,
    _make_typo,
)
from .data_utils import (
    _attach_processor_params,
    _build_eval_dataset_from_config,
    _derive_loader_runtime,
    _prepare_eval_loader,
)
from ..data_loading import load_data_from_config
from .embeddings import _precompute_text_embeddings
from .error_analysis import evaluate_per_column_model
from .eval_dataset_builder import _build_eval_dataset
from .evaluation import _evaluate_corruption_model
from .negative_sampling import (
    _generate_negative_pairs,
    _parse_negative_sampling_config,
    _build_semantic_neighbor_map,
    _build_cell_value_semantic_neighbor_map,
)
from ..runtime import _move_batch_to_device, _resolve_runtime_device
from ...utils.device import resolve_runtime_device


def _process_texts_with_batch_support(
    text_processor,
    texts: List[str],
) -> List[Dict[str, Any]]:
    """
    Prefer text_processor.process_batch for efficiency, fallback to per-text processing.
    """
    if not texts:
        return []

    if hasattr(text_processor, "process_batch"):
        batch_outputs = text_processor.process_batch(texts)
        if isinstance(batch_outputs, list):
            return batch_outputs
        if isinstance(batch_outputs, dict):
            processed: List[Dict[str, Any]] = []
            for idx in range(len(texts)):
                processed.append(
                    {
                        key: (value[idx] if isinstance(value, torch.Tensor) else value)
                        for key, value in batch_outputs.items()
                    }
                )
            return processed

    return [text_processor.process(text) for text in texts]


def _prepare_column_label_encoders(
    rows: List[List[Any]],
    *,
    numeric_ratio_threshold: float,
    max_rows_for_type_inference: int,
    max_vocab_size: Optional[int],
    unknown_token: str = "<UNK>",
) -> Tuple[List[bool], List[Dict[str, int]], List[int]]:
    """
    Infer column types and build per-column categorical vocabularies.
    Returns:
        col_is_numeric: list of booleans indicating numeric columns
        column_value_to_id: per-column mapping from token -> id (categorical columns)
        column_vocab_sizes: per-column vocab sizes (>=1)
    """
    if not rows:
        return [], [], []

    num_cols = len(rows[0])
    numeric_counts = [0] * num_cols
    total_counts = [0] * num_cols
    limit = min(len(rows), max(1, max_rows_for_type_inference))

    for row_idx in range(limit):
        row = rows[row_idx]
        if len(row) != num_cols:
            raise ValueError("dirty_rows must be rectangular (all rows same number of columns).")
        for col_idx, value in enumerate(row):
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            total_counts[col_idx] += 1
            is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
            if is_num:
                numeric_counts[col_idx] += 1

    col_is_numeric: List[bool] = []
    for col_idx in range(num_cols):
        denom = total_counts[col_idx]
        ratio = (numeric_counts[col_idx] / denom) if denom > 0 else 0.0
        col_is_numeric.append(ratio >= numeric_ratio_threshold)

    column_value_to_id: List[Dict[str, int]] = []
    column_vocab_sizes: List[int] = []

    for col_idx in range(num_cols):
        mapping: OrderedDict[str, int] = OrderedDict()
        if unknown_token is not None:
            mapping.setdefault(unknown_token, 0)

        if not col_is_numeric[col_idx]:
            for row in rows:
                if col_idx >= len(row):
                    continue
                token = "" if row[col_idx] is None else str(row[col_idx])
                if token in mapping:
                    continue
                if max_vocab_size is not None and len(mapping) >= max_vocab_size:
                    continue
                mapping[token] = len(mapping)

        if not mapping:
            mapping = OrderedDict()
            if unknown_token is not None:
                mapping[unknown_token] = 0

        column_value_to_id.append(mapping)
        column_vocab_sizes.append(max(1, len(mapping)))

    return col_is_numeric, column_value_to_id, column_vocab_sizes


def run_corruption_experiment(
    exp_cfg: Dict[str, Any],
    device: str,
    device_map: Optional[Dict[str, str]],
    *,
    seed: int,
    config_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    """
    Run corruption-based training experiment.

    Args:
        exp_cfg: Experiment configuration
        device: Device to use
        device_map: Device mapping for model components
        seed: Random seed
        config_dir: Configuration directory
        project_root: Project root directory

    Returns:
        Dictionary of experiment results
    """
    print("\n" + "=" * 80)
    print("实验：Corruption-based 训练（破坏-重建）")
    print("=" * 80)

    clean_rows, text_descriptions, dataset_name, column_names = load_data_from_config(
        exp_cfg,
        default_seed=seed,
        config_dir=config_dir,
        project_root=project_root,
    )

    num_samples = len(clean_rows)
    corruption_prob = exp_cfg.get("corruption_prob", 0.3)
    batch_size = exp_cfg.get("batch_size", 8)
    num_epochs = exp_cfg.get("num_epochs", 10)

    # Import here to avoid circular dependency
    from ..configuration import _instantiate_processors
    from ..components import ENCODER_REGISTRY, FUSION_REGISTRY, DETECTION_HEAD_REGISTRY, build_component
    from ...datasets import CorruptionBasedDataset
    from ...fusion import CrossAttentionFusion, SimpleConcatFusion, SingleColumnFusion
    from ...model import CrossModalErrorDetector
    from ...training import collate_fn_corruption, train_step_corruption

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

        # Check if we should use last token embeddings (like in training)
        use_last_token = text_encoder_params.get("use_text_output_last_token_embedding", False)
        # Check if we should generate response first (NEW FEATURE)
        generate_response = text_encoder_params.get("generate_response", False)
        max_new_tokens = text_encoder_params.get("max_new_tokens", 10)

        if generate_response:
            # NEW: Use generate_and_encode for each prompt
            print("  → 使用生成式模型生成响应并编码...")
            cached_embeddings = {}
            for idx, prompt in enumerate(tqdm(text_descriptions, desc="生成响应并缓存文本 Embeddings", ncols=100)):
                try:
                    last_token_emb = text_encoder.generate_and_encode(
                        prompt=prompt,
                        text_processor=text_processor,
                        max_new_tokens=max_new_tokens,
                        device=cache_device,
                    )
                    cached_embeddings[idx] = last_token_emb
                except Exception as e:
                    print(f"  ⚠ 生成响应失败，使用编码方式: {e}")
                    # Fallback to encoding
                    text_inputs = text_processor.process(prompt)
                    text_inputs = {k: v.to(cache_device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}
                    with torch.no_grad():
                        H_text = text_encoder(text_inputs)
                        if H_text.dim() == 3:
                            last_token_emb = H_text[:, -1, :].squeeze(0).unsqueeze(0).cpu()  # [1, d_model]
                        else:
                            last_token_emb = H_text.unsqueeze(0).cpu()  # [1, d_model]
                        cached_embeddings[idx] = last_token_emb
        else:
            # Original behavior: use _precompute_text_embeddings
            cached_embeddings = _precompute_text_embeddings(
                text_encoder,
                text_descriptions,
                text_processor,
                device=cache_device,
                batch_size=batch_size,
                use_text_output_last_token_embedding=use_last_token,
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

    # Import here
    from ..components import build_optimizer

    optimizer = build_optimizer(model, exp_cfg.get("optimizer", {}))
    _attach_processor_params(optimizer, tabular_processor)

    train_dataset = CorruptionBasedDataset(
        clean_rows=clean_rows,
        text_descriptions=text_descriptions,
        corruption_prob=corruption_prob,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
        cached_text_embeddings=cached_embeddings,
        column_names=column_names,
    )
    num_workers, pin_memory, persistent_workers = _derive_loader_runtime(exp_cfg)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_corruption,
        num_workers=0,  # Set to 0 to avoid multiprocessing serialization issues
        pin_memory=pin_memory,
        persistent_workers=False,
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
                tabular_processor,
                device,
                device_map=device_map,
                column_names=column_names,
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


def run_mcm_experiment(
    exp_cfg: Dict[str, Any],
    device: str,
    device_map: Optional[Dict[str, str]],
    *,
    seed: int,
    config_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    """
    Masked Cell Modeling (MCM) experiment.

    Single-stage self-supervised reconstruction training using masked cell modeling.
    Trains models to reconstruct masked cells, then uses reconstruction loss as error score.

    Args:
        exp_cfg: Experiment configuration
        device: Device to use
        device_map: Device mapping for model components
        seed: Random seed
        config_dir: Configuration directory
        project_root: Project root directory

    Returns:
        Dict containing training losses, scoring results, and evaluation metrics
    """
    print("\n" + "=" * 80)
    print("实验：MCM 无监督重构（结构 + 语义 + 融合）")
    print("=" * 80)

    # Load data
    dirty_rows, text_descriptions, dataset_name, column_names = load_data_from_config(
        exp_cfg,
        default_seed=seed,
        config_dir=config_dir,
        project_root=project_root,
    )

    # Instantiate processors
    from ..configuration import _instantiate_processors
    tabular_processor, text_processor = _instantiate_processors(exp_cfg, config_dir, project_root)

    # Resolve and fix runtime device once to avoid device mismatch
    runtime_device = resolve_runtime_device(device)
    print(f"\n使用设备: {runtime_device}")

    # Import MCM-specific components
    from ..components import ENCODER_REGISTRY, FUSION_REGISTRY, DETECTION_HEAD_REGISTRY, build_component, build_optimizer
    from ...datasets import MaskedCellModelingDataset
    from ...training import collate_fn_mcm, train_step_mcm
    from ...fusion import CrossAttentionFusion, SimpleConcatFusion

    mcm_cfg: Dict[str, Any] = exp_cfg.get("mcm", {})
    batch_size = int(mcm_cfg.get("batch_size", exp_cfg.get("batch_size", 32)))
    num_epochs = int(mcm_cfg.get("num_epochs", exp_cfg.get("num_epochs", 20)))
    mask_ratio = float(mcm_cfg.get("mask_ratio", 0.2))
    num_masked_cells = mcm_cfg.get("num_masked_cells")
    numeric_ratio_threshold = float(mcm_cfg.get("numeric_ratio_threshold", 0.9))
    max_rows_for_type_inference = int(mcm_cfg.get("max_rows_for_type_inference", 2000))
    unknown_token = str(mcm_cfg.get("unknown_token", "<UNK>"))
    w_cat = float(mcm_cfg.get("w_cat", mcm_cfg.get("loss_w_cat", 1.0)))
    w_num = float(mcm_cfg.get("w_num", mcm_cfg.get("loss_w_num", 1.0)))

    # New config: direct comparison for error detection
    use_direct_comparison = bool(mcm_cfg.get("use_direct_comparison", False))
    numeric_tolerance = float(mcm_cfg.get("numeric_tolerance", 0.1))

    # Resolve component configs (prefer mcm.* then top-level)
    tabular_encoder_cfg = mcm_cfg.get("tabular_encoder") or exp_cfg.get("tabular_encoder")
    text_encoder_cfg = mcm_cfg.get("text_encoder") or exp_cfg.get("text_encoder")
    fusion_cfg = mcm_cfg.get("fusion_module") or exp_cfg.get("fusion_module")
    head_cfg = (
        mcm_cfg.get("detection_head")
        or mcm_cfg.get("reconstruction_head")
        or exp_cfg.get("detection_head")
    )

    if tabular_encoder_cfg is None or text_encoder_cfg is None or fusion_cfg is None:
        raise ValueError(
            "MCM experiment requires tabular_encoder, text_encoder, and fusion_module configs "
            "(provide under mcm.* or at top-level)."
        )

    tabular_encoder = build_component(tabular_encoder_cfg, ENCODER_REGISTRY, config_dir, project_root)
    text_encoder = build_component(text_encoder_cfg, ENCODER_REGISTRY, config_dir, project_root)

    # Get number of columns for per-column reconstruction heads and fusion modules
    num_cols = len(column_names) if column_names is not None else (len(dirty_rows[0]) if dirty_rows else 0)

    # Build fusion module with per-column support
    fusion_params = fusion_cfg.get("params", {})
    use_per_column_fusion = fusion_params.pop("per_column", True)  # Default to per-column fusion

    if use_per_column_fusion:
        # Create per-column fusion modules
        fusion_type = fusion_cfg["type"]
        d_model = fusion_params.get("d_model", 32)

        fusion_modules = nn.ModuleList()
        for _ in range(num_cols):
            if fusion_type == "CrossAttentionFusion":
                module = CrossAttentionFusion(
                    d_model=d_model,
                    **{k: v for k, v in fusion_params.items() if k != "d_model"}
                )
            elif fusion_type == "SimpleConcatFusion":
                module = SimpleConcatFusion(d_model=d_model)
            else:
                module = build_component(fusion_cfg, FUSION_REGISTRY, config_dir, project_root)
            fusion_modules.append(module)

        fusion_module = fusion_modules
        print(f"✓ 启用MCM per-column fusion，为{num_cols}列创建独立的fusion模块")
    else:
        # Shared fusion module
        fusion_module = build_component(fusion_cfg, FUSION_REGISTRY, config_dir, project_root)
        print("✓ 使用MCM shared fusion module")
    
    # Default reconstruction head config if not provided
    if head_cfg is None:
        head_cfg = {
            "type": "TabularReconstructionHead",
            "params": {
                "d_model": int(getattr(tabular_encoder, "d_model", 32)),
                "vocab_size": int(getattr(tabular_processor, "vocab_size", 10000)),
            },
        }

    default_vocab_size = int(getattr(tabular_processor, "vocab_size", 10000))
    head_params = head_cfg.get("params", {}) if head_cfg else {}
    max_vocab_size = int(head_params.get("vocab_size", default_vocab_size))

    # Get column types from config first (all columns must be configured)
    column_types = exp_cfg.get("column_types", {})

    # Check that all columns are configured
    if column_names:
        for col_name in column_names:
            if col_name not in column_types:
                raise ValueError(f"Column '{col_name}' not found in column_types configuration. "
                               f"All columns must be explicitly configured in the 'column_types' section.")

    if dirty_rows:
        # Use config-based column types
        col_is_numeric = [False] * num_cols
        for col_idx in range(num_cols):
            col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"
            if column_types and col_name in column_types:
                col_type = column_types[col_name].lower()
                col_is_numeric[col_idx] = col_type in ["numeric", "numerical", "number", "int", "float"]
            else:
                # This should not happen due to the check above, but just in case
                col_is_numeric[col_idx] = False

        # Prepare vocabularies for categorical columns
        column_value_to_id = []
        column_vocab_sizes = []

        for col_idx in range(num_cols):
            mapping: OrderedDict[str, int] = OrderedDict()
            if unknown_token is not None:
                mapping.setdefault(unknown_token, 0)

            # Only build vocab for categorical columns
            if not col_is_numeric[col_idx]:
                for row in dirty_rows:
                    if col_idx >= len(row):
                        continue
                    token = "" if row[col_idx] is None else str(row[col_idx])
                    if token in mapping:
                        continue
                    if max_vocab_size is not None and len(mapping) >= max_vocab_size:
                        continue
                    mapping[token] = len(mapping)

            if not mapping:
                mapping = OrderedDict()
                if unknown_token is not None:
                    mapping[unknown_token] = 0

            column_value_to_id.append(mapping)
            column_vocab_sizes.append(max(1, len(mapping)))
    else:
        col_is_numeric = [False] * num_cols
        column_value_to_id = [
            OrderedDict({unknown_token: 0}) for _ in range(num_cols)
        ]
        column_vocab_sizes = [1] * num_cols

    # Ensure lengths align with num_cols
    if len(col_is_numeric) < num_cols:
        col_is_numeric.extend([False] * (num_cols - len(col_is_numeric)))
    if len(column_value_to_id) < num_cols:
        pad_count = num_cols - len(column_value_to_id)
        column_value_to_id.extend([OrderedDict({unknown_token: 0}) for _ in range(pad_count)])
    if len(column_vocab_sizes) < num_cols:
        column_vocab_sizes.extend([1] * (num_cols - len(column_vocab_sizes)))

    # Register per-column vocabularies with TabularProcessor (no hash collision)
    tabular_processor.register_column_vocabularies(
        column_names=column_names,
        column_value_to_id=[dict(m) for m in column_value_to_id],
        col_is_numeric=col_is_numeric,
    )

    # Create per-column reconstruction heads (each column has its own classifier)
    recon_heads_list: List[nn.Module] = []
    for col_idx in range(num_cols):
        per_col_cfg = copy.deepcopy(head_cfg)
        if per_col_cfg.get("type") == "TabularReconstructionHead":
            per_params = copy.deepcopy(per_col_cfg.get("params", {}))
            vocab_override = column_vocab_sizes[col_idx] if col_idx < len(column_vocab_sizes) else max_vocab_size
            per_params["vocab_size"] = int(max(1, vocab_override))
            per_col_cfg["params"] = per_params
        recon_head = build_component(per_col_cfg, DETECTION_HEAD_REGISTRY, config_dir, project_root)
        recon_heads_list.append(recon_head)
    recon_heads = nn.ModuleList(recon_heads_list)
    print(f"✓ 为{num_cols}列创建了独立的重构头（每列一个分类器）")

    # Freeze text encoder by default for efficiency; enable caching when frozen.
    cached_text_embeddings: Dict[int, torch.Tensor] = {}
    text_encoder_params = (text_encoder_cfg or {}).get("params", {})
    is_frozen = bool(text_encoder_params.get("freeze_pretrained", True) or text_encoder_params.get("freeze_base_model", True))
    if is_frozen:
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    if is_frozen:
        # Check if we should use last token embeddings (like in training)
        use_last_token = mcm_cfg.get("use_text_output_last_token_embedding", False)
        # Check if we should generate response first (NEW FEATURE)
        generate_response = mcm_cfg.get("generate_response", False)
        max_new_tokens = mcm_cfg.get("max_new_tokens", 10)
        generation_row_batch_size = max(1, int(mcm_cfg.get("generation_row_batch_size", 1)))
        generation_backend = mcm_cfg.get("generation_backend", "hf")
        generation_prompt_batch_size = int(mcm_cfg.get("generation_prompt_batch_size", 256))
        vllm_model = mcm_cfg.get("vllm_model") or (text_encoder_params.get("model_name_or_path") if isinstance(text_encoder_params, dict) else None)
        cache_path = mcm_cfg.get("text_embedding_cache_path")
        force_recompute_cache = bool(mcm_cfg.get("force_recompute_text_embedding_cache", False))
        use_multiprocessing = bool(mcm_cfg.get("use_multiprocessing", False))
        num_workers = mcm_cfg.get("generation_num_workers")
        if num_workers is not None:
            num_workers = int(num_workers)
        # Use the intelligent precompute method from text_encoder
        cached_text_embeddings = text_encoder.precompute_embeddings_for_mcm(
            dirty_rows=dirty_rows,
            column_names=column_names,
            text_descriptions=text_descriptions,
            text_processor=text_processor,
            device=runtime_device,
            batch_size=batch_size,
            use_last_token=use_last_token,
            generate_response=generate_response,
            max_new_tokens=max_new_tokens,
            generation_row_batch_size=generation_row_batch_size,
            generation_backend=generation_backend,
            generation_prompt_batch_size=generation_prompt_batch_size,
            vllm_model=vllm_model,
            cache_path=cache_path,
            dataset_name=dataset_name,
            force_recompute_cache=force_recompute_cache,
            use_multiprocessing=use_multiprocessing,
            num_workers=num_workers,
        )
        print(f"✓ 已启用文本embedding缓存，共缓存 {len(cached_text_embeddings)} 条")

    # Trainable mask embedding (in TabularProcessor's input space)
    d_cell = int(getattr(tabular_processor, "d_cell", 8))
    mask_embedding = nn.Parameter(torch.zeros(d_cell, dtype=torch.float32))
    nn.init.normal_(mask_embedding, mean=0.0, std=0.02)

    # Wrap trainables for optimizer creation
    class _MCMTrainable(nn.Module):
        def __init__(self):
            super().__init__()
            self.tabular_processor = tabular_processor
            self.tabular_encoder = tabular_encoder
            self.fusion_module = fusion_module
            self.recon_heads = recon_heads
            self.mask_embedding = mask_embedding

    all_modules = _MCMTrainable().to(runtime_device)

    optimizer = build_optimizer(
        all_modules,
        mcm_cfg.get("optimizer", exp_cfg.get("optimizer", {"type": "Adam", "params": {"lr": 1e-3}})),
    )

    # Dataset / loader
    train_dataset = MaskedCellModelingDataset(
        dirty_rows,
        text_descriptions,
        mask_ratio=mask_ratio,
        num_masked_cells=num_masked_cells,
        numeric_ratio_threshold=numeric_ratio_threshold,
        max_rows_for_type_inference=max_rows_for_type_inference,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
        cached_text_embeddings=cached_text_embeddings,
        column_names=column_names,
        seed=int(mcm_cfg.get("mask_seed", seed)),
        col_is_numeric=col_is_numeric[:num_cols],
        column_value_to_id=[dict(m) for m in column_value_to_id[:num_cols]],
        unknown_token=unknown_token,
    )

    num_workers, pin_memory, persistent_workers = _derive_loader_runtime(exp_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_mcm,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # 逐列训练：外层遍历列，内层遍历epochs和batches
    num_cols = len(column_names) if column_names is not None else (len(dirty_rows[0]) if dirty_rows else 0)

    # 为每列创建独立的数据集
    per_column_datasets = []
    for col_idx in range(num_cols):
        col_dataset = MaskedCellModelingDataset(
            dirty_rows,
            text_descriptions,
            mask_ratio=mask_ratio,
            num_masked_cells=num_masked_cells,
            numeric_ratio_threshold=numeric_ratio_threshold,
            max_rows_for_type_inference=max_rows_for_type_inference,
            tabular_processor=tabular_processor,
            text_processor=text_processor,
            cached_text_embeddings=cached_text_embeddings,
            column_names=column_names,
            col_is_numeric=col_is_numeric[:num_cols],
            column_value_to_id=[dict(m) for m in column_value_to_id[:num_cols]],
            unknown_token=unknown_token,
            seed=int(mcm_cfg.get("mask_seed", seed)),
            fixed_mask_column=col_idx,  # 设置固定mask列
        )
        per_column_datasets.append(col_dataset)

    # 训练每列
    train_losses: List[float] = []

    # 显示总体列进度
    print(f"\n开始逐列训练 ({num_cols} 列，每列 {num_epochs} epochs)...")
    print("=" * 80)

    for col_idx in tqdm(range(num_cols), desc="训练列进度", ncols=100):
        col_dataset = per_column_datasets[col_idx]
        col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"

        # 为该列创建独立的DataLoader
        col_loader = DataLoader(
            col_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_mcm,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        print(f"\n📊 Column {col_idx+1}/{num_cols} ({col_name}):")
        print(f"  Dataset size: {len(col_dataset)} rows, Batch size: {batch_size}")

        # 训练该列的所有epochs
        col_epoch_losses: List[float] = []

        # 简单的epoch进度显示
        for epoch in range(num_epochs):
            epoch_losses: List[float] = []
            epoch_cat: List[float] = []
            epoch_num: List[float] = []

            for batch_idx, batch in enumerate(col_loader):
                stats = train_step_mcm(
                    tabular_encoder,
                    text_encoder,
                    fusion_module,
                    recon_heads,
                    batch,
                    optimizer,
                    tabular_processor,
                    mask_embedding,
                    device=runtime_device,
                    device_map=device_map,
                    column_names=column_names,
                    w_cat=w_cat,
                    w_num=w_num,
                )
                epoch_losses.append(stats["loss"])
                epoch_cat.append(stats["loss_cat"])
                epoch_num.append(stats["loss_num"])

            avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            col_epoch_losses.append(avg)

            # 打印epoch结果
            # 前3个epoch详细显示，中间简化，最后几个详细显示
            if epoch < 3 or epoch >= num_epochs - 3 or epoch % max(1, num_epochs // 5) == 0:
                print(f"    ✅ Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={avg:.4f} (CE={float(np.mean(epoch_cat)) if epoch_cat else 0.0:.4f}, "
                      f"MSE={float(np.mean(epoch_num)) if epoch_num else 0.0:.4f})")
            else:
                # 简化显示：只显示进度和总损失
                print(f"    Epoch {epoch+1}/{num_epochs}: Loss={avg:.4f}")

        # 记录该列的平均损失
        col_avg_loss = float(np.mean(col_epoch_losses)) if col_epoch_losses else 0.0
        train_losses.append(col_avg_loss)

        # 完成该列训练
        print(f"\n  ✅ Column {col_idx+1}/{num_cols} ({col_name}) Complete - Avg Loss: {col_avg_loss:.4f}")
        print("=" * 80)

    # 计算总体平均损失
    overall_avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
    print(f"\n✓ MCM Per-Column Training Complete - Overall Avg Loss: {overall_avg_loss:.4f}")

    # Continue with scoring and evaluation
    # Get column types from config (if specified)
    column_types = exp_cfg.get("column_types", {})

    return _run_mcm_scoring_and_evaluation(
        exp_cfg, dirty_rows, text_descriptions, dataset_name, column_names,
        tabular_processor, text_processor, tabular_encoder, text_encoder,
        fusion_module, recon_heads, mask_embedding, train_dataset,
        cached_text_embeddings, train_losses, runtime_device, device_map,
        config_dir, project_root, mcm_cfg, batch_size,
        use_direct_comparison, numeric_tolerance, column_types
    )


def _run_mcm_scoring_and_evaluation(
    exp_cfg: Dict[str, Any],
    dirty_rows: List[List[Any]],
    text_descriptions: List[str],
    dataset_name: str,
    column_names: Optional[List[str]],
    tabular_processor,
    text_processor,
    tabular_encoder,
    text_encoder,
    fusion_module,
    recon_heads: nn.ModuleList,
    mask_embedding: nn.Parameter,
    train_dataset,
    cached_text_embeddings: Dict[int, torch.Tensor],
    train_losses: List[float],
    runtime_device: str,
    device_map: Optional[Dict[str, str]],
    config_dir: Path,
    project_root: Path,
    mcm_cfg: Dict[str, Any],
    batch_size: int,
    use_direct_comparison: bool,
    numeric_tolerance: float,
    column_types: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Helper function for MCM scoring and evaluation."""
    from ..runtime import _move_batch_to_device
    from ...training import build_tabular_inputs_from_rows
    
    # Lightweight scoring: per-column mean/std + global top-k cells (mask each column once)
    score_cfg: Dict[str, Any] = mcm_cfg.get("scoring", {})
    score_batch_size = int(score_cfg.get("batch_size", batch_size))
    top_k = int(score_cfg.get("top_k", 50))
    return_full = bool(score_cfg.get("return_cell_scores", False))

    # Add direct comparison parameters to score_cfg for use in scoring loop
    score_cfg["use_direct_comparison"] = use_direct_comparison
    score_cfg["numeric_tolerance"] = numeric_tolerance

    def _stable_hash_to_bucket(text: str, vocab_size: int) -> int:
        import hashlib
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % int(vocab_size)

    def _normalize_numeric(value: float, num_numeric_bins: int) -> float:
        return (float(value) % float(num_numeric_bins)) / float(num_numeric_bins)

    tabular_encoder.eval()
    fusion_module.eval()
    for recon_head in recon_heads:
        recon_head.eval()
    tabular_processor.eval()
    
    # Explicitly move all components to runtime_device before scoring to avoid device mismatches
    tabular_processor = tabular_processor.to(runtime_device)
    tabular_encoder = tabular_encoder.to(runtime_device)
    fusion_module = fusion_module.to(runtime_device)
    recon_heads = recon_heads.to(runtime_device)
    mask_embedding = mask_embedding.to(runtime_device)
    # 确保text_encoder也在正确的设备上
    text_encoder = text_encoder.to(runtime_device)

    vocab_size = int(getattr(tabular_processor, "vocab_size", 10000))
    num_numeric_bins = int(getattr(tabular_processor, "num_numeric_bins", 1000))
    col_is_numeric = list(getattr(train_dataset, "col_is_numeric", []))
    column_value_to_id = getattr(train_dataset, "column_value_to_id", None)
    unknown_token = getattr(train_dataset, "unknown_token", "<UNK>")
    num_rows = len(dirty_rows)
    num_cols = len(column_names) if column_names is not None else (len(dirty_rows[0]) if dirty_rows else 0)

    cell_scores: Optional[np.ndarray] = None
    if return_full and num_rows > 0 and num_cols > 0:
        cell_scores = np.zeros((num_rows, num_cols), dtype=np.float32)

    # Store reconstructed values for analysis
    reconstructed_values: Optional[np.ndarray] = None
    if return_full and num_rows > 0 and num_cols > 0:
        reconstructed_values = np.empty((num_rows, num_cols), dtype=object)

    topk_cells: List[Dict[str, Any]] = []
    per_column_stats: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for col_idx in tqdm(range(num_cols), desc="Scoring columns", ncols=100):
            col_losses: List[float] = []
            col_items: List[Tuple[float, int]] = []

            for start in range(0, num_rows, score_batch_size):
                end = min(start + score_batch_size, num_rows)
                batch_rows = dirty_rows[start:end]
                batch_texts = text_descriptions[start:end]

                # Generate per-column text prompts for this column
                col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Column {col_idx}"
                per_column_text_inputs: Dict[int, Dict[str, Any]] = {}
                pending_indices: List[int] = []
                pending_prompts: List[str] = []
                
                for i, r in enumerate(batch_rows):
                    row_idx = start + i
                    # Serialize row data
                    row_serialized = ", ".join([
                        f"{column_names[c] if column_names and c < len(column_names) else f'Col{c}'}: {r[c]}"
                        for c in range(num_cols)
                    ])
                    # Create per-column instruction prompt
                    col_prompt = (
                        f"Instruction: Check for errors (missing, typo, column pattern violations, rule violations) in column '{col_name}' for this record. "
                        f"Record: {row_serialized}. "
                        f"The error checking analysis result is: [MASK]"
                    )
                    
                    cache_key = (row_idx, col_idx)
                    if cached_text_embeddings and cache_key in cached_text_embeddings:
                        per_column_text_inputs[i] = {"cached_embedding": cached_text_embeddings[cache_key]}
                    else:
                        pending_indices.append(i)
                        pending_prompts.append(col_prompt)
                
                if pending_prompts:
                    processed_batch = _process_texts_with_batch_support(text_processor, pending_prompts)
                    for local_idx, batch_idx in enumerate(pending_indices):
                        per_column_text_inputs[batch_idx] = processed_batch[local_idx]
                
                # Collate per-column text inputs
                col_cached_list = []
                col_input_ids_list = []
                col_attention_mask_list = []
                col_has_cached = False
                
                for i in range(end - start):
                    col_text_input = per_column_text_inputs[i]
                    if "cached_embedding" in col_text_input:
                        col_cached_list.append(col_text_input["cached_embedding"])
                        col_has_cached = True
                    else:
                        col_input_ids_list.append(col_text_input["input_ids"])
                        col_attention_mask_list.append(col_text_input["attention_mask"])
                
                if col_has_cached:
                    text_inputs = {
                        "cached_embedding": torch.stack(col_cached_list).to(runtime_device),
                        "per_column": {col_idx: {"cached_embedding": torch.stack(col_cached_list).to(runtime_device)}}
                    }
                else:
                    text_inputs = {
                        "input_ids": torch.stack(col_input_ids_list).to(runtime_device),
                        "attention_mask": torch.stack(col_attention_mask_list).to(runtime_device),
                        "per_column": {
                            col_idx: {
                                "input_ids": torch.stack(col_input_ids_list).to(runtime_device),
                                "attention_mask": torch.stack(col_attention_mask_list).to(runtime_device)
                            }
                        }
                    }

                B = end - start
                mask_indices = torch.full((B, 1), int(col_idx), dtype=torch.long, device=runtime_device)

                # Use column type from config (all columns must be configured)
                col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"
                if column_types and col_name in column_types:
                    col_type = column_types[col_name].lower()
                    is_num_col = col_type in ["numeric", "numerical", "number", "int", "float"]
                else:
                    raise ValueError(f"Column '{col_name}' not found in column_types configuration. "
                                   f"All columns must be explicitly configured in the 'column_types' section.")

                target_types = torch.full((B, 1), 1 if is_num_col else 0, dtype=torch.long, device=runtime_device)
                if is_num_col:
                    target_ids = torch.full((B, 1), -1, dtype=torch.long, device=runtime_device)
                    vals = []
                    for r in batch_rows:
                        v = r[col_idx]
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            vals.append(_normalize_numeric(float(v), num_numeric_bins))
                        else:
                            vals.append(float("nan"))
                    target_nums = torch.tensor(vals, dtype=torch.float32, device=runtime_device).unsqueeze(1)
                else:
                    target_nums = torch.full((B, 1), float("nan"), dtype=torch.float32, device=runtime_device)
                    ids: List[int] = []
                    mapping = None
                    unknown_id = 0
                    if column_value_to_id and col_idx < len(column_value_to_id):
                        mapping = column_value_to_id[col_idx] or {}
                        unknown_id = mapping.get(unknown_token, 0)
                    for r in batch_rows:
                        v = r[col_idx]
                        token = "" if v is None else str(v)
                        if mapping:
                            ids.append(int(mapping.get(token, unknown_id)))
                        else:
                            ids.append(_stable_hash_to_bucket(token, vocab_size))
                    target_ids = torch.tensor(ids, dtype=torch.long, device=runtime_device).unsqueeze(1)

                # Forward path mirrors train_step_mcm (no backward)
                tabular_processor = tabular_processor.to(runtime_device)
                tabular_inputs = tabular_processor.process_batch(
                    batch_rows,
                    row_indices=list(range(start, end)),
                    column_names=column_names,
                )
                tabular_inputs = {k: v.to(runtime_device) for k, v in tabular_inputs.items()}

                # Apply mask
                cell_emb = tabular_inputs["cell_embeddings"].clone()
                batch_idx = torch.arange(B, device=runtime_device).unsqueeze(1)
                cell_emb[batch_idx, mask_indices] = mask_embedding
                tabular_inputs["cell_embeddings"] = cell_emb

                H_table = tabular_encoder(tabular_inputs)  # [B, C, d_model]
                
                # Use per-column text encoding with cell-level alignment
                if "per_column" in text_inputs and col_idx in text_inputs["per_column"]:
                    col_text_input = text_inputs["per_column"][col_idx]
                    if "cached_embedding" in col_text_input:
                        H_text_col = col_text_input["cached_embedding"].to(runtime_device)
                        if H_text_col.dim() == 2:
                            H_text_col = H_text_col.unsqueeze(0)
                        if H_text_col.dim() == 3:
                            H_text_col = H_text_col[:, -1, :]  # [B, d_model]
                    else:
                        col_text_input_device = _move_batch_to_device(col_text_input, runtime_device)
                        H_text_col = text_encoder(col_text_input_device)
                        if H_text_col.dim() == 3:
                            H_text_col = H_text_col[:, -1, :]  # [B, d_model]

                    B, C, d_model = H_table.shape
                    # Cell-level alignment: each cell uses its corresponding text embedding
                    # H_text_col is [B, d_model] - cell-level embeddings for this column across all rows
                    # We want H_text_expanded to have each row's text embedding for each column
                    # For columns other than col_idx, we'll use a fallback strategy
                    H_text_expanded = torch.zeros((B, C, d_model), device=runtime_device, dtype=H_table.dtype)

                    # Fill in the current column's cell-level text embeddings
                    # Each row uses its own text embedding for this column
                    H_text_expanded[:, col_idx, :] = H_text_col

                    # For other columns, we need to get their per-column embeddings if available
                    # This is a simplified approach - in practice, we might want to cache all columns
                    for other_col_idx in range(C):
                        if other_col_idx != col_idx and "per_column" in text_inputs and other_col_idx in text_inputs["per_column"]:
                            other_col_input = text_inputs["per_column"][other_col_idx]
                            if "cached_embedding" in other_col_input:
                                other_H_text = other_col_input["cached_embedding"].to(runtime_device)
                                if other_H_text.dim() == 3:
                                    other_H_text = other_H_text[:, -1, :]  # [B, d_model]
                                H_text_expanded[:, other_col_idx, :] = other_H_text
                            else:
                                # Fallback to global text embedding for this column
                                if "cached_embedding" in text_inputs:
                                    global_H_text = text_inputs["cached_embedding"].to(runtime_device)
                                    if global_H_text.dim() == 3:
                                        global_H_text = global_H_text[:, -1, :]  # [B, d_model]
                                    H_text_expanded[:, other_col_idx, :] = global_H_text
                        else:
                            # Fallback to global text embedding
                            if "cached_embedding" in text_inputs:
                                global_H_text = text_inputs["cached_embedding"].to(runtime_device)
                                if global_H_text.dim() == 3:
                                    global_H_text = global_H_text[:, -1, :]  # [B, d_model]
                                H_text_expanded[:, other_col_idx, :] = global_H_text
                else:
                    # Fallback to global text embedding
                    if "cached_embedding" in text_inputs:
                        H_text = text_inputs["cached_embedding"].to(runtime_device)
                        if H_text.dim() == 2:
                            H_text = H_text.unsqueeze(0)
                        if H_text.dim() == 3:
                            H_text = H_text[:, -1, :]  # [B, d_model]
                        B, C, d_model = H_table.shape
                        H_text_expanded = H_text.unsqueeze(1).expand(-1, C, -1)  # [B, C, d_model]
                    else:
                        H_text = text_encoder(_move_batch_to_device(text_inputs, runtime_device))
                        if H_text.dim() == 3:
                            H_text = H_text[:, -1, :]  # [B, d_model]
                        B, C, d_model = H_table.shape
                        H_text_expanded = H_text.unsqueeze(1).expand(-1, C, -1)  # [B, C, d_model]
                
                # Use per-column fusion module: apply fusion only to the target column
                if isinstance(fusion_module, nn.ModuleList):
                    # Per-column fusion: use dedicated fusion module for this column
                    col_table_embed = H_table[:, col_idx:col_idx+1, :]  # [B, 1, d_model]
                    # Use per-column text embedding if available, otherwise use global
                    if "per_column" in text_inputs and col_idx in text_inputs["per_column"]:
                        col_text_input = text_inputs["per_column"][col_idx]
                        if "cached_embedding" in col_text_input:
                            col_text_emb = col_text_input["cached_embedding"].to(runtime_device)
                            if col_text_emb.dim() == 2:
                                col_text_emb = col_text_emb.unsqueeze(1)  # [B, 1, d_model]
                            elif col_text_emb.dim() == 3:
                                col_text_emb = col_text_emb[:, -1, :].unsqueeze(1)  # [B, 1, d_model]
                        else:
                            # Process through text encoder
                            col_text_input_device = _move_batch_to_device(col_text_input, runtime_device)
                            col_text_emb = text_encoder(col_text_input_device)
                            if col_text_emb.dim() == 3:
                                col_text_emb = col_text_emb[:, -1, :].unsqueeze(1)  # [B, 1, d_model]
                    else:
                        # Use global text embedding for this column
                        col_text_emb = H_text_expanded[:, col_idx:col_idx+1, :]  # [B, 1, d_model]

                    # Apply per-column fusion
                    H_fuse = fusion_module[col_idx](col_table_embed, col_text_emb)  # [B, 1, d_model]
                    col_embedding = H_fuse  # [B, 1, d_model]
                else:
                    # Shared fusion module: compute for all columns, then extract
                    H_fuse = fusion_module(H_table, H_text_expanded)  # [B, C, d_model]
                    col_embedding = H_fuse[:, col_idx:col_idx+1, :]  # [B, 1, d_model]
                col_recon_head = recon_heads[col_idx]
                cat_logits_col, num_pred_col = col_recon_head(col_embedding)  # [B, 1, vocab_size], [B, 1]
                
                # Squeeze to remove column dimension
                cat_logits_col = cat_logits_col.squeeze(1)  # [B, vocab_size]
                num_pred_col = num_pred_col.squeeze(1)  # [B]

                # Compute per-row loss for this (row, col)
                if use_direct_comparison:
                    # New method: direct comparison of reconstructed values
                    if is_num_col:
                        # Numeric data: compute L1 distance in normalized space
                        preds = num_pred_col  # [B]
                        tgt = target_nums.squeeze(1).to(runtime_device)  # [B]
                        finite = torch.isfinite(tgt)
                        per_row = torch.zeros((B,), dtype=torch.float32, device=runtime_device)
                        # Use L1 distance instead of MSE
                        per_row[finite] = torch.abs(preds[finite] - tgt[finite])
                        # Apply tolerance threshold for numeric data
                        errors = per_row > numeric_tolerance
                        per_row = errors.float()
                    else:
                        # Categorical data: directly compare predicted category with ground truth
                        predicted_ids = torch.argmax(cat_logits_col, dim=-1)  # [B]
                        tgt = target_ids.squeeze(1).to(runtime_device)  # [B]
                        # Direct comparison: 1 if different (error), 0 if same (correct)
                        per_row = (predicted_ids != tgt).float()
                else:
                    # Original method: compute loss based scores
                    if is_num_col:
                        preds = num_pred_col
                        tgt = target_nums.squeeze(1).to(runtime_device)
                        finite = torch.isfinite(tgt)
                        per_row = torch.zeros((B,), dtype=torch.float32, device=runtime_device)
                        per_row[finite] = (preds[finite] - tgt[finite]).pow(2)
                    else:
                        logits = cat_logits_col
                        tgt = target_ids.squeeze(1).to(runtime_device)
                        per_row = F.cross_entropy(logits, tgt, reduction="none").to(torch.float32)

                per_row_cpu = per_row.detach().cpu().numpy()

                # Store reconstructed values for analysis
                if reconstructed_values is not None:
                    if is_num_col:
                        # For numeric columns, store the predicted normalized value
                        reconstructed_vals = num_pred_col.detach().cpu().numpy()
                    else:
                        # For categorical columns, convert predicted IDs back to actual values
                        predicted_ids = torch.argmax(cat_logits_col, dim=-1).detach().cpu().numpy()
                        # Create reverse mapping from ID to actual value
                        id_to_value = {}
                        if column_value_to_id and col_idx < len(column_value_to_id):
                            mapping = column_value_to_id[col_idx] or {}
                            for token, token_id in mapping.items():
                                id_to_value[int(token_id)] = token

                        # Get the actual values from the original data
                        reconstructed_vals = []
                        for pred_id in predicted_ids:
                            if int(pred_id) in id_to_value:
                                reconstructed_vals.append(id_to_value[int(pred_id)])
                            else:
                                reconstructed_vals.append(unknown_token)

                for i in range(B):
                    row_idx = start + i
                    loss_val = float(per_row_cpu[i])
                    col_losses.append(loss_val)
                    col_items.append((loss_val, row_idx))
                    if cell_scores is not None:
                        cell_scores[row_idx, col_idx] = loss_val
                    if reconstructed_values is not None:
                        reconstructed_values[row_idx, col_idx] = reconstructed_vals[i] if is_num_col else reconstructed_vals[i]

            if col_losses:
                per_column_stats[str(column_names[col_idx] if column_names else col_idx)] = {
                    "mean": float(np.mean(col_losses)),
                    "std": float(np.std(col_losses)),
                }
                # keep top-k per column candidates, later we will merge globally
                col_items.sort(key=lambda x: x[0], reverse=True)
                for loss_val, row_idx in col_items[: min(top_k, len(col_items))]:
                    v = dirty_rows[row_idx][col_idx]
                    topk_cells.append(
                        {
                            "row_idx": int(row_idx),
                            "col_idx": int(col_idx),
                            "column": str(column_names[col_idx] if column_names else col_idx),
                            "value": None if v is None else str(v),
                            "score": float(loss_val),
                            "type": "numeric" if is_num_col else "categorical",
                        }
                    )

    # Global top-k (dedupe and sort)
    topk_cells.sort(key=lambda x: float(x["score"]), reverse=True)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in topk_cells:
        key = (item["row_idx"], item["col_idx"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    # ============ MCM Evaluation: Compare with clean data ============
    mcm_eval_metrics: Optional[Dict[str, float]] = None
    eval_cfg = exp_cfg.get("evaluation")
    if eval_cfg:
        clean_path_value = eval_cfg.get("clean_data_path")
        if clean_path_value:
            from ..configuration import _resolve_path_like
            from ..data_loading import load_csv_dataset
            from .error_analysis import compute_error_mask
            from .evaluation import _compute_classification_metrics
            
            clean_path = Path(_resolve_path_like(clean_path_value, config_dir, project_root))
            max_rows = eval_cfg.get("max_rows")
            
            print("\n" + "=" * 80)
            print("MCM 评估：对比clean和dirty数据计算错误检测指标")
            print("=" * 80)
            
            try:
                clean_rows, _, _, clean_column_names = load_csv_dataset(
                    clean_path, dataset_name=eval_cfg.get("dataset_name"), max_rows=max_rows
                )
                
                # Ensure clean and dirty have same number of rows
                min_rows = min(len(clean_rows), len(dirty_rows))
                clean_rows = clean_rows[:min_rows]
                dirty_rows_eval = dirty_rows[:min_rows]
                
                # Compute ground truth error mask
                error_masks = compute_error_mask(clean_rows, dirty_rows_eval)
                
                # Convert error masks to flat list: 1 = error, 0 = correct
                ground_truth = []
                for mask in error_masks:
                    ground_truth.extend(mask)
                
                # Generate predictions using per-column optimal thresholds
                predictions = [0] * len(ground_truth)
                num_rows_eval = len(dirty_rows_eval)
                num_cols_eval = len(dirty_rows_eval[0]) if dirty_rows_eval else 0

                # Check if using direct comparison (scores are already binary)
                if use_direct_comparison:
                    print(f"  ✓ 使用直接比较模式，scores已经是二值化的")
                    if cell_scores is not None and cell_scores.shape[0] >= num_rows_eval and cell_scores.shape[1] >= num_cols_eval:
                        cell_scores_eval = cell_scores[:num_rows_eval, :num_cols_eval]
                        # Scores are already binary (0 or 1), just flatten
                        flat_scores = cell_scores_eval.flatten()
                        predictions = [1 if score > 0.5 else 0 for score in flat_scores]
                    else:
                        print(f"  ⚠️  没有cell_scores，使用TopK预测")
                        # Fallback to top-k
                        for cell in deduped[:top_k]:
                            row_idx = cell["row_idx"]
                            col_idx = cell["col_idx"]
                            if row_idx < num_rows_eval and col_idx < num_cols_eval:
                                flat_idx = row_idx * num_cols_eval + col_idx
                                if flat_idx < len(predictions):
                                    predictions[flat_idx] = 1
                else:
                    # Original method: find optimal threshold per column
                    best_thresholds = {}
                    best_f1_per_col = {}

                    if cell_scores is not None and cell_scores.shape[0] >= num_rows_eval and cell_scores.shape[1] >= num_cols_eval:
                        cell_scores_eval = cell_scores[:num_rows_eval, :num_cols_eval]

                        # For each column, find the optimal threshold
                        for col_idx in range(num_cols_eval):
                            col_scores = cell_scores_eval[:, col_idx]  # [num_rows]
                            col_ground_truth = [ground_truth[r * num_cols_eval + col_idx] for r in range(num_rows_eval)]

                            # Get unique score values to use as candidate thresholds
                            unique_scores = np.sort(np.unique(col_scores))[::-1]  # Descending order

                            best_f1 = -1
                            best_thresh = unique_scores[0] if len(unique_scores) > 0 else 0

                            # Try each unique score as a threshold
                            for threshold in unique_scores:
                                col_preds = [1 if score >= threshold else 0 for score in col_scores]

                                # Compute F1 for this threshold
                                tp = sum(1 for i in range(len(col_preds)) if col_preds[i] == 1 and col_ground_truth[i] == 1)
                                fp = sum(1 for i in range(len(col_preds)) if col_preds[i] == 1 and col_ground_truth[i] == 0)
                                fn = sum(1 for i in range(len(col_preds)) if col_preds[i] == 0 and col_ground_truth[i] == 1)

                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                                if f1 > best_f1:
                                    best_f1 = f1
                                    best_thresh = threshold

                            best_thresholds[col_idx] = best_thresh
                            best_f1_per_col[col_idx] = best_f1

                        # Apply per-column thresholds to generate predictions
                        for r in range(num_rows_eval):
                            for c in range(num_cols_eval):
                                flat_idx = r * num_cols_eval + c
                                score = cell_scores_eval[r, c]
                                threshold = best_thresholds.get(c, 0)
                                if score >= threshold:
                                    predictions[flat_idx] = 1

                        # Print per-column thresholds and F1 scores
                        print(f"\n  每列最优阈值及F1-Score:")
                        for col_idx in range(num_cols_eval):
                            col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"
                            print(f"    {col_name}: 阈值={best_thresholds[col_idx]:.4f}, 列级F1={best_f1_per_col[col_idx]:.4f}")
                    else:
                        # Fallback: use global threshold on all cell_scores
                        if cell_scores is not None:
                            print(f"⚠️  无法逐列计算阈值，使用全局阈值")
                            flat_scores = cell_scores.flatten()
                            unique_scores = np.sort(np.unique(flat_scores))[::-1]

                            best_f1 = -1
                            best_global_thresh = unique_scores[0] if len(unique_scores) > 0 else 0

                            for threshold in unique_scores:
                                global_preds = [1 if score >= threshold else 0 for score in flat_scores]
                                tp = sum(1 for i in range(len(global_preds)) if global_preds[i] == 1 and ground_truth[i] == 1)
                                fp = sum(1 for i in range(len(global_preds)) if global_preds[i] == 1 and ground_truth[i] == 0)
                                fn = sum(1 for i in range(len(global_preds)) if global_preds[i] == 0 and ground_truth[i] == 1)
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                if f1 > best_f1:
                                    best_f1 = f1
                                    best_global_thresh = threshold

                            predictions = [1 if score >= best_global_thresh else 0 for score in flat_scores]
                            print(f"    全局最优阈值={best_global_thresh:.4f}, 全局F1={best_f1:.4f}")
                        else:
                            # Last resort: use topk_cells
                            print(f"⚠️  没有cell_scores可用，使用TopK预测")
                            for cell in deduped[:top_k]:
                                row_idx = cell["row_idx"]
                                col_idx = cell["col_idx"]
                                if row_idx < num_rows_eval and col_idx < num_cols_eval:
                                    flat_idx = row_idx * num_cols_eval + col_idx
                                    if flat_idx < len(predictions):
                                        predictions[flat_idx] = 1
                
                # Compute metrics
                preds_tensor = torch.tensor(predictions, dtype=torch.int32)
                targets_tensor = torch.tensor(ground_truth, dtype=torch.int32)

                mcm_eval_metrics = _compute_classification_metrics(preds_tensor, targets_tensor)

                # ===== NEW: Compute per-column metrics =====
                per_column_metrics: Dict[str, Dict[str, int]] = {}

                # Print column type information
                print(f"\n" + "=" * 80)
                print("列类型统计:")
                print("=" * 80)

                numeric_cols = []
                categorical_cols = []

                for col_idx in range(num_cols_eval):
                    col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"

                    # Use column_types from config (all columns must be configured)
                    if column_types and col_name in column_types:
                        col_type = column_types[col_name].lower()
                        is_num = col_type in ["numeric", "numerical", "number", "int", "float"]
                    else:
                        raise ValueError(f"Column '{col_name}' not found in column_types configuration. "
                                       f"All columns must be explicitly configured in the 'column_types' section.")

                    if is_num:
                        numeric_cols.append(col_name)
                    else:
                        categorical_cols.append(col_name)

                print(f"数值型列 ({len(numeric_cols)}个):")
                for i, col in enumerate(numeric_cols):
                    type_source = "(配置)" if (column_types and col in column_types) else "(自动检测)"
                    print(f"  {i+1}. {col} {type_source}")

                print(f"\n分类型列 ({len(categorical_cols)}个):")
                for i, col in enumerate(categorical_cols):
                    type_source = "(配置)" if (column_types and col in column_types) else "(自动检测)"
                    print(f"  {i+1}. {col} {type_source}")

                print(f"\n" + "=" * 80)
                print("MCM 逐列评估结果:")
                print("=" * 80)

                for col_idx in range(num_cols_eval):
                    col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"

                    # Get predictions and ground truth for this column
                    col_preds = [predictions[r * num_cols_eval + col_idx] for r in range(num_rows_eval)]
                    col_gt = [ground_truth[r * num_cols_eval + col_idx] for r in range(num_rows_eval)]

                    # Calculate TP, FP, FN, TN for this column
                    tp = sum(1 for i in range(len(col_preds)) if col_preds[i] == 1 and col_gt[i] == 1)
                    fp = sum(1 for i in range(len(col_preds)) if col_preds[i] == 1 and col_gt[i] == 0)
                    fn = sum(1 for i in range(len(col_preds)) if col_preds[i] == 0 and col_gt[i] == 1)
                    tn = sum(1 for i in range(len(col_preds)) if col_preds[i] == 0 and col_gt[i] == 0)

                    # Calculate metrics
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                    # Calculate cardinality (number of unique values in this column from dirty data)
                    col_values = set()
                    for r in range(num_rows_eval):
                        if r < len(dirty_rows_eval) and col_idx < len(dirty_rows_eval[r]):
                            cell_value = dirty_rows_eval[r][col_idx]
                            col_values.add(str(cell_value) if cell_value is not None else "None")
                    cardinality = len(col_values)

                    per_column_metrics[col_name] = {
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": precision, "recall": recall, "f1": f1,
                        "total_cells": len(col_preds),
                        "true_errors": sum(col_gt),
                        "predicted_errors": sum(col_preds),
                        "cardinality": cardinality
                    }

                # Print per-column results
                print(f"\n{'列名':<20} {'Prec':<8} {'Rec':<8} {'F1':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'真实错误':<8} {'预测错误':<8} {'Cardinality':<12}")
                print("-" * 115)

                for col_idx in range(num_cols_eval):
                    col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"
                    metrics = per_column_metrics[col_name]
                    print(f"{col_name:<20} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f} "
                          f"{metrics['tp']:<6} {metrics['fp']:<6} {metrics['fn']:<6} {metrics['tn']:<6} "
                          f"{metrics['true_errors']:<8} {metrics['predicted_errors']:<8} {metrics['cardinality']:<12}")

                # Store detailed analysis for later printing (after summary)
                detailed_analysis: List[str] = []
                detailed_analysis.append("\n" + "=" * 80)
                detailed_analysis.append("逐列详细分析:")
                detailed_analysis.append("=" * 80)

                for col_idx in range(num_cols_eval):
                    col_name = column_names[col_idx] if column_names and col_idx < len(column_names) else f"Col{col_idx}"
                    metrics = per_column_metrics[col_name]

                    # Get detailed error information for this column
                    tp_details = []
                    fp_details = []
                    fn_details = []

                    for r in range(num_rows_eval):
                        row_idx = r
                        col_pred = predictions[r * num_cols_eval + col_idx]
                        col_gt = ground_truth[r * num_cols_eval + col_idx]

                        cell_value = dirty_rows_eval[r][col_idx] if r < len(dirty_rows_eval) and col_idx < len(dirty_rows_eval[r]) else None
                        cell_value_str = str(cell_value) if cell_value is not None else "None"

                        # Get reconstructed value if available
                        reconstructed_value_str = ""
                        if reconstructed_values is not None and r < reconstructed_values.shape[0] and col_idx < reconstructed_values.shape[1]:
                            recon_val = reconstructed_values[r, col_idx]
                            reconstructed_value_str = str(recon_val) if recon_val is not None else "None"

                        if col_pred == 1 and col_gt == 1:
                            tp_details.append((row_idx, cell_value_str, reconstructed_value_str))
                        elif col_pred == 1 and col_gt == 0:
                            fp_details.append((row_idx, cell_value_str, reconstructed_value_str))
                        elif col_pred == 0 and col_gt == 1:
                            fn_details.append((row_idx, cell_value_str, reconstructed_value_str))

                    # Build detailed analysis for this column
                    col_analysis = []
                    col_analysis.append(f"\n【{col_name}】:")
                    col_analysis.append(f"  a) 正确识别的错误 (TP): {metrics['tp']}")
                    if tp_details:
                        col_analysis.append(f"     具体实例 (最多显示10个):")
                        for i, (row_idx, original_val, recon_val) in enumerate(tp_details[:10]):
                            if recon_val:
                                col_analysis.append(f"       - 行{row_idx}: 原始值='{original_val}' → 重建值='{recon_val}'")
                            else:
                                col_analysis.append(f"       - 行{row_idx}: {original_val}")
                        if len(tp_details) > 10:
                            col_analysis.append(f"       ... 还有{len(tp_details) - 10}个")

                    col_analysis.append(f"  b) 漏掉的错误 (FN): {metrics['fn']}")
                    if fn_details:
                        col_analysis.append(f"     具体实例 (最多显示10个):")
                        for i, (row_idx, original_val, recon_val) in enumerate(fn_details[:10]):
                            if recon_val:
                                col_analysis.append(f"       - 行{row_idx}: 原始值='{original_val}' → 重建值='{recon_val}'")
                            else:
                                col_analysis.append(f"       - 行{row_idx}: {original_val}")
                        if len(fn_details) > 10:
                            col_analysis.append(f"       ... 还有{len(fn_details) - 10}个")

                    col_analysis.append(f"  c) 误判的错误 (FP): {metrics['fp']}")
                    if fp_details:
                        col_analysis.append(f"     具体实例 (最多显示10个):")
                        for i, (row_idx, original_val, recon_val) in enumerate(fp_details[:10]):
                            if recon_val:
                                col_analysis.append(f"       - 行{row_idx}: 原始值='{original_val}' → 重建值='{recon_val}'")
                            else:
                                col_analysis.append(f"       - 行{row_idx}: {original_val}")
                        if len(fp_details) > 10:
                            col_analysis.append(f"       ... 还有{len(fp_details) - 10}个")

                    col_analysis.append(f"  整体指标: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

                    detailed_analysis.extend(col_analysis)

                print("\n" + "=" * 80)
                print("MCM 整体评估结果:")
                print("=" * 80)
                print(f"  总cell数: {len(ground_truth)}")
                print(f"  真实错误cell数: {sum(ground_truth)}")
                print(f"  预测错误cell数: {sum(predictions)}")
                print(f"  Precision (错误检测精确率): {mcm_eval_metrics['precision']:.4f}")
                print(f"  Recall (错误检测召回率): {mcm_eval_metrics['recall']:.4f}")
                print(f"  F1 Score: {mcm_eval_metrics['f1']:.4f}")
                print(f"  TP: {mcm_eval_metrics['tp']}, FP: {mcm_eval_metrics['fp']}, FN: {mcm_eval_metrics['fn']}, TN: {mcm_eval_metrics['tn']}")
                
            except Exception as e:
                print(f"\n⚠️  MCM评估失败: {e}")
                import traceback
                traceback.print_exc()
                mcm_eval_metrics = None

    result: Dict[str, Any] = {
        "stage_a_losses": [],
        "stage_b_losses": [],
        "mcm_losses": train_losses,
        "train_losses": train_losses,
        "dataset": dataset_name,
        "mcm_per_column_stats": per_column_stats,
        "mcm_topk_cells": deduped,
    }
    if cell_scores is not None:
        result["mcm_cell_scores"] = cell_scores
    if reconstructed_values is not None:
        result["mcm_reconstructed_values"] = reconstructed_values
    if mcm_eval_metrics is not None:
        result["mcm_evaluation"] = mcm_eval_metrics
        # Save per-column metrics to results
        if 'per_column_metrics' in locals():
            result["mcm_per_column_evaluation"] = per_column_metrics
        # Save detailed analysis for printing after summary
        if 'detailed_analysis' in locals():
            result["mcm_detailed_analysis"] = detailed_analysis
    return result


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

    Args:
        exp_cfg: Experiment configuration
        device: Device to use
        device_map: Device mapping for model components
        seed: Random seed
        config_dir: Configuration directory
        project_root: Project root directory

    Returns:
        Dict containing training losses and evaluation metrics
    """
    # Check if this is actually an MCM experiment
    objective = str(exp_cfg.get("objective") or "").lower().strip()
    if objective == "mcm":
        # Redirect to MCM experiment function
        return run_mcm_experiment(exp_cfg, device, device_map, seed=seed, config_dir=config_dir, project_root=project_root)

    print("\n" + "=" * 80)
    print("实验：Contrastive 两阶段训练（预训练 + 二分类）")
    print("=" * 80)

    # Load data
    dirty_rows, text_descriptions, dataset_name, column_names = load_data_from_config(
        exp_cfg,
        default_seed=seed,
        config_dir=config_dir,
        project_root=project_root,
    )

    stage_a_cfg = exp_cfg.get("stage_a", {})
    stage_b_cfg = exp_cfg.get("stage_b", {})

    # Instantiate processors
    from ..configuration import _instantiate_processors
    tabular_processor, text_processor = _instantiate_processors(exp_cfg, config_dir, project_root)

    # Resolve and fix runtime device once to avoid device mismatch
    runtime_device = resolve_runtime_device(device)
    print(f"\n使用设备: {runtime_device}")

    # ============ Stage A: Pretraining (Direct Embedding Similarity) ============
    print("\n阶段A：预训练tabular encoder和text encoder（直接embedding相似度）...")

    # Import here
    from ..components import DETECTION_HEAD_REGISTRY, ENCODER_REGISTRY, FUSION_REGISTRY, build_component, build_optimizer
    from ...datasets import ContrastiveDataset
    from ...fusion import CrossAttentionFusion, SimpleConcatFusion, SingleColumnFusion
    from ...training import collate_fn_contrastive, train_step_contrastive_pretrain

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
        # Check if we should use last token embeddings (like in training)
        use_last_token = stage_a_cfg.get("use_text_output_last_token_embedding", False)
        # Check if we should generate response first (NEW FEATURE)
        generate_response = stage_a_cfg.get("generate_response", False)
        max_new_tokens = stage_a_cfg.get("max_new_tokens", 10)

        # Use the simple precompute method from text_encoder
        cached_text_embeddings = text_encoder.precompute_embeddings_simple(
            text_descriptions=text_descriptions,
            text_processor=text_processor,
            device=runtime_device,
            batch_size=stage_a_cfg.get("batch_size", 16),
            use_last_token=use_last_token,
            generate_response=generate_response,
            max_new_tokens=max_new_tokens,
        )
        print(f"✓ Stage A已启用文本embedding缓存，共缓存 {len(cached_text_embeddings)} 条")
    else:
        print("⚠ Stage A未启用文本embedding缓存（text_encoder未冻结）")

    if cached_text_embeddings is None:
        cached_text_embeddings = {}

    # Build pretraining dataset
    pretrain_dataset = ContrastiveDataset(
        dirty_rows,
        text_descriptions,
        tabular_processor=tabular_processor,
        text_processor=text_processor,
        cached_text_embeddings=cached_text_embeddings,
        column_names=column_names,
    )

    batch_size_a = stage_a_cfg.get("batch_size", 16)
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size_a,
        shuffle=True,
        collate_fn=collate_fn_contrastive,
        num_workers=0,  
        pin_memory=bool(exp_cfg.get("pin_memory", True)),
        persistent_workers=False,
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
    _attach_processor_params(optimizer_a, tabular_processor)

    # Move encoders to device before Stage A training
    tabular_encoder = tabular_encoder.to(runtime_device)
    text_encoder = text_encoder.to(runtime_device)
    tabular_processor = tabular_processor.to(runtime_device)
    tabular_processor.train()

    # Train Stage A
    num_epochs_a = stage_a_cfg.get("num_epochs", 30)
    temperature = stage_a_cfg.get("temperature", 0.07)
    pretrain_losses = []

    for epoch in tqdm(range(num_epochs_a), desc=f"Stage A Training ({num_epochs_a} epochs)", ncols=100):
        epoch_losses = []
        for batch in pretrain_dataloader:
            loss = train_step_contrastive_pretrain(
                tabular_encoder,
                text_encoder,
                batch,
                optimizer_a,
                tabular_processor,
                device=runtime_device,
                temperature=temperature,
                device_map=device_map,
                column_names=column_names,
            )
            epoch_losses.append(loss)
        avg_loss = np.mean(epoch_losses)
        pretrain_losses.append(avg_loss)
        print(f"    Epoch {epoch+1}/{num_epochs_a}, Loss: {avg_loss:.4f}")

    # Load eval dataset for Stage A
    eval_dataset = _build_eval_dataset(exp_cfg, config_dir, project_root)
    if eval_dataset is not None:
        # Use the same processors as training to ensure dimension consistency
        eval_dataset.tabular_processor = tabular_processor
        eval_dataset.text_processor = text_processor

    # ============ Stage B: Per-Column Binary Classification (Frozen Encoders) ============
    print("\n阶段B：逐列二分类训练（冻结编码器，每列独立MLP）...")

    # Get number of columns
    num_cols = len(dirty_rows[0]) if dirty_rows else 0
    negative_strategies = _parse_negative_sampling_config(stage_b_cfg)
    strategy_summaries = [
        f"{cfg.get('type', 'random')}@{cfg.get('ratio', 0):.2f}"
        for cfg in negative_strategies
    ]
    print(f"  使用负样本策略: {', '.join(strategy_summaries)}")

    # Build fusion module (check if per-column fusion or single-column optimization is enabled)
    fusion_cfg = stage_b_cfg["fusion_module"]
    # Default to per-column fusion for better column-specific modeling
    use_per_column_fusion = fusion_cfg.get("params", {}).pop("per_column", True)
    use_single_column_fusion = fusion_cfg.get("params", {}).pop("single_column", False)

    if use_per_column_fusion:
        # Per-column fusion: create separate fusion module for each column (like per-column MLPs)
        fusion_type = fusion_cfg["type"]
        fusion_params = fusion_cfg.get("params", {})
        d_model = fusion_params.get("d_model", 32)

        # Create ModuleList of fusion modules, one for each column
        fusion_modules = nn.ModuleList()
        for _ in range(num_cols):
            if fusion_type == "CrossAttentionFusion":
                module = CrossAttentionFusion(
                    d_model=d_model,
                    **{k: v for k, v in fusion_params.items() if k != "d_model"}
                )
            elif fusion_type == "SimpleConcatFusion":
                module = SimpleConcatFusion(d_model=d_model)
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
            fusion_modules.append(module)

        fusion_module = fusion_modules
        print(f"✓ 启用per-column fusion，为{num_cols}列创建独立的fusion模块")
    elif use_single_column_fusion:
        # Single-column fusion: optimized for processing only one column at a time
        fusion_type = fusion_cfg["type"]
        fusion_params = fusion_cfg.get("params", {})
        fusion_module = SingleColumnFusion(
            fusion_type=fusion_type,
            d_model=fusion_params.get("d_model", 32),
            **{k: v for k, v in fusion_params.items() if k != "d_model"}
        )
        print("✓ 启用single-column fusion优化（仅计算当前需要的列）")
    else:
        # Shared fusion module
        fusion_module = build_component(
            fusion_cfg,
            FUSION_REGISTRY,
            config_dir,
            project_root,
        )
        print("✓ 使用shared fusion module")

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
    tabular_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)
    tabular_processor.eval()
    tabular_processor.requires_grad_(False)

    batch_size_b = stage_b_cfg.get("batch_size", 32)
    num_epochs_b = stage_b_cfg.get("num_epochs", 20)
    binary_losses = []
    use_pos_weight = bool(stage_b_cfg.get("use_pos_weight", True))

    # Build optimizer for all column MLPs and fusion
    if isinstance(fusion_module, nn.ModuleList):
        # Per-column fusion: fusion_module is a ModuleList
        optimizer_b = build_optimizer(
            nn.Sequential(fusion_module, column_mlps),
            stage_b_cfg.get("optimizer", {"type": "Adam", "params": {"lr": 1e-3}})
        )
    else:
        # Shared fusion or single-column fusion
        optimizer_b = build_optimizer(
            nn.Sequential(fusion_module, column_mlps),
            stage_b_cfg.get("optimizer", {"type": "Adam", "params": {"lr": 1e-3}})
        )

    # ============ 简化的缓存策略 ============
    # 分别缓存每行的 row embedding 和 text embedding
    # 总缓存量: 2N (N = num_rows)
    print("\n预计算并缓存编码器输出...")

    num_rows = len(dirty_rows)

    num_cols = len(column_names)
    cached_row_embeddings = {}  # row_idx -> [seq_len, d_model]

    # Process rows in batches
    cache_batch_size = 64
    for batch_start in tqdm(range(0, num_rows, cache_batch_size), ncols=100, desc="缓存编码器输出"):
        batch_end = min(batch_start + cache_batch_size, num_rows)

        # Prepare batch data (batched tabular/text processors)
        batch_rows = [dirty_rows[row_idx] for row_idx in range(batch_start, batch_end)]
        batch_texts = [text_descriptions[row_idx] for row_idx in range(batch_start, batch_end)]
        batch_text_inputs = _process_texts_with_batch_support(text_processor, batch_texts)
        with torch.no_grad():
            tabular_batch = tabular_processor.process_batch(
                batch_rows,
                row_indices=[0] * len(batch_rows),
            )

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
            # 只有在没有复用Stage A缓存时才计算text embedding
            compute_text = (len(cached_text_embeddings) == 0)
            if compute_text:
                if "cached_embedding" in text_batch:
                    H_text = text_batch["cached_embedding"]
                else:
                    H_text = text_encoder(text_batch)  # [batch, text_seq_len, d_model]

        # Cache embeddings by row index
        for i, row_idx in enumerate(range(batch_start, batch_end)):
            cached_row_embeddings[row_idx] = H_table[i].cpu()  # [seq_len, d_model]
            # 只有在没有缓存时才存储text embedding
            if compute_text:
                cached_text_embeddings[row_idx] = H_text[i].cpu()  # [text_seq_len, d_model]

    if len(cached_text_embeddings) > 0:
        print(f"✓ 缓存完成！共缓存 {num_rows} 行的 row embeddings + {len(cached_text_embeddings)} 条text embeddings（复用Stage A）")
    else:
        print(f"✓ 缓存完成！共缓存 {num_rows} 行的 row/text embeddings")

    requires_similar_text = any(cfg.get("type") == "similar_text" for cfg in negative_strategies)
    similar_neighbor_map: Optional[Dict[int, List[int]]] = None
    if requires_similar_text:
        max_top_k = max(int(cfg.get("top_k", 5)) for cfg in negative_strategies if cfg.get("type") == "similar_text")
        max_top_k = max(1, max_top_k)
        try:
            similar_neighbor_map = _build_semantic_neighbor_map(cached_text_embeddings, max_top_k)
            print(f"  ✓ 构建similar_text负样本索引（top-{max_top_k} 相似文本）")
        except ValueError as exc:
            print(f"  ⚠ similar_text负样本策略已禁用：{exc}")
            negative_strategies = [cfg for cfg in negative_strategies if cfg.get("type") != "similar_text"]
            requires_similar_text = False

    # 检查是否需要similar_cell_value策略
    requires_similar_cell_value = any(cfg.get("type") == "similar_cell_value" for cfg in negative_strategies)
    cell_value_neighbor_map: Optional[Dict[int, Dict[int, List[int]]]] = None

    if requires_similar_cell_value:
        max_top_k = max(
            int(cfg.get("top_k", 5))
            for cfg in negative_strategies
            if cfg.get("type") == "similar_cell_value"
        )
        max_top_k = max(1, max_top_k)

        print(f"  构建 cell value 相似度映射（top-{max_top_k}）...")
        cell_value_neighbor_map = {}

        for col_idx in tqdm(range(num_cols), desc="构建cell相似度映射", ncols=100):
            try:
                cell_value_neighbor_map[col_idx] = _build_cell_value_semantic_neighbor_map(
                    cached_row_embeddings, col_idx, max_top_k
                )
                print(f"    ✓ 列 {col_idx} 相似度映射构建完成")
            except Exception as exc:
                print(f"    ⚠ 列 {col_idx} 的 similar_cell_value 策略已禁用：{exc}")
                cell_value_neighbor_map[col_idx] = {}

    if not negative_strategies:
        fallback_ratio = float(stage_b_cfg.get("negative_ratio", 1.0))
        negative_strategies = [{"type": "random", "ratio": max(fallback_ratio, 1.0)}]
        print("  ⚠ 未找到有效的负样本策略，已退回到 random 策略。")

    # ============ 重建 Dataset 使用缓存 ============
    # 构建样本列表：(row_idx, text_idx, col_idx, label)
    # 正样本: row_i + text_i
    # 负样本: 根据配置混合 random / row_text_swap / similar_text
    samples_by_column = {col_idx: [] for col_idx in range(num_cols)}
    # Optional: row-side synthetic corruptions (dirty row, same text) to target
    # syntax errors (missing/typo/pattern) and semantic-ish errors (outliers).
    row_corruption_cfg = stage_b_cfg.get("row_corruption_negatives", {})
    enable_row_corruption = bool(row_corruption_cfg.get("enable", False))
    row_corruption_ratio = float(row_corruption_cfg.get("ratio", 0.0))
    row_corruption_types = row_corruption_cfg.get(
        "types", ["missing", "typo", "pattern_violation", "outlier", "swap_with_frequent", "constraint_violation"]
    )
    row_corruption_max_per_col = int(row_corruption_cfg.get("max_per_column", 1000))
    row_corruption_batch_size = int(row_corruption_cfg.get("batch_size", 64))
    row_corruption_seed = int(row_corruption_cfg.get("seed", seed))
    rng_corrupt = random.Random(row_corruption_seed)
    cached_corrupted_row_embeddings: Dict[Tuple[int, int, int], torch.Tensor] = {}
    if enable_row_corruption and row_corruption_ratio > 0:
        print(
            f"  ✓ 启用 row-side 合成错误负样本: ratio={row_corruption_ratio:.2f}, "
            f"types={row_corruption_types}, max_per_col={row_corruption_max_per_col}"
        )
        col_profiles = _build_column_profiles(dirty_rows, column_names)
    else:
        col_profiles = []

    print(f"  构建训练样本 ({num_cols} 列)...")
    for col_idx in tqdm(range(num_cols), desc="构建训练样本", ncols=100):
        # 正样本
        for row_idx in range(num_rows):
            samples_by_column[col_idx].append((row_idx, row_idx, col_idx, 1))  # (row_idx, text_idx, col_idx, label)

        # 负样本
        for strategy in negative_strategies:
            negative_pairs = _generate_negative_pairs(
                strategy,
                num_rows,
                similar_neighbor_map,
                dirty_rows=dirty_rows,
                col_idx=col_idx,
                cell_value_neighbor_map=cell_value_neighbor_map,
            )
            if not negative_pairs:
                continue
            for row_idx, text_idx in negative_pairs:
                samples_by_column[col_idx].append((row_idx, text_idx, col_idx, 0))

        # Row-side synthetic corruption negatives: (dirty_row, same_text) labeled as mismatch (0)
        if enable_row_corruption and row_corruption_ratio > 0 and col_idx < len(col_profiles):
            num_to_make = min(int(num_rows * row_corruption_ratio), row_corruption_max_per_col)
            if num_to_make > 0:
                chosen_rows = [rng_corrupt.randrange(0, num_rows) for _ in range(num_to_make)]
                # Build corrupted rows and embed them in batches
                corrupted_rows: List[List[Any]] = []
                corrupted_keys: List[Tuple[int, int, int]] = []
                for k, r_idx in enumerate(chosen_rows):
                    base = list(dirty_rows[r_idx])
                    ctype = row_corruption_types[k % max(1, len(row_corruption_types))]
                    base[col_idx] = _corrupt_row_cell(base, col_idx, col_profiles, str(ctype), rng_corrupt)
                    corrupted_rows.append(base)
                    corrupted_keys.append((int(r_idx), int(col_idx), int(k)))

                # Embed corrupted rows
                for b0 in range(0, len(corrupted_rows), row_corruption_batch_size):
                    b1 = min(b0 + row_corruption_batch_size, len(corrupted_rows))
                    batch_rows = corrupted_rows[b0:b1]
                    with torch.no_grad():
                        tabular_batch = tabular_processor.process_batch(
                            batch_rows,
                            row_indices=[0] * len(batch_rows),
                        )
                    tabular_batch = {k: v.to(runtime_device) for k, v in tabular_batch.items()}
                    with torch.no_grad():
                        H_dirty = tabular_encoder(tabular_batch)  # [batch, seq_len, d_model]
                    for i, key in enumerate(corrupted_keys[b0:b1]):
                        cached_corrupted_row_embeddings[key] = H_dirty[i].detach().cpu()

                # Add samples: row_ref is a tuple key that will be resolved to corrupted embedding
                for key in corrupted_keys:
                    row_ref = ("corrupt",) + key  # ("corrupt", row_idx, col_idx, k)
                    samples_by_column[col_idx].append((row_ref, key[0], col_idx, 0))

    # Train Stage B - iterate through columns
    for epoch in tqdm(range(num_epochs_b), desc=f"Stage B Training ({num_epochs_b} epochs)", ncols=100):
        epoch_losses = []

        for col_idx in tqdm(range(num_cols), desc=f"Epoch {epoch+1} - 训练列", ncols=100):
            # Get samples for this column from new structure
            col_samples = samples_by_column[col_idx]
            if not col_samples:
                continue

            # Shuffle samples each epoch
            random.shuffle(col_samples)

            # Simple batching using cached embeddings
            pos_weight = None
            if use_pos_weight:
                num_pos = sum(1 for *_, lbl in col_samples if int(lbl) == 1)
                num_neg = sum(1 for *_, lbl in col_samples if int(lbl) == 0)
                if num_pos > 0 and num_neg > 0:
                    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32, device=runtime_device)

            for batch_start in range(0, len(col_samples), batch_size_b):
                batch_end = min(batch_start + batch_size_b, len(col_samples))
                batch_samples = col_samples[batch_start:batch_end]

                # Get cached embeddings by row_idx/text_idx
                batch_tabular_embeds = []
                batch_text_embeds = []
                batch_labels = []

                for row_idx, text_idx, _, label in batch_samples:
                    if isinstance(row_idx, tuple) and len(row_idx) == 4 and row_idx[0] == "corrupt":
                        key = (int(row_idx[1]), int(row_idx[2]), int(row_idx[3]))
                        batch_tabular_embeds.append(cached_corrupted_row_embeddings[key])
                    else:
                        batch_tabular_embeds.append(cached_row_embeddings[int(row_idx)])
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
                # Check if using per-column fusion (ModuleList) or optimized single-column fusion
                if isinstance(fusion_module, nn.ModuleList):
                    # Per-column fusion: use dedicated fusion module for this column
                    col_tabular_embed = batch_tabular_embeds[:, col_idx:col_idx+1, :]  # [batch, 1, d_model]
                    H_fused = fusion_module[col_idx](col_tabular_embed, batch_text_embeds)  # [batch, 1, d_model]
                    col_embedding = H_fused.squeeze(1)  # [batch, d_model]
                elif isinstance(fusion_module, SingleColumnFusion):
                    # Optimized: only compute fusion for the column we need
                    col_tabular_embed = batch_tabular_embeds[:, col_idx:col_idx+1, :]  # [batch, 1, d_model]
                    H_fused = fusion_module(col_tabular_embed, batch_text_embeds)  # [batch, 1, d_model]
                    col_embedding = H_fused.squeeze(1)  # [batch, d_model]
                else:
                    # Original: compute fusion for all columns, then extract the one we need
                    H_fused = fusion_module(batch_tabular_embeds, batch_text_embeds)  # [batch, num_cols, d_model]
                    col_embedding = H_fused[:, col_idx, :]  # [batch, d_model]

                col_embedding = col_embedding.unsqueeze(1)  # [batch, 1, d_model]
                logits = column_mlps[col_idx](col_embedding)  # [batch, 1, 1]
                logits = logits.squeeze(-1).squeeze(-1)  # [batch]

                # Compute loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    batch_labels.squeeze(-1),  # [batch] to match logits shape
                    pos_weight=pos_weight,
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
            eval_dataset,
            runtime_device,
            device_map,
            threshold=float(stage_b_cfg.get("threshold", 0.5)),
            threshold_grid=stage_b_cfg.get("threshold_grid"),
            select_best_threshold=bool(stage_b_cfg.get("select_best_threshold", True)),
            use_per_column_threshold=bool(stage_b_cfg.get("use_per_column_threshold", False)),
            rule_candidate_min_score=stage_b_cfg.get("rule_candidate_min_score"),
            max_cases_per_column=stage_b_cfg.get("max_cases_per_column"),
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
