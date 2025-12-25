"""
Error analysis utilities for experiments.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from .evaluation import _compute_classification_metrics, _evaluate_threshold_with_scores


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


def print_detailed_error_analysis(
    detailed_predictions: List[Dict],
    column_names: Optional[List[str]] = None,
    *,
    max_cases_per_column: Optional[int] = None,
) -> None:
    """
    Print detailed error analysis including TP, FP, and FN cases.

    Args:
        detailed_predictions: List of prediction details with keys:
            - row_idx: row index
            - col_idx: column index
            - col_name: column name
            - dirty_value: value from dirty data
            - clean_value: value from clean data
            - probability: prediction probability
            - pred: predicted label (0 or 1)
            - label: true label (0 or 1)
        column_names: Optional list of column names for display
        max_cases_per_column: Maximum number of cases to print per column
    """
    # Organize cases by type and column
    tp_cases = {}  # True Positives
    fp_cases = {}  # False Positives
    fn_cases = {}  # False Negatives

    for pred_info in detailed_predictions:
        col_idx = pred_info['col_idx']
        col_name = pred_info['col_name']
        dirty_value = pred_info['dirty_value']
        clean_value = pred_info['clean_value']
        probability = pred_info['probability']
        rule_score = pred_info.get("rule_score")
        skipped_by_rule = bool(pred_info.get("skipped_by_rule", False))
        pred = pred_info['pred']
        label = pred_info['label']

        # Determine case type
        if pred == 1 and label == 1:  # TP: Predicted error and truly an error
            if col_name not in tp_cases:
                tp_cases[col_name] = []
            tp_cases[col_name].append({
                'dirty_value': dirty_value,
                'clean_value': clean_value,
                'probability': probability,
                'rule_score': rule_score,
                'skipped_by_rule': skipped_by_rule,
            })
        elif pred == 1 and label == 0:  # FP: Predicted error but actually correct
            if col_name not in fp_cases:
                fp_cases[col_name] = []
            fp_cases[col_name].append({
                'clean_value': clean_value,
                'probability': probability,
                'rule_score': rule_score,
                'skipped_by_rule': skipped_by_rule,
            })
        elif pred == 0 and label == 1:  # FN: Predicted correct but actually an error
            if col_name not in fn_cases:
                fn_cases[col_name] = []
            fn_cases[col_name].append({
                'dirty_value': dirty_value,
                'clean_value': clean_value,
                'probability': probability,
                'rule_score': rule_score,
                'skipped_by_rule': skipped_by_rule,
            })

    # Get all unique column names and sort them
    all_columns = set()
    if column_names:
        all_columns.update(column_names)
    else:
        all_columns.update([pred_info['col_name'] for pred_info in detailed_predictions])

    sorted_columns = sorted(all_columns)

    # Print True Positives
    print("\n=== 详细错误分析 ===")
    print("\n正确识别的错误 (TP):")
    has_tp = False
    for col_name in sorted_columns:
        if col_name in tp_cases:
            has_tp = True
            cases = tp_cases[col_name]
            if max_cases_per_column is not None:
                cases = cases[: int(max_cases_per_column)]
            for case in cases:
                extra = ""
                if "rule_score" in case:
                    extra = f", rule: {case['rule_score']:.2f}"
                print(
                    f"  {col_name}: {case['dirty_value']} -> {case['clean_value']} "
                    f"(概率: {case['probability']:.4f}{extra})"
                )
    if not has_tp:
        print("  无")

    # Print False Positives
    print("\n误判为错误的正确值 (FP):")
    has_fp = False
    for col_name in sorted_columns:
        if col_name in fp_cases:
            has_fp = True
            cases = fp_cases[col_name]
            if max_cases_per_column is not None:
                cases = cases[: int(max_cases_per_column)]
            for case in cases:
                extra = ""
                if "rule_score" in case:
                    extra = f", rule: {case['rule_score']:.2f}"
                if case.get("skipped_by_rule"):
                    extra += ", skipped_by_rule"
                print(
                    f"  {col_name}: {case['clean_value']} (被误判为错误) "
                    f"(概率: {case['probability']:.4f}{extra})"
                )
    if not has_fp:
        print("  无")

    # Print False Negatives
    print("\n漏掉的错误 (FN):")
    has_fn = False
    for col_name in sorted_columns:
        if col_name in fn_cases:
            has_fn = True
            cases = fn_cases[col_name]
            if max_cases_per_column is not None:
                cases = cases[: int(max_cases_per_column)]
            for case in cases:
                extra = ""
                if "rule_score" in case:
                    extra = f", rule: {case['rule_score']:.2f}"
                if case.get("skipped_by_rule"):
                    extra += ", skipped_by_rule"
                print(
                    f"  {col_name}: {case['dirty_value']} (应为{case['clean_value']}，但未检测到) "
                    f"(概率: {case['probability']:.4f}{extra})"
                )
    if not has_fn:
        print("  无")


def evaluate_per_column_model(
    tabular_encoder: nn.Module,
    text_encoder: nn.Module,
    fusion_module: nn.Module,
    column_mlps: nn.ModuleList,
    eval_dataset,
    device: str,
    device_map: Optional[Dict[str, str]],
    *,
    threshold: float = 0.5,
    threshold_grid: Optional[List[float]] = None,
    select_best_threshold: bool = True,
    use_per_column_threshold: bool = False,
    rule_candidate_min_score: Optional[float] = None,
    max_cases_per_column: Optional[int] = None,
    _compute_metrics_fn=_compute_classification_metrics,
    _evaluate_threshold_fn=_evaluate_threshold_with_scores,
) -> Dict[str, float]:
    """
    Evaluate per-column MLPs for error detection.

    Each column has its own MLP that predicts whether the cell value matches the text.

    Args:
        tabular_encoder: Tabular encoder model
        text_encoder: Text encoder model
        fusion_module: Fusion module
        column_mlps: Per-column MLP detection heads
        eval_dataset: Evaluation dataset
        device: Device to use
        device_map: Device mapping for model components
        threshold: Decision threshold
        threshold_grid: Grid of thresholds to search
        select_best_threshold: Whether to select best threshold
        use_per_column_threshold: Whether to use per-column thresholds
        rule_candidate_min_score: Minimum rule score for candidate filtering
        max_cases_per_column: Maximum cases to print per column
        _compute_metrics_fn: Function to compute metrics (injected for testing)
        _evaluate_threshold_fn: Function to evaluate threshold (injected for testing)

    Returns:
        Dictionary of evaluation metrics
    """
    # Import here to avoid circular dependency
    from .column_profiling import _build_column_profiles, _compute_rule_score, _constraint_violation_score
    from .data_utils import _prepare_eval_loader

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
    col_profiles = _build_column_profiles(eval_dataset.clean_rows, eval_dataset.column_names)
    if rule_candidate_min_score is not None:
        rule_candidate_min_score = float(rule_candidate_min_score)
        print(f"  ✓ 启用规则候选过滤：rule_score >= {rule_candidate_min_score:.2f} 才进入模型判别")

    # Predict on dirty data using per-column MLPs
    all_predictions = []
    all_labels = []
    # Collect detailed predictions for error analysis
    detailed_predictions = []
    all_probs: List[float] = []
    all_targets: List[int] = []
    all_cols: List[int] = []

    for row_idx, (dirty_row, text) in enumerate(tqdm(zip(eval_dataset.dirty_rows, eval_dataset.text_descriptions), desc="评估行", ncols=100)):
        with torch.no_grad():
            tabular_inputs = eval_dataset.tabular_processor.process(
                dirty_row, row_idx=row_idx, column_names=eval_dataset.column_names
            )
            text_inputs = eval_dataset.text_processor.process(text)

        # Add batch dimension (filter out non-tensor values like 'raw_text')
        tabular_inputs = {k: v.unsqueeze(0).to(device) for k, v in tabular_inputs.items() if isinstance(v, torch.Tensor)}
        text_inputs = {k: v.unsqueeze(0).to(device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            H_table = tabular_encoder(tabular_inputs)
            H_text = text_encoder(text_inputs)

            # Get predictions from each column's MLP
            row_preds = []

            for col_idx in range(num_cols):
                # Optional candidate filtering using cheap syntax/outlier heuristics.
                if rule_candidate_min_score is not None and col_idx < len(col_profiles):
                    rule_score = max(
                        _compute_rule_score(dirty_row[col_idx], col_profiles[col_idx]),
                        _constraint_violation_score(col_idx, dirty_row, col_profiles),
                    )
                    if float(rule_score) < float(rule_candidate_min_score):
                        prob_value = 1.0
                        pred = 0
                        row_preds.append(pred)

                        if eval_dataset.column_names and col_idx < len(eval_dataset.column_names):
                            col_name = eval_dataset.column_names[col_idx]
                        else:
                            col_name = f"Column {col_idx}"

                        clean_value = eval_dataset.clean_rows[row_idx][col_idx] if row_idx < len(eval_dataset.clean_rows) else None
                        dirty_value = eval_dataset.dirty_rows[row_idx][col_idx] if row_idx < len(eval_dataset.dirty_rows) else None
                        true_label = error_masks[row_idx][col_idx] if row_idx < len(error_masks) and col_idx < len(error_masks[row_idx]) else 0

                        detailed_predictions.append({
                            'row_idx': row_idx,
                            'col_idx': col_idx,
                            'col_name': col_name,
                            'dirty_value': dirty_value,
                            'clean_value': clean_value,
                            'probability': float(prob_value),
                            'pred': int(pred),
                            'label': int(true_label),
                            'rule_score': float(rule_score),
                            'skipped_by_rule': True,
                        })
                        all_probs.append(float(prob_value))
                        all_targets.append(int(true_label))
                        all_cols.append(int(col_idx))
                        continue

                # Check if using per-column fusion (ModuleList) or optimized single-column fusion
                from ...fusion import SingleColumnFusion
                if isinstance(fusion_module, nn.ModuleList):
                    # Per-column fusion: use dedicated fusion module for each column
                    col_tabular_embed = H_table[:, col_idx:col_idx+1, :]  # [batch, 1, d_model]
                    H_fused = fusion_module[col_idx](col_tabular_embed, H_text)  # [batch, 1, d_model]
                    col_embedding = H_fused.squeeze(1)  # [batch, d_model]
                elif isinstance(fusion_module, SingleColumnFusion):
                    # Optimized: only compute fusion for the current column
                    col_tabular_embed = H_table[:, col_idx:col_idx+1, :]  # [batch, 1, d_model]
                    H_fused = fusion_module(col_tabular_embed, H_text)  # [batch, 1, d_model]
                    col_embedding = H_fused.squeeze(1)  # [batch, d_model]
                else:
                    # Original: compute fusion for all columns once, then extract
                    if col_idx == 0:
                        # Compute fusion for all columns (only once)
                        H_fused_all = fusion_module(H_table, H_text)
                    col_embedding = H_fused_all[:, col_idx, :]

                col_embedding = col_embedding.unsqueeze(1)
                logits = column_mlps[col_idx](col_embedding)
                prob = torch.sigmoid(logits.squeeze())
                prob_value = float(prob.item())
                # Predict error if match probability < threshold
                pred = 1 if prob_value < float(threshold) else 0
                row_preds.append(pred)

                # Collect detailed prediction information for error analysis
                # Get column name (use provided name or generate default)
                if eval_dataset.column_names and col_idx < len(eval_dataset.column_names):
                    col_name = eval_dataset.column_names[col_idx]
                else:
                    col_name = f"Column {col_idx}"

                # Get clean and dirty values
                clean_value = eval_dataset.clean_rows[row_idx][col_idx] if row_idx < len(eval_dataset.clean_rows) else None
                dirty_value = eval_dataset.dirty_rows[row_idx][col_idx] if row_idx < len(eval_dataset.dirty_rows) else None

                # Get true label
                true_label = error_masks[row_idx][col_idx] if row_idx < len(error_masks) and col_idx < len(error_masks[row_idx]) else 0

                detailed_predictions.append({
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'col_name': col_name,
                    'dirty_value': dirty_value,
                    'clean_value': clean_value,
                    'probability': prob_value,
                    'pred': pred,
                    'label': true_label,
                    'rule_score': float(
                        max(
                            _compute_rule_score(dirty_value, col_profiles[col_idx]),
                            _constraint_violation_score(col_idx, dirty_row, col_profiles),
                        )
                    ) if col_idx < len(col_profiles) else 0.0,
                    'skipped_by_rule': False,
                })
                all_probs.append(prob_value)
                all_targets.append(int(true_label))
                all_cols.append(int(col_idx))

            all_predictions.append(row_preds)
            all_labels.append(error_masks[row_idx])

    if not all_probs:
        print("阶段B评估结果 - 无可用预测，跳过评估")
        return {
            "error_precision": 0.0,
            "error_recall": 0.0,
            "error_f1": 0.0,
            "best_threshold": float(threshold),
            "num_error_rows": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    probs_tensor = torch.tensor(all_probs, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets, dtype=torch.int32)
    cols_tensor = torch.tensor(all_cols, dtype=torch.int64)

    # Optionally tune threshold for best F1 on this eval set
    if threshold_grid is None:
        threshold_grid = [i / 100.0 for i in range(1, 100)]

    thresholds_by_col: Dict[int, float] = {}
    best_threshold = float(threshold)
    if select_best_threshold and threshold_grid:
        if use_per_column_threshold:
            unique_cols = sorted(set(int(c) for c in all_cols))
            for c in unique_cols:
                mask = cols_tensor == int(c)
                if int(mask.sum().item()) <= 0:
                    continue
                best_f1 = -1.0
                best_t = float(threshold)
                for t in threshold_grid:
                    m = _evaluate_threshold_fn(probs_tensor[mask], targets_tensor[mask], float(t))
                    if m["f1"] > best_f1:
                        best_f1 = m["f1"]
                        best_t = float(t)
                thresholds_by_col[int(c)] = float(best_t)

            for item in detailed_predictions:
                c = int(item["col_idx"])
                t = float(thresholds_by_col.get(c, float(threshold)))
                item["pred"] = 1 if float(item["probability"]) < t else 0
                item["threshold_used"] = float(t)
        else:
            best_f1 = -1.0
            for t in threshold_grid:
                m = _evaluate_threshold_fn(probs_tensor, targets_tensor, float(t))
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    best_threshold = float(t)
            threshold = best_threshold
            for item in detailed_predictions:
                item["pred"] = 1 if float(item["probability"]) < float(threshold) else 0
                item["threshold_used"] = float(threshold)

    if use_per_column_threshold and thresholds_by_col:
        thr_vec = torch.tensor([thresholds_by_col.get(int(c), float(threshold)) for c in all_cols], dtype=torch.float32)
        preds_tensor = (probs_tensor < thr_vec).int()
    else:
        preds_tensor = (probs_tensor < float(threshold)).int()

    final_metrics = _compute_metrics_fn(preds_tensor, targets_tensor)

    print(
        f"阶段B评估结果 - Error Precision: {final_metrics['precision']:.4f}, "
        f"Error Recall: {final_metrics['recall']:.4f}, Error F1: {final_metrics['f1']:.4f}, "
        f"Threshold: {float(best_threshold if (select_best_threshold and not use_per_column_threshold) else threshold):.2f}"
    )

    # Print detailed error analysis
    print_detailed_error_analysis(
        detailed_predictions,
        eval_dataset.column_names,
        max_cases_per_column=max_cases_per_column,
    )

    return {
        "error_precision": float(final_metrics["precision"]),
        "error_recall": float(final_metrics["recall"]),
        "error_f1": float(final_metrics["f1"]),
        "best_threshold": float(best_threshold if (select_best_threshold and not use_per_column_threshold) else threshold),
        "thresholds_by_col": thresholds_by_col,
        "num_error_rows": len([m for m in error_masks if sum(m) > 0]),
        "true_positives": int(final_metrics["tp"]),
        "false_positives": int(final_metrics["fp"]),
        "false_negatives": int(final_metrics["fn"]),
    }
