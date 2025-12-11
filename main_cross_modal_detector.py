"""
跨模态错误检测系统 - 主程序
Cross-Modal Error Detector - Main Runner

该脚本使用配置文件驱动，每个配置文件对应一个独立实验：
1. 破坏-重建（corruption-based）训练
2. 对比学习（contrastive）训练
3. 两阶段对比学习（contrastive_two_stage）训练
4. 消融实验（ablation study）

使用方式：
    python main_cross_modal_detector.py --config configs/hospital_corruption_experiment.json
    python main_cross_modal_detector.py --config configs/hospital_contrastive_experiment.json
    python main_cross_modal_detector.py --config configs/hospital_two_stage_experiment.json
    python main_cross_modal_detector.py --config configs/hospital_ablation_experiment.json

Mock 实验：
    python main_cross_modal_detector.py --config configs/mock_corruption_experiment.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from cross_modal_error_detector.runner import (
    load_config,
    run_contrastive_two_stage_experiment,
    run_corruption_experiment,
    set_seed,
)
from cross_modal_error_detector.utils.device import (
    canonicalize_device_map,
    resolve_runtime_device,
)


def _normalize_device_map(raw_device_map: Optional[Dict[str, str]], default_device: str) -> Optional[Dict[str, str]]:
    if raw_device_map is None:
        return None
    normalized_map = canonicalize_device_map(raw_device_map)
    resolved_default = resolve_runtime_device(default_device)
    resolved_map = {
        key: resolve_runtime_device(value, fallback_device=resolved_default)
        for key, value in normalized_map.items()
    }
    resolved_map.setdefault("default", resolved_default)
    return resolved_map


def _print_header(device: str, seed: int, experiment: str) -> None:
    print("\n" + "=" * 80)
    print("跨模态错误检测系统 - 主程序")
    print("Cross-Modal Error Detector - Main Runner")
    print("=" * 80)
    print(f"\n使用设备: {device}")
    print(f"随机种子: {seed}")
    print(f"实验类型: {experiment}")


def _print_summary(title: str, lines: Dict[str, str]) -> None:
    print("\n" + "=" * 80)
    print("🎉 实验完成 - 总结")
    print("=" * 80)
    print(f"\n[{title}]")
    for label, value in lines.items():
        print(f"  - {label}: {value}")


def main(config_path: Path) -> None:
    config = load_config(config_path)
    device = resolve_runtime_device(config.get("device"))
    device_map = _normalize_device_map(config.get("device_map"), device)
    seed = config.get("seed", 42)
    experiment = config.get("experiment")
    if experiment is None:
        raise ValueError("配置文件缺少 `experiment` 字段，请指定要运行的实验类型。")

    _print_header(device, seed, str(experiment))

    set_seed(seed)
    config_dir = config_path.parent.resolve()
    project_root = config_dir.parent.resolve()

    experiment_key = str(experiment).lower()
    if experiment_key == "corruption":
        results = run_corruption_experiment(
            config,
            device,
            device_map,
            seed=seed,
            config_dir=config_dir,
            project_root=project_root,
        )
        train_losses = results.get("train_losses") or []
        metrics = results.get("metrics") or {}
        dirty_accuracy = metrics.get("dirty_accuracy", metrics.get("accuracy", results.get("accuracy", 0.0)))
        overall_accuracy = metrics.get("overall_accuracy", results.get("overall_accuracy", 0.0))
        summary = {
            "数据集": str(results.get("dataset")),
            "最终Loss": f"{train_losses[-1]:.4f}" if train_losses else "未计算",
            "脏值准确率": f"{dirty_accuracy:.2%}",
        }
        if overall_accuracy:
            summary["整体准确率"] = f"{overall_accuracy:.2%}"
        if metrics:
            summary["Precision(脏值)"] = f"{metrics.get('precision', 0.0):.2%}"
            summary["Recall(脏值)"] = f"{metrics.get('recall', 0.0):.2%}"
            summary["F1(脏值)"] = f"{metrics.get('f1', 0.0):.2%}"
        if "num_samples" in results:
            summary["样本数"] = str(results["num_samples"])
        _print_summary("Corruption-based", summary)
    elif experiment_key == "contrastive_two_stage":
        results = run_contrastive_two_stage_experiment(
            config,
            device,
            device_map,
            seed=seed,
            config_dir=config_dir,
            project_root=project_root,
        )
        stage_a_losses = results.get("stage_a_losses", [])
        stage_b_losses = results.get("stage_b_losses", [])
        summary = {
            "数据集": str(results.get("dataset")),
            "阶段A最终Loss": f"{stage_a_losses[-1]:.4f}" if stage_a_losses else "未计算",
            "阶段B最终Loss": f"{stage_b_losses[-1]:.4f}" if stage_b_losses else "未计算",
        }
        if results.get("best_threshold_stage_a") is not None:
            summary["阶段A最佳阈值"] = f"{results['best_threshold_stage_a']:.4f}"
        if results.get("error_precision", 0.0) > 0:
            summary["阶段B错误检测Precision"] = f"{results.get('error_precision', 0.0):.2%}"
            summary["阶段B错误检测Recall"] = f"{results.get('error_recall', 0.0):.2%}"
            summary["阶段B错误检测F1"] = f"{results.get('error_f1', 0.0):.2%}"
            summary["包含错误的行数"] = str(results.get("num_error_rows", 0))

        _print_summary("Contrastive Two-Stage", summary)
    else:
        raise ValueError(
            f"不支持的实验类型：{experiment}. "
            "请使用 'corruption'、'contrastive'、'contrastive_two_stage' 或 'ablation'。"
        )

    print("\n提示：可通过修改配置文件快速调整组件与超参数。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Modal Error Detector Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hospital_two_stage_experiment.json"),
        help="配置文件路径（每个配置对应一个实验）",
    )
    args = parser.parse_args()
    main(args.config)

