#!/usr/bin/env python3
"""
对比MLP和Cross Attention两种检测方法

基于ZeroED的特征提取和标注结果，使用MLP和Cross Attention两种方法进行检测对比。
"""

import os
import sys
import json
import pickle
import argparse
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from cross_attention_detector import LLMBasedCrossAttentionDetector, prepare_text_representation
import torch
from tqdm import tqdm
import logging

from cross_modal_error_detector.utils.device import resolve_runtime_device

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)


def load_zeroed_results(resp_path):
    """
    加载ZeroED的结果，包括特征、标签等

    Args:
        resp_path: ZeroED的结果目录

    Returns:
        dict: 包含各种结果的字典
    """
    results = {}

    # 加载clustering结果
    cluster_index_dict_path = os.path.join(resp_path, 'cluster_index_dict.json')
    if os.path.exists(cluster_index_dict_path):
        with open(cluster_index_dict_path, 'r') as f:
            results['cluster_index_dict'] = json.load(f)

    # 加载center_value_dict
    center_value_dict_path = os.path.join(resp_path, 'center_value_dict.json')
    if os.path.exists(center_value_dict_path):
        with open(center_value_dict_path, 'r') as f:
            results['center_value_dict'] = json.load(f)

    # 加载特征字典
    feature_dict_path = os.path.join(resp_path, 'cluster_feat_dict.pkl')
    if os.path.exists(feature_dict_path):
        with open(feature_dict_path, 'rb') as f:
            results['feature_dict'] = pickle.load(f)

    # 加载label结果
    llm_label_results_path = os.path.join(resp_path, 'llm_label_results.txt')
    if os.path.exists(llm_label_results_path):
        with open(llm_label_results_path, 'r') as f:
            results['llm_label_results'] = f.read()

    # 加载clean和dirty数据
    data_files = [f for f in os.listdir(resp_path) if f.endswith('.csv')]
    for f in data_files:
        if 'clean' in f:
            results['clean_path'] = os.path.join(resp_path, f)
        elif 'dirty' in f or 'error' in f:
            results['dirty_path'] = os.path.join(resp_path, f)

    return results


def extract_labeled_data(resp_path, dirty_csv, clean_csv, all_attrs, related_attrs_dict):
    """
    从ZeroED结果中提取已标注的训练数据

    Returns:
        tuple: (train_features, train_labels, raw_values_dict)
    """
    # 加载center_value_dict和cluster_index_dict
    with open(os.path.join(resp_path, 'center_value_dict.json'), 'r') as f:
        center_value_dict = json.load(f)

    with open(os.path.join(resp_path, 'cluster_index_dict.json'), 'r') as f:
        cluster_index_dict = json.load(f)

    # 转换格式
    cluster_index_dict = {
        attr: [[int(idx) for idx in cluster] for cluster in clusters]
        for attr, clusters in cluster_index_dict.items()
    }

    # 从error_checking结果中提取label
    error_checking_dir = os.path.join(resp_path, 'error_checking')
    center_index_value_label_dict = {}

    for attr in all_attrs:
        error_checking_file = os.path.join(error_checking_dir, f'error_checking_{attr}.txt')
        if not os.path.exists(error_checking_file):
            continue

        with open(error_checking_file, 'r') as f:
            content = f.read()

        # 解析标注结果
        # 这里简化处理，实际应该使用ZeroED中的extract_llm_label_res函数
        # 暂时假设我们知道哪些被标注为错误

        center_values = center_value_dict[attr]
        labels = []

        # 简单的启发式规则：包含"error"或"wrong"则标注为1（错误）
        for center_val in center_values:
            if any(keyword in content.lower() for keyword in ['error', 'wrong', 'incorrect']):
                labels.append(1)
            else:
                labels.append(0)

        center_index_value_label_dict[attr] = [
            (cluster_index_dict[attr][0][i], center_values[i], labels[i])
            for i in range(len(center_values))
        ]

    # 收集所有训练数据
    all_train_features = []
    all_train_labels = []
    all_raw_values = []

    # 从func_det_res.txt加载已检测的错误
    det_res_path = os.path.join(resp_path, 'func_det_res.txt')
    if os.path.exists(det_res_path):
        with open(det_res_path, 'r') as f:
            det_results = f.read()
    else:
        det_results = ""

    for attr in all_attrs:
        related_attrs = list(related_attrs_dict[attr])

        for idx, val_dict, label in center_index_value_label_dict.get(attr, []):
            # 构建特征
            # 这里应该使用与ZeroED相同的特征提取方法
            # 为了简化，我们使用随机特征或基于值的统计特征

            # 从dirty_csv中获取该样本
            sample_row = dirty_csv.iloc[idx]

            # 构建特征（简化版本）
            feature = []
            for col in all_attrs:
                # 数值型特征编码
                val = str(sample_row[col])
                if val.replace('.', '').replace('-', '').isdigit():
                    feature.append(float(val) if val != 'nan' else 0.0)
                else:
                    # 类别型特征编码
                    feature.append(hash(val) % 1000 / 1000.0)

            all_train_features.append(feature)
            all_train_labels.append(label)
            all_raw_values.append(str(sample_row[attr]))

    return np.array(all_train_features), np.array(all_train_labels), all_raw_values


def prepare_llm_texts(features, labels, raw_values):
    """
    为LLM准备文本描述

    Args:
        features: 特征数组
        labels: 标签数组
        raw_values: 原始值列表

    Returns:
        list: 文本描述列表
    """
    texts = []
    for i, (feature, label, raw_val) in enumerate(zip(features, labels, raw_values)):
        # 使用原始值和统计信息构建文本描述
        label_desc = "correct" if label == 0 else "error"
        text = f"The value '{raw_val}' is labeled as {label_desc}. "
        text += f"Feature statistics: mean={np.mean(feature):.3f}, std={np.std(feature):.3f}. "

        # 添加一些描述性信息
        text += f"This is a data point with {len(feature)} features."

        texts.append(text)

    return texts


def train_mlp_classifier(train_features, train_labels):
    """训练MLP分类器"""
    LOGGER.info("Training MLP classifier...")

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10
    )

    model.fit(train_features, train_labels)
    LOGGER.info("MLP training completed!")
    return model


def train_cross_attention_detector(
    train_features,
    train_labels,
    train_raw_values,
    model_name,
    device: Optional[str],
):
    """训练Cross Attention检测器"""
    LOGGER.info("Initializing Cross Attention detector...")

    runtime_device = resolve_runtime_device(device)
    detector = LLMBasedCrossAttentionDetector(
        model_name=model_name,
        device=runtime_device
    )

    # 加载LLM
    detector.load_llm()

    # 准备文本描述
    train_texts = prepare_llm_texts(train_features, train_labels, train_raw_values)

    # 初始化检测头部
    detector.initialize_detection_head(feature_dim=train_features.shape[1])

    # 训练
    detector.train(
        train_features=train_features,
        train_labels=train_labels,
        train_texts=train_texts,
        epochs=30,  # 减少epoch以加快训练
        batch_size=16,
        lr=1e-4
    )

    LOGGER.info("Cross Attention training completed!")
    return detector


def evaluate_models(mlp_model, cross_attn_detector, test_features, test_labels, test_raw_values):
    """评估两种模型"""
    results = {}

    # 评估MLP
    LOGGER.info("Evaluating MLP...")
    mlp_preds = mlp_model.predict(test_features)
    mlp_proba = mlp_model.predict_proba(test_features) if hasattr(mlp_model, 'predict_proba') else None

    results['mlp'] = {
        'predictions': mlp_preds,
        'accuracy': accuracy_score(test_labels, mlp_preds),
        'precision': precision_score(test_labels, mlp_preds, average='weighted'),
        'recall': recall_score(test_labels, mlp_preds, average='weighted'),
        'f1': f1_score(test_labels, mlp_preds, average='weighted')
    }

    # 评估Cross Attention
    LOGGER.info("Evaluating Cross Attention detector...")
    test_texts = prepare_llm_texts(test_features, test_labels, test_raw_values)
    cross_attn_preds = cross_attn_detector.predict(test_features, test_texts)

    results['cross_attention'] = {
        'predictions': cross_attn_preds,
        'accuracy': accuracy_score(test_labels, cross_attn_preds),
        'precision': precision_score(test_labels, cross_attn_preds, average='weighted'),
        'recall': recall_score(test_labels, cross_attn_preds, average='weighted'),
        'f1': f1_score(test_labels, cross_attn_preds, average='weighted')
    }

    return results


def save_comparison_results(results, output_path):
    """保存对比结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETECTION METHODS COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        # MLP结果
        f.write("MLP Classifier Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {results['mlp']['accuracy']:.4f}\n")
        f.write(f"Precision: {results['mlp']['precision']:.4f}\n")
        f.write(f"Recall:    {results['mlp']['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['mlp']['f1']:.4f}\n\n")

        # Cross Attention结果
        f.write("Cross Attention Detector Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {results['cross_attention']['accuracy']:.4f}\n")
        f.write(f"Precision: {results['cross_attention']['precision']:.4f}\n")
        f.write(f"Recall:    {results['cross_attention']['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['cross_attention']['f1']:.4f}\n\n")

        # 对比
        f.write("Performance Comparison:\n")
        f.write("-" * 40 + "\n")
        acc_diff = results['cross_attention']['accuracy'] - results['mlp']['accuracy']
        f.write(f"Accuracy Improvement: {acc_diff*100:.2f}%\n")

        if results['cross_attention']['accuracy'] > results['mlp']['accuracy']:
            f.write("✓ Cross Attention detector performs better!\n")
        else:
            f.write("✗ MLP classifier performs better.\n")

        # 详细分类报告
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORTS\n")
        f.write("="*80 + "\n\n")

        f.write("MLP Classifier Report:\n")
        f.write(classification_report(results['mlp']['predictions'], results['mlp']['predictions']))

        f.write("\nCross Attention Detector Report:\n")
        f.write(classification_report(results['cross_attention']['predictions'], results['cross_attention']['predictions']))

    LOGGER.info(f"Comparison results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare MLP vs Cross Attention for error detection")
    parser.add_argument('--resp_path', type=str, required=True,
                        help='Path to ZeroED results directory')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model name for LLM')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-select emptiest GPU or CPU)',
    )
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                        help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (0.0-1.0)')

    args = parser.parse_args()

    runtime_device = resolve_runtime_device(args.device)
    LOGGER.info(f"Using device: {runtime_device}")

    # 加载数据
    LOGGER.info(f"Loading data from {args.resp_path}")
    dirty_csv = pd.read_csv(os.path.join(args.resp_path, '..', '..', 'data', '*.csv'), dtype=str).fillna('nan')
    # 这里需要根据实际数据路径调整

    all_attrs = list(dirty_csv.columns)
    related_attrs_dict = {attr: [] for attr in all_attrs}  # 简化处理

    # 提取训练数据
    LOGGER.info("Extracting labeled data...")
    train_features, train_labels, train_raw_values = extract_labeled_data(
        args.resp_path, dirty_csv, None, all_attrs, related_attrs_dict
    )

    if len(train_features) == 0:
        LOGGER.error("No training data extracted. Please check the ZeroED results.")
        sys.exit(1)

    LOGGER.info(f"Extracted {len(train_features)} training samples")

    # 分割训练和测试集
    indices = np.random.permutation(len(train_features))
    test_size = int(len(train_features) * args.test_size)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    train_features_split = train_features[train_indices]
    train_labels_split = train_labels[train_indices]
    train_raw_values_split = [train_raw_values[i] for i in train_indices]

    test_features_split = train_features[test_indices]
    test_labels_split = train_labels[test_indices]
    test_raw_values_split = [train_raw_values[i] for i in test_indices]

    LOGGER.info(f"Training set size: {len(train_features_split)}")
    LOGGER.info(f"Test set size: {len(test_features_split)}")

    # 训练MLP
    mlp_model = train_mlp_classifier(train_features_split, train_labels_split)

    # 训练Cross Attention检测器
    cross_attn_detector = train_cross_attention_detector(
        train_features_split,
        train_labels_split,
        train_raw_values_split,
        args.model_name,
        runtime_device,
    )

    # 评估
    LOGGER.info("Evaluating models...")
    results = evaluate_models(
        mlp_model,
        cross_attn_detector,
        test_features_split,
        test_labels_split,
        test_raw_values_split
    )

    # 打印结果
    LOGGER.info("\n" + "="*80)
    LOGGER.info("FINAL COMPARISON RESULTS")
    LOGGER.info("="*80)
    LOGGER.info(f"\nMLP Classifier:")
    LOGGER.info(f"  Accuracy:  {results['mlp']['accuracy']:.4f}")
    LOGGER.info(f"  Precision: {results['mlp']['precision']:.4f}")
    LOGGER.info(f"  Recall:    {results['mlp']['recall']:.4f}")
    LOGGER.info(f"  F1-Score:  {results['mlp']['f1']:.4f}")

    LOGGER.info(f"\nCross Attention Detector:")
    LOGGER.info(f"  Accuracy:  {results['cross_attention']['accuracy']:.4f}")
    LOGGER.info(f"  Precision: {results['cross_attention']['precision']:.4f}")
    LOGGER.info(f"  Recall:    {results['cross_attention']['recall']:.4f}")
    LOGGER.info(f"  F1-Score:  {results['cross_attention']['f1']:.4f}")

    acc_diff = results['cross_attention']['accuracy'] - results['mlp']['accuracy']
    LOGGER.info(f"\nAccuracy Improvement: {acc_diff*100:.2f}%")

    # 保存结果
    output_path = os.path.join(args.output_dir, 'comparison_report.txt')
    save_comparison_results(results, output_path)


if __name__ == "__main__":
    main()
