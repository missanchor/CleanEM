#!/usr/bin/env python3
"""
基于ZeroED结果的Cross Attention检测器

正确的流程：
1. 运行ZeroED得到：特征、标签、聚类结果
2. 使用ZeroED的专业特征（feature_all_dict）作为Cross Attention的query
3. 生成文本描述获取LLM embedding作为Cross Attention的key和value
4. 训练Cross Attention检测器
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from cross_attention_detector import LLMBasedCrossAttentionDetector
from cross_modal_error_detector.utils.device import resolve_runtime_device
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)


def load_zeroed_results(resp_path):
    """
    加载ZeroED的结果

    Returns:
        dict: 包含所有必要的ZeroED结果
    """
    results = {}

    # 1. 加载clustering结果
    cluster_index_dict_path = os.path.join(resp_path, 'cluster_index_dict.json')
    with open(cluster_index_dict_path, 'r') as f:
        results['cluster_index_dict'] = json.load(f)

    # 2. 加载center_value_dict
    center_value_dict_path = os.path.join(resp_path, 'center_value_dict.json')
    with open(center_value_dict_path, 'r') as f:
        results['center_value_dict'] = json.load(f)

    # 3. 加载特征字典
    feature_dict_path = os.path.join(resp_path, 'cluster_feat_dict.pkl')
    with open(feature_dict_path, 'rb') as f:
        results['feature_all_dict'] = pickle.load(f)

    # 4. 加载标签结果
    error_checking_dir = os.path.join(resp_path, 'error_checking')
    results['error_checking_dir'] = error_checking_dir

    # 5. 加载聚类标注结果（如果有的话）
    # 这里应该从error_checking文件中解析得到center_index_value_label_dict

    LOGGER.info(f"Loaded ZeroED results from {resp_path}")
    LOGGER.info(f"  - cluster_index_dict keys: {list(results['cluster_index_dict'].keys())}")
    LOGGER.info(f"  - feature_all_dict size: {len(results['feature_all_dict'])}")

    return results


def extract_labels_from_zeroed(error_checking_dir, cluster_index_dict, center_value_dict):
    """
    从ZeroED的error_checking结果中提取标签
    正确实现：使用ZeroED/main.py中的extract_llm_label_res函数逻辑
    """
    import re
    import ast

    def normalize_string(s):
        return str(s.replace(" \\", "\\")
                   .replace("\\\\", "\\")
                   .replace("\\", "")
                   .replace(", ", ",")
                   .replace(": ", ":")
                   .replace("'", '"'))

    def err_pat_in_text_attr(attr):
        pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*true'
        return pattern

    def right_pat_in_text_attr(attr):
        pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*false'
        return pattern

    all_extracted_values = {}
    center_index_value_label_dict = {}

    for attr in cluster_index_dict.keys():
        content = ""
        error_checking_file = os.path.join(error_checking_dir, f'error_checking_{attr}.txt')

        if not os.path.exists(error_checking_file):
            LOGGER.warning(f"Error checking file not found: {error_checking_file}")
            continue

        with open(error_checking_file, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('\\+', '').replace('\\n', '\n')

        # 提取错误值
        wrong_pattern = err_pat_in_text_attr(attr)
        matches = re.finditer(wrong_pattern, content)
        wrong_values = [match.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'")
                       for match in matches]
        wrong_values = [normalize_string(match).replace('"{', '{', 1)[:-1] for match in wrong_values]
        wrong_values = list(set(wrong_values))

        # 提取正确值
        right_pattern = right_pat_in_text_attr(attr)
        right_matches = re.finditer(right_pattern, content)
        right_matches = [match.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'")
                        for match in right_matches]
        right_matches = [normalize_string(match).replace('"{', '{', 1)[:-1] for match in right_matches]

        # 过滤冲突项
        wrong_values = [extr_vals for extr_vals in wrong_values if extr_vals not in right_matches]

        # 匹配中心值并打标签
        center_values = center_value_dict[attr]
        labeled_centers = []

        for i, center_val in enumerate(center_values):
            center_idx = cluster_index_dict[attr][0][i]
            center_str = normalize_string(str(center_val))

            if center_str in wrong_values:
                labeled_centers.append((center_idx, center_val, 1))  # 错误
            else:
                labeled_centers.append((center_idx, center_val, 0))  # 正确

        center_index_value_label_dict[attr] = labeled_centers

    LOGGER.info(f"Extracted labels for {len(center_index_value_label_dict)} attributes from LLM")
    return center_index_value_label_dict


def label_propagation(cluster_index_dict, center_index_value_label_dict):
    """
    执行标签传播
    正确实现：使用ZeroED/main.py中的label_prop函数逻辑
    """
    det_wrong_list = []
    det_right_list = []

    for attr, clusters in cluster_index_dict.items():
        center_index = clusters[0]  # 聚类中心列表

        # 获取该属性的所有中心点标签
        for center_idx in center_index:
            # 查找中心点所属的簇
            temp_cluster = []
            for i in range(1, len(clusters)):
                if center_idx in clusters[i]:
                    temp_cluster.extend(clusters[i])
                    break

            # 查找中心点的标签
            temp_label = -1
            for triple_set in center_index_value_label_dict.get(attr, []):
                if triple_set[0] == center_idx:
                    temp_label = triple_set[2]
                    break

            # 将标签传播到整个簇
            if temp_label == 0:
                # 正确标签传播到簇中所有点
                for index in temp_cluster:
                    det_right_list.append((index, attr))
            elif temp_label == 1:
                # 错误标签传播到簇中所有点
                for index in temp_cluster:
                    det_wrong_list.append((index, attr))

    LOGGER.info(f"Label propagation completed:")
    LOGGER.info(f"  - Wrong cells: {len(det_wrong_list)}")
    LOGGER.info(f"  - Right cells: {len(det_right_list)}")

    return det_wrong_list, det_right_list


def build_training_data_zeroed(det_right_list, det_wrong_list, feature_all_dict, dirty_csv, all_attrs):
    """
    基于ZeroED结果构建训练数据

    Args:
        det_right_list: [(row_idx, attr), ...] 正确样本
        det_wrong_list: [(row_idx, attr), ...] 错误样本
        feature_all_dict: ZeroED提取的特征字典
        dirty_csv: 脏数据
        all_attrs: 所有列名

    Returns:
        tuple: (features, labels, texts, raw_values)
    """
    features = []
    labels = []
    texts = []
    raw_values = []

    # 收集所有样本
    all_samples = [(idx, attr, 0) for idx, attr in det_right_list]
    all_samples += [(idx, attr, 1) for idx, attr in det_wrong_list]

    LOGGER.info(f"Building training data from {len(all_samples)} samples...")

    for row_idx, attr, label in all_samples:
        col_idx = all_attrs.index(attr)

        # 获取ZeroED特征
        if (row_idx, col_idx) in feature_all_dict:
            feat_dict = feature_all_dict[(row_idx, col_idx)]

            # 组合所有特征
            feature_parts = []
            if 'occur_cnt_feat' in feat_dict:
                feature_parts.extend(feat_dict['occur_cnt_feat'])
            if 'pat_stats_feat' in feat_dict:
                feature_parts.extend(feat_dict['pat_stats_feat'])
            if 'fasttext_feat' in feat_dict:
                feature_parts.extend(feat_dict['fasttext_feat'])
            if 'pre_funcs_feat' in feat_dict:
                feature_parts.extend(feat_dict['pre_funcs_feat'])

            if feature_parts:
                features.append(feature_parts)

                # 生成文本描述（用于获取LLM embedding）
                raw_val = str(dirty_csv.iloc[row_idx, col_idx])
                text = f"The value '{raw_val}' in column '{attr}' is "
                text += "incorrect" if label == 1 else "correct"
                text += f". Feature context: occur_cnt={feat_dict.get('occur_cnt_feat', [0])[0] if 'occur_cnt_feat' in feat_dict else 'N/A'}"

                texts.append(text)
                raw_values.append(raw_val)
                labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    LOGGER.info(f"Training data built:")
    LOGGER.info(f"  - Features shape: {features.shape}")
    LOGGER.info(f"  - Labels shape: {labels.shape}")
    LOGGER.info(f"  - Error rate: {labels.mean()*100:.1f}%")

    return features, labels, texts, raw_values


def run_zeroed_cross_attention_comparison(resp_path, llm_model_path="/mnt/data/welkinni/Qwen2.5-3B/qwen/Qwen2.5-3B"):
    """
    运行基于ZeroED结果的Cross Attention对比实验
    """
    LOGGER.info("="*80)
    LOGGER.info("Cross Attention vs MLP on ZeroED Results")
    LOGGER.info("="*80)

    # 1. 加载ZeroED结果
    LOGGER.info("\n1. Loading ZeroED results...")
    zeroed_results = load_zeroed_results(resp_path)

    # 2. 加载脏数据
    data_dir = os.path.join(resp_path, '..', '..', 'data')
    dirty_files = [f for f in os.listdir(data_dir) if 'error-01.csv' in f]
    if dirty_files:
        dirty_path = os.path.join(data_dir, dirty_files[0])
        dirty_csv = pd.read_csv(dirty_path, dtype=str).fillna('nan')
        all_attrs = list(dirty_csv.columns)
        LOGGER.info(f"Loaded dirty data: {dirty_csv.shape}")
    else:
        raise ValueError("No dirty data file found")

    # 3. 提取标签（简化版）
    LOGGER.info("\n2. Extracting labels from ZeroED...")
    center_index_value_label_dict = extract_labels_from_zeroed(
        zeroed_results['error_checking_dir'],
        zeroed_results['cluster_index_dict'],
        zeroed_results['center_value_dict']
    )

    # 4. 标签传播
    LOGGER.info("\n3. Propagating labels...")
    det_wrong_list, det_right_list = label_propagation(
        zeroed_results['cluster_index_dict'],
        center_index_value_label_dict
    )

    # 5. 构建训练数据（使用ZeroED特征）
    LOGGER.info("\n4. Building training data from ZeroED features...")
    features, labels, texts, raw_values = build_training_data_zeroed(
        det_right_list,
        det_wrong_list,
        zeroed_results['feature_all_dict'],
        dirty_csv,
        all_attrs
    )

    if len(features) == 0:
        raise ValueError("No training data extracted!")

    # 6. 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        features, labels, texts, test_size=0.3, random_state=42, stratify=labels
    )

    LOGGER.info(f"  - Train: {len(X_train)} samples")
    LOGGER.info(f"  - Test: {len(X_test)} samples")

    # 7. 训练MLP（基线）
    LOGGER.info("\n5. Training MLP baseline...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    LOGGER.info(f"  ✓ MLP Accuracy: {mlp_acc:.4f}")

    # 8. 训练Cross Attention
    LOGGER.info("\n6. Training Cross Attention detector...")
    detector = LLMBasedCrossAttentionDetector(
        model_name=llm_model_path,
        device=resolve_runtime_device("cuda")
    )

    detector.load_llm()
    detector.initialize_detection_head(feature_dim=X_train.shape[1])

    detector.train(
        train_features=X_train,
        train_labels=y_train,
        train_texts=texts_train,
        epochs=30,
        batch_size=16,
        lr=1e-4
    )

    cross_attn_preds = detector.predict(X_test, texts_test)
    cross_attn_acc = accuracy_score(y_test, cross_attn_preds)
    LOGGER.info(f"  ✓ Cross Attention Accuracy: {cross_attn_acc:.4f}")

    # 9. 对比结果
    LOGGER.info("\n" + "="*80)
    LOGGER.info("RESULTS")
    LOGGER.info("="*80)
    LOGGER.info(f"MLP Accuracy:           {mlp_acc:.4f}")
    LOGGER.info(f"Cross Attention:        {cross_attn_acc:.4f}")
    LOGGER.info(f"Improvement:            {(cross_attn_acc - mlp_acc)*100:.2f}%")

    if cross_attn_acc > mlp_acc:
        LOGGER.info("✓ Cross Attention performs better!")
    else:
        LOGGER.info("✗ MLP performs better.")

    # 10. 保存结果
    results = {
        'mlp_accuracy': float(mlp_acc),
        'cross_attention_accuracy': float(cross_attn_acc),
        'improvement': float(cross_attn_acc - mlp_acc),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'error_rate': float(labels.mean())
    }

    import json
    output_path = os.path.join(resp_path, 'cross_attention_vs_mlp.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    LOGGER.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Cross Attention on ZeroED results")
    parser.add_argument('--resp_path', type=str, required=True,
                        help='Path to ZeroED results directory')
    parser.add_argument('--llm_model', type=str,
                        default="/mnt/data/welkinni/Qwen2.5-3B/qwen/Qwen2.5-3B",
                        help='Path to LLM model')

    args = parser.parse_args()

    try:
        results = run_zeroed_cross_attention_comparison(
            resp_path=args.resp_path,
            llm_model_path=args.llm_model
        )
        LOGGER.info("\n" + "="*80)
        LOGGER.info("Experiment completed successfully!")
        LOGGER.info("="*80)
    except Exception as e:
        LOGGER.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
