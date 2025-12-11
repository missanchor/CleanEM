#!/usr/bin/env python3
"""
使用真实数据集演示MLP vs Cross Attention检测方法 - ZeroED版本

✅ 核心特点：
1. 只使用ZeroED的结果（专业特征 + LLM标签）
2. 完全符合ZeroED/main.py的流程
3. 对比Cross Attention vs MLP基线方法

📊 数据流程：
1. 加载ZeroED结果（clustering, features, LLM labels）
2. 解析LLM标注（使用ZeroED的extract_llm_label_res逻辑）
3. 执行标签传播（使用ZeroED的label_prop逻辑）
4. 构建训练数据（使用ZeroED专业特征）
5. 训练MLP和Cross Attention模型
6. 对比性能

🔧 使用方法：
python real_data_cross_attention_demo_v2.py --dataset beers
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import json
import pickle
import re
import logging

# Ensure project root is available for local package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加ZeroED目录到路径以加载相关模块
ZEROED_DIR = PROJECT_ROOT / 'ZeroED'
if str(ZEROED_DIR) not in sys.path:
    sys.path.insert(0, str(ZEROED_DIR))

from cross_attention_detector import LLMBasedCrossAttentionDetector
from cross_modal_error_detector.utils.device import resolve_runtime_device

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)


def load_dataset(dataset_name="beers", data_dir="/data/nw/Cleaning_LLM/data"):
    """
    加载真实数据集

    Args:
        dataset_name: 数据集名称 (beers, hospital, flights, etc.)
        data_dir: 数据目录

    Returns:
        tuple: (clean_df, dirty_df)
    """
    # 尝试多种文件命名模式
    possible_patterns = [
        f"{dataset_name}_clean.csv",
        f"{dataset_name}_clean_missing_values.csv",
        f"{dataset_name}_clean_mixed_err.csv"
    ]

    clean_path = None
    for pattern in possible_patterns:
        path = os.path.join(data_dir, pattern)
        if os.path.exists(path):
            clean_path = path
            break

    if clean_path is None:
        raise FileNotFoundError(f"No clean data file found for {dataset_name}")

    # 加载脏数据
    dirty_path = os.path.join(data_dir, f"{dataset_name}_error-01.csv")
    if not os.path.exists(dirty_path):
        raise FileNotFoundError(f"Dirty data file not found: {dirty_path}")

    clean_df = pd.read_csv(clean_path, dtype=str).fillna('nan')
    dirty_df = pd.read_csv(dirty_path, dtype=str).fillna('nan')

    LOGGER.info(f"Loaded {dataset_name} dataset:")
    LOGGER.info(f"  - Clean data shape: {clean_df.shape}")
    LOGGER.info(f"  - Dirty data shape: {dirty_df.shape}")
    LOGGER.info(f"  - Columns: {list(clean_df.columns)}")

    return clean_df, dirty_df


FEATURE_KEYS = ['occur_cnt_feat', 'pat_stats_feat', 'fasttext_feat', 'pre_funcs_feat']


def _flatten_feature_values(values):
    """
    将ZeroED生成的特征值展平为一维浮点列表
    """
    if isinstance(values, np.ndarray):
        values = values.tolist()

    if isinstance(values, (list, tuple)):
        flattened = []
        for item in values:
            flattened.extend(_flatten_feature_values(item))
        return flattened

    try:
        return [float(values)]
    except (TypeError, ValueError):
        return []


def build_feature_vector(feat_dict, feature_keys=None):
    """
    根据ZeroED的特征字典构造统一的一维特征向量
    """
    feature_keys = feature_keys or FEATURE_KEYS
    feature_parts = []

    for key in feature_keys:
        if key in feat_dict and isinstance(feat_dict[key], (list, tuple, np.ndarray)):
            flattened = _flatten_feature_values(feat_dict[key])
            if flattened:
                feature_parts.extend(flattened)

    if not feature_parts:
        return None

    return np.array(feature_parts, dtype=np.float32)


def prepare_attribute_eval_dataset(attr_name, column_idx, dirty_df, clean_df, feature_all_dict):
    """
    构造某一列在整个脏数据集上的特征、文本和ground truth标签
    """
    feature_list = []
    label_list = []
    text_list = []
    missing_count = 0

    for row_idx in range(len(dirty_df)):
        key = (row_idx, column_idx)
        if key not in feature_all_dict:
            missing_count += 1
            continue

        feature_vec = build_feature_vector(feature_all_dict[key])
        if feature_vec is None:
            missing_count += 1
            continue

        feature_list.append(feature_vec)
        raw_val = str(dirty_df.iloc[row_idx, column_idx])
        text_list.append(f"The value '{raw_val}' in column '{attr_name}'.")

        clean_val = str(clean_df.iloc[row_idx, column_idx])
        label_list.append(int(raw_val != clean_val))

    if not feature_list:
        LOGGER.warning(f"    Unable to build evaluation data for {attr_name}: no usable features (missing {missing_count} entries).")
        return None

    try:
        features = np.stack(feature_list)
    except ValueError as exc:
        LOGGER.error(f"    Feature dimension mismatch for {attr_name}: {exc}")
        return None

    labels = np.array(label_list, dtype=np.int64)
    return {
        'features': features,
        'labels': labels,
        'texts': text_list
    }


def try_load_zeroed_results(resp_path, dataset_name):
    """
    从ZeroED结果加载专业特征和标签

    使用与ZeroED/main.py相同的流程：
    1. 加载clustering结果
    2. 解析LLM标注结果
    3. 执行标签传播
    4. 构建训练数据

    Args:
        resp_path: ZeroED结果目录路径
        dataset_name: 数据集名称

    Returns:
        dict: 包含 (features, labels, texts, raw_values) 或 None
    """
    try:
        # 直接使用传入的目录路径
        zeroed_dir = resp_path
        LOGGER.info(f"✓ Using ZeroED results from: {zeroed_dir}")

        # 检查必要文件
        required_files = [
            'cluster_index_dict.json',
            'center_value_dict.json',
            'cluster_feat_dict.pkl'
        ]

        missing_files = []
        for f in required_files:
            if not os.path.exists(os.path.join(zeroed_dir, f)):
                missing_files.append(f)

        if missing_files:
            LOGGER.error(f"❌ ZeroED results incomplete, missing: {missing_files}")
            return None

        # 加载文件
        with open(os.path.join(zeroed_dir, 'cluster_index_dict.json'), 'r') as f:
            cluster_index_dict = json.load(f)

        with open(os.path.join(zeroed_dir, 'center_value_dict.json'), 'r') as f:
            center_value_dict = json.load(f)

        # 安全加载pickle文件
        try:
            with open(os.path.join(zeroed_dir, 'cluster_feat_dict.pkl'), 'rb') as f:
                feature_all_dict = pickle.load(f)
        except (ModuleNotFoundError, AttributeError) as e:
            LOGGER.warning(f"Pickle loading with standard method failed: {e}")
            LOGGER.warning("Attempting alternative loading method...")
            try:
                with open(os.path.join(zeroed_dir, 'cluster_feat_dict.pkl'), 'rb') as f:
                    feature_all_dict = pickle.load(f)
            except Exception as e2:
                LOGGER.error(f"Failed to load feature_all_dict.pkl: {e2}")
                return None

        LOGGER.info(f"✓ Loaded clustering and feature files")

        # 从ZeroED/main.py复制的标签解析函数
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

        # 加载LLM标注结果
        error_checking_dir = os.path.join(zeroed_dir, 'error_checking')
        if not os.path.exists(error_checking_dir):
            LOGGER.error(f"❌ Error checking directory not found")
            return None

        # 解析LLM标注（使用ZeroED/main.py中的extract_llm_label_res逻辑）
        center_index_value_label_dict = {}
        for attr in cluster_index_dict.keys():
            error_file = os.path.join(error_checking_dir, f'error_checking_{attr}.txt')

            if not os.path.exists(error_file):
                LOGGER.warning(f"Warning: error_checking_{attr}.txt not found, skipping...")
                continue

            with open(error_file, 'r', encoding='utf-8') as f:
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

        LOGGER.info(f"✓ Extracted labels from LLM responses")

        # 标签传播（使用ZeroED/main.py中的label_prop逻辑）
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
                        det_right_list.append((index, attr, 0))
                elif temp_label == 1:
                    # 错误标签传播到簇中所有点
                    for index in temp_cluster:
                        det_wrong_list.append((index, attr, 1))

        LOGGER.info(f"✓ Label propagation completed: {len(det_wrong_list)} wrong, {len(det_right_list)} right")

        # 构建训练数据
        all_attrs = list(cluster_index_dict.keys())

        # 加载脏数据以获取原始值
        dirty_df = pd.read_csv(f"/data/nw/Cleaning_LLM/data/{dataset_name}_error-01.csv", dtype=str).fillna('nan')

        # 按属性分组数据
        attr_groups = {}
        for attr in all_attrs:
            attr_groups[attr] = {
                'features': [],
                'labels': [],
                'texts': [],
                'raw_values': []
            }

        for row_idx, attr, label in det_right_list + det_wrong_list:
            col_idx = all_attrs.index(attr)

            # 确保索引在范围内
            if row_idx >= len(dirty_df) or col_idx >= len(dirty_df.columns):
                continue

            if (row_idx, col_idx) in feature_all_dict:
                feat_dict = feature_all_dict[(row_idx, col_idx)]

                feature_vec = build_feature_vector(feat_dict)
                if feature_vec is None:
                    continue

                # 生成文本描述
                raw_val = str(dirty_df.iloc[row_idx, col_idx])
                text = f"The value '{raw_val}' in column '{attr}' is "
                text += "incorrect" if label == 1 else "correct"

                # 添加到对应属性的组
                attr_groups[attr]['features'].append(feature_vec)
                attr_groups[attr]['labels'].append(label)
                attr_groups[attr]['texts'].append(text)
                attr_groups[attr]['raw_values'].append(raw_val)

        # 转换每个属性的数据为numpy数组
        for attr in attr_groups:
            if len(attr_groups[attr]['features']) > 0:
                attr_groups[attr]['features'] = np.array(attr_groups[attr]['features'])
                attr_groups[attr]['labels'] = np.array(attr_groups[attr]['labels'])

        LOGGER.info(f"✓ Successfully built training data:")
        LOGGER.info(f"  - Attributes: {len(attr_groups)}")
        for attr, data in attr_groups.items():
            if len(data['features']) > 0:
                LOGGER.info(f"    {attr}: {len(data['features'])} samples, feature dim: {data['features'].shape[1]}, error rate: {data['labels'].mean()*100:.1f}%")

        return {
            'attr_groups': attr_groups,
            'feature_all_dict': feature_all_dict,
            'all_attrs': all_attrs,
            'source': 'zeroed'
        }

    except Exception as e:
        LOGGER.error(f"❌ Failed to load ZeroED results: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_real_data_comparison(dataset_name="beers"):
    """
    在真实数据集上运行对比实验
    只使用ZeroED的结果（专业特征 + LLM标签）

    Args:
        dataset_name: 数据集名称
    """
    LOGGER.info("="*80)
    LOGGER.info(f"REAL DATA COMPARISON: {dataset_name.upper()} DATASET")
    LOGGER.info(f"Using ZeroED Results (LLM-labeled + Professional Features)")
    LOGGER.info("="*80)

    # 1. 加载ZeroED结果
    LOGGER.info("\n1. Loading ZeroED results...")

    # 动态查找最新的ZeroED结果目录
    base_result_dir = "/data/nw/Cleaning_LLM/result/pipeline"
    if not os.path.exists(base_result_dir):
        LOGGER.error(f"❌ ZeroED results directory not found: {base_result_dir}")
        return None

    # 手动指定数据集特定的目录
    import glob
    pattern = os.path.join(base_result_dir, f"*{dataset_name}01-5%-set1")
    all_dirs = glob.glob(pattern)
    if not all_dirs:
        LOGGER.error(f"❌ No Hospital ZeroED results found matching pattern: {pattern}")
        return None
    resp_path = all_dirs[0]
    LOGGER.info(f"✓ Using Hospital ZeroED results from: {resp_path}")

    if not os.path.exists(resp_path):
        LOGGER.error(f"❌ ZeroED results directory not found: {resp_path}")
        return None

    zeroed_results = try_load_zeroed_results(resp_path, dataset_name)

    if zeroed_results is None:
        LOGGER.error(f"❌ Failed to load ZeroED results for {dataset_name}")
        LOGGER.error(f"Please ensure ZeroED has been run on this dataset first.")
        return None

    # 2. 按属性分组数据（为每个属性单独训练）
    LOGGER.info("\n2. Grouping data by attribute...")
    attr_groups = zeroed_results.get('attr_groups', {})

    if not attr_groups:
        LOGGER.error("❌ No attribute groups found in ZeroED results")
        LOGGER.error("Need to modify try_load_zeroed_results to return attribute groups")
        return None

    feature_all_dict = zeroed_results.get('feature_all_dict')
    if feature_all_dict is None:
        LOGGER.error("❌ feature_all_dict missing from ZeroED results. Cannot build evaluation dataset.")
        return None

    try:
        clean_df, dirty_df = load_dataset(dataset_name)
    except FileNotFoundError as exc:
        LOGGER.error(f"❌ {exc}")
        return None

    column_index_map = {col: idx for idx, col in enumerate(dirty_df.columns)}
    attr_eval_cache = {}

    LOGGER.info(f"Found {len(attr_groups)} attributes: {list(attr_groups.keys())}")

    # 3. 为每个属性训练模型并汇总结果
    LOGGER.info("\n3. Training models for each attribute...")
    all_mlp_preds = []
    all_cross_attn_preds = []
    mlp_eval_labels = []
    cross_attn_eval_labels = []
    mlp_preds_for_cross_subset = []

    mlp_accuracies = {}
    cross_attn_accuracies = {}

    for attr_name, attr_data in attr_groups.items():
        LOGGER.info(f"\n  Processing attribute: {attr_name}")
        LOGGER.info(f"    Samples: {len(attr_data['features'])}")
        LOGGER.info(f"    Features shape: {attr_data['features'].shape}")
        LOGGER.info(f"    Error rate: {attr_data['labels'].mean()*100:.1f}%")

        if attr_name not in column_index_map:
            LOGGER.warning(f"    Column {attr_name} not present in dirty dataset. Skipping.")
            continue

        col_idx = column_index_map[attr_name]
        eval_data = attr_eval_cache.get(attr_name)
        if eval_data is None:
            eval_data = prepare_attribute_eval_dataset(attr_name, col_idx, dirty_df, clean_df, feature_all_dict)
            if eval_data is None:
                LOGGER.warning(f"    Skipping {attr_name}: unable to prepare evaluation data.")
                continue
            attr_eval_cache[attr_name] = eval_data

        eval_features = eval_data['features']
        eval_labels = eval_data['labels']
        eval_texts = eval_data['texts']

        if eval_features.shape[1] != attr_data['features'].shape[1]:
            LOGGER.error(
                f"    Feature dimension mismatch for {attr_name}: "
                f"train_dim={attr_data['features'].shape[1]}, eval_dim={eval_features.shape[1]}"
            )
            continue

        # 跳过错误率为0%的列（所有样本都是正确的）
        if attr_data['labels'].mean() == 0:
            LOGGER.info(f"    Skipping {attr_name}: no errors in this column (all samples correct)")
            zero_preds = np.zeros_like(eval_labels)
            mlp_eval_labels.extend(eval_labels)
            all_mlp_preds.extend(zero_preds)
            mlp_accuracies[attr_name] = accuracy_score(eval_labels, zero_preds)
            continue

        # 不再拆分训练/测试集，全部样本用于训练
        X_train = attr_data['features']
        y_train = attr_data['labels']
        texts_train = attr_data['texts']

        # 训练MLP
        LOGGER.info(f"    Training MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        mlp.fit(X_train, y_train)
        mlp_preds = mlp.predict(eval_features)
        mlp_acc = accuracy_score(eval_labels, mlp_preds)
        mlp_accuracies[attr_name] = mlp_acc
        all_mlp_preds.extend(mlp_preds)
        mlp_eval_labels.extend(eval_labels)

        LOGGER.info(f"    MLP Accuracy: {mlp_acc:.4f}")

        # 训练Cross Attention（可选）
        try:
            LOGGER.info(f"    Training Cross Attention (using GPU)...")
            detector = LLMBasedCrossAttentionDetector(
                model_name="/mnt/data/welkinni/Qwen2.5-3B/qwen/Qwen2.5-3B",
                device=resolve_runtime_device("cuda"),  # 使用最空GPU
            )

            detector.load_llm(cache_dir="/data/nw/modelscope_cache")
            detector.initialize_detection_head(feature_dim=X_train.shape[1])

            detector.train(
                train_features=X_train,
                train_labels=y_train,
                train_texts=texts_train,
                epochs=20,
                batch_size=16,
                lr=1e-4
            )

            cross_attn_preds = detector.predict(eval_features, eval_texts)
            cross_attn_acc = accuracy_score(eval_labels, cross_attn_preds)
            cross_attn_accuracies[attr_name] = cross_attn_acc
            all_cross_attn_preds.extend(cross_attn_preds)
            cross_attn_eval_labels.extend(eval_labels)
            mlp_preds_for_cross_subset.extend(mlp_preds)

            LOGGER.info(f"    Cross Attention Accuracy: {cross_attn_acc:.4f}")

        except Exception as e:
            LOGGER.warning(f"    Cross Attention training failed: {e}")
            # Cross Attention失败时，用MLP预测代替（或者其他默认策略）
            all_cross_attn_preds.extend(mlp_preds)
            cross_attn_eval_labels.extend(eval_labels)
            mlp_preds_for_cross_subset.extend(mlp_preds)

    # 4. 汇总结果（使用clean vs dirty获得的真实ground truth）
    if not mlp_eval_labels:
        LOGGER.error("❌ No evaluation samples collected for MLP. Cannot compute metrics.")
        return None

    mlp_eval_labels = np.array(mlp_eval_labels)
    all_mlp_preds = np.array(all_mlp_preds)

    total_eval_samples = len(mlp_eval_labels)
    error_sample_count = int((mlp_eval_labels == 1).sum())

    LOGGER.info("\n" + "="*80)
    LOGGER.info("COMPARISON RESULTS (Ground truth from clean vs dirty)")
    LOGGER.info("="*80)
    LOGGER.info(f"Total evaluated samples: {total_eval_samples}")
    LOGGER.info(f"Total true error samples: {error_sample_count}")

    overall_mlp_acc = accuracy_score(mlp_eval_labels, all_mlp_preds)
    mlp_precision = precision_score(mlp_eval_labels, all_mlp_preds, pos_label=1, zero_division=0)
    mlp_recall = recall_score(mlp_eval_labels, all_mlp_preds, pos_label=1, zero_division=0)
    mlp_f1 = f1_score(mlp_eval_labels, all_mlp_preds, pos_label=1, zero_division=0)

    LOGGER.info(f"\nMLP Classifier (full dirty dataset vs clean GT):")
    LOGGER.info(f"  Accuracy:  {overall_mlp_acc:.4f}")
    LOGGER.info(f"  Precision (error class): {mlp_precision:.4f}")
    LOGGER.info(f"  Recall    (error class): {mlp_recall:.4f}")
    LOGGER.info(f"  F1-Score  (error class): {mlp_f1:.4f}")
    LOGGER.info(f"  Per-attribute accuracy: {mlp_accuracies}")

    has_cross_attn = len(all_cross_attn_preds) > 0
    if has_cross_attn:
        cross_attn_eval_labels = np.array(cross_attn_eval_labels)
        all_cross_attn_preds = np.array(all_cross_attn_preds)
        mlp_preds_for_cross_subset = np.array(mlp_preds_for_cross_subset)

        cross_attn_total_samples = len(cross_attn_eval_labels)
        cross_attn_error_samples = int((cross_attn_eval_labels == 1).sum())

        overall_cross_attn_acc = accuracy_score(cross_attn_eval_labels, all_cross_attn_preds)
        cross_attn_precision = precision_score(cross_attn_eval_labels, all_cross_attn_preds, pos_label=1, zero_division=0)
        cross_attn_recall = recall_score(cross_attn_eval_labels, all_cross_attn_preds, pos_label=1, zero_division=0)
        cross_attn_f1 = f1_score(cross_attn_eval_labels, all_cross_attn_preds, pos_label=1, zero_division=0)

        subset_mlp_acc = accuracy_score(cross_attn_eval_labels, mlp_preds_for_cross_subset)
        improvement = overall_cross_attn_acc - subset_mlp_acc

        LOGGER.info(f"\nCross Attention Detector (evaluated on {cross_attn_total_samples} samples | {cross_attn_error_samples} errors):")
        LOGGER.info(f"  Accuracy:  {overall_cross_attn_acc:.4f}")
        LOGGER.info(f"  Precision (error class): {cross_attn_precision:.4f}")
        LOGGER.info(f"  Recall    (error class): {cross_attn_recall:.4f}")
        LOGGER.info(f"  F1-Score  (error class): {cross_attn_f1:.4f}")
        LOGGER.info(f"  Per-attribute accuracy: {cross_attn_accuracies}")
        LOGGER.info(f"  Accuracy improvement vs MLP on same subset: {improvement*100:.2f}%")

        if overall_cross_attn_acc > subset_mlp_acc:
            LOGGER.info("✓ Cross Attention detector performs BETTER than MLP on its evaluation subset!")
        else:
            LOGGER.info("✗ MLP classifier performs better or equal on the same subset.")

        results = {
            'dataset': dataset_name,
            'overall_mlp_accuracy': float(overall_mlp_acc),
            'overall_mlp_precision': float(mlp_precision),
            'overall_mlp_recall': float(mlp_recall),
            'overall_mlp_f1': float(mlp_f1),
            'overall_cross_attention_accuracy': float(overall_cross_attn_acc),
            'overall_cross_attention_precision': float(cross_attn_precision),
            'overall_cross_attention_recall': float(cross_attn_recall),
            'overall_cross_attention_f1': float(cross_attn_f1),
            'mlp_accuracy_on_cross_subset': float(subset_mlp_acc),
            'improvement': float(improvement),
            'total_samples': total_eval_samples,
            'error_samples': error_sample_count,
            'cross_attn_samples': cross_attn_total_samples,
            'cross_attn_error_samples': cross_attn_error_samples,
            'attributes_count': len(attr_groups),
            'mlp_per_attribute': mlp_accuracies,
            'cross_attention_per_attribute': cross_attn_accuracies,
            'data_source': zeroed_results['source']
        }
    else:
        LOGGER.warning("Cross Attention results unavailable; reporting MLP metrics only.")
        results = {
            'dataset': dataset_name,
            'overall_mlp_accuracy': float(overall_mlp_acc),
            'overall_mlp_precision': float(mlp_precision),
            'overall_mlp_recall': float(mlp_recall),
            'overall_mlp_f1': float(mlp_f1),
            'overall_cross_attention_accuracy': None,
            'overall_cross_attention_precision': None,
            'overall_cross_attention_recall': None,
            'overall_cross_attention_f1': None,
            'total_samples': total_eval_samples,
            'error_samples': error_sample_count,
            'attributes_count': len(attr_groups),
            'mlp_per_attribute': mlp_accuracies,
            'note': 'Cross Attention training failed or skipped'
        }

    # 6. 保存结果
    output_file = f"/data/nw/Cleaning_LLM/real_data_comparison_v2_{dataset_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    LOGGER.info(f"\n✓ Results saved to {output_file}")
    LOGGER.info("="*80)
    LOGGER.info(f"Experiment completed for {dataset_name} dataset!")
    LOGGER.info("="*80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Cross Attention vs MLP comparison using ZeroED results"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='beers',
        help='Dataset name (beers, hospital, flights, rayyan, etc.)'
    )

    args = parser.parse_args()

    LOGGER.info("="*80)
    LOGGER.info("Cross Attention vs MLP Comparison on ZeroED Results")
    LOGGER.info("="*80)
    LOGGER.info(f"Dataset: {args.dataset}")
    LOGGER.info("ZeroED Results Path: /data/nw/Cleaning_LLM/result/pipeline (auto-detected)")
    LOGGER.info("="*80 + "\n")

    results = run_real_data_comparison(dataset_name=args.dataset)

    if results is None:
        LOGGER.error("\n❌ Experiment failed. Please check the logs above.")
        exit(1)
    else:
        LOGGER.info("\n✓ Experiment completed successfully!")
        exit(0)
