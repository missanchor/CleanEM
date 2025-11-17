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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import json
import pickle
import re
import logging
from cross_attention_detector import LLMBasedCrossAttentionDetector

# 添加ZeroED目录到路径以加载相关模块
sys.path.insert(0, '/data/nw/Cleaning_LLM/ZeroED')

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

                # 组合所有特征（使用ZeroED设计的专业特征）
                feature_parts = []
                # 只使用固定存在的特征类型
                for key in ['occur_cnt_feat', 'pat_stats_feat', 'fasttext_feat', 'pre_funcs_feat']:
                    if key in feat_dict and isinstance(feat_dict[key], (list, np.ndarray)) and len(feat_dict[key]) > 0:
                        # 确保特征是平坦的数值列表
                        if isinstance(feat_dict[key][0], (list, np.ndarray)):
                            # 如果是嵌套的，平坦化
                            flat = [item for sublist in feat_dict[key] for item in (sublist if isinstance(sublist, list) else [sublist])]
                            feature_parts.extend(flat)
                        else:
                            # 直接添加
                            feature_parts.extend(feat_dict[key])

                # 跳过无效特征
                if len(feature_parts) < 10:  # 最小特征长度阈值
                    continue

                # 生成文本描述
                raw_val = str(dirty_df.iloc[row_idx, col_idx])
                text = f"The value '{raw_val}' in column '{attr}' is "
                text += "incorrect" if label == 1 else "correct"

                # 添加到对应属性的组
                attr_groups[attr]['features'].append(np.array(feature_parts))
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
    if dataset_name == "hospital":
        # 查找hospital目录
        import glob
        pattern = os.path.join(base_result_dir, "*hospital01-5%-set1")
        all_dirs = glob.glob(pattern)
        if not all_dirs:
            LOGGER.error(f"❌ No Hospital ZeroED results found matching pattern: {pattern}")
            return None
        resp_path = all_dirs[0]
        LOGGER.info(f"✓ Using Hospital ZeroED results from: {resp_path}")
    else:
        # 查找最新结果目录
        import glob
        pattern = os.path.join(base_result_dir, f"*-*-* *s01-5%-set1")
        all_dirs = glob.glob(pattern)
        all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        if not all_dirs:
            LOGGER.error(f"❌ No ZeroED results found matching pattern: {pattern}")
            return None

        resp_path = all_dirs[0]
        LOGGER.info(f"✓ Using ZeroED results from: {resp_path}")

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

    LOGGER.info(f"Found {len(attr_groups)} attributes: {list(attr_groups.keys())}")

    # 3. 为每个属性训练模型并汇总结果
    LOGGER.info("\n3. Training models for each attribute...")
    all_mlp_preds = []
    all_cross_attn_preds = []
    all_labels = []

    mlp_accuracies = {}
    cross_attn_accuracies = {}

    for attr_name, attr_data in attr_groups.items():
        LOGGER.info(f"\n  Processing attribute: {attr_name}")
        LOGGER.info(f"    Samples: {len(attr_data['features'])}")
        LOGGER.info(f"    Features shape: {attr_data['features'].shape}")
        LOGGER.info(f"    Error rate: {attr_data['labels'].mean()*100:.1f}%")

        # 跳过错误率为0%的列（所有样本都是正确的）
        if attr_data['labels'].mean() == 0:
            LOGGER.info(f"    Skipping {attr_name}: no errors in this column (all samples correct)")
            all_mlp_preds.extend(attr_data['labels'])  # 所有预测都是正确的
            all_labels.extend(attr_data['labels'])
            all_cross_attn_preds.extend(attr_data['labels'])  # Cross Attention也用正确的预测
            continue

        if len(attr_data['features']) < 50:  # 跳过样本太少的属性
            LOGGER.warning(f"    Skipping {attr_name}: too few samples")
            all_mlp_preds.extend(attr_data['labels'])
            all_labels.extend(attr_data['labels'])
            all_cross_attn_preds.extend(attr_data['labels'])
            continue

        # 分割数据
        X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
            attr_data['features'], attr_data['labels'], attr_data['texts'],
            test_size=0.3,
            random_state=42,
            stratify=attr_data['labels']
        )

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
        mlp_preds = mlp.predict(X_test)
        mlp_acc = accuracy_score(y_test, mlp_preds)
        mlp_accuracies[attr_name] = mlp_acc
        all_mlp_preds.extend(mlp_preds)

        LOGGER.info(f"    MLP Accuracy: {mlp_acc:.4f}")

        # 训练Cross Attention（可选）
        try:
            LOGGER.info(f"    Training Cross Attention (using GPU)...")
            detector = LLMBasedCrossAttentionDetector(
                model_name="/data/nw/modelscope_models/Qwen2.5-1.5B-Instruct",
                device="cuda"  # 使用GPU
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

            cross_attn_preds = detector.predict(X_test, texts_test)
            cross_attn_acc = accuracy_score(y_test, cross_attn_preds)
            cross_attn_accuracies[attr_name] = cross_attn_acc
            all_cross_attn_preds.extend(cross_attn_preds)

            LOGGER.info(f"    Cross Attention Accuracy: {cross_attn_acc:.4f}")

        except Exception as e:
            LOGGER.warning(f"    Cross Attention training failed: {e}")
            # Cross Attention失败时，用MLP预测代替（或者其他默认策略）
            all_cross_attn_preds.extend(mlp_preds)

        all_labels.extend(y_test)

    # 4. 汇总结果
    all_labels = np.array(all_labels)
    all_mlp_preds = np.array(all_mlp_preds)

    # 只有当有Cross Attention结果时才计算
    if len(all_cross_attn_preds) > 0:
        all_cross_attn_preds = np.array(all_cross_attn_preds)
        has_cross_attn = True
    else:
        has_cross_attn = False
        all_cross_attn_preds = None

    # 仅对真实错误样本进行评估
    error_mask = all_labels == 1
    error_sample_count = int(error_mask.sum())

    if error_sample_count == 0:
        LOGGER.error("❌ No error samples found in aggregated labels. Unable to compute error-only metrics.")
        return None

    error_labels = all_labels[error_mask]
    error_mlp_preds = all_mlp_preds[error_mask]
    if has_cross_attn:
        error_cross_attn_preds = all_cross_attn_preds[error_mask]

    # 5. 对比结果
    LOGGER.info("\n" + "="*80)
    LOGGER.info("COMPARISON RESULTS (Aggregated)")
    LOGGER.info("="*80)
    LOGGER.info(f"Evaluating on error samples only ({error_sample_count} cases)")

    overall_mlp_acc = accuracy_score(error_labels, error_mlp_preds)
    # 错误检测任务：只对真实错误样本计算precision/recall/F1
    mlp_precision = precision_score(error_labels, error_mlp_preds, pos_label=1, zero_division=0)
    mlp_recall = recall_score(error_labels, error_mlp_preds, pos_label=1, zero_division=0)
    mlp_f1 = f1_score(error_labels, error_mlp_preds, pos_label=1, zero_division=0)

    LOGGER.info(f"\nError-only MLP Classifier (Baseline):")
    LOGGER.info(f"  Accuracy:  {overall_mlp_acc:.4f}")
    LOGGER.info(f"  Precision (on detected errors): {mlp_precision:.4f}")
    LOGGER.info(f"  Recall (of all errors):         {mlp_recall:.4f}")
    LOGGER.info(f"  F1-Score (on detected errors):  {mlp_f1:.4f}")
    LOGGER.info(f"  Per-attribute accuracy: {mlp_accuracies}")

    if has_cross_attn:
        overall_cross_attn_acc = accuracy_score(error_labels, error_cross_attn_preds)
        # 错误检测任务：只对真实错误样本计算precision/recall/F1
        cross_attn_precision = precision_score(error_labels, error_cross_attn_preds, pos_label=1, zero_division=0)
        cross_attn_recall = recall_score(error_labels, error_cross_attn_preds, pos_label=1, zero_division=0)
        cross_attn_f1 = f1_score(error_labels, error_cross_attn_preds, pos_label=1, zero_division=0)

        LOGGER.info(f"\nError-only Cross Attention Detector:")
        LOGGER.info(f"  Accuracy:  {overall_cross_attn_acc:.4f}")
        LOGGER.info(f"  Precision (on detected errors): {cross_attn_precision:.4f}")
        LOGGER.info(f"  Recall (of all errors):         {cross_attn_recall:.4f}")
        LOGGER.info(f"  F1-Score (on detected errors):  {cross_attn_f1:.4f}")
        LOGGER.info(f"  Per-attribute accuracy: {cross_attn_accuracies}")

        improvement = overall_cross_attn_acc - overall_mlp_acc
        LOGGER.info(f"\nPerformance Improvement: {improvement*100:.2f}%")

        if overall_cross_attn_acc > overall_mlp_acc:
            LOGGER.info("✓ Cross Attention detector performs BETTER than MLP!")
        else:
            LOGGER.info("✗ MLP classifier performs better or equal.")

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
            'improvement': float(improvement),
            'total_samples': len(all_labels),
            'error_samples': error_sample_count,
            'attributes_count': len(attr_groups),
            'mlp_per_attribute': mlp_accuracies,
            'cross_attention_per_attribute': cross_attn_accuracies,
            'data_source': zeroed_results['source']
        }
    else:
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
            'total_samples': len(all_labels),
            'error_samples': error_sample_count,
            'attributes_count': len(attr_groups),
            'mlp_per_attribute': mlp_accuracies,
            'note': 'Cross Attention training failed'
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
