#!/usr/bin/env python3
"""
快速测试真实数据上的对比效果
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from cross_attention_detector import LLMBasedCrossAttentionDetector
from cross_modal_error_detector.utils.device import resolve_runtime_device
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def quick_test():
    print("="*80)
    print("Quick Real Data Test: MLP vs Cross Attention")
    print("="*80)

    # 使用beers数据集
    dirty_df = pd.read_csv("/data/nw/Cleaning_LLM/data/beers_error-01.csv", dtype=str).fillna('nan')
    clean_df = pd.read_csv("/data/nw/Cleaning_LLM/data/beers_clean.csv", dtype=str).fillna('nan')

    print(f"\nDataset: beers")
    print(f"Shape: {dirty_df.shape}")

    # 提取错误标签
    labels = (clean_df != dirty_df).any(axis=1).astype(int)
    print(f"Error rate: {labels.mean()*100:.1f}%")

    # 简单特征提取
    features = []
    for i in range(len(dirty_df)):
        row_features = []
        for j in range(min(5, len(dirty_df.columns))):  # 只用前5列
            val = str(dirty_df.iloc[i, j])
            row_features.extend([
                len(val),
                int(any(c.isdigit() for c in val)),
                int(any(c.isalpha() for c in val)),
                int(' ' in val),
                int(val.lower() == 'nan')
            ])
        features.append(row_features)
    features = np.array(features)
    print(f"Features shape: {features.shape}")

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 1. 训练MLP
    print("\n1. Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    print(f"   MLP Accuracy: {mlp_acc:.4f}")

    # 2. 训练Cross Attention
    print("\n2. Training Cross Attention...")

    # 创建文本描述
    texts_train = [f"Row {i}: {dirty_df.iloc[idx].to_dict()}" for i, idx in enumerate(X_train)]
    texts_test = [f"Row {i}: {dirty_df.iloc[idx].to_dict()}" for i, idx in enumerate(X_test)]

    detector = LLMBasedCrossAttentionDetector(
        model_name="/mnt/data/welkinni/Qwen2.5-3B/qwen/Qwen2.5-3B",
        device=resolve_runtime_device("cuda")
    )
    detector.load_llm()
    detector.initialize_detection_head(feature_dim=X_train.shape[1])

    # 训练（用较少的epoch以节省时间）
    detector.train(
        train_features=X_train,
        train_labels=y_train,
        train_texts=texts_train,
        epochs=10,  # 只用10个epoch
        batch_size=16,
        lr=1e-4
    )

    cross_attn_preds = detector.predict(X_test, texts_test)
    cross_attn_acc = accuracy_score(y_test, cross_attn_preds)
    print(f"   Cross Attention Accuracy: {cross_attn_acc:.4f}")

    # 3. 对比结果
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"MLP Accuracy:           {mlp_acc:.4f}")
    print(f"Cross Attention:        {cross_attn_acc:.4f}")
    print(f"Improvement:            {(cross_attn_acc - mlp_acc)*100:.2f}%")

    if cross_attn_acc > mlp_acc:
        print("✓ Cross Attention performs better!")
    else:
        print("✗ MLP performs better.")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

if __name__ == "__main__":
    quick_test()
