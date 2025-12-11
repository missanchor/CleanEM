#!/usr/bin/env python3
"""
演示脚本：对比MLP和Cross Attention检测方法

使用模拟数据展示两种方法的工作原理和性能对比
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn.functional as F
from cross_attention_detector import LLMBasedCrossAttentionDetector, prepare_text_representation
from cross_modal_error_detector.utils.device import resolve_runtime_device
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=1000, n_features=50, n_errors=200, random_state=42):
    """
    生成合成数据集用于演示

    Args:
        n_samples: 总样本数
        n_features: 特征维度
        n_errors: 错误样本数
        random_state: 随机种子

    Returns:
        tuple: (特征, 标签, 原始值列表)
    """
    np.random.seed(random_state)

    # 生成正常数据
    normal_data = np.random.randn(n_samples - n_errors, n_features)

    # 生成错误数据（添加噪声或异常值）
    error_data = np.random.randn(n_errors, n_features)
    error_data += np.random.randn(n_features) * 2  # 添加更多噪声

    # 合并数据
    features = np.vstack([normal_data, error_data])

    # 生成标签
    labels = np.hstack([np.zeros(n_samples - n_errors), np.ones(n_errors)])

    # 生成原始值（模拟数据库中的实际值）
    raw_values = []
    for i in range(n_samples):
        if labels[i] == 0:
            # 正常值
            raw_values.append(f"value_{i}_correct")
        else:
            # 错误值
            error_types = ["typo", "outlier", "inconsistent", "missing", "format_error"]
            raw_values.append(f"value_{i}_error_{error_types[i % len(error_types)]}")

    return features, labels, raw_values


def create_enriched_texts(features, labels, raw_values):
    """
    创建丰富的文本描述，用于LLM embedding

    Args:
        features: 特征数组
        labels: 标签数组
        raw_values: 原始值列表

    Returns:
        list: 文本描述列表
    """
    texts = []
    for i, (feature, label, raw_val) in enumerate(zip(features, labels, raw_values)):
        # 分析特征统计
        mean_feat = np.mean(feature)
        std_feat = np.std(feature)
        max_feat = np.max(feature)
        min_feat = np.min(feature)

        # 确定状态
        status = "correct" if label == 0 else "error"

        # 构建详细描述
        text = f"Data point {i}: The value '{raw_val}' is a {status} entry. "
        text += f"Statistical summary - Mean: {mean_feat:.3f}, Std: {std_feat:.3f}, "
        text += f"Range: [{min_feat:.3f}, {max_feat:.3f}]. "

        # 添加一些解释性文本
        if status == "error":
            text += "This entry exhibits anomalous characteristics that deviate from expected patterns. "
            text += "Possible issues include typographical errors, inconsistent formatting, or outlier values. "
            text += f"The feature vector has {len(feature)} dimensions with complex interactions."
        else:
            text += "This entry follows expected data patterns and appears valid. "
            text += "The values are within normal ranges and consistent with data constraints. "
            text += f"The feature vector contains {len(feature)} well-structured attributes."

        texts.append(text)

    return texts


class SimplifiedCrossAttentionClassifier(torch.nn.Module):
    """
    简化版的Cross Attention分类器，用于演示
    不依赖外部LLM，使用简单的文本嵌入
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 投影层
        self.feature_proj = torch.nn.Linear(feature_dim, hidden_dim)
        self.text_proj = torch.nn.Linear(300, hidden_dim)  # 假设文本嵌入为300维

        # Cross Attention
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, 2)
        )

        self.device = torch.device(resolve_runtime_device("cuda"))

    def text_to_embedding(self, text):
        """简单的文本嵌入（演示用）"""
        # 使用基于长度和字符的简单哈希嵌入
        embedding = torch.zeros(300)
        for i, char in enumerate(text[:300]):
            embedding[i % 300] += ord(char)
        return embedding / len(text[:300]) if len(text) > 0 else embedding

    def fit(self, features, labels, texts, epochs=50, lr=0.001):
        """训练模型"""
        self.to(self.device)
        torch.nn.Module.train(self)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        features = torch.FloatTensor(features).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        for epoch in range(epochs):
            total_loss = 0

            # 生成文本嵌入
            text_embeddings = torch.stack([
                self.text_to_embedding(text) for text in texts
            ]).to(self.device)

            # 前向传播
            projected_features = self.feature_proj(features).unsqueeze(1)
            projected_texts = self.text_proj(text_embeddings).unsqueeze(0).expand(features.size(0), -1, -1)

            # Cross Attention
            attn_output, _ = self.cross_attn(
                query=projected_features,
                key=projected_texts,
                value=projected_texts
            )
            attn_output = attn_output.squeeze(1)

            # 分类
            logits = self.classifier(attn_output)

            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                LOGGER.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def predict(self, features, texts):
        """预测"""
        self.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)

            # 生成文本嵌入
            text_embeddings = torch.stack([
                self.text_to_embedding(text) for text in texts
            ]).to(self.device)

            # 前向传播
            projected_features = self.feature_proj(features).unsqueeze(1)
            projected_texts = self.text_proj(text_embeddings).unsqueeze(0).expand(features.size(0), -1, -1)

            # Cross Attention
            attn_output, _ = self.cross_attn(
                query=projected_features,
                key=projected_texts,
                value=projected_texts
            )
            attn_output = attn_output.squeeze(1)

            # 分类
            logits = self.classifier(attn_output)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy()


def run_comparison_demo():
    """运行对比演示"""
    LOGGER.info("="*80)
    LOGGER.info("MLP vs Cross Attention Detection Methods Comparison Demo")
    LOGGER.info("="*80)

    # 1. 生成数据
    LOGGER.info("\n1. Generating synthetic dataset...")
    features, labels, raw_values = generate_synthetic_data(
        n_samples=1000,
        n_features=50,
        n_errors=300,
        random_state=42
    )

    LOGGER.info(f"   - Total samples: {len(features)}")
    LOGGER.info(f"   - Feature dimensions: {features.shape[1]}")
    LOGGER.info(f"   - Error samples: {np.sum(labels)}")
    LOGGER.info(f"   - Error rate: {np.mean(labels)*100:.1f}%")

    # 2. 分割数据
    LOGGER.info("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        features, labels, raw_values,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )

    LOGGER.info(f"   - Training set: {len(X_train)} samples")
    LOGGER.info(f"   - Test set: {len(X_test)} samples")

    # 3. 创建文本描述
    LOGGER.info("\n3. Creating text descriptions for LLM embedding...")
    train_texts = create_enriched_texts(X_train, y_train, raw_train)
    test_texts = create_enriched_texts(X_test, y_test, raw_test)

    LOGGER.info(f"   - Sample text: {train_texts[0][:100]}...")

    # 4. 训练MLP
    LOGGER.info("\n4. Training MLP classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)

    LOGGER.info(f"   ✓ MLP training completed!")
    LOGGER.info(f"   - Test Accuracy: {mlp_acc:.4f}")

    # 5. 训练Cross Attention模型
    LOGGER.info("\n5. Training Cross Attention detector...")
    cross_attn = SimplifiedCrossAttentionClassifier(
        feature_dim=X_train.shape[1],
        hidden_dim=128
    )
    cross_attn.fit(X_train, y_train, train_texts, epochs=50, lr=0.001)
    cross_attn_preds = cross_attn.predict(X_test, test_texts)
    cross_attn_acc = accuracy_score(y_test, cross_attn_preds)

    LOGGER.info(f"   ✓ Cross Attention training completed!")
    LOGGER.info(f"   - Test Accuracy: {cross_attn_acc:.4f}")

    # 6. 对比结果
    LOGGER.info("\n" + "="*80)
    LOGGER.info("COMPARISON RESULTS")
    LOGGER.info("="*80)

    LOGGER.info(f"\nMLP Classifier:")
    LOGGER.info(f"  Accuracy:  {mlp_acc:.4f}")
    LOGGER.info(f"  Precision: {np.sum((mlp_preds == 1) & (y_test == 1)) / (np.sum(mlp_preds == 1) + 1e-10):.4f}")
    LOGGER.info(f"  Recall:    {np.sum((mlp_preds == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-10):.4f}")

    LOGGER.info(f"\nCross Attention Detector:")
    LOGGER.info(f"  Accuracy:  {cross_attn_acc:.4f}")
    LOGGER.info(f"  Precision: {np.sum((cross_attn_preds == 1) & (y_test == 1)) / (np.sum(cross_attn_preds == 1) + 1e-10):.4f}")
    LOGGER.info(f"  Recall:    {np.sum((cross_attn_preds == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-10):.4f}")

    # 7. 性能改进
    improvement = cross_attn_acc - mlp_acc
    LOGGER.info(f"\nPerformance Improvement: {improvement*100:.2f}%")

    if cross_attn_acc > mlp_acc:
        LOGGER.info("✓ Cross Attention detector performs better!")
    else:
        LOGGER.info("✗ MLP classifier performs better.")

    # 8. 详细分析
    LOGGER.info("\n" + "="*80)
    LOGGER.info("DETAILED ANALYSIS")
    LOGGER.info("="*80)

    # MLP Confusion Matrix
    LOGGER.info("\nMLP Confusion Matrix:")
    mlp_cm = confusion_matrix(y_test, mlp_preds)
    LOGGER.info(f"  True Negatives (TN):  {mlp_cm[0, 0]}")
    LOGGER.info(f"  False Positives (FP): {mlp_cm[0, 1]}")
    LOGGER.info(f"  False Negatives (FN): {mlp_cm[1, 0]}")
    LOGGER.info(f"  True Positives (TP):  {mlp_cm[1, 1]}")

    # Cross Attention Confusion Matrix
    LOGGER.info("\nCross Attention Confusion Matrix:")
    cross_attn_cm = confusion_matrix(y_test, cross_attn_preds)
    LOGGER.info(f"  True Negatives (TN):  {cross_attn_cm[0, 0]}")
    LOGGER.info(f"  False Positives (FP): {cross_attn_cm[0, 1]}")
    LOGGER.info(f"  False Negatives (FN): {cross_attn_cm[1, 0]}")
    LOGGER.info(f"  True Positives (TP):  {cross_attn_cm[1, 1]}")

    # 9. 错误案例分析
    LOGGER.info("\n" + "="*80)
    LOGGER.info("ERROR CASE ANALYSIS")
    LOGGER.info("="*80)

    # MLP的错误案例
    mlp_errors = (y_test != mlp_preds)
    LOGGER.info(f"\nMLP Errors: {np.sum(mlp_errors)}/{len(y_test)} ({np.mean(mlp_errors)*100:.1f}%)")

    # Cross Attention的错误案例
    cross_attn_errors = (y_test != cross_attn_preds)
    LOGGER.info(f"\nCross Attention Errors: {np.sum(cross_attn_errors)}/{len(y_test)} ({np.mean(cross_attn_errors)*100:.1f}%)")

    # 共同错误的样本
    common_errors = mlp_errors & cross_attn_errors
    LOGGER.info(f"\nCommon Errors: {np.sum(common_errors)} samples")

    # 只被MLP错过的错误
    mlp_only_errors = mlp_errors & (~cross_attn_errors)
    LOGGER.info(f"Errors caught only by Cross Attention: {np.sum(mlp_only_errors)} samples")

    # 只被Cross Attention错过的错误
    cross_attn_only_errors = cross_attn_errors & (~mlp_errors)
    LOGGER.info(f"Errors caught only by MLP: {np.sum(cross_attn_only_errors)} samples")

    # 10. 保存结果
    LOGGER.info("\n" + "="*80)
    LOGGER.info("Saving results...")

    results = {
        'mlp_accuracy': mlp_acc,
        'cross_attention_accuracy': cross_attn_acc,
        'improvement': improvement,
        'mlp_predictions': mlp_preds.tolist(),
        'cross_attention_predictions': cross_attn_preds.tolist(),
        'true_labels': y_test.tolist()
    }

    import json
    with open('/data/nw/Cleaning_LLM/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Results saved to /data/nw/Cleaning_LLM/comparison_results.json")

    LOGGER.info("\n" + "="*80)
    LOGGER.info("Demo completed successfully!")
    LOGGER.info("="*80)

    return results


if __name__ == "__main__":
    results = run_comparison_demo()
