#!/usr/bin/env python3
"""
Cross Attention Detection Model

基于UnIMP的CrossAttentionFusion机制，将zeroed特征与LLM embedding进行cross attention，
用于数据错误检测，并与MLP方法进行对比。
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm
import logging

from cross_modal_error_detector.utils.device import resolve_runtime_device

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class CrossAttentionDetectionHead(nn.Module):
    """
    基于UnIMP的CrossAttentionFusion的检测头部
    将zeroed特征与LLM embedding进行cross attention，然后进行分类
    """
    def __init__(self, feature_dim, llm_dim=1024, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim

        # 特征投影层 - 将zeroed特征投影到LLM embedding维度
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # LLM embedding投影层（如果LLM维度不匹配）
        self.llm_proj = nn.Linear(llm_dim, hidden_dim) if llm_dim != hidden_dim else nn.Identity()

        # Cross Attention层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, llm_embeddings):
        """
        Args:
            features: [batch_size, feature_dim] - zeroed特征
            llm_embeddings: [batch_size, seq_len, llm_dim] - LLM embedding序列

        Returns:
            logits: [batch_size, 2] - 分类logits
        """
        # 确保所有张量使用与检测头相同的dtype/device
        target_dtype = self.feature_proj.weight.dtype
        target_device = self.feature_proj.weight.device
        features = features.to(device=target_device, dtype=target_dtype)
        llm_embeddings = llm_embeddings.to(device=target_device, dtype=target_dtype)

        batch_size = features.size(0)

        # 投影特征到hidden_dim
        projected_features = self.feature_proj(features)  # [batch_size, hidden_dim]
        projected_features = projected_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # 投影LLM embedding
        projected_llm = self.llm_proj(llm_embeddings)  # [batch_size, seq_len, hidden_dim]

        # Cross Attention: 使用features作为query, llm作为key和value
        attn_output, attn_weights = self.cross_attn(
            query=projected_features,
            key=projected_llm,
            value=projected_llm
        )
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]

        # 残差连接和LayerNorm
        attn_output = self.ln1(projected_features.squeeze(1) + self.dropout(attn_output))

        # FFN
        ffn_output = self.ffn(attn_output)
        ffn_output = self.ln2(attn_output + self.dropout(ffn_output))

        # 分类
        logits = self.classifier(ffn_output)

        return logits


class LLMBasedCrossAttentionDetector:
    """
    基于LLM cross attention的检测器
    """
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device: Optional[str] = None):
        resolved_device = resolve_runtime_device(device)
        self.device = torch.device(resolved_device)
        self.model_name = model_name
        self.tokenizer = None
        self.llm_model = None
        self.detection_head = None
        self.is_trained = False

    def load_llm(self, cache_dir=None):
        """加载LLM模型和tokenizer"""
        LOGGER.info(f"Loading LLM model: {self.model_name}")

        # 检查是否是本地路径
        if os.path.exists(self.model_name):
            # 直接使用本地路径
            model_dir = self.model_name
            LOGGER.info(f"Using local model at: {model_dir}")
        else:
            # 使用ModelScope下载模型
            model_dir = snapshot_download(
                self.model_name,
                cache_dir=cache_dir or "~/.cache/modelscope"
            )
            LOGGER.info(f"Downloaded model to: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载LLM模型
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            dtype=dtype,
        )
        self.llm_model.to(self.device)
        self.llm_model.eval()

        # 冻结LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        LOGGER.info(f"LLM model loaded successfully on {self.device}")
        return self

    def get_llm_embeddings(self, texts, max_length=128):
        """获取LLM embedding"""
        if self.llm_model is None or self.tokenizer is None:
            raise ValueError("LLM model not loaded. Call load_llm() first.")

        # Token化
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 获取embedding
        with torch.no_grad():
            outputs = self.llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # 最后一层

        return hidden_states, attention_mask

    def initialize_detection_head(self, feature_dim, hidden_dim=512):
        """初始化检测头部"""
        llm_dim = self.llm_model.config.hidden_size if hasattr(self.llm_model.config, 'hidden_size') else 1024

        self.detection_head = CrossAttentionDetectionHead(
            feature_dim=feature_dim,
            llm_dim=llm_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        llm_dtype = next(self.llm_model.parameters()).dtype
        head_dtype = next(self.detection_head.parameters()).dtype

        LOGGER.info(
            f"Detection head initialized with feature_dim={feature_dim}, llm_dim={llm_dim}, "
            f"head_dtype={head_dtype}, llm_dtype={llm_dtype}"
        )
        return self

    def train(self, train_features, train_labels, train_texts, val_features=None, val_labels=None, val_texts=None,
              epochs=50, batch_size=32, lr=1e-4, weight_decay=0.01):
        """
        训练检测模型

        Args:
            train_features: numpy array, 训练特征 [n_samples, feature_dim]
            train_labels: numpy array, 训练标签 [n_samples]
            train_texts: list, 训练样本对应的文本描述
            val_features: numpy array, 验证特征
            val_labels: numpy array, 验证标签
            val_texts: list, 验证样本对应的文本描述
            epochs: int, 训练轮数
            batch_size: int, 批次大小
            lr: float, 学习率
            weight_decay: float, 权重衰减
        """
        if self.detection_head is None:
            raise ValueError("Detection head not initialized. Call initialize_detection_head() first.")

        # 转换为tensor - 使用与LLM相同的dtype
        head_dtype = next(self.detection_head.parameters()).dtype
        train_features = torch.tensor(train_features, dtype=head_dtype, device=self.device)
        train_labels = torch.as_tensor(train_labels, dtype=torch.long, device=self.device)

        if val_features is not None:
            val_features = torch.tensor(val_features, dtype=head_dtype, device=self.device)
            val_labels = torch.as_tensor(val_labels, dtype=torch.long, device=self.device)

        # 优化器
        optimizer = torch.optim.AdamW(self.detection_head.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_acc = 0
        self.detection_head.train()

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            # 分批训练
            for i in range(0, len(train_features), batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                batch_texts = train_texts[i:i+batch_size]

                # 获取LLM embedding
                llm_embeds, attention_mask = self.get_llm_embeddings(batch_texts)

                # 前向传播
                optimizer.zero_grad()
                logits = self.detection_head(batch_features, llm_embeds)
                loss = criterion(logits, batch_labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            # 验证
            if val_features is not None and epoch % 10 == 0:
                val_acc = self.evaluate(val_features, val_labels, val_texts, batch_size)
                LOGGER.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.detection_head.state_dict(), "best_cross_attn_detector.pth")
            else:
                LOGGER.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True
        LOGGER.info("Training completed!")
        return self

    def evaluate(self, features, labels, texts, batch_size=32):
        """评估模型"""
        self.detection_head.eval()

        all_preds = []
        all_labels = []

        # 使用与LLM相同的dtype
        head_dtype = next(self.detection_head.parameters()).dtype

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = torch.tensor(
                    features[i:i+batch_size], dtype=head_dtype, device=self.device
                )
                batch_labels = labels[i:i+batch_size]
                batch_texts = texts[i:i+batch_size]

                llm_embeds, _ = self.get_llm_embeddings(batch_texts)
                logits = self.detection_head(batch_features, llm_embeds)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

    def predict(self, features, texts, batch_size=32):
        """预测"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        self.detection_head.eval()
        predictions = []

        # 使用与LLM相同的dtype
        head_dtype = next(self.detection_head.parameters()).dtype

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = torch.tensor(
                    features[i:i+batch_size], dtype=head_dtype, device=self.device
                )
                batch_texts = texts[i:i+batch_size]

                llm_embeds, _ = self.get_llm_embeddings(batch_texts)
                logits = self.detection_head(batch_features, llm_embeds)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)


def prepare_text_representation(features, labels, raw_values=None):
    """
    基于特征准备文本描述，用于获取LLM embedding
    这里可以使用简单的策略，比如将特征转换为描述性文本
    """
    texts = []
    for i, (feature, label) in enumerate(zip(features, labels)):
        if raw_values is not None:
            # 使用原始值作为文本描述
            text = f"This data point has value: {raw_values[i]}"
        else:
            # 使用特征向量的统计信息作为文本描述
            text = f"This data point has feature statistics: mean={np.mean(feature):.2f}, std={np.std(feature):.2f}"
        texts.append(text)
    return texts


def compare_mlp_vs_cross_attention(
    train_features,
    train_labels,
    test_features,
    test_labels,
    train_raw_values=None,
    test_raw_values=None,
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device: Optional[str] = None,
):
    """
    对比MLP和Cross Attention两种方法

    Returns:
        dict: 包含两种方法的结果
    """
    results = {}

    # 1. 训练MLP分类器
    LOGGER.info("Training MLP classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    mlp.fit(train_features, train_labels)
    mlp_preds = mlp.predict(test_features)
    mlp_acc = accuracy_score(test_labels, mlp_preds)
    results['mlp_accuracy'] = mlp_acc
    results['mlp_predictions'] = mlp_preds
    LOGGER.info(f"MLP Accuracy: {mlp_acc:.4f}")

    # 2. 训练Cross Attention检测器
    LOGGER.info("Training Cross Attention detector...")
    runtime_device = resolve_runtime_device(device)
    detector = LLMBasedCrossAttentionDetector(model_name=model_name, device=runtime_device)
    detector.load_llm()

    # 准备文本描述
    train_texts = prepare_text_representation(train_features, train_labels, train_raw_values)
    test_texts = prepare_text_representation(test_features, test_labels, test_raw_values)

    # 初始化检测头部
    detector.initialize_detection_head(feature_dim=train_features.shape[1])

    # 训练
    detector.train(
        train_features=train_features,
        train_labels=train_labels,
        train_texts=train_texts,
        epochs=50,
        batch_size=16,
        lr=1e-4
    )

    # 预测
    cross_attn_preds = detector.predict(test_features, test_texts)
    cross_attn_acc = accuracy_score(test_labels, cross_attn_preds)
    results['cross_attention_accuracy'] = cross_attn_acc
    results['cross_attention_predictions'] = cross_attn_preds
    LOGGER.info(f"Cross Attention Accuracy: {cross_attn_acc:.4f}")

    # 3. 详细对比报告
    LOGGER.info("\n" + "="*50)
    LOGGER.info("DETAILED COMPARISON")
    LOGGER.info("="*50)
    LOGGER.info(f"MLP Accuracy: {mlp_acc:.4f}")
    LOGGER.info(f"Cross Attention Accuracy: {cross_attn_acc:.4f}")
    LOGGER.info(f"Improvement: {(cross_attn_acc - mlp_acc)*100:.2f}%")

    LOGGER.info("\nMLP Classification Report:")
    LOGGER.info(classification_report(test_labels, mlp_preds))

    LOGGER.info("\nCross Attention Classification Report:")
    LOGGER.info(classification_report(test_labels, cross_attn_preds))

    return results


if __name__ == "__main__":
    # 示例用法
    import numpy as np

    # 生成示例数据
    np.random.seed(42)
    train_features = np.random.randn(1000, 100)
    train_labels = np.random.randint(0, 2, 1000)
    test_features = np.random.randn(200, 100)
    test_labels = np.random.randint(0, 2, 200)
    train_raw_values = [f"value_{i}" for i in range(1000)]
    test_raw_values = [f"test_value_{i}" for i in range(200)]

    # 对比实验
    results = compare_mlp_vs_cross_attention(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        train_raw_values=train_raw_values,
        test_raw_values=test_raw_values
    )

    print("\nFinal Results:")
    print(f"MLP Accuracy: {results['mlp_accuracy']:.4f}")
    print(f"Cross Attention Accuracy: {results['cross_attention_accuracy']:.4f}")
