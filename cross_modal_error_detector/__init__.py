"""
跨模态错误检测系统 - 高度模块化架构实现
Cross-Modal Error Detector - Highly Modular Architecture Implementation

这个实现遵循以下设计原则：
1. 清晰的接口定义（Base Classes）
2. 可插拔的组件（Pluggable Components）
3. 易于扩展和进行消融实验（Ablation Studies）
4. 支持多种训练策略（Training Strategies）
"""

# Base interfaces
from .base import BaseEncoder, BaseFusion, BaseDetectionHead

# Core model
from .model import CrossModalErrorDetector

# Encoders
from .encoders import (
    StructureAwareTransformer,
    PretrainedTextEncoder,
    SimpleMLPEncoder
)

# Fusion modules
from .fusion import (
    CrossAttentionFusion,
    SimpleConcatFusion
)

# Detection heads
from .heads import (
    MLPDetectionHead,
    ContrastiveDetectionHead
)

# Data processing
from .processors import (
    TabularProcessor,
    TextProcessor
)

# Datasets
from .datasets import (
    CorruptionBasedDataset,
    ContrastiveDataset,
    CleanDirtyEvaluationDataset,
    PerColumnBinaryDataset
)

# Training utilities
from .training import (
    collate_fn_corruption,
    collate_fn_contrastive,
    collate_fn_contrastive_cell_level,
    train_step_corruption,
    compute_embedding_similarity,
    train_step_contrastive_pretrain
)

__all__ = [
    # Base interfaces
    'BaseEncoder',
    'BaseFusion',
    'BaseDetectionHead',

    # Core model
    'CrossModalErrorDetector',

    # Encoders
    'StructureAwareTransformer',
    'PretrainedTextEncoder',
    'SimpleMLPEncoder',

    # Fusion modules
    'CrossAttentionFusion',
    'SimpleConcatFusion',

    # Detection heads
    'MLPDetectionHead',
    'ContrastiveDetectionHead',

    # Data processing
    'TabularProcessor',
    'TextProcessor',

    # Datasets
    'CorruptionBasedDataset',
    'ContrastiveDataset',
    'CleanDirtyEvaluationDataset',
    'PerColumnBinaryDataset',

    # Training utilities
    'collate_fn_corruption',
    'collate_fn_contrastive',
    'collate_fn_contrastive_cell_level',
    'train_step_corruption',
    'compute_embedding_similarity',
    'train_step_contrastive_pretrain',
]

if __name__ == "__main__":
    print("跨模态错误检测系统 - 模块化架构实现完成！")
    print("\n可用的组件：")
    print("=" * 60)
    print("\n1. 编码器 (Encoders):")
    print("   - StructureAwareTransformer: 结构感知的表格编码器")
    print("   - PretrainedTextEncoder: 预训练文本编码器")
    print("   - SimpleMLPEncoder: 简单MLP编码器（消融实验）")
    print("\n2. 融合模块 (Fusion Modules):")
    print("   - CrossAttentionFusion: 跨注意力融合")
    print("   - SimpleConcatFusion: 简单拼接融合（消融实验）")
    print("\n3. 检测头 (Detection Heads):")
    print("   - MLPDetectionHead: 单元格级二分类")
    print("   - ContrastiveDetectionHead: 行-文本匹配")
    print("\n4. 训练策略 (Training Strategies):")
    print("   - Corruption-based: 破坏-重建")
    print("   - Contrastive: 对比学习")
    print("=" * 60)


