"""
自定义组件示例
Example: How to Add Custom Components

展示如何扩展架构，添加自己的编码器、融合模块等
"""

import torch
import torch.nn as nn
from cross_modal_error_detector import (
    BaseEncoder,
    BaseFusion,
    CrossModalErrorDetector,
    PretrainedTextEncoder,
    MLPDetectionHead,
    TabularProcessor,
    TextProcessor,
)


# ============================================================================
# 示例1: 自定义表格编码器 - GNN-based Encoder
# ============================================================================

class GNNTableEncoder(BaseEncoder):
    """
    基于图神经网络的表格编码器
    
    思想：将表格视为图
    - 节点：单元格
    - 边：同行/同列的单元格之间有边
    """
    
    def __init__(self, d_cell: int, d_model: int, num_layers: int = 2):
        super().__init__()
        
        self.cell_projection = nn.Linear(d_cell, d_model)
        
        # 简化的图卷积层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers)
        ])
        
        self.activation = nn.ReLU()
        self.d_model = d_model
    
    def forward(self, tabular_inputs):
        """
        Args:
            tabular_inputs:
                - cell_embeddings: [batch_size, num_cols, d_cell]
                - row_indices: [batch_size, num_cols]
                - col_indices: [batch_size, num_cols]
        
        Returns:
            H_table: [batch_size, num_cols, d_model]
        """
        cell_embeddings = tabular_inputs['cell_embeddings']
        row_indices = tabular_inputs['row_indices']
        col_indices = tabular_inputs['col_indices']
        
        # 初始特征
        x = self.cell_projection(cell_embeddings)  # [B, num_cols, d_model]
        
        # 图卷积传播
        for gnn_layer in self.gnn_layers:
            # 简化版：对同行/同列的节点取平均（邻居聚合）
            x_agg = self._aggregate_neighbors(x, row_indices, col_indices)
            x = self.activation(gnn_layer(x_agg))
        
        return x
    
    def _aggregate_neighbors(self, x, row_indices, col_indices):
        """
        邻居聚合：同行或同列的单元格
        
        简化实现：这里只做全局平均作为示例
        实际应该根据行/列索引构建邻接矩阵
        """
        # 简化版：全局平均
        batch_size, num_cols, d_model = x.shape
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, d_model]
        x_agg = (x + x_mean.expand(-1, num_cols, -1)) / 2
        return x_agg


# ============================================================================
# 示例2: 自定义融合模块 - Bilinear Fusion
# ============================================================================

class BilinearFusion(BaseFusion):
    """
    双线性融合模块
    
    思想：通过双线性变换捕获表格和文本特征之间的二阶交互
    H_fuse = W(H_table ⊙ H_text)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # 双线性层
        self.bilinear = nn.Bilinear(d_model, d_model, d_model)
        
        # 额外的投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
    
    def forward(self, H_table, H_text):
        """
        Args:
            H_table: [batch_size, num_cols, d_model]
            H_text: [batch_size, text_seq_len, d_model]
        
        Returns:
            H_fuse: [batch_size, num_cols, d_model]
        """
        batch_size, num_cols, d_model = H_table.shape
        
        # 1. 对文本特征进行池化
        H_text_pooled = H_text.mean(dim=1)  # [B, d_model]
        
        # 2. 扩展到每个单元格
        H_text_expanded = H_text_pooled.unsqueeze(1).expand(-1, num_cols, -1)  # [B, num_cols, d_model]
        
        # 3. 双线性交互
        H_bilinear = self.bilinear(H_table, H_text_expanded)  # [B, num_cols, d_model]
        
        # 4. 结合原始特征
        H_combined = torch.cat([H_table, H_bilinear], dim=-1)  # [B, num_cols, 2*d_model]
        
        # 5. 输出投影
        H_fuse = self.output_projection(H_combined)  # [B, num_cols, d_model]
        
        return H_fuse


# ============================================================================
# 示例3: 自定义融合模块 - Gated Fusion
# ============================================================================

class GatedFusion(BaseFusion):
    """
    门控融合模块
    
    思想：使用门控机制动态控制表格和文本特征的融合比例
    gate = σ(W[H_table; H_text])
    H_fuse = gate ⊙ H_table + (1 - gate) ⊙ H_aligned
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 用于对齐文本特征
        self.text_projection = nn.Linear(d_model, d_model)
    
    def forward(self, H_table, H_text):
        """
        Args:
            H_table: [batch_size, num_cols, d_model]
            H_text: [batch_size, text_seq_len, d_model]
        
        Returns:
            H_fuse: [batch_size, num_cols, d_model]
        """
        batch_size, num_cols, d_model = H_table.shape
        
        # 1. 池化文本特征
        H_text_pooled = H_text.mean(dim=1)  # [B, d_model]
        H_text_expanded = H_text_pooled.unsqueeze(1).expand(-1, num_cols, -1)
        
        # 2. 投影文本特征
        H_text_aligned = self.text_projection(H_text_expanded)  # [B, num_cols, d_model]
        
        # 3. 计算门控值
        gate_input = torch.cat([H_table, H_text_aligned], dim=-1)  # [B, num_cols, 2*d_model]
        gate = self.gate_network(gate_input)  # [B, num_cols, d_model]
        
        # 4. 门控融合
        H_fuse = gate * H_table + (1 - gate) * H_text_aligned
        
        return H_fuse


# ============================================================================
# 使用示例
# ============================================================================

def demo_custom_components():
    """
    演示如何使用自定义组件
    """
    
    print("\n" + "=" * 80)
    print("自定义组件演示")
    print("=" * 80)
    
    device = 'cpu'
    d_model = 128
    
    # 准备简单的测试数据
    batch_size = 2
    num_cols = 6
    text_seq_len = 64
    
    tabular_inputs = {
        'cell_embeddings': torch.randn(batch_size, num_cols, 32),
        'row_indices': torch.zeros(batch_size, num_cols, dtype=torch.long),
        'col_indices': torch.arange(num_cols).unsqueeze(0).expand(batch_size, -1)
    }
    
    text_inputs = {
        'input_ids': torch.randint(0, 30522, (batch_size, text_seq_len)),
        'attention_mask': torch.ones(batch_size, text_seq_len, dtype=torch.long)
    }
    
    # 测试不同的自定义组件组合
    configs = [
        {
            'name': 'GNN + Bilinear Fusion',
            'tabular_encoder': GNNTableEncoder(32, d_model, num_layers=2),
            'fusion': BilinearFusion(d_model)
        },
        {
            'name': 'GNN + Gated Fusion',
            'tabular_encoder': GNNTableEncoder(32, d_model, num_layers=2),
            'fusion': GatedFusion(d_model)
        },
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n配置 {i}: {config['name']}")
        print("-" * 80)
        
        # 构建模型
        model = CrossModalErrorDetector(
            tabular_encoder=config['tabular_encoder'],
            text_encoder=PretrainedTextEncoder(30522, d_model, nhead=4, num_layers=2, max_seq_len=64),
            fusion_module=config['fusion'],
            detection_head=MLPDetectionHead(d_model, hidden_dim=64, output_dim=1)
        ).to(device)
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            logits = model(tabular_inputs, text_inputs)
        
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✓ 参数量: {num_params:,}")
        print(f"  ✓ 输出形状: {logits.shape}")
        print(f"  ✓ 前向传播成功！")
    
    print("\n" + "=" * 80)
    print("✅ 所有自定义组件测试通过！")
    print("=" * 80)
    
    print("\n💡 关键要点:")
    print("  1. 继承基类（BaseEncoder, BaseFusion）")
    print("  2. 实现forward方法")
    print("  3. 确保输入/输出维度匹配")
    print("  4. 可以随意组合不同的组件")
    
    print("\n🚀 你可以实现:")
    print("  - 不同的编码器（CNN, RNN, Graph-based等）")
    print("  - 不同的融合方式（Attention, Bilinear, Gating等）")
    print("  - 不同的检测头（Multi-class, Hierarchical等）")
    
    print("\n📝 只需3步:")
    print("  1. 定义你的类，继承基类")
    print("  2. 实现__init__和forward方法")
    print("  3. 像乐高积木一样组装到CrossModalErrorDetector中")


if __name__ == "__main__":
    demo_custom_components()


