"""
Modality fusion implementations.
"""

import torch
import torch.nn as nn

from .base import BaseFusion


class CrossAttentionFusion(BaseFusion):
    """
    Uses cross-attention to align table queries with text keys/values.
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_projection = nn.Linear(d_model * 3, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_table: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        H_aligned, _ = self.cross_attention(
            query=H_table,
            key=H_text,
            value=H_text,
        )
        diff = torch.abs(H_table - H_aligned)
        H_concat = torch.cat([H_table, H_aligned, diff], dim=-1)
        H_fuse = self.fusion_projection(H_concat)
        H_fuse = self.dropout(H_fuse)
        H_fuse = self.layer_norm(H_fuse)
        return H_fuse


class SimpleConcatFusion(BaseFusion):
    """
    Naive pooling + concatenation fusion for ablation studies.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fusion_projection = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, H_table: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        batch_size, num_cols, _ = H_table.shape
        H_text_pooled = H_text.mean(dim=1)
        H_text_expanded = H_text_pooled.unsqueeze(1).expand(-1, num_cols, -1)
        H_concat = torch.cat([H_table, H_text_expanded], dim=-1)
        H_fuse = self.fusion_projection(H_concat)
        H_fuse = self.layer_norm(H_fuse)
        return H_fuse


class SingleColumnFusion(BaseFusion):
    """
    Optimized fusion module that processes only a single column at a time.
    This avoids computing fusion for all columns when only one column is needed.

    Args:
        fusion_type: Type of fusion to use ('CrossAttentionFusion' or 'SimpleConcatFusion')
        d_model: Hidden dimension size
        **kwargs: Additional parameters for the fusion module
    """

    def __init__(self, fusion_type: str, d_model: int, **kwargs):
        super().__init__()
        if fusion_type == "CrossAttentionFusion":
            self.fusion_module = CrossAttentionFusion(d_model=d_model, **kwargs)
        elif fusion_type == "SimpleConcatFusion":
            self.fusion_module = SimpleConcatFusion(d_model=d_model)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, H_table: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single column.

        Args:
            H_table: [batch, 1, d_model] - single column embedding
            H_text: [batch, seq_len, d_model] - text embeddings

        Returns:
            H_fused: [batch, 1, d_model] - fused single column embedding
        """
        # Ensure H_table has shape [batch, 1, d_model]
        if H_table.dim() == 2:
            H_table = H_table.unsqueeze(1)  # [batch, d_model] -> [batch, 1, d_model]
        elif H_table.shape[1] != 1:
            raise ValueError(f"SingleColumnFusion expects H_table with 1 column, got shape {H_table.shape}")

        # Forward pass through the underlying fusion module
        H_fused = self.fusion_module(H_table, H_text)  # [batch, 1, d_model]
        return H_fused


class PerColumnFusionWrapper(BaseFusion):
    """
    Wrapper that creates separate fusion modules for each column.
    Each column gets its own fusion module, similar to per-column MLPs.
    """

    def __init__(self, fusion_type: str, d_model: int, num_cols: int, **kwargs):
        super().__init__()
        # 创建num_cols个独立的fusion模块
        self.fusion_modules = nn.ModuleList()

        for _ in range(num_cols):
            if fusion_type == "CrossAttentionFusion":
                module = CrossAttentionFusion(d_model=d_model, **kwargs)
            elif fusion_type == "SimpleConcatFusion":
                module = SimpleConcatFusion(d_model=d_model)
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
            self.fusion_modules.append(module)

    def forward(self, H_table: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        batch_size, num_cols, d_model = H_table.shape

        # 为每列应用对应的fusion模块
        fused_embeddings = []
        for col_idx in range(num_cols):
            col_embedding = H_table[:, col_idx, :].unsqueeze(1)  # [batch, 1, d_model]
            fused = self.fusion_modules[col_idx](col_embedding, H_text)  # [batch, 1, d_model]
            fused_embeddings.append(fused.squeeze(1))  # [batch, d_model]

        # 重新堆叠为 [batch, num_cols, d_model]
        return torch.stack(fused_embeddings, dim=1)


class CellLevelFusionWrapper(BaseFusion):
    """
    Wrapper that creates separate fusion modules for each column and supports cell-level text embeddings.
    Each cell uses its corresponding text embedding for fusion, enabling true cell-level semantic alignment.

    This is different from PerColumnFusionWrapper which still uses the same text embedding for all cells in a column.
    Here, each (row, col) pair has its own text embedding.

    Args:
        fusion_type: Type of fusion to use ('CrossAttentionFusion' or 'SimpleConcatFusion')
        d_model: Hidden dimension size
        num_cols: Number of columns in the table
        **kwargs: Additional parameters for the fusion module
    """

    def __init__(self, fusion_type: str, d_model: int, num_cols: int, **kwargs):
        super().__init__()
        # 创建num_cols个独立的fusion模块
        self.fusion_modules = nn.ModuleList()

        for _ in range(num_cols):
            if fusion_type == "CrossAttentionFusion":
                module = CrossAttentionFusion(d_model=d_model, **kwargs)
            elif fusion_type == "SimpleConcatFusion":
                module = SimpleConcatFusion(d_model=d_model)
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
            self.fusion_modules.append(module)

    def forward(self, H_table: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cell-level text embeddings.

        Args:
            H_table: [batch, num_cols, d_model] - table cell embeddings
            H_text: [batch, num_cols, d_model] - cell-level text embeddings
                    Each H_text[b, c] corresponds to the text embedding for cell (b, c)

        Returns:
            H_fused: [batch, num_cols, d_model] - fused embeddings with cell-level alignment
        """
        batch_size, num_cols, d_model = H_table.shape

        # Verify H_text has the expected shape
        if H_text.shape != (batch_size, num_cols, d_model):
            raise ValueError(
                f"CellLevelFusionWrapper expects H_text with shape [{batch_size}, {num_cols}, {d_model}], "
                f"got {H_text.shape}"
            )

        # 为每列应用对应的fusion模块，每个cell使用其对应的文本embedding
        fused_embeddings = []
        for col_idx in range(num_cols):
            # 提取该列的所有cell embeddings
            col_table_embedding = H_table[:, col_idx, :].unsqueeze(1)  # [batch, 1, d_model]

            # 提取该列所有cell对应的文本embeddings
            col_text_embedding = H_text[:, col_idx, :].unsqueeze(1)  # [batch, 1, d_model]

            # 应用该列的fusion模块，每个cell使用其对应的文本embedding
            fused = self.fusion_modules[col_idx](col_table_embedding, col_text_embedding)  # [batch, 1, d_model]
            fused_embeddings.append(fused.squeeze(1))  # [batch, d_model]

        # 重新堆叠为 [batch, num_cols, d_model]
        return torch.stack(fused_embeddings, dim=1)


__all__ = [
    "CrossAttentionFusion",
    "SimpleConcatFusion",
    "PerColumnFusionWrapper",
    "SingleColumnFusion",
    "CellLevelFusionWrapper",
]

