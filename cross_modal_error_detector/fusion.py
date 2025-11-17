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


__all__ = ["CrossAttentionFusion", "SimpleConcatFusion"]


