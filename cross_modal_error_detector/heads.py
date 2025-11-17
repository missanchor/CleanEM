"""
Detection head implementations.
"""

import torch
import torch.nn as nn

from .base import BaseDetectionHead


class MLPDetectionHead(BaseDetectionHead):
    """
    Cell-level binary classification head.
    """

    def __init__(self, d_input: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, H_fuse: torch.Tensor) -> torch.Tensor:
        return self.mlp(H_fuse)


class ContrastiveDetectionHead(BaseDetectionHead):
    """
    Global row-text matching head for contrastive training.
    """

    def __init__(self, d_input: int):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.ReLU(),
            nn.Linear(d_input // 2, 1),
        )

    def forward(self, H_fuse: torch.Tensor) -> torch.Tensor:
        H_pooled = self.pooling(H_fuse.transpose(1, 2)).squeeze(-1)
        return self.projection(H_pooled)


__all__ = ["MLPDetectionHead", "ContrastiveDetectionHead"]


