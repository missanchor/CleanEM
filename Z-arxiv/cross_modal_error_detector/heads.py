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


class TabularReconstructionHead(BaseDetectionHead):
    """
    Reconstruction head for Masked Cell Modeling (MCM).

    It supports mixed-type columns by providing:
      - categorical logits over a hashed bucket vocabulary
      - numeric regression for normalized scalar values
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 10000,
        numeric_hidden_dim: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.categorical_head = nn.Linear(d_model, self.vocab_size)

        if numeric_hidden_dim and int(numeric_hidden_dim) > 0:
            h = int(numeric_hidden_dim)
            self.numeric_head = nn.Sequential(
                nn.Linear(d_model, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
        else:
            self.numeric_head = nn.Linear(d_model, 1)

    def forward(
        self,
        H_fuse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H_fuse: [B, C, d_model]

        Returns:
            cat_logits: [B, C, vocab_size]
            num_pred: [B, C] (normalized scalar prediction)
        """
        cat_logits = self.categorical_head(H_fuse)
        num_pred = self.numeric_head(H_fuse).squeeze(-1)
        return cat_logits, num_pred


__all__ = ["MLPDetectionHead", "ContrastiveDetectionHead", "TabularReconstructionHead"]


