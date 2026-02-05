"""
Base interface definitions for the Cross-Modal Error Detector package.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """
    Base class for modality-specific encoders.
    """

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: Modality-specific inputs.

        Returns:
            torch.Tensor: Encoded representations with shape
            [batch_size, seq_len, d_model].
        """
        raise NotImplementedError


class BaseFusion(nn.Module, ABC):
    """
    Base class for fusion modules that combine table and text features.
    """

    @abstractmethod
    def forward(self, H_table: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_table: Table representations [batch_size, num_cols, d_model].
            H_text: Text representations [batch_size, text_seq_len, d_model].

        Returns:
            torch.Tensor: Fused representations
            [batch_size, num_cols, d_fused].
        """
        raise NotImplementedError


class BaseDetectionHead(nn.Module, ABC):
    """
    Base class for detection heads.
    """

    @abstractmethod
    def forward(self, H_fuse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_fuse: Fused representations [batch_size, num_cols, d_fused].

        Returns:
            torch.Tensor: Predictions whose shapes depend on the task.
        """
        raise NotImplementedError


__all__ = ["BaseEncoder", "BaseFusion", "BaseDetectionHead"]

