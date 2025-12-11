"""
Core Cross-Modal Error Detector model assembly.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDetectionHead, BaseEncoder, BaseFusion
from .utils.device import canonicalize_device_map, resolve_runtime_device


class CrossModalErrorDetector(nn.Module):
    """
    Container model that wires together modality encoders, fusion, and detection
    heads using dependency injection.
    """

    def __init__(
        self,
        tabular_encoder: BaseEncoder,
        text_encoder: BaseEncoder,
        fusion_module: BaseFusion,
        detection_head: BaseDetectionHead,
        *,
        device_map: Optional[Dict[str, str]] = None,
        default_device: str = "cpu",
    ):
        super().__init__()
        self.tabular_encoder = tabular_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        self.detection_head = detection_head
        self.default_device = resolve_runtime_device(default_device)
        raw_device_map = canonicalize_device_map(device_map) if device_map else {}
        self.device_map = {
            key: resolve_runtime_device(value, fallback_device=self.default_device)
            for key, value in raw_device_map.items()
        }
        self.device_map.setdefault("default", self.default_device)
        self.module_devices: Dict[str, torch.device] = {}
        self._apply_device_allocation()

    @staticmethod
    def _move_inputs_to_device(
        inputs: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if not inputs:
            return inputs
        return {k: (v if v.device == device else v.to(device)) for k, v in inputs.items()}

    @staticmethod
    def _maybe_to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        return tensor.to(device)

    def _resolve_device(self, key: str) -> torch.device:
        if key in self.device_map:
            return torch.device(self.device_map[key])
        if "default" in self.device_map:
            return torch.device(self.device_map["default"])
        return torch.device(self.default_device)

    def _apply_device_allocation(self) -> None:
        modules = {
            "tabular_encoder": self.tabular_encoder,
            "text_encoder": self.text_encoder,
            "fusion_module": self.fusion_module,
            "detection_head": self.detection_head,
        }
        for name, module in modules.items():
            target_device = self._resolve_device(name)
            module.to(target_device)
            self.module_devices[name] = target_device

    def forward(
        self,
        tabular_inputs: Dict[str, torch.Tensor],
        text_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            tabular_inputs: Dictionary of table inputs.
            text_inputs: Dictionary of text inputs.

        Returns:
            torch.Tensor: Cell-level logits of shape
            [batch_size, num_cols] or [batch_size, num_cols, 1].
        """
        tabular_device = self.module_devices.get("tabular_encoder", torch.device(self.default_device))
        text_device = self.module_devices.get("text_encoder", torch.device(self.default_device))
        fusion_device = self.module_devices.get("fusion_module", torch.device(self.default_device))
        head_device = self.module_devices.get("detection_head", torch.device(self.default_device))

        tabular_inputs = self._move_inputs_to_device(tabular_inputs, tabular_device)
        text_inputs = self._move_inputs_to_device(text_inputs, text_device)

        H_table = self.tabular_encoder(tabular_inputs)
        
        if "cached_embedding" in text_inputs:
            H_text = text_inputs["cached_embedding"]
        else:
            H_text = self.text_encoder(text_inputs)

        H_table = self._maybe_to(H_table, fusion_device)
        H_text = self._maybe_to(H_text, fusion_device)

        H_fuse = self.fusion_module(H_table, H_text)
        H_fuse = self._maybe_to(H_fuse, head_device)
        return self.detection_head(H_fuse)


__all__ = ["CrossModalErrorDetector"]


