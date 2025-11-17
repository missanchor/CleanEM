from __future__ import annotations

from typing import Dict, Optional

import torch


def _resolve_runtime_device(
    default_device: str,
    device_map: Optional[Dict[str, str]],
    key: str,
) -> str:
    if device_map:
        if key in device_map:
            return device_map[key]
        if "default" in device_map:
            return device_map["default"]
    return default_device


def _move_batch_to_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    target = torch.device(device)
    return {k: (v if v.device == target else v.to(target)) for k, v in inputs.items()}


__all__ = ["_resolve_runtime_device", "_move_batch_to_device"]


