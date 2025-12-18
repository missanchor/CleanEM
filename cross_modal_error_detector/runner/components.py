from __future__ import annotations

from typing import Any, Dict

import torch

from ..encoders import PretrainedTextEncoder, SimpleMLPEncoder, StructureAwareTransformer
from ..fusion import CrossAttentionFusion, SimpleConcatFusion
from ..heads import ContrastiveDetectionHead, MLPDetectionHead, TabularReconstructionHead


ENCODER_REGISTRY = {
    "StructureAwareTransformer": StructureAwareTransformer,
    "PretrainedTextEncoder": PretrainedTextEncoder,
    "SimpleMLPEncoder": SimpleMLPEncoder,
}

FUSION_REGISTRY = {
    "CrossAttentionFusion": CrossAttentionFusion,
    "SimpleConcatFusion": SimpleConcatFusion,
}

DETECTION_HEAD_REGISTRY = {
    "MLPDetectionHead": MLPDetectionHead,
    "ContrastiveDetectionHead": ContrastiveDetectionHead,
    "TabularReconstructionHead": TabularReconstructionHead,
}

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "Adam": torch.optim.Adam,
}


def build_component(
    component_cfg: Dict[str, Any],
    registry: Dict[str, Any],
    config_dir,
    project_root,
) -> Any:
    component_type = component_cfg.get("type")
    if component_type not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown component type '{component_type}'. Available: {available}")
    params = component_cfg.get("params", {})
    from .configuration import _resolve_component_params  # Local import to avoid cycle

    resolved_params = _resolve_component_params(params, config_dir, project_root)
    return registry[component_type](**resolved_params)


def build_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_type = optimizer_cfg.get("type", "Adam")
    params = optimizer_cfg.get("params", {"lr": 1e-4})
    optimizer_cls = OPTIMIZER_REGISTRY.get(optimizer_type, None)
    if optimizer_cls is None:
        available = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(f"Unknown optimizer '{optimizer_type}'. Available: {available}")
    return optimizer_cls(model.parameters(), **params)


__all__ = [
    "ENCODER_REGISTRY",
    "FUSION_REGISTRY",
    "DETECTION_HEAD_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "build_component",
    "build_optimizer",
]


