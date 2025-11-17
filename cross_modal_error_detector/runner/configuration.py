from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..processors import TabularProcessor, TextProcessor


def _resolve_path_like(value: Any, config_dir: Path, project_root: Path) -> Any:
    if isinstance(value, (str, Path)):
        path_obj = Path(value).expanduser()
        if path_obj.is_absolute():
            return path_obj.resolve()

        first_part = path_obj.parts[0] if path_obj.parts else ""
        base_dir = config_dir if first_part in {".", ".."} else project_root
        return (base_dir / path_obj).resolve()
    return value


def _resolve_component_params(
    params: Dict[str, Any],
    config_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in params.items():
        if key in {"model_name_or_path", "tokenizer_name_or_path", "cache_dir"}:
            resolved[key] = str(_resolve_path_like(value, config_dir, project_root))
        elif isinstance(value, dict):
            resolved[key] = _resolve_component_params(value, config_dir, project_root)
        elif isinstance(value, list):
            resolved_list = []
            for item in value:
                if isinstance(item, (str, Path)):
                    resolved_list.append(str(_resolve_path_like(item, config_dir, project_root)))
                else:
                    resolved_list.append(item)
            resolved[key] = resolved_list
        else:
            resolved[key] = value
    return resolved


def _instantiate_processors(
    exp_cfg: Dict[str, Any],
    config_dir: Path,
    project_root: Path,
) -> Tuple[TabularProcessor, TextProcessor]:
    tabular_kwargs = exp_cfg.get("tabular_processor", {})
    text_kwargs = exp_cfg.get("text_processor", {})
    resolved_tabular_kwargs = (
        _resolve_component_params(tabular_kwargs, config_dir, project_root) if tabular_kwargs else {}
    )
    resolved_text_kwargs = (
        _resolve_component_params(text_kwargs, config_dir, project_root) if text_kwargs else {}
    )
    tabular_processor = TabularProcessor(**resolved_tabular_kwargs)
    text_processor = TextProcessor(**resolved_text_kwargs)
    return tabular_processor, text_processor


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "_resolve_path_like",
    "_resolve_component_params",
    "_instantiate_processors",
    "load_config",
    "set_seed",
]


