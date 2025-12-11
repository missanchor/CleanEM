"""
Device utility helpers for canonicalizing runtime device strings.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


_DEVICE_ALIASES = {
    "gpu": "cuda",
    "cuda": "cuda",
    "cpu": "cpu",
    "xpu": "xpu",
    "ipu": "ipu",
    "mlu": "mlu",
    "mps": "mps",
    "hpu": "hpu",
    "dipu": "dipu",
    "meta": "meta",
    "hip": "hip",
    "privateuseone": "privateuseone",
}


def canonicalize_device(device: str) -> str:
    """
    Normalize device strings to values accepted by ``torch.device``.

    Supports legacy aliases like ``"gpu"`` -> ``"cuda"`` and preserves explicit
    device indices (e.g. ``"gpu:1"`` -> ``"cuda:1"``).
    """

    if device is None:
        raise ValueError("Device string cannot be None.")

    device_str = str(device).strip()
    if not device_str:
        raise ValueError("Device string cannot be empty.")

    if ":" in device_str:
        prefix, suffix = device_str.split(":", 1)
        normalized_prefix = _DEVICE_ALIASES.get(prefix.lower(), prefix.lower())
        return f"{normalized_prefix}:{suffix}"

    return _DEVICE_ALIASES.get(device_str.lower(), device_str.lower())


def _safe_cuda_memory_snapshot(device_index: int) -> Tuple[int, int]:
    """
    Return (free_bytes, total_bytes) for a CUDA device, tolerating driver quirks.
    """

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        return int(free_bytes), int(total_bytes)
    except (RuntimeError, AttributeError):
        # Fall back to properties + allocated memory estimation.
        try:
            props = torch.cuda.get_device_properties(device_index)
            total_bytes = int(getattr(props, "total_memory", 0))
            allocated = int(torch.cuda.memory_allocated(device_index))
            free_bytes = max(total_bytes - allocated, 0)
            return free_bytes, total_bytes
        except Exception:
            return 0, 0


def select_optimal_cuda_device() -> Optional[str]:
    """
    Pick the CUDA device with the most free memory.
    """

    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None

    best_index: Optional[int] = None
    best_free_bytes = -1

    for idx in range(device_count):
        free_bytes, _ = _safe_cuda_memory_snapshot(idx)
        if free_bytes > best_free_bytes:
            best_free_bytes = free_bytes
            best_index = idx

    if best_index is None:
        return None

    return f"cuda:{best_index}"


def resolve_runtime_device(device: Optional[str], fallback_device: str = "cpu") -> str:
    """
    Resolve the concrete runtime device, automatically picking the optimal GPU
    with the most free memory when CUDA is available.
    """

    candidate = device or ("cuda" if torch.cuda.is_available() else fallback_device)
    normalized = canonicalize_device(candidate)

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            return fallback_device

        # Always use select_optimal_cuda_device() to pick the best GPU
        best_device = select_optimal_cuda_device()
        if best_device:
            return best_device
        return fallback_device

    return normalized


def canonicalize_device_map(
    device_map: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """
    Canonicalize device map entries; returns a new dict (never mutates input).
    """

    if not device_map:
        return {}

    return {key: canonicalize_device(value) for key, value in device_map.items()}


__all__ = [
    "canonicalize_device",
    "canonicalize_device_map",
    "resolve_runtime_device",
    "select_optimal_cuda_device",
]


