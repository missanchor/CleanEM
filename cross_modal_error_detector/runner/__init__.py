"""
Runner utilities for the Cross-Modal Error Detector experiments.

This submodule provides a thin layer that decouples the oversized
`main_cross_modal_detector.py` script into focused components:

- configuration helpers
- dataset utilities
- component factories
- standardized experiment runners
"""

from .configuration import load_config, set_seed
from .experiments import (
    run_ablation_experiment,
    run_contrastive_experiment,
    run_corruption_experiment,
)

__all__ = [
    "load_config",
    "set_seed",
    "run_corruption_experiment",
    "run_contrastive_experiment",
    "run_ablation_experiment",
]


