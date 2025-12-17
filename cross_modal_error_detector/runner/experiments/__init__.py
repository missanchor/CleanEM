"""
Experiments module for cross-modal error detection.

This module contains the refactored experiment functions that were previously
all in a single large experiments.py file. The functions have been split into
focused modules based on functionality.
"""

from .core import (
    run_corruption_experiment,
    run_contrastive_two_stage_experiment,
)

__all__ = [
    "run_corruption_experiment",
    "run_contrastive_two_stage_experiment",
]
