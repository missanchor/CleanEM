"""
Backward compatibility layer for experiments module.

This file has been refactored into multiple modules in the experiments/ directory.
All exports are now re-exported from the new modular structure.
"""
from __future__ import annotations

# Re-export everything from the new modular structure
from .experiments.core import (
    run_corruption_experiment,
    run_contrastive_two_stage_experiment,
)

# Re-export utility functions for backward compatibility
from .experiments.column_profiling import (
    _build_column_profiles,
    _compute_rule_score,
    _constraint_violation_score,
    _corrupt_row_cell,
    _corrupt_value,
    _infer_dominant_string_pattern,
    _make_typo,
)
from .experiments.data_utils import (
    _attach_processor_params,
    _build_eval_dataset_from_config,
    _derive_loader_runtime,
    _prepare_eval_loader,
    _resolve_num_workers,
    _safe_str,
    _try_parse_float,
)
from .experiments.embeddings import _precompute_text_embeddings
from .experiments.error_analysis import (
    compute_error_mask,
    evaluate_per_column_model,
    print_detailed_error_analysis,
)
from .experiments.eval_dataset_builder import _build_eval_dataset
from .experiments.evaluation import (
    _compute_classification_metrics,
    _evaluate_corruption_model,
    _evaluate_threshold_with_scores,
)
from .experiments.negative_sampling import (
    _generate_negative_pairs,
    _parse_negative_sampling_config,
    _sample_different_index,
    _sample_row_text_mismatch,
    _sample_row_text_mismatch_different_column_value,
    _sample_row_text_swap,
    _sample_similar_text_pairs,
    _build_semantic_neighbor_map,
)
from .experiments.types import _ColumnProfile

__all__ = [
    "run_corruption_experiment",
    "run_contrastive_two_stage_experiment",
    "_ColumnProfile",
    "_build_column_profiles",
    "_compute_rule_score",
    "_constraint_violation_score",
    "_corrupt_row_cell",
    "_corrupt_value",
    "_infer_dominant_string_pattern",
    "_make_typo",
    "_attach_processor_params",
    "_build_eval_dataset_from_config",
    "_derive_loader_runtime",
    "_prepare_eval_loader",
    "_resolve_num_workers",
    "_safe_str",
    "_try_parse_float",
    "_precompute_text_embeddings",
    "compute_error_mask",
    "evaluate_per_column_model",
    "print_detailed_error_analysis",
    "_build_eval_dataset",
    "_compute_classification_metrics",
    "_evaluate_corruption_model",
    "_evaluate_threshold_with_scores",
    "_generate_negative_pairs",
    "_parse_negative_sampling_config",
    "_sample_different_index",
    "_sample_row_text_mismatch",
    "_sample_row_text_mismatch_different_column_value",
    "_sample_row_text_swap",
    "_sample_similar_text_pairs",
    "_build_semantic_neighbor_map",
]
