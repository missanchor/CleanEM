"""
Evaluation dataset builder utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ...datasets import CleanDirtyEvaluationDataset
from ..configuration import _resolve_path_like
from ..data_loading import load_csv_dataset


def _build_eval_dataset(
    exp_cfg: Dict[str, Any],
    config_dir: Path,
    project_root: Path,
) -> Optional[CleanDirtyEvaluationDataset]:
    """
    Helper to build evaluation dataset from config.

    Args:
        exp_cfg: Experiment configuration
        config_dir: Configuration directory
        project_root: Project root directory

    Returns:
        Evaluation dataset or None if not available
    """
    eval_cfg = exp_cfg.get("evaluation")
    if not eval_cfg:
        # Check if using mock data - if so, generate eval data from mock
        mock_cfg = exp_cfg.get("mock_data")
        if mock_cfg is not None:
            print("\n检测到mock数据，正在生成评估数据集...")
            from ..data_loading import MockDataGenerator

            generator = MockDataGenerator(seed=mock_cfg.get("seed", 42))
            dataset_type = mock_cfg.get("type", "employee")
            num_samples = min(mock_cfg.get("num_samples", 50), 20)  # Use fewer samples for eval

            clean_rows, text_descriptions = generator.generate(dataset_type, num_samples)
            # Generate dirty rows by corrupting some cells
            dirty_rows = []
            for row in clean_rows:
                dirty_row = list(row)
                # Randomly corrupt 1-2 cells
                num_cols = len(dirty_row)
                num_corruptions = np.random.randint(1, min(3, num_cols))
                corrupt_indices = np.random.choice(num_cols, num_corruptions, replace=False)

                for idx in corrupt_indices:
                    if isinstance(dirty_row[idx], (int, float)):
                        # Corrupt numeric value
                        if isinstance(dirty_row[idx], int):
                            dirty_row[idx] = int(dirty_row[idx] * np.random.uniform(0.5, 2.0))
                        else:
                            dirty_row[idx] = dirty_row[idx] * np.random.uniform(0.5, 2.0)
                    else:
                        # Corrupt string value
                        dirty_row[idx] = f"CORRUPTED_{dirty_row[idx]}"

                dirty_rows.append(dirty_row)

            # Generate default column names for mock data
            if dataset_type == "employee":
                column_names = ["employee_id", "name", "age", "department", "salary", "city"]
            elif dataset_type == "sales":
                column_names = ["order_id", "product", "quantity", "price", "region", "date"]
            else:
                column_names = [f"col_{i}" for i in range(len(dirty_rows[0]) if dirty_rows else 0)]

            return CleanDirtyEvaluationDataset(
                clean_rows=clean_rows,
                dirty_rows=dirty_rows,
                text_descriptions=text_descriptions,
                column_names=column_names,
            )
        return None

    try:
        clean_data_path = _resolve_path_like(eval_cfg["clean_data_path"], config_dir, project_root)
        dirty_data_path = _resolve_path_like(eval_cfg["dirty_data_path"], config_dir, project_root)

        clean_rows, clean_text_descriptions, _, clean_column_names = load_csv_dataset(clean_data_path)
        dirty_rows, dirty_text_descriptions, _, dirty_column_names = load_csv_dataset(dirty_data_path)

        # Use text descriptions from clean data
        text_descriptions = dirty_text_descriptions
        # Use column names from dirty data (or clean if dirty doesn't have them)
        column_names = dirty_column_names or clean_column_names

        return CleanDirtyEvaluationDataset(
            clean_rows=clean_rows,
            dirty_rows=dirty_rows,
            text_descriptions=text_descriptions,
            column_names=column_names,
        )
    except Exception as e:
        print(f"警告: 无法构建评估数据集: {e}")
        return None
