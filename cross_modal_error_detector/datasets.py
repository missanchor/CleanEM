"""
Dataset implementations supporting different training strategies.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .processors import TabularProcessor, TextProcessor


class CorruptionBasedDataset(Dataset):
    """
    Corrupts clean rows to train cell-level detectors.
    """

    def __init__(
        self,
        clean_rows: List[List[Any]],
        text_descriptions: List[str],
        corruption_prob: float = 0.3,
        tabular_processor: Optional[TabularProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        cached_text_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        column_names: Optional[List[str]] = None,
    ):
        self.clean_rows = clean_rows
        self.text_descriptions = text_descriptions
        self.corruption_prob = corruption_prob
        self.tabular_processor = tabular_processor or TabularProcessor()
        self.text_processor = text_processor or TextProcessor()
        self.cached_text_embeddings = cached_text_embeddings
        self.column_names = column_names

    def __len__(self) -> int:
        return len(self.clean_rows)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, torch.Tensor]:
        clean_row = self.clean_rows[idx]
        text = self.text_descriptions[idx]
        corrupted_row, labels = self._corrupt_row(clean_row)
        # TODO: 这里corruption最好可以同时考虑text和row
        row_payload = {
            "row_data": corrupted_row,
            "row_idx": 0,
        }
        if self.column_names is not None:
            row_payload["column_names"] = self.column_names

        if self.cached_text_embeddings is not None and idx in self.cached_text_embeddings:
            # Use cached embedding if available
            # Wrap in a dict to match expected text_inputs format for model forwarding
            # But model expects input_ids/attention_mask usually. 
            # We need to adjust the model or collation to handle pre-computed embeddings.
            # For now, let's return a special key that collate_fn can recognize
            text_inputs = {"cached_embedding": self.cached_text_embeddings[idx]}
        else:
            text_inputs = self.text_processor.process(text)
            
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return row_payload, text_inputs, labels_tensor

    def _corrupt_row(self, clean_row: List[Any]) -> Tuple[List[Any], List[int]]:
        corrupted_row = []
        labels = []
        for cell in clean_row:
            if np.random.rand() < self.corruption_prob:
                corrupted_row.append(self._corrupt_cell(cell))
                labels.append(0)
            else:
                corrupted_row.append(cell)
                labels.append(1)
        return corrupted_row, labels

    def _corrupt_cell(self, cell: Any) -> Any:
        if isinstance(cell, (int, float)):
            return cell * np.random.uniform(0.5, 1.5) + np.random.randn() * 10
        if isinstance(cell, str):
            # TODO: 这里corruption最好有更多选择
            corruptions = ["ERROR", "NULL", "INVALID", "###", str(np.random.randint(0, 1000))]
            return np.random.choice(corruptions)
        return None


class ContrastiveDataset(Dataset):
    """
    Provides matching rows and texts for contrastive learning.
    """

    def __init__(
        self,
        dirty_rows: List[List[Any]],
        text_descriptions: List[str],
        tabular_processor: Optional[TabularProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        cached_text_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        column_names: Optional[List[str]] = None,
    ):
        self.dirty_rows = dirty_rows
        self.text_descriptions = text_descriptions
        self.tabular_processor = tabular_processor or TabularProcessor()
        self.text_processor = text_processor or TextProcessor()
        self.cached_text_embeddings = cached_text_embeddings
        self.column_names = column_names

    def __len__(self) -> int:
        return len(self.dirty_rows)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        dirty_row = self.dirty_rows[idx]
        text = self.text_descriptions[idx]
        row_payload = {
            "row_data": dirty_row,
            "row_idx": 0,
        }
        if self.column_names is not None:
            row_payload["column_names"] = self.column_names

        if self.cached_text_embeddings is not None and idx in self.cached_text_embeddings:
            text_inputs = {"cached_embedding": self.cached_text_embeddings[idx]}
        else:
            text_inputs = self.text_processor.process(text)
            
        return row_payload, text_inputs


class CleanDirtyEvaluationDataset(Dataset):
    """
    Uses paired clean + dirty rows to create deterministic evaluation samples.
    """

    def __init__(
        self,
        clean_rows: List[List[Any]],
        dirty_rows: List[List[Any]],
        text_descriptions: List[str],
        tabular_processor: Optional[TabularProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        column_names: Optional[List[str]] = None,
    ):
        if len(clean_rows) != len(dirty_rows):
            raise ValueError("clean_rows 与 dirty_rows 的长度不一致，无法对齐监督标签。")
        if len(text_descriptions) != len(dirty_rows):
            raise ValueError("text_descriptions 数量应与 dirty_rows 相同。")

        self.clean_rows = clean_rows
        self.dirty_rows = dirty_rows
        self.text_descriptions = text_descriptions
        self.tabular_processor = tabular_processor or TabularProcessor()
        self.text_processor = text_processor or TextProcessor()
        self.column_names = column_names
        self.labels = self._build_labels()

    def _build_labels(self) -> List[List[int]]:
        labels: List[List[int]] = []
        for clean_row, dirty_row in zip(self.clean_rows, self.dirty_rows):
            if len(clean_row) != len(dirty_row):
                raise ValueError("clean_row 与 dirty_row 的列数不同，无法生成标签。")
            row_labels = [1 if dirty_cell == clean_cell else 0 for clean_cell, dirty_cell in zip(clean_row, dirty_row)]
            labels.append(row_labels)
        return labels

    def __len__(self) -> int:
        return len(self.dirty_rows)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, torch.Tensor]:
        dirty_row = self.dirty_rows[idx]
        text = self.text_descriptions[idx]
        labels = self.labels[idx]

        row_payload = {
            "row_data": dirty_row,
            "row_idx": idx,
        }
        if self.column_names is not None:
            row_payload["column_names"] = self.column_names

        text_inputs = self.text_processor.process(text)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return row_payload, text_inputs, labels_tensor


class PerColumnBinaryDataset(Dataset):
    """
    Per-column binary classification dataset for Stage B.
    Each sample is a row + text description, grouped by column index.
    Used for training column-specific MLP classifiers.

    Samples are organized as: (row_data, text, col_idx, label)
    - Positive: clean row + correct text (label=1)
    - Negative: clean row + wrong text OR dirty row + correct text (label=0)
    """

    def __init__(
        self,
        clean_rows: List[List[Any]],
        text_descriptions: List[str],
        negative_ratio: float = 1.0,
        negative_strategy: str = "random",
        dirty_rows: Optional[List[List[Any]]] = None,
        tabular_processor: Optional[TabularProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        seed: Optional[int] = None,
        column_names: Optional[List[str]] = None,
    ):
        self.clean_rows = clean_rows
        self.text_descriptions = text_descriptions
        self.dirty_rows = dirty_rows
        self.negative_ratio = negative_ratio
        self.negative_strategy = negative_strategy
        self.tabular_processor = tabular_processor or TabularProcessor()
        self.text_processor = text_processor or TextProcessor()
        self.column_names = column_names
        self.num_cols = len(clean_rows[0]) if clean_rows else 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build row-level samples grouped by column
        self.samples = self._build_samples()

        # Group samples by column for column-wise training
        self.samples_by_column = self._group_by_column()

    def _build_samples(self) -> List[Tuple[List[Any], str, int, int]]:
        """
        Build row-level samples for each column.
        Returns: List of (row_data, text, col_idx, label)
        """
        samples = []
        num_rows = len(self.clean_rows)

        for col_idx in range(self.num_cols):
            # Positive samples: clean row + correct text for this column
            for row_idx, (row, text) in enumerate(zip(self.clean_rows, self.text_descriptions)):
                samples.append((row, text, col_idx, 1))

            # Negative samples
            num_negatives = int(num_rows * self.negative_ratio)

            if self.negative_strategy == "random":
                # Random row-text mismatch
                for _ in range(num_negatives):
                    row_idx = random.randint(0, num_rows - 1)
                    # Choose a different text (wrong description)
                    wrong_text_idx = random.choice([i for i in range(num_rows) if i != row_idx])
                    samples.append((self.clean_rows[row_idx], self.text_descriptions[wrong_text_idx], col_idx, 0))

            elif self.negative_strategy == "dirty" and self.dirty_rows:
                # Use dirty rows as negative samples
                for row_idx in range(min(num_negatives, len(self.dirty_rows))):
                    samples.append((self.dirty_rows[row_idx], self.text_descriptions[row_idx], col_idx, 0))

        return samples

    def _group_by_column(self) -> Dict[int, List[int]]:
        """Group sample indices by column."""
        by_column = {col_idx: [] for col_idx in range(self.num_cols)}
        for idx, (_, _, col_idx, _) in enumerate(self.samples):
            by_column[col_idx].append(idx)
        return by_column

    def get_column_samples(self, col_idx: int) -> List[int]:
        """Get sample indices for a specific column."""
        return self.samples_by_column.get(col_idx, [])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, torch.Tensor, int]:
        row_data, text, col_idx, label = self.samples[idx]

        row_payload = {
            "row_data": row_data,
            "row_idx": 0,
        }
        if self.column_names is not None:
            row_payload["column_names"] = self.column_names

        text_inputs = self.text_processor.process(text)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        col_idx_tensor = torch.tensor(col_idx, dtype=torch.long)
        return row_payload, text_inputs, label_tensor, col_idx_tensor


__all__ = [
    "CorruptionBasedDataset",
    "ContrastiveDataset",
    "CleanDirtyEvaluationDataset",
    "PerColumnBinaryDataset",
]

