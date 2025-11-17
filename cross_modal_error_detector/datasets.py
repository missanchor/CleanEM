"""
Dataset implementations supporting different training strategies.
"""

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
    ):
        self.clean_rows = clean_rows
        self.text_descriptions = text_descriptions
        self.corruption_prob = corruption_prob
        self.tabular_processor = tabular_processor or TabularProcessor()
        self.text_processor = text_processor or TextProcessor()

    def __len__(self) -> int:
        return len(self.clean_rows)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, torch.Tensor]:
        clean_row = self.clean_rows[idx]
        text = self.text_descriptions[idx]
        corrupted_row, labels = self._corrupt_row(clean_row)
        # TODO: 这里corruption最好可以同时考虑text和row
        tabular_inputs = self.tabular_processor.process(corrupted_row, row_idx=0)
        text_inputs = self.text_processor.process(text)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return tabular_inputs, text_inputs, labels_tensor

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
        clean_rows: List[List[Any]],
        text_descriptions: List[str],
        tabular_processor: Optional[TabularProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
    ):
        self.clean_rows = clean_rows
        self.text_descriptions = text_descriptions
        self.tabular_processor = tabular_processor or TabularProcessor()
        self.text_processor = text_processor or TextProcessor()

    def __len__(self) -> int:
        return len(self.clean_rows)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        clean_row = self.clean_rows[idx]
        text = self.text_descriptions[idx]
        tabular_inputs = self.tabular_processor.process(clean_row, row_idx=0)
        text_inputs = self.text_processor.process(text)
        return tabular_inputs, text_inputs


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

        tabular_inputs = self.tabular_processor.process(dirty_row, row_idx=idx)
        text_inputs = self.text_processor.process(text)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return tabular_inputs, text_inputs, labels_tensor


__all__ = ["CorruptionBasedDataset", "ContrastiveDataset", "CleanDirtyEvaluationDataset"]

