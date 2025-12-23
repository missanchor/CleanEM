"""
Dataset implementations supporting different training strategies.
"""

import random
import hashlib
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
        cached_text_embeddings: Optional[Dict] = None,
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
        cached_text_embeddings: Optional[Dict] = None,
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


class MaskedCellModelingDataset(Dataset):
    """
    Masked Cell Modeling (MCM) dataset for self-supervised reconstruction.

    Each sample returns:
      - row_payload: {"row_data": [...], "row_idx": idx, "column_names": [...]?}
      - text_inputs: tokenized text or {"cached_embedding": ...}
      - mask_indices: LongTensor[K] (column indices masked in this row)
      - target_ids: LongTensor[K] (categorical targets; -1 for numeric positions)
      - target_nums: FloatTensor[K] (numeric targets in [0,1]; NaN for categorical positions)
      - target_types: LongTensor[K] (0=categorical, 1=numeric)
    """

    def __init__(
        self,
        dirty_rows: List[List[Any]],
        text_descriptions: List[str],
        *,
        mask_ratio: float = 0.2,
        num_masked_cells: Optional[int] = None,
        numeric_ratio_threshold: float = 0.9,
        max_rows_for_type_inference: int = 2000,
        tabular_processor: Optional[TabularProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        cached_text_embeddings: Optional[Dict] = None,
        column_names: Optional[List[str]] = None,
        col_is_numeric: Optional[List[bool]] = None,
        column_value_to_id: Optional[List[Dict[str, int]]] = None,
        unknown_token: str = "<UNK>",
        seed: Optional[int] = None,
    ):
        if len(dirty_rows) != len(text_descriptions):
            raise ValueError("dirty_rows 与 text_descriptions 的长度不一致，无法对齐。")
        self.dirty_rows = dirty_rows
        self.text_descriptions = text_descriptions
        self.tabular_processor = tabular_processor or TabularProcessor()
        self.text_processor = text_processor or TextProcessor()
        self.cached_text_embeddings = cached_text_embeddings
        self.column_names = column_names
        self.mask_ratio = float(mask_ratio)
        self.num_masked_cells = num_masked_cells
        self.numeric_ratio_threshold = float(numeric_ratio_threshold)
        self.max_rows_for_type_inference = int(max_rows_for_type_inference)
        self.seed = seed
        self.unknown_token = unknown_token

        self.num_cols = len(dirty_rows[0]) if dirty_rows else 0
        self.num_rows = len(dirty_rows)
        if col_is_numeric is not None:
            if self.num_cols and len(col_is_numeric) != self.num_cols:
                raise ValueError("Length of col_is_numeric must match number of columns.")
            self.col_is_numeric = list(col_is_numeric)
        else:
            self.col_is_numeric = self._infer_column_types()

        if column_value_to_id is not None:
            if self.num_cols and len(column_value_to_id) != self.num_cols:
                raise ValueError("Length of column_value_to_id must match number of columns.")
            self.column_value_to_id = [dict(mapping) for mapping in column_value_to_id]
        else:
            self.column_value_to_id = None

        if self.column_value_to_id is not None:
            self.column_unknown_ids = [
                mapping.get(self.unknown_token) if isinstance(mapping, dict) else None
                for mapping in self.column_value_to_id
            ]
        else:
            self.column_unknown_ids: List[Optional[int]] = []

        if self.num_cols > 0 and self.num_rows > 0:
            if self.num_masked_cells is None:
                # 计算整个数据集的总单元格数，然后按 mask_ratio 计算总掩码数量
                total_cells = self.num_rows * self.num_cols
                inferred = int(round(total_cells * self.mask_ratio))
                self.num_masked_cells = max(1, inferred)
            else:
                self.num_masked_cells = max(1, int(self.num_masked_cells))

            # 计算每行应该掩码的单元格数
            self.cells_per_row = max(1, self.num_masked_cells // self.num_rows)
            # 确保不超过列数
            self.cells_per_row = min(self.cells_per_row, self.num_cols)

    def __len__(self) -> int:
        return len(self.dirty_rows)

    def _stable_hash_to_bucket(self, text: str, vocab_size: int) -> int:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % int(vocab_size)

    def _normalize_numeric(self, value: float, num_numeric_bins: int) -> float:
        # Mirror TabularProcessor normalization behavior.
        return (float(value) % float(num_numeric_bins)) / float(num_numeric_bins)

    def _infer_column_types(self) -> List[bool]:
        """
        Infer per-column type (numeric vs categorical) by scanning values.
        """
        if not self.dirty_rows:
            return []
        num_cols = len(self.dirty_rows[0])
        numeric_counts = [0] * num_cols
        total_counts = [0] * num_cols

        limit = min(len(self.dirty_rows), max(1, self.max_rows_for_type_inference))
        for r in range(limit):
            row = self.dirty_rows[r]
            if len(row) != num_cols:
                raise ValueError("dirty_rows must be rectangular (all rows same number of columns).")
            for c, v in enumerate(row):
                if v is None:
                    continue
                if isinstance(v, str) and v.strip() == "":
                    continue
                total_counts[c] += 1
                # Treat bool as categorical-ish by default to reduce accidental numeric inference
                is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                if is_num:
                    numeric_counts[c] += 1

        col_is_numeric: List[bool] = []
        for c in range(num_cols):
            denom = total_counts[c]
            ratio = (numeric_counts[c] / denom) if denom > 0 else 0.0
            col_is_numeric.append(ratio >= self.numeric_ratio_threshold)
        return col_is_numeric

    def __getitem__(self, idx: int):
        row = self.dirty_rows[idx]
        text = self.text_descriptions[idx]
        if self.num_cols and len(row) != self.num_cols:
            raise ValueError("dirty_rows must be rectangular (all rows same number of columns).")

        row_payload: Dict[str, Any] = {
            "row_data": row,
            "row_idx": idx,
        }
        if self.column_names is not None:
            row_payload["column_names"] = self.column_names

        # Generate per-column text prompts for each column
        # Format: Instruction-based prompt following TimeCMA strategy
        # Serialize row data: "Attribute: Value, Attribute: Value..."
        row_serialized = ", ".join([
            f"{self.column_names[c] if self.column_names and c < len(self.column_names) else f'Col{c}'}: {row[c]}"
            for c in range(self.num_cols)
        ])
        
        # For each column, create a column-specific instruction prompt
        # The instruction asks LLM to check consistency for a specific column
        # The output (last token embedding) represents LLM's understanding of the instruction
        per_column_text_inputs: Dict[int, Dict[str, Any]] = {}
        
        first_cached_column_embedding = None

        for col_idx in range(self.num_cols):
            col_name = self.column_names[col_idx] if self.column_names and col_idx < len(self.column_names) else f"Column {col_idx}"
            # Create per-column instruction prompt
            # Format: Instruction + Data + Output placeholder
            # The LLM processes this instruction and outputs embeddings
            # Last token embedding captures LLM's response to the instruction
            col_prompt = (
                f"Instruction: Check the consistency/errorneousness of this record on column '{col_name}'. "
                f"Record: {row_serialized}. "
                f"The summary consistency/errorneousness check result is:"
            )

            cache_key = (idx, col_idx)
            if self.cached_text_embeddings is not None and cache_key in self.cached_text_embeddings:
                cached_embedding = self.cached_text_embeddings[cache_key]
                per_column_text_inputs[col_idx] = {"cached_embedding": cached_embedding}
                if first_cached_column_embedding is None:
                    first_cached_column_embedding = cached_embedding
            else:
                per_column_text_inputs[col_idx] = self.text_processor.process(col_prompt)
        
        # For backward compatibility, also provide a default text input (for non-masked columns)
        # But we'll use per-column inputs for masked columns
        if self.cached_text_embeddings is not None:
            if first_cached_column_embedding is not None:
                # Fall back to the first cached per-column embedding when only (idx, col_idx) keys exist
                text_inputs = {"cached_embedding": first_cached_column_embedding}
            elif idx in self.cached_text_embeddings:
                text_inputs = {"cached_embedding": self.cached_text_embeddings[idx]}
            else:
                text_inputs = self.text_processor.process(text)
        else:
            text_inputs = self.text_processor.process(text)
        
        # Store per-column text inputs in row_payload for use in training
        row_payload["per_column_text_inputs"] = per_column_text_inputs
        
        # Also provide default text_inputs for backward compatibility
        # (will be used if per-column inputs are not available)

        # Deterministic masking per row if seed is provided.
        rng = random.Random((self.seed or 0) + int(idx))

        if self.num_cols <= 0:
            mask_indices = torch.empty((0,), dtype=torch.long)
            target_ids = torch.empty((0,), dtype=torch.long)
            target_nums = torch.empty((0,), dtype=torch.float32)
            target_types = torch.empty((0,), dtype=torch.long)
            return row_payload, text_inputs, mask_indices, target_ids, target_nums, target_types

        # 使用每行固定的掩码数量
        k = self.cells_per_row
        chosen = rng.sample(range(self.num_cols), k=k)
        mask_indices_list = list(chosen)

        vocab_size = int(getattr(self.tabular_processor, "vocab_size", 10000))
        num_numeric_bins = int(getattr(self.tabular_processor, "num_numeric_bins", 1000))

        target_ids_list: List[int] = []
        target_nums_list: List[float] = []
        target_types_list: List[int] = []

        for c in mask_indices_list:
            v = row[c]
            is_num_col = bool(self.col_is_numeric[c]) if c < len(self.col_is_numeric) else False
            if is_num_col and isinstance(v, (int, float)) and not isinstance(v, bool):
                target_types_list.append(1)
                target_ids_list.append(-1)
                target_nums_list.append(self._normalize_numeric(float(v), num_numeric_bins))
            else:
                target_types_list.append(0)
                text_value = "" if v is None else str(v)
                target_id: Optional[int] = None
                if self.column_value_to_id and c < len(self.column_value_to_id):
                    mapping = self.column_value_to_id[c] or {}
                    target_id = mapping.get(text_value)
                    if target_id is None:
                        unknown_id = (
                            self.column_unknown_ids[c]
                            if c < len(self.column_unknown_ids)
                            else None
                        )
                        target_id = unknown_id if unknown_id is not None else 0
                if target_id is None:
                    target_id = self._stable_hash_to_bucket(text_value, vocab_size)
                target_ids_list.append(int(target_id))
                target_nums_list.append(float("nan"))

        mask_indices = torch.tensor(mask_indices_list, dtype=torch.long)
        target_ids = torch.tensor(target_ids_list, dtype=torch.long)
        target_nums = torch.tensor(target_nums_list, dtype=torch.float32)
        target_types = torch.tensor(target_types_list, dtype=torch.long)

        return row_payload, text_inputs, mask_indices, target_ids, target_nums, target_types


__all__ = [
    "CorruptionBasedDataset",
    "ContrastiveDataset",
    "CleanDirtyEvaluationDataset",
    "PerColumnBinaryDataset",
    "MaskedCellModelingDataset",
]

