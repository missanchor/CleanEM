"""
Data processing utilities.
"""

from typing import Any, Dict, List, Optional

import torch


class TabularProcessor:
    """
    Converts raw table rows into tensor inputs.
    """

    def __init__(self, num_numeric_bins: int = 100, d_cell: int = 64):
        self.num_numeric_bins = num_numeric_bins
        self.d_cell = d_cell

    def process(self, row_data: List[Any], row_idx: int = 0) -> Dict[str, torch.Tensor]:
        num_cols = len(row_data)
        cell_embeddings = [self._embed_cell(cell_value) for cell_value in row_data]
        cell_embeddings = torch.stack(cell_embeddings)
        row_indices = torch.full((num_cols,), row_idx, dtype=torch.long)
        col_indices = torch.arange(num_cols, dtype=torch.long)
        return {
            "cell_embeddings": cell_embeddings,
            "row_indices": row_indices,
            "col_indices": col_indices,
        }

    def _embed_cell(self, value: Any) -> torch.Tensor:
        if isinstance(value, (int, float)):
            normalized = (value % self.num_numeric_bins) / self.num_numeric_bins
            embedding = torch.full((self.d_cell,), normalized, dtype=torch.float32)
        elif isinstance(value, str):
            hash_val = hash(value) % self.num_numeric_bins / self.num_numeric_bins
            embedding = torch.full((self.d_cell,), hash_val, dtype=torch.float32)
        else:
            embedding = torch.zeros(self.d_cell, dtype=torch.float32)

        embedding = embedding + torch.randn(self.d_cell) * 0.01
        return embedding


class TextProcessor:
    """
    Converts schema/constraint text into token ids/attention masks.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_seq_len: int = 512,
        tokenizer_name_or_path: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.use_hf_tokenizer = tokenizer_name_or_path is not None

        if self.use_hf_tokenizer:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "The `transformers` package is required to use a pretrained tokenizer. "
                    "Install it via `pip install transformers`."
                ) from exc

            tokenizer_kwargs = dict(tokenizer_kwargs or {})
            tokenizer_kwargs.setdefault("trust_remote_code", trust_remote_code)

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                **tokenizer_kwargs,
            )
            self.vocab_size = getattr(self.tokenizer, "vocab_size", vocab_size)
        else:
            self.vocab_size = vocab_size

    def process(self, text: str) -> Dict[str, torch.Tensor]:
        if self.use_hf_tokenizer:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
                return_tensors="pt",
            )

            output: Dict[str, torch.Tensor] = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

            if "token_type_ids" in encoded:
                output["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
            if "position_ids" in encoded:
                output["position_ids"] = encoded["position_ids"].squeeze(0)

            return output

        tokens = text.lower().split()[: self.max_seq_len]
        input_ids = [hash(token) % self.vocab_size for token in tokens]

        seq_len = len(input_ids)
        if seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - seq_len
            input_ids = input_ids + [0] * padding_len
            attention_mask = [1] * seq_len + [0] * padding_len
        else:
            attention_mask = [1] * seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


__all__ = ["TabularProcessor", "TextProcessor"]


