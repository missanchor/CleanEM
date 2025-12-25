"""
Data processing utilities.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class TabularProcessor(nn.Module):
    """
    Converts raw table rows into tensor inputs.

    Each categorical column has its own learnable embedding table (no hash collision).
    Numeric columns use linear projection.
    """

    def __init__(
        self,
        num_numeric_bins: int = 1000,
        d_cell: int = 8,
        use_column_names: bool = True,
        vocab_size: int = 10000,  # kept for backward compat, not used for per-column embeddings
        numeric_embedding_dim: int = 16,
        unknown_token: str = "<UNK>",
    ):
        super().__init__()
        self.num_numeric_bins = num_numeric_bins
        self.d_cell = d_cell
        self.use_column_names = use_column_names
        self.vocab_size = vocab_size  # backward compat
        self.numeric_embedding_dim = numeric_embedding_dim
        self.unknown_token = unknown_token

        # Column name related
        self.column_name_to_id: Dict[str, int] = {}
        self.num_cols: int = 0

        if self.use_column_names:
            self.d_name = max(1, d_cell // 2)
        else:
            self.d_name = 0
        self.d_value = d_cell - self.d_name

        # Column name embedding (shared across all rows)
        self.column_embedding: Optional[nn.Embedding] = None

        # Per-column value embeddings for categorical columns
        # Will be initialized when column vocabularies are registered
        self.per_column_value_embeddings: nn.ModuleList = nn.ModuleList()
        self.per_column_value_to_id: List[Dict[str, int]] = []
        self.per_column_is_numeric: List[bool] = []
        self._columns_initialized: bool = False

        # Numeric embedding layers (shared across all numeric columns)
        self.numeric_embedding = nn.Linear(1, numeric_embedding_dim)
        self.numeric_projection = nn.Linear(numeric_embedding_dim, self.d_value)

        # Interaction / projection layers
        self.name_value_interaction: Optional[nn.Module] = None
        self.name_only_projection: Optional[nn.Module] = None
        self.value_only_projection: Optional[nn.Module] = None

        if self.d_name > 0 and self.d_value > 0:
            interaction_in_dim = self.d_name + self.d_value
            self.name_value_interaction = nn.Sequential(
                nn.Linear(interaction_in_dim, d_cell),
                nn.ReLU(),
                nn.Linear(d_cell, d_cell),
            )
        elif self.d_name > 0:
            self.name_only_projection = (
                nn.Identity() if self.d_name == d_cell else nn.Linear(self.d_name, d_cell)
            )

        if self.d_value > 0 and (self.d_name == 0 or self.name_value_interaction is None):
            self.value_only_projection = (
                nn.Identity() if self.d_value == d_cell else nn.Linear(self.d_value, d_cell)
            )

    def _infer_device(self) -> torch.device:
        return self.numeric_embedding.weight.device

    @property
    def device(self) -> torch.device:
        return self._infer_device()

    def register_column_vocabularies(
        self,
        column_names: List[str],
        column_value_to_id: List[Dict[str, int]],
        col_is_numeric: Optional[List[bool]] = None,
    ) -> None:
        """
        Register per-column vocabularies and create embedding tables.

        Must be called before processing data.

        Args:
            column_names: List of column names
            column_value_to_id: Per-column mapping from value -> id
            col_is_numeric: Per-column flag indicating if column is numeric
        """
        device = self._infer_device()
        num_cols = len(column_names)
        self.num_cols = num_cols

        # Register column names
        self.column_name_to_id = {name: idx for idx, name in enumerate(column_names)}
        if self.d_name > 0:
            self.column_embedding = nn.Embedding(num_cols, self.d_name).to(device)
            nn.init.xavier_uniform_(self.column_embedding.weight)

        # Store per-column info
        self.per_column_value_to_id = column_value_to_id
        self.per_column_is_numeric = col_is_numeric if col_is_numeric else [False] * num_cols

        # Create per-column embedding tables
        self.per_column_value_embeddings = nn.ModuleList()
        for col_idx in range(num_cols):
            if self.per_column_is_numeric[col_idx]:
                # Numeric column: no embedding table needed, use linear projection
                # Add a placeholder to keep indices aligned
                self.per_column_value_embeddings.append(nn.Identity())
            else:
                # Categorical column: create embedding table
                vocab = column_value_to_id[col_idx] if col_idx < len(column_value_to_id) else {}
                vocab_size = max(1, len(vocab))
                embedding = nn.Embedding(vocab_size, self.d_value).to(device)
                nn.init.xavier_uniform_(embedding.weight)
                self.per_column_value_embeddings.append(embedding)

        self._columns_initialized = True
        print(f"✓ TabularProcessor: 注册了 {num_cols} 列的词汇表")
        for col_idx, name in enumerate(column_names):
            if self.per_column_is_numeric[col_idx]:
                print(f"    {name}: 数值列 (线性投影)")
            else:
                vocab_size = len(column_value_to_id[col_idx]) if col_idx < len(column_value_to_id) else 0
                print(f"    {name}: 分类列 (vocab_size={vocab_size})")

    def _get_or_register_column_ids(self, column_names: List[str]) -> torch.Tensor:
        """Get column IDs, auto-register if not initialized."""
        device = self._infer_device()

        if not self.column_name_to_id:
            # Auto-register columns (backward compatibility)
            for idx, name in enumerate(column_names):
                self.column_name_to_id[name] = idx
            self.num_cols = len(column_names)
            if self.d_name > 0:
                self.column_embedding = nn.Embedding(len(self.column_name_to_id), self.d_name).to(device)
                nn.init.xavier_uniform_(self.column_embedding.weight)
        else:
            unknown = [name for name in column_names if name not in self.column_name_to_id]
            if unknown:
                raise ValueError(f"检测到未知列名: {unknown}. 请提前注册所有列名。")

        ids = [self.column_name_to_id[name] for name in column_names]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _embed_value_for_column(
        self,
        value: Any,
        col_idx: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Embed a single cell value using the appropriate column embedding.

        Args:
            value: Cell value
            col_idx: Column index
            device: Target device

        Returns:
            Tensor of shape [d_value]
        """
        # Check if column is numeric
        is_numeric_col = (
            col_idx < len(self.per_column_is_numeric)
            and self.per_column_is_numeric[col_idx]
        )

        # Check if value itself is numeric
        is_numeric_value = isinstance(value, (int, float)) and not isinstance(value, bool)

        if is_numeric_col or is_numeric_value:
            # Numeric: use linear projection
            if is_numeric_value:
                numeric_value = float(value)
            else:
                # Try to parse string as number
                try:
                    numeric_value = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    numeric_value = 0.0

            normalized = (numeric_value % self.num_numeric_bins) / self.num_numeric_bins
            numeric_input = torch.tensor([[normalized]], dtype=torch.float32, device=device)
            numeric_hidden = self.numeric_embedding(numeric_input)
            numeric_output = self.numeric_projection(numeric_hidden)
            return numeric_output.squeeze(0)

        # Categorical: use per-column embedding table
        text_value = "" if value is None else str(value)

        if self._columns_initialized and col_idx < len(self.per_column_value_to_id):
            vocab = self.per_column_value_to_id[col_idx]
            value_id = vocab.get(text_value)
            if value_id is None:
                # Unknown value: use UNK token or fallback to 0
                value_id = vocab.get(self.unknown_token, 0)

            embedding_table = self.per_column_value_embeddings[col_idx]
            if isinstance(embedding_table, nn.Embedding):
                value_id_tensor = torch.tensor(value_id, dtype=torch.long, device=device)
                return embedding_table(value_id_tensor)

        # Fallback: return zeros (should not happen if properly initialized)
        return torch.zeros(self.d_value, dtype=torch.float32, device=device)

    def _embed_values_flat_per_column(
        self,
        values: List[Any],
        col_indices: List[int],
        device: torch.device
    ) -> torch.Tensor:
        """
        Vectorized embedding for a flat list of values with their column indices.

        Args:
            values: Flat list of values (length N)
            col_indices: Column index for each value (length N)
            device: Target device

        Returns:
            Tensor of shape [N, d_value]
        """
        n = len(values)
        if n == 0:
            return torch.empty((0, self.d_value), dtype=torch.float32, device=device)

        embeddings = []
        for i, (value, col_idx) in enumerate(zip(values, col_indices)):
            emb = self._embed_value_for_column(value, col_idx, device)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)

    def _embed_values_flat(self, values: List[Any], *, device: torch.device) -> torch.Tensor:
        """
        Vectorized embedding for a flat list of values.

        Note: This method assumes values are in row-major order (all cols of row 0, then row 1, etc.)
        """
        n = len(values)
        if n == 0:
            return torch.empty((0, self.d_value), dtype=torch.float32, device=device)

        if not self._columns_initialized or self.num_cols == 0:
            # Fallback: treat all as numeric or use zeros
            embeddings = []
            for v in values:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    normalized = (float(v) % self.num_numeric_bins) / self.num_numeric_bins
                    numeric_input = torch.tensor([[normalized]], dtype=torch.float32, device=device)
                    numeric_hidden = self.numeric_embedding(numeric_input)
                    numeric_output = self.numeric_projection(numeric_hidden)
                    embeddings.append(numeric_output.squeeze(0))
                else:
                    embeddings.append(torch.zeros(self.d_value, dtype=torch.float32, device=device))
            return torch.stack(embeddings, dim=0)

        # Compute column indices for each value
        col_indices = [i % self.num_cols for i in range(n)]
        return self._embed_values_flat_per_column(values, col_indices, device)

    def _embed_row_values(self, row_data: List[Any], *, device: torch.device) -> torch.Tensor:
        """
        Embed all values in a single row.

        Args:
            row_data: List of cell values
            device: Target device

        Returns:
            Tensor of shape [num_cols, d_value]
        """
        num_cols = len(row_data)
        if num_cols == 0:
            return torch.empty((0, self.d_value), dtype=torch.float32, device=device)

        embeddings = []
        for col_idx, value in enumerate(row_data):
            emb = self._embed_value_for_column(value, col_idx, device)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)

    def process(
        self,
        row_data: List[Any],
        row_idx: int = 0,
        column_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        num_cols = len(row_data)
        device = self._infer_device()

        # Use column names if provided and enabled
        name_embeddings = None
        if self.use_column_names and column_names is not None and self.d_name > 0:
            if len(column_names) != num_cols:
                raise ValueError(f"列名数量 ({len(column_names)}) 与数据列数 ({num_cols}) 不匹配")
            column_ids = self._get_or_register_column_ids(column_names)
            name_embeddings = self.column_embedding(column_ids)

        value_embeddings = None
        if self.d_value > 0:
            value_embeddings = self._embed_row_values(row_data, device=device)

        if (
            name_embeddings is not None
            and value_embeddings is not None
            and self.name_value_interaction is not None
        ):
            interaction_input = torch.cat([name_embeddings, value_embeddings], dim=-1)
            cell_embeddings = self.name_value_interaction(interaction_input)
        elif name_embeddings is not None:
            if self.name_only_projection is None:
                raise ValueError("缺少 name_only_projection，无法仅使用列名构建嵌入。")
            cell_embeddings = self.name_only_projection(name_embeddings)
        elif value_embeddings is not None:
            if self.value_only_projection is None:
                raise ValueError("缺少 value_only_projection，无法仅使用列值构建嵌入。")
            cell_embeddings = self.value_only_projection(value_embeddings)
        else:
            raise ValueError("d_cell must be positive to build embeddings.")

        row_indices = torch.full((num_cols,), row_idx, dtype=torch.long, device=device)
        col_indices = torch.arange(num_cols, dtype=torch.long, device=device)
        return {
            "cell_embeddings": cell_embeddings,
            "row_indices": row_indices,
            "col_indices": col_indices,
        }

    def process_batch(
        self,
        batch_rows: List[List[Any]],
        *,
        row_indices: Optional[List[int]] = None,
        column_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Batched version of `process`.

        Args:
            batch_rows: List of rows, each row is a list of cell values. Must be rectangular.
            row_indices: Optional list of row indices, length B. Defaults to 0s.
            column_names: Optional list of column names, length C.

        Returns:
            Dict with:
              - cell_embeddings: [B, C, d_cell]
              - row_indices: [B, C]
              - col_indices: [B, C]
        """
        device = self._infer_device()
        batch_size = len(batch_rows)
        if batch_size == 0:
            return {
                "cell_embeddings": torch.empty((0, 0, self.d_cell), dtype=torch.float32, device=device),
                "row_indices": torch.empty((0, 0), dtype=torch.long, device=device),
                "col_indices": torch.empty((0, 0), dtype=torch.long, device=device),
            }

        num_cols = len(batch_rows[0])
        for r in batch_rows:
            if len(r) != num_cols:
                raise ValueError("batch_rows must be rectangular (all rows have the same number of columns).")

        if row_indices is None:
            row_indices = [0] * batch_size
        if len(row_indices) != batch_size:
            raise ValueError("row_indices length must match batch size.")

        # Column name embeddings (shared across the batch)
        name_embeddings = None
        if self.use_column_names and column_names is not None and self.d_name > 0:
            if len(column_names) != num_cols:
                raise ValueError(f"列名数量 ({len(column_names)}) 与数据列数 ({num_cols}) 不匹配")
            column_ids = self._get_or_register_column_ids(column_names)
            name_embeddings = self.column_embedding(column_ids)  # [C, d_name]
            name_embeddings = name_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C, d_name]

        # Value embeddings (batched)
        value_embeddings = None
        if self.d_value > 0:
            flat_values: List[Any] = []
            for row in batch_rows:
                flat_values.extend(row)
            flat_embeds = self._embed_values_flat(flat_values, device=device)  # [B*C, d_value]
            value_embeddings = flat_embeds.view(batch_size, num_cols, self.d_value)  # [B, C, d_value]

        if (
            name_embeddings is not None
            and value_embeddings is not None
            and self.name_value_interaction is not None
        ):
            interaction_input = torch.cat([name_embeddings, value_embeddings], dim=-1)  # [B, C, d_name+d_value]
            cell_embeddings = self.name_value_interaction(interaction_input)  # [B, C, d_cell]
        elif name_embeddings is not None:
            if self.name_only_projection is None:
                raise ValueError("缺少 name_only_projection，无法仅使用列名构建嵌入。")
            cell_embeddings = self.name_only_projection(name_embeddings)
        elif value_embeddings is not None:
            if self.value_only_projection is None:
                raise ValueError("缺少 value_only_projection，无法仅使用列值构建嵌入。")
            cell_embeddings = self.value_only_projection(value_embeddings)
        else:
            raise ValueError("d_cell must be positive to build embeddings.")

        row_idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=device).unsqueeze(1).expand(-1, num_cols)
        col_idx_tensor = torch.arange(num_cols, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        return {
            "cell_embeddings": cell_embeddings,
            "row_indices": row_idx_tensor,
            "col_indices": col_idx_tensor,
        }


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

            output: Dict[str, Any] = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "raw_text": text,
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

    def process_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """批量处理文本，显著提高效率"""
        if self.use_hf_tokenizer:
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            # 返回所有张量，保留 batch 维度
            output: Dict[str, torch.Tensor] = {
                k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)
            }
            return output

        # 非 HF tokenizer 回退到循环处理并 stack
        results = [self.process(t) for t in texts]
        return {
            k: torch.stack([r[k] for r in results])
            for k in ["input_ids", "attention_mask"]
            if k in results[0]
        }


__all__ = ["TabularProcessor", "TextProcessor"]
