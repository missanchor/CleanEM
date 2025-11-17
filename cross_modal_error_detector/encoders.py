"""
Modal encoder implementations.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseEncoder


class StructureAwareTransformer(BaseEncoder):
    """
    Transformer encoder that leverages row/column structure via custom masks.
    """

    def __init__(
        self,
        d_cell: int,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_projection = nn.Linear(d_cell, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.d_model = d_model

    def _create_structure_mask(
        self,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_cols = row_indices.shape
        row_i = row_indices.unsqueeze(2)
        row_j = row_indices.unsqueeze(1)
        col_i = col_indices.unsqueeze(2)
        col_j = col_indices.unsqueeze(1)

        same_row = row_i == row_j
        same_col = col_i == col_j
        can_attend = same_row | same_col

        mask = torch.zeros_like(can_attend, dtype=torch.float)
        mask = mask.masked_fill(~can_attend, float("-inf"))
        return mask

    def forward(self, tabular_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        cell_embeddings = tabular_inputs["cell_embeddings"]
        row_indices = tabular_inputs["row_indices"]
        col_indices = tabular_inputs["col_indices"]

        x = self.cell_projection(cell_embeddings)
        attn_mask = self._create_structure_mask(row_indices, col_indices)

        # Simplified: assume shared structure within the batch.
        attn_mask_2d = attn_mask[0]

        H_table = self.transformer_encoder(
            x,
            mask=attn_mask_2d,
        )
        return H_table


class PretrainedTextEncoder(BaseEncoder):
    """
    Lightweight Transformer text encoder that can optionally wrap pretrained
    HuggingFace/ModelScope models stored on disk.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        *,
        model_name_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        pretrained_model_kwargs: Optional[Dict[str, Any]] = None,
        output_proj_dim: Optional[int] = None,
        freeze_base_model: bool = True,
        adapter_hidden_dim: Optional[int] = None,
        adapter_dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of the synthetic vocabulary (ignored when loading a
                pretrained model).
            d_model: Hidden size of the lightweight encoder or projection
                dimension when wrapping a pretrained model.
            nhead: Number of attention heads for the lightweight encoder.
            num_layers: Number of transformer layers for the lightweight encoder.
            max_seq_len: Maximum sequence length supported by the lightweight encoder.
            dropout: Dropout probability applied before the transformer or on top
                of the pretrained model outputs.
            model_name_or_path: Local path or identifier of a pretrained model.
                When provided, the encoder will load the model via
                `transformers.AutoModel.from_pretrained`.
            trust_remote_code: Whether to allow custom model code when loading
                from `transformers`. Defaults to True to support ModelScope
                checkpoints.
            pretrained_model_kwargs: Additional keyword arguments forwarded to
                `AutoModel.from_pretrained`.
            output_proj_dim: Optional dimension to project the pretrained model
                hidden states to. If None, the original hidden size is used.
            freeze_base_model: Whether to freeze the pretrained encoder weights.
                Set to True to keep the backbone fixed and only train the adapter.
            adapter_hidden_dim: Size of the adapter bottleneck. Defaults to the
                output projection dimension when not specified.
            adapter_dropout: Dropout probability applied inside the adapter.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.adapter: Optional[nn.Module] = None
        self.use_transformers_model = model_name_or_path is not None
        self.freeze_base_model = freeze_base_model
        self.adapter_hidden_dim = adapter_hidden_dim
        self.adapter_dropout = adapter_dropout

        # Placeholders to keep attribute access consistent.
        self.model: Optional[nn.Module] = None
        self.token_embedding: Optional[nn.Embedding] = None
        self.position_embedding: Optional[nn.Embedding] = None
        self.transformer: Optional[nn.TransformerEncoder] = None

        if self.use_transformers_model:
            load_kwargs: Dict[str, Any]
            if pretrained_model_kwargs is None:
                load_kwargs = {}
            else:
                load_kwargs = dict(pretrained_model_kwargs)
            load_kwargs.setdefault("trust_remote_code", trust_remote_code)

            try:
                from transformers import AutoModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "The `transformers` package is required to load pretrained "
                    "text encoders. Install it via `pip install transformers`."
                ) from exc

            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                **load_kwargs,
            )

            if self.freeze_base_model:
                for param in self.model.parameters():
                    param.requires_grad = False

            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.model.config, "d_model", None)
            if hidden_size is None:
                raise ValueError(
                    "Unable to determine the hidden size of the pretrained model. "
                    "Please ensure the config exposes `hidden_size` or `d_model`."
                )

            adapter_out_dim = output_proj_dim or hidden_size
            self.adapter = self._build_adapter(hidden_size, adapter_out_dim)
            self.d_model = adapter_out_dim
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.d_model = d_model

    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_transformers_model:
            if self.model is None:
                raise RuntimeError("Pretrained model not initialized.")

            model_kwargs: Dict[str, torch.Tensor] = {}
            for key in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "position_ids",
                "inputs_embeds",
            ):
                if key in text_inputs:
                    model_kwargs[key] = text_inputs[key]

            if self.freeze_base_model:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**model_kwargs)
            else:
                outputs = self.model(**model_kwargs)

            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs[0]

            hidden_states = self.dropout(hidden_states)
            if self.adapter is not None:
                hidden_states = self.adapter(hidden_states)
            return hidden_states

        if self.token_embedding is None or self.position_embedding is None or self.transformer is None:
            raise RuntimeError("Lightweight encoder components not initialized.")

        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        batch_size, seq_len = input_ids.shape

        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        embeddings = self.dropout(embeddings)

        src_key_padding_mask = attention_mask == 0

        H_text = self.transformer(
            embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )
        return H_text

    def _build_adapter(self, input_dim: int, output_dim: int) -> nn.Module:
        bottleneck_dim = self.adapter_hidden_dim or output_dim
        layers = [
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(self.adapter_dropout),
            nn.Linear(bottleneck_dim, output_dim),
        ]
        return nn.Sequential(*layers)


class SimpleMLPEncoder(BaseEncoder):
    """
    Ablation-friendly MLP encoder that ignores table structure.
    """

    def __init__(self, d_cell: int, d_model: int, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = d_cell
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = d_model
        layers.append(nn.Linear(in_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        self.d_model = d_model

    def forward(self, tabular_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        cell_embeddings = tabular_inputs["cell_embeddings"]
        return self.mlp(cell_embeddings)


__all__ = [
    "StructureAwareTransformer",
    "PretrainedTextEncoder",
    "SimpleMLPEncoder",
]


