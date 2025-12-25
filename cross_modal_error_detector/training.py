"""
Collate utilities and training steps for different strategies.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CrossModalErrorDetector
from .utils.device import resolve_runtime_device


def build_tabular_inputs_from_rows(
    row_samples: Sequence[Any],
    tabular_processor,
    default_column_names: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build a batched tabular_inputs dict from raw row samples.

    This should be called in the main process / training step so that:
    - autograd graphs stay local to the training process (no multiprocessing serialization)
    - tabular_processor can live on GPU and be trained
    """
    if tabular_processor is None:
        raise ValueError("tabular_processor must be provided.")

    batch_rows: List[List[Any]] = []
    batch_row_indices: List[int] = []
    resolved_column_names: Optional[Sequence[str]] = None

    for sample in row_samples:
        if isinstance(sample, dict):
            row_data = sample.get("row_data")
            row_idx = sample.get("row_idx", 0)
            column_names = sample.get("column_names", default_column_names)
        elif isinstance(sample, (list, tuple)):
            if not sample:
                raise ValueError("Row sample tuple cannot be empty.")
            row_data = sample[0]
            row_idx = sample[1] if len(sample) > 1 else 0
            column_names = sample[2] if len(sample) > 2 else default_column_names
        else:
            raise TypeError(f"Unsupported row sample type: {type(sample)}")

        if row_data is None:
            raise ValueError("row_data is required to build tabular inputs.")

        if resolved_column_names is None:
            resolved_column_names = column_names
        else:
            # Ensure consistent column_names within a batch
            if column_names is not None and list(column_names) != list(resolved_column_names):
                raise ValueError("Inconsistent column_names within the same batch.")

        batch_rows.append(list(row_data))
        batch_row_indices.append(int(row_idx))

    return tabular_processor.process_batch(
        batch_rows,
        row_indices=batch_row_indices,
        column_names=list(resolved_column_names) if resolved_column_names is not None else None,
    )


def collate_fn_corruption(batch):
    """
    Collate function for corruption-based training.
    """

    row_samples, text_inputs_list, labels_list = zip(*batch)
    row_samples = list(row_samples)

    input_ids_list = []
    attention_mask_list = []
    cached_embeddings_list = []
    has_cached = False

    for x in text_inputs_list:
        if "cached_embedding" in x:
            cached_embeddings_list.append(x["cached_embedding"])
            has_cached = True
        else:
            input_ids_list.append(x["input_ids"])
            attention_mask_list.append(x["attention_mask"])

    if has_cached:
        text_inputs = {
            "cached_embedding": torch.stack(cached_embeddings_list)
        }
    else:
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    labels = torch.stack(labels_list)
    return row_samples, text_inputs, labels


def collate_fn_contrastive(batch):
    """
    Collate function for contrastive training.
    """

    row_samples, text_inputs_list = zip(*batch)
    row_samples = list(row_samples)

    input_ids_list = []
    attention_mask_list = []
    cached_embeddings_list = []
    has_cached = False

    for x in text_inputs_list:
        if "cached_embedding" in x:
            cached_embeddings_list.append(x["cached_embedding"])
            has_cached = True
        else:
            input_ids_list.append(x["input_ids"])
            attention_mask_list.append(x["attention_mask"])

    if has_cached:
        text_inputs = {
            "cached_embedding": torch.stack(cached_embeddings_list)
        }
    else:
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return row_samples, text_inputs


def collate_fn_contrastive_cell_level(batch):
    """
    Collate function for two-stage contrastive cell-level binary classification training.
    Expects batch items with 4 values: (tabular_inputs, text_inputs, labels, cell_idx)
    """

    row_samples, text_inputs_list, labels_list, cell_indices = zip(*batch)
    row_samples = list(row_samples)

    input_ids_list = []
    attention_mask_list = []
    cached_embeddings_list = []
    has_cached = False

    for x in text_inputs_list:
        if "cached_embedding" in x:
            cached_embeddings_list.append(x["cached_embedding"])
            has_cached = True
        else:
            input_ids_list.append(x["input_ids"])
            attention_mask_list.append(x["attention_mask"])

    if has_cached:
        text_inputs = {
            "cached_embedding": torch.stack(cached_embeddings_list)
        }
    else:
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    labels = torch.stack(labels_list)
    cell_indices = torch.stack(cell_indices)
    return row_samples, text_inputs, labels, cell_indices


def collate_fn_mcm(batch):
    """
    Collate function for Masked Cell Modeling (MCM).
    Expects batch items:
      (row_payload, text_inputs, mask_indices, target_ids, target_nums, target_types)
    
    Now supports per-column text inputs stored in row_payload["per_column_text_inputs"]
    """
    (
        row_samples,
        text_inputs_list,
        mask_indices_list,
        target_ids_list,
        target_nums_list,
        target_types_list,
    ) = zip(*batch)

    row_samples = list(row_samples)

    # Extract per-column text inputs from row_payloads
    per_column_text_inputs_dict: Dict[int, List[Dict[str, Any]]] = {}
    for sample in row_samples:
        if isinstance(sample, dict) and "per_column_text_inputs" in sample:
            per_col_inputs = sample["per_column_text_inputs"]
            for col_idx, col_text_input in per_col_inputs.items():
                if col_idx not in per_column_text_inputs_dict:
                    per_column_text_inputs_dict[col_idx] = []
                per_column_text_inputs_dict[col_idx].append(col_text_input)

    # Legacy: also support global text inputs for backward compatibility
    input_ids_list = []
    attention_mask_list = []
    cached_embeddings_list = []
    has_cached = False

    for x in text_inputs_list:
        if "cached_embedding" in x:
            cached_embeddings_list.append(x["cached_embedding"])
            has_cached = True
        else:
            input_ids_list.append(x["input_ids"])
            attention_mask_list.append(x["attention_mask"])

    if has_cached:
        text_inputs = {
            "cached_embedding": torch.stack(cached_embeddings_list),
        }
    else:
        text_inputs = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
        }

    # Collate per-column text inputs
    per_column_text_inputs_collated: Dict[int, Dict[str, torch.Tensor]] = {}
    for col_idx, col_text_list in per_column_text_inputs_dict.items():
        col_cached_list = []
        col_input_ids_list = []
        col_attention_mask_list = []
        col_has_cached = False
        
        for x in col_text_list:
            if "cached_embedding" in x:
                col_cached_list.append(x["cached_embedding"])
                col_has_cached = True
            else:
                col_input_ids_list.append(x["input_ids"])
                col_attention_mask_list.append(x["attention_mask"])
        
        if col_has_cached:
            per_column_text_inputs_collated[col_idx] = {
                "cached_embedding": torch.stack(col_cached_list),
            }
        else:
            per_column_text_inputs_collated[col_idx] = {
                "input_ids": torch.stack(col_input_ids_list),
                "attention_mask": torch.stack(col_attention_mask_list),
            }
    
    # Store per-column text inputs in text_inputs dict
    text_inputs["per_column"] = per_column_text_inputs_collated

    mask_indices = torch.stack(mask_indices_list)  # [B, K]
    target_ids = torch.stack(target_ids_list)      # [B, K]
    target_nums = torch.stack(target_nums_list)    # [B, K]
    target_types = torch.stack(target_types_list)  # [B, K]

    return row_samples, text_inputs, mask_indices, target_ids, target_nums, target_types


def _resolve_device(default_device: str, device_map: Optional[Dict[str, str]], key: str) -> str:
    if device_map:
        if key in device_map:
            return resolve_runtime_device(device_map[key], fallback_device=default_device)
        if "default" in device_map:
            return resolve_runtime_device(device_map["default"], fallback_device=default_device)
    return default_device


def _move_to_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    target = torch.device(device)
    return {k: (v if v.device == target else v.to(target)) for k, v in inputs.items()}


def train_step_corruption(
    model: CrossModalErrorDetector,
    batch: Tuple,
    optimizer: torch.optim.Optimizer,
    tabular_processor,
    device: Optional[str] = None,
    *,
    device_map: Optional[Dict[str, str]] = None,
    column_names: Optional[Sequence[str]] = None,
) -> float:
    """
    Single training step for corruption-based strategy.
    """

    model.train()
    row_samples, text_inputs, labels = batch

    runtime_device = resolve_runtime_device(device)
    tabular_device = _resolve_device(runtime_device, device_map, "tabular_encoder")
    text_device = _resolve_device(runtime_device, device_map, "text_encoder")
    label_device = _resolve_device(runtime_device, device_map, "detection_head")

    if tabular_processor is None:
        raise ValueError("tabular_processor is required for train_step_corruption.")
    tabular_processor = tabular_processor.to(tabular_device)
    tabular_inputs = build_tabular_inputs_from_rows(row_samples, tabular_processor, column_names)
    text_inputs = _move_to_device(text_inputs, text_device)
    labels = labels.to(label_device)

    logits = model(tabular_inputs, text_inputs).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_embedding_similarity(
    tabular_encoder,
    text_encoder,
    tabular_inputs: Dict[str, torch.Tensor],
    text_inputs: Dict[str, torch.Tensor],
    *,
    device: Optional[str] = None,
    device_map: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """
    Compute cosine similarity between tabular and text encoder embeddings.
    This is used in Stage A (pretraining) without fusion module.
    """
    device = device or "cpu"

    # Determine devices for each component from device_map or use default
    default_device = device
    text_encoder_device = device
    if device_map:
        default_device = device_map.get("default", default_device)
        text_encoder_device = device_map.get("text_encoder", text_encoder_device)

    # Ensure encoders are on correct devices
    tabular_encoder = tabular_encoder.to(default_device)
    if not isinstance(text_inputs.get("cached_embedding"), torch.Tensor):
        text_encoder = text_encoder.to(text_encoder_device)

    # Add batch dimension if not present (for single-row evaluation)
    # Process tabular inputs
    if "cell_embeddings" in tabular_inputs and tabular_inputs["cell_embeddings"].dim() == 2:
        # Single row case - add batch dimension
        tabular_inputs = {
            k: v.unsqueeze(0)
            for k, v in tabular_inputs.items()
        }

    # Move tabular inputs to the correct device (tabular encoder device)
    tabular_inputs = {
        k: v.to(default_device)
        for k, v in tabular_inputs.items()
    }

    # Get tabular embeddings on its device
    H_table = tabular_encoder(tabular_inputs)

    # Handle text inputs
    if "cached_embedding" in text_inputs:
        H_text = text_inputs["cached_embedding"]
        # Determine if this is a pooled embedding (last token) or full sequence
        # If cached_embedding is 2D (already pooled), or if it's 3D but we want to use last token
        if H_text.dim() == 2:
            # Already pooled (e.g., last token from _precompute_text_embeddings with use_text_output_last_token_embedding=True)
            # Shape: [B, d_model] - add sequence dimension for consistency with downstream code
            H_text = H_text.unsqueeze(1)  # [B, 1, d_model]
        elif H_text.dim() == 3:
            # Full sequence [B, seq_len, d_model] - extract last token
            H_text = H_text[:, -1, :]  # [B, d_model]
            H_text = H_text.unsqueeze(1)  # [B, 1, d_model] for consistency
        H_text = H_text.to(default_device)
    else:
        # Add batch dimension if needed
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            if k in text_inputs and text_inputs[k].dim() == 1:
                text_inputs[k] = text_inputs[k].unsqueeze(0)

        # Move text inputs to text encoder device
        text_inputs = {
            k: v.to(text_encoder_device)
            for k, v in text_inputs.items()
        }

        # Get text embeddings on text encoder device
        H_text = text_encoder(text_inputs)

    # Pool text embeddings to get one representation per row
    if "attention_mask" in text_inputs:
        # Mean pooling with attention mask
        attention_mask = text_inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(H_text.size()).float()
        sum_embeddings = torch.sum(H_text * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        H_text_pooled = sum_embeddings / sum_mask  # [batch_size, d_model]
    else:
        # Simple mean pooling
        H_text_pooled = torch.mean(H_text, dim=1)  # [batch_size, d_model]

    # Move to default device after pooling
    H_text_pooled = H_text_pooled.to(default_device)

    # Expand text pooled embeddings to match tabular structure
    H_table = H_table.to(default_device)
    seq_len = H_table.size(1)
    H_text_expanded = H_text_pooled.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, d_model]

    # L2 normalize per cell
    H_table_norm = F.normalize(H_table, p=2, dim=2)  # [batch_size, seq_len, d_model]
    H_text_norm = F.normalize(H_text_expanded, p=2, dim=2)  # [batch_size, seq_len, d_model]

    # Cosine similarity per cell
    cell_similarities = torch.sum(H_table_norm * H_text_norm, dim=2)  # [batch_size, seq_len]

    # Average across cells in each row to get a single similarity score per row
    similarity = torch.mean(cell_similarities, dim=1)  # [batch_size]

    return similarity


def train_step_contrastive_pretrain(
    tabular_encoder,
    text_encoder,
    batch: Tuple,
    optimizer: torch.optim.Optimizer,
    tabular_processor,
    device: Optional[str] = None,
    temperature: float = 0.07,
    *,
    device_map: Optional[Dict[str, str]] = None,
    column_names: Optional[Sequence[str]] = None,
) -> float:
    """
    Single training step for Stage A pretraining using direct embedding similarity.
    Uses InfoNCE loss based on cosine similarity (no fusion module).
    Optimized to use matrix operations for efficiency.
    """
    device = device or "cpu"

    # Determine devices for each component from device_map
    default_device = device
    text_encoder_device = device
    if device_map:
        default_device = device_map.get("default", default_device)
        text_encoder_device = device_map.get("text_encoder", text_encoder_device)

    tabular_encoder.train()
    text_encoder.eval()

    row_samples, text_inputs = batch

    # Move encoders to their respective devices
    tabular_encoder = tabular_encoder.to(default_device)
    if "cached_embedding" not in text_inputs:
        text_encoder = text_encoder.to(text_encoder_device)

    if tabular_processor is None:
        raise ValueError("tabular_processor is required for train_step_contrastive_pretrain.")
    tabular_processor = tabular_processor.to(default_device)
    tabular_inputs = build_tabular_inputs_from_rows(row_samples, tabular_processor, column_names)
    batch_size = tabular_inputs["cell_embeddings"].shape[0]

    # Move inputs to their respective devices
    tabular_inputs = {k: v.to(default_device) for k, v in tabular_inputs.items()}
    if "cached_embedding" in text_inputs:
        text_inputs = {"cached_embedding": text_inputs["cached_embedding"].to(default_device)}
    else:
        text_inputs = {k: v.to(text_encoder_device) for k, v in text_inputs.items()}

    # Compute embeddings for all items in the batch at once
    H_table_all = tabular_encoder(tabular_inputs)  # [batch_size, seq_len, d_model]

    # Text embeddings - need to pool from token-level to sequence-level
    if "cached_embedding" in text_inputs:
        H_text_all = text_inputs["cached_embedding"]  # [batch_size, seq_len, d_model] or [batch_size, d_model]
        # If cached_embedding is 2D (already pooled), use as is; otherwise mean pool
        if H_text_all.dim() == 2:
            H_text_pooled = H_text_all  # Already pooled
        else:
            # Full sequence, use mean pooling
            H_text_pooled = torch.mean(H_text_all, dim=1)  # [batch_size, d_model]
    else:
        H_text_all = text_encoder(text_inputs)  # [batch_size, text_seq_len, d_model]
        # Pool text embeddings to get one representation per row
        # Use attention mask if available, otherwise use mean pooling
        if "attention_mask" in text_inputs:
            # Mean pooling with attention mask
            attention_mask = text_inputs["attention_mask"]  # [batch_size, text_seq_len]
            mask_expanded = attention_mask.unsqueeze(-1).expand(H_text_all.size()).float()
            sum_embeddings = torch.sum(H_text_all * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            H_text_pooled = sum_embeddings / sum_mask  # [batch_size, d_model]
        else:
            H_text_pooled = torch.mean(H_text_all, dim=1)  # [batch_size, d_model]

    # Move to default device after pooling
    H_text_pooled = H_text_pooled.to(default_device)

    # Keep cell-level embeddings for cell-level error detection
    # H_table_all: [batch_size, seq_len, d_model]
    seq_len = H_table_all.size(1)
    d_model = H_table_all.size(2)

    # L2 normalize both embeddings for cosine similarity
    H_text_norm = F.normalize(H_text_pooled, p=2, dim=1)  # [batch_size, d_model]
    H_table_norm = F.normalize(H_table_all, p=2, dim=2)   # [batch_size, seq_len, d_model]

    # Flatten cell embeddings: each cell compares against all row texts
    H_table_flat = H_table_norm.view(-1, d_model)  # [batch_size * seq_len, d_model]

    # Compute similarity: each cell vs all text representations
    # cell_to_text_sim[i*seq_len + j, k] = similarity between cell(i,j) and text_k
    cell_to_text_sim = torch.mm(H_table_flat, H_text_norm.T) / temperature
    # [batch_size * seq_len, batch_size]

    # Labels: cell (i, j) should match text i
    # [0,0,...,0, 1,1,...,1, ..., batch_size-1,...] with seq_len repetitions each
    labels = torch.arange(batch_size, device=cell_to_text_sim.device)
    labels = labels.unsqueeze(1).expand(-1, seq_len).reshape(-1)
    # [batch_size * seq_len]

    # InfoNCE loss: for each cell, its row's text should have highest similarity
    loss = F.cross_entropy(cell_to_text_sim, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_mcm(
    tabular_encoder,
    text_encoder,
    fusion_module,
    recon_heads,  # Changed from recon_head to recon_heads (ModuleList)
    batch: Tuple,
    optimizer: torch.optim.Optimizer,
    tabular_processor,
    mask_embedding: torch.Tensor,
    device: Optional[str] = None,
    *,
    device_map: Optional[Dict[str, str]] = None,
    column_names: Optional[Sequence[str]] = None,
    w_cat: float = 1.0,
    w_num: float = 1.0,
) -> Dict[str, float]:
    """
    Single training step for Masked Cell Modeling (MCM).

    - Mask selected cells in tabular_processor embeddings using a learned mask vector.
    - Encode -> fuse with text -> reconstruct masked cells using per-column heads.
    - Loss: categorical CE + numeric MSE (masked positions only).
    
    Args:
        recon_heads: nn.ModuleList of per-column reconstruction heads
    """
    # 使用统一的runtime_device，确保整个流程设备一致性
    runtime_device = resolve_runtime_device(device)
    # 所有组件都使用统一的runtime_device
    tabular_device = fusion_device = head_device = text_device = runtime_device

    tabular_encoder.train()
    fusion_module.train()
    # Set all reconstruction heads to train mode
    if isinstance(recon_heads, torch.nn.ModuleList):
        for recon_head in recon_heads:
            recon_head.train()
    else:
        recon_heads.train()
    # text_encoder should be frozen/cached; keep in eval to avoid dropout.
    text_encoder.eval()

    (
        row_samples,
        text_inputs,
        mask_indices,
        target_ids,
        target_nums,
        target_types,
    ) = batch

    if tabular_processor is None:
        raise ValueError("tabular_processor is required for train_step_mcm.")

    tabular_processor = tabular_processor.to(tabular_device)
    tabular_inputs = build_tabular_inputs_from_rows(row_samples, tabular_processor, column_names)
    tabular_inputs = {k: v.to(tabular_device) for k, v in tabular_inputs.items()}

    mask_indices = mask_indices.to(tabular_device)
    target_ids = target_ids.to(head_device)
    target_nums = target_nums.to(head_device)
    target_types = target_types.to(head_device)

    # Prepare text embeddings - use per-column text inputs if available
    # For each masked position, use the corresponding column's text embedding
    per_column_text_embeddings: Dict[int, torch.Tensor] = {}
    
    if "per_column" in text_inputs:
        # Use per-column text inputs
        per_col_inputs = text_inputs["per_column"]
        text_encoder = text_encoder.to(text_device)
        
        for col_idx, col_text_input in per_col_inputs.items():
            if "cached_embedding" in col_text_input:
                H_text_col = col_text_input["cached_embedding"]
                # Check if already pooled (2D) or full sequence (3D)
                if H_text_col.dim() == 2:
                    # Already pooled (e.g., last token from _precompute_text_embeddings with use_text_output_last_token_embedding=True)
                    # Shape: [B, d_model] - use as is
                    pass
                elif H_text_col.dim() == 3:
                    # Full sequence [B, seq_len, d_model] - extract last token
                    H_text_col = H_text_col[:, -1, :]  # [B, d_model]
                H_text_col = H_text_col.to(fusion_device)
            else:
                col_text_input_device = _move_to_device(col_text_input, text_device)
                H_text_col = text_encoder(col_text_input_device).to(fusion_device)
            
            # Extract last token embedding
            # The last token embedding represents LLM's response to the instruction
            # This is the "output" of the instruction, capturing LLM's understanding
            # H_text_col shape: [B, seq_len, d_model]
            if H_text_col.dim() == 3:
                # Use last token embedding - this is the instruction output
                # It represents LLM's understanding of the instruction for this column
                H_text_col = H_text_col[:, -1, :]  # [B, d_model]
            elif H_text_col.dim() == 2:
                # Already pooled or single token
                # If it's [B, d_model], use as is (already the instruction output)
                pass
            
            per_column_text_embeddings[col_idx] = H_text_col
    
    # Fallback to global text embedding if per-column not available
    if not per_column_text_embeddings:
        if "cached_embedding" in text_inputs:
            H_text = text_inputs["cached_embedding"]
            # Check if already pooled (2D) or full sequence (3D)
            if H_text.dim() == 2:
                # Already pooled (e.g., last token from _precompute_text_embeddings with use_text_output_last_token_embedding=True)
                # Shape: [B, d_model] - use as is
                pass
            elif H_text.dim() == 3:
                # Full sequence [B, seq_len, d_model] - extract last token
                H_text = H_text[:, -1, :]  # [B, d_model]
            H_text = H_text.to(fusion_device)
        else:
            text_encoder = text_encoder.to(text_device)
            text_inputs_device = _move_to_device(text_inputs, text_device)
            H_text = text_encoder(text_inputs_device).to(fusion_device)
            # Extract last token embedding
            if H_text.dim() == 3:
                H_text = H_text[:, -1, :]  # [B, d_model]

    # Mask tabular embeddings (operate on tabular_device)
    cell_embeddings = tabular_inputs["cell_embeddings"]
    if cell_embeddings.dim() != 3:
        raise ValueError(f"Expected cell_embeddings as [B,C,d_cell], got {cell_embeddings.shape}")

    B, _, d_cell = cell_embeddings.shape
    masked = cell_embeddings.clone()

    if mask_indices.numel() > 0:
        if mask_embedding.dim() != 1 or mask_embedding.numel() != d_cell:
            raise ValueError(
                f"mask_embedding must have shape [d_cell]={d_cell}, got {tuple(mask_embedding.shape)}"
            )
        mask_vec = mask_embedding.to(tabular_device)
        batch_idx = torch.arange(B, device=tabular_device).unsqueeze(1).expand_as(mask_indices)
        masked[batch_idx, mask_indices] = mask_vec

    tabular_inputs["cell_embeddings"] = masked

    # Forward
    tabular_encoder = tabular_encoder.to(tabular_device)
    H_table = tabular_encoder(tabular_inputs).to(fusion_device)  # [B, C, d_model]
    fusion_module = fusion_module.to(fusion_device)
    
    # Use per-column text embeddings if available
    B, C, d_model = H_table.shape
    H_text_expanded = None
    
    if per_column_text_embeddings:
        # For each column, use its corresponding text embedding
        # Create H_text with shape [B, C, d_model] where each column uses its own text embedding
        H_text_expanded = torch.zeros((B, C, d_model), device=fusion_device, dtype=H_table.dtype)

        # Fill in per-column text embeddings
        for col_idx, H_text_col in per_column_text_embeddings.items():
            if col_idx >= 0 and col_idx < C:
                # H_text_col shape should be [B, d_model]
                if H_text_col.dim() == 2 and H_text_col.shape[0] == B:
                    H_text_expanded[:, col_idx, :] = H_text_col
                elif H_text_col.dim() == 1:
                    # Single embedding, expand to batch
                    H_text_expanded[:, col_idx, :] = H_text_col.unsqueeze(0).expand(B, -1)

        # For columns without per-column embeddings, use the first available or mean
        for c in range(C):
            if H_text_expanded[:, c, :].sum() == 0:  # Not set yet
                if per_column_text_embeddings:
                    # Use first available column's embedding
                    first_col_idx = next(iter(per_column_text_embeddings.keys()))
                    first_col_emb = per_column_text_embeddings[first_col_idx]
                    if first_col_emb.dim() == 2:
                        H_text_expanded[:, c, :] = first_col_emb
                    else:
                        H_text_expanded[:, c, :] = first_col_emb.unsqueeze(0).expand(B, -1)
    else:
        # Fallback to global text embedding
        if "cached_embedding" in text_inputs:
            H_text = text_inputs["cached_embedding"]
            # Check if already pooled (2D) or full sequence (3D)
            if H_text.dim() == 2:
                # Already pooled (e.g., last token from _precompute_text_embeddings with use_text_output_last_token_embedding=True)
                # Shape: [B, d_model] - use as is
                pass
            elif H_text.dim() == 3:
                # Full sequence [B, seq_len, d_model] - extract last token
                H_text = H_text[:, -1, :]  # [B, d_model]
            H_text = H_text.to(fusion_device)
        else:
            text_encoder = text_encoder.to(text_device)
            text_inputs_device = _move_to_device(text_inputs, text_device)
            H_text = text_encoder(text_inputs_device).to(fusion_device)
            # Extract last token embedding
            if H_text.dim() == 3:
                H_text = H_text[:, -1, :]  # [B, d_model]
        
        # Expand H_text to match column dimension: [B, d_model] -> [B, C, d_model]
        if H_text.dim() == 2:
            H_text_expanded = H_text.unsqueeze(1).expand(-1, C, -1)  # [B, C, d_model]
        else:
            H_text_expanded = H_text
    
    # Apply fusion: check if using per-column fusion modules
    if isinstance(fusion_module, nn.ModuleList):
        # Per-column fusion: apply fusion per column, only for masked positions
        # Initialize H_fuse with zeros, will be filled for masked positions
        B, C, d_model = H_table.shape
        H_fuse = torch.zeros((B, C, d_model), device=fusion_device, dtype=H_table.dtype)

        # Process each masked position with its corresponding column fusion module
        for b in range(B):
            for k in range(mask_indices.shape[1]):
                col_idx = int(mask_indices[b, k].item())
                if col_idx >= 0 and col_idx < C:
                    # Extract column embeddings
                    col_table_embed = H_table[b:b+1, col_idx:col_idx+1, :]  # [1, 1, d_model]

                    # Get corresponding text embedding for this column
                    if per_column_text_embeddings and col_idx in per_column_text_embeddings:
                        col_text_emb = per_column_text_embeddings[col_idx][b:b+1]  # [1, d_model]
                    else:
                        col_text_emb = H_text_expanded[b:b+1, col_idx:col_idx+1, :]  # [1, 1, d_model]
                        if col_text_emb.dim() == 3:
                            col_text_emb = col_text_emb.squeeze(1)  # [1, d_model]

                    # Apply per-column fusion
                    col_fused = fusion_module[col_idx](col_table_embed, col_text_emb.unsqueeze(1))  # [1, 1, d_model]
                    H_fuse[b, col_idx:col_idx+1, :] = col_fused
    else:
        # Shared fusion module: compute for all columns
        H_fuse = fusion_module(H_table, H_text_expanded).to(head_device)  # [B, C, d_model]

    H_fuse = H_fuse.to(head_device)

    # Move recon_heads to device
    if isinstance(recon_heads, torch.nn.ModuleList):
        recon_heads = recon_heads.to(head_device)
    else:
        recon_heads = recon_heads.to(head_device)

    # Compute losses on masked positions only using per-column heads
    mask_indices_on_head = mask_indices.to(head_device)
    target_ids = target_ids.to(head_device)
    target_nums = target_nums.to(head_device)
    target_types = target_types.to(head_device)

    # Verify single-column training mode (all rows must mask the same column)
    def _is_single_column_mask(mask_indices):
        """Check if all rows mask the same column"""
        if mask_indices.numel() == 0:
            return True
        unique_indices = torch.unique(mask_indices)
        return len(unique_indices) <= 1

    if not _is_single_column_mask(mask_indices_on_head):
        raise ValueError(
            f"train_step_mcm requires single-column training mode. "
            f"All rows in the batch must mask the same column, "
            f"but got mask_indices with unique values: {torch.unique(mask_indices_on_head).tolist()}. "
            f"Please ensure your dataset sets fixed_mask_column before training."
        )

    # Single-column training: all rows mask the same column
    col_idx = int(mask_indices_on_head[0, 0].item()) if mask_indices_on_head.numel() > 0 else 0

    if col_idx < 0 or col_idx >= len(recon_heads):
        loss_cat = torch.tensor(0.0, device=head_device)
        loss_num = torch.tensor(0.0, device=head_device)
    else:
        # Extract column embeddings for all rows at once
        col_embeddings = H_fuse[:, col_idx:col_idx+1, :]  # [B, 1, d_model]
        col_recon_head = recon_heads[col_idx]
        cat_logits, num_preds = col_recon_head(col_embeddings)  # [B, 1, vocab_size], [B, 1]

        # Extract targets for this column
        target_ids_col = target_ids[:, 0] if target_ids.shape[1] > 0 else target_ids.new_zeros(B)
        target_nums_col = target_nums[:, 0] if target_nums.shape[1] > 0 else target_nums.new_zeros(B)
        target_types_col = target_types[:, 0] if target_types.shape[1] > 0 else target_types.new_zeros(B)

        # Vectorized loss computation
        cat_mask = (target_types_col == 0)
        num_mask = (target_types_col == 1)

        # Categorical loss
        if cat_mask.any():
            loss_cat = F.cross_entropy(
                cat_logits[cat_mask].squeeze(1),  # [N, vocab_size]
                target_ids_col[cat_mask]  # [N]
            )
        else:
            loss_cat = torch.tensor(0.0, device=head_device)

        # Numeric loss
        if num_mask.any():
            finite_mask = torch.isfinite(target_nums_col[num_mask])
            if finite_mask.any():
                loss_num = F.mse_loss(
                    num_preds[num_mask][finite_mask].squeeze(),
                    target_nums_col[num_mask][finite_mask]
                )
            else:
                loss_num = torch.tensor(0.0, device=head_device)
        else:
            loss_num = torch.tensor(0.0, device=head_device)

    loss = (float(w_cat) * loss_cat) + (float(w_num) * loss_num)

    optimizer.zero_grad()
    loss.backward()

    # Debug check: ensure params and grads share device/dtype
    device_groups = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not isinstance(param, torch.Tensor) or param.grad is None:
                continue
            if param.device != param.grad.device:
                raise RuntimeError(
                    f"[train_step_mcm] Gradient device mismatch for param {param.shape}: "
                    f"param={param.device}, grad={param.grad.device}"
                )
            if param.dtype != param.grad.dtype:
                raise RuntimeError(
                    f"[train_step_mcm] Gradient dtype mismatch for param {param.shape}: "
                    f"param={param.dtype}, grad={param.grad.dtype}"
                )
            # Track devices
            device_key = (param.device, param.dtype)
            if device_key not in device_groups:
                device_groups[device_key] = []
            device_groups[device_key].append(param)

    # Check if we have params on multiple devices
    if len(device_groups) > 1:
        print("\n[DEBUG] Parameters found on multiple devices/dtypes:")
        for (device, dtype), params in device_groups.items():
            print(f"  Device: {device}, Dtype: {dtype}, Num params: {len(params)}")
        raise RuntimeError(
            f"[train_step_mcm] Found parameters on {len(device_groups)} different device/dtype combinations. "
            "Adam optimizer requires all params in each param_group to be on the same device/dtype."
        )

    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "loss_cat": float(loss_cat.detach().item()),
        "loss_num": float(loss_num.detach().item()),
    }


__all__ = [
    "collate_fn_corruption",
    "collate_fn_contrastive",
    "collate_fn_contrastive_cell_level",
    "collate_fn_mcm",
    "build_tabular_inputs_from_rows",
    "train_step_corruption",
    "compute_embedding_similarity",
    "train_step_contrastive_pretrain",
    "train_step_mcm",
]
