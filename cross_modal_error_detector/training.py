"""
Collate utilities and training steps for different strategies.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import CrossModalErrorDetector
from .utils.device import resolve_runtime_device


def collate_fn_corruption(batch):
    """
    Collate function for corruption-based training.
    """

    tabular_inputs_list, text_inputs_list, labels_list = zip(*batch)

    cell_embeddings = torch.stack([x["cell_embeddings"] for x in tabular_inputs_list])
    row_indices = torch.stack([x["row_indices"] for x in tabular_inputs_list])
    col_indices = torch.stack([x["col_indices"] for x in tabular_inputs_list])

    tabular_inputs = {
        "cell_embeddings": cell_embeddings,
        "row_indices": row_indices,
        "col_indices": col_indices,
    }

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
    return tabular_inputs, text_inputs, labels


def collate_fn_contrastive(batch):
    """
    Collate function for contrastive training.
    """

    tabular_inputs_list, text_inputs_list = zip(*batch)

    cell_embeddings = torch.stack([x["cell_embeddings"] for x in tabular_inputs_list])
    row_indices = torch.stack([x["row_indices"] for x in tabular_inputs_list])
    col_indices = torch.stack([x["col_indices"] for x in tabular_inputs_list])

    tabular_inputs = {
        "cell_embeddings": cell_embeddings,
        "row_indices": row_indices,
        "col_indices": col_indices,
    }

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

    return tabular_inputs, text_inputs


def collate_fn_contrastive_cell_level(batch):
    """
    Collate function for two-stage contrastive cell-level binary classification training.
    Expects batch items with 4 values: (tabular_inputs, text_inputs, labels, cell_idx)
    """

    tabular_inputs_list, text_inputs_list, labels_list, cell_indices = zip(*batch)

    cell_embeddings = torch.stack([x["cell_embeddings"] for x in tabular_inputs_list])
    row_indices = torch.stack([x["row_indices"] for x in tabular_inputs_list])
    col_indices = torch.stack([x["col_indices"] for x in tabular_inputs_list])

    tabular_inputs = {
        "cell_embeddings": cell_embeddings,
        "row_indices": row_indices,
        "col_indices": col_indices,
    }

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
    return tabular_inputs, text_inputs, labels, cell_indices


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
    device: Optional[str] = None,
    *,
    device_map: Optional[Dict[str, str]] = None,
) -> float:
    """
    Single training step for corruption-based strategy.
    """

    model.train()
    tabular_inputs, text_inputs, labels = batch

    runtime_device = resolve_runtime_device(device)
    tabular_device = _resolve_device(runtime_device, device_map, "tabular_encoder")
    text_device = _resolve_device(runtime_device, device_map, "text_encoder")
    label_device = _resolve_device(runtime_device, device_map, "detection_head")

    tabular_inputs = _move_to_device(tabular_inputs, tabular_device)
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
        # Ensure batch dimension
        if H_text.dim() == 2:
            H_text = H_text.unsqueeze(0)
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
    device: Optional[str] = None,
    temperature: float = 0.07,
    *,
    device_map: Optional[Dict[str, str]] = None,
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

    tabular_inputs, text_inputs = batch
    batch_size = tabular_inputs["cell_embeddings"].shape[0]

    # Move encoders to their respective devices
    tabular_encoder = tabular_encoder.to(default_device)
    if "cached_embedding" not in text_inputs:
        text_encoder = text_encoder.to(text_encoder_device)

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
        H_text_all = text_inputs["cached_embedding"]  # [batch_size, seq_len, d_model]
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


__all__ = [
    "collate_fn_corruption",
    "collate_fn_contrastive",
    "collate_fn_contrastive_cell_level",
    "train_step_corruption",
    "compute_embedding_similarity",
    "train_step_contrastive_pretrain",
]
