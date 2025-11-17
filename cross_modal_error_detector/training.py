"""
Collate utilities and training steps for different strategies.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import CrossModalErrorDetector


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

    input_ids = torch.stack([x["input_ids"] for x in text_inputs_list])
    attention_mask = torch.stack([x["attention_mask"] for x in text_inputs_list])

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

    input_ids = torch.stack([x["input_ids"] for x in text_inputs_list])
    attention_mask = torch.stack([x["attention_mask"] for x in text_inputs_list])

    text_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    return tabular_inputs, text_inputs


def _resolve_device(default_device: str, device_map: Optional[Dict[str, str]], key: str) -> str:
    if device_map:
        if key in device_map:
            return device_map[key]
        if "default" in device_map:
            return device_map["default"]
    return default_device


def _move_to_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    target = torch.device(device)
    return {k: (v if v.device == target else v.to(target)) for k, v in inputs.items()}


def train_step_corruption(
    model: CrossModalErrorDetector,
    batch: Tuple,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    *,
    device_map: Optional[Dict[str, str]] = None,
) -> float:
    """
    Single training step for corruption-based strategy.
    """

    model.train()
    tabular_inputs, text_inputs, labels = batch

    tabular_device = _resolve_device(device, device_map, "tabular_encoder")
    text_device = _resolve_device(device, device_map, "text_encoder")
    label_device = _resolve_device(device, device_map, "detection_head")

    tabular_inputs = _move_to_device(tabular_inputs, tabular_device)
    text_inputs = _move_to_device(text_inputs, text_device)
    labels = labels.to(label_device)

    logits = model(tabular_inputs, text_inputs).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_contrastive(
    model: CrossModalErrorDetector,
    batch: Tuple,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    temperature: float = 0.07,
    *,
    device_map: Optional[Dict[str, str]] = None,
) -> float:
    """
    Single training step for contrastive strategy using InfoNCE loss.
    """

    model.train()
    tabular_inputs, text_inputs = batch
    batch_size = tabular_inputs["cell_embeddings"].shape[0]

    tabular_device = _resolve_device(device, device_map, "tabular_encoder")
    text_device = _resolve_device(device, device_map, "text_encoder")
    score_device = _resolve_device(device, device_map, "detection_head")

    tabular_inputs = _move_to_device(tabular_inputs, tabular_device)
    text_inputs = _move_to_device(text_inputs, text_device)

    scores_matrix = torch.zeros(batch_size, batch_size, device=score_device)

    for i in range(batch_size):
        tab_i = {
            "cell_embeddings": tabular_inputs["cell_embeddings"][i : i + 1],
            "row_indices": tabular_inputs["row_indices"][i : i + 1],
            "col_indices": tabular_inputs["col_indices"][i : i + 1],
        }

        for j in range(batch_size):
            txt_j = {
                "input_ids": text_inputs["input_ids"][j : j + 1],
                "attention_mask": text_inputs["attention_mask"][j : j + 1],
            }
            score = model(tab_i, txt_j)
            scores_matrix[i, j] = score.squeeze()

    scores_matrix = scores_matrix / temperature
    labels = torch.arange(batch_size, device=score_device)
    loss = F.cross_entropy(scores_matrix, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


__all__ = [
    "collate_fn_corruption",
    "collate_fn_contrastive",
    "train_step_corruption",
    "train_step_contrastive",
]


