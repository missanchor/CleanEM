"""
Text embedding preprocessing utilities for experiments.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


def _precompute_text_embeddings(
    text_encoder: torch.nn.Module,
    text_descriptions: List[str],
    text_processor,
    device: str,
    batch_size: int = 32,
    per_column_prompts: Optional[List[Dict[int, str]]] = None,
    column_names: Optional[List[str]] = None,
    use_text_output_last_token_embedding: bool = False,
) -> Dict:
    """
    Precompute text embeddings for all descriptions.

    Args:
        text_encoder: Text encoder model
        text_descriptions: List of text descriptions (one per row)
        text_processor: Text processor
        device: Device to use for computation
        batch_size: Batch size for processing
        per_column_prompts: Optional list of dicts mapping col_idx to prompt for each row.
                          If provided, will compute embeddings per (row, col) instead of per row.
        column_names: List of column names (used for generating per-column prompts if needed)
        use_text_output_last_token_embedding: If True, extract and cache the last token embedding
                                             from the text encoder output instead of full sequence.
                                             Default: False (cache full sequence embeddings).

    Returns:
        Dictionary mapping (row_idx, col_idx) or row_idx to cached embedding tensor.
        Returns Dict[Tuple[int, int], torch.Tensor] if per_column_prompts is provided,
        otherwise returns Dict[int, torch.Tensor].
    """
    print("正在预计算文本 Embeddings...")
    text_encoder.eval()
    target_device = torch.device(device)
    text_encoder.to(target_device)

    cached_embeddings = {}

    # Case 1: Per-column embeddings (for MCM experiments with per-column prompts)
    if per_column_prompts is not None:
        print("  → 为每行每列分别计算文本 Embeddings...")
        num_rows = len(per_column_prompts)

        for row_idx in tqdm(range(num_rows), desc="Caching Per-Column Embeddings", ncols=100):
            row_prompts = per_column_prompts[row_idx]

            # Process all columns for this row
            batch_inputs = []
            col_indices = []

            for col_idx, prompt in row_prompts.items():
                batch_inputs.append(text_processor.process(prompt))
                col_indices.append(col_idx)

            if not batch_inputs:
                continue

            # Collate
            input_ids = torch.stack([x["input_ids"] for x in batch_inputs]).to(target_device)
            attention_mask = torch.stack([x["attention_mask"] for x in batch_inputs]).to(target_device)

            text_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            with torch.no_grad():
                embeddings = text_encoder(text_inputs)

            # Extract last token embedding if requested
            if use_text_output_last_token_embedding and embeddings.dim() == 3:
                embeddings = embeddings[:, -1, :]  # [B, d_model]

            # Store each column embedding with (row_idx, col_idx) key
            for col_idx, emb in zip(col_indices, embeddings):
                emb_cpu = emb.detach().cpu()
                if emb_cpu.dim() == 1:
                    emb_cpu = emb_cpu.unsqueeze(0)
                cached_embeddings[(row_idx, col_idx)] = emb_cpu

        print(f"✓ 已缓存 {len(cached_embeddings)} 个(行,列)文本 Embeddings")
        return cached_embeddings

    # Case 2: Per-row embeddings (default, for backward compatibility)
    print("  → 为每行计算文本 Embeddings...")
    num_samples = len(text_descriptions)
    for i in tqdm(range(0, num_samples, batch_size), desc="Caching Embeddings", ncols=100):
        batch_texts = text_descriptions[i : i + batch_size]
        indices = list(range(i, min(i + batch_size, num_samples)))

        # Process batch
        batch_inputs = [text_processor.process(t) for t in batch_texts]

        # Collate manually
        input_ids = torch.stack([x["input_ids"] for x in batch_inputs]).to(target_device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch_inputs]).to(target_device)

        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            embeddings = text_encoder(text_inputs)

        # Extract last token embedding if requested
        if use_text_output_last_token_embedding and embeddings.dim() == 3:
            embeddings = embeddings[:, -1, :]  # [B, d_model]

        # Store individually
        for idx, emb in zip(indices, embeddings):
            emb_cpu = emb.detach().cpu()
            # Ensure cached embeddings are at least 2D so downstream pooling (dim=1) is well-defined.
            # train_step_contrastive_pretrain expects cached_embedding shaped like [seq_len, d_model] per sample.
            if emb_cpu.dim() == 1:
                emb_cpu = emb_cpu.unsqueeze(0)
            cached_embeddings[idx] = emb_cpu

    print(f"✓ 已缓存 {len(cached_embeddings)} 条文本 Embeddings")
    return cached_embeddings
