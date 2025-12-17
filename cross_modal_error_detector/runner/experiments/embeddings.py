"""
Text embedding preprocessing utilities for experiments.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from tqdm import tqdm


def _precompute_text_embeddings(
    text_encoder: torch.nn.Module,
    text_descriptions: List[str],
    text_processor,
    device: str,
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """
    Precompute text embeddings for all descriptions.

    Args:
        text_encoder: Text encoder model
        text_descriptions: List of text descriptions
        text_processor: Text processor
        device: Device to use for computation
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping index to cached embedding tensor
    """
    print("正在预计算文本 Embeddings...")
    text_encoder.eval()
    target_device = torch.device(device)
    text_encoder.to(target_device)

    cached_embeddings = {}

    # Process in batches
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
