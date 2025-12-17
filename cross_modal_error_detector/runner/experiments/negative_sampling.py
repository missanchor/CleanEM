"""
Negative sampling strategies for contrastive learning.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _parse_negative_sampling_config(stage_b_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize Stage B negative sampling configuration into a list of strategy dicts.

    Args:
        stage_b_cfg: Stage B configuration dictionary

    Returns:
        List of normalized strategy dictionaries
    """
    default_ratio = float(stage_b_cfg.get("negative_ratio", 1.0))
    strategy_cfg = stage_b_cfg.get("negative_strategy", "random")

    def _normalize_entry(entry) -> Dict[str, Any]:
        if isinstance(entry, str):
            return {"type": entry, "ratio": default_ratio}
        if isinstance(entry, dict):
            if "type" not in entry:
                raise ValueError("negative_strategy dict entries must include a 'type' key.")
            normalized = dict(entry)
            normalized.setdefault("ratio", default_ratio)
            return normalized
        raise TypeError("negative_strategy must be a string, dict, or list of them.")

    strategies: List[Dict[str, Any]] = []
    if isinstance(strategy_cfg, list):
        for cfg in strategy_cfg:
            strategies.append(_normalize_entry(cfg))
    else:
        strategies.append(_normalize_entry(strategy_cfg))

    # Filter out zero/negative ratios to avoid useless work
    filtered = [s for s in strategies if s.get("ratio", 0) > 0]
    if not filtered:
        filtered = [{"type": "random", "ratio": max(default_ratio, 1.0)}]
    return filtered


def _sample_different_index(num_rows: int, exclude_idx: int) -> int:
    """
    Return a random index in [0, num_rows) that is different from exclude_idx.

    Args:
        num_rows: Total number of rows
        exclude_idx: Index to exclude

    Returns:
        Random index different from exclude_idx
    """
    if num_rows <= 1:
        return exclude_idx
    candidate = random.randint(0, num_rows - 2)
    if candidate >= exclude_idx:
        candidate += 1
    return candidate


def _sample_row_text_mismatch(num_rows: int, num_samples: int) -> List[Tuple[int, int]]:
    """
    Generate (row_i, text_j) pairs with i != j.

    Args:
        num_rows: Number of rows
        num_samples: Number of samples to generate

    Returns:
        List of (row_idx, text_idx) pairs
    """
    pairs: List[Tuple[int, int]] = []
    for _ in range(num_samples):
        row_idx = random.randint(0, num_rows - 1)
        text_idx = _sample_different_index(num_rows, row_idx)
        pairs.append((row_idx, text_idx))
    return pairs


def _sample_row_text_mismatch_different_column_value(
    rows: List[List[Any]],
    col_idx: int,
    num_samples: int,
    *,
    max_tries: int = 20,
) -> List[Tuple[int, int]]:
    """
    Generate (row_i, text_j) pairs with i != j and rows[i][col_idx] != rows[j][col_idx] (string-compared).

    This reduces label noise for per-column training when a column has many repeated values
    (random row/text mismatch can still accidentally match on that column).

    Args:
        rows: List of rows
        col_idx: Column index to check
        num_samples: Number of samples to generate
        max_tries: Maximum attempts to find a valid pair

    Returns:
        List of (row_idx, text_idx) pairs
    """
    num_rows = len(rows)
    if num_rows == 0:
        return []
    if col_idx < 0 or (rows and col_idx >= len(rows[0])):
        raise ValueError(f"col_idx out of range: {col_idx}")

    col_values = [str(r[col_idx]) for r in rows]

    pairs: List[Tuple[int, int]] = []
    for _ in range(num_samples):
        row_idx = random.randint(0, num_rows - 1)
        v = col_values[row_idx]

        chosen = None
        for _try in range(max_tries):
            cand = _sample_different_index(num_rows, row_idx)
            if col_values[cand] != v:
                chosen = cand
                break

        text_idx = chosen if chosen is not None else _sample_different_index(num_rows, row_idx)
        pairs.append((row_idx, text_idx))
    return pairs


def _sample_row_text_swap(num_rows: int, num_samples: int) -> List[Tuple[int, int]]:
    """
    Generate (row_j, text_i) pairs by swapping the tabular/text indices.

    Args:
        num_rows: Number of rows
        num_samples: Number of samples to generate

    Returns:
        List of (row_idx, text_idx) pairs
    """
    pairs: List[Tuple[int, int]] = []
    for _ in range(num_samples):
        text_idx = random.randint(0, num_rows - 1)
        row_idx = _sample_different_index(num_rows, text_idx)
        pairs.append((row_idx, text_idx))
    return pairs


def _build_semantic_neighbor_map(
    cached_text_embeddings: Dict[int, torch.Tensor],
    max_top_k: int,
) -> Dict[int, List[int]]:
    """
    Build a map row_idx -> similar text indices based on cosine similarity.

    Args:
        cached_text_embeddings: Dictionary of cached text embeddings
        max_top_k: Maximum number of top neighbors to consider

    Returns:
        Dictionary mapping row index to list of similar text indices
    """
    if not cached_text_embeddings:
        raise ValueError("similar_text negative sampling requires cached text embeddings.")
    sorted_indices = sorted(cached_text_embeddings.keys())
    text_vectors = []
    for idx in sorted_indices:
        emb = cached_text_embeddings[idx]
        if emb.dim() == 1:
            pooled = emb
        else:
            pooled = emb.mean(dim=0)
        text_vectors.append(pooled)
    text_matrix = torch.stack(text_vectors, dim=0).to(torch.float32)
    with torch.no_grad():
        normalized = F.normalize(text_matrix, dim=-1)
        similarity = torch.matmul(normalized, normalized.T)
        similarity.fill_diagonal_(float("-inf"))
        top_k = min(max_top_k, similarity.shape[1] - 1)
        if top_k <= 0:
            return {idx: [] for idx in sorted_indices}
        _, top_indices = torch.topk(similarity, k=top_k, dim=-1)

    neighbor_map: Dict[int, List[int]] = {}
    for pos, row_idx in enumerate(sorted_indices):
        neighbor_map[row_idx] = [sorted_indices[idx.item()] for idx in top_indices[pos]]
    return neighbor_map


def _sample_similar_text_pairs(
    num_rows: int,
    num_samples: int,
    neighbor_map: Dict[int, List[int]],
    top_k: int,
) -> List[Tuple[int, int]]:
    """
    Sample (row_i, text_j) where text_j is among the top-K similar descriptions to row_i.

    Args:
        num_rows: Number of rows
        num_samples: Number of samples to generate
        neighbor_map: Map of similar neighbors
        top_k: Number of top neighbors to consider

    Returns:
        List of (row_idx, text_idx) pairs
    """
    pairs: List[Tuple[int, int]] = []
    if not neighbor_map:
        return pairs
    for _ in range(num_samples):
        row_idx = random.randint(0, num_rows - 1)
        neighbors = neighbor_map.get(row_idx, [])
        if not neighbors:
            continue
        effective_k = min(top_k, len(neighbors))
        text_idx = random.choice(neighbors[:effective_k])
        pairs.append((row_idx, text_idx))
    return pairs


def _generate_negative_pairs(
    strategy: Dict[str, Any],
    num_rows: int,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    *,
    dirty_rows: Optional[List[List[Any]]] = None,
    col_idx: Optional[int] = None,
    cell_value_neighbor_map: Optional[Dict[int, Dict[int, List[int]]]] = None,
) -> List[Tuple[int, int]]:
    """
    Generate (row_idx, text_idx) negative pairs according to the strategy.

    Args:
        strategy: Sampling strategy configuration
        num_rows: Number of rows
        neighbor_map: Map of similar neighbors
        dirty_rows: List of rows
        col_idx: Column index

    Returns:
        List of (row_idx, text_idx) negative pairs
    """
    ratio = float(strategy.get("ratio", 0))
    num_samples = max(0, int(num_rows * ratio))
    if num_samples <= 0:
        return []

    strategy_type = strategy.get("type", "random")
    if strategy_type == "random":
        return _sample_row_text_mismatch(num_rows, num_samples)
    if strategy_type == "row_text_swap":
        return _sample_row_text_swap(num_rows, num_samples)
    if strategy_type in {"column_value_mismatch", "col_value_mismatch"}:
        if dirty_rows is None or col_idx is None:
            raise ValueError("column_value_mismatch requires dirty_rows and col_idx.")
        return _sample_row_text_mismatch_different_column_value(
            dirty_rows,
            int(col_idx),
            num_samples,
            max_tries=int(strategy.get("max_tries", 20)),
        )
    if strategy_type == "similar_text":
        top_k = int(strategy.get("top_k", 5))
        top_k = max(1, top_k)
        return _sample_similar_text_pairs(
            num_rows,
            num_samples,
            neighbor_map or {},
            top_k,
        )
    if strategy_type == "similar_cell_value":
        top_k = int(strategy.get("top_k", 5))
        top_k = max(1, top_k)
        if cell_value_neighbor_map is None:
            raise ValueError("similar_cell_value requires cell_value_neighbor_map.")
        return _sample_similar_cell_value_pairs(
            num_rows,
            num_samples,
            cell_value_neighbor_map.get(col_idx, {}),
            top_k,
        )
    raise ValueError(f"Unknown negative sampling strategy: {strategy_type}")


def _build_cell_value_semantic_neighbor_map(
    cached_row_embeddings: Dict[int, torch.Tensor],
    col_idx: int,
    max_top_k: int,
) -> Dict[int, List[int]]:
    """
    构建特定列的cell value语义邻居映射

    Args:
        cached_row_embeddings: {row_idx: [num_cols, d_model]}
        col_idx: 目标列索引
        max_top_k: 最大邻居数

    Returns:
        {row_idx: [similar_row_idx1, similar_row_idx2, ...]}
    """
    # 1. 提取指定列的所有cell embeddings
    sorted_indices = sorted(cached_row_embeddings.keys())
    cell_vectors = []
    for row_idx in sorted_indices:
        cell_emb = cached_row_embeddings[row_idx][col_idx, :]  # [d_model]
        cell_vectors.append(cell_emb)

    # 2. 计算余弦相似度矩阵
    cell_matrix = torch.stack(cell_vectors, dim=0)  # [num_rows, d_model]
    with torch.no_grad():
        normalized = F.normalize(cell_matrix, dim=-1)
        similarity = torch.matmul(normalized, normalized.T)  # [num_rows, num_rows]
        similarity.fill_diagonal_(float("-inf"))  # 排除自身
        _, top_indices = torch.topk(similarity, k=min(max_top_k, len(sorted_indices)-1), dim=-1)

    # 3. 构建邻居映射
    neighbor_map = {}
    for pos, row_idx in enumerate(sorted_indices):
        neighbor_map[row_idx] = [sorted_indices[idx.item()] for idx in top_indices[pos]]
    return neighbor_map


def _sample_similar_cell_value_pairs(
    num_rows: int,
    num_samples: int,
    neighbor_map: Dict[int, List[int]],
    top_k: int,
) -> List[Tuple[int, int]]:
    """
    从相似单元格值中采样负样本对

    Args:
        neighbor_map: {row_idx: [similar_row_indices]}
        top_k: 实际使用的邻居数

    Returns:
        [(row_idx, text_idx), ...]
    """
    pairs = []
    for _ in range(num_samples):
        row_idx = random.randint(0, num_rows - 1)
        neighbors = neighbor_map.get(row_idx, [])
        if not neighbors:
            # 没有相似邻居，退化为随机采样
            text_idx = _sample_different_index(num_rows, row_idx)
        else:
            effective_k = min(top_k, len(neighbors))
            text_idx = random.choice(neighbors[:effective_k])
        pairs.append((row_idx, text_idx))
    return pairs
