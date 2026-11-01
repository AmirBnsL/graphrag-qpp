"""Retrieval evaluation metrics: Precision@k, Recall@k, Access Gini, and PPR Entropy."""

from __future__ import annotations

import numpy as np


def compute_precision_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int = 5,
) -> float:
    """Compute Precision@k for retrieved passage IDs against gold evidence IDs."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    gold_set = set(gold_ids)
    hits = sum(1 for rid in top_k if rid in gold_set)
    return float(hits / k)


def compute_recall_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int = 5,
) -> float:
    """Compute Recall@k for retrieved passage IDs against gold evidence IDs."""
    gold_set = set(gold_ids)
    if not gold_set:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in gold_set)
    return float(hits / len(gold_set))


def compute_access_gini(
    frequencies: np.ndarray | list[float],
    only_positive: bool = False,
) -> float:
    """Compute the Gini coefficient measuring access/visit concentration across nodes.
    
    Implements G_access from Heavy-Tailed Memory Traces in Long-Horizon Language Agents
    (Song & Cai, 2026, arXiv:2610.00010). Lower values indicate more balanced, uniform
    distribution; higher values indicate concentration in a small set of super-hubs.
    If only_positive is True, filters to nodes/states with positive access frequency.
    """
    arr = np.asarray(frequencies, dtype=np.float64)
    if only_positive:
        arr = arr[arr > 0]
    if arr.size <= 1:
        return 0.0
    arr = np.sort(np.maximum(arr, 0.0))
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    n = len(arr)
    # Discrete Gini formulation: (2 * sum_{i=1}^n i * y_i) / (n * sum y_i) - (n + 1) / n
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(index * arr)) / (n * total) - (n + 1.0) / n
    return float(max(0.0, min(1.0, gini)))


def compute_ppr_entropy(p: np.ndarray | list[float]) -> float:
    """Compute Shannon entropy (base 2) of a probability distribution."""
    arr = np.asarray(p, dtype=np.float64)
    pos = arr[arr > 0]
    if pos.size == 0:
        return 0.0
    s = float(pos.sum())
    if s <= 0.0:
        return 0.0
    prob = pos / s
    return float(-np.sum(prob * np.log2(prob)))
