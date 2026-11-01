"""Unit tests for retrieval metrics: Precision@k, Recall@k, Access Gini, and PPR Entropy."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.retrieval_metrics import (
    compute_access_gini,
    compute_ppr_entropy,
    compute_precision_at_k,
    compute_recall_at_k,
)


def test_precision_and_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    gold = ["doc2", "doc5"]

    assert compute_precision_at_k(retrieved, gold, k=5) == 2 / 5
    assert compute_precision_at_k(retrieved, gold, k=2) == 1 / 2
    assert compute_precision_at_k(retrieved, gold, k=1) == 0.0

    assert compute_recall_at_k(retrieved, gold, k=5) == 1.0
    assert compute_recall_at_k(retrieved, gold, k=2) == 0.5
    assert compute_recall_at_k(retrieved, gold, k=1) == 0.0

    # Edge cases
    assert compute_precision_at_k([], gold, k=5) == 0.0
    assert compute_recall_at_k([], gold, k=5) == 0.0
    assert compute_recall_at_k(retrieved, [], k=5) == 1.0


def test_access_gini():
    # Uniform distribution -> Gini should be 0.0
    uniform = np.array([5.0, 5.0, 5.0, 5.0])
    assert pytest.approx(compute_access_gini(uniform), abs=1e-5) == 0.0

    # Skewed distribution (one hub has all visits) -> Gini near 1.0
    skewed = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
    # For n=5, delta has Gini = (5-1)/5 = 0.8
    assert pytest.approx(compute_access_gini(skewed), abs=1e-5) == 0.8

    # Monotonicity: higher skew -> higher Gini
    mild_skew = np.array([1.0, 2.0, 3.0, 4.0])
    heavy_skew = np.array([0.1, 0.1, 0.1, 10.0])
    assert compute_access_gini(heavy_skew) > compute_access_gini(mild_skew)

    # Empty and single element edge cases
    assert compute_access_gini([]) == 0.0
    assert compute_access_gini([10.0]) == 0.0
    assert compute_access_gini([0.0, 0.0]) == 0.0


def test_ppr_entropy():
    # Uniform over 4 items -> log2(4) = 2.0 bits
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    assert pytest.approx(compute_ppr_entropy(p_uniform), abs=1e-5) == 2.0

    # Delta distribution (all mass on one item) -> 0.0 bits
    p_delta = np.array([1.0, 0.0, 0.0, 0.0])
    assert pytest.approx(compute_ppr_entropy(p_delta), abs=1e-5) == 0.0

    # Empty
    assert compute_ppr_entropy([]) == 0.0
