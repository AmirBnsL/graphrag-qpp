"""Behavioral invariant tests for Candidate Subgraph, Passage-Budget Parity, and Conservative Abstention."""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.fake import FakeEmbedder, FakeOracleGeneratorJudge
from src.data.loader import QuerySample
from src.study import StudyConfig, run_qpp_study
from tests.test_study_tracer_bullet import create_planted_scenarios


def test_passage_budget_parity_across_different_k():
    """Verify that Flat and Graph retrieval strictly match passage counts across k budgets (ADR-0001)."""
    samples = create_planted_scenarios()
    embedder = FakeEmbedder()
    judge = FakeOracleGeneratorJudge()

    for k in [1, 3, 5]:
        config = StudyConfig(retrieval_top_k=k)
        report = run_qpp_study(samples=samples, config=config, embedder=embedder, oracle_judge=judge)
        for res in report.query_results.values():
            assert len(res.flat_passages) == k
            assert len(res.graph_passages) == k


def test_candidate_subgraph_structural_invariants():
    """Verify Candidate Subgraph invariants (ADR-0003, ADR-0006)."""
    samples = create_planted_scenarios()
    config = StudyConfig(retrieval_top_k=5)
    report = run_qpp_study(samples=samples, config=config, embedder=FakeEmbedder(), oracle_judge=FakeOracleGeneratorJudge())

    for res in report.query_results.values():
        sub = res.candidate_subgraph
        # 1. Feature dimensionality is strictly compact and domain-agnostic (no raw 1024-d text embeddings)
        assert sub.node_features.ndim == 2
        assert sub.node_features.shape[1] < 32
        assert sub.node_features.shape[0] == sub.num_nodes

        # 2. Probability vectors are non-negative and properly bounded
        assert np.all(sub.ppr_shallow >= 0.0)
        assert np.all(sub.ppr_deep >= 0.0)
        assert len(sub.ppr_drift) == sub.num_nodes

        # 3. Directional distance encoding (DDE) values are valid distances in [0, 4]
        assert np.all(sub.directional_distance_encoding >= 0)
        assert np.all(sub.directional_distance_encoding <= 4)

        # 4. Edge index validity
        assert sub.edge_index.shape[0] == 2
        if sub.edge_index.shape[1] > 0:
            assert np.max(sub.edge_index) < sub.num_nodes


def test_conservative_graph_abstention_monotonicity():
    """Verify that increasing tau monotonically decreases Graph selections (ADR-0005)."""
    samples = create_planted_scenarios()
    config = StudyConfig(
        retrieval_top_k=5,
        gain_margin_thresholds=[0.0, 0.05, 0.10, 0.15, 0.20],
    )
    report = run_qpp_study(samples=samples, config=config, embedder=FakeEmbedder(), oracle_judge=FakeOracleGeneratorJudge())

    prev_graph_count = None
    for tau in [0.0, 0.10, 0.60, 1.10]:
        stats = report.evaluate_conservative_abstention(tau=tau)
        graph_count = stats["graph_selected_count"]
        flat_count = stats["flat_selected_count"]

        assert graph_count + flat_count == len(samples)
        if prev_graph_count is not None:
            # Monotonically non-increasing as bar gets higher
            assert graph_count <= prev_graph_count
        prev_graph_count = graph_count

    # At tau > 1.0 (impossible margin), Flat must be selected 100% of the time
    high_tau_stats = report.evaluate_conservative_abstention(tau=1.5)
    assert high_tau_stats["graph_selected_count"] == 0
    assert high_tau_stats["flat_selected_count"] == len(samples)
