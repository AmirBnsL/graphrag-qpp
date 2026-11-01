"""Unit and invariant tests for handcrafted topological feature extraction (ADR-0003, Issue #4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.fake import FakeEmbedder
from src.graph.ppr_retriever import CandidateSubgraphData, PPRRetriever
from src.graph.trigraph_builder import CleanupStage, TriGraphBuilder
from src.qpp.topological_features import extract_topological_features


EXPECTED_FEATURE_KEYS = [
    "seed_deg_mean",
    "seed_deg_max",
    "neighbor_to_seed_hub_ratio",
    "subgraph_deg_gini",
    "lcc_ratio",
    "leaf_fraction",
    "max_k_core",
    "ppr_shallow_entropy",
    "ppr_deep_entropy",
    "ppr_entropy_drift",
    "dde_passage_mean",
    "dde_passage_max",
    "dde_passage_min",
    "anti_homophily_bridge_score",
    "flat_graph_jaccard",
]


def test_topological_feature_keys_and_finite_invariants():
    """Verify that extract_topological_features produces all 15 finite bounded scalar descriptors."""
    embedder = FakeEmbedder()
    builder = TriGraphBuilder(embedder=embedder)
    retriever = PPRRetriever()

    passages = [
        {
            "passage_id": "p1",
            "session_id": "s1",
            "text": "FluxQubit design incorporates a JosephsonJunction loop.",
            "sentences": ["FluxQubit design incorporates a JosephsonJunction loop."],
        },
        {
            "passage_id": "p2",
            "session_id": "s1",
            "text": "JosephsonJunction enables persistent macroscopic quantum coherence.",
            "sentences": ["JosephsonJunction enables persistent macroscopic quantum coherence."],
        },
        {
            "passage_id": "p3",
            "session_id": "s1",
            "text": "Quantum inductance is essential for superconducting circuit stability.",
            "sentences": ["Quantum inductance is essential for superconducting circuit stability."],
        },
    ]

    graph = builder.build(passages, stage=CleanupStage.RAW_PERSISTENT)
    q_vec = embedder.embed_texts(["FluxQubit macroscopic quantum coherence"])[0]

    flat_passages = retriever.retrieve_flat(q_vec, graph, top_k=2)
    graph_passages, cand_subgraph = retriever.retrieve_graph(
        query_vec=q_vec,
        graph=graph,
        flat_top_passages=flat_passages,
        top_k=2,
    )

    features = extract_topological_features(
        candidate_subgraph=cand_subgraph,
        flat_passages=flat_passages,
        graph_passages=graph_passages,
    )

    # 1. Check all 15 expected keys exist
    for key in EXPECTED_FEATURE_KEYS:
        assert key in features, f"Missing expected topological feature: {key}"

    # 2. Check no NaNs or infs
    for key, val in features.items():
        assert isinstance(val, (int, float)), f"Feature {key} is not numeric: {type(val)}"
        assert np.isfinite(val), f"Feature {key} is non-finite: {val}"

    # 3. Check structural invariants
    assert 0.0 <= features["lcc_ratio"] <= 1.0
    assert 0.0 <= features["leaf_fraction"] <= 1.0
    assert 0.0 <= features["flat_graph_jaccard"] <= 1.0
    assert features["max_k_core"] >= 0.0
    assert features["subgraph_deg_gini"] >= 0.0
    assert features["seed_deg_mean"] >= 0.0


def test_topological_features_on_empty_and_trivial_subgraphs():
    """Verify handling of degenerate or minimal candidate subgraphs."""
    # Trivial 1-node subgraph with no edges
    trivial_subgraph = CandidateSubgraphData(
        num_nodes=1,
        node_features=np.zeros((1, 12), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        ppr_shallow=np.array([1.0], dtype=np.float32),
        ppr_deep=np.array([1.0], dtype=np.float32),
        ppr_drift=np.array([0.0], dtype=np.float32),
        ppr_entropy_drift=0.0,
        directional_distance_encoding=np.array([0.0], dtype=np.float32),
        node_labels=["p_0"],
    )

    features = extract_topological_features(
        candidate_subgraph=trivial_subgraph,
        flat_passages=[],
        graph_passages=[],
    )

    for key in EXPECTED_FEATURE_KEYS:
        assert key in features
        assert np.isfinite(features[key])

    assert features["lcc_ratio"] == 1.0
    assert features["flat_graph_jaccard"] == 0.0
    assert features["max_k_core"] == 0.0
