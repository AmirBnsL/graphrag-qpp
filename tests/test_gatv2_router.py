"""Unit, invariant, and cross-validation tests for Domain-Agnostic GATv2 GNN Router (ADR-0003, Issue #5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.adapters.fake import FakeEmbedder, FakeOracleGeneratorJudge
from src.data.loader import QuerySample
from src.graph.ppr_retriever import CandidateSubgraphData, PPRRetriever
from src.graph.trigraph_builder import CleanupStage, TriGraphBuilder
from src.qpp.gnn_router import (
    GATv2ContextSelector,
    TopologicalGNNRouter,
    candidate_subgraph_to_pyg_data,
    evaluate_gnn_in_domain_cv,
    evaluate_gnn_zero_shot_transfer,
)
from src.qpp.tabular_router import TabularCVResult, TransferResult
from src.study import StudyConfig, run_qpp_study


def _create_mock_candidate_subgraph(
    n_nodes: int = 10,
    bridge_strength: float = 1.0,
    seed: int = 42,
) -> CandidateSubgraphData:
    """Create a synthetic CandidateSubgraphData with domain-agnostic scalar features."""
    rng = np.random.default_rng(seed)

    # 12 scalar features (ADR-0003: D < 32, zero 1024-d text embeddings)
    features = np.zeros((n_nodes, 12), dtype=np.float32)
    # Col 0: query_sim
    features[:, 0] = rng.uniform(0.2, 0.8, size=n_nodes)
    features[0, 0] = 0.9  # Seed entity high query sim
    # Col 1: sub_p_shallow, Col 2: sub_p_deep, Col 3: sub_drift
    features[:, 1] = rng.uniform(0.01, 0.2, size=n_nodes)
    features[:, 2] = rng.uniform(0.01, 0.2, size=n_nodes)
    features[:, 3] = features[:, 2] - features[:, 1]
    # Col 4: sub_deg, Col 5: global_deg
    features[:, 4] = np.log1p(rng.integers(1, 5, size=n_nodes))
    features[:, 5] = np.log1p(rng.integers(2, 8, size=n_nodes))
    # Col 6: node_idf_feat
    features[:, 6] = rng.uniform(0.5, 2.5, size=n_nodes)
    # Col 7: dde_feat (dde / 4.0)
    dde = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 4][:n_nodes], dtype=np.float32)
    features[:, 7] = dde / 4.0
    # Col 8: is_flat, Col 9: is_p, Col 10: is_s, Col 11: is_e
    features[:3, 9] = 1.0  # Passages
    features[3:5, 10] = 1.0  # Sentences
    features[5:, 11] = 1.0  # Entities

    # Simple connected path / tree edge_index
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    edge_index = np.array([src, dst], dtype=np.int64)

    return CandidateSubgraphData(
        num_nodes=n_nodes,
        node_features=features,
        edge_index=edge_index,
        ppr_shallow=features[:, 1],
        ppr_deep=features[:, 2],
        ppr_drift=features[:, 3],
        ppr_entropy_drift=float(bridge_strength * 0.2),
        directional_distance_encoding=dde,
        node_labels=[f"node_{i}" for i in range(n_nodes)],
    )


def test_candidate_subgraph_to_pyg_data_invariants():
    """Verify PyG Data conversion strictly respects ADR-0003 domain-agnostic inputs."""
    cand = _create_mock_candidate_subgraph(n_nodes=8, seed=1)
    pyg_data = candidate_subgraph_to_pyg_data(cand, label=0.25)

    assert isinstance(pyg_data, Data)
    # Feature dimension must be strictly less than 32 (ADR-0003, zero 1024-d text embeddings)
    assert pyg_data.x.shape == (8, 12)
    assert pyg_data.x.shape[1] < 32
    assert pyg_data.edge_index.shape[0] == 2
    assert pyg_data.edge_index.shape[1] == 14  # 7 undirected edges * 2
    assert pyg_data.y.item() == pytest.approx(0.25)
    assert torch.all(torch.isfinite(pyg_data.x))


def test_gatv2_architecture_forward_pass_and_gradient_flow():
    """Verify 2-layer GATv2 architecture with query-similarity attention readout and backprop."""
    cand1 = _create_mock_candidate_subgraph(n_nodes=6, seed=1)
    cand2 = _create_mock_candidate_subgraph(n_nodes=8, seed=2)

    data1 = candidate_subgraph_to_pyg_data(cand1, label=0.3)
    data2 = candidate_subgraph_to_pyg_data(cand2, label=-0.2)
    batch = Batch.from_data_list([data1, data2])

    model = GATv2ContextSelector(in_channels=12, hidden_channels=16, heads=2)
    model.train()

    preds = model(batch.x, batch.edge_index, batch.batch)
    assert preds.shape == (2,)

    # Test backward pass and valid gradient flow across all parameters
    loss = torch.nn.functional.mse_loss(preds, batch.y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for parameter: {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in: {name}"


def test_topological_gnn_router_fit_predict_route():
    """Verify TopologicalGNNRouter training, prediction, and Conservative Graph Abstention routing."""
    graphs = [
        candidate_subgraph_to_pyg_data(
            _create_mock_candidate_subgraph(n_nodes=8, bridge_strength=float(i % 3), seed=i),
            label=float(0.15 * (i % 3) - 0.1),
        )
        for i in range(20)
    ]
    targets = np.array([g.y.item() for g in graphs], dtype=np.float32)

    router = TopologicalGNNRouter(
        in_channels=12,
        hidden_channels=16,
        heads=2,
        epochs=15,
        lr=0.01,
        random_state=42,
    )
    router.fit(graphs, targets)

    preds = router.predict(graphs)
    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))

    # Conservative Graph Abstention check across tau
    for tau in [0.0, 0.05, 0.10, 0.15, 0.20]:
        decisions = router.route(graphs, tau=tau)
        assert decisions.shape == (20,)
        expected = (preds >= tau).astype(int)
        np.testing.assert_array_equal(decisions, expected)


def test_gnn_five_fold_in_domain_cv():
    """Verify 5-fold cross-validation on GATv2 router computes out-of-fold metrics and sweep results."""
    # Planted correlation between feature signal and target gain
    graphs = []
    targets = []
    for i in range(25):
        strength = float((i % 5) - 2) * 0.2  # Range [-0.4, 0.4]
        cand = _create_mock_candidate_subgraph(n_nodes=8, bridge_strength=strength, seed=i)
        # Inject strong signal into query_sim of bridge entity to enable learnability
        cand.node_features[5, 0] = float(0.5 + strength)
        data = candidate_subgraph_to_pyg_data(cand, label=strength)
        graphs.append(data)
        targets.append(strength)

    targets_arr = np.array(targets, dtype=np.float32)

    cv_result: TabularCVResult = evaluate_gnn_in_domain_cv(
        graphs=graphs,
        targets=targets_arr,
        n_splits=5,
        tau_sweep=[0.0, 0.05, 0.10, 0.15, 0.20],
        epochs=15,
        random_state=42,
    )

    assert isinstance(cv_result, TabularCVResult)
    assert len(cv_result.out_of_fold_predictions) == 25
    assert np.all(np.isfinite(cv_result.out_of_fold_predictions))
    assert cv_result.spearman_rho > 0.0
    assert len(cv_result.tau_sweep_results) == 5


def test_gnn_zero_shot_topological_transfer():
    """Verify Zero-Shot Topological Transfer of GATv2 router across benchmarks (GLOSSARY.md)."""
    # Source dataset: 20 graphs with planted signal
    source_graphs = []
    source_targets = []
    for i in range(20):
        strength = float((i % 4) - 1.5) * 0.25
        cand = _create_mock_candidate_subgraph(n_nodes=8, bridge_strength=strength, seed=100 + i)
        cand.node_features[5, 0] = float(0.5 + strength)
        source_graphs.append(candidate_subgraph_to_pyg_data(cand, label=strength))
        source_targets.append(strength)

    # Target dataset: 15 graphs with different seed/topology
    target_graphs = []
    target_targets = []
    for i in range(15):
        strength = float((i % 3) - 1.0) * 0.25
        cand = _create_mock_candidate_subgraph(n_nodes=6, bridge_strength=strength, seed=200 + i)
        cand.node_features[5, 0] = float(0.5 + strength)
        target_graphs.append(candidate_subgraph_to_pyg_data(cand, label=strength))
        target_targets.append(strength)

    router = TopologicalGNNRouter(
        in_channels=12, hidden_channels=16, heads=2, epochs=20, lr=0.01, random_state=42
    )
    router.fit(source_graphs, np.array(source_targets, dtype=np.float32))

    transfer_result: TransferResult = evaluate_gnn_zero_shot_transfer(
        router=router,
        graphs_target=target_graphs,
        targets_target=np.array(target_targets, dtype=np.float32),
        tau_sweep=[0.0, 0.05, 0.10, 0.15, 0.20],
    )

    assert isinstance(transfer_result, TransferResult)
    assert len(transfer_result.predictions) == 15
    assert transfer_result.spearman_rho > 0.0
    assert len(transfer_result.tau_sweep_results) == 5


def test_study_report_evaluate_topological_gnn():
    """Verify StudyReport seam method evaluate_topological_gnn."""
    embedder = FakeEmbedder()
    oracle_judge = FakeOracleGeneratorJudge()
    config = StudyConfig(retrieval_top_k=2)

    passages = [
        {"passage_id": "p1", "text": "FluxQubit macroscopic quantum coherence loop.", "sentences": ["FluxQubit macroscopic quantum coherence loop."]},
        {"passage_id": "p2", "text": "JosephsonJunction enables quantum coherence.", "sentences": ["JosephsonJunction enables quantum coherence."]},
    ]
    samples = [
        QuerySample(
            query_id=f"q{i}", dataset="bench", question_type="multi-hop",
            question=f"FluxQubit quantum coherence query {i}?", gold_answers=["yes"],
            gold_evidence_ids=["p1", "p2"], corpus_passages=passages,
        )
        for i in range(1, 5)
    ]

    report = run_qpp_study(samples, config=config, embedder=embedder, oracle_judge=oracle_judge)

    gnn_cv_result = report.evaluate_topological_gnn(target="semantic", n_splits=2, epochs=5)
    assert isinstance(gnn_cv_result, TabularCVResult)
    assert len(gnn_cv_result.out_of_fold_predictions) == 4
