"""Unit, invariant, and cross-validation tests for Topological LightGBM Router (Issue #4)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.adapters.fake import FakeEmbedder, FakeOracleGeneratorJudge
from src.data.loader import QuerySample
from src.qpp.tabular_router import (
    DisagreementMatrix,
    TabularCVResult,
    TopologicalLightGBMRouter,
    TransferResult,
    compute_router_disagreement,
    evaluate_conservative_abstention_sweep,
    evaluate_in_domain_cv,
    evaluate_router_disagreement_sweep,
    evaluate_zero_shot_transfer,
)
from src.study import StudyConfig, run_qpp_study


def _generate_synthetic_subgraph_features(
    n_samples: int = 40, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic topological features, lexical gains, and semantic gains with planted correlation."""
    rng = np.random.default_rng(seed)

    # 15 features matching extract_topological_features
    # Positive bridge signal correlates with positive gain
    bridge_signal = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    hub_signal = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    X = np.zeros((n_samples, 15), dtype=np.float32)
    # Feature 13 is anti_homophily_bridge_score
    X[:, 13] = bridge_signal + rng.normal(0, 0.2, size=n_samples)
    # Feature 2 is neighbor_to_seed_hub_ratio
    X[:, 2] = hub_signal + rng.normal(0, 0.2, size=n_samples)
    # Feature 9 is ppr_entropy_drift
    X[:, 9] = np.abs(rng.normal(0.5, 0.3, size=n_samples))
    # Other features
    for col in range(15):
        if col not in (2, 9, 13):
            X[:, col] = rng.uniform(0.1, 0.9, size=n_samples)

    # Lexical gain: driven by bridge signal minus hub contamination
    y_lexical = np.clip(
        0.4 * bridge_signal - 0.3 * hub_signal + rng.normal(0, 0.1, size=n_samples),
        -0.5,
        0.5,
    )

    # Semantic gain: correlated but with some divergence (e.g. paraphrase tolerance)
    y_semantic = np.clip(
        0.35 * bridge_signal - 0.25 * hub_signal + rng.normal(0, 0.15, size=n_samples),
        -0.5,
        0.5,
    )

    return X, y_lexical, y_semantic


def test_topological_lightgbm_router_fit_predict_route():
    """Verify basic fit, continuous prediction, and Conservative Graph Abstention routing."""
    X, y_lexical, _ = _generate_synthetic_subgraph_features(n_samples=50, seed=123)

    router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    router.fit(X, y_lexical)

    preds = router.predict(X)
    assert preds.shape == (50,)
    assert np.all(np.isfinite(preds))

    # Test Conservative Graph Abstention across tau
    for tau in [0.0, 0.05, 0.10, 0.15, 0.20]:
        decisions = router.route(X, tau=tau)
        assert decisions.shape == (50,)
        # Invariant: decision == 1 strictly when pred >= tau
        expected_decisions = (preds >= tau).astype(int)
        np.testing.assert_array_equal(decisions, expected_decisions)


def test_conservative_graph_abstention_sweep_monotonicity_and_ternary_labels():
    """Verify that sweeping tau monotonically decreases or maintains Graph selection rate (ADR-0005) and reports ternary labels."""
    X, y_lexical, _ = _generate_synthetic_subgraph_features(n_samples=50, seed=123)
    router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    router.fit(X, y_lexical)
    preds = router.predict(X)

    tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]
    sweep_results = evaluate_conservative_abstention_sweep(
        y_pred=preds, y_true=y_lexical, tau_sweep=tau_sweep
    )

    selection_rates = [sweep_results[tau]["graph_selected_rate"] for tau in tau_sweep]

    # Monotonically non-increasing: higher tau -> more conservative -> fewer graph selections
    for i in range(len(selection_rates) - 1):
        assert selection_rates[i] >= selection_rates[i + 1]

    # Ternary label definitions: positive, negative, and neutral (User Story 17)
    for tau in tau_sweep:
        res = sweep_results[tau]
        total_ternary = res["positive_count"] + res["negative_count"] + res["neutral_count"]
        assert total_ternary == 50

    # Test monoclass ROC-AUC behavior: when all labels belong to a single class, ROC-AUC must be NaN
    monoclass_y_true = np.zeros(50, dtype=np.float32)
    monoclass_sweep = evaluate_conservative_abstention_sweep(
        y_pred=preds, y_true=monoclass_y_true, tau_sweep=[0.10]
    )
    assert math.isnan(monoclass_sweep[0.10]["roc_auc"])


def test_five_fold_in_domain_cv():
    """Verify 5-fold cross-validation computes out-of-fold Spearman rho and sweep results."""
    X, y_lexical, _ = _generate_synthetic_subgraph_features(n_samples=50, seed=42)

    cv_result: TabularCVResult = evaluate_in_domain_cv(
        X=X,
        y=y_lexical,
        n_splits=5,
        tau_sweep=[0.0, 0.05, 0.10, 0.15, 0.20],
        random_state=42,
    )

    # 1. Predictions cover all samples
    assert len(cv_result.out_of_fold_predictions) == len(y_lexical)
    assert np.all(np.isfinite(cv_result.out_of_fold_predictions))

    # 2. Planted correlation yields positive Spearman rho
    assert cv_result.spearman_rho > 0.0

    # 3. Sweep results exist for all requested tau
    for tau in [0.0, 0.05, 0.10, 0.15, 0.20]:
        assert tau in cv_result.tau_sweep_results
        assert "mean_achieved_gain" in cv_result.tau_sweep_results[tau]
        assert "graph_selected_rate" in cv_result.tau_sweep_results[tau]


def test_zero_shot_topological_transfer():
    """Verify Zero-Shot Topological Transfer between source and target datasets (GLOSSARY.md)."""
    X_source, y_source, _ = _generate_synthetic_subgraph_features(n_samples=50, seed=10)
    X_target, y_target, _ = _generate_synthetic_subgraph_features(n_samples=40, seed=20)

    source_router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    source_router.fit(X_source, y_source)

    transfer_result: TransferResult = evaluate_zero_shot_transfer(
        router=source_router,
        X_target=X_target,
        y_target=y_target,
        tau_sweep=[0.0, 0.05, 0.10, 0.15, 0.20],
    )

    assert isinstance(transfer_result, TransferResult)
    assert len(transfer_result.predictions) == 40
    # Topological signal should transfer positively across synthetic datasets
    assert transfer_result.spearman_rho > 0.0
    assert len(transfer_result.tau_sweep_results) == 5


def test_router_disagreement_matrix_and_sweep():
    """Verify computation of router disagreement matrix and sweep between Lexical and Semantic routers."""
    X, y_lexical, y_semantic = _generate_synthetic_subgraph_features(n_samples=60, seed=99)

    lexical_router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    lexical_router.fit(X, y_lexical)
    lexical_preds = lexical_router.predict(X)
    lexical_decisions = (lexical_preds >= 0.05).astype(int)

    semantic_router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    semantic_router.fit(X, y_semantic)
    semantic_preds = semantic_router.predict(X)
    semantic_decisions = (semantic_preds >= 0.05).astype(int)

    disagreement: DisagreementMatrix = compute_router_disagreement(
        lexical_decisions=lexical_decisions,
        semantic_decisions=semantic_decisions,
        tau=0.05,
    )

    assert isinstance(disagreement, DisagreementMatrix)
    assert disagreement.sample_count == 60
    assert (
        disagreement.both_flat_count
        + disagreement.both_graph_count
        + disagreement.lexical_graph_semantic_flat_count
        + disagreement.lexical_flat_semantic_graph_count
    ) == 60
    assert pytest.approx(disagreement.agreement_rate + disagreement.disagreement_rate) == 1.0

    # Sweep evaluation across tau
    disagreement_sweep = evaluate_router_disagreement_sweep(
        lexical_preds=lexical_preds,
        semantic_preds=semantic_preds,
        tau_sweep=[0.0, 0.05, 0.10, 0.15, 0.20],
    )
    assert len(disagreement_sweep) == 5
    for tau in [0.0, 0.05, 0.10, 0.15, 0.20]:
        assert tau in disagreement_sweep
        assert isinstance(disagreement_sweep[tau], DisagreementMatrix)


def test_study_report_dual_routers_and_zero_shot_transfer():
    """Verify primary seam StudyReport methods for dual router CV and zero-shot transfer."""
    embedder = FakeEmbedder()
    oracle_judge = FakeOracleGeneratorJudge()
    config = StudyConfig(retrieval_top_k=2)

    # Create minimal distinct query samples representing two benchmarks (at least 4 samples for 2-fold CV)
    passages_a = [
        {"passage_id": "pA1", "text": "FluxQubit macroscopic quantum coherence loop.", "sentences": ["FluxQubit macroscopic quantum coherence loop."]},
        {"passage_id": "pA2", "text": "JosephsonJunction enables quantum coherence.", "sentences": ["JosephsonJunction enables quantum coherence."]},
    ]
    samples_a = [
        QuerySample(
            query_id=f"qA{i}", dataset="bench_A", question_type="multi-hop",
            question=f"FluxQubit quantum coherence item {i}?", gold_answers=["yes"],
            gold_evidence_ids=["pA1", "pA2"], corpus_passages=passages_a,
        )
        for i in range(1, 5)
    ]

    report_a = run_qpp_study(samples_a, config=config, embedder=embedder, oracle_judge=oracle_judge)

    # 1. Test error handling on invalid target
    with pytest.raises(ValueError, match="Invalid target"):
        report_a.evaluate_topological_lightgbm(target="invalid_target")

    # 2. Test dual router evaluation on primary seam
    dual_result = report_a.evaluate_dual_routers(n_splits=2, tau_sweep=[0.0, 0.10])
    assert "lexical_cv" in dual_result
    assert "semantic_cv" in dual_result
    assert "disagreement_sweep" in dual_result
    assert isinstance(dual_result["lexical_cv"], TabularCVResult)
    assert isinstance(dual_result["semantic_cv"], TabularCVResult)

    # 3. Test zero-shot transfer from report A to report B
    passages_b = [
        {"passage_id": "pB1", "text": "User discussed AlphaProject quarterly budget.", "sentences": ["User discussed AlphaProject quarterly budget."]},
        {"passage_id": "pB2", "text": "Assistant booked flight travel conference.", "sentences": ["Assistant booked flight travel conference."]},
    ]
    samples_b = [
        QuerySample(
            query_id=f"qB{i}", dataset="bench_B", question_type="conversational",
            question=f"What budget was discussed in session {i}?", gold_answers=["quarterly budget"],
            gold_evidence_ids=["pB1"], corpus_passages=passages_b,
        )
        for i in range(1, 3)
    ]
    report_b = run_qpp_study(samples_b, config=config, embedder=embedder, oracle_judge=oracle_judge)

    transfer = report_a.evaluate_zero_shot_transfer(report_b, target="semantic", tau_sweep=[0.0, 0.10])
    assert isinstance(transfer, TransferResult)
    assert len(transfer.predictions) == 2
