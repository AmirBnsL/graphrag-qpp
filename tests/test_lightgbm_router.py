"""Unit, invariant, and cross-validation tests for Topological LightGBM Router (Issue #4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.qpp.tabular_router import (
    TabularCVResult,
    TopologicalLightGBMRouter,
    compute_router_disagreement,
    evaluate_conservative_abstention_sweep,
    evaluate_in_domain_cv,
    evaluate_zero_shot_transfer,
)


def _generate_synthetic_subgraph_features(n_samples: int = 40, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    y_lexical = np.clip(0.4 * bridge_signal - 0.3 * hub_signal + rng.normal(0, 0.1, size=n_samples), -0.5, 0.5)

    # Semantic gain: correlated but with some divergence (e.g. paraphrase tolerance)
    y_semantic = np.clip(0.35 * bridge_signal - 0.25 * hub_signal + rng.normal(0, 0.15, size=n_samples), -0.5, 0.5)

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


def test_conservative_graph_abstention_sweep_monotonicity():
    """Verify that sweeping tau monotonically decreases or maintains Graph selection rate (ADR-0005)."""
    X, y_lexical, _ = _generate_synthetic_subgraph_features(n_samples=50, seed=123)
    router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    router.fit(X, y_lexical)
    preds = router.predict(X)

    tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]
    sweep_results = evaluate_conservative_abstention_sweep(y_pred=preds, y_true=y_lexical, tau_sweep=tau_sweep)

    selection_rates = [sweep_results[tau]["graph_selected_rate"] for tau in tau_sweep]

    # Monotonically non-increasing: higher tau -> more conservative -> fewer graph selections
    for i in range(len(selection_rates) - 1):
        assert selection_rates[i] >= selection_rates[i + 1]


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

    transfer_result = evaluate_zero_shot_transfer(
        router=source_router,
        X_target=X_target,
        y_target=y_target,
        tau_sweep=[0.0, 0.05, 0.10, 0.15, 0.20],
    )

    assert "spearman_rho" in transfer_result
    assert "tau_sweep_results" in transfer_result
    assert len(transfer_result["predictions"]) == 40
    # Topological signal should transfer positively across synthetic datasets
    assert transfer_result["spearman_rho"] > 0.0


def test_router_disagreement_matrix():
    """Verify computation of router disagreement matrix between Lexical and Semantic routers."""
    X, y_lexical, y_semantic = _generate_synthetic_subgraph_features(n_samples=60, seed=99)

    lexical_router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    lexical_router.fit(X, y_lexical)
    lexical_decisions = lexical_router.route(X, tau=0.05)

    semantic_router = TopologicalLightGBMRouter(n_estimators=30, max_depth=3, random_state=42)
    semantic_router.fit(X, y_semantic)
    semantic_decisions = semantic_router.route(X, tau=0.05)

    disagreement = compute_router_disagreement(
        lexical_decisions=lexical_decisions,
        semantic_decisions=semantic_decisions,
    )

    assert "both_flat_count" in disagreement
    assert "both_graph_count" in disagreement
    assert "lexical_graph_semantic_flat_count" in disagreement
    assert "lexical_flat_semantic_graph_count" in disagreement
    assert "agreement_rate" in disagreement
    assert "disagreement_rate" in disagreement
    assert "cohen_kappa" in disagreement

    total = (
        disagreement["both_flat_count"]
        + disagreement["both_graph_count"]
        + disagreement["lexical_graph_semantic_flat_count"]
        + disagreement["lexical_flat_semantic_graph_count"]
    )
    assert total == 60
    assert pytest.approx(disagreement["agreement_rate"] + disagreement["disagreement_rate"]) == 1.0
