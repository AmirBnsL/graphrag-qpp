"""Topological LightGBM Post-Retrieval Context Selector (ADR-0002, ADR-0005, ADR-0006, Issue #4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold


@dataclass
class TabularCVResult:
    """Consolidated results of in-domain cross-validation for a Post-Retrieval Context Selector."""

    out_of_fold_predictions: np.ndarray
    true_gains: np.ndarray
    spearman_rho: float
    spearman_pvalue: float
    mse: float
    mae: float
    tau_sweep_results: dict[float, dict[str, float]] = field(default_factory=dict)


class TopologicalLightGBMRouter:
    """LightGBM tabular Post-Retrieval Context Selector trained on Candidate Subgraph geometry."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 42,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        feature_names: list[str] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.feature_names = feature_names

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            verbose=-1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> TopologicalLightGBMRouter:
        """Fit the regressor to predict continuous GraphRAG Gain (Lexical or Semantic)."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous GraphRAG Gain (y_hat_q)."""
        if not self.is_fitted:
            raise RuntimeError("TopologicalLightGBMRouter must be fitted before predict.")
        X_arr = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=np.float32)

    def route(self, X: np.ndarray, tau: float = 0.10) -> np.ndarray:
        """Apply Conservative Graph Abstention: choose Graph if y_hat >= tau, else Flat (ADR-0005).

        Returns:
            Binary array where 1 indicates Graph RAG passages (C_graph), and 0 indicates Flat RAG (C_flat).
        """
        preds = self.predict(X)
        return (preds >= tau).astype(int)

    def get_feature_importances(self) -> dict[str, float]:
        """Return feature importance map."""
        if not self.is_fitted:
            raise RuntimeError("Router must be fitted to get feature importances.")
        importances = self.model.feature_importances_
        if self.feature_names and len(self.feature_names) == len(importances):
            return {name: float(imp) for name, imp in zip(self.feature_names, importances)}
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}


def evaluate_conservative_abstention_sweep(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    tau_sweep: list[float] | None = None,
) -> dict[float, dict[str, float]]:
    """Evaluate Conservative Graph Abstention across Gain Margin Threshold sweep tau (ADR-0002, ADR-0005)."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]

    y_p = np.asarray(y_pred, dtype=np.float32)
    y_t = np.asarray(y_true, dtype=np.float32)
    n = len(y_p)
    if n == 0:
        return {}

    results: dict[float, dict[str, float]] = {}

    for tau in tau_sweep:
        decisions = (y_p >= tau).astype(int)
        binary_ground_truth = (y_t >= tau).astype(int)

        graph_count = int(np.sum(decisions == 1))
        flat_count = int(np.sum(decisions == 0))
        graph_rate = float(graph_count / n)

        # Achieved gain under Conservative Graph Abstention:
        # If Graph selected: achieve y_true. If Flat selected: achieve 0.0 (baseline fallback).
        achieved_gains = np.where(decisions == 1, y_t, 0.0)
        mean_achieved_gain = float(np.mean(achieved_gains))

        # Oracle gain: select Graph whenever y_true >= tau
        oracle_gains = np.where(y_t >= tau, y_t, 0.0)
        mean_oracle_gain = float(np.mean(oracle_gains))

        mean_flat_gain = 0.0
        mean_graph_gain = float(np.mean(y_t))
        accuracy = float(np.mean(decisions == binary_ground_truth))

        # ROC-AUC if both positive and negative classes exist
        if len(np.unique(binary_ground_truth)) > 1:
            try:
                auc = float(roc_auc_score(binary_ground_truth, y_p))
            except Exception:
                auc = 0.5
        else:
            auc = 1.0

        results[tau] = {
            "tau": tau,
            "graph_selected_count": graph_count,
            "flat_selected_count": flat_count,
            "graph_selected_rate": graph_rate,
            "mean_achieved_gain": mean_achieved_gain,
            "mean_oracle_gain": mean_oracle_gain,
            "mean_flat_gain": mean_flat_gain,
            "mean_graph_gain": mean_graph_gain,
            "accuracy": accuracy,
            "roc_auc": auc,
        }

    return results


def evaluate_in_domain_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    tau_sweep: list[float] | None = None,
    random_state: int = 42,
    router_kwargs: dict[str, Any] | None = None,
) -> TabularCVResult:
    """Evaluate 5-fold in-domain cross-validation of Topological LightGBM Post-Retrieval Context Selector."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]
    if router_kwargs is None:
        router_kwargs = {"n_estimators": 50, "max_depth": 3, "random_state": random_state}

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    n = len(X_arr)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(n, dtype=np.float32)

    for train_idx, val_idx in kf.split(X_arr):
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_val = X_arr[val_idx]

        router = TopologicalLightGBMRouter(**router_kwargs)
        router.fit(X_train, y_train)
        oof_predictions[val_idx] = router.predict(X_val)

    # Compute correlation and error metrics
    rho, pval = stats.spearmanr(oof_predictions, y_arr)
    if np.isnan(rho):
        rho, pval = 0.0, 1.0

    mse = float(mean_squared_error(y_arr, oof_predictions))
    mae = float(mean_absolute_error(y_arr, oof_predictions))

    sweep_results = evaluate_conservative_abstention_sweep(
        y_pred=oof_predictions,
        y_true=y_arr,
        tau_sweep=tau_sweep,
    )

    return TabularCVResult(
        out_of_fold_predictions=oof_predictions,
        true_gains=y_arr,
        spearman_rho=float(rho),
        spearman_pvalue=float(pval),
        mse=mse,
        mae=mae,
        tau_sweep_results=sweep_results,
    )


def evaluate_zero_shot_transfer(
    router: TopologicalLightGBMRouter,
    X_target: np.ndarray,
    y_target: np.ndarray,
    tau_sweep: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate Zero-Shot Topological Transfer on an unseen benchmark (GLOSSARY.md, Issue #4)."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]

    y_p = router.predict(X_target)
    y_t = np.asarray(y_target, dtype=np.float32)

    rho, pval = stats.spearmanr(y_p, y_t)
    if np.isnan(rho):
        rho, pval = 0.0, 1.0

    mse = float(mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))

    sweep_results = evaluate_conservative_abstention_sweep(
        y_pred=y_p,
        y_true=y_t,
        tau_sweep=tau_sweep,
    )

    return {
        "predictions": y_p,
        "true_gains": y_t,
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "mse": mse,
        "mae": mae,
        "tau_sweep_results": sweep_results,
    }


def compute_router_disagreement(
    lexical_decisions: np.ndarray,
    semantic_decisions: np.ndarray,
) -> dict[str, Any]:
    """Compute router disagreement matrix and Cohen's kappa between Lexical and Semantic routers."""
    lex = np.asarray(lexical_decisions, dtype=int)
    sem = np.asarray(semantic_decisions, dtype=int)
    n = len(lex)
    if n == 0 or len(sem) != n:
        raise ValueError("Decisions arrays must be non-empty and equal in length.")

    both_flat = int(np.sum((lex == 0) & (sem == 0)))
    both_graph = int(np.sum((lex == 1) & (sem == 1)))
    lex_graph_sem_flat = int(np.sum((lex == 1) & (sem == 0)))
    lex_flat_sem_graph = int(np.sum((lex == 0) & (sem == 1)))

    agreement_count = both_flat + both_graph
    agreement_rate = float(agreement_count / n)
    disagreement_rate = float(1.0 - agreement_rate)

    try:
        kappa = float(cohen_kappa_score(lex, sem))
        if np.isnan(kappa):
            kappa = 1.0 if agreement_rate == 1.0 else 0.0
    except Exception:
        kappa = 1.0 if agreement_rate == 1.0 else 0.0

    return {
        "both_flat_count": both_flat,
        "both_graph_count": both_graph,
        "lexical_graph_semantic_flat_count": lex_graph_sem_flat,
        "lexical_flat_semantic_graph_count": lex_flat_sem_graph,
        "agreement_count": agreement_count,
        "disagreement_count": n - agreement_count,
        "agreement_rate": agreement_rate,
        "disagreement_rate": disagreement_rate,
        "cohen_kappa": kappa,
        "sample_count": n,
    }
