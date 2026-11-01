"""Topological LightGBM Post-Retrieval Context Selector (ADR-0002, ADR-0005, ADR-0006, Issue #4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold


@dataclass
class DisagreementMatrix:
    """Pairwise decision cross-tabulation and inter-router agreement metrics (Issue #4)."""

    tau: float
    both_flat_count: int
    both_graph_count: int
    lexical_graph_semantic_flat_count: int
    lexical_flat_semantic_graph_count: int
    agreement_count: int
    disagreement_count: int
    agreement_rate: float
    disagreement_rate: float
    cohen_kappa: float
    sample_count: int


@dataclass
class TransferResult:
    """Consolidated evaluation of Zero-Shot Topological Transfer on an unseen benchmark (GLOSSARY.md)."""

    predictions: np.ndarray
    true_gains: np.ndarray
    spearman_rho: float
    spearman_pvalue: float
    mse: float
    mae: float
    tau_sweep_results: dict[float, dict[str, float]] = field(default_factory=dict)


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

    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    n = len(y_pred_arr)
    if n == 0:
        return {}

    results: dict[float, dict[str, float]] = {}

    for tau in tau_sweep:
        decisions = (y_pred_arr >= tau).astype(int)
        binary_ground_truth = (y_true_arr >= tau).astype(int)

        graph_count = int(np.sum(decisions == 1))
        flat_count = int(np.sum(decisions == 0))
        graph_rate = float(graph_count / n)

        # Ternary label counts (User Story 17, GLOSSARY.md)
        positive_count = int(np.sum(y_true_arr >= tau))
        negative_count = int(np.sum(y_true_arr <= -tau))
        neutral_count = int(np.sum(np.abs(y_true_arr) < tau))

        # Achieved gain under Conservative Graph Abstention:
        # If Graph selected: achieve y_true. If Flat selected: achieve 0.0 (baseline default selection).
        achieved_gains = np.where(decisions == 1, y_true_arr, 0.0)
        mean_achieved_gain = float(np.mean(achieved_gains))

        # Oracle gain: select Graph whenever y_true >= tau
        oracle_gains = np.where(y_true_arr >= tau, y_true_arr, 0.0)
        mean_oracle_gain = float(np.mean(oracle_gains))

        mean_flat_gain = 0.0
        mean_graph_gain = float(np.mean(y_true_arr))
        accuracy = float(np.mean(decisions == binary_ground_truth))

        # ROC-AUC if both positive and negative classes exist in ground truth
        if len(np.unique(binary_ground_truth)) > 1:
            try:
                auc = float(roc_auc_score(binary_ground_truth, y_pred_arr))
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")

        results[tau] = {
            "tau": tau,
            "graph_selected_count": float(graph_count),
            "flat_selected_count": float(flat_count),
            "graph_selected_rate": graph_rate,
            "positive_count": float(positive_count),
            "negative_count": float(negative_count),
            "neutral_count": float(neutral_count),
            "mean_achieved_gain": mean_achieved_gain,
            "mean_oracle_gain": mean_oracle_gain,
            "mean_flat_gain": mean_flat_gain,
            "mean_graph_gain": mean_graph_gain,
            "accuracy": accuracy,
            "roc_auc": auc,
        }

    return results


def _compute_gain_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute Spearman rho, p-value, MSE, and MAE between true and predicted gains."""
    # Handle constant arrays cleanly without raising scipy warnings
    if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        rho, pval = 0.0, 1.0
    else:
        rho, pval = stats.spearmanr(y_pred, y_true)
        if np.isnan(rho):
            rho, pval = 0.0, 1.0

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return float(rho), float(pval), mse, mae


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

    rho, pval, mse, mae = _compute_gain_regression_metrics(y_arr, oof_predictions)

    sweep_results = evaluate_conservative_abstention_sweep(
        y_pred=oof_predictions,
        y_true=y_arr,
        tau_sweep=tau_sweep,
    )

    return TabularCVResult(
        out_of_fold_predictions=oof_predictions,
        true_gains=y_arr,
        spearman_rho=rho,
        spearman_pvalue=pval,
        mse=mse,
        mae=mae,
        tau_sweep_results=sweep_results,
    )


def evaluate_zero_shot_transfer(
    router: TopologicalLightGBMRouter,
    X_target: np.ndarray,
    y_target: np.ndarray,
    tau_sweep: list[float] | None = None,
) -> TransferResult:
    """Evaluate Zero-Shot Topological Transfer on an unseen benchmark (GLOSSARY.md, Issue #4)."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]

    y_pred_arr = router.predict(X_target)
    y_true_arr = np.asarray(y_target, dtype=np.float32)

    rho, pval, mse, mae = _compute_gain_regression_metrics(y_true_arr, y_pred_arr)

    sweep_results = evaluate_conservative_abstention_sweep(
        y_pred=y_pred_arr,
        y_true=y_true_arr,
        tau_sweep=tau_sweep,
    )

    return TransferResult(
        predictions=y_pred_arr,
        true_gains=y_true_arr,
        spearman_rho=rho,
        spearman_pvalue=pval,
        mse=mse,
        mae=mae,
        tau_sweep_results=sweep_results,
    )


def compute_router_disagreement(
    lexical_decisions: np.ndarray,
    semantic_decisions: np.ndarray,
    tau: float = 0.10,
) -> DisagreementMatrix:
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kappa = float(cohen_kappa_score(lex, sem, labels=[0, 1]))
            if np.isnan(kappa):
                kappa = 1.0 if agreement_rate == 1.0 else 0.0
        except Exception:
            kappa = 1.0 if agreement_rate == 1.0 else 0.0

    return DisagreementMatrix(
        tau=tau,
        both_flat_count=both_flat,
        both_graph_count=both_graph,
        lexical_graph_semantic_flat_count=lex_graph_sem_flat,
        lexical_flat_semantic_graph_count=lex_flat_sem_graph,
        agreement_count=agreement_count,
        disagreement_count=n - agreement_count,
        agreement_rate=agreement_rate,
        disagreement_rate=disagreement_rate,
        cohen_kappa=kappa,
        sample_count=n,
    )


def evaluate_router_disagreement_sweep(
    lexical_preds: np.ndarray,
    semantic_preds: np.ndarray,
    tau_sweep: list[float] | None = None,
) -> dict[float, DisagreementMatrix]:
    """Evaluate router disagreement matrix across Gain Margin Threshold sweep tau."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]

    lex_p = np.asarray(lexical_preds, dtype=np.float32)
    sem_p = np.asarray(semantic_preds, dtype=np.float32)

    sweep_results: dict[float, DisagreementMatrix] = {}
    for tau in tau_sweep:
        lex_decisions = (lex_p >= tau).astype(int)
        sem_decisions = (sem_p >= tau).astype(int)
        sweep_results[tau] = compute_router_disagreement(
            lexical_decisions=lex_decisions,
            semantic_decisions=sem_decisions,
            tau=tau,
        )

    return sweep_results
