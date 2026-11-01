"""Domain-Agnostic GATv2 GNN Post-Retrieval Context Selector (ADR-0003, ADR-0005, ADR-0006, Issue #5)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.utils import softmax

from src.graph.ppr_retriever import CandidateSubgraphData
from src.qpp.tabular_router import (
    TabularCVResult,
    TransferResult,
    _compute_gain_regression_metrics,
    evaluate_conservative_abstention_sweep,
)

# Domain-agnostic feature indices (ADR-0003)
COL_QUERY_SIM: int = 0


def candidate_subgraph_to_pyg_data(
    candidate_subgraph: CandidateSubgraphData,
    label: float | None = None,
) -> Data:
    """Convert CandidateSubgraphData into PyTorch Geometric Data strictly using domain-agnostic scalar features."""
    x = torch.from_numpy(candidate_subgraph.node_features.astype(np.float32))
    edge_index = torch.from_numpy(candidate_subgraph.edge_index.astype(np.int64))

    y_tensor = (
        torch.tensor([label], dtype=torch.float32)
        if label is not None
        else None
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=y_tensor,
        num_nodes=candidate_subgraph.num_nodes,
    )


class QuerySimilarityAttentionReadout(nn.Module):
    """Global graph readout weighted by query-similarity conditioned attention (Issue #1 User Story 16)."""

    def __init__(self, in_features: int, hidden_features: int = 16) -> None:
        super().__init__()
        # Takes node representation concatenated with query-similarity scalar
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_features + 1, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, h: torch.Tensor, query_sim: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.int64, device=h.device)

        gate_input = torch.cat([h, query_sim], dim=-1)
        gate_scores = self.gate_mlp(gate_input)
        weights = softmax(gate_scores, batch)
        pooled = global_add_pool(weights * h, batch)
        return pooled


class GATv2ContextSelector(nn.Module):
    """2-layer GATv2 Graph Attention Network with query-similarity attention readout (ADR-0003, Issue #5)."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 16,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
        )
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
        )
        self.act2 = nn.ELU()

        self.readout = QuerySimilarityAttentionReadout(
            in_features=hidden_channels,
            hidden_features=hidden_channels,
        )

        self.regressor_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Layer 1 message passing
        h1 = self.drop1(self.act1(self.conv1(x, edge_index)))

        # Layer 2 message passing
        h2 = self.act2(self.conv2(h1, edge_index))

        # Query-similarity attention readout
        query_sim = x[:, COL_QUERY_SIM : COL_QUERY_SIM + 1]
        graph_embed = self.readout(h2, query_sim, batch)

        # Predict continuous GraphRAG Gain
        pred = self.regressor_head(graph_embed).squeeze(-1)
        return pred


class TopologicalGNNRouter:
    """Wrapper managing training and routing for GATv2 Context Selector under Conservative Graph Abstention."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 16,
        heads: int = 4,
        dropout: float = 0.1,
        epochs: int = 20,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device(device)

        self._set_seed(random_state)
        self.model = GATv2ContextSelector(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            dropout=self.dropout,
        ).to(self.device)
        self.is_fitted = False

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def fit(self, graphs: list[Data], targets: np.ndarray | None = None) -> TopologicalGNNRouter:
        """Fit GATv2 regressor on training graphs to predict continuous GraphRAG Gain."""
        self._set_seed(self.random_state)
        self.model.train()

        train_data_list: list[Data] = []
        for i, g in enumerate(graphs):
            data_copy = g.clone()
            if targets is not None:
                data_copy.y = torch.tensor([targets[i]], dtype=torch.float32)
            train_data_list.append(data_copy)

        loader = DataLoader(train_data_list, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(preds, batch.y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict(self, graphs: list[Data]) -> np.ndarray:
        """Predict continuous GraphRAG Gain (y_hat_q) for Candidate Subgraphs."""
        if not self.is_fitted:
            raise RuntimeError("TopologicalGNNRouter must be fitted before predict.")

        self.model.eval()
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        all_preds: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model(batch.x, batch.edge_index, batch.batch)
                all_preds.append(preds.cpu().numpy().reshape(-1))

        if not all_preds:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(all_preds, axis=0).astype(np.float32)

    def route(self, graphs: list[Data], tau: float = 0.10) -> np.ndarray:
        """Apply Conservative Graph Abstention: choose Graph if y_hat >= tau, else Flat (ADR-0005)."""
        preds = self.predict(graphs)
        return (preds >= tau).astype(int)


def evaluate_gnn_in_domain_cv(
    graphs: list[Data],
    targets: np.ndarray,
    n_splits: int = 5,
    tau_sweep: list[float] | None = None,
    epochs: int = 20,
    lr: float = 0.01,
    random_state: int = 42,
    router_kwargs: dict[str, Any] | None = None,
) -> TabularCVResult:
    """Evaluate 5-fold in-domain cross-validation of GATv2 Post-Retrieval Context Selector."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]
    if router_kwargs is None:
        router_kwargs = {
            "in_channels": 12,
            "hidden_channels": 16,
            "heads": 2,
            "epochs": epochs,
            "lr": lr,
            "random_state": random_state,
        }

    targets_arr = np.asarray(targets, dtype=np.float32)
    n = len(graphs)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(n, dtype=np.float32)

    for train_idx, val_idx in kf.split(graphs):
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        train_y = targets_arr[train_idx]

        router = TopologicalGNNRouter(**router_kwargs)
        router.fit(train_graphs, train_y)
        oof_predictions[val_idx] = router.predict(val_graphs)

    rho, pval, mse, mae = _compute_gain_regression_metrics(targets_arr, oof_predictions)

    sweep_results = evaluate_conservative_abstention_sweep(
        y_pred=oof_predictions,
        y_true=targets_arr,
        tau_sweep=tau_sweep,
    )

    return TabularCVResult(
        out_of_fold_predictions=oof_predictions,
        true_gains=targets_arr,
        spearman_rho=rho,
        spearman_pvalue=pval,
        mse=mse,
        mae=mae,
        tau_sweep_results=sweep_results,
    )


def evaluate_gnn_zero_shot_transfer(
    router: TopologicalGNNRouter,
    graphs_target: list[Data],
    targets_target: np.ndarray,
    tau_sweep: list[float] | None = None,
) -> TransferResult:
    """Evaluate Zero-Shot Topological Transfer of GATv2 router on an unseen benchmark (GLOSSARY.md)."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]

    y_pred_arr = router.predict(graphs_target)
    y_true_arr = np.asarray(targets_target, dtype=np.float32)

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
