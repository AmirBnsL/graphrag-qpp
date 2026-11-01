"""QPP module containing Classical QPP, Tabular LightGBM, and GATv2 GNN Post-Retrieval Context Selectors."""

from src.qpp.classical import compute_classical_qpp
from src.qpp.gnn_router import (
    GATv2ContextSelector,
    TopologicalGNNRouter,
    candidate_subgraph_to_pyg_data,
    evaluate_gnn_in_domain_cv,
    evaluate_gnn_zero_shot_transfer,
)
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
from src.qpp.topological_features import extract_topological_features

__all__ = [
    "compute_classical_qpp",
    "extract_topological_features",
    "TopologicalLightGBMRouter",
    "TabularCVResult",
    "TransferResult",
    "DisagreementMatrix",
    "evaluate_conservative_abstention_sweep",
    "evaluate_in_domain_cv",
    "evaluate_zero_shot_transfer",
    "compute_router_disagreement",
    "evaluate_router_disagreement_sweep",
    "candidate_subgraph_to_pyg_data",
    "GATv2ContextSelector",
    "TopologicalGNNRouter",
    "evaluate_gnn_in_domain_cv",
    "evaluate_gnn_zero_shot_transfer",
]
