"""Evaluation, metrics, LaTeX reporting, and figure plotting module for GraphRAG-QPP."""

from src.eval.figures import (
    plot_conservative_abstention_sweep,
    plot_hub_cleanup_ablation,
    plot_router_disagreement_heatmap,
    plot_zero_shot_transfer_comparison,
)
from src.eval.reporting import (
    export_conservative_abstention_table_latex,
    export_hub_cleanup_ablation_table_latex,
    export_qpp_correlation_table_latex,
    export_router_disagreement_table_latex,
    export_zero_shot_transfer_table_latex,
)
from src.eval.retrieval_metrics import (
    compute_access_gini,
    compute_ppr_entropy,
    compute_ppr_entropy_drift,
    compute_precision_at_k,
    compute_recall_at_k,
)

__all__ = [
    "compute_access_gini",
    "compute_ppr_entropy",
    "compute_ppr_entropy_drift",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "export_qpp_correlation_table_latex",
    "export_conservative_abstention_table_latex",
    "export_hub_cleanup_ablation_table_latex",
    "export_router_disagreement_table_latex",
    "export_zero_shot_transfer_table_latex",
    "plot_conservative_abstention_sweep",
    "plot_hub_cleanup_ablation",
    "plot_router_disagreement_heatmap",
    "plot_zero_shot_transfer_comparison",
]
