"""Publication-ready LaTeX table exporters for GraphRAG-QPP Study."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.qpp.tabular_router import DisagreementMatrix


def _write_latex_file(tex_str: str, filepath: str | Path | None) -> str:
    """Helper to persist LaTeX string to disk if path is provided."""
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tex_str, encoding="utf-8")
    return tex_str


def export_qpp_correlation_table_latex(
    metrics_by_model: dict[str, dict[str, Any]],
    filepath: str | Path | None = None,
) -> str:
    """Export formatted LaTeX booktabs table of QPP predictor correlations (ADR-0006)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Post-Retrieval GraphRAG Gain Prediction Performance on Multi-Hop and Conversational Benchmarks.}",
        r"\label{tab:qpp_correlations}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Predictor / Router} & \textbf{Type} & \textbf{Spearman $\rho$} & \textbf{$p$-value} & \textbf{MSE} & \textbf{MAE} \\",
        r"\midrule",
    ]

    for name, m in metrics_by_model.items():
        m_type = m.get("type", "Topological")
        rho = m.get("spearman_rho", 0.0)
        pval = m.get("spearman_pvalue", 1.0)
        mse = m.get("mse", 0.0)
        mae = m.get("mae", 0.0)

        pval_str = "< 0.001" if pval < 0.001 else f"{pval:.3f}"
        lines.append(
            f"{name} & {m_type} & {rho:+.3f} & {pval_str} & {mse:.3f} & {mae:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(lines)
    return _write_latex_file(tex_str, filepath)


def export_conservative_abstention_table_latex(
    sweep_results: dict[float, dict[str, Any]],
    filepath: str | Path | None = None,
) -> str:
    """Export formatted LaTeX table for Conservative Graph Abstention sweep across tau (ADR-0005)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Conservative Graph Abstention Performance Across Gain Margin Thresholds $\tau$.}",
        r"\label{tab:conservative_abstention}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"\textbf{$\tau$} & \textbf{Graph Selected (\%)} & \textbf{Mean Achieved Gain} & \textbf{Oracle Gain} & \textbf{Accuracy} & \textbf{ROC-AUC} \\",
        r"\midrule",
    ]

    for tau, res in sorted(sweep_results.items(), key=lambda x: x[0]):
        rate = res.get("graph_selected_rate", 0.0) * 100.0
        achieved = res.get("mean_achieved_gain", 0.0)
        oracle = res.get("mean_oracle_gain", 0.0)
        acc = res.get("accuracy", 0.0)
        auc = res.get("roc_auc", 0.5)

        lines.append(
            f"{tau:.2f} & {rate:5.1f}\\% & {achieved:+.4f} & {oracle:+.4f} & {acc:.3f} & {auc:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(lines)
    return _write_latex_file(tex_str, filepath)


def export_hub_cleanup_ablation_table_latex(
    stages_data: list[dict[str, Any]],
    filepath: str | Path | None = None,
) -> str:
    """Export formatted LaTeX table for 4-Stage Hub Contamination Cleanup Ablation."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-Session Hub Contamination Regularization Ablation Across 4 Cleanup Stages.}",
        r"\label{tab:hub_cleanup_ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Cleanup Stage} & \textbf{Hub Access Gini $\downarrow$} & \textbf{Precision@5 $\uparrow$} & \textbf{Recall@5 $\uparrow$} & \textbf{PPR Entropy $\downarrow$} \\",
        r"\midrule",
    ]

    for row in stages_data:
        stage = row.get("stage", "")
        gini = row.get("hub_access_gini", 0.0)
        precision_5 = row.get("precision_at_5", 0.0)
        recall_5 = row.get("recall_at_5", 0.0)
        ppr_entropy = row.get("ppr_entropy", 0.0)

        lines.append(
            f"{stage} & {gini:.3f} & {precision_5:.3f} & {recall_5:.3f} & {ppr_entropy:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(lines)
    return _write_latex_file(tex_str, filepath)


def export_router_disagreement_table_latex(
    disagreement_sweep: dict[float, DisagreementMatrix],
    filepath: str | Path | None = None,
) -> str:
    """Export formatted LaTeX table comparing Lexical vs Semantic Router decisions (ADR-0002)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Lexical Router vs. Semantic Router Routing Disagreement Across $\tau$.}",
        r"\label{tab:router_disagreement}",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"\textbf{$\tau$} & \textbf{Agree (\%)} & \textbf{Disagree (\%)} & \textbf{Both Flat} & \textbf{Both Graph} & \textbf{Lex Only} & \textbf{Sem Only} & \textbf{Cohen's $\kappa$} \\",
        r"\midrule",
    ]

    for tau, mat in sorted(disagreement_sweep.items(), key=lambda x: x[0]):
        agreement_pct = mat.agreement_rate * 100.0
        disagreement_pct = mat.disagreement_rate * 100.0
        both_flat = mat.both_flat_count
        both_graph = mat.both_graph_count
        lexical_only = mat.lexical_graph_semantic_flat_count
        semantic_only = mat.lexical_flat_semantic_graph_count
        cohen_kappa = mat.cohen_kappa

        lines.append(
            f"{tau:.2f} & {agreement_pct:5.1f}\\% & {disagreement_pct:5.1f}\\% & "
            f"{both_flat} & {both_graph} & {lexical_only} & {semantic_only} & {cohen_kappa:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(lines)
    return _write_latex_file(tex_str, filepath)


def export_zero_shot_transfer_table_latex(
    transfer_results_map: dict[str, dict[str, Any]],
    filepath: str | Path | None = None,
) -> str:
    """Export formatted LaTeX table for Zero-Shot Topological Transfer (GLOSSARY.md)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Zero-Shot Topological Transfer Across Multi-Hop QA and Conversational Memory.}",
        r"\label{tab:zero_shot_transfer}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Source Corpus} & \textbf{Target Corpus} & \textbf{Setting} & \textbf{Spearman $\rho$} & \textbf{Mean Achieved Gain ($\tau=0.10$)} \\",
        r"\midrule",
    ]

    for key, data in transfer_results_map.items():
        src = data.get("source", "")
        tgt = data.get("target", "")
        setting = data.get("setting", "Zero-Shot")
        rho = data.get("spearman_rho", 0.0)
        gain = data.get("mean_achieved_gain", 0.0)

        lines.append(
            f"{src} & {tgt} & {setting} & {rho:+.3f} & {gain:+.4f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(lines)
    return _write_latex_file(tex_str, filepath)
