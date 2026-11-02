"""Publication-ready diagnostic figure generation (PDF and PNG) using Matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.qpp.tabular_router import DisagreementMatrix


def plot_conservative_abstention_sweep(
    sweep_results: dict[float, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot Mean Achieved Gain and Graph Selection Rate vs Gain Margin Threshold tau (ADR-0005)."""
    taus = sorted(sweep_results.keys())
    achieved = [sweep_results[t].get("mean_achieved_gain", 0.0) for t in taus]
    oracle = [sweep_results[t].get("mean_oracle_gain", 0.0) for t in taus]
    rates = [sweep_results[t].get("graph_selected_rate", 0.0) * 100.0 for t in taus]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

    color_gain = "#1f77b4"
    ax1.set_xlabel(r"Gain Margin Threshold ($\tau$)", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Mean Achieved GraphRAG Gain", color=color_gain, fontsize=11, fontweight="medium")
    l1 = ax1.plot(taus, achieved, "o-", color=color_gain, linewidth=2.2, label=r"Achieved Gain ($\hat{y} \geq \tau$)")
    l2 = ax1.plot(taus, oracle, "--", color="#2ca02c", linewidth=1.8, label="Oracle Context Selector")
    l3 = ax1.axhline(0.0, color="#d62728", linestyle=":", linewidth=1.5, label="Flat Dense Baseline")
    ax1.tick_params(axis="y", labelcolor=color_gain)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_rate = "#9467bd"
    ax2.set_ylabel("Graph Selection Rate (%)", color=color_rate, fontsize=11, fontweight="medium")
    l4 = ax2.plot(taus, rates, "s--", color=color_rate, linewidth=1.8, label="Graph Selected (%)")
    ax2.tick_params(axis="y", labelcolor=color_rate)
    ax2.set_ylim(-5, 105)

    lines = l1 + l2 + [l3] + l4
    labels: list[str] = [str(line.get_label()) for line in lines]
    ax1.legend(handles=lines, labels=labels, loc="upper right", framealpha=0.9, fontsize=9)

    plt.title("Conservative Graph Abstention Sensitivity", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_hub_cleanup_ablation(
    stages_data: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot Hub Access Gini suppression and Precision@5 across the 4 cleanup stages."""
    raw_stages = [s.get("stage", f"Step {i}") for i, s in enumerate(stages_data)]
    stages = [st.replace(" (", "\n(") for st in raw_stages]
    ginis = [s.get("hub_access_gini", 0.0) for s in stages_data]
    precisions = [s.get("precision_at_5", 0.0) for s in stages_data]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    color_gini = "#d62728"
    ax1.set_xlabel("Cross-Session Contamination Cleanup Stage", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Hub Access Gini Coefficient (lower = less hub drift)", color=color_gini, fontsize=11)
    l1 = ax1.plot(stages, ginis, "o-", color=color_gini, linewidth=2.2, label="Hub Access Gini (lower is better)")
    ax1.tick_params(axis="y", labelcolor=color_gini)
    ax1.tick_params(axis="x", labelsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_p5 = "#1f77b4"
    ax2.set_ylabel("Retrieval Precision@5 (higher = better)", color=color_p5, fontsize=11)
    l2 = ax2.plot(stages, precisions, "s-", color=color_p5, linewidth=2.2, label="Precision@5 (higher is better)")
    ax2.tick_params(axis="y", labelcolor=color_p5)

    lines = l1 + l2
    labels: list[str] = [str(l.get_label()) for l in lines]
    ax1.legend(handles=lines, labels=labels, loc="center right", framealpha=0.9, fontsize=9)

    plt.title("4-Stage Cross-Session Hub Cleanup Ablation", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_router_disagreement_heatmap(
    matrix: DisagreementMatrix,
    output_path: str | Path,
) -> Path:
    """Plot 2x2 confusion heatmap comparing Lexical Router and Semantic Router decisions."""
    data = np.array([
        [matrix.both_flat_count, matrix.lexical_flat_semantic_graph_count],
        [matrix.lexical_graph_semantic_flat_count, matrix.both_graph_count],
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    cax = ax.imshow(data, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Flat (0)", "Graph (1)"], fontsize=10)
    ax.set_yticklabels(["Flat (0)", "Graph (1)"], fontsize=10)
    ax.set_xlabel("Semantic Router Decision", fontsize=11, fontweight="medium")
    ax.set_ylabel("Lexical Router Decision", fontsize=11, fontweight="medium")

    # Annotate counts and percentages
    total = max(1, matrix.sample_count)
    for i in range(2):
        for j in range(2):
            val = data[i, j]
            pct = val / total * 100.0
            color = "white" if val > (matrix.sample_count * 0.4) else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold")

    plt.title(
        f"Router Disagreement (tau={matrix.tau:.2f}, kappa={matrix.cohen_kappa:.3f})\n"
        f"Agreement: {matrix.agreement_rate*100:.1f}% | Disagreement: {matrix.disagreement_rate*100:.1f}%",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_zero_shot_transfer_comparison(
    transfer_data: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot grouped bar chart comparing in-domain CV vs zero-shot transfer Spearman rho."""
    labels = [d.get("name", f"Benchmark {i}") for i, d in enumerate(transfer_data)]
    in_domain = [d.get("in_domain_rho", 0.0) for d in transfer_data]
    zero_shot = [d.get("zero_shot_rho", 0.0) for d in transfer_data]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    rects1 = ax.bar(x - width / 2, in_domain, width, label="In-Domain 5-Fold CV", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, zero_shot, width, label="Zero-Shot Topological Transfer", color="#ff7f0e")

    ax.set_ylabel("Spearman Rank Correlation (rho)", fontsize=11, fontweight="medium")
    ax.set_title("Zero-Shot Topological Transfer vs In-Domain CV", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Add bar values
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:+.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out
