"""Tests for Ticket 5: Production Adapters, Parquet/pt Caching, and Publication Figures (Issue #6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters.fake import FakeEmbedder, FakeOracleGeneratorJudge
from src.adapters.qwen_embedder import QwenEmbedder
from src.adapters.vllm_adapter import (
    QuantizedVLLMGeneratorJudge,
    build_generator_prompt,
    build_judge_prompt,
    parse_judge_score,
)
from src.data.caching import (
    export_study_summary_json,
    load_study_artifacts,
    save_study_artifacts,
)
from src.data.loader import QuerySample
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
from src.graph.ppr_retriever import CandidateSubgraphData
from src.qpp.tabular_router import DisagreementMatrix, compute_router_disagreement
from src.study import QueryResult, StudyConfig, StudyReport, run_qpp_study
from scripts.run_benchmarks import run_benchmark_pipeline


def test_vllm_prompt_builders_and_score_parsers():
    """Verify vLLM prompt templates maintain Passage-Budget Parity and parse discrete rubric."""
    passages = [
        {"passage_id": "p1", "text": "Passage one text about quantum flux qubit."},
        {"passage_id": "p2", "text": "Passage two text about Josephson junctions."},
    ]
    query = "How does flux qubit operate?"

    # 1. Generator prompt template parity (ADR-0001)
    prompt = build_generator_prompt(query, passages)
    assert "[1] Passage one text about quantum flux qubit." in prompt
    assert "[2] Passage two text about Josephson junctions." in prompt
    assert f"Question: {query}" in prompt
    assert prompt.endswith("Answer:")

    # 2. Judge prompt rubric (ADR-0002)
    judge_prompt = build_judge_prompt(query, "It uses Josephson junctions.", ["Josephson junctions operate in loops."])
    assert "1.0: Correct and completely answers" in judge_prompt
    assert "0.5: Partially correct" in judge_prompt
    assert "0.0: Factually incorrect" in judge_prompt

    # 3. Score parser robustness
    assert parse_judge_score("Evaluation:\nScore: 1.0\nExplanation: Matches.") == 1.0
    assert parse_judge_score("Score: 0.5") == 0.5
    assert parse_judge_score("score:0.0") == 0.0
    assert parse_judge_score("Overall, 1.0 is given.") == 1.0
    assert parse_judge_score("Completely wrong answer.") == 0.0


def test_quantized_vllm_generator_judge_lifecycle():
    """Verify VRAM budgeting and sequential lifecycle unloading in QuantizedVLLMGeneratorJudge (ADR-0004)."""
    adapter = QuantizedVLLMGeneratorJudge(
        generator_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        judge_model="google/gemma-2-9b-it",
        gpu_memory_utilization=0.40,
        max_model_len=4096,
    )
    # Weights must consume < 10GB (< 45% of 24GB VRAM)
    assert adapter.gpu_memory_utilization <= 0.45
    assert adapter._generator_llm is None
    assert adapter._judge_llm is None

    # Unload releases memory cleanly
    adapter._unload_active_model()
    adapter.cleanup()
    assert adapter._generator_llm is None
    assert adapter._judge_llm is None


def test_parquet_and_pt_artifact_caching_roundtrip(tmp_path: Path):
    """Verify Parquet tabular results and PyTorch .pt Candidate Subgraph caching round-trip."""
    embedder = FakeEmbedder()
    oracle_judge = FakeOracleGeneratorJudge()
    config = StudyConfig(retrieval_top_k=2)

    passages = [
        {"passage_id": "p1", "text": "FluxQubit quantum coherence loop.", "sentences": ["FluxQubit quantum coherence loop."]},
        {"passage_id": "p2", "text": "JosephsonJunction enables coherence.", "sentences": ["JosephsonJunction enables coherence."]},
    ]
    samples = [
        QuerySample(
            query_id=f"q_{i}", dataset="bench", question_type="multi-hop",
            question=f"FluxQubit quantum coherence query {i}?", gold_answers=["yes"],
            gold_evidence_ids=["p1", "p2"], corpus_passages=passages,
        )
        for i in range(1, 4)
    ]

    report = run_qpp_study(samples, config=config, embedder=embedder, oracle_judge=oracle_judge)

    # 1. Save artifacts
    parquet_path, pt_path = save_study_artifacts(report, tmp_path)
    assert parquet_path.exists()
    assert pt_path.exists()
    assert parquet_path.name == "study_results.parquet"
    assert pt_path.name == "candidate_subgraphs.pt"

    # 2. Load artifacts
    df, subgraphs = load_study_artifacts(tmp_path)
    assert len(df) == 3
    assert len(subgraphs) == 3
    assert set(df["query_id"]) == {"q_1", "q_2", "q_3"}
    assert "semantic_gain" in df.columns
    assert "lexical_gain" in df.columns
    assert "ppr_entropy_drift" in df.columns

    for qid in ["q_1", "q_2", "q_3"]:
        assert qid in subgraphs
        cand = subgraphs[qid]
        assert isinstance(cand, CandidateSubgraphData)
        assert cand.node_features.shape[1] < 32  # ADR-0003
        assert hasattr(cand, "edge_attr")

    # 3. Summary JSON
    summary_path = export_study_summary_json(report, tmp_path / "summary.json")
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["total_queries"] == 3
    assert "dataset_counts" in summary_data


def test_latex_table_exporters(tmp_path: Path):
    """Verify publication-ready LaTeX tables export with valid booktabs structure."""
    # 1. Correlation table
    corr_metrics = {
        "NQC": {"type": "Classical Post", "spearman_rho": -0.05, "spearman_pvalue": 0.45, "mse": 0.12, "mae": 0.28},
        "GATv2 GNN": {"type": "GNN Topo", "spearman_rho": 0.65, "spearman_pvalue": 0.0001, "mse": 0.04, "mae": 0.15},
    }
    tex_corr = export_qpp_correlation_table_latex(corr_metrics, tmp_path / "correlations.tex")
    assert r"\begin{table}" in tex_corr
    assert r"\toprule" in tex_corr
    assert "GATv2 GNN" in tex_corr
    assert "+0.650" in tex_corr
    assert (tmp_path / "correlations.tex").exists()

    # 2. Conservative abstention table
    sweep_results = {
        0.0: {"graph_selected_rate": 0.8, "mean_achieved_gain": 0.05, "mean_oracle_gain": 0.10, "accuracy": 0.75, "roc_auc": 0.80},
        0.10: {"graph_selected_rate": 0.4, "mean_achieved_gain": 0.08, "mean_oracle_gain": 0.10, "accuracy": 0.85, "roc_auc": 0.80},
    }
    tex_cga = export_conservative_abstention_table_latex(sweep_results, tmp_path / "cga.tex")
    assert "Conservative Graph Abstention" in tex_cga
    assert r"80.0\%" in tex_cga

    # 3. Hub cleanup ablation table
    ablation_data = [
        {"stage": "Step 0 (Raw)", "hub_access_gini": 0.82, "precision_at_5": 0.30, "recall_at_5": 0.45, "ppr_entropy": 3.8},
        {"stage": "Step 3 (Context)", "hub_access_gini": 0.31, "precision_at_5": 0.65, "recall_at_5": 0.70, "ppr_entropy": 2.1},
    ]
    tex_abl = export_hub_cleanup_ablation_table_latex(ablation_data, tmp_path / "ablation.tex")
    assert "Step 0 (Raw)" in tex_abl
    assert "Step 3 (Context)" in tex_abl

    # 4. Router disagreement table
    lex_decisions = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    sem_decisions = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    dis_mat = compute_router_disagreement(lex_decisions, sem_decisions, tau=0.10)
    dis_sweep = {0.10: dis_mat}

    tex_dis = export_router_disagreement_table_latex(dis_sweep, tmp_path / "disagreement.tex")
    assert r"\%" in tex_dis
    assert f"{dis_mat.cohen_kappa:.3f}" in tex_dis

    # 5. Zero-shot transfer table
    transfer_map = {
        "HotpotQA -> LongMem": {
            "source": "HotpotQA", "target": "LongMemEval-S", "setting": "Zero-Shot",
            "spearman_rho": 0.55, "mean_achieved_gain": 0.065,
        }
    }
    tex_zs = export_zero_shot_transfer_table_latex(transfer_map, tmp_path / "zero_shot.tex")
    assert "HotpotQA" in tex_zs
    assert "LongMemEval-S" in tex_zs


def test_diagnostic_figure_generation(tmp_path: Path):
    """Verify publication diagnostic figures render properly to PDF and PNG."""
    # 1. Conservative abstention sweep
    sweep_results = {
        0.0: {"mean_achieved_gain": 0.02, "mean_oracle_gain": 0.12, "graph_selected_rate": 0.90},
        0.05: {"mean_achieved_gain": 0.06, "mean_oracle_gain": 0.12, "graph_selected_rate": 0.70},
        0.10: {"mean_achieved_gain": 0.09, "mean_oracle_gain": 0.12, "graph_selected_rate": 0.50},
        0.15: {"mean_achieved_gain": 0.07, "mean_oracle_gain": 0.12, "graph_selected_rate": 0.30},
        0.20: {"mean_achieved_gain": 0.03, "mean_oracle_gain": 0.12, "graph_selected_rate": 0.10},
    }
    p_cga_pdf = plot_conservative_abstention_sweep(sweep_results, tmp_path / "cga.pdf")
    p_cga_png = plot_conservative_abstention_sweep(sweep_results, tmp_path / "cga.png")
    assert p_cga_pdf.exists() and p_cga_pdf.stat().st_size > 0
    assert p_cga_png.exists() and p_cga_png.stat().st_size > 0

    # 2. Hub cleanup ablation
    ablation_data = [
        {"stage": "Step 0", "hub_access_gini": 0.85, "precision_at_5": 0.25},
        {"stage": "Step 1", "hub_access_gini": 0.65, "precision_at_5": 0.40},
        {"stage": "Step 2", "hub_access_gini": 0.45, "precision_at_5": 0.55},
        {"stage": "Step 3", "hub_access_gini": 0.30, "precision_at_5": 0.65},
    ]
    p_abl_pdf = plot_hub_cleanup_ablation(ablation_data, tmp_path / "abl.pdf")
    p_abl_png = plot_hub_cleanup_ablation(ablation_data, tmp_path / "abl.png")
    assert p_abl_pdf.exists() and p_abl_pdf.stat().st_size > 0
    assert p_abl_png.exists() and p_abl_png.stat().st_size > 0

    # 3. Router disagreement heatmap
    lex_decisions = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    sem_decisions = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    matrix = compute_router_disagreement(lex_decisions, sem_decisions, tau=0.10)
    p_heat_pdf = plot_router_disagreement_heatmap(matrix, tmp_path / "heat.pdf")
    p_heat_png = plot_router_disagreement_heatmap(matrix, tmp_path / "heat.png")
    assert p_heat_pdf.exists() and p_heat_pdf.stat().st_size > 0
    assert p_heat_png.exists() and p_heat_png.stat().st_size > 0

    # 4. Zero-shot transfer comparison
    transfer_data = [
        {"name": "Multi-Hop -> Conv", "in_domain_rho": 0.62, "zero_shot_rho": 0.51},
        {"name": "Conv -> Multi-Hop", "in_domain_rho": 0.58, "zero_shot_rho": 0.48},
    ]
    p_zs_pdf = plot_zero_shot_transfer_comparison(transfer_data, tmp_path / "zs.pdf")
    p_zs_png = plot_zero_shot_transfer_comparison(transfer_data, tmp_path / "zs.png")
    assert p_zs_pdf.exists() and p_zs_pdf.stat().st_size > 0
    assert p_zs_png.exists() and p_zs_png.stat().st_size > 0


def test_end_to_end_benchmark_pipeline_execution(tmp_path: Path):
    """Verify run_benchmark_pipeline completes end-to-end in mock mode and saves all expected outputs."""
    out_dir = tmp_path / "pipeline_run"
    results = run_benchmark_pipeline(
        mode="mock",
        max_queries_lme=4,
        max_queries_multihop=4,
        output_dir=out_dir,
        cleanup_stage_str="3",
        tau_sweep=[0.0, 0.05, 0.10, 0.15, 0.20],
    )

    assert "report" in results
    assert "parquet_path" in results
    assert "pt_path" in results
    assert "tables_dir" in results
    assert "figures_dir" in results

    # Verify artifacts exist on disk
    assert results["parquet_path"].exists()
    assert results["pt_path"].exists()
    assert results["summary_path"].exists()

    tables = list(results["tables_dir"].glob("*.tex"))
    assert len(tables) >= 5

    figures = list(results["figures_dir"].glob("*.*"))
    assert len(figures) >= 8  # 4 figures in both PDF and PNG format
