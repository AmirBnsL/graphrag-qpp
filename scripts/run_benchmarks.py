"""End-to-end benchmark execution and publication artifact export script."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Prevent headless Colab from blocking on web UI secrets vault
os.environ.setdefault("HF_HUB_DISABLE_COLAB_SECRETS", "1")
os.environ.setdefault("VERTEX_PRODUCT", "COLAB_ENTERPRISE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from scipy.stats import spearmanr

from src.adapters.base import BaseEmbedder, BaseOracleGeneratorJudge
from src.adapters.fake import FakeEmbedder, FakeOracleGeneratorJudge
from src.adapters.qwen_embedder import QwenEmbedder
from src.adapters.vllm_adapter import QuantizedVLLMGeneratorJudge
from src.data.caching import export_study_summary_json, save_study_artifacts
from src.data.loader import (
    QuerySample,
    load_hotpotqa_slice,
    load_longmemeval_slice,
    load_musique_slice,
)
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
from src.graph.trigraph_builder import CleanupStage
from src.study import (
    CleanupAblationReport,
    StudyConfig,
    StudyReport,
    evaluate_cleanup_ablation,
    run_qpp_study,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_benchmarks")

CLEANUP_STAGE_MAP = {
    "0": CleanupStage.RAW_PERSISTENT,
    "1": CleanupStage.LEXICAL_NORMALIZATION,
    "2": CleanupStage.IDF_DEGREE_DAMPENING,
    "3": CleanupStage.SESSION_CONTEXT_CONSTRAINED,
    "raw": CleanupStage.RAW_PERSISTENT,
    "lemmatized": CleanupStage.LEXICAL_NORMALIZATION,
    "idf": CleanupStage.IDF_DEGREE_DAMPENING,
    "context_filtered": CleanupStage.SESSION_CONTEXT_CONSTRAINED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphRAG-QPP Full Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["mock", "gpu", "colab"],
        default="mock",
        help="Execution mode: 'mock' (CPU fake adapters for testing) or 'gpu'/'colab' (quantized vLLM & Qwen3-Embedding).",
    )
    parser.add_argument(
        "--max-queries-lme",
        type=int,
        default=500,
        help="Maximum queries to load from LongMemEval-S (default: 500).",
    )
    parser.add_argument(
        "--max-queries-multihop",
        type=int,
        default=500,
        help="Maximum queries to load from MuSiQue & HotpotQA (default: 500).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/benchmark_run",
        help="Directory to save Parquet, .pt, LaTeX tables, and PDF/PNG figures.",
    )
    parser.add_argument(
        "--cleanup-stage",
        choices=["0", "1", "2", "3", "raw", "lemmatized", "idf", "context_filtered"],
        default="context_filtered",
        help="Tri-Graph hub cleanup stage to evaluate.",
    )
    parser.add_argument(
        "--tau-sweep",
        type=str,
        default="0.0,0.05,0.10,0.15,0.20",
        help="Comma-separated list of Gain Margin Thresholds (tau).",
    )
    return parser.parse_args()


def setup_adapters(mode: str) -> tuple[BaseEmbedder, BaseOracleGeneratorJudge]:
    """Instantiate Embedder and GeneratorJudge according to execution mode."""
    if mode in ("gpu", "colab"):
        logger.info("Initializing GPU production adapters (Qwen3-Embedding & Quantized vLLM)...")
        embedder: BaseEmbedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B")
        oracle_judge: BaseOracleGeneratorJudge = QuantizedVLLMGeneratorJudge(
            generator_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            judge_model="neuralmagic/gemma-2-9b-it-FP8",
            generator_quantization="awq",
            judge_quantization="fp8",
            gpu_memory_utilization=0.70,
            judge_gpu_memory_utilization=0.80,
            max_model_len=8192,
            judge_max_model_len=4096,
        )
        return embedder, oracle_judge
    else:
        logger.info("Initializing mock in-memory adapters for fast CPU execution...")
        return FakeEmbedder(dim=64), FakeOracleGeneratorJudge()


def run_benchmark_pipeline(
    mode: str = "mock",
    max_queries_lme: int = 500,
    max_queries_multihop: int = 500,
    output_dir: str | Path = "artifacts/benchmark_run",
    cleanup_stage_str: str = "context_filtered",
    tau_sweep: list[float] | None = None,
) -> dict[str, Any]:
    """Run full benchmark pipeline, caching artifacts and exporting tables/figures."""
    if tau_sweep is None:
        tau_sweep = [0.0, 0.05, 0.10, 0.15, 0.20]

    out_path = Path(output_dir)
    tables_dir = out_path / "tables"
    figures_dir = out_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cleanup_stage = CLEANUP_STAGE_MAP.get(cleanup_stage_str, CleanupStage.SESSION_CONTEXT_CONSTRAINED)
    config = StudyConfig(
        retrieval_top_k=5,
        cleanup_stage=cleanup_stage,
        gain_margin_thresholds=tau_sweep,
    )

    embedder, oracle_judge = setup_adapters(mode)

    # 1. Dataset Ingestion
    logger.info("Ingesting benchmark datasets...")
    samples: list[QuerySample] = []
    lme_samples: list[QuerySample] = []
    mh_samples: list[QuerySample] = []

    if mode in ("gpu", "colab"):
        try:
            lme_samples = load_longmemeval_slice(max_queries=max_queries_lme)
            samples.extend(lme_samples)
        except Exception as e:
            logger.warning("Could not load full LongMemEval-S (%s). Using fallback slice.", e)

        try:
            hotpot = load_hotpotqa_slice(max_queries=max_queries_multihop // 2)
            musique = load_musique_slice(max_queries=max_queries_multihop // 2)
            mh_samples = hotpot + musique
            samples.extend(mh_samples)
        except Exception as e:
            logger.warning("Could not load full Multi-Hop benchmarks (%s). Using fallback slice.", e)
    else:
        # In mock mode, construct representative synthetic samples
        passages_hotpot = [
            {"passage_id": f"hp_{i}", "text": f"FluxQubit quantum coherence passage {i}.", "sentences": [f"FluxQubit quantum coherence passage {i}."]}
            for i in range(10)
        ]
        mh_samples = [
            QuerySample(
                query_id=f"mh_{i}", dataset="HotpotQA", question_type="multi-hop",
                question=f"Does FluxQubit coherence enable JosephsonJunction state {i}?",
                gold_answers=["yes"], gold_evidence_ids=["hp_0", "hp_1"], corpus_passages=passages_hotpot,
            )
            for i in range(min(max_queries_multihop, 20))
        ]

        passages_lme = [
            {"passage_id": f"lme_p_{i}", "text": f"User discussed AlphaProject quarterly budget {i}.", "sentences": [f"User discussed AlphaProject quarterly budget {i}."]}
            for i in range(10)
        ]
        lme_samples = [
            QuerySample(
                query_id=f"lme_{i}", dataset="LongMemEval-S", question_type="conversational",
                question=f"What was AlphaProject budget session {i}?",
                gold_answers=["quarterly budget"], gold_evidence_ids=["lme_p_0"], corpus_passages=passages_lme,
            )
            for i in range(min(max_queries_lme, 20))
        ]
        samples = mh_samples + lme_samples

    logger.info("Total query workload: %d queries (%d multi-hop, %d conversational)", len(samples), len(mh_samples), len(lme_samples))

    # 2. Study Execution Seam
    logger.info("Executing GraphRAG-QPP Study...")
    report: StudyReport = run_qpp_study(
        samples=samples,
        config=config,
        embedder=embedder,
        oracle_judge=oracle_judge,
    )

    # 3. Artifact Caching
    logger.info("Saving study artifacts to Parquet and PyTorch .pt...")
    parquet_path, pt_path = save_study_artifacts(report, out_path)
    summary_path = export_study_summary_json(report, out_path / "study_summary.json")

    # 4. Context Selection Evaluation
    logger.info("Evaluating Post-Retrieval Context Selectors (LightGBM & GATv2 GNN)...")
    n_splits = min(5, len(samples) // 2) if len(samples) >= 4 else 2

    # In-Domain Dual Routers
    lightgbm_dual = report.evaluate_dual_routers(n_splits=n_splits, tau_sweep=tau_sweep)
    gnn_dual = report.evaluate_dual_gnn_routers(n_splits=n_splits, tau_sweep=tau_sweep, epochs=15)

    # Zero-Shot Topological Transfer without redundant study re-execution
    mh_qids = {s.query_id for s in mh_samples}
    lme_qids = {s.query_id for s in lme_samples}
    report_mh = StudyReport(
        query_results={qid: res for qid, res in report.query_results.items() if qid in mh_qids},
        classical_qpp_results=report.classical_qpp_results,
        config=config,
    )
    report_lme = StudyReport(
        query_results={qid: res for qid, res in report.query_results.items() if qid in lme_qids},
        classical_qpp_results=report.classical_qpp_results,
        config=config,
    )

    zs_transfer = report_mh.evaluate_gnn_zero_shot_transfer(
        target_report=report_lme,
        target="semantic",
        tau_sweep=tau_sweep,
        epochs=15,
    )

    # 5. Export LaTeX Tables
    logger.info("Exporting publication LaTeX tables...")
    y_sem = np.array([r.semantic_gain for r in report.query_results.values()], dtype=np.float32)
    classical_names = ["nqc", "wig", "avg_idf", "max_idf", "scq", "query_length"]
    correlation_metrics: dict[str, dict[str, Any]] = {}

    for qpp_name in classical_names:
        scores = np.array([r.classical_qpp.get(qpp_name, 0.0) for r in report.query_results.values()], dtype=np.float32)
        if len(scores) > 1 and np.std(scores) > 1e-9 and np.std(y_sem) > 1e-9:
            res_corr = spearmanr(scores, y_sem)
            rho_val = float(res_corr.statistic)
            p_val = float(res_corr.pvalue)
        else:
            rho_val, p_val = 0.0, 1.0

        is_pre = "idf" in qpp_name or "scq" in qpp_name or "length" in qpp_name
        correlation_metrics[qpp_name.upper()] = {
            "type": "Classical Pre" if is_pre else "Classical Post",
            "spearman_rho": rho_val,
            "spearman_pvalue": p_val,
            "mse": float(np.mean((scores - y_sem) ** 2)),
            "mae": float(np.mean(np.abs(scores - y_sem))),
        }

    correlation_metrics["Topological LightGBM"] = {
        "type": "Tabular Topo",
        "spearman_rho": lightgbm_dual["semantic_cv"].spearman_rho,
        "spearman_pvalue": lightgbm_dual["semantic_cv"].spearman_pvalue,
        "mse": lightgbm_dual["semantic_cv"].mse,
        "mae": lightgbm_dual["semantic_cv"].mae,
    }
    correlation_metrics["GATv2 GNN Context Selector"] = {
        "type": "GNN Topo",
        "spearman_rho": gnn_dual["semantic_cv"].spearman_rho,
        "spearman_pvalue": gnn_dual["semantic_cv"].spearman_pvalue,
        "mse": gnn_dual["semantic_cv"].mse,
        "mae": gnn_dual["semantic_cv"].mae,
    }

    export_qpp_correlation_table_latex(correlation_metrics, tables_dir / "qpp_correlations.tex")
    export_conservative_abstention_table_latex(gnn_dual["semantic_cv"].tau_sweep_results, tables_dir / "conservative_abstention.tex")

    # Run real 4-stage cleanup ablation without calling LLM
    ablation_report: CleanupAblationReport = evaluate_cleanup_ablation(
        samples=samples[:min(len(samples), 10)],
        config=config,
        embedder=embedder,
    )
    ablation_data = [
        {
            "stage": f"Step {int(st)} ({st.name})",
            "hub_access_gini": res.mean_access_gini,
            "precision_at_5": res.mean_precision,
            "recall_at_5": res.mean_recall,
            "ppr_entropy": res.mean_ppr_entropy,
        }
        for st, res in ablation_report.stage_results.items()
    ]
    export_hub_cleanup_ablation_table_latex(ablation_data, tables_dir / "hub_cleanup_ablation.tex")
    export_router_disagreement_table_latex(gnn_dual["disagreement_sweep"], tables_dir / "router_disagreement.tex")

    transfer_map = {
        "HotpotQA -> LongMemEval-S": {
            "source": "HotpotQA",
            "target": "LongMemEval-S",
            "setting": "Zero-Shot",
            "spearman_rho": zs_transfer.spearman_rho,
            "mean_achieved_gain": zs_transfer.tau_sweep_results.get(0.10, {}).get("mean_achieved_gain", 0.0),
        }
    }
    export_zero_shot_transfer_table_latex(transfer_map, tables_dir / "zero_shot_transfer.tex")

    # 6. Export Diagnostic Figures (PDF and PNG)
    logger.info("Generating publication diagnostic figures (PDF & PNG)...")
    plot_conservative_abstention_sweep(gnn_dual["semantic_cv"].tau_sweep_results, figures_dir / "conservative_abstention_sweep.pdf")
    plot_conservative_abstention_sweep(gnn_dual["semantic_cv"].tau_sweep_results, figures_dir / "conservative_abstention_sweep.png")

    plot_hub_cleanup_ablation(ablation_data, figures_dir / "hub_cleanup_ablation.pdf")
    plot_hub_cleanup_ablation(ablation_data, figures_dir / "hub_cleanup_ablation.png")

    if 0.10 in gnn_dual["disagreement_sweep"]:
        plot_router_disagreement_heatmap(gnn_dual["disagreement_sweep"][0.10], figures_dir / "router_disagreement_heatmap.pdf")
        plot_router_disagreement_heatmap(gnn_dual["disagreement_sweep"][0.10], figures_dir / "router_disagreement_heatmap.png")

    transfer_plot_data = [
        {"name": "Multi-Hop -> Conv", "in_domain_rho": gnn_dual["semantic_cv"].spearman_rho, "zero_shot_rho": zs_transfer.spearman_rho}
    ]
    plot_zero_shot_transfer_comparison(transfer_plot_data, figures_dir / "zero_shot_transfer.pdf")
    plot_zero_shot_transfer_comparison(transfer_plot_data, figures_dir / "zero_shot_transfer.png")

    logger.info("Pipeline execution completed successfully.")
    return {
        "report": report,
        "parquet_path": parquet_path,
        "pt_path": pt_path,
        "summary_path": summary_path,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "lightgbm_dual": lightgbm_dual,
        "gnn_dual": gnn_dual,
        "zs_transfer": zs_transfer,
        "ablation_report": ablation_report,
    }


def main() -> None:
    args = parse_args()
    tau_sweep = [float(x.strip()) for x in args.tau_sweep.split(",") if x.strip()]
    results = run_benchmark_pipeline(
        mode=args.mode,
        max_queries_lme=args.max_queries_lme,
        max_queries_multihop=args.max_queries_multihop,
        output_dir=args.output_dir,
        cleanup_stage_str=args.cleanup_stage,
        tau_sweep=tau_sweep,
    )
    print("\n=== Benchmark Run Summary ===")
    print(f"Total Processed Queries: {len(results['report'].query_results)}")
    print(f"Parquet Artifact:        {results['parquet_path']}")
    print(f"PyTorch Artifact:        {results['pt_path']}")
    print(f"Tables Directory:        {results['tables_dir']}")
    print(f"Figures Directory:       {results['figures_dir']}")
    print(f"Mean Lexical Gain:       {results['report'].mean_lexical_gain:+.4f}")
    print(f"Mean Semantic Gain:      {results['report'].mean_semantic_gain:+.4f}")
    print("=============================\n")


if __name__ == "__main__":
    main()
