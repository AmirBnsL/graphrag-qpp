"""Parquet and PyTorch (.pt) artifact caching for GraphRAG-QPP Study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

from src.graph.ppr_retriever import CandidateSubgraphData

if TYPE_CHECKING:
    from src.study import StudyReport


def save_study_artifacts(
    report: StudyReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Serialize tabular study results to Parquet and Candidate Subgraphs to PyTorch .pt."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_file = out_path / "study_results.parquet"
    pt_file = out_path / "candidate_subgraphs.pt"

    rows: list[dict[str, Any]] = []
    subgraphs: dict[str, CandidateSubgraphData] = {}
    retrieval_outputs: dict[str, dict[str, Any]] = {}

    for qid, res in report.query_results.items():
        row: dict[str, Any] = {
            "query_id": res.query_id,
            "dataset": res.dataset,
            "question_type": res.question_type,
            "question": res.question,
            "no_rag_answer": getattr(res, "no_rag_answer", ""),
            "flat_answer": res.flat_answer,
            "graph_answer": res.graph_answer,
            "flat_f1": float(res.flat_f1),
            "graph_f1": float(res.graph_f1),
            "lexical_gain": float(res.lexical_gain),
            "flat_judge_score": float(res.flat_judge_score),
            "graph_judge_score": float(res.graph_judge_score),
            "semantic_gain": float(res.semantic_gain),
            "ppr_entropy_drift": float(res.candidate_subgraph.ppr_entropy_drift),
            "flat_passages_json": json.dumps(res.flat_passages),
            "graph_passages_json": json.dumps(res.graph_passages),
        }
        # Classical QPP features
        for k, v in res.classical_qpp.items():
            row[f"qpp_{k}"] = float(v)

        rows.append(row)
        subgraphs[qid] = res.candidate_subgraph
        retrieval_outputs[qid] = {
            "flat_passages": res.flat_passages,
            "graph_passages": res.graph_passages,
        }

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_file, index=False)

    torch.save({"subgraphs": subgraphs, "retrieval_outputs": retrieval_outputs}, pt_file)

    return parquet_file, pt_file


def load_study_artifacts(
    input_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, CandidateSubgraphData]]:
    """Load cached Parquet tabular dataframe and deserialized Candidate Subgraphs (.pt)."""
    in_path = Path(input_dir)
    parquet_file = in_path / "study_results.parquet"
    pt_file = in_path / "candidate_subgraphs.pt"

    if not parquet_file.exists():
        raise FileNotFoundError(f"Missing study results Parquet file: {parquet_file}")
    if not pt_file.exists():
        raise FileNotFoundError(f"Missing candidate subgraphs PyTorch file: {pt_file}")

    df = pd.read_parquet(parquet_file)
    raw_pt: Any = torch.load(pt_file, weights_only=False)
    if isinstance(raw_pt, dict) and "subgraphs" in raw_pt:
        subgraphs: dict[str, CandidateSubgraphData] = raw_pt["subgraphs"]
    else:
        subgraphs = raw_pt

    return df, subgraphs


def export_study_summary_json(
    report: StudyReport,
    output_path: str | Path,
) -> Path:
    """Export high-level summary JSON of study results and configurations."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_queries": len(report.query_results),
        "mean_lexical_gain": float(report.mean_lexical_gain),
        "mean_semantic_gain": float(report.mean_semantic_gain),
        "mean_access_gini": float(getattr(report, "mean_access_gini", 0.0)),
        "mean_ppr_entropy_drift": float(getattr(report, "mean_ppr_entropy_drift", 0.0)),
        "config": {
            "retrieval_top_k": report.config.retrieval_top_k,
            "cleanup_stage": int(report.config.cleanup_stage),
            "gain_margin_thresholds": report.config.gain_margin_thresholds,
        },
        "dataset_counts": {},
    }

    counts: dict[str, int] = {}
    for res in report.query_results.values():
        ds = res.dataset
        counts[ds] = counts.get(ds, 0) + 1
    summary["dataset_counts"] = counts

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_file
