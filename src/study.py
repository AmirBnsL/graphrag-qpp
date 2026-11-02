"""Primary seam implementation for GraphRAG-QPP Study."""

from __future__ import annotations

import collections
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from src.adapters.base import BaseEmbedder, BaseOracleGeneratorJudge
from src.data.loader import QuerySample
from src.eval.retrieval_metrics import (
    compute_access_gini,
    compute_ppr_entropy,
    compute_precision_at_k,
    compute_recall_at_k,
)
from src.graph.ppr_retriever import CandidateSubgraphData, PPRRetriever
from src.graph.trigraph_builder import CleanupStage, TriGraphBuilder
from src.qpp.classical import compute_classical_qpp
from src.qpp.gnn_router import (
    TopologicalGNNRouter,
    candidate_subgraph_to_pyg_data,
    evaluate_gnn_dual_routers,
    evaluate_gnn_in_domain_cv,
    evaluate_gnn_zero_shot_transfer,
)
from src.qpp.tabular_router import (
    DisagreementMatrix,
    TabularCVResult,
    TopologicalLightGBMRouter,
    TransferResult,
    evaluate_in_domain_cv,
    evaluate_router_disagreement_sweep,
    evaluate_zero_shot_transfer,
)
from src.qpp.topological_features import extract_topological_features


def format_prompt(query: str, passages: list[dict[str, Any]]) -> str:
    """Format prompt with strict template parity across Flat and Graph RAG (ADR-0001)."""
    context = "\n\n".join(f"[{i+1}] {p.get('text', '')}" for i, p in enumerate(passages))
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def compute_token_f1(prediction: str, gold: str) -> float:
    """Compute token-level SQuAD-style F1 overlap score."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gold_tokens = re.findall(r"\w+", gold.lower())
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


@dataclass
class StudyConfig:
    """Configuration for GraphRAG-QPP Study."""

    retrieval_top_k: int = 5
    ppr_alpha: float = 0.15
    ppr_shallow_steps: int = 2
    ppr_deep_steps: int = 10
    cleanup_stage: CleanupStage = CleanupStage.RAW_PERSISTENT
    gain_margin_thresholds: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )


@dataclass
class QueryResult:
    """Execution results for a single query."""

    query_id: str
    dataset: str
    question_type: str
    question: str
    flat_passages: list[dict[str, Any]]
    graph_passages: list[dict[str, Any]]
    flat_answer: str
    graph_answer: str
    flat_f1: float
    graph_f1: float
    lexical_gain: float
    flat_judge_score: float
    graph_judge_score: float
    semantic_gain: float
    candidate_subgraph: CandidateSubgraphData
    classical_qpp: dict[str, float]
    no_rag_answer: str = ""


@dataclass
class StudyReport:
    """Consolidated report across all queries and QPP models."""

    query_results: dict[str, QueryResult]
    classical_qpp_results: dict[str, float]
    config: StudyConfig

    @property
    def mean_lexical_gain(self) -> float:
        """Mean token F1 gain across all evaluated queries."""
        if not self.query_results:
            return 0.0
        return float(np.mean([r.lexical_gain for r in self.query_results.values()]))

    @property
    def mean_semantic_gain(self) -> float:
        """Mean LLM judge score gain across all evaluated queries."""
        if not self.query_results:
            return 0.0
        return float(np.mean([r.semantic_gain for r in self.query_results.values()]))

    def evaluate_conservative_abstention(
        self,
        tau: float = 0.10,
        target: str = "semantic",
    ) -> dict[str, Any]:
        """Evaluate Conservative Graph Abstention policy: select Flat if gain < tau, Graph if >= tau."""
        flat_selected = 0
        graph_selected = 0
        achieved_gains = []

        for res in self.query_results.values():
            gain = res.semantic_gain if target == "semantic" else res.lexical_gain
            if gain >= tau:
                graph_selected += 1
                achieved_gains.append(gain)
            else:
                flat_selected += 1
                achieved_gains.append(0.0)

        return {
            "tau": tau,
            "target": target,
            "flat_selected_count": flat_selected,
            "graph_selected_count": graph_selected,
            "mean_achieved_gain": float(np.mean(achieved_gains)) if achieved_gains else 0.0,
        }

    def evaluate_classical_qpp_routing(
        self,
        predictor_key: str = "nqc",
        percentile_threshold: float = 50.0,
        target: str = "semantic",
    ) -> dict[str, Any]:
        """Evaluate Classical Flat QPP as a routing policy under Conservative Graph Abstention."""
        scores = [res.classical_qpp.get(predictor_key, 0.0) for res in self.query_results.values()]
        cutoff = float(np.percentile(scores, percentile_threshold)) if scores else 0.0

        flat_selected = 0
        graph_selected = 0
        achieved_gains = []

        for res in self.query_results.values():
            pred_score = res.classical_qpp.get(predictor_key, 0.0)
            actual_gain = res.semantic_gain if target == "semantic" else res.lexical_gain
            # If QPP score exceeds cutoff, route to Graph; else default to Flat
            if pred_score >= cutoff:
                graph_selected += 1
                achieved_gains.append(actual_gain)
            else:
                flat_selected += 1
                achieved_gains.append(0.0)

        return {
            "predictor_key": predictor_key,
            "cutoff": cutoff,
            "flat_selected_count": flat_selected,
            "graph_selected_count": graph_selected,
            "mean_achieved_gain": float(np.mean(achieved_gains)) if achieved_gains else 0.0,
        }

    def extract_topological_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Extract handcrafted topological feature matrix and targets from all query results."""
        feature_list: list[list[float]] = []
        lexical_gains: list[float] = []
        semantic_gains: list[float] = []
        feature_names: list[str] = []

        for res in self.query_results.values():
            feats = extract_topological_features(
                candidate_subgraph=res.candidate_subgraph,
                flat_passages=res.flat_passages,
                graph_passages=res.graph_passages,
            )
            if not feature_names:
                feature_names = list(feats.keys())
            feature_list.append([feats[k] for k in feature_names])
            lexical_gains.append(res.lexical_gain)
            semantic_gains.append(res.semantic_gain)

        X = np.array(feature_list, dtype=np.float32) if feature_list else np.empty((0, 15), dtype=np.float32)
        y_lex = np.array(lexical_gains, dtype=np.float32)
        y_sem = np.array(semantic_gains, dtype=np.float32)
        return X, y_lex, y_sem, feature_names

    def evaluate_topological_lightgbm(
        self,
        target: str = "semantic",
        n_splits: int = 5,
        tau_sweep: list[float] | None = None,
        random_state: int = 42,
    ) -> TabularCVResult:
        """Train and cross-validate Topological LightGBM Post-Retrieval Context Selector."""
        if target not in ("semantic", "lexical"):
            raise ValueError(f"Invalid target: '{target}'. Expected 'semantic' or 'lexical'.")
        X, y_lex, y_sem, names = self.extract_topological_features()
        y = y_sem if target == "semantic" else y_lex
        sweep = tau_sweep or self.config.gain_margin_thresholds
        return evaluate_in_domain_cv(
            X=X,
            y=y,
            n_splits=n_splits,
            tau_sweep=sweep,
            random_state=random_state,
            router_kwargs={"feature_names": names, "random_state": random_state},
        )

    def evaluate_dual_routers(
        self,
        n_splits: int = 5,
        tau_sweep: list[float] | None = None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Train separate Lexical and Semantic routers, returning CV results and router disagreement sweep (Issue #4)."""
        X, y_lex, y_sem, names = self.extract_topological_features()
        sweep = tau_sweep or self.config.gain_margin_thresholds

        lexical_cv = evaluate_in_domain_cv(
            X=X,
            y=y_lex,
            n_splits=n_splits,
            tau_sweep=sweep,
            random_state=random_state,
            router_kwargs={"feature_names": names, "random_state": random_state},
        )
        semantic_cv = evaluate_in_domain_cv(
            X=X,
            y=y_sem,
            n_splits=n_splits,
            tau_sweep=sweep,
            random_state=random_state,
            router_kwargs={"feature_names": names, "random_state": random_state},
        )

        disagreement_sweep = evaluate_router_disagreement_sweep(
            lexical_preds=lexical_cv.out_of_fold_predictions,
            semantic_preds=semantic_cv.out_of_fold_predictions,
            tau_sweep=sweep,
        )

        return {
            "lexical_cv": lexical_cv,
            "semantic_cv": semantic_cv,
            "disagreement_sweep": disagreement_sweep,
        }

    def evaluate_zero_shot_transfer(
        self,
        target_report: StudyReport,
        target: str = "semantic",
        tau_sweep: list[float] | None = None,
        random_state: int = 42,
    ) -> TransferResult:
        """Evaluate Zero-Shot Topological Transfer from this study to another benchmark (GLOSSARY.md)."""
        if target not in ("semantic", "lexical"):
            raise ValueError(f"Invalid target: '{target}'. Expected 'semantic' or 'lexical'.")
        X_train, y_lex_train, y_sem_train, names = self.extract_topological_features()
        X_test, y_lex_test, y_sem_test, _ = target_report.extract_topological_features()

        y_train = y_sem_train if target == "semantic" else y_lex_train
        y_test = y_sem_test if target == "semantic" else y_lex_test
        sweep = tau_sweep or self.config.gain_margin_thresholds

        router = TopologicalLightGBMRouter(feature_names=names, random_state=random_state)
        router.fit(X_train, y_train)

        return evaluate_zero_shot_transfer(
            router=router,
            X_target=X_test,
            y_target=y_test,
            tau_sweep=sweep,
        )

    def evaluate_topological_gnn(
        self,
        target: str = "semantic",
        n_splits: int = 5,
        tau_sweep: list[float] | None = None,
        epochs: int = 20,
        lr: float = 0.01,
        random_state: int = 42,
    ) -> TabularCVResult:
        """Train and cross-validate 2-layer GATv2 Post-Retrieval Context Selector (ADR-0003, Issue #5)."""
        if target not in ("semantic", "lexical"):
            raise ValueError(f"Invalid target: '{target}'. Expected 'semantic' or 'lexical'.")
        sweep = tau_sweep or self.config.gain_margin_thresholds

        graphs: list[Any] = []
        targets: list[float] = []
        for res in self.query_results.values():
            y = res.semantic_gain if target == "semantic" else res.lexical_gain
            pyg_data = candidate_subgraph_to_pyg_data(res.candidate_subgraph, label=y)
            graphs.append(pyg_data)
            targets.append(y)

        return evaluate_gnn_in_domain_cv(
            graphs=graphs,
            targets=np.array(targets, dtype=np.float32),
            n_splits=n_splits,
            tau_sweep=sweep,
            epochs=epochs,
            lr=lr,
            random_state=random_state,
        )

    def evaluate_dual_gnn_routers(
        self,
        n_splits: int = 5,
        tau_sweep: list[float] | None = None,
        epochs: int = 20,
        lr: float = 0.01,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Train separate Lexical and Semantic GNN routers, returning CV results and router disagreement sweep (Issue #5)."""
        sweep = tau_sweep or self.config.gain_margin_thresholds
        graphs: list[Any] = []
        y_lex: list[float] = []
        y_sem: list[float] = []
        for res in self.query_results.values():
            y_l = res.lexical_gain
            y_s = res.semantic_gain
            graphs.append(candidate_subgraph_to_pyg_data(res.candidate_subgraph, label=y_s))
            y_lex.append(y_l)
            y_sem.append(y_s)

        return evaluate_gnn_dual_routers(
            graphs=graphs,
            y_lexical=np.array(y_lex, dtype=np.float32),
            y_semantic=np.array(y_sem, dtype=np.float32),
            n_splits=n_splits,
            tau_sweep=sweep,
            epochs=epochs,
            lr=lr,
            random_state=random_state,
        )

    def evaluate_gnn_zero_shot_transfer(
        self,
        target_report: StudyReport,
        target: str = "semantic",
        tau_sweep: list[float] | None = None,
        epochs: int = 20,
        lr: float = 0.01,
        random_state: int = 42,
    ) -> TransferResult:
        """Evaluate Zero-Shot Topological Transfer of GNN from this study to another benchmark (GLOSSARY.md)."""
        if target not in ("semantic", "lexical"):
            raise ValueError(f"Invalid target: '{target}'. Expected 'semantic' or 'lexical'.")
        sweep = tau_sweep or self.config.gain_margin_thresholds

        graphs_train: list[Any] = []
        targets_train: list[float] = []
        for res in self.query_results.values():
            y = res.semantic_gain if target == "semantic" else res.lexical_gain
            graphs_train.append(candidate_subgraph_to_pyg_data(res.candidate_subgraph, label=y))
            targets_train.append(y)

        graphs_target: list[Any] = []
        targets_target: list[float] = []
        for res in target_report.query_results.values():
            y = res.semantic_gain if target == "semantic" else res.lexical_gain
            graphs_target.append(candidate_subgraph_to_pyg_data(res.candidate_subgraph, label=y))
            targets_target.append(y)

        router = TopologicalGNNRouter(
            epochs=epochs,
            lr=lr,
            random_state=random_state,
        )
        router.fit(graphs_train, np.array(targets_train, dtype=np.float32))

        return evaluate_gnn_zero_shot_transfer(
            router=router,
            graphs_target=graphs_target,
            targets_target=np.array(targets_target, dtype=np.float32),
            tau_sweep=sweep,
        )


class ContextSelectionStudy:
    """Deep module orchestrating Tri-Graph building, retrieval, and QPP evaluation."""

    def __init__(
        self,
        config: StudyConfig,
        embedder: BaseEmbedder,
        oracle_judge: BaseOracleGeneratorJudge,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.oracle_judge = oracle_judge
        self.builder = TriGraphBuilder(embedder=embedder)
        self.retriever = PPRRetriever(
            alpha=config.ppr_alpha,
            shallow_steps=config.ppr_shallow_steps,
            deep_steps=config.ppr_deep_steps,
        )

    def run(self, samples: list[QuerySample], checkpoint_path: str | Path | None = None) -> StudyReport:
        query_results: dict[str, QueryResult] = {}
        all_qpp: dict[str, list[float]] = collections.defaultdict(list)
        intermediate: list[dict[str, Any]] = []

        # Check if Step A checkpoint exists
        loaded_checkpoint = False
        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                try:
                    import torch
                    data = torch.load(ckpt, weights_only=False)
                    if len(data.get("intermediate", [])) == len(samples):
                        intermediate = data["intermediate"]
                        all_qpp = collections.defaultdict(list, data.get("all_qpp", {}))
                        loaded_checkpoint = True
                        logger.info("Loaded cached Step A checkpoint from %s (%d samples)", ckpt, len(intermediate))
                except Exception as e:
                    logger.warning("Failed loading checkpoint %s: %s. Re-running Step A.", ckpt, e)

        if not loaded_checkpoint:
            # Step A: Perform retrieval and classical QPP indexing for all samples
            logger.info("Step A: Batch-embedding %d query questions...", len(samples))
            all_q_vecs = self.embedder.embed_texts([sample.question for sample in samples])
            for idx, sample in enumerate(samples):
                if (idx + 1) % 50 == 0 or idx == len(samples) - 1:
                    logger.info("Step A Progress: [%d/%d] queries indexed", idx + 1, len(samples))
                q_vec = all_q_vecs[idx] if len(all_q_vecs) > idx else self.embedder.embed_texts([sample.question])[0]
                trigraph = self.builder.build(sample.corpus_passages, stage=self.config.cleanup_stage)

                flat_top = self.retriever.retrieve_flat(
                    query_vec=q_vec,
                    graph=trigraph,
                    top_k=self.config.retrieval_top_k,
                )
                graph_top, candidate_subgraph = self.retriever.retrieve_graph(
                    query_vec=q_vec,
                    graph=trigraph,
                    flat_top_passages=flat_top,
                    top_k=self.config.retrieval_top_k,
                )
                cqpp = compute_classical_qpp(
                    query=sample.question,
                    flat_passages=flat_top,
                    corpus_passages=sample.corpus_passages,
                    embedder=self.embedder,
                    query_vec=q_vec,
                )
                for k, v in cqpp.items():
                    all_qpp[k].append(v)

                intermediate.append({
                    "sample": sample,
                    "flat_top": flat_top,
                    "graph_top": graph_top,
                    "candidate_subgraph": candidate_subgraph,
                    "cqpp": cqpp,
                })

            if checkpoint_path is not None:
                try:
                    import torch
                    ckpt = Path(checkpoint_path)
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"intermediate": intermediate, "all_qpp": dict(all_qpp)}, ckpt)
                    logger.info("Saved Step A checkpoint to %s", ckpt)
                except Exception as e:
                    logger.warning("Could not save Step A checkpoint: %s", e)

        # Free embedder GPU VRAM before loading the generator LLM
        if hasattr(self.embedder, "unload"):
            self.embedder.unload()

        # Step B: Generator stage (No-RAG, Flat-RAG, Graph-RAG under Passage-Budget Parity)
        if hasattr(self.oracle_judge, "generate_batch"):
            pairs_to_generate: list[tuple[str, list[dict[str, Any]]]] = []
            for item in intermediate:
                q = item["sample"].question
                pairs_to_generate.append((q, []))  # No-RAG baseline
                pairs_to_generate.append((q, item["flat_top"]))
                pairs_to_generate.append((q, item["graph_top"]))

            generated_texts = self.oracle_judge.generate_batch(pairs_to_generate)
            no_rag_answers = generated_texts[0::3]
            flat_answers = generated_texts[1::3]
            graph_answers = generated_texts[2::3]
        else:
            no_rag_answers = [self.oracle_judge.generate(item["sample"].question, []) for item in intermediate]
            flat_answers = [self.oracle_judge.generate(item["sample"].question, item["flat_top"]) for item in intermediate]
            graph_answers = [self.oracle_judge.generate(item["sample"].question, item["graph_top"]) for item in intermediate]

        # Step C: Judge stage (Discrete rubric & pairwise preference margin)
        if hasattr(self.oracle_judge, "judge_batch"):
            tuples_to_judge = [
                (item["sample"].question, flat_answers[i], graph_answers[i], item["sample"].gold_answers)
                for i, item in enumerate(intermediate)
            ]
            judged_tuples = self.oracle_judge.judge_batch(tuples_to_judge)
        else:
            judged_tuples = [
                self.oracle_judge.judge(
                    query=item["sample"].question,
                    flat_answer=flat_answers[i],
                    graph_answer=graph_answers[i],
                    gold_answers=item["sample"].gold_answers,
                )
                for i, item in enumerate(intermediate)
            ]

        # Step C.1: Clean up judge GPU memory before selector training
        if hasattr(self.oracle_judge, "cleanup"):
            self.oracle_judge.cleanup()

        # Step D: Assemble QueryResult objects
        for i, item in enumerate(intermediate):
            sample = item["sample"]
            flat_ans = flat_answers[i]
            graph_ans = graph_answers[i]
            no_rag_ans = no_rag_answers[i]
            flat_score, graph_score, _ = judged_tuples[i]

            gold_ans = sample.gold_answers[0] if sample.gold_answers else ""
            flat_f1 = compute_token_f1(flat_ans, gold_ans)
            graph_f1 = compute_token_f1(graph_ans, gold_ans)
            lexical_gain = graph_f1 - flat_f1
            semantic_gain = graph_score - flat_score

            query_results[sample.query_id] = QueryResult(
                query_id=sample.query_id,
                dataset=sample.dataset,
                question_type=sample.question_type,
                question=sample.question,
                flat_passages=item["flat_top"],
                graph_passages=item["graph_top"],
                flat_answer=flat_ans,
                graph_answer=graph_ans,
                flat_f1=flat_f1,
                graph_f1=graph_f1,
                lexical_gain=lexical_gain,
                flat_judge_score=flat_score,
                graph_judge_score=graph_score,
                semantic_gain=semantic_gain,
                candidate_subgraph=item["candidate_subgraph"],
                classical_qpp=item["cqpp"],
                no_rag_answer=no_rag_ans,
            )

        summary_qpp = {k: float(np.mean(vals)) for k, vals in all_qpp.items()}

        return StudyReport(
            query_results=query_results,
            classical_qpp_results=summary_qpp,
            config=self.config,
        )


def run_qpp_study(
    samples: list[QuerySample],
    config: StudyConfig,
    embedder: BaseEmbedder,
    oracle_judge: BaseOracleGeneratorJudge,
    checkpoint_path: str | Path | None = None,
) -> StudyReport:
    """Convenience functional wrapper around ContextSelectionStudy."""
    study = ContextSelectionStudy(config=config, embedder=embedder, oracle_judge=oracle_judge)
    return study.run(samples, checkpoint_path=checkpoint_path)


@dataclass
class StageResult:
    """Consolidated retrieval metrics for a Tri-Graph cleanup stage."""

    stage: CleanupStage
    mean_precision: float
    mean_recall: float
    mean_access_gini: float
    mean_ppr_entropy: float
    mean_ppr_entropy_drift: float
    per_query_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CleanupAblationReport:
    """Report comparing retrieval metrics across the 4-stage Tri-Graph cleanup hierarchy."""

    stage_results: dict[CleanupStage, StageResult]
    config: StudyConfig


def evaluate_cleanup_ablation(
    samples: list[QuerySample],
    config: StudyConfig,
    embedder: BaseEmbedder,
) -> CleanupAblationReport:
    """Evaluate 4-stage cleanup ablation on retrieval metrics without LLM calls (ADR-0006, Issue #3).

    Evaluates:
      Stage 0 (RAW_PERSISTENT): Unconstrained persistent graph, uniform weights.
      Stage 1 (LEXICAL_NORMALIZATION): POS filter, lemmatization, conversational stop-list.
      Stage 2 (IDF_DEGREE_DAMPENING): Corpus IDF edge weighting + degree-penalized PPR.
      Stage 3 (SESSION_CONTEXT_CONSTRAINED): Cross-session context-similarity edge gating.
    """
    builder = TriGraphBuilder(embedder=embedder)
    retriever = PPRRetriever(
        alpha=config.ppr_alpha,
        shallow_steps=config.ppr_shallow_steps,
        deep_steps=config.ppr_deep_steps,
    )

    stage_results: dict[CleanupStage, StageResult] = {}

    for stage in [
        CleanupStage.RAW_PERSISTENT,
        CleanupStage.LEXICAL_NORMALIZATION,
        CleanupStage.IDF_DEGREE_DAMPENING,
        CleanupStage.SESSION_CONTEXT_CONSTRAINED,
    ]:
        precisions: list[float] = []
        recalls: list[float] = []
        ginis: list[float] = []
        entropies: list[float] = []
        drifts: list[float] = []
        per_query: list[dict[str, Any]] = []

        for sample in samples:
            q_vec = embedder.embed_texts([sample.question])[0]
            graph = builder.build(sample.corpus_passages, stage=stage)

            flat_passages = retriever.retrieve_flat(
                query_vec=q_vec,
                graph=graph,
                top_k=config.retrieval_top_k,
            )

            graph_passages, cand_subgraph = retriever.retrieve_graph(
                query_vec=q_vec,
                graph=graph,
                flat_top_passages=flat_passages,
                top_k=config.retrieval_top_k,
            )

            retrieved_ids = [p.get("passage_id", "") for p in graph_passages]
            gold_ids = sample.gold_evidence_ids

            prec = compute_precision_at_k(retrieved_ids, gold_ids, k=config.retrieval_top_k)
            rec = compute_recall_at_k(retrieved_ids, gold_ids, k=config.retrieval_top_k)
            # Hub Access Gini: Gini coefficient measuring access concentration across active candidate nodes (Song & Cai, 2026)
            gini = compute_access_gini(cand_subgraph.ppr_deep, only_positive=True)
            ent = compute_ppr_entropy(cand_subgraph.ppr_deep)
            drift = cand_subgraph.ppr_entropy_drift

            precisions.append(prec)
            recalls.append(rec)
            ginis.append(gini)
            entropies.append(ent)
            drifts.append(drift)

            per_query.append(
                {
                    "query_id": sample.query_id,
                    "precision": prec,
                    "recall": rec,
                    "access_gini": gini,
                    "ppr_entropy": ent,
                    "ppr_entropy_drift": drift,
                    "retrieved_ids": retrieved_ids,
                }
            )

        stage_results[stage] = StageResult(
            stage=stage,
            mean_precision=float(np.mean(precisions)) if precisions else 0.0,
            mean_recall=float(np.mean(recalls)) if recalls else 0.0,
            mean_access_gini=float(np.mean(ginis)) if ginis else 0.0,
            mean_ppr_entropy=float(np.mean(entropies)) if entropies else 0.0,
            mean_ppr_entropy_drift=float(np.mean(drifts)) if drifts else 0.0,
            per_query_results=per_query,
        )

    return CleanupAblationReport(
        stage_results=stage_results,
        config=config,
    )
