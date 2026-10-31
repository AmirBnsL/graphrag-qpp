"""Primary seam implementation for GraphRAG-QPP Study."""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.adapters.base import BaseEmbedder, BaseOracleGeneratorJudge
from src.data.loader import QuerySample
from src.graph.ppr_retriever import CandidateSubgraphData, PPRRetriever
from src.graph.trigraph_builder import TriGraphBuilder
from src.qpp.classical import compute_classical_qpp


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


@dataclass
class StudyReport:
    """Consolidated report across all queries and QPP models."""

    query_results: dict[str, QueryResult]
    classical_qpp_results: dict[str, float]
    config: StudyConfig

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

    def run(self, samples: list[QuerySample]) -> StudyReport:
        query_results: dict[str, QueryResult] = {}
        all_qpp: dict[str, list[float]] = collections.defaultdict(list)

        for sample in samples:
            # 1. Embed query
            q_vec = self.embedder.embed_texts([sample.question])[0]

            # 2. Build Tri-Graph from corpus passages
            trigraph = self.builder.build(sample.corpus_passages)

            # 3. Flat Dense retrieval (strict top-k passage budget parity)
            flat_top = self.retriever.retrieve_flat(
                query_vec=q_vec,
                graph=trigraph,
                top_k=self.config.retrieval_top_k,
            )

            # 4. Tri-Graph PPR retrieval & Candidate Subgraph extraction
            graph_top, candidate_subgraph = self.retriever.retrieve_graph(
                query_vec=q_vec,
                graph=trigraph,
                flat_top_passages=flat_top,
                top_k=self.config.retrieval_top_k,
            )

            # 5. Oracle Generator pass with strict prompt template parity (ADR-0001)
            flat_prompt = format_prompt(sample.question, flat_top)
            graph_prompt = format_prompt(sample.question, graph_top)
            flat_answer = self.oracle_judge.generate(sample.question, flat_top)
            graph_answer = self.oracle_judge.generate(sample.question, graph_top)

            # 6. Lexical Gain (Token F1 difference)
            gold_ans = sample.gold_answers[0] if sample.gold_answers else ""
            flat_f1 = compute_token_f1(flat_answer, gold_ans)
            graph_f1 = compute_token_f1(graph_answer, gold_ans)
            lexical_gain = graph_f1 - flat_f1

            # 7. Semantic Gain (LLM judge rubric & preference)
            flat_judge_score, graph_judge_score, sem_margin = self.oracle_judge.judge(
                query=sample.question,
                flat_answer=flat_answer,
                graph_answer=graph_answer,
                gold_answers=sample.gold_answers,
            )
            semantic_gain = graph_judge_score - flat_judge_score

            # 8. Classical QPP baselines
            cqpp = compute_classical_qpp(
                query=sample.question,
                flat_passages=flat_top,
                corpus_passages=sample.corpus_passages,
                embedder=self.embedder,
            )
            for k, v in cqpp.items():
                all_qpp[k].append(v)

            query_results[sample.query_id] = QueryResult(
                query_id=sample.query_id,
                dataset=sample.dataset,
                question_type=sample.question_type,
                question=sample.question,
                flat_passages=flat_top,
                graph_passages=graph_top,
                flat_answer=flat_answer,
                graph_answer=graph_answer,
                flat_f1=flat_f1,
                graph_f1=graph_f1,
                lexical_gain=lexical_gain,
                flat_judge_score=flat_judge_score,
                graph_judge_score=graph_judge_score,
                semantic_gain=semantic_gain,
                candidate_subgraph=candidate_subgraph,
                classical_qpp=cqpp,
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
) -> StudyReport:
    """Convenience functional wrapper around ContextSelectionStudy."""
    study = ContextSelectionStudy(config=config, embedder=embedder, oracle_judge=oracle_judge)
    return study.run(samples)
