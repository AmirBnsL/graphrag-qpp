"""Tracer-bullet test for Ticket 1: run_qpp_study primary seam and classical QPP."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.loader import QuerySample
from src.study import (
    ContextSelectionStudy,
    StudyConfig,
    StudyReport,
    run_qpp_study,
)
from src.adapters.fake import FakeEmbedder, FakeOracleGeneratorJudge


def create_planted_scenarios() -> list[QuerySample]:
    """Create two planted query samples: one clean multi-hop bridge, one cross-session hub contamination."""
    # 1. Multi-hop bridge query: query needs bridge connection
    # doc_bridge_1: contains "QuantumInductance" and "JosephsonJunction"
    # doc_bridge_2: contains "JosephsonJunction" and "FluxQubit"
    # Distractors share words "quantum", "coherence" with query, so Flat Dense retrieval
    # picks doc_bridge_2 and 4 distractors, missing doc_bridge_1.
    # Tri-Graph PPR follows JosephsonJunction bridge from doc_bridge_2 to doc_bridge_1!
    passages_bridge = [
        {
            "passage_id": "p_bridge_1",
            "title": "Inductance Hardware",
            "text": "Superconducting circuits use JosephsonJunction elements for inductive loops.",
            "sentences": ["Superconducting circuits use JosephsonJunction elements for inductive loops."],
            "timestamp": None,
        },
        {
            "passage_id": "p_bridge_2",
            "title": "Flux Qubit Design",
            "text": "FluxQubit designs exploit JosephsonJunction loops to maintain quantum coherence.",
            "sentences": ["FluxQubit designs exploit JosephsonJunction loops to maintain quantum coherence."],
            "timestamp": None,
        },
        {
            "passage_id": "p_distractor_1",
            "title": "Quantum Decoherence",
            "text": "Quantum noise degrades coherence in high temperature quantum environments.",
            "sentences": ["Quantum noise degrades coherence in high temperature quantum environments."],
            "timestamp": None,
        },
        {
            "passage_id": "p_distractor_2",
            "title": "Quantum Optics",
            "text": "Laser beams maintain quantum phase coherence in optical cavities.",
            "sentences": ["Laser beams maintain quantum phase coherence in optical cavities."],
            "timestamp": None,
        },
        {
            "passage_id": "p_distractor_3",
            "title": "Quantum Annealing",
            "text": "Annealing processors optimize coherence protocols across adiabatic schedules.",
            "sentences": ["Annealing processors optimize coherence protocols across adiabatic schedules."],
            "timestamp": None,
        },
        {
            "passage_id": "p_distractor_4",
            "title": "Quantum Theory",
            "text": "Quantum measurement theory analyzes coherence loss during wave function collapse.",
            "sentences": ["Quantum measurement theory analyzes coherence loss during wave function collapse."],
            "timestamp": None,
        },
        {
            "passage_id": "p_distractor_5",
            "title": "Quantum Computing",
            "text": "Quantum gates manipulate coherence vectors in Hilbert state space.",
            "sentences": ["Quantum gates manipulate coherence vectors in Hilbert state space."],
            "timestamp": None,
        },
    ]

    sample_bridge = QuerySample(
        query_id="q_bridge",
        dataset="synthetic",
        question_type="bridge",
        question="How does FluxQubit achieve quantum coherence?",
        gold_answers=["Through JosephsonJunction loops providing inductive coherence."],
        gold_evidence_ids=["p_bridge_1", "p_bridge_2"],
        corpus_passages=passages_bridge,
    )

    # 2. Cross-Session Hub Contamination query: doc_target has the answer.
    # In Flat retrieval: doc_target ranks #1, followed by clean session passages.
    # In Graph retrieval: doc_target contains super-hub entity "AssistantUser" which is linked to 10 noise passages.
    # Unconstrained PPR diffuses into the super-hub, corrupting context (Recall-Relevance Inversion).
    passages_hub = [
        {
            "passage_id": "p_target",
            "title": "Favorite Color",
            "text": "AssistantUser recorded that user favorite color is cerulean blue.",
            "sentences": ["AssistantUser recorded that user favorite color is cerulean blue."],
            "timestamp": None,
        },
        {
            "passage_id": "p_clean_1",
            "title": "Daily Preferences",
            "text": "User prefers morning coffee and quiet rooms.",
            "sentences": ["User prefers morning coffee and quiet rooms."],
            "timestamp": None,
        },
        {
            "passage_id": "p_clean_2",
            "title": "Travel Preferences",
            "text": "User likes seaside destinations and sunny climates.",
            "sentences": ["User likes seaside destinations and sunny climates."],
            "timestamp": None,
        },
        {
            "passage_id": "p_clean_3",
            "title": "Reading Habits",
            "text": "User enjoys historical fiction novels on weekends.",
            "sentences": ["User enjoys historical fiction novels on weekends."],
            "timestamp": None,
        },
        {
            "passage_id": "p_clean_4",
            "title": "Music Habits",
            "text": "User listens to classical symphony orchestras in the evening.",
            "sentences": ["User listens to classical symphony orchestras in the evening."],
            "timestamp": None,
        },
    ]
    # Connect 10 noise passages to AssistantUser and User super-hubs
    for i in range(1, 11):
        passages_hub.append(
            {
                "passage_id": f"p_hub_noise_{i}",
                "title": f"Meeting Note {i}",
                "text": f"AssistantUser scheduled sync call agenda with User for item {i}.",
                "sentences": [f"AssistantUser scheduled sync call agenda with User for item {i}."],
                "timestamp": None,
            }
        )

    sample_hub = QuerySample(
        query_id="q_hub_contamination",
        dataset="synthetic",
        question_type="single-session-assistant",
        question="What is the user favorite color?",
        gold_answers=["The user favorite color is cerulean blue."],
        gold_evidence_ids=["p_target"],
        corpus_passages=passages_hub,
    )

    return [sample_bridge, sample_hub]


def test_run_qpp_study_tracer_bullet_behavior():
    """Verify primary study seam, passage-budget parity, candidate subgraph, and classical QPP."""
    samples = create_planted_scenarios()
    config = StudyConfig(
        retrieval_top_k=5,
        ppr_alpha=0.15,
        ppr_shallow_steps=2,
        ppr_deep_steps=10,
        gain_margin_thresholds=[0.0, 0.05, 0.10, 0.15],
    )
    embedder = FakeEmbedder()
    oracle_judge = FakeOracleGeneratorJudge()

    report: StudyReport = run_qpp_study(
        samples=samples,
        config=config,
        embedder=embedder,
        oracle_judge=oracle_judge,
    )

    # 1. Report structure and query execution
    assert len(report.query_results) == 2
    res_bridge = report.query_results["q_bridge"]
    res_hub = report.query_results["q_hub_contamination"]

    # 2. Strict Passage-Budget Parity (ADR-0001): exactly top-5 passages each
    assert len(res_bridge.flat_passages) == 5
    assert len(res_bridge.graph_passages) == 5
    assert len(res_hub.flat_passages) == 5
    assert len(res_hub.graph_passages) == 5

    # 3. Ground-truth gain behavior:
    # Bridge query should have positive GraphRAG gain (graph connects the bridge)
    assert res_bridge.lexical_gain > 0
    assert res_bridge.semantic_gain > 0

    # Cross-session hub contamination query should suffer Recall-Relevance Inversion (negative GraphRAG gain)
    assert res_hub.lexical_gain < 0
    assert res_hub.semantic_gain < 0

    # 4. Candidate Subgraph extraction and invariants (ADR-0003, ADR-0006)
    subgraph = res_bridge.candidate_subgraph
    assert subgraph is not None
    assert subgraph.num_nodes > 0
    assert subgraph.ppr_shallow is not None
    assert subgraph.ppr_deep is not None
    assert subgraph.ppr_drift is not None
    assert isinstance(subgraph.ppr_entropy_drift, float)
    assert subgraph.directional_distance_encoding is not None
    # Verify no raw 1024-dim text embeddings in candidate subgraph node features
    assert subgraph.node_features.shape[1] < 32  # Strictly compact domain-agnostic features (12 features)

    # 5. Classical Flat QPP evaluation under Conservative Graph Abstention (ADR-0005)
    assert "avg_idf" in report.classical_qpp_results
    assert "nqc" in report.classical_qpp_results
    assert "wig" in report.classical_qpp_results
    assert "mst_homophily" in report.classical_qpp_results

    # Conservative Graph Abstention invariant: routing selection with tau=0.10
    abstention_stats = report.evaluate_conservative_abstention(tau=0.10, target="semantic")
    assert "flat_selected_count" in abstention_stats
    assert "graph_selected_count" in abstention_stats

    # Classical QPP routing policy evaluation
    classical_routing = report.evaluate_classical_qpp_routing(predictor_key="nqc", percentile_threshold=50.0)
    assert "flat_selected_count" in classical_routing
    assert "graph_selected_count" in classical_routing
