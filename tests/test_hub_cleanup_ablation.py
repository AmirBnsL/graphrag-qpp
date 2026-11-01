"""Tests for Ticket 2: 4-stage Cross-Session Hub Contamination Cleanup and Retrieval Ablation."""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.fake import FakeEmbedder
from src.data.loader import QuerySample
from src.graph.trigraph_builder import CleanupStage, TriGraphBuilder
from src.study import CleanupAblationReport, StudyConfig, evaluate_cleanup_ablation


def create_multi_session_scenario() -> list[QuerySample]:
    """Create multi-session conversational samples vulnerable to cross-session hub contamination."""
    # Target session 1 contains favorite color
    # Noise sessions 2 and 3 share generic conversational entities ("User", "AssistantUser", "Meeting")
    passages = [
        {
            "passage_id": "sess1_target",
            "session_id": "sess_1",
            "title": "Color Preferences",
            "text": "User told AssistantUser that favorite color is cerulean blue.",
            "sentences": ["User told AssistantUser that favorite color is cerulean blue."],
            "timestamp": "2026-01-01",
        },
        {
            "passage_id": "sess1_context",
            "session_id": "sess_1",
            "title": "Room Painting",
            "text": "User wants to paint bedroom walls with favorite color cerulean.",
            "sentences": ["User wants to paint bedroom walls with favorite color cerulean."],
            "timestamp": "2026-01-01",
        },
    ]

    # Session 2: Work & Budgeting (Noise linked via User and AssistantUser)
    for i in range(1, 5):
        passages.append(
            {
                "passage_id": f"sess2_budget_{i}",
                "session_id": "sess_2",
                "title": f"Budget Note {i}",
                "text": f"User and AssistantUser discussed quarterly budget review items {i}.",
                "sentences": [f"User and AssistantUser discussed quarterly budget review items {i}."],
                "timestamp": "2026-02-15",
            }
        )

    # Session 3: Travel & Flights (Noise linked via User and AssistantUser)
    for i in range(1, 5):
        passages.append(
            {
                "passage_id": f"sess3_travel_{i}",
                "session_id": "sess_3",
                "title": f"Travel Note {i}",
                "text": f"User and AssistantUser booked flight reservations for conference {i}.",
                "sentences": [f"User and AssistantUser booked flight reservations for conference {i}."],
                "timestamp": "2026-03-20",
            }
        )

    sample = QuerySample(
        query_id="q_multisession_color",
        dataset="longmemeval_synthetic",
        question_type="multi-session",
        question="What is the user favorite color?",
        gold_answers=["cerulean blue"],
        gold_evidence_ids=["sess1_target", "sess1_context"],
        corpus_passages=passages,
    )
    return [sample]


def test_four_stage_cleanup_ablation_behavior():
    """Verify that moving from Step 0 to Step 3 monotonically mitigates hub contamination."""
    samples = create_multi_session_scenario()
    config = StudyConfig(retrieval_top_k=5)
    embedder = FakeEmbedder()

    # Run the 4-stage ablation (zero LLM calls required)
    ablation_report: CleanupAblationReport = evaluate_cleanup_ablation(
        samples=samples,
        config=config,
        embedder=embedder,
    )

    # 1. All 4 stages (0, 1, 2, 3) must be present in the report
    assert len(ablation_report.stage_results) == 4
    for stage in [
        CleanupStage.RAW_PERSISTENT,
        CleanupStage.LEXICAL_NORMALIZATION,
        CleanupStage.IDF_DEGREE_DAMPENING,
        CleanupStage.SESSION_CONTEXT_CONSTRAINED,
    ]:
        assert stage in ablation_report.stage_results

    stage0_raw = ablation_report.stage_results[CleanupStage.RAW_PERSISTENT]
    stage1_lexical = ablation_report.stage_results[CleanupStage.LEXICAL_NORMALIZATION]
    stage2_idf_damped = ablation_report.stage_results[CleanupStage.IDF_DEGREE_DAMPENING]
    stage3_session_gated = ablation_report.stage_results[CleanupStage.SESSION_CONTEXT_CONSTRAINED]

    # 2. Precision@5 must improve or stay high as hub contamination is cleaned up
    # Step 0 suffers from cross-session hub diffusion into sessions 2 and 3
    assert stage3_session_gated.mean_precision >= stage0_raw.mean_precision

    # 3. Hub Access Gini must decrease (indicating more uniform diffusion rather than hub concentration)
    assert stage3_session_gated.mean_access_gini < stage0_raw.mean_access_gini

    # 4. Recall@5 on target evidence should be preserved or improved
    assert stage3_session_gated.mean_recall >= stage0_raw.mean_recall

    # 5. PPR Shannon entropy must decrease (proving suppression of global hub noise dispersion)
    assert stage3_session_gated.mean_ppr_entropy < stage0_raw.mean_ppr_entropy

    # 6. PPR Diffusion Drift ΔH_PPR = H(p_deep) - H(p_shallow) is strictly computed and reported (GLOSSARY.md, ADR-0006)
    assert stage3_session_gated.mean_ppr_entropy_drift > 0.0
    assert stage0_raw.mean_ppr_entropy_drift > 0.0


def test_trigraph_construction_stage_properties():
    """Verify structural graph properties across the 4 stages."""
    embedder = FakeEmbedder()
    builder = TriGraphBuilder(embedder=embedder)
    passages = [
        {
            "passage_id": "p1",
            "session_id": "sess_A",
            "text": "User discussed AlphaProject plans for quarterly execution.",
            "sentences": ["User discussed AlphaProject plans for quarterly execution."],
        },
        {
            "passage_id": "p2",
            "session_id": "sess_B",
            "text": "Assistant discussed AlphaProject cooking recipes with vegetable ingredients.",
            "sentences": ["Assistant discussed AlphaProject cooking recipes with vegetable ingredients."],
        },
    ]

    # Stage 0: Conversational hubs ("User", "Assistant") must be present
    graph_raw = builder.build(passages, stage=CleanupStage.RAW_PERSISTENT)
    assert "User" in graph_raw.entities
    assert "Assistant" in graph_raw.entities
    assert graph_raw.cleanup_stage == CleanupStage.RAW_PERSISTENT

    # Stage 1: Conversational hubs must be filtered by stop-list
    graph_lexical = builder.build(passages, stage=CleanupStage.LEXICAL_NORMALIZATION)
    assert "User" not in graph_lexical.entities
    assert "Assistant" not in graph_lexical.entities
    assert "AlphaProject" in graph_lexical.entities

    # Stage 2: Entity IDF weights should be computed and applied to edges
    graph_idf_damped = builder.build(passages, stage=CleanupStage.IDF_DEGREE_DAMPENING)
    assert graph_idf_damped.cleanup_stage == CleanupStage.IDF_DEGREE_DAMPENING
    # Adjacency matrix should exist and be non-empty
    assert graph_idf_damped.adjacency_matrix.nnz > 0

    # Stage 3: Cross-session context-similarity edge gating gates AlphaProject if sessions A and B are unrelated
    graph_session_gated = builder.build(passages, stage=CleanupStage.SESSION_CONTEXT_CONSTRAINED)
    assert graph_session_gated.cleanup_stage == CleanupStage.SESSION_CONTEXT_CONSTRAINED
    assert "AlphaProject" in graph_session_gated.entities
    # Unrelated sessions A and B have cross-session AlphaProject edges gated, reducing edge count relative to stage 2
    assert graph_session_gated.adjacency_matrix.nnz < graph_idf_damped.adjacency_matrix.nnz


def test_cross_session_context_similarity_retention():
    """Verify that when cross-session passages are contextually similar, edges are preserved."""
    embedder = FakeEmbedder()
    # When threshold is set very low (-1.0), all sessions are deemed related
    builder = TriGraphBuilder(embedder=embedder, session_similarity_threshold=-1.0)
    passages = [
        {
            "passage_id": "p1",
            "session_id": "sess_A",
            "text": "User discussed AlphaProject quarterly plans.",
            "sentences": ["User discussed AlphaProject quarterly plans."],
        },
        {
            "passage_id": "p2",
            "session_id": "sess_B",
            "text": "User discussed AlphaProject quarterly plans.",
            "sentences": ["User discussed AlphaProject quarterly plans."],
        },
    ]

    graph_gated = builder.build(passages, stage=CleanupStage.SESSION_CONTEXT_CONSTRAINED)
    graph_idf = builder.build(passages, stage=CleanupStage.IDF_DEGREE_DAMPENING)
    # With related sessions, edges to AlphaProject are kept open in Stage 3
    assert graph_gated.adjacency_matrix.nnz == graph_idf.adjacency_matrix.nnz

