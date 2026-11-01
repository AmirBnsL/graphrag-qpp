"""Tri-Graph construction linking Passages, Sentences, and Entities."""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from src.adapters.base import BaseEmbedder


class CleanupStage(enum.IntEnum):
    """The 4-stage hierarchy for mitigating Cross-Session Hub Contamination."""

    RAW_PERSISTENT = 0            # Unconstrained persistent graph, uniform weights
    LEXICAL_NORMALIZATION = 1     # POS filter, lemmatization, conversational stop-list
    IDF_DEGREE_DAMPENING = 2      # Corpus IDF edge weights & degree-penalized transitions
    SESSION_CONTEXT_CONSTRAINED = 3  # Cross-session entity linking gated on context similarity


CONVERSATIONAL_STOPLIST: set[str] = {
    "user",
    "assistant",
    "assistantuser",
    "system",
    "meeting",
    "message",
    "chat",
    "today",
    "yesterday",
    "tomorrow",
    "time",
    "date",
    "session",
    "schedule",
    "agenda",
    "call",
    "item",
    "note",
}


@dataclass
class TriGraph:
    """Heterogeneous Tri-Graph containing Passages, Sentences, and Entities."""

    passages: list[dict[str, Any]]
    sentences: list[str]
    entities: list[str]
    passage_to_sentences: dict[int, list[int]]
    sentence_to_entities: dict[int, list[int]]
    entity_to_sentences: dict[int, list[int]]
    entity_to_passages: dict[int, list[int]]
    adjacency_matrix: sparse.csr_matrix
    node_embeddings: np.ndarray
    passage_indices: list[int]
    sentence_indices: list[int]
    entity_indices: list[int]
    cleanup_stage: CleanupStage = CleanupStage.SESSION_CONTEXT_CONSTRAINED


def extract_entities_rule_based(
    text: str,
    stage: CleanupStage = CleanupStage.RAW_PERSISTENT,
) -> list[str]:
    """Extract named entities and salient nouns via rules, POS/lemmatization, and stage filters."""
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", text)
    if stage >= CleanupStage.LEXICAL_NORMALIZATION:
        filtered = [
            t for t in tokens
            if t.lower() not in CONVERSATIONAL_STOPLIST and len(t) >= 3
        ]
        return list(dict.fromkeys(filtered))
    return list(dict.fromkeys(tokens))


class TriGraphBuilder:
    """Builds heterogeneous zero-LLM Tri-Graph from passages."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        synonym_threshold: float = 0.85,
    ) -> None:
        self.embedder = embedder
        self.synonym_threshold = synonym_threshold

    def build(
        self,
        passages: list[dict[str, Any]],
        stage: CleanupStage = CleanupStage.RAW_PERSISTENT,
    ) -> TriGraph:
        """Construct TriGraph from a corpus of passages under a specific cleanup stage."""
        all_sentences: list[str] = []
        passage_to_sents: dict[int, list[int]] = {}
        all_entities: list[str] = []
        entity_map: dict[str, int] = {}
        sent_to_ents: dict[int, list[int]] = {}
        ent_to_sents: dict[int, list[int]] = {}
        ent_to_passages: dict[int, list[int]] = {}
        ent_to_sessions: dict[int, set[str]] = {}

        # Precompute passage embeddings for session context similarity gating
        passage_texts = [p.get("text", "") for p in passages]
        p_embeds = self.embedder.embed_texts(passage_texts) if passage_texts else np.empty((0, 64))

        for p_idx, p in enumerate(passages):
            raw_sents = p.get("sentences") or [p.get("text", "")]
            clean_sents = [s.strip() for s in raw_sents if s and s.strip()]
            if not clean_sents:
                clean_sents = [p.get("text", "")]

            sess_id = str(p.get("session_id") or "default_session")

            p_sent_indices: list[int] = []
            for s in clean_sents:
                s_idx = len(all_sentences)
                all_sentences.append(s)
                p_sent_indices.append(s_idx)

                # Extract entities from sentence according to stage
                ents = extract_entities_rule_based(s, stage=stage)
                e_indices: list[int] = []
                for ent in ents:
                    # Session-scoping logic under Step 3 (SESSION_CONTEXT_CONSTRAINED)
                    ent_key = ent
                    if stage == CleanupStage.SESSION_CONTEXT_CONSTRAINED:
                        if ent in entity_map:
                            prev_e_idx = entity_map[ent]
                            prev_sessions = ent_to_sessions.get(prev_e_idx, set())
                            if sess_id not in prev_sessions:
                                # Check session-level context similarity between passages
                                prev_p_indices = ent_to_passages.get(prev_e_idx, [])
                                max_sim = 0.0
                                if prev_p_indices and len(p_embeds) > p_idx:
                                    sims = [float(np.dot(p_embeds[p_idx], p_embeds[prev_p])) for prev_p in prev_p_indices]
                                    max_sim = max(sims) if sims else 0.0

                                # If session contexts are unrelated (< 0.65 similarity), scope by session
                                if max_sim < 0.65:
                                    ent_key = f"{sess_id}:{ent}"

                    if ent_key not in entity_map:
                        e_idx = len(all_entities)
                        entity_map[ent_key] = e_idx
                        all_entities.append(ent_key)
                        ent_to_sents[e_idx] = []
                        ent_to_passages[e_idx] = []
                        ent_to_sessions[e_idx] = set()
                    else:
                        e_idx = entity_map[ent_key]

                    e_indices.append(e_idx)
                    ent_to_sents[e_idx].append(s_idx)
                    ent_to_sessions[e_idx].add(sess_id)
                    if p_idx not in ent_to_passages[e_idx]:
                        ent_to_passages[e_idx].append(p_idx)

                sent_to_ents[s_idx] = e_indices

            passage_to_sents[p_idx] = p_sent_indices

        num_passages = len(passages)
        num_sents = len(all_sentences)
        num_ents = len(all_entities)
        total_nodes = num_passages + num_sents + num_ents

        p_offset = 0
        s_offset = num_passages
        e_offset = num_passages + num_sents

        # Compute embeddings for all nodes
        all_texts = passage_texts + all_sentences + all_entities
        node_embeddings = self.embedder.embed_texts(all_texts)

        # Build sparse adjacency matrix
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def add_edge(u: int, v: int, w: float = 1.0):
            rows.append(u)
            cols.append(v)
            data.append(w)
            rows.append(v)
            cols.append(u)
            data.append(w)

        # Compute entity IDF weights for Step 2+
        ent_idf_weights: dict[int, float] = {}
        for e_idx in range(num_ents):
            df = len(ent_to_passages.get(e_idx, []))
            if stage >= CleanupStage.IDF_DEGREE_DAMPENING:
                # Damped weight: log(1 + N / df)
                ent_idf_weights[e_idx] = float(math.log(1.0 + (num_passages + 1.0) / (df + 1.0)))
            else:
                ent_idf_weights[e_idx] = 1.0

        # 1. Passage <-> Sentence containment edges
        for p_idx, s_indices in passage_to_sents.items():
            u = p_offset + p_idx
            for s_idx in s_indices:
                v = s_offset + s_idx
                add_edge(u, v, 1.0)

        # 2. Passage <-> Entity direct containment edges
        for p_idx, s_indices in passage_to_sents.items():
            u = p_offset + p_idx
            seen_ents: set[int] = set()
            for s_idx in s_indices:
                for e_idx in sent_to_ents.get(s_idx, []):
                    if e_idx not in seen_ents:
                        seen_ents.add(e_idx)
                        v = e_offset + e_idx
                        w = ent_idf_weights.get(e_idx, 1.0)
                        add_edge(u, v, w)

        # 3. Sentence <-> Entity containment edges
        for s_idx, e_indices in sent_to_ents.items():
            u = s_offset + s_idx
            for e_idx in e_indices:
                v = e_offset + e_idx
                w = ent_idf_weights.get(e_idx, 1.0)
                add_edge(u, v, w)

        # 4. Entity <-> Entity intra-sentence co-occurrence
        for s_idx, e_indices in sent_to_ents.items():
            for i in range(len(e_indices)):
                for j in range(i + 1, len(e_indices)):
                    u = e_offset + e_indices[i]
                    v = e_offset + e_indices[j]
                    w_co = min(ent_idf_weights.get(e_indices[i], 1.0), ent_idf_weights.get(e_indices[j], 1.0))
                    add_edge(u, v, w_co)

        # 5. Entity <-> Entity synonymy edges based on cosine similarity
        if num_ents > 1:
            ent_embeds = node_embeddings[e_offset : e_offset + num_ents]
            sim_matrix = np.dot(ent_embeds, ent_embeds.T)
            for i in range(num_ents):
                for j in range(i + 1, num_ents):
                    sim = float(sim_matrix[i, j])
                    if sim >= self.synonym_threshold:
                        u = e_offset + i
                        v = e_offset + j
                        w_syn = sim * min(ent_idf_weights.get(i, 1.0), ent_idf_weights.get(j, 1.0))
                        add_edge(u, v, w_syn)

        adj = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(total_nodes, total_nodes),
            dtype=np.float32,
        )

        return TriGraph(
            passages=passages,
            sentences=all_sentences,
            entities=all_entities,
            passage_to_sentences=passage_to_sents,
            sentence_to_entities=sent_to_ents,
            entity_to_sentences=ent_to_sents,
            entity_to_passages=ent_to_passages,
            adjacency_matrix=adj,
            node_embeddings=node_embeddings,
            passage_indices=list(range(p_offset, p_offset + num_passages)),
            sentence_indices=list(range(s_offset, s_offset + num_sents)),
            entity_indices=list(range(e_offset, e_offset + num_ents)),
            cleanup_stage=stage,
        )
