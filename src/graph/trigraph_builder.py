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


DEFAULT_SESSION_SIMILARITY_THRESHOLD: float = 0.65


def _lemmatize_token(token: str) -> str:
    """Rule-based lemmatization for common noun inflections (Issue #3)."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses") or token.endswith("shes") or token.endswith("ches") or token.endswith("xes") or token.endswith("zes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def extract_entities_rule_based(
    text: str,
    stage: CleanupStage = CleanupStage.RAW_PERSISTENT,
) -> list[str]:
    """Extract named entities and salient nouns via rules, POS/lemmatization, and stage filters."""
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", text)
    if stage >= CleanupStage.LEXICAL_NORMALIZATION:
        lemmatized: list[str] = []
        for t in tokens:
            t_clean = _lemmatize_token(t)
            if t_clean.lower() not in CONVERSATIONAL_STOPLIST and len(t_clean) >= 3:
                lemmatized.append(t_clean)
        return list(dict.fromkeys(lemmatized))
    return list(dict.fromkeys(tokens))


class TriGraphBuilder:
    """Builds heterogeneous zero-LLM Tri-Graph from passages."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        synonym_threshold: float = 0.85,
        session_similarity_threshold: float = DEFAULT_SESSION_SIMILARITY_THRESHOLD,
    ) -> None:
        self.embedder = embedder
        self.synonym_threshold = synonym_threshold
        self.session_similarity_threshold = session_similarity_threshold

    def build(
        self,
        passages: list[dict[str, Any]],
        stage: CleanupStage = CleanupStage.RAW_PERSISTENT,
    ) -> TriGraph:
        """Construct TriGraph from a corpus of passages under a specific cleanup stage."""
        passage_sessions: dict[int, str] = {}
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

            raw_sess = p.get("session_id")
            if not raw_sess:
                pid_str = str(p.get("passage_id") or f"p_{p_idx}")
                raw_sess = re.sub(r"_c\d+$", "", pid_str)
            sess_id = str(raw_sess)
            passage_sessions[p_idx] = sess_id

            p_sent_indices: list[int] = []
            for s in clean_sents:
                s_idx = len(all_sentences)
                all_sentences.append(s)
                p_sent_indices.append(s_idx)

                # Extract entities from sentence according to stage (canonical names preserved)
                ents = extract_entities_rule_based(s, stage=stage)
                e_indices: list[int] = []
                for ent in ents:
                    if ent not in entity_map:
                        e_idx = len(all_entities)
                        entity_map[ent] = e_idx
                        all_entities.append(ent)
                        ent_to_sents[e_idx] = []
                        ent_to_passages[e_idx] = []
                        ent_to_sessions[e_idx] = set()
                    else:
                        e_idx = entity_map[ent]

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

        # Compute embeddings for non-passage nodes and concatenate without redundant passage embedding
        remaining_texts = all_sentences + all_entities
        if remaining_texts:
            remaining_embeds = self.embedder.embed_texts(remaining_texts)
            if len(p_embeds) > 0:
                node_embeddings = np.concatenate([p_embeds, remaining_embeds], axis=0)
            else:
                node_embeddings = remaining_embeds
        else:
            node_embeddings = p_embeds

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

        def should_gate_edge(p_index: int, e_index: int) -> bool:
            """Cross-session context-similarity edge gating (ADR-0006, Issue #3)."""
            if stage != CleanupStage.SESSION_CONTEXT_CONSTRAINED:
                return False
            sess = passage_sessions[p_index]
            other_sessions = ent_to_sessions[e_index] - {sess}
            if not other_sessions:
                return False
            other_p_indices = [
                prev_p for prev_p in ent_to_passages[e_index]
                if passage_sessions[prev_p] in other_sessions
            ]
            if not other_p_indices or len(p_embeds) <= p_index:
                return False
            sims = [float(np.dot(p_embeds[p_index], p_embeds[other_p])) for other_p in other_p_indices]
            max_sim = max(sims) if sims else 0.0
            return max_sim < self.session_similarity_threshold

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
                        if should_gate_edge(p_idx, e_idx):
                            continue
                        v = e_offset + e_idx
                        w = ent_idf_weights.get(e_idx, 1.0)
                        add_edge(u, v, w)

        # 3. Sentence <-> Entity containment edges
        for p_idx, s_indices in passage_to_sents.items():
            for s_idx in s_indices:
                u = s_offset + s_idx
                for e_idx in sent_to_ents.get(s_idx, []):
                    if should_gate_edge(p_idx, e_idx):
                        continue
                    v = e_offset + e_idx
                    w = ent_idf_weights.get(e_idx, 1.0)
                    add_edge(u, v, w)

        # 4. Entity <-> Entity intra-sentence co-occurrence
        for p_idx, s_indices in passage_to_sents.items():
            for s_idx in s_indices:
                e_indices = sent_to_ents.get(s_idx, [])
                for i in range(len(e_indices)):
                    if should_gate_edge(p_idx, e_indices[i]):
                        continue
                    for j in range(i + 1, len(e_indices)):
                        if should_gate_edge(p_idx, e_indices[j]):
                            continue
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
                        if stage == CleanupStage.SESSION_CONTEXT_CONSTRAINED:
                            sessions_i = ent_to_sessions.get(i, set())
                            sessions_j = ent_to_sessions.get(j, set())
                            if not (sessions_i & sessions_j):
                                p_i_list = ent_to_passages.get(i, [])
                                p_j_list = ent_to_passages.get(j, [])
                                max_p_sim = 0.0
                                for pi in p_i_list:
                                    for pj in p_j_list:
                                        if len(p_embeds) > max(pi, pj):
                                            psim = float(np.dot(p_embeds[pi], p_embeds[pj]))
                                            if psim > max_p_sim:
                                                max_p_sim = psim
                                if max_p_sim < self.session_similarity_threshold:
                                    continue
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
