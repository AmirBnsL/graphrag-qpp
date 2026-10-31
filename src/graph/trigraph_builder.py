"""Tri-Graph construction linking Passages, Sentences, and Entities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from src.adapters.base import BaseEmbedder


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


def extract_entities_rule_based(text: str) -> list[str]:
    """Extract named entities and salient nouns via rules/patterns."""
    # Matches CamelCase or capitalized multi-word phrases and salient domain nouns
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", text)
    # Also extract specific phrases
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

    def build(self, passages: list[dict[str, Any]]) -> TriGraph:
        """Construct TriGraph from a corpus of passages."""
        all_sentences: list[str] = []
        passage_to_sents: dict[int, list[int]] = {}
        all_entities: list[str] = []
        entity_map: dict[str, int] = {}
        sent_to_ents: dict[int, list[int]] = {}
        ent_to_sents: dict[int, list[int]] = {}
        ent_to_passages: dict[int, list[int]] = {}

        for p_idx, p in enumerate(passages):
            raw_sents = p.get("sentences") or [p.get("text", "")]
            clean_sents = [s.strip() for s in raw_sents if s and s.strip()]
            if not clean_sents:
                clean_sents = [p.get("text", "")]

            p_sent_indices: list[int] = []
            for s in clean_sents:
                s_idx = len(all_sentences)
                all_sentences.append(s)
                p_sent_indices.append(s_idx)

                # Extract entities from sentence
                ents = extract_entities_rule_based(s)
                e_indices: list[int] = []
                for ent in ents:
                    if ent not in entity_map:
                        e_idx = len(all_entities)
                        entity_map[ent] = e_idx
                        all_entities.append(ent)
                        ent_to_sents[e_idx] = []
                        ent_to_passages[e_idx] = []
                    else:
                        e_idx = entity_map[ent]

                    e_indices.append(e_idx)
                    ent_to_sents[e_idx].append(s_idx)
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
        passage_texts = [p.get("text", "") for p in passages]
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
                        add_edge(u, v, 1.0)

        # 3. Sentence <-> Entity containment edges
        for s_idx, e_indices in sent_to_ents.items():
            u = s_offset + s_idx
            for e_idx in e_indices:
                v = e_offset + e_idx
                add_edge(u, v, 1.0)

        # 4. Entity <-> Entity intra-sentence co-occurrence
        for s_idx, e_indices in sent_to_ents.items():
            for i in range(len(e_indices)):
                for j in range(i + 1, len(e_indices)):
                    u = e_offset + e_indices[i]
                    v = e_offset + e_indices[j]
                    add_edge(u, v, 1.0)

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
                        add_edge(u, v, sim)

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
        )
