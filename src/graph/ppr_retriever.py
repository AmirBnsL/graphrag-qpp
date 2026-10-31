"""PPR Retriever and Candidate Subgraph Extractor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from src.graph.trigraph_builder import TriGraph


@dataclass
class CandidateSubgraphData:
    """Extracted Candidate Subgraph containing domain-agnostic features and topology."""

    num_nodes: int
    node_features: np.ndarray  # Shape: [N, D] where D < 32
    edge_index: np.ndarray     # Shape: [2, E]
    ppr_shallow: np.ndarray    # Shape: [N]
    ppr_deep: np.ndarray       # Shape: [N]
    ppr_drift: np.ndarray      # Shape: [N]
    directional_distance_encoding: np.ndarray  # Shape: [N]
    node_labels: list[str]


def _power_iteration_ppr(
    adj: sparse.csr_matrix,
    p0: np.ndarray,
    alpha: float,
    steps: int,
) -> np.ndarray:
    """Run power-iteration Personalized PageRank for a fixed number of steps."""
    n = adj.shape[0]
    deg = np.array(adj.sum(axis=1)).flatten()
    inv_deg = np.zeros(n, dtype=np.float32)
    nonzero_mask = deg > 0
    inv_deg[nonzero_mask] = 1.0 / deg[nonzero_mask]

    # Row-normalized transition matrix P
    diag_inv_deg = sparse.diags(inv_deg)
    p_trans = diag_inv_deg.dot(adj)

    # Power iteration: p_{k+1} = (1 - alpha) * P^T * p_k + alpha * p0
    p = p0.copy()
    for _ in range(steps):
        p_next = (1.0 - alpha) * p_trans.T.dot(p) + alpha * p0
        p = p_next

    # Re-normalize
    s = p.sum()
    if s > 0:
        p = p / s
    return p


class PPRRetriever:
    """Executes Flat Dense retrieval, Tri-Graph PPR retrieval, and Candidate Subgraph slicing."""

    def __init__(
        self,
        alpha: float = 0.15,
        shallow_steps: int = 2,
        deep_steps: int = 10,
        subgraph_max_entities: int = 30,
        subgraph_max_passages: int = 15,
    ) -> None:
        self.alpha = alpha
        self.shallow_steps = shallow_steps
        self.deep_steps = deep_steps
        self.subgraph_max_entities = subgraph_max_entities
        self.subgraph_max_passages = subgraph_max_passages

    def retrieve_flat(
        self,
        query_vec: np.ndarray,
        graph: TriGraph,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Flat dense cosine similarity retrieval over passage nodes."""
        p_indices = graph.passage_indices
        p_embeds = graph.node_embeddings[p_indices]
        scores = np.dot(p_embeds, query_vec)

        ranked = np.argsort(-scores)[:top_k]
        return [graph.passages[i] for i in ranked]

    def retrieve_graph(
        self,
        query_vec: np.ndarray,
        graph: TriGraph,
        flat_top_passages: list[dict[str, Any]],
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], CandidateSubgraphData]:
        """Dual-horizon PPR retrieval and candidate subgraph extraction."""
        adj = graph.adjacency_matrix
        num_nodes = adj.shape[0]

        # 1. Compute seed personalization vector p0
        p0 = np.zeros(num_nodes, dtype=np.float32)
        deg = np.array(adj.sum(axis=1)).flatten()

        # Entity seeds: rank entities by cosine similarity with query, weighted by inverse degree
        e_indices = graph.entity_indices
        if e_indices:
            e_embeds = graph.node_embeddings[e_indices]
            e_scores = np.dot(e_embeds, query_vec)
            # Pick top-5 entity seeds with score > 0
            top_e_rel = np.argsort(-e_scores)[:5]
            for rel_idx in top_e_rel:
                sc = float(e_scores[rel_idx])
                if sc > 0:
                    global_idx = e_indices[rel_idx]
                    d = max(deg[global_idx], 1.0)
                    # Degree penalty (inverse log degree)
                    p0[global_idx] = sc / np.log1p(d)

        # Passage seeds: top-3 passages by cosine similarity
        p_indices = graph.passage_indices
        p_embeds = graph.node_embeddings[p_indices]
        p_scores = np.dot(p_embeds, query_vec)
        top_p_rel = np.argsort(-p_scores)[:3]
        for rel_idx in top_p_rel:
            sc = float(p_scores[rel_idx])
            if sc > 0:
                global_idx = p_indices[rel_idx]
                p0[global_idx] += sc * 0.5

        if p0.sum() == 0:
            p0 = np.ones(num_nodes, dtype=np.float32) / num_nodes
        else:
            p0 = p0 / p0.sum()

        seed_nodes = np.where(p0 > 0)[0].tolist()

        # 2. Run dual-horizon PPR
        p_shallow = _power_iteration_ppr(adj, p0, self.alpha, self.shallow_steps)
        p_deep = _power_iteration_ppr(adj, p0, self.alpha, self.deep_steps)
        p_drift = p_deep - p_shallow

        # 3. Select top-k Graph Passages using full PPR propagation (ADR-0006)
        p_ppr_scores = p_deep[p_indices]
        graph_ranked = np.argsort(-p_ppr_scores)[:top_k]
        graph_passages = [graph.passages[i] for i in graph_ranked]

        # 4. Extract Candidate Subgraph
        # Nodes included: seeds, top entities from shallow PPR, top passages
        candidate_nodes: list[int] = list(seed_nodes)

        # Top entities from shallow PPR
        if e_indices:
            e_ppr = p_shallow[e_indices]
            top_e = [e_indices[i] for i in np.argsort(-e_ppr)[: self.subgraph_max_entities]]
            candidate_nodes.extend(top_e)

        # Top passages
        top_p = [p_indices[i] for i in np.argsort(-p_ppr_scores)[: self.subgraph_max_passages]]
        candidate_nodes.extend(top_p)

        # Also include connecting sentences for selected passages/entities
        for p_idx in graph_ranked:
            candidate_nodes.extend([graph.sentence_indices[s] for s in graph.passage_to_sentences.get(p_idx, [])])

        subgraph_nodes = sorted(list(dict.fromkeys(candidate_nodes)))
        node_map = {orig: new for new, orig in enumerate(subgraph_nodes)}
        n_sub = len(subgraph_nodes)

        # Sliced adjacency and edge index
        sub_adj = adj[subgraph_nodes, :][:, subgraph_nodes].tocoo()
        edge_index = np.stack([sub_adj.row, sub_adj.col], axis=0)

        # Directional Distance Encoding (BFS from seed entities to nodes in subgraph)
        dde = np.full(n_sub, fill_value=4, dtype=np.int32)
        adj_list: dict[int, list[int]] = {i: [] for i in range(n_sub)}
        for u, v in zip(sub_adj.row, sub_adj.col):
            adj_list[u].append(v)

        sub_seeds = [node_map[s] for s in seed_nodes if s in node_map]
        queue = deque([(s, 0) for s in sub_seeds])
        visited = set(sub_seeds)
        for s in sub_seeds:
            dde[s] = 0

        while queue:
            curr, dist = queue.popleft()
            for neighbor in adj_list[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_dist = min(dist + 1, 4)
                    dde[neighbor] = new_dist
                    queue.append((neighbor, new_dist))

        # Node Features (Domain-Agnostic): D < 32
        sub_embeds = graph.node_embeddings[subgraph_nodes]
        query_sim = np.dot(sub_embeds, query_vec).reshape(-1, 1)

        sub_p_shallow = p_shallow[subgraph_nodes].reshape(-1, 1)
        sub_p_deep = p_deep[subgraph_nodes].reshape(-1, 1)
        sub_drift = p_drift[subgraph_nodes].reshape(-1, 1)

        sub_deg = np.log1p(np.array(sub_adj.sum(axis=1))).reshape(-1, 1)
        global_deg = np.log1p(np.array(adj[subgraph_nodes, :].sum(axis=1))).reshape(-1, 1)

        dde_feat = (dde / 4.0).reshape(-1, 1)

        flat_pids = {p["passage_id"] for p in flat_top_passages}
        is_flat = np.array(
            [
                1.0
                if (orig in graph.passage_indices and graph.passages[orig]["passage_id"] in flat_pids)
                else 0.0
                for orig in subgraph_nodes
            ]
        ).reshape(-1, 1)

        is_p = np.array([1.0 if orig in graph.passage_indices else 0.0 for orig in subgraph_nodes]).reshape(-1, 1)
        is_s = np.array([1.0 if orig in graph.sentence_indices else 0.0 for orig in subgraph_nodes]).reshape(-1, 1)
        is_e = np.array([1.0 if orig in graph.entity_indices else 0.0 for orig in subgraph_nodes]).reshape(-1, 1)

        node_features = np.concatenate(
            [
                query_sim,
                sub_p_shallow,
                sub_p_deep,
                sub_drift,
                sub_deg,
                global_deg,
                dde_feat,
                is_flat,
                is_p,
                is_s,
                is_e,
            ],
            axis=1,
        ).astype(np.float32)

        node_labels = []
        for orig in subgraph_nodes:
            if orig in graph.passage_indices:
                node_labels.append(graph.passages[orig]["passage_id"])
            elif orig in graph.sentence_indices:
                node_labels.append(f"sent_{orig}")
            else:
                e_idx = orig - len(graph.passage_indices) - len(graph.sentence_indices)
                node_labels.append(graph.entities[e_idx])

        subgraph_data = CandidateSubgraphData(
            num_nodes=n_sub,
            node_features=node_features,
            edge_index=edge_index,
            ppr_shallow=sub_p_shallow.flatten(),
            ppr_deep=sub_p_deep.flatten(),
            ppr_drift=sub_drift.flatten(),
            directional_distance_encoding=dde,
            node_labels=node_labels,
        )

        return graph_passages, subgraph_data
