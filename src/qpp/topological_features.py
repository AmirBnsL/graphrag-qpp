"""Handcrafted topological feature extraction from Candidate Subgraphs (ADR-0003, Issue #4)."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from src.eval.retrieval_metrics import compute_access_gini, compute_ppr_entropy
from src.graph.ppr_retriever import CandidateSubgraphData


def extract_topological_features(
    candidate_subgraph: CandidateSubgraphData,
    flat_passages: list[dict[str, Any]],
    graph_passages: list[dict[str, Any]],
) -> dict[str, float]:
    """Extract a compact, domain-agnostic vector of handcrafted topological descriptors (ADR-0003, Issue #4).

    Features computed:
      1. seed_deg_mean: Mean degree of seed nodes in candidate subgraph.
      2. seed_deg_max: Maximum degree of seed nodes in candidate subgraph.
      3. neighbor_to_seed_hub_ratio: Ratio of average seed neighbor degree to seed degree.
      4. subgraph_deg_gini: Gini coefficient of subgraph node degree distribution.
      5. lcc_ratio: Relative size of largest connected component (|V_LCC| / |V_q|).
      6. leaf_fraction: Fraction of subgraph nodes with degree <= 1.
      7. max_k_core: Maximum core number in candidate subgraph.
      8. ppr_shallow_entropy: Shannon entropy H(p_shallow) at t=2.
      9. ppr_deep_entropy: Shannon entropy H(p_deep) at t=10.
      10. ppr_entropy_drift: PPR Diffusion Drift ΔH_PPR = H(p_deep) - H(p_shallow).
      11. dde_passage_mean: Mean Directional Distance Encoding to candidate passages.
      12. dde_passage_max: Max Directional Distance Encoding to candidate passages.
      13. dde_passage_min: Min Directional Distance Encoding to candidate passages.
      14. anti_homophily_bridge_score: Anti-homophily multi-hop bridge dispersion score.
      15. flat_graph_jaccard: Jaccard overlap between Flat and Graph retrieved passages.
    """
    n = candidate_subgraph.num_nodes
    if n == 0:
        return {
            "seed_deg_mean": 0.0,
            "seed_deg_max": 0.0,
            "neighbor_to_seed_hub_ratio": 0.0,
            "subgraph_deg_gini": 0.0,
            "lcc_ratio": 0.0,
            "leaf_fraction": 0.0,
            "max_k_core": 0.0,
            "ppr_shallow_entropy": 0.0,
            "ppr_deep_entropy": 0.0,
            "ppr_entropy_drift": 0.0,
            "dde_passage_mean": 0.0,
            "dde_passage_max": 0.0,
            "dde_passage_min": 0.0,
            "anti_homophily_bridge_score": 0.0,
            "flat_graph_jaccard": 0.0,
        }

    # Build NetworkX subgraph for structural topology analysis
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if candidate_subgraph.edge_index.shape[1] > 0:
        edges = list(
            zip(
                candidate_subgraph.edge_index[0].tolist(),
                candidate_subgraph.edge_index[1].tolist(),
            )
        )
        g.add_edges_from(edges)

    degrees = np.array([d for _, d in g.degree()], dtype=np.float32)

    # 1. Seed degree statistics & 3. Neighbor-to-seed hub ratio
    dde = candidate_subgraph.directional_distance_encoding
    seed_mask = dde == 0
    seed_indices = np.where(seed_mask)[0]

    if len(seed_indices) > 0:
        seed_deg_mean = float(np.mean(degrees[seed_indices]))
        seed_deg_max = float(np.max(degrees[seed_indices]))

        neighbor_degs: list[float] = []
        for s in seed_indices:
            for nbr in g.neighbors(int(s)):
                neighbor_degs.append(float(g.degree(nbr)))

        if neighbor_degs and seed_deg_mean > 0:
            neighbor_to_seed_hub_ratio = float(np.mean(neighbor_degs)) / seed_deg_mean
        else:
            neighbor_to_seed_hub_ratio = 0.0
    else:
        seed_deg_mean = 0.0
        seed_deg_max = 0.0
        neighbor_to_seed_hub_ratio = 0.0

    # 4. Subgraph degree Gini
    subgraph_deg_gini = compute_access_gini(degrees, only_positive=False)

    # 5. Largest connected component ratio
    components = list(nx.connected_components(g))
    lcc_size = max(len(c) for c in components) if components else 0
    lcc_ratio = float(lcc_size / n)

    # 6. Low-degree leaf fraction
    leaf_fraction = float(np.sum(degrees <= 1.0) / n)

    # 7. Maximum k-core number
    core_numbers = nx.core_number(g) if g.number_of_nodes() > 0 else {}
    max_k_core = float(max(core_numbers.values())) if core_numbers else 0.0

    # 8. Shallow PPR entropy & 9. Deep PPR entropy & 10. Drift
    ppr_shallow_entropy = compute_ppr_entropy(candidate_subgraph.ppr_shallow)
    ppr_deep_entropy = compute_ppr_entropy(candidate_subgraph.ppr_deep)
    ppr_entropy_drift = float(candidate_subgraph.ppr_entropy_drift)

    # 11-13. Directional Distance Encoding statistics to candidate passages
    # Column 9 of node_features is is_p indicator
    is_p_mask = candidate_subgraph.node_features[:, 9] > 0.5
    if np.any(is_p_mask):
        p_dde = dde[is_p_mask]
        dde_passage_mean = float(np.mean(p_dde))
        dde_passage_max = float(np.max(p_dde))
        dde_passage_min = float(np.min(p_dde))
    else:
        dde_passage_mean = 0.0
        dde_passage_max = 0.0
        dde_passage_min = 0.0

    # 14. Anti-homophily bridge score (Issue #1 User Story 14)
    # Reflects disparity between high seed query similarity and low passage query similarity
    query_sims = candidate_subgraph.node_features[:, 0]
    if len(seed_indices) > 0 and np.any(is_p_mask):
        seed_sim_mean = float(np.mean(query_sims[seed_indices]))
        passage_sim_mean = float(np.mean(query_sims[is_p_mask]))
        anti_homophily_bridge_score = float(
            max(0.0, seed_sim_mean - passage_sim_mean) + (1.0 - passage_sim_mean) / 2.0
        )
    else:
        anti_homophily_bridge_score = 0.0

    # 15. Flat-vs-Graph passage Jaccard overlap
    flat_ids = {p.get("passage_id", "") for p in flat_passages if p.get("passage_id")}
    graph_ids = {p.get("passage_id", "") for p in graph_passages if p.get("passage_id")}
    union = len(flat_ids | graph_ids)
    intersection = len(flat_ids & graph_ids)
    flat_graph_jaccard = float(intersection / union) if union > 0 else 0.0

    return {
        "seed_deg_mean": seed_deg_mean,
        "seed_deg_max": seed_deg_max,
        "neighbor_to_seed_hub_ratio": neighbor_to_seed_hub_ratio,
        "subgraph_deg_gini": subgraph_deg_gini,
        "lcc_ratio": lcc_ratio,
        "leaf_fraction": leaf_fraction,
        "max_k_core": max_k_core,
        "ppr_shallow_entropy": ppr_shallow_entropy,
        "ppr_deep_entropy": ppr_deep_entropy,
        "ppr_entropy_drift": ppr_entropy_drift,
        "dde_passage_mean": dde_passage_mean,
        "dde_passage_max": dde_passage_max,
        "dde_passage_min": dde_passage_min,
        "anti_homophily_bridge_score": anti_homophily_bridge_score,
        "flat_graph_jaccard": flat_graph_jaccard,
    }
