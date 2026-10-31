"""Classical Flat QPP predictors and MST Passage Graph Homophily baseline."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from src.adapters.base import BaseEmbedder


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def compute_idf_dict(corpus_passages: list[dict[str, Any]]) -> tuple[dict[str, float], int]:
    """Compute token document frequencies across corpus passages."""
    total_docs = len(corpus_passages)
    df: dict[str, int] = {}
    for p in corpus_passages:
        tokens = set(_tokenize(p.get("text", "")))
        for t in tokens:
            df[t] = df.get(t, 0) + 1

    idf: dict[str, float] = {}
    for t, count in df.items():
        idf[t] = math.log((total_docs - count + 0.5) / (count + 0.5) + 1.0)
    return idf, total_docs


def compute_classical_qpp(
    query: str,
    flat_passages: list[dict[str, Any]],
    graph_passages: list[dict[str, Any]],
    corpus_passages: list[dict[str, Any]],
    embedder: BaseEmbedder,
) -> dict[str, float]:
    """Compute pre-retrieval and post-retrieval classical QPP signals."""
    idf_dict, n_docs = compute_idf_dict(corpus_passages)
    q_tokens = _tokenize(query)

    # 1. Pre-retrieval IDF signals
    q_idfs = [idf_dict.get(t, 0.0) for t in q_tokens]
    avg_idf = float(np.mean(q_idfs)) if q_idfs else 0.0
    max_idf = float(np.max(q_idfs)) if q_idfs else 0.0

    # Simplified Clarity Score (SCQ)
    scq = sum((1.0 + math.log(1.0 + q_tokens.count(t))) * idf_dict.get(t, 0.0) for t in set(q_tokens))

    # 2. Post-retrieval Score Variance (NQC & WIG) on Flat top-k
    flat_texts = [p.get("text", "") for p in flat_passages]
    if flat_texts:
        q_vec = embedder.embed_texts([query])[0]
        flat_embeds = embedder.embed_texts(flat_texts)
        scores = np.dot(flat_embeds, q_vec)

        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        nqc = std_s / (mean_s + 1e-9)
        wig = float(np.mean(scores - 0.2))  # Relative to background baseline score
    else:
        nqc = 0.0
        wig = 0.0

    # 3. Minimum Spanning Tree (MST) Pairwise Passage Homophily (IQS5MK62)
    # Evaluates pairwise cosine similarity graph among retrieved passages
    if len(flat_texts) > 1:
        pairwise_sim = np.dot(flat_embeds, flat_embeds.T)
        # Convert similarity to distance matrix: dist = 1 - sim (clipped at 0)
        dist_matrix = np.clip(1.0 - pairwise_sim, 0.0, 2.0)
        mst = minimum_spanning_tree(dist_matrix)
        mst_weight = float(mst.toarray().sum())
    else:
        mst_weight = 0.0

    return {
        "avg_idf": avg_idf,
        "max_idf": max_idf,
        "scq": float(scq),
        "nqc": nqc,
        "wig": wig,
        "mst_homophily": mst_weight,
    }
