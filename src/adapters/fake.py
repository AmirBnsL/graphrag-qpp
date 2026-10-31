"""Deterministic in-memory fake adapters for testing without GPU compute."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from src.adapters.base import BaseEmbedder, BaseOracleGeneratorJudge


def _text_to_feature_vector(text: str, dim: int = 64) -> np.ndarray:
    """Hash tokens into a normalized vector space with deterministic semantic clusters."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        vec[0] = 1.0
        return vec

    # Semantic keyword boosts for planted tests
    concept_map = {
        "fluxqubit": 1,
        "coherence": 2,
        "josephsonjunction": 3,
        "quantum": 4,
        "inductance": 5,
        "favorite": 10,
        "color": 11,
        "cerulean": 12,
        "assistantuser": 15,
        "user": 16,
        "agenda": 20,
        "meeting": 21,
    }

    for tok in tokens:
        if tok in concept_map:
            idx = concept_map[tok]
            vec[idx] += 3.0  # Dominant semantic weight
        else:
            # Deterministic MD5 hash index
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little")
            idx = 25 + (h % (dim - 25))
            vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


class FakeEmbedder(BaseEmbedder):
    """In-memory embedder producing reproducible normalized embeddings."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = [_text_to_feature_vector(t, self.dim) for t in texts]
        return np.stack(vectors, axis=0)


class FakeOracleGeneratorJudge(BaseOracleGeneratorJudge):
    """Deterministic generator and judge that checks planted evidence presence."""

    def generate(self, query: str, passages: list[dict[str, Any]]) -> str:
        pids = [p["passage_id"] for p in passages]
        q_lower = query.lower()

        # Bridge query logic
        if "fluxqubit" in q_lower:
            has_p1 = "p_bridge_1" in pids
            has_p2 = "p_bridge_2" in pids
            if has_p1 and has_p2:
                return "Through JosephsonJunction loops providing inductive coherence."
            elif has_p2:
                return "Through quantum loops."
            return "Unknown quantum architecture."

        # Cross-session hub contamination query logic
        if "favorite color" in q_lower:
            has_target = "p_target" in pids
            noise_count = sum(1 for pid in pids if "noise" in pid)
            if has_target and noise_count <= 1:
                return "The user favorite color is cerulean blue."
            elif has_target and noise_count > 1:
                # Distractor pollution / hub confusion
                return "The user discussed meetings and might like various colors."
            return "No color information found."

        return "Default answer."

    def judge(
        self,
        query: str,
        flat_answer: str,
        graph_answer: str,
        gold_answers: list[str],
    ) -> tuple[float, float, float]:
        gold = gold_answers[0].lower() if gold_answers else ""

        def score(ans: str) -> float:
            ans_l = ans.lower()
            if "cerulean blue" in gold:
                if "cerulean blue" in ans_l:
                    return 1.0
                if "various colors" in ans_l:
                    return 0.3
                return 0.0
            if "josephsonjunction" in gold.lower():
                if "josephsonjunction loops" in ans_l:
                    return 1.0
                if "quantum loops" in ans_l:
                    return 0.5
                return 0.0
            return 1.0 if gold in ans_l else 0.0

        s_flat = score(flat_answer)
        s_graph = score(graph_answer)
        margin = s_graph - s_flat
        return (s_flat, s_graph, margin)
