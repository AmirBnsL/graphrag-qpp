"""Base interfaces for Embedder and OracleGeneratorJudge adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for dense embedding models."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Compute normalized dense embeddings for a list of texts."""
        pass


class BaseOracleGeneratorJudge(ABC):
    """Abstract interface for LLM RAG generator and cross-family judge."""

    @abstractmethod
    def generate(self, query: str, passages: list[dict[str, Any]]) -> str:
        """Generate RAG response given query and retrieved passage chunks."""
        pass

    @abstractmethod
    def judge(
        self,
        query: str,
        flat_answer: str,
        graph_answer: str,
        gold_answers: list[str],
    ) -> tuple[float, float, float]:
        """Judge Flat vs Graph answers on {0, 0.5, 1.0} rubric and pairwise preference margin."""
        pass
