"""Production dense embedder using Qwen3-Embedding (ADR-0004)."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from src.adapters.base import BaseEmbedder

# Prevent headless Colab from blocking on web UI secrets vault
os.environ.setdefault("HF_HUB_DISABLE_COLAB_SECRETS", "1")
os.environ.setdefault("VERTEX_PRODUCT", "COLAB_ENTERPRISE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)


class QwenEmbedder(BaseEmbedder):
    """Dense text embedder wrapping Qwen3-Embedding models with batching and L2 normalization."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str | None = None,
        batch_size: int = 16,
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading Qwen embedding model '%s' on %s (max_seq_length=%d)", self.model_name, self.device, self.max_seq_length)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Compute L2-normalized dense embeddings for a list of texts."""
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        model = self._load_model()
        with torch.inference_mode():
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
        vecs = np.asarray(embeddings, dtype=np.float32)
        if not self.normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
        return vecs

    def unload(self) -> None:
        """Unload embedding model and clear PyTorch CUDA cache."""
        if self._model is not None:
            import gc

            logger.info("Unloading embedder model '%s'", self.model_name)
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
