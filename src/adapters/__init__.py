"""Adapter package containing abstract interfaces, test fakes, and GPU production adapters."""

from src.adapters.base import BaseEmbedder, BaseOracleGeneratorJudge
from src.adapters.fake import FakeEmbedder, FakeOracleGeneratorJudge
from src.adapters.qwen_embedder import QwenEmbedder
from src.adapters.vllm_adapter import (
    QuantizedVLLMGeneratorJudge,
    build_generator_prompt,
    build_judge_prompt,
    parse_judge_score,
)

__all__ = [
    "BaseEmbedder",
    "BaseOracleGeneratorJudge",
    "FakeEmbedder",
    "FakeOracleGeneratorJudge",
    "QwenEmbedder",
    "QuantizedVLLMGeneratorJudge",
    "build_generator_prompt",
    "build_judge_prompt",
    "parse_judge_score",
]
