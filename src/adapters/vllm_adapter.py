"""Quantized vLLM Generator and Cross-Family Judge Adapter (ADR-0004)."""

from __future__ import annotations

import gc
import json
import logging
import re
from typing import Any

import numpy as np
import torch

from src.adapters.base import BaseOracleGeneratorJudge

logger = logging.getLogger(__name__)


def build_generator_prompt(query: str, passages: list[dict[str, Any]]) -> str:
    """Format RAG prompt with strict Passage-Budget Parity template (ADR-0001)."""
    context = "\n\n".join(f"[{i+1}] {p.get('text', '')}" for i, p in enumerate(passages))
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer the question directly and concisely based on the context above. If the context does not contain enough information, state that clearly.\n"
        f"Answer:"
    )


def build_judge_prompt(
    query: str,
    candidate_answer: str,
    gold_answers: list[str],
) -> str:
    """Format single-answer evaluation prompt with discrete {0.0, 0.5, 1.0} rubric (ADR-0002)."""
    gold_str = " | ".join(gold_answers)
    return (
        f"You are an impartial evaluator for a question-answering benchmark.\n\n"
        f"Question: {query}\n"
        f"Reference ground-truth answer: {gold_str}\n"
        f"Candidate model answer: {candidate_answer}\n\n"
        f"Evaluate the candidate answer on this strict rubric:\n"
        f"- 1.0: Correct and completely answers the question consistent with reference.\n"
        f"- 0.5: Partially correct, incomplete, or contains minor factual discrepancies.\n"
        f"- 0.0: Factually incorrect, contradictory, or completely fails to answer.\n\n"
        f"Provide your rating in this exact format: Score: <0.0, 0.5, or 1.0>"
    )


def parse_judge_score(response_text: str) -> float:
    """Extract numeric score {0.0, 0.5, 1.0} from judge LLM response text."""
    match = re.search(r"Score:\s*([01](?:\.[05])?)", response_text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if val in (0.0, 0.5, 1.0):
            return val
    # Fallback search for any 1.0, 0.5, 0.0 in output
    for token in ["1.0", "0.5", "0.0"]:
        if token in response_text:
            return float(token)
    return 0.0


class QuantizedVLLMGeneratorJudge(BaseOracleGeneratorJudge):
    """Production sequential quantized generator and cross-family judge via vLLM (ADR-0004)."""

    def __init__(
        self,
        generator_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        judge_model: str = "google/gemma-2-9b-it",
        generator_quantization: str = "awq",
        judge_quantization: str | None = None,
        gpu_memory_utilization: float = 0.40,
        max_model_len: int = 4096,
        seed: int = 42,
    ) -> None:
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.generator_quantization = generator_quantization
        self.judge_quantization = judge_quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.seed = seed

        self._generator_llm: Any = None
        self._judge_llm: Any = None

    def _unload_active_model(self) -> None:
        """Explicitly unload active LLM instance and flush PyTorch CUDA cache."""
        if self._generator_llm is not None:
            logger.info("Unloading generator LLM '%s'", self.generator_model)
            del self._generator_llm
            self._generator_llm = None

        if self._judge_llm is not None:
            logger.info("Unloading judge LLM '%s'", self.judge_model)
            del self._judge_llm
            self._judge_llm = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_generator_llm(self) -> Any:
        if self._generator_llm is None:
            # Enforce single model in VRAM at a time (ADR-0004)
            self._unload_active_model()
            from vllm import LLM

            logger.info(
                "Loading quantized generator '%s' (quant=%s, mem_util=%.2f)",
                self.generator_model,
                self.generator_quantization,
                self.gpu_memory_utilization,
            )
            kwargs: dict[str, Any] = {
                "model": self.generator_model,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "seed": self.seed,
                "trust_remote_code": True,
            }
            if self.generator_quantization:
                kwargs["quantization"] = self.generator_quantization

            self._generator_llm = LLM(**kwargs)
        return self._generator_llm

    def _get_judge_llm(self) -> Any:
        if self._judge_llm is None:
            # Enforce single model in VRAM at a time (ADR-0004)
            self._unload_active_model()
            from vllm import LLM

            logger.info(
                "Loading quantized judge '%s' (quant=%s, mem_util=%.2f)",
                self.judge_model,
                self.judge_quantization,
                self.gpu_memory_utilization,
            )
            kwargs: dict[str, Any] = {
                "model": self.judge_model,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "seed": self.seed,
                "trust_remote_code": True,
            }
            if self.judge_quantization:
                kwargs["quantization"] = self.judge_quantization

            self._judge_llm = LLM(**kwargs)
        return self._judge_llm

    def generate(self, query: str, passages: list[dict[str, Any]]) -> str:
        """Generate response for a single query using active generator."""
        results = self.generate_batch([(query, passages)])
        return results[0] if results else ""

    def generate_batch(self, query_passages_pairs: list[tuple[str, list[dict[str, Any]]]]) -> list[str]:
        """Batch-generate answers with temperature=0.0 deterministic greedy decoding."""
        if not query_passages_pairs:
            return []

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=["\nQuestion:", "\nContext:"],
        )
        prompts = [build_generator_prompt(q, p) for q, p in query_passages_pairs]
        llm = self._get_generator_llm()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        return [out.outputs[0].text.strip() for out in outputs]

    def judge(
        self,
        query: str,
        flat_answer: str,
        graph_answer: str,
        gold_answers: list[str],
    ) -> tuple[float, float, float]:
        """Evaluate Flat vs Graph answers on {0.0, 0.5, 1.0} rubric and pairwise preference margin."""
        results = self.judge_batch([(query, flat_answer, graph_answer, gold_answers)])
        return results[0] if results else (0.0, 0.0, 0.0)

    def judge_batch(
        self,
        judge_tuples: list[tuple[str, str, str, list[str]]],
    ) -> list[tuple[float, float, float]]:
        """Batch-judge candidate answers with cross-family judge LLM."""
        if not judge_tuples:
            return []

        from vllm import SamplingParams

        prompts: list[str] = []
        for query, flat_ans, graph_ans, golds in judge_tuples:
            prompts.append(build_judge_prompt(query, flat_ans, golds))
            prompts.append(build_judge_prompt(query, graph_ans, golds))

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
        )
        llm = self._get_judge_llm()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        results: list[tuple[float, float, float]] = []
        for i in range(len(judge_tuples)):
            flat_out = outputs[2 * i].outputs[0].text
            graph_out = outputs[2 * i + 1].outputs[0].text
            score_flat = parse_judge_score(flat_out)
            score_graph = parse_judge_score(graph_out)
            delta_judge = score_graph - score_flat
            results.append((score_flat, score_graph, delta_judge))

        return results

    def cleanup(self) -> None:
        """Release all model weights and GPU allocations."""
        self._unload_active_model()
