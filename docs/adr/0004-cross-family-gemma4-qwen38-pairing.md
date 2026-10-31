# Quantized Cross-Family 2026 Model Pairing (Gemma 4 and Qwen 3.5/3.8) for Context-Safe VRAM

We use `Qwen/Qwen3-Embedding-0.6B` for dense embeddings, and pair quantized (`FP8` / 4-bit `AWQ`) 8B–12B models from Alibaba's **Qwen 3 / 3.5 / 3.8** family and Google's **Gemma 4** family (`google/gemma-4-12B-it` quantized) as the RAG Generator and Cross-Family LLM Judge via `vLLM`, loading only one LLM into VRAM at a time.

On a 24 GB Colab Pro L4 GPU, unquantized 9B–12B models consume 18–24 GB for weights alone, starving the `vLLM` KV cache when processing 5-passage RAG contexts across batched prompts. Running quantized 8B–12B models caps weight memory at ~6–9 GB VRAM, reserving 11–13 GB of VRAM exclusively for prompt context and KV cache while preserving 8B–12B reasoning quality and eliminating self-preference bias.
