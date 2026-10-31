# Dual Lexical and Semantic Gain Targets with Margin Sensitivity

We record and train routers against both **Lexical Gain** ($\Delta \text{F1}$) and **Semantic Gain** ($\Delta \text{Judge}$ via LLM-as-a-Judge and answer log-probability ratio), and sweep the **Gain Margin Threshold** $\tau \in \{0.0, 0.05, 0.10, 0.15, 0.20\}$ rather than fixing a single binarization cutoff.

Conversational benchmarks (`LongMemEval-S`) exhibit valid paraphrases where token F1 penalizes semantically correct answers, whereas multi-hop QA (`HotpotQA`, `MuSiQue`) relies on exact entity spans. Comparing routers trained on Lexical Gain versus Semantic Gain turns potential metric noise into a controlled experiment testing whether lexical and semantic supervision induce distinct topological routing boundaries.
