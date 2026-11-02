# Dual Lexical and Semantic Gain Targets with Margin Sensitivity

We record and train routers against both **Lexical Gain** ($\Delta \text{F1}$) and continuous **Semantic Gain** ($\Delta \text{Judge} \in [-1.0, +1.0]$, combining the $\{0.0, 0.5, 1.0\}$ probability-weighted rubric delta with a $\pm 0.25$ pairwise preference margin between $C_{\text{flat}}$ and $C_{\text{graph}}$), and sweep the **Gain Margin Threshold** $\tau \in \{0.0, 0.05, 0.10, 0.15, 0.20\}$ rather than fixing a single binarization cutoff.

Conversational benchmarks (`LongMemEval-S`) exhibit valid paraphrases where token F1 penalizes semantically correct answers, whereas multi-hop QA (`HotpotQA`, `MuSiQue`) relies on exact entity spans. Combining log-probability-weighted rubric scoring with pairwise preference comparison eliminates discretization dead-zones in $(0.0, 0.5)$ and turns router comparison across $\tau$ into a smooth sensitivity experiment.
