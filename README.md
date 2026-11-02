# GraphRAG-QPP: Subgraph Topological & GNN Query Performance Prediction for GraphRAG

Pilot study evaluating **GraphRAG Gain Prediction** ($y_q = \text{F1}_{\text{GraphRAG}}(q) - \text{F1}_{\text{FlatRAG}}(q)$) across conversational agent memory (**LongMemEval-S**) and multi-hop QA (**MuSiQue** & **HotpotQA**).

## Core Research Question
Can a lightweight (`<3ms`) GNN or topological predictor over a query's candidate retrieval subgraph predict whether GraphRAG traversal will **help** (connect multi-hop bridges) or **hurt** (induce Recall–Relevance inversion and PPR hub drift) compared to Flat Dense Retrieval?

## 4-Anchor Literature Stress Test
1. **RAG Gain Target (`ALJ84L3P`, 2026):** Extends flat RAG Gain to GraphRAG vs. Flat RAG and shows why classical flat QPP (`IDF`, `SCQ`, `NQC`, `WIG`) collapses ($\rho \approx 0$).
2. **Recall–Relevance Inversion & Hub Contamination (`FDB5XWZV`, 2026 & `2DK8CGT8` *LinearRAG*, 2025):** Quantifies when multi-hop expansion hurts (`single-session-assistant` and cross-session super-hubs).
3. **Subgraph Geometry Without SPARQL Plans (`S66FD27N` *PlanRGCN*, 2025 & `VRQVWNRE` *SubgraphRAG*, 2025):** Replaces formal SPARQL execution plans with **Directional Distance Encoding (`DDE`)** and **Dual-Horizon PPR Entropy Drift** ($\Delta H_{\text{PPR}} = H(\pi^{(10)}) - H(\pi^{(2)})$ from *NodeRAG* `C4SCNXSY`).
4. **Sub-5ms Routing vs. Expensive RL/LLM Routers (`XHXSY3AA` *RouteRAG*, 2025 & `GSEMW9TE` *DRAG*, 2026):** Achieves Pareto-optimal selective routing without $8\times\text{A100}$ GRPO RL or $360\text{ms}$ 4B-LLM routers.

## How to Run the Benchmark

You can run the benchmark in three ways depending on your environment:

### Option 1: One-Click Google Colab Notebook (Recommended)
Open `notebooks/run_benchmarks.ipynb` directly in Google Colab (with an **L4 or A100 GPU** runtime).
- All setup, package installation, fast pilot testing, full 1,000-query benchmark execution, and diagnostic plot rendering are packaged into clean, sequential notebook cells.
- You can attach directly to your existing CLI session in the browser at any time:
  ```bash
  colab url --open -s graphrag-v2
  ```

### Option 2: Local CLI / Workstation Execution
Use the unified runner script [`scripts/run_benchmarks.sh`](file:///home/amirbnsl/Projects/graphrag-qpp/scripts/run_benchmarks.sh):

```bash
# Fast mock CPU run (~10 seconds, zero GPU needed)
./scripts/run_benchmarks.sh --mode mock

# 20-query validation pilot on GPU (~3 minutes)
./scripts/run_benchmarks.sh --mode colab --pilot

# Full 1,000-query benchmark on GPU (~65 minutes)
./scripts/run_benchmarks.sh --mode colab --full
```

### Option 3: Headless Colab CLI Execution
If running via the `colab` CLI, always supply `--timeout` for long-running jobs (the default CLI timeout is 30s):

```bash
# 1. Provision L4 GPU instance
colab new --gpu L4 -s graphrag-v2

# 2. Package and upload codebase
tar --exclude='.git' --exclude='.venv' --exclude='artifacts' -czf /tmp/graphrag-qpp.tar.gz .
colab upload -s graphrag-v2 /tmp/graphrag-qpp.tar.gz graphrag-qpp.tar.gz

# 3. Extract and install dependencies
echo "import tarfile; tarfile.open('/graphrag-qpp.tar.gz').extractall('/content/graphrag-qpp')" | colab exec -s graphrag-v2 --timeout 60
echo "!pip install -q -e /content/graphrag-qpp vllm torch-geometric sentence-transformers lightgbm datasets" | colab exec -s graphrag-v2 --timeout 180

# 4. Run benchmark (with 2-hour timeout to prevent socket disconnections)
colab exec -s graphrag-v2 -f /content/graphrag-qpp/scripts/run_benchmarks.py --timeout 7200

# 5. Stop instance when complete to conserve compute units
colab stop -s graphrag-v2
```

## Connection Lost Troubleshooting
If you encounter `RuntimeError: Connection was lost` or `TimeoutError`:
- **Default Timeout**: `colab exec` defaults to a 30-second timeout. Always specify `--timeout <seconds>` (e.g. `--timeout 1800` or `--timeout 7200`) for model loading and benchmark sweeps.
- **Kernel Restart**: If a kernel hangs or an unhandled exception wedges memory, run `colab restart-kernel -s <session>` to reset the kernel without losing the provisioned GPU VM.
- **Browser Monitoring**: Run `colab url --open -s <session>` to monitor progress directly in the web UI. Colab's web UI runs persistently on Google's backend and will never drop execution due to local terminal connection loss.
