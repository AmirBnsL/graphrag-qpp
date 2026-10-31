# GraphRAG-QPP: Subgraph Topological & GNN Query Performance Prediction for GraphRAG

Pilot study evaluating **GraphRAG Gain Prediction** ($y_q = \text{F1}_{\text{GraphRAG}}(q) - \text{F1}_{\text{FlatRAG}}(q)$) across conversational agent memory (**LongMemEval-S**) and multi-hop QA (**MuSiQue** & **HotpotQA**).

## Core Research Question
Can a lightweight (`<3ms`) GNN or topological predictor over a query's candidate retrieval subgraph predict whether GraphRAG traversal will **help** (connect multi-hop bridges) or **hurt** (induce Recall–Relevance inversion and PPR hub drift) compared to Flat Dense Retrieval?

## 4-Anchor Literature Stress Test
1. **RAG Gain Target (`ALJ84L3P`, 2026):** Extends flat RAG Gain to GraphRAG vs. Flat RAG and shows why classical flat QPP (`IDF`, `SCQ`, `NQC`, `WIG`) collapses ($\rho \approx 0$).
2. **Recall–Relevance Inversion & Hub Contamination (`FDB5XWZV`, 2026 & `2DK8CGT8` *LinearRAG*, 2025):** Quantifies when multi-hop expansion hurts (`single-session-assistant` and cross-session super-hubs).
3. **Subgraph Geometry Without SPARQL Plans (`S66FD27N` *PlanRGCN*, 2025 & `VRQVWNRE` *SubgraphRAG*, 2025):** Replaces formal SPARQL execution plans with **Directional Distance Encoding (`DDE`)** and **Dual-Horizon PPR Entropy Drift** ($\Delta H_{\text{PPR}} = H(\pi^{(10)}) - H(\pi^{(2)})$ from *NodeRAG* `C4SCNXSY`).
4. **Sub-5ms Routing vs. Expensive RL/LLM Routers (`XHXSY3AA` *RouteRAG*, 2025 & `GSEMW9TE` *DRAG*, 2026):** Achieves Pareto-optimal selective routing without $8\times\text{A100}$ GRPO RL or $360\text{ms}$ 4B-LLM routers.

## 100% Colab Pro Execution Workflow (`colab` CLI)

```bash
# Stage A: Build Tri-Graphs, extract PyG candidate subgraphs, and run vLLM Oracle (L4/A100 GPU)
colab new --gpu L4 -s qpp-oracle
colab upload -s qpp-oracle . /content/graphrag-qpp
colab exec -s qpp-oracle -f scripts/00_colab_bootstrap.py
colab exec -s qpp-oracle -f scripts/01_build_and_retrieve.py
colab exec -s qpp-oracle -f scripts/02_run_vllm_oracle.py
colab download -s qpp-oracle /content/artifacts/oracle_dataset.parquet artifacts/oracle_dataset.parquet
colab download -s qpp-oracle /content/artifacts/pyg_subgraphs.pt artifacts/pyg_subgraphs.pt
colab stop -s qpp-oracle

# Stage B: Train & Evaluate Classical QPP, Topological LightGBM, and PyG SubgraphQPP-GNN (T4/L4 GPU)
colab new --gpu T4 -s qpp-train
colab upload -s qpp-train . /content/graphrag-qpp
colab upload -s qpp-train artifacts/ /content/artifacts/
colab exec -s qpp-train -f scripts/03_train_evaluate_qpp.py
colab exec -s qpp-train -f scripts/04_plot_paper_figures.py
colab download -s qpp-train /content/artifacts/results/ artifacts/results/
colab stop -s qpp-train
```
