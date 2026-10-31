# Post-Retrieval Quality Prediction Over Latency-Constrained Routing

We formulate the GNN and topological predictors strictly as a **Post-Retrieval Context Selector** focused on retrieval quality rather than retrieval speed: for every query, both Flat Dense retrieval and full `Tri-Graph` Personalized PageRank ($t=2$ and $t=10$) are executed, and the GNN inspects the resulting **Candidate Subgraph** to select the higher-quality passage set ($C_{\text{flat}}$ vs. $C_{\text{graph}}$) prior to LLM generation.

Both dense vector search and sparse matrix PPR execute in a few milliseconds, making retrieval latency optimization irrelevant compared to downstream answer quality. Running full PPR before the GNN eliminates artificial early-stopping heuristics and gives the GNN complete visibility into **PPR Diffusion Drift** ($\Delta H_{\text{PPR}}$ between $t=2$ and $t=10$) to detect **Recall-Relevance Inversion**.
