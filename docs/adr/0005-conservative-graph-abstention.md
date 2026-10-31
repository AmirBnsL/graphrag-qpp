# Conservative Graph Abstention Inside the Gain Margin Threshold

We enforce **Conservative Graph Abstention**: the **Post-Retrieval Context Selector** chooses Flat Dense RAG passages ($C_{\text{flat}}$) whenever predicted **GraphRAG Gain** is below the **Gain Margin Threshold** ($\hat{y}_q < \tau$), selecting GraphRAG passages ($C_{\text{graph}}$) only when $\hat{y}_q \ge \tau$.

When Flat RAG and GraphRAG are predicted to tie ($|\hat{y}_q| < \tau$), selecting GraphRAG passages risks **Recall-Relevance Inversion** (`Gap 3`) without expected quality improvement. Defaulting to Flat RAG in the neutral zone guarantees that graph-retrieved passages are only passed to the generator LLM when topological evidence predicts a clear quality gain.
