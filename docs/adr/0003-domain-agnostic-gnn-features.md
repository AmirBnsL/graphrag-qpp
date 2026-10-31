# Domain-Agnostic Structural and Similarity Features for the Subgraph GNN

We exclude raw high-dimensional dense text embeddings from the GNN node feature vector, passing only domain-agnostic structural and query-relative scalar signals: query-node cosine similarity, pairwise edge semantic similarity, corpus IDF, global and local node degree, shallow/deep PPR probabilities, **PPR Diffusion Drift**, and **Directional Distance Encoding**.

Passing 1024-dimensional text vectors into a GNN trained on 1,000 queries causes the input projection layer to memorize dataset-specific lexical topics (such as Wikipedia trivia versus chat logs) rather than learning retrieval graph topology. Restricting node features to query-relative similarity and graph geometry forces the GNN to learn a transferable structural QPP function that can be evaluated zero-shot across datasets.
