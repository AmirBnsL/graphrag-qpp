# GraphRAG-QPP

Query Performance Prediction (QPP) and topological context selection for Graph-based Retrieval-Augmented Generation and conversational agent memory.

## Language

### Core Metrics & Targets

**GraphRAG Gain**:
The difference in downstream quality between GraphRAG retrieval and Flat Dense retrieval for a given query: $y_q = \text{Score}_{\text{GraphRAG}}(q) - \text{Score}_{\text{FlatRAG}}(q)$.
_Avoid_: Retrieval benefit, RAG boost, delta score

**Lexical Gain**:
GraphRAG Gain measured via surface token overlap ($\Delta \text{F1} = \text{F1}_{\text{GraphRAG}}(q) - \text{F1}_{\text{FlatRAG}}(q)$), sensitive to verbatim phrasing.
_Avoid_: String gain, token delta

**Semantic Gain**:
GraphRAG Gain measured via cross-family LLM-as-a-Judge correctness or ground-truth answer log-likelihood ratio, invariant to surface paraphrase.
_Avoid_: Judge delta, meaning gain

**Gain Margin Threshold**:
The minimum absolute GraphRAG Gain $\tau \ge 0$ required to classify a query as a positive ($y_q \ge \tau$) or negative ($y_q \le -\tau$) routing instance rather than a neutral tie ($|y_q| < \tau$).
_Avoid_: Cutoff, deadzone, noise floor

**Passage-Budget Parity**:
The evaluation invariant requiring both Flat RAG and GraphRAG to supply the exact same number of retrieved passage chunks ($k$) to an identical generator prompt template, without injecting extra entity triples or graph summaries.
_Avoid_: Context matching, prompt normalization

**Recall-Relevance Inversion**:
The structural failure mode where expanding retrieval through graph traversal increases evidence recall but drastically degrades context precision and downstream LLM accuracy.
_Avoid_: Precision drop, graph dilution, noisy retrieval

### Graph Representations & Retrieval

**Tri-Graph**:
A heterogeneous three-layer graph consisting of passage nodes, sentence nodes, and entity nodes connected by containment, co-occurrence, and semantic similarity edges without requiring write-time LLM triple extraction.
_Avoid_: Knowledge graph, triple store, entity graph

**Cross-Session Hub Contamination**:
The topological pathology where lemmatized or surface-identical entities shared across distinct conversational sessions merge into high-degree super-hubs, causing random walks to leak irrelevant cross-session context.
_Avoid_: Session bleed, entity collision, graph pollution

**Candidate Subgraph**:
The post-retrieval query-conditioned induced subgraph containing the seed entities, candidate passages, and intermediate bridge entities annotated with full Personalized PageRank walk distributions ($t=2$ and $t=10$).
_Avoid_: Ego-network, retrieved graph, query neighborhood

**Directional Distance Encoding**:
A structural node feature capturing the directed shortest-path distance from query seed entities to candidate passage nodes in the candidate subgraph.
_Avoid_: Hop count, graph distance, positional embedding

**PPR Diffusion Drift**:
The divergence in probability distribution between shallow random walks ($t=2$) and deep random walks ($t=10$) caused by stationary convergence to high-degree graph hubs.
_Avoid_: Random walk degradation, hub trap, PageRank decay

### Routing & Evaluation

**Post-Retrieval Context Selector**:
A quality predictor (GNN or tabular model) that runs after both Flat RAG and Tri-Graph PPR retrieval have completed, inspecting the Candidate Subgraph to select the higher-quality passage set ($C_{\text{flat}}$ vs. $C_{\text{graph}}$) for the generator LLM.
_Avoid_: Pre-retrieval router, latency gatekeeper, early-stop controller

**Conservative Graph Abstention**:
The selection policy that defaults to Flat Dense RAG passages ($C_{\text{flat}}$) whenever predicted GraphRAG Gain is below the Gain Margin Threshold ($\hat{y}_q < \tau$), selecting GraphRAG passages ($C_{\text{graph}}$) strictly when $\hat{y}_q \ge \tau$.
_Avoid_: Flat fallback, tie-breaker, safe routing

**Lexical Router**:
A Post-Retrieval Context Selector trained to predict Lexical Gain ($\Delta \text{F1}$).
_Avoid_: F1 router, token router

**Semantic Router**:
A Post-Retrieval Context Selector trained to predict Semantic Gain ($\Delta \text{Judge}$).
_Avoid_: Judge router, LLM-graded router

**Zero-Shot Topological Transfer**:
Evaluating a Post-Retrieval Context Selector trained on one benchmark's Candidate Subgraph geometry directly on an unseen benchmark without retraining.
_Avoid_: Cross-dataset generalization, out-of-distribution test
