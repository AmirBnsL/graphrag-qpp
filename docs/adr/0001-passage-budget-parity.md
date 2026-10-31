# Passage-Budget Parity Between Flat RAG and GraphRAG

We enforce strict Passage-Budget Parity: both Flat RAG and GraphRAG retrieve and pass the exact same number of raw passage chunks ($k=5$) into an identical generator prompt, omitting extracted entity triples or community summaries from the GraphRAG prompt.

Many GraphRAG systems append entity-relation triples alongside passages, which doubles prompt token count and confounds graph-guided retrieval ranking with prompt length effects. Isolating passage ranking ensures that any non-zero GraphRAG Gain ($y_q$) is strictly attributable to topological graph traversal versus dense similarity.
