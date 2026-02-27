# RAG Evaluation Pipeline To-Do List

## Phase 1: Project Setup & Base Infrastructure
- [ ] Initialize Python environment and project structure (`src/`, `tests/`, etc.).
- [ ] Set up `requirements.txt` (RAGAS, LangChain/LlamaIndex, Qdrant/Milvus, etc.).
- [ ] Define the `AbstractRAGStrategy` interface.
- [ ] Define the unified `UnifiedDatasetReader` interface.

## Phase 2: Data Ingestion & Normalization
- [ ] Implement data loader for **HotpotQA**.
- [ ] Implement data loader for **MuSiQue**.
- [ ] Implement data loader for **MultiHop-RAG**.
- [ ] Implement data loader for **2WikiMultiHopQA**.
- [ ] Validate alignment of all loaders with the unified data schema.

## Phase 3: Baseline RAG Evaluation
- [ ] Implement **Standard/Naive RAG** strategy (Vector search + chunking).
- [ ] Implement the **RAGAS Evaluator** wrapper to process responses against datasets.
- [ ] Run the end-to-end evaluation for Standard RAG on one dataset to confirm pipeline viability.
- [ ] Integrate experiment logging (e.g., W&B or MLflow).

## Phase 4: Advanced RAG Strategies implementation
- [ ] Implement **Query Decomposition** strategy.
- [ ] Evaluate Query Decomposition with RAGAS.
- [ ] Implement **Agentic (Iterative) Retrieval** strategy.
- [ ] Evaluate Agentic Retrieval with RAGAS.
- [ ] Implement **GraphRAG** strategy (Entity extraction + graph search).
- [ ] Evaluate GraphRAG with RAGAS.

## Phase 5: Result Analysis
- [ ] Compile performance reports across all techniques and datasets.
- [ ] Analyze which retrieval setup optimally balances latency, cost, and RAGAS score.
