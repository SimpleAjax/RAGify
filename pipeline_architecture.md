# RAG Evaluation Pipeline Architecture

## 1. Executive Summary
This document outlines the technical architecture for an end-to-end framework designed to benchmark various Retrieval-Augmented Generation (RAG) methodologies against complex, multi-hop datasets. The primary goal is to establish a robust base infrastructure capable of loading standardized datasets, running pluggable retrieval/generation strategies, and evaluating their performance using the **RAGAS** framework. 

Once the base infrastructure and pipeline are established, we will incrementally implement and benchmark advanced RAG techniques such as GraphRAG, Agentic Retrieval, and Query Decomposition.

---

## 2. High-Level System Architecture

The pipeline follows a highly modular, interface-driven design (adhering to DRY principles). This ensures that adding a new dataset or a new RAG methodology does not require structural changes to the evaluation engine.

### Core Components
1. **Data Ingestion Layer**: Standardizes diverse datasets (MuSiQue, MultiHop-RAG, etc.) into a unified schema.
2. **Indexing Engine**: Handles document processing, chunking, and storage (e.g., Vector DBs, Graph DBs).
3. **Strategy Controller (RAG Implementations)**: Pluggable modules conforming to a standard `BaseRAGStrategy` interface.
4. **Evaluation Engine (RAGAS)**: Runs inferences, collects trajectories (context + generated answers), and computes metrics.
5. **Analytics & Reporting**: Stores results and visualizes comparative performance.

---

## 3. Phase 1: Data Ingestion & Normalization

To ensure fair evaluation, all multi-hop QA datasets will be ingested and mapped to a unified internal schema. 

**Target Datasets:**
*   **MuSiQue**: Multi-hop questions requiring compositional reasoning.
*   **MultiHop-RAG**: Benchmarking retrieval across disparate documents.
*   **HotpotQA**: Questions requiring reasoning over multiple Wikipedia articles.
*   **2WikiMultiHopQA**: Complex multi-hop QA leveraging Wikipedia.

**Unified Data Schema:**
```json
{
  "dataset_name": "hotpot_qa",
  "sample_id": "12345",
  "query": "Who is the director of the movie starring X?",
  "ground_truth_answer": "Director Name",
  "supporting_facts": ["fact 1", "fact 2"], // For context recall eval
  "corpus": [
    {"doc_id": "d1", "text": "...", "metadata": {}}
  ]
}
```

---

## 4. Phase 2: Core RAG Implementations

Each implementation will inherit from a base class `AbstractRAGStrategy` which defines a standard `retrieve_and_generate(query: str) -> dict` method. The output must include the generated `answer` and the `retrieved_contexts` for RAGAS evaluation.

### A. Standard/Naive RAG (Baseline)
*   **Chunking**: Semantic or fixed-size chunking.
*   **Retrieval**: Dense Vector Search (e.g., embeddings) + BM25 (Lexical) followed by Cross-Encoder Reranking.
*   **Generation**: Single-pass prompt with retrieved context.

### B. GraphRAG
*   **Indexing**: Uses LLMs to extract entities and relationships from the corpus, building a Knowledge Graph (e.g., using NetworkX or Neo4j).
*   **Retrieval**: Traverses graph edges or uses community summaries to gather context, highly effective for dataset-level questions.

### C. Agentic (Iterative) Retrieval
*   **Mechanism**: A ReAct (Reasoning + Acting) or Self-Ask agent strategy.
*   **Execution**: The LLM queries the Vector/Graph DB, reads the retrieved context, determines if it has enough information, and if not, formulates a new, more specific query to run a subsequent retrieval. 

### D. Query Decomposition (Sub-Querying)
*   **Mechanism**: An LLM decomposes the initial complex multi-hop query into smaller, independent sub-queries.
*   **Execution**: Retrieves context for each sub-query in parallel, concatenates the contexts, and passes the aggregate context to a final synthesis prompt.

---

## 5. Phase 3: Evaluation Engine (RAGAS framework)

The pipeline will integrate **RAGAS** (Retrieval Augmented Generation Assessment) to compute automated, reference-free (where possible) and reference-based metrics.

### Key Metrics to Track:
1.  **Context Precision**: How relevant are the retrieved contexts to the query? (Are the most relevant chunks ranked highest?)
2.  **Context Recall**: Did the retrieval system fetch all the necessary information requires to answer the multi-hop query? (Compared against dataset `supporting_facts`).
3.  **Faithfulness**: Is the generated answer hallucinatory, or is it strictly derived from the retrieved context?
4.  **Answer Relevancy**: Does the generated answer directly address the user's query?

### Execution Flow:
1. Pipeline loads a test split from the normalized dataset (e.g., 500 samples).
2. For each sample, the pipeline invokes the selected `RAGStrategy`.
3. The strategy returns `{"answer": "...", "contexts": [...]}`.
4. Results are pushed to a `RagasEvaluator` class alongside the `ground_truth`.
5. RAGAS computes the scores and logs them to an experiment tracking system.

---

## 6. Infrastructure & Tech Stack Recommendations

*   **Orchestration / Framework:** `LangChain` or `LlamaIndex` (provides great primitives for Agentic and Graph strategies).
*   **Vector Database:** `Qdrant` or `Milvus` (local containerized instances for fast iteration).
*   **Graph Database:** `Neo4j` (if GraphRAG needs persistence) or simple in-memory `NetworkX` for smaller prototype datasets.
*   **LLM Provider:** OpenAI (`gpt-4o-mini` for fast/cheap baseline, `gpt-4o` for complex multi-hop agentic reasoning) or Anthropic `Claude 3.5 Sonnet`. 
*   **Experiment Tracking:** `MLflow`, `Weights & Biases (W&B)`, or `LangSmith` to track traces and evaluation scores across different RAG variants.
*   **Language:** Python 3.11+ using `Pydantic` heavily for data validation.

---

## 7. Next Steps for Implementation

1.  **Setup Project Structure**: Initialize Python environment, set up the `src/` directory with `data_loaders`, `rag_strategies`, and `evaluation` modules.
2.  **Implement Data Loaders**: Write the ingestion script to pull a sample (e.g., 100 records) of HotpotQA and map it to our unified schema.
3.  **Build Base Infrastructure**: Implement the `AbstractRAGStrategy` and a stub/mock evaluator.
4.  **Implement Baseline RAG**: Build the Naive RAG approach and wire it through the RAGAS evaluator to establish baseline metrics.
5.  **Iterate**: Begin implementing advanced strategies (Decomposition, Agentic, Graph) one by one, comparing results.
