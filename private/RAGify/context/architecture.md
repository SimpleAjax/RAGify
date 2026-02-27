# RAG Evaluation Pipeline Architecture

## 1. Executive Summary
This document outlines the technical architecture for an end-to-end framework designed to benchmark various Retrieval-Augmented Generation (RAG) methodologies against complex, multi-hop datasets. The primary goal is to establish a robust base infrastructure capable of loading standardized datasets, running pluggable retrieval/generation strategies, and evaluating their performance using the **RAGAS** framework. 

Once the base infrastructure and pipeline are established, we will incrementally implement and benchmark advanced RAG techniques such as GraphRAG, Agentic Retrieval, and Query Decomposition.

---

## 2. High-Level System Architecture

The pipeline follows a highly modular, interface-driven design. This ensures that adding a new dataset or a new RAG methodology does not require structural changes to the evaluation engine.

### Core Components
1. **Data Ingestion Layer**: Standardizes diverse datasets (MuSiQue, MultiHop-RAG, etc.) into a unified schema `UnifiedQASample`.
2. **Indexing Engine**: Handles document processing, chunking, and storage (e.g., Vector DBs, Graph DBs).
3. **Strategy Controller (RAG Implementations)**: Pluggable modules conforming to a standard `AbstractRAGStrategy` interface.
4. **Evaluation Engine (RAGAS)**: Runs inferences, collects trajectories (context + generated answers), and computes metrics.
5. **Analytics & Reporting**: Stores results and visualizes comparative performance.

---

## 3. Phase 1: Data Ingestion & Normalization

All multi-hop QA datasets will be ingested and mapped to a unified internal schema. 

**Target Datasets and Format:**
*   **HotpotQA**: JSON format containing `question`, `answer`, `context` (paragraphs) and `supporting_facts` (indices connecting facts to sentences).
*   **2WikiMultiHopQA**: Similar to HotpotQA but includes relation graphs (`evidences`) and reasoning `type`.
*   **MuSiQue**: JSON format containing questions, answers, supporting paragraphs, and multi-hop `question_decomposition`.
*   **MultiHop-RAG**: JSON format with queries, answers, and `evidence_list` (full articles/chunks).

**Unified Data Schema (`UnifiedQASample`):**
```python
class UnifiedQASample(BaseModel):
    dataset_name: str           
    sample_id: str              
    query: str                 
    ground_truth_answer: str    
    supporting_contexts: List[Dict[str, str]] # The actual text required to answer the query
    corpus: List[Dict[str, str]]              # All documents provided in the dataset sample
    metadata: Dict[str, Any]                  # Crucial for dataset-specific artifacts (Graph triples, Sub-queries)
```

---

## 4. Phase 2: Core RAG Implementations

Each implementation will inherit from a base class `AbstractRAGStrategy` which defines a standard `retrieve_and_generate(query: str) -> dict` method. The output must include the generated `answer` and the `retrieved_contexts` for RAGAS evaluation.

### A. Standard/Naive RAG (Baseline)
*   **Chunking**: Semantic or fixed-size chunking.
*   **Retrieval**: Dense Vector Search + BM25 (Lexical) followed by Cross-Encoder Reranking.
*   **Generation**: Single-pass prompt with retrieved context.

### B. GraphRAG
*   **Indexing**: Uses metadata (if available, e.g., 2WikiMultiHopQA `evidences`) or LLMs to extract entities/relationships, building a Knowledge Graph.
*   **Retrieval**: Traverses graph edges or uses community summaries to gather context.

### C. Agentic (Iterative) Retrieval
*   **Mechanism**: A ReAct (Reasoning + Acting) or Self-Ask agent strategy.
*   **Execution**: Multi-turn LLM loop querying index until sufficient information is gathered.

### D. Query Decomposition (Sub-Querying)
*   **Mechanism**: LLM decomposes the query (can be evaluated against MuSiQue's ground-truth decompositions).
*   **Execution**: Retrieves context for sub-queries in parallel, concatenates, and synthesizes.

---

## 5. Phase 3: Evaluation Engine (RAGAS framework)

Integrates **RAGAS** to compute automated reference-free and reference-based metrics.
-  **Context Precision**
-  **Context Recall**
-  **Faithfulness**
-  **Answer Relevancy**

---

## 6. Infrastructure Recommendations
*   **Orchestration:** LangChain / LlamaIndex.
*   **Database:** Qdrant / Milvus (Vector), Neo4j / NetworkX (Graph).
*   **Experiment Tracking:** MLflow / Weights & Biases (W&B).
