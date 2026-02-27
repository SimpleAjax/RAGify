# Phase 3: Data Indexing & Storage Architecture

## Overview
Phase 1 handled pulling remote datasets and normalizing them in memory into `UnifiedQASample`.
Phase 2 built the strategy logic expecting a `retriever` to feed it documents.
Phase 3 bridges the gap: We will persist the `UnifiedQASample` data into long-term retrieval storage (Qdrant for vector, Neo4j for Graph) so our Phase 2 strategies can query them.

## 1. Differentiating Datasets in Vector / Graph DBs

A critical requirement is ensuring that when evaluating a query from *HotpotQA*, the retriever does **not** accidentally return context paragraphs from *MuSiQue* or *2WikiMultiHop*. Otherwise, the evaluation is contaminated.

### The Solution: Multi-Tenancy via Metadata Filtering
We will store all datasets in the same central databases (one Qdrant cluster, one Neo4j cluster) but strictly segregate them using **Metadata Payload Filtering**.

*   **Qdrant (Vector DB):** Every embedded chunk will be upserted with a payload: `{"dataset": "hotpotqa", "sample_id": "12345"}`. When `NaiveRAG` queries the DB, the retriever will be dynamically configured with a Qdrant `Filter` ensuring `{"key": "dataset", "match": {"value": "hotpotqa"}}`.
*   **Neo4j (Graph DB):** Every Node and Edge will be labeled or assigned a property `dataset: "2wikimultihop"`. Cypher queries will include `WHERE n.dataset = "2wikimultihop"`.

This allows us to scale horizontally without provisioning dedicated database instances for every tiny dataset.

## 2. Component Architecture

### A. The Indexer Pipeline (`src/ragify/indexing/`)
We will build an Indexer module that takes a list of `UnifiedQASample` objects and routes them to the appropriate database.

1.  **Vector Indexer (Qdrant):**
    *   **Chunking:** LangChain `RecursiveCharacterTextSplitter`.
    *   **Embedding:** Use a local embedding model via HuggingFace (e.g., `BAAI/bge-small-en-v1.5` to save costs, or OpenAI if requested).
    *   **Storage:** Upsert vectors + payloads to Qdrant.
2.  **Graph Indexer (Neo4j):**
    *   For Phase 3, we will start with datasets that *already* provide graph evidences (like 2WikiMultiHopQA) to prove the pipeline works end-to-end before writing expensive LLM Entity-Extraction ingestion scripts.
    *   **Storage:** Execute Cypher queries to `MERGE` nodes and edges based on the dataset's metadata.

### B. The Storage Infrastructure (`docker-compose.yml`)
We need local, containerized instances of the databases for rapid iteration and testing.

*   **Qdrant:** `qdrant/qdrant:latest` (Exposed on port 6333)
*   **Neo4j:** `neo4j:5-community` (Exposed on ports 7687 and 7474, using APOC plugins if needed).

## 3. Implementation Steps

1.  **Docker Setup:** Create `docker-compose.yml` for Qdrant and Neo4j.
2.  **Vector Indexing:** Implement `QdrantIndexer` class.
3.  **Graph Indexing:** Implement `Neo4jIndexer` class.
4.  **CLI / Script:** Create an entrypoint `scripts/index_data.py` that loads Phase 1 data and pushes it through the Indexers.
