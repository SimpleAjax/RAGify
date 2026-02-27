# Enterprise RAG: Tuning Parameters & Scaling Contentions

This document outlines the various parameters that can be tuned across the implemented RAG strategies to optimize for accuracy and cost, as well as the potential system contentions when scaling these systems to enterprise levels (high volume, high concurrency) and their solutions.

---

## 1. Tuneable Parameters for Optimization

### A. General / Overarching Parameters
*   **LLM Selection:** 
    *   *Heavy tasks* (complex synthesis, zero-shot reasoning): GPT-4o, Claude 3.5 Sonnet.
    *   *Light tasks* (routing, entity extraction, decomposition): Fast, cheap SLMs like LLaMA-3-8B, Mistral-7B, or specialized fine-tuned models (e.g., NuExtract for NER).
*   **Embedding Models:** High-dimension embeddings (e.g., `text-embedding-3-large`, `BGE-m3`) yield better semantic capture but increase Vector DB storage cost/latency.
*   **Prompts:** Few-shot prompting, dynamic prompt selection based on query intent.

### B. Naive RAG Parameters
*   **Chunking Strategy:** Chunk size (e.g., 512 tokens), overlap (e.g., 50 tokens). Semantic chunking (splitting at logical boundaries) vs. fixed-size.
*   **Retrieval K:** Number of documents to retrieve from Vector Search vs. BM25 (e.g., K=10).
*   **Reranking:** Using a Cross-Encoder (e.g., `Cohere Rerank`, `BGE-Reranker`) to re-sort the top 100 results down to top 5.

### C. GraphRAG Parameters
*   **Extraction Schema:** How granular the entities/relationships are (e.g., `Person -> works_at -> Company` vs coarse semantic triples).
*   **Traversal Depth (Max Hops):** How far into the graph to traverse (1-hop vs 3-hop). More hops = more context but higher noise and slower latency.
*   **Community Summarization:** (Microsoft GraphRAG approach) Tuning the hierarchical clustering algorithms (like Leiden) that pre-summarize sub-graphs.

### D. Agentic Retrieval & Query Decomposition Parameters
*   **Max Iterations:** Hard limit on the ReAct loop to prevent infinite tool-calling loops (e.g., max 3 steps).
*   **Parallelization:** Tuning the map-reduce fan-out for sub-queries (how many sub-query network requests to fire concurrently).

---

## 2. Enterprise Scale Contentions & Solutions

When scaling a RAG system to high-frequency, high-volume production traffic, different RAG architectures face distinct bottlenecks.

### A. LLM API / Inference Contention 
*   **Contention Point:** Heavy RAG strategies leverage the LLM heavily.
    *   *Agentic RAG* makes sequential LLM calls in a loop. A single user query might invoke the LLM 4 times. High concurrency will lead to API rate limiting or GPU OOM (Out of Memory) if self-hosted.
    *   *Decomposition RAG* fires multiple parallel LLM/Search calls, multiplying concurrency spikes.
*   **Solutions:**
    *   **Semantic Caching:** Cache previous answers based on vector similarity of the query (e.g., if User A asks a 95% similar query to User B, return cached answer without hitting the LLM).
    *   **Routing / Triage:** Not all queries need GraphRAG or Agentic RAG. Use a cheap classifier (SLM) to route simple fact-checks to Naive RAG, and only run expensive Multi-Hop / Agentic workflows for complex analytical queries.
    *   **Inference Engines:** If self-hosting, use optimized serving frameworks like vLLM or TGI for continuous batching and PagedAttention.

### B. Database Load (Vector & Graph)
*   **Contention Point:** High QPS (Queries Per Second) on the underlying knowledge bases.
    *   *Graph Databases (Neo4j)* can struggle with deep, multi-hop traversals under high concurrent load due to the exponential growth of traversed nodes.
    *   *Vector Databases (Qdrant/Milvus)* become memory-constrained as the millions of embeddings exceed RAM capacities, leading to disk swapping and latency spikes.
*   **Solutions:**
    *   **Read Replicas & Load Balancing:** Separate read and write traffic. Run multiple read-only replicas of Qdrant/Neo4j behind a load balancer.
    *   **Tenant Isolation (Sharding):** In B2B SaaS, data is partitioned by customer/tenant. Partition Vector spaces and Graphs so searches only execute over a tiny fraction of the total data.

### C. Indexing & Data Ingestion Bottlenecks
*   **Contention Point:** Keeping the Graph/Vector databases up to date in real-time. GraphRAG specifically requires running LLMs over *every single ingested document* to extract entities, which is massively expensive and slow.
*   **Solutions:**
    *   **Asynchronous Pipelines:** Offload data ingestion to async message queues (Kafka, RabbitMQ) and worker pools (Celery, Ray) that process chunks in the background without blocking the main application.
    *   **Delta Updates:** Only re-embed or re-extract graphs for the specific paragraphs that changed, rather than re-indexing entire documents.

### D. Network & I/O Latency
*   **Contention Point:** ReAct agents suffer from "Chain of Thought" latency. The user is waiting while the system does network calls to tools, waits for responses, runs LLM thought generation, and repeats.
*   **Solutions:**
    *   **Streaming UI:** Stream the LLM's intermediate thoughts and tool calls to the frontend (like ChatGPT's "Searching the web..." UI) so the user perceives the system as fast and working, mitigating frustration during high-latency cyclical loops.
