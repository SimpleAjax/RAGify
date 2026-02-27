# Phase 2: Core RAG Implementations Technical Architecture

## Overview
This document outlines the technical design for Phase 2 of RAGify. The goal is to implement four distinct Retrieval-Augmented Generation strategies. Each strategy must ingest a standardized query, retrieve context, and generate an answer, yielding an output format compatible with the RAGAS evaluation engine.

To ensure consistency, traceability, and support for complex agent loops, we will utilize a **Hybrid Orchestration Approach** (Option B):
- Linear, non-cyclic pipelines (Naive RAG, basic GraphRAG) will use **LangChain Expression Language (LCEL)**.
- Complex, cyclic, and dynamic pipelines (Agentic Retrieval, Query Decomposition) will use **LangGraph**.

## 1. Shared Strategy Abstraction
All strategies implement `AbstractRAGStrategy` and utilize a standardized state schema. 

```python
from typing import TypedDict, List, Dict, Any
from abc import ABC, abstractmethod

class RAGState(TypedDict):
    query: str
    retrieved_contexts: List[str]  # Must be populated for Context Precision/Recall
    answer: str                    # Must be populated for Answer Relevancy/Faithfulness
    metadata: Dict[str, Any]       # Strategy-specific internal data (e.g., intermediate steps, sub-queries)

class AbstractRAGStrategy(ABC):
    @abstractmethod
    def retrieve_and_generate(self, query: str) -> dict:
        """
        Executes the implemented LangGraph workflow.
        Returns a dict shaped like RAGState containing the final 'answer' and 'retrieved_contexts'.
        """
        pass
```

## 2. RAG Strategy Implementations

Each of the following strategies compiles into an executable object (either LCEL `Runnable` or LangGraph `StateGraph`) which is wrapped uniformly by `AbstractRAGStrategy`.

### Strategy A: Standard/Naive RAG
**Architecture:** Linear LCEL Chain `Query -> Retrieval -> Generation -> End`
1.  **Retrieval Step**: Executes a dense hybrid search (Qdrant + BM25). Contexts are reranked via a Cross-Encoder and the top-K chunks are injected into the chain context.
2.  **Generation Step**: Constructs a single prompt containing the query and retrieved contexts. Yields `answer` and passes through `retrieved_contexts`.

### Strategy B: GraphRAG
**Architecture:** Linear LCEL Chain `Query -> Graph Retrieval -> Generation -> End`
1.  **Graph Retrieval Step**: Given the input query, use an LLM or Named Entity Recognition (NER) to extract entities. Traverses Neo4j to find relationship paths (e.g., matching multi-hop `evidences` from datasets like 2WikiMultiHopQA) and community summaries.
2.  **Generation Step**: Synthesizes the final answer from graph-based context.

### Strategy C: Agentic Retrieval (Iterative)
**Architecture:** Cyclic Graph (ReAct loop) `Query -> Agent <--> Tools -> Generation -> End`
1.  **Agent Node**: Evaluates the `query` against current `retrieved_contexts`. Decides if more context is needed. If true, emits a tool call. If false, transitions to `Generate Node`.
2.  **Tools Node**: Executes the requested retrieval operations (e.g., query vector DB for a specific sub-topic). Appends new information to `state["retrieved_contexts"]`. Loops back to the Agent Node.
3.  **Generate Node**: Synthesizes the final answer.

### Strategy D: Query Decomposition
**Architecture:** Map-Reduce Graph `Query -> Decompose -> Parallel Retrieve -> Synthesize -> End`
1.  **Decompose Node**: LLM analyzes the multi-hop query and outputs a sequence of simpler sub-queries (stored in `state["metadata"]["sub_queries"]`).
2.  **Parallel Retrieve Node**: Iterates (or uses LangGraph's Send API) over sub-queries, running a simple Vector+BM25 retrieval for each. Accumulates all unique contexts into `state["retrieved_contexts"]`.
3.  **Synthesize Node**: Reviews the aggregated context to synthesize the comprehensive `state["answer"]`.

## 3. Directory Layout
```text
src/
└── ragify/
    └── strategies/
        ├── __init__.py
        ├── config.py                 # Configuration parameters for strategies 
        ├── abstract_strategy.py      # Contains AbstractRAGStrategy and RAGState
        ├── naive/
        │   ├── base.py               # LCEL chain and implementation
        ├── graph_rag/
        │   └── ...
        ├── agentic/
        │   └── ...
        └── decomposition/
            └── ...
```
