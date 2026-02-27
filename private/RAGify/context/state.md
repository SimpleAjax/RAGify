# RAG Evaluation Pipeline State

**Current Phase:** Planning
**Last Updated:** {{CURRENT_DATE}}

### State Summary
- Initial architecture and plan documented.
- Project requires initial structure setup.
- We have chosen 4 Multi-hop QA datasets: MuSiQue, MultiHop-RAG, HotpotQA, 2WikiMultiHopQA.
- We have chosen to use RAGAS as the unified evaluation engine.
- We have defined 4 RAG architectures to benchmark: Standard, GraphRAG, Agentic Iterative, Query Decomposition.

### Next Immediate Steps
- Review architecture with user.
- Upon approval, initialize Python project, define `requirements.txt`, and build the Base Interfaces (Data Ingestion + Strategy Controller + Ragas Evaluator).

### Blockers / Open Questions
- Awaiting user approval on the initial architecture and flow. 
- Need to decide if we strictly use LangChain or LlamaIndex for orchestration, or handle things without a heavy framework.
- Need to lock in an experiment tracking platform (e.g., W&B or MLflow).
