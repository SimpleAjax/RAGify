# RAGify - Complete User Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Configuration Parameters](#configuration-parameters)
5. [How to Run](#how-to-run)
6. [Testing Guide](#testing-guide)
7. [Architecture Overview](#architecture-overview)
8. [Dataset Loaders](#dataset-loaders)
9. [RAG Strategies](#rag-strategies)
10. [Evaluation Framework](#evaluation-framework)

---

## Project Overview

RAGify is an end-to-end framework for benchmarking various Retrieval-Augmented Generation (RAG) methodologies against complex, multi-hop QA datasets. It supports:

- **Datasets**: HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHop-RAG
- **RAG Strategies**: Naive RAG, GraphRAG, Agentic RAG, Query Decomposition RAG
- **Evaluation**: RAGAS framework with MLflow tracking
- **Storage**: Qdrant (Vector DB) + Neo4j (Graph DB)

---

## Prerequisites & Installation

### System Requirements
- Python 3.11+
- Docker & Docker Compose
- Git

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd RAGify

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies (requirements.txt)
```
datasets
pydantic
pytest
langchain-core>=0.3.74,<1.0.0
tqdm
langchain-huggingface
qdrant-client
sentence-transformers
langgraph
neo4j
ragas
litellm
mlflow
```

---

## Infrastructure Setup

### Start Infrastructure Services

```bash
# Start Qdrant and Neo4j containers
docker compose up -d

# Verify services are running
docker ps
```

### Service Endpoints

| Service | URL/Port | Purpose |
|---------|----------|---------|
| Qdrant REST API | http://localhost:6333 | Vector search operations |
| Qdrant gRPC | localhost:6334 | High-performance vector operations |
| Neo4j HTTP | http://localhost:7474 | Graph browser (user: neo4j, pass: password) |
| Neo4j Bolt | localhost:7687 | Graph database protocol |

### Default Credentials
- **Neo4j**: Username `neo4j`, Password `password`

### Stop Infrastructure
```bash
docker compose down

# To remove data volumes (WARNING: deletes all indexed data)
docker compose down -v
```

---

## Configuration Parameters

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key for LLM calls |

*Required when using OpenAI models. For local models (Ollama), set to any non-empty string.

### Setting Environment Variables

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Configurable Parameters by Component

#### QdrantManager (`src/indexing/qdrant_manager.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `collection_name` | `"ragify_evaluator"` | Qdrant collection name |
| `embedding_model_name` | `"BAAI/bge-small-en-v1.5"` | HuggingFace embedding model |
| `host` | `"localhost"` | Qdrant server host |
| `port` | `6333` | Qdrant server port |
| `chunk_size` | `512` | Text splitter chunk size |
| `chunk_overlap` | `50` | Text splitter overlap |

#### Neo4jManager (`src/indexing/neo4j_manager.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `uri` | `"bolt://localhost:7687"` | Neo4j connection URI |
| `user` | `"neo4j"` | Neo4j username |
| `password` | `"password"` | Neo4j password |

#### RagasEvaluator (`src/evaluation/evaluator.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"gpt-4o-mini"` | LiteLLM model identifier |
| `api_base` | `None` | Custom API base URL (e.g., Ollama) |

**Supported Model Formats:**
- OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
- Ollama: `ollama/llama3`, `ollama/mistral`
- Anthropic: `claude-3-sonnet`

#### ExperimentTracker (`src/evaluation/tracker.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment_name` | `"RAGify_Evaluations"` | MLflow experiment name |
| `tracking_uri` | `"./mlruns"` | MLflow tracking URI |

---

## How to Run

### 1. Demo Dataset Loaders

Verify dataset loaders are working:

```bash
python scripts/demo_loaders.py
```

This displays sample records from all 4 datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHop-RAG).

### 2. Index Data

Index datasets into vector and graph databases:

```bash
# Index both vector and graph data
python scripts/index_data.py

# Skip vector indexing (Qdrant)
python scripts/index_data.py --skip-vector

# Skip graph indexing (Neo4j)
python scripts/index_data.py --skip-graph
```

**What it does:**
- Loads 100 samples from each of the 4 datasets
- Chunks and embeds texts using BAAI/bge-small-en-v1.5
- Stores vectors in Qdrant with dataset metadata tagging
- Extracts and stores graph triples in Neo4j (for 2WikiMultiHopQA)

### 3. Run Evaluation

Evaluate RAG strategies using RAGAS metrics:

```bash
# Using OpenAI (default)
python scripts/evaluate_strategy.py --model gpt-4o-mini --output results.csv

# Using Ollama (local)
python scripts/evaluate_strategy.py \
  --model ollama/llama3 \
  --api-base http://localhost:11434 \
  --output results.csv

# With custom input file
python scripts/evaluate_strategy.py \
  --model gpt-4o-mini \
  --input eval_data.json \
  --output results.csv \
  --run-name "experiment_1"
```

**Input JSON format (`eval_data.json`):**
```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris is the capital of France.",
    "contexts": ["Paris is the capital city of France."],
    "ground_truth": "Paris"
  }
]
```

### 4. View MLflow Results

```bash
mlflow ui
# Open http://localhost:5000 in browser
```

---

## Testing Guide

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only (no integration tests)
pytest tests/ -k "not integration" -v

# Integration tests only (requires Docker services running)
pytest tests/strategies/integration/ -v

# Specific component tests
pytest tests/indexing/ -v
pytest tests/loaders/ -v
pytest tests/strategies/ -v
pytest tests/evaluation/ -v
```

### Test Structure

```
tests/
├── __init__.py
├── evaluation/
│   ├── test_evaluator.py      # RAGAS evaluator tests
│   └── test_tracker.py        # MLflow tracker tests
├── indexing/
│   ├── test_neo4j_manager.py  # Neo4j manager tests
│   └── test_qdrant_manager.py # Qdrant manager tests
├── loaders/
│   ├── test_hotpotqa.py       # HotpotQA loader tests
│   ├── test_multihop_rag.py   # MultiHop-RAG loader tests
│   ├── test_musique.py        # MuSiQue loader tests
│   └── test_twowiki.py        # 2Wiki loader tests
└── strategies/
    ├── integration/           # Integration tests (needs Docker)
    │   ├── conftest.py        # Shared fixtures
    │   ├── test_agentic_rag_int.py
    │   ├── test_decomposition_rag_int.py
    │   ├── test_graph_rag_int.py
    │   └── test_naive_rag_int.py
    ├── test_abstract_strategy.py
    ├── test_agentic_rag.py    # Unit tests
    ├── test_decomposition_rag.py
    ├── test_graph_rag.py
    └── test_naive_rag.py
```

### Integration Test Requirements

Integration tests require:
1. Docker containers running (`docker compose up -d`)
2. Services accessible on default ports
3. Test data indexed (run `python scripts/index_data.py` first)

**Skip integration tests if infrastructure is unavailable:**
```bash
pytest tests/ -k "not integration"
```

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  (HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHop-RAG)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Unified Schema                            │
│         (UnifiedQASample - standardized format)             │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│    Vector Indexing      │      │     Graph Indexing      │
│      (Qdrant)           │      │      (Neo4j)            │
│  - Embeddings           │      │  - Entity Extraction    │
│  - Chunking             │      │  - Relationship Mapping │
│  - Multi-tenancy        │      │  - Triple Storage       │
└─────────────────────────┘      └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG Strategies                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │  Naive RAG  │ │  GraphRAG   │ │ Agentic RAG │            │
│  │   (LCEL)    │ │   (LCEL)    │ │(LangGraph)  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────┐                                            │
│  │Decomposition│                                            │
│  │   RAG       │                                            │
│  │(LangGraph)  │                                            │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation Engine (RAGAS)                       │
│  - Context Precision                                        │
│  - Context Recall                                           │
│  - Faithfulness                                             │
│  - Answer Relevancy                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Experiment Tracking (MLflow)                    │
│  - Metrics logging                                          │
│  - Artifact storage                                         │
│  - Run comparison                                           │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector DB | Qdrant | Dense vector search, document retrieval |
| Graph DB | Neo4j | Knowledge graph storage, relationship traversal |
| Embeddings | HuggingFace (BGE) | Text vectorization |
| LLM Interface | LiteLLM | Multi-provider LLM abstraction |
| Evaluation | RAGAS | Automated RAG metrics |
| Tracking | MLflow | Experiment management |
| Orchestration | LangChain/LangGraph | Pipeline composition |

---

## Dataset Loaders

### Available Loaders

| Dataset | Class | Source | Splits |
|---------|-------|--------|--------|
| HotpotQA | `HotpotQALoader` | HuggingFace | train, validation |
| 2WikiMultiHopQA | `TwoWikiMultiHopQALoader` | HuggingFace | train, validation |
| MuSiQue | `MuSiQueLoader` | HuggingFace | train, validation |
| MultiHop-RAG | `MultiHopRAGLoader` | HuggingFace | train |

### Usage Example

```python
from src.loaders.hotpotqa import HotpotQALoader
from src.loaders.twowiki import TwoWikiMultiHopQALoader
from src.loaders.musique import MuSiQueLoader
from src.loaders.multihop_rag import MultiHopRAGLoader

# Load HotpotQA
loader = HotpotQALoader(config_name="distractor", split="validation")
for sample in loader.load():
    print(sample.query)
    print(sample.ground_truth_answer)
    break
```

### Unified Schema (`UnifiedQASample`)

```python
{
    "dataset_name": str,           # Source dataset name
    "sample_id": str,              # Unique identifier
    "query": str,                  # Question text
    "ground_truth_answer": str,    # Correct answer
    "supporting_contexts": List[   # Evidence passages
        {"title": str, "text": str}
    ],
    "corpus": List[                # All available documents
        {"title": str, "text": str}
    ],
    "metadata": Dict               # Dataset-specific extras
}
```

---

## RAG Strategies

All strategies implement `AbstractRAGStrategy` with the interface:

```python
class AbstractRAGStrategy(ABC):
    @abstractmethod
    def retrieve_and_generate(self, query: str, **kwargs) -> RAGState:
        """Returns dict with query, retrieved_contexts, answer, metadata"""
        pass
```

### 1. Naive RAG (`NaiveRAG`)

Standard retrieval-generation pipeline using LCEL.

```python
from src.strategies.naive.base import NaiveRAG
from src.indexing.qdrant_manager import QdrantManager
from langchain_openai import ChatOpenAI

# Setup
qdrant = QdrantManager()
retriever = qdrant.get_langchain_retriever(target_dataset="HotpotQA", k=5)
llm = ChatOpenAI(model="gpt-4o-mini")

# Execute
rag = NaiveRAG(retriever=retriever, llm=llm)
result = rag.retrieve_and_generate("What is the capital of France?")
```

**Flow:** Query → Vector Search → Context Formatting → LLM Generation

### 2. GraphRAG (`GraphRAG`)

Uses knowledge graph traversal for context retrieval.

```python
from src.strategies.graph_rag.base import GraphRAG

# Define graph retriever function
def graph_retriever(entities: List[str]) -> List[Document]:
    # Custom logic to query Neo4j based on entities
    pass

rag = GraphRAG(graph_retriever=graph_retriever, llm=llm)
result = rag.retrieve_and_generate("What is the relationship between X and Y?")
```

**Flow:** Query → Entity Extraction → Graph Traversal → Context Formatting → LLM Generation

### 3. Agentic RAG (`AgenticRAG`)

Uses LangGraph ReAct agent for iterative retrieval.

```python
from src.strategies.agentic.base import AgenticRAG
from langchain_core.tools import tool

@tool
def vector_search(query: str) -> str:
    """Search vector database"""
    return "Search results..."

rag = AgenticRAG(
    tools=[vector_search],
    llm=llm,
    system_prompt="Custom agent instructions..."
)
result = rag.retrieve_and_generate("Complex multi-hop question?")
```

**Flow:** Query → ReAct Loop (Tool Calls → Observation → Reasoning) → Final Answer

### 4. Decomposition RAG (`DecompositionRAG`)

Breaks complex queries into sub-queries using Map-Reduce.

```python
from src.strategies.decomposition.base import DecompositionRAG

rag = DecompositionRAG(retriever=retriever, llm=llm)
result = rag.retrieve_and_generate("What is X and how does it relate to Y?")
```

**Flow:** Query → Decompose into Sub-queries → Parallel Retrieval → Synthesize → Final Answer

---

## Evaluation Framework

### RAGAS Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Context Precision | Relevance of retrieved contexts to query | 0-1 |
| Context Recall | Coverage of ground truth in retrieved contexts | 0-1 |
| Faithfulness | Answer groundedness in retrieved contexts | 0-1 |
| Answer Relevancy | Directness of answer to query | 0-1 |

### Running Evaluation

```python
from src.evaluation.evaluator import RagasEvaluator

evaluator = RagasEvaluator(model_name="gpt-4o-mini")

eval_data = [
    {
        "question": "What is X?",
        "answer": "X is...",
        "contexts": ["Context 1", "Context 2"],
        "ground_truth": "X is defined as..."
    }
]

results_df = evaluator.evaluate_strategy(eval_data)
print(results_df.mean())  # Average scores
```

### MLflow Tracking

```python
from src.evaluation.tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="My_Experiment",
    tracking_uri="./mlruns"
)

with tracker.start_run(run_name="run_1"):
    tracker.log_parameters({"model": "gpt-4o-mini", "k": 5})
    tracker.log_metrics(results_df)
    tracker.log_evaluation_artifact(results_df, "results.csv")
```

### Viewing Results

```bash
# Start MLflow UI
mlflow ui

# Open browser
http://localhost:5000
```

---

## Troubleshooting

### Common Issues

**1. Docker services not starting**
```bash
# Check Docker status
docker ps
docker compose logs

# Restart services
docker compose down
docker compose up -d
```

**2. Connection errors to Qdrant/Neo4j**
- Verify containers are running: `docker ps`
- Check ports are not in use: `netstat -an | findstr 6333`
- Verify firewall settings

**3. OpenAI API errors**
- Check `OPENAI_API_KEY` is set: `echo $env:OPENAI_API_KEY`
- Verify API key is valid
- Check rate limits

**4. Import errors**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

**5. Embedding model download issues**
- First run downloads BGE model from HuggingFace
- Requires internet connection
- Model cached in `~/.cache/huggingface/`

### Getting Help

- Check logs in Docker: `docker compose logs <service>`
- Review MLflow logs: `mlruns/` directory
- Run tests with verbose output: `pytest -v -s`

---

## Project Structure

```
RAGify/
├── docker-compose.yml          # Infrastructure services
├── requirements.txt            # Python dependencies
├── GUIDE.md                    # This guide
├── pipeline_architecture.md    # Architecture documentation
├── scripts/
│   ├── demo_loaders.py         # Dataset loader demo
│   ├── index_data.py           # Data indexing script
│   └── evaluate_strategy.py    # Evaluation runner
├── src/
│   ├── schema.py               # Unified data schema
│   ├── evaluation/
│   │   ├── evaluator.py        # RAGAS wrapper
│   │   └── tracker.py          # MLflow wrapper
│   ├── indexing/
│   │   ├── qdrant_manager.py   # Vector DB operations
│   │   └── neo4j_manager.py    # Graph DB operations
│   ├── loaders/
│   │   ├── base.py             # Abstract loader
│   │   ├── hotpotqa.py         # HotpotQA loader
│   │   ├── twowiki.py          # 2Wiki loader
│   │   ├── musique.py          # MuSiQue loader
│   │   └── multihop_rag.py     # MultiHop-RAG loader
│   └── strategies/
│       ├── abstract_strategy.py
│       ├── naive/base.py
│       ├── graph_rag/base.py
│       ├── agentic/base.py
│       └── decomposition/base.py
├── tests/                      # Test suite
├── qdrant_data/               # Qdrant persistence
└── neo4j_data/                # Neo4j persistence
```

---

## Quick Start Checklist

- [ ] Clone repository
- [ ] Create and activate virtual environment
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] Start Docker services: `docker compose up -d`
- [ ] Run demo: `python scripts/demo_loaders.py`
- [ ] Index data: `python scripts/index_data.py`
- [ ] Run evaluation: `python scripts/evaluate_strategy.py`
- [ ] View results: `mlflow ui`
- [ ] Run tests: `pytest`

---

*Generated for RAGify Project - Last Updated: 2026-02-27*
