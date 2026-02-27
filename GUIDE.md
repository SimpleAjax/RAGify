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

## Configuration via Environment Variables (.env)

RAGify uses a `.env` file for configuration. Copy `.env.example` to `.env` and customize:

```bash
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

Then edit `.env` with your API keys and preferred models.

### Required API Keys

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes* | OpenRouter API key (recommended) |
| `OPENAI_API_KEY` | Alternative | OpenAI API key (if not using OpenRouter) |

*Get your OpenRouter key from https://openrouter.ai/keys

### Model Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVALUATION_MODEL` | `openrouter/anthropic/claude-3-haiku` | RAGAS evaluation (frequent calls) |
| `DECOMPOSITION_MODEL` | `openrouter/openai/gpt-4o-mini` | Query decomposition (JSON output) |
| `ENTITY_EXTRACTION_MODEL` | `openrouter/anthropic/claude-3-haiku` | GraphRAG entity extraction |
| `GENERATION_MODEL` | `openrouter/openai/gpt-4o-mini` | Answer generation |
| `AGENTIC_MODEL` | `openrouter/anthropic/claude-3-haiku` | Agentic RAG reasoning |

### Alternative: Single Model for All RAG Tasks

Set `RAG_MODEL` to use one model for all strategies instead of individual models:

```bash
RAG_MODEL=openrouter/openai/gpt-4o-mini
RAG_API_BASE=https://openrouter.ai/api/v1
```

### Other Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Local embedding model (FREE) |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `CHUNK_SIZE` | `512` | Text chunking size |
| `CHUNK_OVERLAP` | `50` | Text chunking overlap |

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
| `api_base` | `None` | Custom API base URL (e.g., OpenRouter, Ollama) |
| `api_key` | `None` | API key (uses env vars if not provided) |

**Supported Model Formats:**
- OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
- Ollama: `ollama/llama3`, `ollama/mistral`
- Anthropic: `claude-3-sonnet`
- **OpenRouter**: `openrouter/openai/gpt-4o`, `openrouter/anthropic/claude-3.5-sonnet`, `openrouter/google/gemini-pro`, etc.

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
# Show current dataset configuration
python scripts/index_data.py --show-config

# Index both vector and graph data (uses settings from .env)
python scripts/index_data.py

# Skip vector indexing (Qdrant)
python scripts/index_data.py --skip-vector

# Skip graph indexing (Neo4j)
python scripts/index_data.py --skip-graph
```

**What it does:**
- Loads samples from each dataset (configurable via .env)
- Chunks and embeds texts using configured embedding model
- Stores vectors in Qdrant with dataset metadata tagging
- Extracts and stores graph triples in Neo4j (for 2WikiMultiHopQA)

**Configuration via .env:**
```bash
# Number of samples to index (-1 = ALL, 0 = skip, N = N samples)
HOTPOTQA_SAMPLE_SIZE=100
TWOWIKI_SAMPLE_SIZE=100
MUSIQUE_SAMPLE_SIZE=100
MULTIHOP_RAG_SAMPLE_SIZE=100

# Which splits to use
HOTPOTQA_SPLIT=validation
TWOWIKI_SPLIT=train
MUSIQUE_SPLIT=validation
MULTIHOP_RAG_SPLIT=train
```

**Dataset Sizes:**
| Dataset | Split | Total Samples | Default Indexed |
|---------|-------|--------------|-----------------|
| HotpotQA | validation | ~7,405 | 100 |
| 2WikiMultiHopQA | train | ~167,454 | 100 |
| MuSiQue | validation | ~2,417 | 100 |
| MultiHop-RAG | train | ~2,556 | 100 |

### 3. Run Evaluation

Evaluate RAG strategies using RAGAS metrics:

```bash
# Show current configuration
python scripts/evaluate_strategy.py --show-config

# Run with default model from .env (EVALUATION_MODEL)
python scripts/evaluate_strategy.py --output results.csv

# Override with specific model
python scripts/evaluate_strategy.py \
  --model openrouter/anthropic/claude-3.5-sonnet \
  --output results.csv

# With custom input file
python scripts/evaluate_strategy.py \
  --input eval_data.json \
  --output results.csv \
  --run-name "experiment_1"
```

The script will automatically use models from your `.env` file. No need to specify `--model` and `--api-base` every time!

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

## Using OpenRouter

[OpenRouter](https://openrouter.ai/) provides a unified API for accessing 200+ LLMs from various providers (OpenAI, Anthropic, Google, Meta, etc.) through a single endpoint.

### Why Use OpenRouter?
- Access to 200+ models from different providers
- Pay-as-you-go pricing
- Standardized API format
- No need to manage multiple API keys

### Setup

**1. Get an API Key**
- Sign up at [openrouter.ai](https://openrouter.ai/)
- Generate an API key from your dashboard

**2. Set Environment Variable**

```bash
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "your-openrouter-key"

# Windows (CMD)
set OPENROUTER_API_KEY=your-openrouter-key

# Linux/Mac
export OPENROUTER_API_KEY="your-openrouter-key"
```

### Usage Examples

**Using OpenRouter with any model:**

```bash
# GPT-4o via OpenRouter
python scripts/evaluate_strategy.py \
  --model openrouter/openai/gpt-4o \
  --api-base https://openrouter.ai/api/v1 \
  --output results.csv

# Claude 3.5 Sonnet via OpenRouter
python scripts/evaluate_strategy.py \
  --model openrouter/anthropic/claude-3.5-sonnet \
  --api-base https://openrouter.ai/api/v1 \
  --output results.csv

# Google Gemini Pro via OpenRouter
python scripts/evaluate_strategy.py \
  --model openrouter/google/gemini-pro \
  --api-base https://openrouter.ai/api/v1 \
  --output results.csv

# Meta Llama 3 via OpenRouter
python scripts/evaluate_strategy.py \
  --model openrouter/meta-llama/llama-3-70b-instruct \
  --api-base https://openrouter.ai/api/v1 \
  --output results.csv
```

**Passing API key directly (not recommended for production):**

```bash
python scripts/evaluate_strategy.py \
  --model openrouter/anthropic/claude-3.5-sonnet \
  --api-base https://openrouter.ai/api/v1 \
  --api-key sk-or-v1-your-key-here \
  --output results.csv
```

### Popular OpenRouter Models

| Model | OpenRouter Identifier | Provider |
|-------|----------------------|----------|
| GPT-4o | `openrouter/openai/gpt-4o` | OpenAI |
| GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | OpenAI |
| Claude 3.5 Sonnet | `openrouter/anthropic/claude-3.5-sonnet` | Anthropic |
| Claude 3 Opus | `openrouter/anthropic/claude-3-opus` | Anthropic |
| Gemini Pro 1.5 | `openrouter/google/gemini-pro-1.5` | Google |
| Llama 3 70B | `openrouter/meta-llama/llama-3-70b-instruct` | Meta |
| Mistral Large | `openrouter/mistralai/mistral-large` | Mistral |
| DeepSeek V3 | `openrouter/deepseek/deepseek-chat` | DeepSeek |

### Using OpenRouter in Custom Code

```python
from src.evaluation.evaluator import RagasEvaluator
from src.strategies.naive.base import NaiveRAG
from langchain_openai import ChatOpenAI

# Initialize evaluator with OpenRouter
evaluator = RagasEvaluator(
    model_name="openrouter/anthropic/claude-3.5-sonnet",
    api_base="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key"  # Or use OPENROUTER_API_KEY env var
)

# Or use with RAG strategies
llm = ChatOpenAI(
    model_name="openrouter/openai/gpt-4o",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="your-openrouter-key"
)

rag = NaiveRAG(retriever=retriever, llm=llm)
```

### OpenRouter-Specific Headers (Optional)

For tracking and analytics, OpenRouter supports custom headers:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="openrouter/anthropic/claude-3.5-sonnet",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="your-openrouter-key",
    model_kwargs={
        "extra_headers": {
            "HTTP-Referer": "https://your-app.com",  # Optional
            "X-Title": "RAGify Evaluation"  # Optional
        }
    }
)
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

---

## Complete End-to-End Workflow

This section provides a comprehensive step-by-step guide to run the entire RAGify pipeline, test all functionalities, and use MLflow to compare results.

### Phase 1: Initial Setup

```bash
# 1. Clone and navigate to project (if not already done)
cd RAGify

# 2. Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies (if not already done)
pip install -r requirements.txt

# 4. Copy environment template
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# 5. Edit .env with your API keys and preferences
# Minimum required: OPENROUTER_API_KEY or OPENAI_API_KEY
```

### Phase 2: Configure Your Environment

Edit `.env` file with your settings:

```bash
# Required API Key
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# For quick testing (small sample)
HOTPOTQA_SAMPLE_SIZE=100
TWOWIKI_SAMPLE_SIZE=100
MUSIQUE_SAMPLE_SIZE=100
MULTIHOP_RAG_SAMPLE_SIZE=100

# For full benchmark (uncomment when ready)
# HOTPOTQA_SAMPLE_SIZE=-1
# TWOWIKI_SAMPLE_SIZE=-1
# MUSIQUE_SAMPLE_SIZE=-1
# MULTIHOP_RAG_SAMPLE_SIZE=-1

# Recommended balanced models
EVALUATION_MODEL=openrouter/anthropic/claude-3-haiku
DECOMPOSITION_MODEL=openrouter/openai/gpt-4o-mini
ENTITY_EXTRACTION_MODEL=openrouter/anthropic/claude-3-haiku
GENERATION_MODEL=openrouter/openai/gpt-4o-mini
AGENTIC_MODEL=openrouter/anthropic/claude-3-haiku
```

### Phase 3: Start Infrastructure

```bash
# Start Qdrant and Neo4j containers
docker compose up -d

# Verify services are running
docker ps

# Check configuration
python scripts/index_data.py --show-config
python scripts/evaluate_strategy.py --show-config
```

### Phase 4: Index Data

```bash
# Index data (uses settings from .env)
python scripts/index_data.py

# Skip graph indexing (if you only need vector search)
python scripts/index_data.py --skip-graph

# Skip vector indexing (if you only need graph)
python scripts/index_data.py --skip-vector

# For full dataset indexing, update .env first, then:
# python scripts/index_data.py
```

### Phase 5: Run Tests

```bash
# Run all unit tests (no Docker required for these)
pytest tests/ -v -k "not integration"

# Run specific test categories
pytest tests/evaluation/ -v
pytest tests/indexing/ -v
pytest tests/loaders/ -v
pytest tests/strategies/ -v -k "not integration"

# Run with coverage report
pytest tests/ -v -k "not integration" --cov=src --cov-report=html
# Then open: htmlcov/index.html

# Run integration tests (requires Docker containers running)
pytest tests/strategies/integration/ -v
```

### Phase 6: Understanding the Evaluation Scripts

RAGify has **two evaluation scripts**:

#### Script 1: `evaluate_strategy.py` - Quick RAGAS Evaluation Only
This script **only evaluates pre-generated answers** (dummy data or your JSON file). It does NOT run RAG strategies.

```bash
# Use this when you already have answers + contexts to evaluate
python scripts/evaluate_strategy.py --input my_results.json --output results.csv
```

#### Script 2: `run_rag_evaluation.py` - Full RAG Pipeline (NEW)
This script **runs the complete RAG pipeline**:
1. Loads questions from indexed datasets
2. Runs through specified RAG strategy (Naive/Graph/Agentic/Decomposition)
3. Generates answers using LLM + retrieval
4. Evaluates with RAGAS
5. Logs to MLflow

```bash
# Evaluate NaiveRAG on HotpotQA (100 samples)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100

# Evaluate GraphRAG on 2WikiMultiHopQA
python scripts/run_rag_evaluation.py --strategy graph --dataset 2WikiMultiHopQA --samples 50

# Evaluate with specific model
python scripts/run_rag_evaluation.py --strategy decomposition --dataset HotpotQA \
  --model openrouter/anthropic/claude-3.5-sonnet --samples 100
```

### Phase 7: Run Complete RAG Evaluations with Different Strategies

Test different RAG strategies and models, track everything in MLflow:

#### Step 1: Evaluate Each RAG Strategy

```bash
# --- NAIVE RAG ---
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100 \
  --run-name "naive_hotpot_100"

# --- GRAPH RAG ---
python scripts/run_rag_evaluation.py --strategy graph --dataset HotpotQA --samples 100 \
  --run-name "graph_hotpot_100"

# --- AGENTIC RAG ---
python scripts/run_rag_evaluation.py --strategy agentic --dataset HotpotQA --samples 100 \
  --run-name "agentic_hotpot_100"

# --- DECOMPOSITION RAG ---
python scripts/run_rag_evaluation.py --strategy decomposition --dataset HotpotQA --samples 100 \
  --run-name "decomposition_hotpot_100"
```

#### Step 2: Compare Different Models (Same Strategy)

```bash
# Test different generation models with NaiveRAG

# GPT-4o Mini (fast & cheap)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100 \
  --model openrouter/openai/gpt-4o-mini --run-name "naive_gpt4omini"

# Claude 3 Haiku (balanced)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100 \
  --model openrouter/anthropic/claude-3-haiku --run-name "naive_haiku"

# Claude 3.5 Sonnet (premium)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100 \
  --model openrouter/anthropic/claude-3.5-sonnet --run-name "naive_sonnet"

# Mistral 7B (budget)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100 \
  --model openrouter/mistralai/mistral-7b-instruct --run-name "naive_mistral"
```

#### Step 3: Test Different Datasets

```bash
# Same strategy, different datasets
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100 \
  --run-name "naive_hotpot"

python scripts/run_rag_evaluation.py --strategy naive --dataset 2WikiMultiHopQA --samples 100 \
  --run-name "naive_2wiki"

python scripts/run_rag_evaluation.py --strategy naive --dataset MuSiQue --samples 100 \
  --run-name "naive_musique"

python scripts/run_rag_evaluation.py --strategy naive --dataset MultiHopRAG --samples 100 \
  --run-name "naive_multihop"
```

### Phase 8: Compare Results in MLflow

```bash
# Start MLflow UI
mlflow ui

# Open browser and go to: http://localhost:5000
```

In MLflow UI:
1. **View All Runs**: See all your evaluation runs in the "RAGify_Evaluations" experiment
2. **Compare Runs**: Select multiple runs and click "Compare" to see side-by-side metrics
3. **View Metrics**: See average scores for:
   - Context Precision
   - Context Recall
   - Faithfulness
   - Answer Relevancy
4. **Download Artifacts**: Each run includes the detailed CSV results

### Phase 9: Advanced Evaluation Scenarios

#### Scenario A: Test with Different Sample Counts

```bash
# Test with different evaluation sample sizes (not indexing size)
# Uses same indexed data, just evaluates on different number of questions

# Small test (50 questions)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 50 \
  --run-name "naive_50samples"

# Medium test (200 questions)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 200 \
  --run-name "naive_200samples"

# Full test (1000 questions - make sure you indexed enough!)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 1000 \
  --run-name "naive_1000samples"
```

#### Scenario B: Test on Different Datasets

```bash
# First, make sure you indexed all datasets you want to test
# Edit .env:
HOTPOTQA_SAMPLE_SIZE=1000
TWOWIKI_SAMPLE_SIZE=1000
MUSIQUE_SAMPLE_SIZE=1000
MULTIHOP_RAG_SAMPLE_SIZE=1000

python scripts/index_data.py

# Now test same strategy on all datasets
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100 \
  --run-name "naive_hotpot"

python scripts/run_rag_evaluation.py --strategy naive --dataset 2WikiMultiHopQA --samples 100 \
  --run-name "naive_2wiki"

python scripts/run_rag_evaluation.py --strategy naive --dataset MuSiQue --samples 100 \
  --run-name "naive_musique"

python scripts/run_rag_evaluation.py --strategy naive --dataset MultiHopRAG --samples 100 \
  --run-name "naive_multihop"
```

#### Scenario C: Compare All 4 RAG Strategies Side-by-Side

```bash
# Index enough data first
HOTPOTQA_SAMPLE_SIZE=500
python scripts/index_data.py

# Run all strategies on same dataset
for strategy in naive graph agentic decomposition; do
  python scripts/run_rag_evaluation.py --strategy $strategy --dataset HotpotQA --samples 100 \
    --run-name "${strategy}_comparison" --output "results_${strategy}.csv"
done

# Then view all in MLflow to compare
mlflow ui
```

### Phase 10: View and Analyze Results

```bash
# View CSV results directly
type results_baseline.csv  # Windows
# cat results_baseline.csv  # Linux/Mac

# Python analysis
python -c "
import pandas as pd
df = pd.read_csv('results_baseline.csv')
print(df.describe())
print('\nAverage Metrics:')
print(df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean())
"
```

### Phase 11: Cleanup

```bash
# Stop Docker containers
docker compose down

# To remove all data (WARNING: deletes indexed data)
docker compose down -v

# Remove MLflow runs (if needed)
# rm -rf mlruns/
```

---

## Quick Reference Command Cheat Sheet

| Task | Command |
|------|---------|
| **Setup** | `copy .env.example .env` → Edit with API keys |
| **Start Infra** | `docker compose up -d` |
| **Show Config** | `python scripts/evaluate_strategy.py --show-config` |
| **Index Data** | `python scripts/index_data.py` |
| **Run Tests** | `pytest tests/ -v -k "not integration"` |
| **Full RAG Eval** | `python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100` |
| **MLflow UI** | `mlflow ui` → http://localhost:5000 |
| **Stop Infra** | `docker compose down` |

### Script Differences

| Script | Purpose | Use When |
|--------|---------|----------|
| `evaluate_strategy.py` | Evaluates pre-generated answers | You have JSON with answers/contexts already |
| `run_rag_evaluation.py` | Full RAG pipeline + evaluation | You want to test RAG strategies end-to-end |

---

## Example Complete Workflow (Copy & Paste)

```bash
# 1. Setup
copy .env.example .env
# (Edit .env with OPENROUTER_API_KEY)

# 2. Start infrastructure
docker compose up -d

# 3. Index small sample for testing (100 samples each)
python scripts/index_data.py

# 4. Run quick test
pytest tests/ -v -k "not integration" --tb=short

# 5. Test ONE RAG strategy
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 50 \
  --run-name "test_naive"

# 6. Compare ALL RAG strategies (after confirming one works)
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 50 --run-name "naive_test"
python scripts/run_rag_evaluation.py --strategy graph --dataset HotpotQA --samples 50 --run-name "graph_test"
python scripts/run_rag_evaluation.py --strategy agentic --dataset HotpotQA --samples 50 --run-name "agentic_test"
python scripts/run_rag_evaluation.py --strategy decomposition --dataset HotpotQA --samples 50 --run-name "decomposition_test"

# 7. View results in MLflow
mlflow ui
# Open http://localhost:5000

# 8. For full benchmark:
#    - Edit .env to set all SAMPLE_SIZE=-1 (or larger number)
#    - Re-index with more data
#    - Re-run evaluations with more samples

# Edit .env: HOTPOTQA_SAMPLE_SIZE=1000, etc.
python scripts/index_data.py

python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 500 \
  --run-name "naive_full"
python scripts/run_rag_evaluation.py --strategy graph --dataset HotpotQA --samples 500 \
  --run-name "graph_full"
```

---

## How RAG Evaluation Works (Data Flow)

Understanding how data flows through the evaluation pipeline:

### 1. Data Loading Flow

```
HuggingFace Datasets
├── HotpotQA (7,405 val samples)
├── 2WikiMultiHopQA (167,454 train samples)
├── MuSiQue (2,417 val samples)
└── MultiHopRAG (2,556 train samples)
         ↓
   Dataset Loaders (src/loaders/)
         ↓
   UnifiedQASample (standardized format)
         ↓
   Indexing (scripts/index_data.py)
         ├── Qdrant (vector DB) - for Naive/Agentic/Decomposition
         └── Neo4j (graph DB) - for GraphRAG
```

### 2. Evaluation Flow

```
Evaluation Script (scripts/run_rag_evaluation.py)
         ↓
   Load Questions from Dataset
   (e.g., 100 questions from HotpotQA)
         ↓
   For Each Question:
         ├── RAG Strategy retrieves context
         │   ├── NaiveRAG: Vector search (Qdrant)
         │   ├── GraphRAG: Graph traversal (Neo4j)
         │   ├── AgenticRAG: Multi-step tool use
         │   └── DecompositionRAG: Sub-query + parallel retrieval
         │
         └── LLM generates answer
         ↓
   RAGAS Evaluation
         ├── Context Precision: Was relevant context retrieved?
         ├── Context Recall: Was all necessary context retrieved?
         ├── Faithfulness: Is answer grounded in context?
         └── Answer Relevancy: Does answer match question?
         ↓
   MLflow Logging
         ├── Parameters (strategy, model, dataset)
         ├── Metrics (average scores)
         └── Artifacts (detailed CSV results)
```

### 3. What Gets Evaluated?

| Component | Source | What It Does |
|-----------|--------|--------------|
| **Questions** | HuggingFace datasets | Multi-hop QA questions |
| **Contexts** | Qdrant/Neo4j (indexed) | Retrieved by RAG strategy |
| **Answers** | LLM (OpenRouter) | Generated from context |
| **Ground Truth** | HuggingFace datasets | Correct answer from dataset |
| **Metrics** | RAGAS framework | Automated quality scoring |

### 4. Example Data Flow

For a single HotpotQA question:

```json
{
  "question": "Who is the director of the movie starring Tom Hanks and Meg Ryan?",
  "ground_truth": "Nora Ephron",
  
  // RAG Strategy retrieves:
  "contexts": [
    "Tom Hanks and Meg Ryan starred in 'Sleepless in Seattle' (1993)...",
    "'Sleepless in Seattle' was directed by Nora Ephron..."
  ],
  
  // LLM generates:
  "answer": "Nora Ephron directed 'Sleepless in Seattle' starring Tom Hanks and Meg Ryan.",
  
  // RAGAS evaluates:
  "context_precision": 1.0,  // Both contexts relevant
  "context_recall": 1.0,     // All needed info retrieved
  "faithfulness": 1.0,       // Answer supported by context
  "answer_relevancy": 1.0    // Directly answers question
}
```

---

## FAQ: Common Questions

### Q: Why two evaluation scripts?

**A:** 
- `evaluate_strategy.py`: For evaluating pre-generated results (you already have answers)
- `run_rag_evaluation.py`: For running full RAG pipeline (generates answers + evaluates)

### Q: Do I need to index data before running evaluations?

**A:** Yes! For `run_rag_evaluation.py`, you must:
1. Run `python scripts/index_data.py` first
2. The strategy needs to retrieve from Qdrant/Neo4j

### Q: Can I evaluate on different datasets than I indexed?

**A:** No. You can only evaluate on datasets you indexed. The retrievers query the indexed data.

### Q: How do I compare strategies fairly?

**A:** Use the **same** dataset and **same** questions:
```bash
# Same 100 questions, different strategies
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100
python scripts/run_rag_evaluation.py --strategy graph --dataset HotpotQA --samples 100
python scripts/run_rag_evaluation.py --strategy agentic --dataset HotpotQA --samples 100
```

### Q: What's the difference between indexing sample size and evaluation sample size?

**A:** 
- **Indexing sample size** (`HOTPOTQA_SAMPLE_SIZE`): How many documents to index in the database
- **Evaluation sample size** (`--samples`): How many questions to test (must be ≤ indexed)

Example:
```bash
# Index 1000 documents
HOTPOTQA_SAMPLE_SIZE=1000
python scripts/index_data.py

# Evaluate on 100 of them
python scripts/run_rag_evaluation.py --samples 100

# Later evaluate on all 1000
python scripts/run_rag_evaluation.py --samples 1000
```

*Generated for RAGify Project - Last Updated: 2026-02-27*
