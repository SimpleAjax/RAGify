# OpenRouter Model Recommendations for RAGify

This guide helps you choose the right models from OpenRouter for different components of the RAG pipeline, balancing **cost** and **quality**.

---

## Current Architecture Overview

| Component | Current Model | Type | Runs On |
|-----------|---------------|------|---------|
| Embeddings | BAAI/bge-small-en-v1.5 | Local HuggingFace | Your Machine (FREE) |
| RAGAS Evaluation | gpt-4o-mini (default) | LiteLLM/OpenAI | API ($) |
| RAG Strategies | User-provided | Various | API ($) |

---

## Model Selection Strategy

### Keep Embeddings Local (FREE)

**Current: `BAAI/bge-small-en-v1.5`**
- ✅ **Keep this** - It's local, fast, and high quality
- Cost: **$0** (runs on your machine)
- Dimensions: 384
- Performance: Excellent for retrieval
- Alternative upgrade: `BAAI/bge-large-en-v1.5` (1024 dims, better quality, still free)

```python
# In src/indexing/qdrant_manager.py
embedding_model_name="BAAI/bge-small-en-v1.5"  # Current - good balance
# OR
embedding_model_name="BAAI/bge-large-en-v1.5"  # Better quality, same cost (FREE)
```

---

## OpenRouter Model Recommendations by Use Case

### 1. RAGAS Evaluation (Quality Critical, Frequent Calls)

**Purpose:** Judging answer quality, faithfulness, context precision/recall

| Budget | Model | OpenRouter ID | Why |
|--------|-------|---------------|-----|
| **Ultra-Low** | Mistral 7B Instruct | `openrouter/mistralai/mistral-7b-instruct` | Cheapest, decent evaluation |
| **Balanced** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | Fast, good at judging |
| **Best Quality** | Claude 3.5 Sonnet | `openrouter/anthropic/claude-3.5-sonnet` | Best evaluator |

**Recommendation: `claude-3-haiku`**
- Excellent at evaluation tasks
- 10x cheaper than GPT-4/Claude-3.5-Sonnet
- Fast response times

```python
# For evaluation
python scripts/evaluate_strategy.py \
  --model openrouter/anthropic/claude-3-haiku \
  --api-base https://openrouter.ai/api/v1
```

**Estimated Cost:** ~$0.0001-0.0002 per sample evaluated

---

### 2. Query Decomposition (Structured Output Required)

**Purpose:** Breaking complex queries into sub-queries (needs structured JSON output)

| Budget | Model | OpenRouter ID | Why |
|--------|-------|---------------|-----|
| **Ultra-Low** | Mistral 7B Instruct | `openrouter/mistralai/mistral-7b-instruct` | Cheap, good JSON |
| **Balanced** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | Reliable structured output |
| **Best Quality** | GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | Best JSON adherence |

**Recommendation: `gpt-4o-mini` or `claude-3-haiku`**
- Structured output is critical here
- GPT-4o-mini is cheap and excellent at JSON

```python
# In your RAG strategy code
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="openrouter/openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="your-key"
)

decomposition_rag = DecompositionRAG(retriever=retriever, llm=llm)
```

---

### 3. Entity Extraction for GraphRAG (Structured Output Required)

**Purpose:** Extracting entities from queries for graph traversal

| Budget | Model | OpenRouter ID | Why |
|--------|-------|---------------|-----|
| **Ultra-Low** | Mistral 7B Instruct | `openrouter/mistralai/mistral-7b-instruct` | Cheap NER |
| **Balanced** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | Good entity recognition |
| **Best Quality** | GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | Precise extraction |

**Recommendation: `claude-3-haiku`**
- Good at recognizing named entities
- Reliable structured output
- Cost-effective

---

### 4. Answer Generation (NaiveRAG, GraphRAG Final Answer)

**Purpose:** Synthesizing final answers from retrieved context

| Budget | Model | OpenRouter ID | Why |
|--------|-------|---------------|-----|
| **Ultra-Low** | Llama 3.1 8B | `openrouter/meta-llama/llama-3.1-8b-instruct` | Very cheap, decent |
| **Low** | Mistral 7B Instruct | `openrouter/mistralai/mistral-7b-instruct` | Good quality/cost |
| **Balanced** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | Fast, good reasoning |
| **High Quality** | GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | Best cheap option |
| **Premium** | Claude 3.5 Sonnet | `openrouter/anthropic/claude-3.5-sonnet` | Best quality |

**Recommendation: `gpt-4o-mini`**
- Excellent quality for the price
- Good at following instructions (sticking to context)
- Reduces hallucinations

```python
# Cost-effective generation
llm = ChatOpenAI(
    model_name="openrouter/openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="your-key"
)

naive_rag = NaiveRAG(retriever=retriever, llm=llm)
graph_rag = GraphRAG(graph_retriever=graph_retriever, llm=llm)
```

---

### 5. Agentic RAG (Tool Calling + Reasoning)

**Purpose:** Multi-step reasoning, tool calling, iterative retrieval

| Budget | Model | OpenRouter ID | Why |
|--------|-------|---------------|-----|
| **Balanced** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | Good tool use |
| **Better** | GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | Reliable tool calling |
| **Best** | Claude 3.5 Sonnet | `openrouter/anthropic/claude-3.5-sonnet` | Best reasoning |

**Recommendation: `claude-3-haiku` or `gpt-4o-mini`**
- Agentic tasks need reliable tool calling
- Haiku is surprisingly good at this for the price

```python
# For agentic RAG
llm = ChatOpenAI(
    model_name="openrouter/anthropic/claude-3-haiku",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="your-key"
)

agentic_rag = AgenticRAG(tools=tools, llm=llm)
```

---

## Recommended Configurations by Budget

### 💰 Ultra-Low Budget (~$0.001-0.005 per query)

**Best for:** Development, testing, high-volume low-stakes scenarios

```bash
# Evaluation (cheapest viable option)
--model openrouter/mistralai/mistral-7b-instruct

# RAG Strategies (all use same cheap model)
LLM Model: openrouter/meta-llama/llama-3.1-8b-instruct
```

**Pros:** Extremely cheap  
**Cons:** May have lower quality evaluations, more hallucinations

---

### ⚖️ Balanced Budget (~$0.005-0.02 per query) ⭐ RECOMMENDED

**Best for:** Production use, good quality at reasonable cost

```bash
# Evaluation (good judge, cheap)
--model openrouter/anthropic/claude-3-haiku

# RAG Strategies
# - Query Decomposition: openrouter/openai/gpt-4o-mini
# - Entity Extraction: openrouter/anthropic/claude-3-haiku
# - Answer Generation: openrouter/openai/gpt-4o-mini
# - Agentic: openrouter/anthropic/claude-3-haiku
```

**Setup Example:**

```python
from langchain_openai import ChatOpenAI

# For structured tasks (decomposition, entity extraction)
structured_llm = ChatOpenAI(
    model_name="openrouter/openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="your-key"
)

# For generation and agentic
generation_llm = ChatOpenAI(
    model_name="openrouter/openai/gpt-4o-mini",  # or claude-3-haiku
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="your-key"
)

# Build strategies
decomposition_rag = DecompositionRAG(retriever=retriever, llm=structured_llm)
graph_rag = GraphRAG(graph_retriever=graph_retriever, llm=structured_llm)
naive_rag = NaiveRAG(retriever=retriever, llm=generation_llm)
agentic_rag = AgenticRAG(tools=tools, llm=generation_llm)
```

---

### 🏆 High Quality Budget (~$0.02-0.10 per query)

**Best for:** Critical applications, research, benchmarking

```bash
# Evaluation (best judge)
--model openrouter/anthropic/claude-3.5-sonnet

# RAG Strategies (all use best models)
# - All tasks: openrouter/anthropic/claude-3.5-sonnet
# OR
# - Complex reasoning: claude-3.5-sonnet
# - Simple generation: openrouter/openai/gpt-4o
```

---

## Specific Model Deep Dive

### 1. Claude 3 Haiku (`openrouter/anthropic/claude-3-haiku`)
- **Speed:** Very Fast
- **Cost:** ~$0.25/M tokens input, ~$1.25/M tokens output
- **Best for:** Evaluation, entity extraction, simple generation
- **Why:** Excellent price/performance, reliable structured outputs

### 2. GPT-4o Mini (`openrouter/openai/gpt-4o-mini`)
- **Speed:** Fast
- **Cost:** ~$0.15/M tokens input, ~$0.60/M tokens output
- **Best for:** Structured outputs, generation, tool calling
- **Why:** Cheapest reliable model, excellent JSON adherence

### 3. Claude 3.5 Sonnet (`openrouter/anthropic/claude-3.5-sonnet`)
- **Speed:** Medium
- **Cost:** ~$3/M tokens input, ~$15/M tokens output
- **Best for:** Complex reasoning, evaluation, agentic tasks
- **Why:** Best quality for reasoning and evaluation

### 4. Mistral 7B Instruct (`openrouter/mistralai/mistral-7b-instruct`)
- **Speed:** Fast
- **Cost:** ~$0.07/M tokens input, ~$0.07/M tokens output
- **Best for:** Ultra-low budget scenarios
- **Why:** Cheapest viable option

---

## Cost Estimation Examples

### Scenario: Evaluating 100 samples with RAGAS

| Model | Est. Cost |
|-------|-----------|
| mistral-7b-instruct | ~$0.01-0.02 |
| claude-3-haiku | ~$0.03-0.05 |
| gpt-4o-mini | ~$0.02-0.04 |
| claude-3.5-sonnet | ~$0.50-1.00 |

### Scenario: 100 RAG queries with decomposition

| Model Strategy | Est. Cost |
|----------------|-----------|
| All mistral-7b | ~$0.05-0.10 |
| All claude-3-haiku | ~$0.20-0.40 |
| All gpt-4o-mini | ~$0.15-0.30 |
| All claude-3.5-sonnet | ~$3.00-6.00 |
| Mixed (recommended) | ~$0.15-0.30 |

---

## Configuration via .env File

RAGify now supports easy model configuration via `.env` file. Copy `.env.example` to `.env` and customize:

```bash
# Recommended balanced configuration (.env)
OPENROUTER_API_KEY=sk-or-v1-your-key

EVALUATION_MODEL=openrouter/anthropic/claude-3-haiku
DECOMPOSITION_MODEL=openrouter/openai/gpt-4o-mini
ENTITY_EXTRACTION_MODEL=openrouter/anthropic/claude-3-haiku
GENERATION_MODEL=openrouter/openai/gpt-4o-mini
AGENTIC_MODEL=openrouter/anthropic/claude-3-haiku
```

Or use a single model for simplicity:
```bash
RAG_MODEL=openrouter/openai/gpt-4o-mini
```

### 1. Use Different Models for Different Tasks

```python
# Don't use one model for everything - optimize by task

# Cheap model for evaluation (called frequently)
eval_llm = ChatOpenAI(
    model_name="openrouter/anthropic/claude-3-haiku",
    openai_api_base="https://openrouter.ai/api/v1"
)

# Good model for generation (called once per query)
gen_llm = ChatOpenAI(
    model_name="openrouter/openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1"
)

# Best model for critical reasoning (called rarely)
reasoning_llm = ChatOpenAI(
    model_name="openrouter/anthropic/claude-3.5-sonnet",
    openai_api_base="https://openrouter.ai/api/v1"
)
```

### 2. Set Up Cost Tracking

```python
# Track costs in MLflow
from src.evaluation.tracker import ExperimentTracker

tracker = ExperimentTracker()
with tracker.start_run(run_name="cost_tracking"):
    tracker.log_parameters({
        "eval_model": "claude-3-haiku",
        "gen_model": "gpt-4o-mini",
        "query_count": 100
    })
```

### 3. Use Local Embeddings

```python
# NEVER use API embeddings - use local HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"  # FREE
)
```

### 4. Cache Where Possible

```python
# Use LangChain's caching for repeated queries
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

---

## Summary: My Top Recommendations

| Use Case | Recommended Model | OpenRouter ID | Est. Cost per 1K tokens |
|----------|------------------|---------------|------------------------|
| **Embeddings** | BGE Small (local) | FREE | $0 |
| **RAGAS Evaluation** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | ~$0.25-1.25 |
| **Query Decomposition** | GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | ~$0.15-0.60 |
| **Entity Extraction** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | ~$0.25-1.25 |
| **Answer Generation** | GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | ~$0.15-0.60 |
| **Agentic Tasks** | Claude 3 Haiku | `openrouter/anthropic/claude-3-haiku` | ~$0.25-1.25 |
| **Premium Quality** | Claude 3.5 Sonnet | `openrouter/anthropic/claude-3.5-sonnet` | ~$3-15 |

---

## Quick Start Command

```bash
# 1. Copy and configure .env
copy .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 2. Index data first
python scripts/index_data.py

# 3. Run full RAG evaluation
python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 100

# 4. Show current configuration
python scripts/evaluate_strategy.py --show-config
```

### Important: Two Different Scripts

| Script | Purpose | Use When |
|--------|---------|----------|
| `run_rag_evaluation.py` | **Full RAG pipeline** - runs strategy, generates answers, evaluates | Testing RAG strategies end-to-end |
| `evaluate_strategy.py` | **RAGAS only** - evaluates pre-generated answers | You already have answers in JSON format |

### Switching Models

Edit `.env` to switch models:

```bash
# Budget mode
EVALUATION_MODEL=openrouter/mistralai/mistral-7b-instruct

# Balanced mode (recommended)
EVALUATION_MODEL=openrouter/anthropic/claude-3-haiku

# Premium mode
EVALUATION_MODEL=openrouter/anthropic/claude-3.5-sonnet
```

Or use a single model for everything:
```bash
RAG_MODEL=openrouter/openai/gpt-4o-mini
```

---

*Last Updated: 2026-02-27*  
*OpenRouter pricing subject to change - check https://openrouter.ai/models for current rates*
