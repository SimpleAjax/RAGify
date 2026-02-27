# Phase 5: Experiment Tracking (MLflow) Architecture

## Overview
To systematically compare the performance of different RAG pipelines (Naive vs. GraphRAG vs. Agentic) across multiple datasets (HotpotQA, MuSiQue, etc.), we need a robust experiment tracking system. MLflow is the industry standard for this.

## 1. Component Architecture
We will introduce an `ExperimentTracker` module responsible for logging all parameters, metrics, and artifacts generated during an evaluation run.

### The `ExperimentTracker` Wrapper
This component (`src/evaluation/tracker.py`) will wrap MLflow's tracking logic to:
*   **Initialize:** Set up the MLflow tracking URI (defaulting to local `./mlruns`).
*   **Log Parameters:** Record RAG strategy type, LLM used, chunk size, vector/graph DB parameters, and dataset name.
*   **Log Metrics:** Record the final RAGAS scores (Context Precision, Recall, Faithfulness, Answer Relevancy).
*   **Log Artifacts:** Save the detailed Pandas DataFrame evaluation results as a CSV artifact directly into the MLflow run.

## 2. Integration
The tracking module will be injected into our existing execution CLI (`scripts/evaluate_strategy.py`). The evaluation will happen inside an `mlflow.start_run()` context block to tie metrics directly to the specific execution parameters.

## 3. Storage
For local development, we will use a **Local File-Based Tracking URI** creating an `mlruns` directory. This avoids setting up an external tracking server while providing full dashboard visibility through `mlflow ui`.
