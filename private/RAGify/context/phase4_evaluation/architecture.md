# Phase 4 Evaluation Engine Architecture

*Status: Pending User Decision*

## Overview
This component integrates the **RAGAS** framework to evaluate the RAG strategies built in Phase 2. RAGAS provides automated metrics based on LLM-as-a-judge techniques.
The core metrics we will compute are:
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

## Architecture Options Selected
- **Option C (LiteLLM abstraction layer):** We wrap the RAGAS judge in Litellm, allowing us to route requests to local models (e.g., Ollama) or OpenAI as needed.


## Interfaces
The evaluator will take:
1.  **A list of evaluation samples:** (query, ground_truth, generated_answer, retrieved_contexts)
2.  **An initialized LLM Judge:** configured to score the samples.

It will output:
-   A Pandas DataFrame containing score distributions.
-   Aggregated metric averages.
