# Phase 1: Data Ingestion & Normalization Architecture

## Approach
We are using **Option A: HuggingFace Datasets Integration**. This means we will fetch datasets natively from the Hugging Face hub (when available) and process them directly to our internal schema.

## Schema
Target Schema: `UnifiedQASample` (Pydantic model)
Fields:
- `dataset_name` (str)
- `sample_id` (str)
- `query` (str)
- `ground_truth_answer` (str)
- `supporting_contexts` (List[Dict[str, str]])
- `corpus` (List[Dict[str, str]])
- `metadata` (Dict[str, Any])

## Loaders
Each dataset loader will inherit from an `AbstractDatasetLoader` in `src.loaders.base`.
The loader will be responsible for mapping specific dataset structures:
- **HotpotQA**: Mapping paragraphs and supporting facts to contexts.
- **2WikiMultiHopQA**: Mapping relation graphs (evidences) into metadata.
- **MuSiQue**: Mapping question_decomposition.
- **MultiHop-RAG**: Mapping evidence_list.
