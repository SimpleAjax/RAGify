"""
RAG Strategy Evaluation Pipeline

This script runs a complete RAG evaluation:
1. Loads test questions from datasets
2. Runs them through a specified RAG strategy
3. Generates answers using LLM + retrieval
4. Evaluates results using RAGAS metrics
5. Logs everything to MLflow

Usage:
    # Evaluate with NaiveRAG
    python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 50
    
    # Evaluate with GraphRAG
    python scripts/run_rag_evaluation.py --strategy graph --dataset HotpotQA --samples 50
    
    # Evaluate with AgenticRAG
    python scripts/run_rag_evaluation.py --strategy agentic --dataset HotpotQA --samples 50
    
    # Evaluate with DecompositionRAG
    python scripts/run_rag_evaluation.py --strategy decomposition --dataset HotpotQA --samples 50
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.evaluation.evaluator import RagasEvaluator
from src.evaluation.tracker import ExperimentTracker
from src.indexing.qdrant_manager import QdrantManager
from src.indexing.neo4j_manager import Neo4jManager
from src.schema import UnifiedQASample

# Dataset loaders
from src.loaders.hotpotqa import HotpotQALoader
from src.loaders.twowiki import TwoWikiMultiHopQALoader
from src.loaders.musique import MuSiQueLoader
from src.loaders.multihop_rag import MultiHopRAGLoader

# RAG Strategies
from src.strategies.naive.base import NaiveRAG
from src.strategies.graph_rag.base import GraphRAG
from src.strategies.agentic.base import AgenticRAG
from src.strategies.decomposition.base import DecompositionRAG

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document


def get_dataset_loader(dataset_name: str, split: str = None, max_samples: int = None):
    """Get the appropriate dataset loader."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == "hotpotqa":
        split = split or config.HOTPOTQA_SPLIT
        loader = HotpotQALoader(split=split)
    elif dataset_name == "2wikimultihopqa" or dataset_name == "twowiki":
        split = split or config.TWOWIKI_SPLIT
        loader = TwoWikiMultiHopQALoader(split=split)
    elif dataset_name == "musique":
        split = split or config.MUSIQUE_SPLIT
        loader = MuSiQueLoader(split=split)
    elif dataset_name == "multihoprag" or dataset_name == "multihop":
        split = split or config.MULTIHOP_RAG_SPLIT
        loader = MultiHopRAGLoader(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return loader


def load_test_samples(dataset_name: str, split: str = None, max_samples: int = 100) -> List[UnifiedQASample]:
    """Load test samples from specified dataset."""
    print(f"\nLoading up to {max_samples} samples from {dataset_name}...")
    
    loader = get_dataset_loader(dataset_name, split)
    samples = []
    
    for i, sample in enumerate(loader.load()):
        if i >= max_samples:
            break
        samples.append(sample)
    
    print(f"Loaded {len(samples)} samples.")
    return samples


def create_naive_rag(qdrant: QdrantManager, dataset_name: str, llm: ChatOpenAI) -> NaiveRAG:
    """Create NaiveRAG strategy."""
    retriever = qdrant.get_langchain_retriever(target_dataset=dataset_name, k=5)
    return NaiveRAG(retriever=retriever, llm=llm)


def create_graph_rag(neo4j: Neo4jManager, dataset_name: str, llm: ChatOpenAI) -> GraphRAG:
    """Create GraphRAG strategy."""
    def graph_retriever(entities: List[str]) -> List[Document]:
        docs = []
        with neo4j.driver.session() as session:
            for entity in entities:
                # Query graph for this entity
                query = f"""
                MATCH (start:Entity {{name: '{entity}', dataset: '{dataset_name}'}})-[rel]->(end:Entity)
                RETURN start.name as start_name, type(rel) as rel_type, end.name as end_name
                LIMIT 5
                """
                results = session.run(query)
                for record in results:
                    content = f"{record['start_name']} is {record['rel_type']} {record['end_name']}"
                    docs.append(Document(page_content=content))
        return docs
    
    return GraphRAG(graph_retriever=graph_retriever, llm=llm)


def create_agentic_rag(qdrant: QdrantManager, dataset_name: str, llm: ChatOpenAI) -> AgenticRAG:
    """Create AgenticRAG strategy with tools."""
    retriever = qdrant.get_langchain_retriever(target_dataset=dataset_name, k=5)
    
    @tool
    def vector_search(query: str) -> str:
        """Search the vector database for relevant context."""
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    
    return AgenticRAG(tools=[vector_search], llm=llm)


def create_decomposition_rag(qdrant: QdrantManager, dataset_name: str, llm: ChatOpenAI) -> DecompositionRAG:
    """Create DecompositionRAG strategy."""
    retriever = qdrant.get_langchain_retriever(target_dataset=dataset_name, k=3)
    return DecompositionRAG(retriever=retriever, llm=llm)


def get_rag_strategy(strategy_name: str, qdrant: QdrantManager, neo4j: Neo4jManager, dataset_name: str, llm: ChatOpenAI):
    """Create the specified RAG strategy."""
    strategy_name = strategy_name.lower()
    
    if strategy_name == "naive":
        return create_naive_rag(qdrant, dataset_name, llm)
    elif strategy_name == "graph":
        return create_graph_rag(neo4j, dataset_name, llm)
    elif strategy_name == "agentic":
        return create_agentic_rag(qdrant, dataset_name, llm)
    elif strategy_name == "decomposition":
        return create_decomposition_rag(qdrant, dataset_name, llm)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from: naive, graph, agentic, decomposition")


def run_evaluation(
    strategy_name: str,
    dataset_name: str,
    model_name: str,
    api_base: str,
    max_samples: int,
    output_file: str,
    run_name: str = None
):
    """Run complete RAG evaluation pipeline."""
    
    print("=" * 70)
    print("RAG EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Strategy: {strategy_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Max Samples: {max_samples}")
    print("=" * 70)
    
    # 1. Load test samples from dataset
    samples = load_test_samples(dataset_name, max_samples=max_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return
    
    # 2. Setup LLM
    print(f"\nInitializing LLM: {model_name}")
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_base=api_base,
        openai_api_key=config.get_api_key()
    )
    
    # 3. Setup databases
    print("\nConnecting to databases...")
    try:
        qdrant = QdrantManager()
        print("✓ Qdrant connected")
    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
        qdrant = None
    
    try:
        neo4j = Neo4jManager()
        print("✓ Neo4j connected")
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")
        neo4j = None
    
    if strategy_name.lower() == "graph" and neo4j is None:
        print("ERROR: GraphRAG requires Neo4j but connection failed.")
        return
    
    if strategy_name.lower() != "graph" and qdrant is None:
        print("ERROR: This strategy requires Qdrant but connection failed.")
        return
    
    # 4. Create RAG strategy
    print(f"\nInitializing {strategy_name} RAG strategy...")
    rag = get_rag_strategy(strategy_name, qdrant, neo4j, dataset_name, llm)
    print(f"✓ {strategy_name} RAG ready")
    
    # 5. Generate answers for each sample
    print(f"\nGenerating answers for {len(samples)} questions...")
    eval_data = []
    
    for sample in tqdm(samples, desc="Processing"):
        try:
            # Run RAG pipeline
            result = rag.retrieve_and_generate(query=sample.query)
            
            # Prepare data for RAGAS evaluation
            eval_item = {
                "question": sample.query,
                "answer": result["answer"],
                "contexts": result["retrieved_contexts"],
                "ground_truth": sample.ground_truth_answer
            }
            eval_data.append(eval_item)
            
        except Exception as e:
            print(f"\nError processing sample {sample.sample_id}: {e}")
            # Add empty result to maintain alignment
            eval_item = {
                "question": sample.query,
                "answer": "ERROR: Generation failed",
                "contexts": [],
                "ground_truth": sample.ground_truth_answer
            }
            eval_data.append(eval_item)
    
    print(f"\nGenerated answers for {len(eval_data)} questions.")
    
    # 6. Evaluate with RAGAS
    print("\nEvaluating with RAGAS metrics...")
    evaluator = RagasEvaluator(model_name=config.EVALUATION_MODEL, api_base=config.EVALUATION_API_BASE)
    results_df = evaluator.evaluate_strategy(eval_data)
    
    # 7. Log to MLflow
    tracker = ExperimentTracker()
    with tracker.start_run(run_name=run_name):
        # Log parameters
        tracker.log_parameters({
            "strategy": strategy_name,
            "dataset": dataset_name,
            "generation_model": model_name,
            "evaluation_model": config.EVALUATION_MODEL,
            "max_samples": max_samples,
            "actual_samples": len(eval_data)
        })
        
        # Log metrics
        tracker.log_metrics(results_df)
        tracker.log_evaluation_artifact(results_df, filename=os.path.basename(output_file))
    
    # 8. Save results
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # 9. Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    averages = results_df.select_dtypes(include='number').mean()
    for metric, score in averages.items():
        print(f"  {metric}: {score:.4f}")
    print("=" * 70)
    print(f"\nView detailed results in MLflow: mlflow ui")
    
    # Cleanup
    if neo4j:
        neo4j.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run complete RAG evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate NaiveRAG on HotpotQA
  python scripts/run_rag_evaluation.py --strategy naive --dataset HotpotQA --samples 50
  
  # Evaluate GraphRAG on 2WikiMultiHopQA
  python scripts/run_rag_evaluation.py --strategy graph --dataset 2WikiMultiHopQA --samples 50
  
  # Evaluate with specific model
  python scripts/run_rag_evaluation.py --strategy decomposition --dataset HotpotQA \\
      --model openrouter/anthropic/claude-3.5-sonnet --samples 100
        """
    )
    
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["naive", "graph", "agentic", "decomposition"],
                        help="RAG strategy to evaluate")
    
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["HotpotQA", "2WikiMultiHopQA", "MuSiQue", "MultiHopRAG"],
                        help="Dataset to use for evaluation")
    
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to evaluate (default: 100)")
    
    parser.add_argument("--model", type=str, default=None,
                        help="Model for answer generation (default: GENERATION_MODEL from config)")
    
    parser.add_argument("--api-base", type=str, default=None,
                        help="API base URL (default: GENERATION_API_BASE from config)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file (default: results_<strategy>_<dataset>.csv)")
    
    parser.add_argument("--run-name", type=str, default=None,
                        help="MLflow run name (default: <strategy>_<dataset>_<timestamp>)")
    
    args = parser.parse_args()
    
    # Set defaults from config
    model_name = args.model or config.GENERATION_MODEL
    api_base = args.api_base or config.GENERATION_API_BASE
    output_file = args.output or f"results_{args.strategy}_{args.dataset.lower()}.csv"
    run_name = args.run_name or f"{args.strategy}_{args.dataset.lower()}_{args.samples}samples"
    
    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check your .env file.")
        sys.exit(1)
    
    # Run evaluation
    run_evaluation(
        strategy_name=args.strategy,
        dataset_name=args.dataset,
        model_name=model_name,
        api_base=api_base,
        max_samples=args.samples,
        output_file=output_file,
        run_name=run_name
    )


if __name__ == "__main__":
    main()
