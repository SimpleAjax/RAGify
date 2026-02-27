import argparse
import json
import sys
import os

# Add the project root to the python path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import RagasEvaluator
from src.evaluation.tracker import ExperimentTracker

def main():
    parser = argparse.ArgumentParser(description="Run Ragas Evaluation using LiteLLM Wrapper")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LiteLLM model name (e.g., 'ollama/llama3', 'gpt-4o-mini', 'openrouter/openai/gpt-4o')")
    parser.add_argument("--api-base", type=str, default=None, help="Base URL for the model API (e.g., 'https://openrouter.ai/api/v1', 'http://localhost:11434' for Ollama)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (if not set, uses OPENAI_API_KEY or OPENROUTER_API_KEY env var)")
    parser.add_argument("--input", type=str, help="Path to JSON file containing array of samples (requires keys: question, answer, contexts, ground_truth).")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Path to save the output CSV")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for the MLflow run")
    
    args = parser.parse_args()

    print(f"Initializing Ragas Evaluator with model: {args.model}")
    if args.api_base:
        print(f"Using API Base: {args.api_base}")
    if args.api_key:
        print("Using provided API key")
        
    evaluator = RagasEvaluator(model_name=args.model, api_base=args.api_base, api_key=args.api_key)

    if args.input:
        print(f"Loading data from {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    else:
        print("\nNo input file provided. Running a built-in sample dummy evaluation.")
        eval_data = [
            {
                "question": "What is the capital of France?",
                "answer": "France's capital city is Paris.",
                "contexts": ["Paris is the capital and most populous city of France."],
                "ground_truth": "Paris"
            },
            {
                "question": "How many states are in the USA?",
                "answer": "There are 50 states in the United States of America.",
                "contexts": ["The United States of America is a country primarily located in North America, consisting of 50 states."],
                "ground_truth": "50 states"
            }
        ]

    print(f"Evaluating {len(eval_data)} sample(s)...")
    
    # Initialize Tracker
    tracker = ExperimentTracker()
    
    try:
        with tracker.start_run(run_name=args.run_name):
            # Log initial parameters
            tracker.log_parameters({
                "model_name": args.model,
                "api_base": args.api_base,
                "sample_count": len(eval_data),
                "input_file": args.input if args.input else "built-in_dummy"
            })
            
            results_df = evaluator.evaluate_strategy(eval_data)
            
            print("\n=== Evaluation Complete ===")
            print("Metric Averages:")
            
            # Print to stdout
            averages = results_df.select_dtypes(include='number').mean()
            for metric, score in averages.items():
                print(f"- {metric}: {score:.4f}")
                
            # Log to MLflow
            tracker.log_metrics(results_df)
            tracker.log_evaluation_artifact(results_df, filename=os.path.basename(args.output))
                
            results_df.to_csv(args.output, index=False)
            print(f"\nDetailed results saved to {args.output}")
            print("Run `mlflow ui` to view the experiment dashboard.")
        
        
    except Exception as e:
        print(f"Evaluation failed. Please check your model endpoint and API keys. Error: {e}")

if __name__ == "__main__":
    main()
