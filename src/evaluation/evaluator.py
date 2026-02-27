import os
import pandas as pd
from typing import List, Dict, Any, Optional

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.chat_models import ChatLiteLLM
from langchain_huggingface import HuggingFaceEmbeddings

class RagasEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini", api_base: Optional[str] = None):
        """
        Initializes the Ragas Evaluator with a LiteLLM backend.
        This provides the abstraction layer requested in Phase 4 Option C.
        It allows seamless switching between OpenAI, Ollama, Anthropic, etc.
        """
        # Ensure we don't fail immediately if keys are missing during test instantiation
        api_key = os.environ.get("OPENAI_API_KEY", "dummy-key-for-tests")
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
            
        # We explicitly set up LiteLLM wrapped for LangChain and Ragas
        try:
            self.llm = ChatLiteLLM(model=model_name, api_base=api_base)
            self.ragas_llm = LangchainLLMWrapper(self.llm)
        except Exception as e:
            # Fallback for old litellm/langchain mappings if any
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name, base_url=api_base)
            self.ragas_llm = LangchainLLMWrapper(self.llm)
        
        # Use a small local embeddings model for evaluating contexts
        hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.ragas_embeds = LangchainEmbeddingsWrapper(hf_embeddings)

        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]

    def evaluate_strategy(self, eval_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluates a list of samples.
        Required keys in each dictionary:
        - question (str)
        - answer (str): generated string
        - contexts (List[str]): retrieved contexts
        - ground_truth (str)
        """
        # Validate data
        for d in eval_data:
            if not all(k in d for k in ["question", "answer", "contexts", "ground_truth"]):
                raise ValueError("eval_data missing required keys for RAGAS.")
                
        # RAGAS requires a HuggingFace Dataset format
        dataset = Dataset.from_list(eval_data)

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeds,
            raise_exceptions=False
        )

        return result.to_pandas()
