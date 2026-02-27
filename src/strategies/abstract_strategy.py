from typing import TypedDict, List, Dict, Any, Optional
from abc import ABC, abstractmethod

class RAGState(TypedDict, total=False):
    """
    Standardized state object for all RAG pipelines.
    Compatible with both LCEL chains and LangGraph.
    """
    query: str
    retrieved_contexts: List[str]  # Must be populated for Context Precision/Recall
    answer: str                    # Must be populated for Answer Relevancy/Faithfulness
    metadata: Dict[str, Any]       # Strategy-specific internal data


class AbstractRAGStrategy(ABC):
    """
    Abstract base class for all RAG evaluation strategies.
    Ensures a consistent interface for the RAGAS evaluation runner.
    """

    @abstractmethod
    def retrieve_and_generate(self, query: str, **kwargs) -> RAGState:
        """
        Executes the implemented RAG workflow (either via LCEL or LangGraph).
        
        Args:
            query (str): The input user question.
            **kwargs: Additional parameters (e.g., config for tools or vectorstore overrides).
            
        Returns:
            RAGState: Dict-like object containing `query`, `retrieved_contexts`, and `answer`.
        """
        pass
