import pytest
from src.strategies.abstract_strategy import AbstractRAGStrategy, RAGState

class MockStrategy(AbstractRAGStrategy):
    """A concrete implementation of AbstractRAGStrategy for testing."""
    def retrieve_and_generate(self, query: str, **kwargs) -> RAGState:
        return {
            "query": query,
            "retrieved_contexts": ["Mock context 1", "Mock context 2"],
            "answer": "Mock answer based on context.",
            "metadata": {"kwargs": kwargs}
        }

def test_abstract_strategy_interface():
    # Verify the concrete class handles the interface properly
    strategy = MockStrategy()
    result = strategy.retrieve_and_generate(query="What is testing?", mock_param=True)
    
    assert isinstance(result, dict)
    assert result["query"] == "What is testing?"
    assert len(result["retrieved_contexts"]) == 2
    assert result["answer"] == "Mock answer based on context."
    assert result["metadata"]["kwargs"]["mock_param"] is True

def test_abstract_strategy_enforces_implementation():
    # Verify that a class failing to implement abstract method raises TypeError
    class IncompleteStrategy(AbstractRAGStrategy):
        pass

    with pytest.raises(TypeError):
        _ = IncompleteStrategy()
