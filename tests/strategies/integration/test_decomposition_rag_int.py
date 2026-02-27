import pytest
from src.strategies.decomposition.base import DecompositionRAG

def test_decomposition_rag_integration(qdrant_retriever, fake_llm):
    """
    Tests the DecompositionRAG Map-Reduce strategy interacting with real Qdrant backend.
    """
    strategy = DecompositionRAG(retriever=qdrant_retriever, llm=fake_llm)
    
    query = "Where is the Eiffel Tower and what is it?"
    
    result = strategy.retrieve_and_generate(query=query)
    
    assert result["query"] == query
    assert result["metadata"]["strategy"] == "DecompositionRAG"
    
    # Check that Qdrant actually returned the chunk for the decomposed sub-queries
    contexts = result["retrieved_contexts"]
    assert len(contexts) >= 1
    assert any("marvel" in ctx.lower() for ctx in contexts)
    
    # Check that the Synthesis node produced the final answer
    assert result["answer"] == "This is the deterministic final answer from Fake LLM based on context."
    assert "sub_queries_generated" in result["metadata"]
