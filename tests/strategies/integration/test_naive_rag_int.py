import pytest
from src.strategies.naive.base import NaiveRAG

def test_naive_rag_integration(qdrant_retriever, fake_llm):
    """
    Tests the NaiveRAG strategy end-to-end against the real Qdrant backend using the Fake LLM.
    """
    # Re-instantiate the fake LLM to guarantee predictable responses for this specific test
    fake_llm.responses = ["This is the deterministic final answer from Fake LLM based on context."]
    
    strategy = NaiveRAG(retriever=qdrant_retriever, llm=fake_llm)
    
    # We query for something that should match the indexed text in Qdrant
    query = "Where is the Eiffel Tower integration test?"
    
    result = strategy.retrieve_and_generate(query=query)
    
    assert result["query"] == query
    assert result["metadata"]["strategy"] == "NaiveRAG"
    
    # Check that Qdrant actually returned the chunk we embedded
    contexts = result["retrieved_contexts"]
    assert len(contexts) >= 1
    assert "marvel" in contexts[0].lower() # From "testing marvel."
    
    # Check that the LLM chain produced the final answer
    assert result["answer"] == "This is the deterministic final answer from Fake LLM based on context."
