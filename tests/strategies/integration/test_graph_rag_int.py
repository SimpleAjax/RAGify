import pytest
from src.strategies.graph_rag.base import GraphRAG
from tests.strategies.integration.conftest import CustomFakeLLM

def test_graph_rag_integration(neo4j_graph_retriever):
    """
    Tests the GraphRAG strategy by hitting the real Neo4j local instance test graph.
    """
    fake_llm = CustomFakeLLM(
        responses=[
            '{"entities": ["Eiffel Tower"]}',
            "This is the deterministic final answer from Fake LLM based on context."
        ]
    )
    
    strategy = GraphRAG(graph_retriever=neo4j_graph_retriever, llm=fake_llm)
    
    query = "Tell me about the Eiffel Tower."
    
    result = strategy.retrieve_and_generate(query=query)
    
    assert result["query"] == query
    assert result["metadata"]["strategy"] == "GraphRAG"
    assert "Eiffel Tower" in result["metadata"].get("extracted_entities", [])
    
    # Ensure Neo4j actually returned the relationship we indexed in the fixture
    contexts = result["retrieved_contexts"]
    assert len(contexts) >= 1
    assert "Eiffel Tower is LOCATED_IN Paris" in contexts[0]
    
    assert result["answer"] == "This is the deterministic final answer from Fake LLM based on context."
