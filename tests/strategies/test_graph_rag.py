import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.strategies.graph_rag.base import GraphRAG, EntityList

def test_graph_rag_execution():
    # 1. Mock Graph Retriever (Callable[[List[str]], List[Document]])
    mock_retriever = MagicMock()
    mock_retriever.return_value = [
        Document(page_content="Eiffel Tower is located in Paris, France.")
    ]
    
    # 2. Mock LLM
    mock_llm = MagicMock()
    
    # 2a. Mock Entity Extraction (with_structured_output)
    mock_structured_llm = MagicMock()
    # It should return the Pydantic model EntityList when called as a function (via LCEL wrap)
    mock_structured_llm.return_value = EntityList(entities=["Eiffel Tower", "Paris"])
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    # 2b. Mock Answer Generation (standard invoke)
    mock_llm.return_value = AIMessage(content="Based on the graph context, the Eiffel Tower is in Paris.")
    
    # 3. Instantiate GraphRAG
    rag = GraphRAG(graph_retriever=mock_retriever, llm=mock_llm)
    
    # 4. Execute standard interface
    query = "Where is the Eiffel Tower?"
    result = rag.retrieve_and_generate(query=query)
    
    # 5. Assertions
    # Verify Retriever was called with the extracted entities
    mock_retriever.assert_called_once_with(["Eiffel Tower", "Paris"])
    
    # Verify the final state matches RAGState expectations
    assert result["query"] == query
    assert len(result["retrieved_contexts"]) == 1
    assert result["retrieved_contexts"][0] == "Eiffel Tower is located in Paris, France."
    assert result["answer"] == "Based on the graph context, the Eiffel Tower is in Paris."
    
    assert "metadata" in result
    assert result["metadata"]["strategy"] == "GraphRAG"
    assert result["metadata"]["extracted_entities"] == ["Eiffel Tower", "Paris"]

def test_graph_rag_retrieval_failure_fallback():
    # Test that the pipeline doesn't crash if the graph retriever fails
    mock_retriever = MagicMock(side_effect=Exception("Database Connection Error"))
    
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.return_value = EntityList(entities=["Unknown Entity"])
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = AIMessage(content="I do not have enough context to answer that.")
    
    rag = GraphRAG(graph_retriever=mock_retriever, llm=mock_llm)
    
    query = "What is the secret code?"
    result = rag.retrieve_and_generate(query=query)
    
    assert len(result["retrieved_contexts"]) == 0
    assert result["answer"] == "I do not have enough context to answer that."
