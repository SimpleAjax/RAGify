import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.strategies.naive.base import NaiveRAG

def test_naive_rag_execution():
    # 1. Mock Retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="The Eiffel Tower is located in Paris, France."),
        Document(page_content="It was completed in 1889.")
    ]
    
    # 2. Mock LLM
    mock_llm = FakeListChatModel(responses=["Based on the context, the Eiffel Tower is in Paris, France."])
    
    # 3. Instantiate Naive RAG
    rag = NaiveRAG(retriever=mock_retriever, llm=mock_llm)
    
    # 4. Execute standard interface
    query = "Where is the Eiffel Tower?"
    result = rag.retrieve_and_generate(query=query)
    
    # 5. Assertions
    # Verify Retriever was called with the query
    mock_retriever.invoke.assert_called_once_with(query)
    
    # Verify the final state matches RAGState expectations
    assert result["query"] == query
    assert len(result["retrieved_contexts"]) == 2
    assert result["retrieved_contexts"][0] == "The Eiffel Tower is located in Paris, France."
    assert result["answer"] == "Based on the context, the Eiffel Tower is in Paris, France."
    assert result["metadata"] == {"strategy": "NaiveRAG"}
