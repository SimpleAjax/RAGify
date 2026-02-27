import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from src.strategies.decomposition.base import DecompositionRAG, SubQueriesList, SubQuery

def test_decomposition_rag_execution():
    # 1. Mock Retriever
    mock_retriever = MagicMock()
    # Give different context based on sub-query
    def return_docs(query):
        if "Paris" in query:
            return [Document(page_content="Eiffel Tower is in Paris.")]
        elif "tall" in query:
            return [Document(page_content="Eiffel Tower is 330 meters tall.")]
        return []
        
    mock_retriever.side_effect = return_docs
    
    # 2. Mock LLM
    mock_llm = MagicMock()
    
    # 2a. Mock Structured Output for Decomposition
    mock_structured_llm = MagicMock()
    # LCEL wraps non-runnables so use return_value
    mock_structured_llm.return_value = SubQueriesList(
        sub_queries=[
            SubQuery(query="Where is the Eiffel Tower located in Paris?"),
            SubQuery(query="How tall is the Eiffel Tower?")
        ]
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    # 2b. Mock Standard Output for Synthesis
    mock_llm.invoke.return_value = AIMessage(content="The Eiffel Tower is in Paris and is 330 meters tall.")
    # For LCEL chains without RunnableLambda
    mock_llm.return_value = AIMessage(content="The Eiffel Tower is in Paris and is 330 meters tall.")
    
    # 3. Instantiate Strategy
    rag = DecompositionRAG(retriever=mock_retriever, llm=mock_llm)
    
    # 4. Execute standard interface
    query = "Where is the Eiffel Tower and how tall is it?"
    result = rag.retrieve_and_generate(query=query)
    
    # 5. Assertions
    # Final state shape checks
    assert result["query"] == query
    assert result["answer"] == "The Eiffel Tower is in Paris and is 330 meters tall."
    
    # Context aggregation checks (De-duplicated array from 2 parallel nodes)
    assert len(result["retrieved_contexts"]) == 2
    assert "Eiffel Tower is in Paris." in result["retrieved_contexts"]
    assert "Eiffel Tower is 330 meters tall." in result["retrieved_contexts"]
    
    # Metadata checks
    assert result["metadata"]["strategy"] == "DecompositionRAG"
    assert len(result["metadata"]["sub_queries_generated"]) == 2
    assert "How tall is the Eiffel Tower?" in result["metadata"]["sub_queries_generated"]
