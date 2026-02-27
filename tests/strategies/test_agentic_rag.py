import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool

from src.strategies.agentic.base import AgenticRAG

@tool
def dummy_search(query: str) -> str:
    """A dummy search tool for testing."""
    return "Dummy context about " + query


def test_agentic_rag_initialization():
    mock_llm = MagicMock()
    
    agent = AgenticRAG(tools=[dummy_search], llm=mock_llm)
    assert len(agent.tools) == 1
    assert agent.tools[0] == dummy_search
    assert agent.llm == mock_llm


@patch("src.strategies.agentic.base.create_react_agent")
def test_agentic_rag_execution(mock_create_react_agent):
    # 1. Setup mocks
    mock_llm = MagicMock()
    
    # Mock the returned agent executor 
    mock_agent_executor = MagicMock()
    mock_create_react_agent.return_value = mock_agent_executor
    
    # 2. Mock a simulated trajectory from LangGraph
    # Human asks question -> AI decides to use tool -> Tool returns context -> AI answers
    simulated_messages = [
        HumanMessage(content="What is testing?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "dummy_search", "args": {"query": "testing"}, "id": "call_123"}]
        ),
        ToolMessage(
            content="Testing is a method to verify software.",
            tool_call_id="call_123",
            name="dummy_search"
        ),
        AIMessage(content="Testing is a method to verify software.")
    ]
    
    mock_agent_executor.invoke.return_value = {"messages": simulated_messages}
    
    # 3. Instantiate AgenticRAG
    agent = AgenticRAG(tools=[dummy_search], llm=mock_llm)
    
    # 4. Execute interface
    query = "What is testing?"
    result = agent.retrieve_and_generate(query=query)
    
    # 5. Assertions
    # Verify the inner agent was invoked with the right message shape
    mock_agent_executor.invoke.assert_called_once_with({"messages": [("user", query)]})
    
    assert result["query"] == query
    assert result["answer"] == "Testing is a method to verify software."
    
    # The tool output should be extracted as the retrieved context
    assert len(result["retrieved_contexts"]) == 1
    assert result["retrieved_contexts"][0] == "Testing is a method to verify software."
    
    # Check Metadata
    assert result["metadata"]["strategy"] == "AgenticRAG"
    assert result["metadata"]["trajectory_length"] == 4
    assert len(result["metadata"]["tool_calls"]) == 1
