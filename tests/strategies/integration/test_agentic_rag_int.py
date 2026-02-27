import pytest
from src.strategies.agentic.base import AgenticRAG
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, ToolMessage
from tests.strategies.integration.conftest import CustomFakeLLM

def test_agentic_rag_integration(qdrant_retriever):
    """
    Tests Agentic RAG end-to-end, passing the real Qdrant retriever as a tool.
    Because the agent loop depends on the LLM's dynamic decisions, we mock the exact 
    ReAct response trajectory from the fake LLM to force it to use the tool.
    (Testing real agent intelligence against real DBs requires an expensive model like GPT-4).
    """
    
    # 1. Turn the Qdrant backend into an Agent Tool
    qdrant_tool = create_retriever_tool(
        retriever=qdrant_retriever,
        name="qdrant_integration_db",
        description="Search for context about integration tests."
    )
    
    # 2. Force the Fake LLM to make a tool call, then give the final answer.
    fake_llm = CustomFakeLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "qdrant_integration_db", "args": {"query": "Eiffel Tower"}, "id": "call_integration"}]
            ),
            AIMessage(content="This is the deterministic final answer from Fake LLM based on context.")
        ]
    )
    
    strategy = AgenticRAG(tools=[qdrant_tool], llm=fake_llm)
    
    query = "Where is the Eiffel Tower integration test?"
    
    # 3. Execute
    result = strategy.retrieve_and_generate(query=query)
    
    assert result["query"] == query
    assert result["metadata"]["strategy"] == "AgenticRAG"
    
    # 4. Assert Qdrant was successfully queried by the Agent tool execution
    # `retrieved_contexts` will contain the stringified output of the tool.
    contexts = result["retrieved_contexts"]
    assert len(contexts) >= 1
    assert "marvel" in contexts[0].lower()
    
    # 5. Assert Final Output
    assert result["answer"] == "This is the deterministic final answer from Fake LLM based on context."
    assert len(result["metadata"]["tool_calls"]) == 1
    assert result["metadata"]["tool_calls"][0]["name"] == "qdrant_integration_db"
