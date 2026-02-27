from typing import Any, Dict, List
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent

from src.strategies.abstract_strategy import AbstractRAGStrategy, RAGState

class AgenticRAG(AbstractRAGStrategy):
    """
    Agentic Retrieval strategy using a LangGraph ReAct loop.
    Flow Pipeline: Query -> ReAct Agent <--> Tools -> End
    The Agent autonomously decides when to search, gather context, and answer.
    """
    def __init__(self, tools: List[BaseTool], llm: BaseChatModel, system_prompt: str = None):
        """
        Initializes the Agentic RAG strategy.
        
        Args:
            tools (List[BaseTool]): A list of LangChain tools the agent can use to search for context 
                (e.g., VectorStore search tool, Web search tool, Graph traversal tool).
            llm (BaseChatModel): Chat model capable of function calling/tool usage.
            system_prompt (str, optional): Custom instructions for the ReAct agent.
        """
        self.tools = tools
        self.llm = llm
        
        self.system_prompt = system_prompt or (
            "You are a helpful research assistant. "
            "Use the provided tools to search for context and answer the user's question. "
            "If you do not know the answer, use a tool to find it. "
            "Base your final answer STRICTLY on the facts discovered using tools, do not hallucinate."
        )
        
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt
        )

    def retrieve_and_generate(self, query: str, **kwargs) -> RAGState:
        """
        Executes the LangGraph ReAct agent and normalizes the output to RAGState format.
        
        Args:
            query (str): The input user question.
            
        Returns:
            RAGState: Dict containing `query`, `retrieved_contexts`, `answer`, and `metadata`.
        """
        
        # Invoke the LangGraph agent
        # The prebuilt ReAct agent expects {"messages": [("user", query)]} as standard input
        result = self.agent_executor.invoke({"messages": [("user", query)]})
        
        # Extract the sequence of messages
        messages = result.get("messages", [])
        
        # The final AI message containing the final answer is usually the last message at the end of the loop
        final_answer = ""
        if messages:
            final_answer = messages[-1].content
            
        # We need to extract all context the agent actually retrieved using its tools 
        # to properly populate `retrieved_contexts` for the RAGAS Context Recall/Precision metrics.
        # LangGraph ReAct agents store tool outputs as ToolMessages
        retrieved_contexts = []
        for msg in messages:
            if msg.type == "tool":
                # Assuming the tool returns string context or JSON-serializable context
                retrieved_contexts.append(str(msg.content))
                
        # Optional: Extract the exact tool calls made for transparency
        tool_calls_made = []
        for msg in messages:
            if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made.extend(msg.tool_calls)

        return {
            "query": query,
            "retrieved_contexts": retrieved_contexts,
            "answer": final_answer,
            "metadata": {
                "strategy": "AgenticRAG",
                "trajectory_length": len(messages),
                "tool_calls": tool_calls_made
            }
        }
