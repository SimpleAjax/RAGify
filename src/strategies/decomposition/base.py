import operator
from typing import Annotated, Any, Dict, List, Callable
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.strategies.abstract_strategy import AbstractRAGStrategy, RAGState

# --- State Definitions ---

class SubQuery(BaseModel):
    query: str = Field(description="A distinct, individual sub-question that needs answering to help solve the overall query.")

class SubQueriesList(BaseModel):
    sub_queries: List[SubQuery] = Field(description="List of required sub-queries.")

class SubQueryState(RAGState):
    """
    Extended state for Query Decomposition.
    Uses `operator.add` for `retrieved_contexts` so parallel nodes can append to the list.
    """
    sub_queries: List[str]
    # We redefine retrieved_contexts to explicitly use the `add` reducer for Map-Reduce fan-in
    retrieved_contexts: Annotated[List[str], operator.add] 

# State passed to the individual worker nodes during the `Send` mapping
class ParallelRetrieveState(BaseModel):
    sub_query: str


class DecompositionRAG(AbstractRAGStrategy):
    """
    Query Decomposition strategy using LangGraph Map-Reduce architecture.
    Flow: 
      1. Decompose Node: LLM breaks down the main query into sub-queries.
      2. Parallel Retrieve Nodes (Send API): Retrieve and answer Sub-Queries concurrently.
      3. Synthesize Node: Combine all context into final RAGState answer.
    """
    
    def __init__(self, retriever: Callable[[str], List[Document]], llm: BaseChatModel):
        """
        Initializes the DecompositionRAG strategy.
        
        Args:
            retriever (Callable[[str], List[Document]]): Function that takes a query string 
                and returns relevant Document objects.
            llm (BaseChatModel): Chat model used for decomposition and synthesis.
        """
        self.retriever = retriever
        self.llm = llm
        
        # 1. Decompose Prompt
        self.decompose_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at breaking down complex, multi-hop questions into simpler sub-questions. Break down the user's question into 2-4 sub-questions necessary to fully answer it."),
            ("user", "{query}")
        ])
        
        # 2. Synthesis Prompt
        self.synthesize_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent assistant. Synthesize a comprehensive final answer to the user's original query based strictly on the aggregated context retrieved from sub-queries.\n\nAggregated Context:\n{context}"),
            ("user", "{query}")
        ])

        # Compile the state graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Builds and compiles the `Map-Reduce` LangGraph Workflow."""
        builder = StateGraph(SubQueryState)
        
        # --- Node Definitions ---
        
        def decompose_node(state: SubQueryState):
            """Uses LLM with structured output to break down the query."""
            chain = self.decompose_prompt | self.llm.with_structured_output(SubQueriesList)
            result = chain.invoke({"query": state["query"]})
            
            # Extract plain strings from the Pydantic models
            sub_query_strings = [sq.query for sq in result.sub_queries]
            
            return {"sub_queries": sub_query_strings}

        def retrieve_worker(state: ParallelRetrieveState):
            """Worker node executed in parallel for each sub-query."""
            sub_query = state.sub_query
            
            try:
                # If it's a LangChain retriever, use .invoke(), otherwise if callable fallback
                if hasattr(self.retriever, "invoke"):
                    docs = self.retriever.invoke(sub_query)
                else:
                    docs = self.retriever(sub_query)
                texts = [d.page_content for d in docs]
            except Exception:
                texts = []
                
            # The reducer (operator.add) in SubQueryState will aggregate these lists
            return {"retrieved_contexts": texts}

        def synthesize_node(state: SubQueryState):
            """Combines all parallel contexts to form the final answer."""
            contexts = state.get("retrieved_contexts", [])
            
            # De-duplicate contexts in case multiple sub-queries retrieved the same chunk
            unique_contexts = list(dict.fromkeys(contexts))
            
            formatted_context = "\n\n".join(unique_contexts)
            
            chain = self.synthesize_prompt | self.llm
            result = chain.invoke({
                "query": state["query"],
                "context": formatted_context
            })
            
            return {
                "answer": str(result.content),
                # Overwrite retrieved_contexts to ensure we pass exactly the de-duped unique list forward
                # Note: Given `operator.add` is on the schema, simply returning a dict here appended by default in LangGraph.
                # To clean it up perfectly for RAGAS eval, we rely on the retrieve_and_generate wrapper mapping instead.
            }

        # --- Edge Definitions ---
        
        # Map out to workers
        def continue_to_retrieve(state: SubQueryState):
            """LangGraph `Send` edge logic mapping sub-queries to parallel retrieve nodes."""
            return [Send("retrieve_worker", ParallelRetrieveState(sub_query=sq)) for sq in state["sub_queries"]]
            
        builder.add_node("decompose", decompose_node)
        builder.add_node("retrieve_worker", retrieve_worker)
        builder.add_node("synthesize", synthesize_node)
        
        builder.add_edge(START, "decompose")
        
        # Map over Sub-Queries
        builder.add_conditional_edges("decompose", continue_to_retrieve, ["retrieve_worker"])
        
        # Reduce: Once all retrieve_workers finish, they transition to synthesize
        builder.add_edge("retrieve_worker", "synthesize")
        
        builder.add_edge("synthesize", END)
        
        return builder.compile()

    def retrieve_and_generate(self, query: str, **kwargs) -> RAGState:
        """
        Executes the LangGraph Map-Reduce Pipeline for query decomposition.
        
        Args:
            query (str): The input user question.
            
        Returns:
            RAGState: Dict containing `query`, `retrieved_contexts`, `answer`, and `metadata`.
        """
        # Execute the Graph
        initial_state = {"query": query, "retrieved_contexts": [], "sub_queries": []}
        final_state = self.graph.invoke(initial_state)
        
        # De-duplicate context list for the final output
        unique_contexts = list(dict.fromkeys(final_state.get("retrieved_contexts", [])))
        
        return {
            "query": final_state["query"],
            "retrieved_contexts": unique_contexts,
            "answer": final_state.get("answer", ""),
            "metadata": {
                "strategy": "DecompositionRAG",
                "sub_queries_generated": final_state.get("sub_queries", [])
            }
        }
