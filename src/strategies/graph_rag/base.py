from typing import Any, Dict, List, Callable

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from src.strategies.abstract_strategy import AbstractRAGStrategy, RAGState

class EntityList(BaseModel):
    """Schema for extracting entities from a query."""
    entities: List[str] = Field(description="A list of key entities extracted from the user's query.")

class GraphRAG(AbstractRAGStrategy):
    """
    GraphRAG strategy using a linear LCEL pipeline.
    Flow Pipeline: Query -> Entity Extraction -> Graph Retrieval -> Answer Generation -> End
    """
    def __init__(self, graph_retriever: Callable[[List[str]], List[Document]], llm: BaseChatModel):
        """
        Initializes the GraphRAG strategy.
        
        Args:
            graph_retriever (Callable[[List[str]], List[Document]]): Function or Retriever that 
                takes a list of entity strings and returns relevant Document objects from a Graph DB.
            llm (BaseChatModel): Chat model with structured output capabilities for extraction 
                and standard output for generation.
        """
        self.graph_retriever = graph_retriever
        self.llm = llm
        
        # Setup Prompts
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert entity extractor. Extract the most important named entities from the given user query. Return them as a structured list."),
            ("user", "{query}")
        ])
        
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent assistant. Answer the user's question based strictly on the provided knowledge graph context.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        
        # The underlying LCEL chain
        self.chain = self._build_chain()
        
    def _format_docs(self, docs: List[Document]) -> str:
        """Formats LangChain Document objects into a single string."""
        if not docs:
            return "No relevant graph context found."
        return "\n\n".join([d.page_content for d in docs])

    def _build_chain(self) -> RunnableSequence:
        """
        Builds the underlying LCEL chain for GraphRAG.
        """
        # 1. Entity Extraction Step
        extraction_chain = self.extraction_prompt | self.llm.with_structured_output(EntityList)
        
        def extract_entities(inputs: dict) -> dict:
            extracted = extraction_chain.invoke(inputs)
            # Add extracted entities to the dictionary sequence
            return {
                "query": inputs["query"],
                "extracted_entities": extracted.entities
            }
            
        # 2. Graph Retrieval Step
        def retrieve_graph(inputs: dict) -> dict:
            try:
                # Retrieve documents based on entities
                docs = self.graph_retriever(inputs["extracted_entities"])
            except Exception as e:
                # Fallback if retrieval fails
                docs = []
            
            return {
                "query": inputs["query"],
                "extracted_entities": inputs["extracted_entities"],
                "retrieved_docs": docs
            }
            
        # 3. Context Formatting Step
        def format_context(inputs: dict) -> dict:
            docs = inputs.get("retrieved_docs", [])
            return {
                "query": inputs["query"],
                "extracted_entities": inputs["extracted_entities"],
                "context": self._format_docs(docs),
                "retrieved_contexts": [d.page_content for d in docs]
            }

        # 4. Generate Answer Step
        def generate_answer(inputs: dict) -> dict:
            chain = self.generation_prompt | self.llm
            result = chain.invoke(inputs)
            
            return {
                "query": inputs["query"],
                "extracted_entities": inputs["extracted_entities"],
                "retrieved_contexts": inputs["retrieved_contexts"],
                "answer": str(result.content)
            }
            
        return (
            RunnableLambda(extract_entities) 
            | RunnableLambda(retrieve_graph) 
            | RunnableLambda(format_context) 
            | RunnableLambda(generate_answer)
        )

    def retrieve_and_generate(self, query: str, **kwargs) -> RAGState:
        """
        Executes the GraphRAG LCEL chain and returns a standardized state.
        
        Args:
            query (str): The input user question.
            
        Returns:
            RAGState: Dict containing `query`, `retrieved_contexts`, `answer`, and `metadata`.
        """
        response = self.chain.invoke({"query": query})
        
        return {
            "query": response["query"],
            "retrieved_contexts": response["retrieved_contexts"],
            "answer": response["answer"],
            "metadata": {
                "strategy": "GraphRAG",
                "extracted_entities": response["extracted_entities"]
            }
        }
