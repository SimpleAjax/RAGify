from typing import Any, Dict, List

from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from src.strategies.abstract_strategy import AbstractRAGStrategy, RAGState

class NaiveRAG(AbstractRAGStrategy):
    """
    Standard/Naive RAG strategy using a linear LCEL pipeline.
    """
    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel):
        """
        Initializes the Naive RAG strategy with required LangChain components.
        
        Args:
            retriever (VectorStoreRetriever): Configured retriever (e.g., Qdrant).
            llm (BaseChatModel): Chat model used for answer generation.
        """
        self.retriever = retriever
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent assistant. Answer the user's question based strictly on the provided context.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        
        self.chain = self._build_chain()
        
    def _format_docs(self, docs: List[Any]) -> str:
        """Formats LangChain Document objects into a single string."""
        return "\n\n".join([d.page_content for d in docs])

    def _build_chain(self) -> RunnableSequence:
        """
        Builds the underlying LCEL chain to retrieve contexts and generate the answer.
        Input dictionary matches `{"query": "user question"}`.
        """
        
        # Step 1: Execute retrieval
        retrieve_step = {
            "retrieved_docs": lambda x: self.retriever.invoke(x["query"]),
            "query": lambda x: x["query"]
        }
        
        # Step 2: Format the contexts and pass through required variables
        def format_step(inputs: dict) -> dict:
            docs = inputs["retrieved_docs"]
            return {
                "context": self._format_docs(docs),
                "query": inputs["query"],
                "retrieved_contexts": [d.page_content for d in docs]
            }

        # Step 3: Run the prompt -> LLM and merge with pass-through values
        def generate_step(inputs: dict) -> dict:
            # We explicitly execute prompt | llm | string output parser inside the nested dictionary structure
            chain = self.prompt | self.llm 
            ai_msg = chain.invoke(inputs)
            
            return {
                "answer": ai_msg.content,
                "retrieved_contexts": inputs["retrieved_contexts"],
                "query": inputs["query"]
            }
            
        return retrieve_step | RunnableLambda(format_step) | RunnableLambda(generate_step)

    def retrieve_and_generate(self, query: str, **kwargs) -> RAGState:
        """
        Executes the Naive RAG LCEL chain and returns a standardized state.
        
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
            "metadata": {"strategy": "NaiveRAG"}
        }
