import pytest
import uuid
import json
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.documents import Document

from src.indexing.qdrant_manager import QdrantManager
from src.indexing.neo4j_manager import Neo4jManager
from src.schema import UnifiedQASample

# We use a unique dataset name to avoid collision in Qdrant/Neo4j
# between different local test runs.
TEST_DATASET_NAME = f"integration_test_{uuid.uuid4().hex[:8]}"

@pytest.fixture(scope="session")
def qdrant_retriever():
    """
    Sets up a small text index in the real local Qdrant instance.
    """
    try:
        manager = QdrantManager(embedding_model_name="BAAI/bge-small-en-v1.5", host="localhost", port=6333)
    except Exception as e:
        pytest.skip(f"Could not connect to or initialize local Qdrant. Is the docker container running? Err: {e}")
        
    test_sample = UnifiedQASample(
        sample_id="test_qdrant_1",
        dataset_name=TEST_DATASET_NAME,
        query="Where is the Eiffel Tower integration test?",
        ground_truth_answer="Paris, France",
        supporting_contexts=[{"title": "Doc1", "text": "The Eiffel Tower is located in Paris, France. It is an integration testing marvel."}],
        corpus=[{"title": "Doc1", "text": "The Eiffel Tower is located in Paris, France. It is an integration testing marvel."}],
        metadata={}
    )
    
    manager.process_and_index_samples([test_sample])
    
    # Return a retriever configured for this test dataset
    retriever = manager.get_langchain_retriever(target_dataset=TEST_DATASET_NAME, k=1)
    return retriever


@pytest.fixture(scope="session")
def neo4j_graph_retriever():
    """
    Sets up a small graph in the real local Neo4j instance and returns a retrieval function.
    """
    try:
        manager = Neo4jManager(uri="bolt://localhost:7687", user="neo4j", password="password")
    except Exception as e:
        pytest.skip(f"Could not connect to local Neo4j. Is the docker container running? Err: {e}")

    test_sample = UnifiedQASample(
        sample_id="test_neo4j_1",
        dataset_name=TEST_DATASET_NAME,
        query="GraphRAG test query",
        ground_truth_answer="Paris answers",
        supporting_contexts=[],
        corpus=[],
        metadata={
            "evidences": [
                ["Eiffel Tower", "LOCATED_IN", "Paris"]
            ]
        }
    )
    
    manager.process_and_index_samples([test_sample])
    
    def retrieve_method(entities: list[str]) -> list[Document]:
        docs = []
        with manager.driver.session() as session:
            for entity in entities:
                # Basic cypher to retrieve neighbours 1 hop away
                query = f"""
                MATCH (start:Entity {{name: '{entity}', dataset: '{TEST_DATASET_NAME}'}})-[rel]->(end:Entity)
                RETURN start.name as start_name, type(rel) as rel_type, end.name as end_name
                """
                results = session.run(query)
                for record in results:
                    content = f"{record['start_name']} is {record['rel_type']} {record['end_name']}"
                    docs.append(Document(page_content=content))
        return docs

    yield retrieve_method
    manager.close()

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional
from pydantic import PrivateAttr

class CustomFakeLLM(BaseChatModel):
    """
    A custom fake LLM that properly implements bind_tools and with_structured_output 
    by returning a RunnableLambda that directly spits out the configured Pydantic model.
    """
    responses: List[Any]
    _response_idx: int = PrivateAttr(default=0)
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        if self._response_idx >= len(self.responses):
            current_response = self.responses[-1]  # Repeat last if exhausted
        else:
            current_response = self.responses[self._response_idx]
            self._response_idx += 1
            
        if isinstance(current_response, str):
            message = AIMessage(content=current_response)
        else:
            message = current_response
            
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    @property
    def _llm_type(self) -> str:
        return "custom_fake_llm"
    
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self
        
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        from langchain_core.runnables import RunnableLambda
        
        def mock_hardcoded_output(inputs: Any) -> Any:
            # Advance the response iterator
            res_content = self.invoke(inputs).content
            
            # Since the LLM returns JSON strings strings, we convert it to dict, then feed to Pydantic
            try:
                data = json.loads(res_content)
                return schema(**data)
            except Exception as e:
                print(f"FAILED TO PARSE STRUCTURED OUTPUT: {e}")
                raise e
            
        return RunnableLambda(mock_hardcoded_output)

@pytest.fixture
def fake_llm():
    """
    Provides a deterministic Custom Fake LLM so tests don't incur cost / latency.
    """
    return CustomFakeLLM(
        responses=[
            '{"sub_queries": [{"query": "Where is the Eiffel Tower?"}]}',
            "This is the deterministic final answer from Fake LLM based on context.",
            '{"entities": ["Eiffel Tower"]}',
            "This is the deterministic final answer from Fake LLM based on context."
        ]
    )
