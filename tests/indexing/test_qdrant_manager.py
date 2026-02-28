import pytest
from unittest.mock import MagicMock, patch

from src.schema import UnifiedQASample
from src.indexing.qdrant_manager import QdrantManager

@patch('src.indexing.qdrant_manager.QdrantClient')
@patch('src.indexing.qdrant_manager.HuggingFaceEmbeddings')
def test_qdrant_manager_initialization(mock_embeddings_class, mock_client_class):
    # Setup Mocks
    mock_embeddings_instance = MagicMock()
    # Mocking embed_query to return a 3-dimensional vector for sizing
    mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embeddings_class.return_value = mock_embeddings_instance
    
    mock_client_instance = MagicMock()
    mock_client_instance.collection_exists.return_value = False
    mock_client_class.return_value = mock_client_instance
    
    # Init Manager
    manager = QdrantManager(embedding_model_name="dummy_model")
    
    # Asserts
    assert manager.vector_size == 3
    mock_client_instance.create_collection.assert_called_once()
    
    # Should create indexes for dataset, sample_id, and composite_id
    assert mock_client_instance.create_payload_index.call_count == 3
    
    # Check that all expected indexes are created
    index_calls = mock_client_instance.create_payload_index.call_args_list
    indexed_fields = [call.kwargs.get('field_name') or call[1].get('field_name') 
                      for call in index_calls]
    
    assert "dataset" in indexed_fields
    assert "sample_id" in indexed_fields
    assert "composite_id" in indexed_fields

@patch('src.indexing.qdrant_manager.uuid')
@patch('src.indexing.qdrant_manager.QdrantClient')
@patch('src.indexing.qdrant_manager.HuggingFaceEmbeddings')
def test_qdrant_manager_process_and_index(mock_embeddings_class, mock_client_class, mock_uuid):
    # Setup Mocks
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.embed_query.return_value = [0.1]
    # Mock embedding documents to return a list of vectors
    mock_embeddings_instance.embed_documents.return_value = [[0.1], [0.2]]
    mock_embeddings_class.return_value = mock_embeddings_instance
    
    mock_client_instance = MagicMock()
    mock_client_instance.collection_exists.return_value = True # skip creation
    mock_client_class.return_value = mock_client_instance
    
    mock_uuid.uuid4.return_value = "mocked-uuid"
    
    # Init Manager
    manager = QdrantManager(embedding_model_name="dummy_model")
    
    # Mock dataset sample
    sample = UnifiedQASample(
        sample_id="test_id_1",
        dataset_name="test_dataset",
        query="What is this?",
        ground_truth_answer="A test.",
        supporting_contexts=[{"title": "Doc1", "text": "This is chunk one of the document."}],
        corpus=[
            {"title": "Doc1", "text": "This is chunk one of the document."}
        ],
        metadata={}
    )
    
    # Run method
    manager.process_and_index_samples([sample])
    
    # Asserts
    # Should embed the extracted text
    mock_embeddings_instance.embed_documents.assert_called_once()
    args, _ = mock_embeddings_instance.embed_documents.call_args
    assert "This is chunk one of the document." in args[0][0]
    
    # Should upsert to Qdrant client with payload
    mock_client_instance.upsert.assert_called_once()
    args, kwargs = mock_client_instance.upsert.call_args
    
    points_upserted = kwargs["points"]
    assert len(points_upserted) == 1
    
    point = points_upserted[0]
    assert point.id == "mocked-uuid"
    # Crucial assertion: Did we correctly apply the Metadata filter tag?
    assert point.payload["dataset"] == "test_dataset"
    assert point.payload["sample_id"] == "test_id_1"
    assert point.payload["composite_id"] == "test_dataset_test_id_1"  # NEW: composite_id
    assert point.payload["text"] == "This is chunk one of the document."
