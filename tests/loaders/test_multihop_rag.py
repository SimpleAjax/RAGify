import pytest
from unittest.mock import patch
from src.schema import UnifiedQASample
from src.loaders.multihop_rag import MultiHopRAGLoader

@pytest.fixture
def mock_multihop_dataset():
    return [
        {
            "query": "Who is the CEO of the company that makes the iPhone?",
            "answer": "Tim Cook",
            "evidence_list": [
                {
                    "title": "Apple Inc.",
                    "fact": "Apple Inc. makes the iPhone."
                },
                {
                    "title": "Tim Cook",
                    "fact": "Tim Cook is the CEO of Apple Inc."
                }
            ],
            "question_type": "bridge"
        }
    ]

@patch("src.loaders.multihop_rag.load_dataset")
def test_multihop_rag_loader(mock_load_dataset, mock_multihop_dataset):
    mock_load_dataset.return_value = mock_multihop_dataset
    
    loader = MultiHopRAGLoader()
    samples = list(loader.load())
    
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, UnifiedQASample)
    assert sample.dataset_name == "MultiHop-RAG"
    # The loader prefixes the query to 30 chars for sample_id if ID is missing
    assert sample.sample_id == "Who is the CEO of the company "[:30]
    
    assert len(sample.corpus) == 2
    assert sample.corpus[0]["title"] == "Apple Inc."
    
    assert len(sample.supporting_contexts) == 2
    assert sample.supporting_contexts[1]["title"] == "Tim Cook"
    assert sample.metadata["question_type"] == "bridge"
