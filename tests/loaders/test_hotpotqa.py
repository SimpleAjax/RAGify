import pytest
from unittest.mock import patch, MagicMock
from src.schema import UnifiedQASample
from src.loaders.hotpotqa import HotpotQALoader

@pytest.fixture
def mock_hotpotqa_dataset():
    return [
        {
            "id": "123",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "type": "bridge",
            "level": "hard",
            "context": {
                "title": ["France", "Paris"],
                "sentences": [
                    ["France is a country in Europe.", "It is known for its culture."],
                    ["Paris is the capital of France.", "It has the Eiffel Tower."]
                ]
            },
            "supporting_facts": {
                "title": ["Paris"],
                "sent_id": [0]
            }
        }
    ]

@patch("src.loaders.hotpotqa.load_dataset")
def test_hotpotqa_loader(mock_load_dataset, mock_hotpotqa_dataset):
    mock_load_dataset.return_value = mock_hotpotqa_dataset
    
    loader = HotpotQALoader()
    samples = list(loader.load())
    
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, UnifiedQASample)
    assert sample.dataset_name == "HotpotQA"
    assert sample.sample_id == "123"
    assert sample.query == "What is the capital of France?"
    assert sample.ground_truth_answer == "Paris"
    
    assert len(sample.corpus) == 2
    assert sample.corpus[0]["title"] == "France"
    assert "France is a country" in sample.corpus[0]["text"]
    
    assert len(sample.supporting_contexts) == 1
    assert sample.supporting_contexts[0]["title"] == "Paris"
    assert "Paris is the capital" in sample.supporting_contexts[0]["text"]
    assert sample.metadata["type"] == "bridge"
