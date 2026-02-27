import pytest
import os
import json
from unittest.mock import patch
from src.schema import UnifiedQASample
from src.loaders.twowiki import TwoWikiMultiHopQALoader

@pytest.fixture
def mock_twowiki_dataset(tmpdir):
    data = [
        {
            "_id": "456",
            "question": "Who directed Inception?",
            "answer": "Christopher Nolan",
            "type": "compositional",
            "evidences": [
                ["Inception", "director", "Christopher Nolan"]
            ],
            "context": [
                ["Inception", ["Inception is a 2010 movie.", "It was directed by Christopher Nolan."]],
                ["Christopher Nolan", ["Christopher Nolan is a director."]]
            ],
            "supporting_facts": [
                ["Inception", 1]
            ]
        }
    ]
    file_path = os.path.join(tmpdir, "dev.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path

def test_twowiki_loader(mock_twowiki_dataset):
    loader = TwoWikiMultiHopQALoader(split="validation", data_path=mock_twowiki_dataset)
    samples = list(loader.load())
    
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, UnifiedQASample)
    assert sample.dataset_name == "2WikiMultiHopQA"
    assert sample.sample_id == "456"
    
    assert len(sample.corpus) == 2
    assert sample.corpus[0]["title"] == "Inception"
    assert "directed by Christopher Nolan" in sample.corpus[0]["text"]
    
    assert len(sample.supporting_contexts) == 1
    assert sample.supporting_contexts[0]["title"] == "Inception"
    assert "directed by Christopher Nolan" in sample.supporting_contexts[0]["text"]
    assert sample.metadata["evidences"][0][1] == "director"
