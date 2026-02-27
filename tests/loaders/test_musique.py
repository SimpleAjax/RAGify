import pytest
import os
import json
from src.schema import UnifiedQASample
from src.loaders.musique import MuSiQueLoader

@pytest.fixture
def mock_musique_dataset(tmpdir):
    data = {"id": "789", "question": "What album features the song 'Halo'?", "answer": "I Am... Sasha Fierce", "paragraphs": [{"title": "Halo (song)", "paragraph_text": "'Halo' is a song recorded by American singer Beyoncé.", "is_supporting": True}], "question_decomposition": [], "type": "compositional"}
    file_path = os.path.join(tmpdir, "dev.jsonl")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
    return file_path

def test_musique_loader(mock_musique_dataset):
    loader = MuSiQueLoader(split="validation", data_path=mock_musique_dataset)
    samples = list(loader.load())
    
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, UnifiedQASample)
    assert sample.dataset_name == "MuSiQue"
    assert sample.sample_id == "789"
    assert sample.corpus[0]["title"] == "Halo (song)"
