from typing import Iterator
from datasets import load_dataset
from src.schema import UnifiedQASample
from src.loaders.base import AbstractDatasetLoader

class MuSiQueLoader(AbstractDatasetLoader):
    """
    Loader for the MuSiQue dataset.
    Uses reliable HuggingFace mirrors as the original StonyBrookNLP github and HuggingFace repos 
    were deleted / return 404s.
    """
    
    # Map our generic splits to the specific mirror repos
    DATASETS = {
        "train": "jerry128/Musique-Ans-Train-Baseline",
        "validation": "jerry128/Musique-Ans-Eval"
    }
    
    def __init__(self, split: str = "validation", data_path: str = None):
        super().__init__(data_path=data_path, split=split)
        
        if self.data_path:
            self.dataset = load_dataset("json", data_files={"train": self.data_path})["train"]
        else:
            dataset_repo = self.DATASETS.get(self.split)
            if not dataset_repo:
                raise ValueError(f"Split '{self.split}' not mapped to a valid MuSiQue HuggingFace mirror.")
            # These specific mirrors only have a 'train' split inside them
            self.dataset = load_dataset(dataset_repo, split="train")
            
    def load(self) -> Iterator[UnifiedQASample]:
        for row in self.dataset:
            corpus = []
            supporting_contexts = []
            
            for para in row.get("paragraphs", []):
                title = para.get("title", "")
                text = para.get("paragraph_text", para.get("text", ""))
                corpus_entry = {"title": title, "text": text}
                corpus.append(corpus_entry)
                
                if para.get("is_supporting", False):
                    supporting_contexts.append({"title": title, "text": text})
                    
            sample = UnifiedQASample(
                dataset_name="MuSiQue",
                sample_id=str(row.get("id", "")),
                query=row.get("question", ""),
                ground_truth_answer=row.get("answer", ""),
                supporting_contexts=supporting_contexts,
                corpus=corpus,
                metadata={
                    "question_decomposition": row.get("question_decomposition", []),
                    "type": row.get("type", "unknown")
                }
            )
            yield sample
