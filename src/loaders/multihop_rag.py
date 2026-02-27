from typing import Iterator
from datasets import load_dataset
from src.schema import UnifiedQASample
from src.loaders.base import AbstractDatasetLoader

class MultiHopRAGLoader(AbstractDatasetLoader):
    """
    Loader for the MultiHop-RAG dataset.
    Can load from Hugging Face hub (e.g. 'yixuantt/MultiHopRAG') or local JSON if data_path is provided.
    """
    
    def __init__(self, split: str = "dataset", data_path: str = None, dataset_name: str = "yixuantt/MultiHopRAG"):
        # MultiHop-RAG typically doesn't use standard train/val splits natively, mostly just 'dataset'
        super().__init__(data_path=data_path, split=split)
        
        if self.data_path:
            self.dataset = load_dataset("json", data_files={self.split: self.data_path})[self.split]
        else:
            self.dataset = load_dataset(dataset_name, 'MultiHopRAG', split=self.split)
            
    def load(self) -> Iterator[UnifiedQASample]:
        for row in self.dataset:
            # MultiHop-RAG schema: query, answer, evidence_list (list of dicts {title, fact}) 
            # Note: The 'corpus' might not be fully provided per-sample like HotpotQA. 
            # Often, MultiHop-RAG provides a separate KB (Knowledge Base). If evidence_list is provided per doc:
            
            corpus = []
            supporting_contexts = []
            
            # Using evidence_list as both corpus (what was retrieved) and supporting context
            for evidence in row.get("evidence_list", []):
                title = evidence.get("title", "")
                text = evidence.get("fact", "")
                
                context_entry = {"title": title, "text": text}
                corpus.append(context_entry)
                supporting_contexts.append(context_entry)
                
            sample = UnifiedQASample(
                dataset_name="MultiHop-RAG",
                sample_id=str(row.get("query", ""))[:30], # Using query prefix as ID if true ID missing
                query=row.get("query", ""),
                ground_truth_answer=row.get("answer", ""),
                supporting_contexts=supporting_contexts,
                corpus=corpus,
                metadata={
                    "question_type": row.get("question_type", "unknown")
                }
            )
            yield sample
