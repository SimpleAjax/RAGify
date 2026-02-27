from typing import Iterator, Dict, Any
from datasets import load_dataset
from src.schema import UnifiedQASample
from src.loaders.base import AbstractDatasetLoader

class HotpotQALoader(AbstractDatasetLoader):
    """
    Loader for the HotpotQA dataset via Hugging Face ('hotpot_qa').
    """
    
    def __init__(self, config_name: str = "distractor", split: str = "validation", data_path: str = None):
        """
        Args:
            config_name (str): Configuration mapping for HotpotQA ('distractor' or 'fullwiki').
            split (str): Dataset split ('train', 'validation').
            data_path (str, optional): Not used for fetching via Hugging Face.
        """
        super().__init__(data_path=data_path, split=split)
        self.config_name = config_name
        self.dataset = load_dataset("hotpot_qa", self.config_name, split=self.split)
        
    def load(self) -> Iterator[UnifiedQASample]:
        for row in self.dataset:
            # Map HotpotQA 'context' into corpus
            # Context comes as dict: {'title': [t1, t2], 'sentences': [[s1, s2], [s3, s4]]}
            corpus = []
            title_to_paragraphs = {}
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
                paragraph_text = "".join(sentences)
                corpus_entry = {"title": title, "text": paragraph_text}
                corpus.append(corpus_entry)
                title_to_paragraphs[title] = sentences
                
            # Map supporting facts into supporting_contexts
            # Supporting facts: {'title': [t1, t2], 'sent_id': [0, 2]}
            supporting_contexts = []
            
            sf_titles = row["supporting_facts"]["title"]
            sf_sent_ids = row["supporting_facts"]["sent_id"]
            
            # Using a set to avoid duplicating full paragraphs if multiple sentences from the same paragraph are supporting facts
            added_titles = set()
            
            for index, sf_title in enumerate(sf_titles):
                if sf_title not in added_titles and sf_title in title_to_paragraphs:
                    # In HotpotQA, usually the whole paragraph acts as the context, or specific sentences.
                    # RAG engines typically ingest chunks. Let's provide the whole paragraph where the facts reside
                    # as the supporting context or we could just provide the exact sentence. For general RAG, 
                    # the full paragraph is typical. Here we store the exact sentence and the title.
                    sent_id = sf_sent_ids[index]
                    try:
                        sf_text = title_to_paragraphs[sf_title][sent_id]
                        supporting_contexts.append({"title": sf_title, "text": sf_text, "sent_id": str(sent_id)})
                    except IndexError:
                        pass # Sometimes sent_id can be out of bounds due to noisy dataset annotations
            
            sample = UnifiedQASample(
                dataset_name="HotpotQA",
                sample_id=row["id"],
                query=row["question"],
                ground_truth_answer=row["answer"],
                supporting_contexts=supporting_contexts,
                corpus=corpus,
                metadata={
                    "type": row.get("type", "unknown"),
                    "level": row.get("level", "unknown"),
                    "config_name": self.config_name
                }
            )
            yield sample
