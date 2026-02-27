import json
import os
import urllib.request
from typing import Iterator
from src.schema import UnifiedQASample
from src.loaders.base import AbstractDatasetLoader

class TwoWikiMultiHopQALoader(AbstractDatasetLoader):
    """
    Loader for the 2WikiMultiHopQA dataset.
    Downloads the raw JSON directly to bypass pyarrow schema strictness issues 
    present in the huggingface datasets version.
    """
    
    # Hugging Face direct raw file URLs for the dataset
    URLS = {
        "train": "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/train.json",
        "validation": "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json",
        "test": "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/test.json"
    }

    def __init__(self, split: str = "validation", data_path: str = None):
        super().__init__(data_path=data_path, split=split)
        
        # If no local path is provided, download/cache it locally
        if not self.data_path:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ragify_datasets", "2wiki")
            os.makedirs(cache_dir, exist_ok=True)
            self.data_path = os.path.join(cache_dir, f"{self.split}.json")
            
            if not os.path.exists(self.data_path):
                url = self.URLS.get(self.split)
                if not url:
                    raise ValueError(f"Unknown split '{self.split}' for 2WikiMultiHopQA.")
                    
                print(f"Downloading 2WikiMultiHopQA ({self.split}) from {url}...")
                urllib.request.urlretrieve(url, self.data_path)
                print("Download complete.")
                
    def load(self) -> Iterator[UnifiedQASample]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for row in data:
            # Map context into corpus
            # context: [ ["title1", ["sentence1", "sentence2..."]], ["title2", ["sentence1..."]] ]
            corpus = []
            title_to_paragraphs = {}
            for title, sentences in row.get("context", []):
                # sentences is sometimes just a string instead of array due to bad schema
                if isinstance(sentences, str):
                    sentences = [sentences]
                    
                paragraph_text = "".join(sentences)
                corpus_entry = {"title": title, "text": paragraph_text}
                corpus.append(corpus_entry)
                title_to_paragraphs[title] = sentences
                
            # For 2Wiki, 'supporting_facts' structure: [ ["title1", sent_id1], ["title2", sent_id2] ]
            supporting_contexts = []
            
            for sf in row.get("supporting_facts", []):
                # Handle cases where sf might be malformed
                if isinstance(sf, list) and len(sf) >= 2:
                    sf_title = sf[0]
                    sf_sent_id = sf[1]
                else:
                    continue
                    
                if sf_title in title_to_paragraphs:
                    try:
                        sentences = title_to_paragraphs[sf_title]
                        if isinstance(sf_sent_id, int) and 0 <= sf_sent_id < len(sentences):
                            sf_text = sentences[sf_sent_id]
                            supporting_contexts.append({"title": sf_title, "text": sf_text, "sent_id": str(sf_sent_id)})
                    except Exception:
                        pass
                        
            sample = UnifiedQASample(
                dataset_name="2WikiMultiHopQA",
                sample_id=str(row.get("_id", row.get("id", ""))),
                query=row.get("question", ""),
                ground_truth_answer=row.get("answer", ""),
                supporting_contexts=supporting_contexts,
                corpus=corpus,
                metadata={
                    "type": row.get("type", "unknown"),
                    "evidences": row.get("evidences", [])
                }
            )
            yield sample
