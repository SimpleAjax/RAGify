from typing import List, Dict, Any
from pydantic import BaseModel

class UnifiedQASample(BaseModel):
    """
    A unified representation of a Multi-Hop QA sample that normalizes diverse datasets 
    (HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHop-RAG) into a standard format for downstream 
    RAG engines and RAGAS evaluation.
    """
    dataset_name: str           
    sample_id: str              
    query: str                 
    ground_truth_answer: str    
    
    # The actual text chunks/paragraphs containing the information required to answer the query
    supporting_contexts: List[Dict[str, str]] 
    
    # All documents/chunks provided as potential context in the dataset sample
    # (used for testing retrieval performance from a specific corpus subset)
    corpus: List[Dict[str, str]]              
    
    # Crucial for dataset-specific artifacts (e.g., Graph triples, Sub-queries, reasoning types)
    metadata: Dict[str, Any]                  
