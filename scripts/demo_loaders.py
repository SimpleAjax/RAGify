import json
import logging
import sys
import os

# Add the project root to sys.path so we can import 'src' even when running from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loaders.hotpotqa import HotpotQALoader
from src.loaders.twowiki import TwoWikiMultiHopQALoader
from src.loaders.musique import MuSiQueLoader
from src.loaders.multihop_rag import MultiHopRAGLoader

logging.basicConfig(level=logging.INFO)

def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))

def print_sample(sample):
    data = sample.model_dump()
    safe_print(f"\n{'='*60}")
    safe_print(f"Dataset: {data['dataset_name']}")
    safe_print(f"ID:      {data['sample_id']}")
    safe_print(f"Query:   {data['query']}")
    safe_print(f"Answer:  {data['ground_truth_answer']}")
    
    safe_print("\n--- Supporting Contexts ---")
    for ctx in data['supporting_contexts']:
        title = ctx.get('title', 'No Title')
        text_preview = ctx['text'][:100] + "..." if len(ctx['text']) > 100 else ctx['text']
        safe_print(f" - [{title}] {text_preview}")
        
    safe_print("\n--- Corpus Extract ---")
    safe_print(f"Total Corpus Documents provided for context: {len(data['corpus'])}")
    if data['corpus']:
        first_doc = data['corpus'][0]
        title = first_doc.get('title', 'No Title')
        preview = first_doc['text'][:100] + "..." if len(first_doc['text']) > 100 else first_doc['text']
        safe_print(f" - Document 1: [{title}] {preview}")
        
    safe_print("\n--- Metadata ---")
    safe_print(json.dumps(data['metadata'], indent=2, ensure_ascii=True))
    safe_print(f"{'='*60}\n")

def run_demo():
    print("\n" + "#"*40)
    print("1. HotpotQA Demo")
    print("#"*40)
    hotpot_loader = HotpotQALoader(split="validation")
    for i, sample in enumerate(hotpot_loader.load()):
        if i >= 1: break
        print_sample(sample)


    print("\n" + "#"*40)
    print("2. 2WikiMultiHopQA Demo")
    print("#"*40)
    twowiki_loader = TwoWikiMultiHopQALoader(split="validation")
    for i, sample in enumerate(twowiki_loader.load()):
        if i >= 1: break
        print_sample(sample)


    print("\n" + "#"*40)
    print("3. MuSiQue Demo")
    print("#"*40)
    musique_loader = MuSiQueLoader(split="validation")
    for i, sample in enumerate(musique_loader.load()):
        if i >= 1: break
        print_sample(sample)


    print("\n" + "#"*40)
    print("4. MultiHop-RAG Demo")
    print("#"*40)
    # MultiHop-RAG uses 'train' split when configs are applied
    multihop_loader = MultiHopRAGLoader(split="train")
    for i, sample in enumerate(multihop_loader.load()):
        if i >= 1: break
        print_sample(sample)

if __name__ == "__main__":
    run_demo()
