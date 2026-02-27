import argparse
from typing import List
import sys
import os

# Add the project root to sys.path so we can import 'src' even when running from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.schema import UnifiedQASample
from src.loaders.hotpotqa import HotpotQALoader
from src.loaders.twowiki import TwoWikiMultiHopQALoader
from src.loaders.musique import MuSiQueLoader
from src.loaders.multihop_rag import MultiHopRAGLoader

from src.indexing.qdrant_manager import QdrantManager
from src.indexing.neo4j_manager import Neo4jManager
from src.config import config
from itertools import islice


def load_datasets() -> List[UnifiedQASample]:
    print("--- [1] Loading Datasets from HuggingFace ---")
    all_samples = []
    
    # 1. Load HotpotQA
    if config.HOTPOTQA_SAMPLE_SIZE != 0:
        sample_size_str = "ALL" if config.HOTPOTQA_SAMPLE_SIZE == -1 else config.HOTPOTQA_SAMPLE_SIZE
        print(f"Loading HotpotQA ({sample_size_str} samples from {config.HOTPOTQA_SPLIT} split)...")
        hotpot_loader = HotpotQALoader(split=config.HOTPOTQA_SPLIT)
        
        if config.HOTPOTQA_SAMPLE_SIZE == -1:
            # Load all samples
            hotpot_samples = list(hotpot_loader.load())
        else:
            # Load subset
            hotpot_samples = list(islice(hotpot_loader.load(), config.HOTPOTQA_SAMPLE_SIZE))
        
        all_samples.extend(hotpot_samples)
        print(f"Loaded {len(hotpot_samples)} HotpotQA samples.")
    else:
        print("Skipping HotpotQA (HOTPOTQA_SAMPLE_SIZE=0)")
    
    # 2. Load 2WikiMultiHopQA (Provides Graph Evidences)
    if config.TWOWIKI_SAMPLE_SIZE != 0:
        sample_size_str = "ALL" if config.TWOWIKI_SAMPLE_SIZE == -1 else config.TWOWIKI_SAMPLE_SIZE
        print(f"Loading 2WikiMultiHopQA ({sample_size_str} samples from {config.TWOWIKI_SPLIT} split)...")
        wiki_loader = TwoWikiMultiHopQALoader(split=config.TWOWIKI_SPLIT)
        
        if config.TWOWIKI_SAMPLE_SIZE == -1:
            wiki_samples = list(wiki_loader.load())
        else:
            wiki_samples = list(islice(wiki_loader.load(), config.TWOWIKI_SAMPLE_SIZE))
        
        all_samples.extend(wiki_samples)
        print(f"Loaded {len(wiki_samples)} 2WikiMultiHopQA samples.")
    else:
        print("Skipping 2WikiMultiHopQA (TWOWIKI_SAMPLE_SIZE=0)")
    
    # 3. Load MuSiQue
    if config.MUSIQUE_SAMPLE_SIZE != 0:
        sample_size_str = "ALL" if config.MUSIQUE_SAMPLE_SIZE == -1 else config.MUSIQUE_SAMPLE_SIZE
        print(f"Loading MuSiQue ({sample_size_str} samples from {config.MUSIQUE_SPLIT} split)...")
        musique_loader = MuSiQueLoader(split=config.MUSIQUE_SPLIT)
        
        if config.MUSIQUE_SAMPLE_SIZE == -1:
            musique_samples = list(musique_loader.load())
        else:
            musique_samples = list(islice(musique_loader.load(), config.MUSIQUE_SAMPLE_SIZE))
        
        all_samples.extend(musique_samples)
        print(f"Loaded {len(musique_samples)} MuSiQue samples.")
    else:
        print("Skipping MuSiQue (MUSIQUE_SAMPLE_SIZE=0)")
    
    # 4. Load MultiHop-RAG
    if config.MULTIHOP_RAG_SAMPLE_SIZE != 0:
        sample_size_str = "ALL" if config.MULTIHOP_RAG_SAMPLE_SIZE == -1 else config.MULTIHOP_RAG_SAMPLE_SIZE
        print(f"Loading MultiHop-RAG ({sample_size_str} samples from {config.MULTIHOP_RAG_SPLIT} split)...")
        multihop_loader = MultiHopRAGLoader(split=config.MULTIHOP_RAG_SPLIT)
        
        if config.MULTIHOP_RAG_SAMPLE_SIZE == -1:
            multihop_samples = list(multihop_loader.load())
        else:
            multihop_samples = list(islice(multihop_loader.load(), config.MULTIHOP_RAG_SAMPLE_SIZE))
        
        all_samples.extend(multihop_samples)
        print(f"Loaded {len(multihop_samples)} MultiHop-RAG samples.")
    else:
        print("Skipping MultiHop-RAG (MULTIHOP_RAG_SAMPLE_SIZE=0)")
    
    return all_samples

def index_data(skip_vector=False, skip_graph=False, show_config=False):
    if show_config:
        config.print_config()
        return
    
    samples = load_datasets()
    
    if not samples:
        print("\nWARNING: No samples loaded! Check your dataset configuration in .env")
        return
    
    print(f"\nTotal samples to index: {len(samples)}")
    
    if not skip_vector:
        print("\n--- [2] Indexing Vector Data to Qdrant ---")
        try:
            # Uses embedding model from config (default: BAAI/bge-small-en-v1.5)
            qdrant = QdrantManager()
            qdrant.process_and_index_samples(samples, batch_size=50)
            print("Successfully Indexed Vectors!")
        except Exception as e:
            print(f"ERROR reaching Vector DB: {e}")
            print("Please ensure `docker compose up -d` is running.")
            
    if not skip_graph:
        print("\n--- [3] Indexing Graph Data to Neo4j ---")
        try:
            # Uses Neo4j connection from config
            neo4j = Neo4jManager()
            neo4j.process_and_index_samples(samples, batch_size=50)
            neo4j.close()
            print("Successfully Indexed Graph Triples!")
        except Exception as e:
            print(f"ERROR reaching Graph DB: {e}")
            print("Please ensure `docker compose up -d` is running AND Bolt port is exposed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Phase 1 Datasets into Vector and Graph DBs.")
    parser.add_argument("--skip-vector", action="store_true", help="Skip Qdrant embedding and indexing")
    parser.add_argument("--skip-graph", action="store_true", help="Skip Neo4j cypher extraction and indexing")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration and exit")
    
    args = parser.parse_args()
    index_data(skip_vector=args.skip_vector, skip_graph=args.skip_graph, show_config=args.show_config)
