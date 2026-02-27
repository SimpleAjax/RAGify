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

def load_datasets() -> List[UnifiedQASample]:
    print("--- [1] Loading Datasets from HuggingFace ---")
    all_samples = []
    
    # 1. Load HotpotQA
    print("Loading HotpotQA (Subset of 100 docs for fast local indexing)...")
    hotpot_loader = HotpotQALoader(split="validation")
    
    # The loader returns an iterator, we slice the first 100
    from itertools import islice
    
    hotpot_samples = list(islice(hotpot_loader.load(), 100))
    all_samples.extend(hotpot_samples)
    print(f"Loaded {len(hotpot_samples)} HotpotQA samples.")
    
    # 2. Load 2WikiMultiHopQA (Provides Graph Evidences)
    print("Loading 2WikiMultiHopQA (Subset of 100 docs)...")
    wiki_loader = TwoWikiMultiHopQALoader(split="train")
    wiki_samples = list(islice(wiki_loader.load(), 100))
    all_samples.extend(wiki_samples)
    print(f"Loaded {len(wiki_samples)} 2WikiMultiHopQA samples.")
    
    # 3. Load MuSiQue
    print("Loading MuSiQue (Subset of 100 docs)...")
    musique_loader = MuSiQueLoader(split="validation")
    musique_samples = list(islice(musique_loader.load(), 100))
    all_samples.extend(musique_samples)
    print(f"Loaded {len(musique_samples)} MuSiQue samples.")
    
    # 4. Load MultiHop-RAG
    print("Loading MultiHop-RAG (Subset of 100 docs)...")
    multihop_loader = MultiHopRAGLoader(split="train")
    multihop_samples = list(islice(multihop_loader.load(), 100))
    all_samples.extend(multihop_samples)
    print(f"Loaded {len(multihop_samples)} MultiHop-RAG samples.")
    
    return all_samples

def index_data(skip_vector=False, skip_graph=False):
    samples = load_datasets()
    
    if not skip_vector:
        print("\n--- [2] Indexing Vector Data to Qdrant ---")
        try:
            # Requires Qdrant Docker container running on port 6333
            qdrant = QdrantManager(embedding_model_name="BAAI/bge-small-en-v1.5")
            qdrant.process_and_index_samples(samples, batch_size=50)
            print("Successfully Indexed Vectors!")
        except Exception as e:
            print(f"ERROR reaching Vector DB: {e}")
            print("Please ensure `docker compose up -d` is running.")
            
    if not skip_graph:
        print("\n--- [3] Indexing Graph Data to Neo4j ---")
        try:
            # Requires Neo4j Docker container running on port 7687
            neo4j = Neo4jManager(uri="bolt://localhost:7687", user="neo4j", password="password")
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
    
    args = parser.parse_args()
    index_data(skip_vector=args.skip_vector, skip_graph=args.skip_graph)
