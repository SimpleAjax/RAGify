from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

from src.schema import UnifiedQASample
from src.config import config

class QdrantManager:
    """
    Handles chunking, embedding, and upserting Phase 1 unified data into Qdrant.
    Ensures strict Multi-Tenancy segregation via Metadata payload tagging.
    """
    def __init__(
        self, 
        collection_name: str = None, 
        embedding_model_name: str = None,
        host: str = None,
        port: int = None
    ):
        # Use config defaults if not provided
        self.collection_name = collection_name or config.QDRANT_COLLECTION_NAME
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL_NAME
        self.host = host or config.QDRANT_HOST
        self.port = port or config.QDRANT_PORT
        
        print(f"Initializing HuggingFace Embeddings: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Determine vector size based on the model. BAAI/bge-small-en-v1.5 has 384 dims
        # To make this dynamic, we embed a dummy string to check
        sample_vector = self.embeddings.embed_query("test")
        self.vector_size = len(sample_vector)
        
        print(f"Connecting to Qdrant at {self.host}:{self.port}")
        self.client = QdrantClient(host=self.host, port=self.port)
        
        self._ensure_collection_exists()
        
        # Setup Text Splitter with config values
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def _ensure_collection_exists(self):
        """Creates the collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating new Qdrant collection: {self.collection_name} with {self.vector_size} dims.")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            
            # Create payload indexes for efficient filtering and grouping
            # These indexes enable fast filter queries and aggregations
            indexes = [
                ("dataset", "keyword"),           # For dataset segregation
                ("sample_id", "keyword"),         # For sample-level lookups
                ("composite_id", "keyword"),      # For grouping by dataset+sample
            ]
            
            for field_name, field_schema in indexes:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_schema
                    )
                    print(f"  Created index on '{field_name}'")
                except Exception as e:
                    print(f"  Warning: Could not create index on '{field_name}': {e}")

    def process_and_index_samples(self, samples: List[UnifiedQASample], batch_size: int = 100):
        """
        Takes UnifiedQASample objects, extracts raw corpus texts, chunks them, embeds them,
        and pushes them to Qdrant WITH the 'dataset' metadata tag to ensure segregation.
        """
        all_points = []
        
        for sample in samples:
            # We index the full provided `corpus` from the dataset sample 
            # (which contains both supporting facts and decoy texts)
            raw_texts = [doc.get("text", "") for doc in sample.corpus if doc.get("text")]
            
            # Chunk the documents
            chunks = self.text_splitter.create_documents(raw_texts)
            
            # Prepare metadata mapping
            for chunk in chunks:
                # Create composite_id for easy grouping: dataset_sample_id
                composite_id = f"{sample.dataset_name}_{sample.sample_id}"
                
                payload = {
                    "dataset": sample.dataset_name,
                    "sample_id": sample.sample_id,
                    "composite_id": composite_id,  # For easy grouping/filtering
                    "text": chunk.page_content, # The actual text used for generation later
                }
                
                # Combine source chunk metadata if any exists
                payload.update(chunk.metadata)
                
                # We need a unique ID for the point. Use UUID built from dataset + sample
                point_id = str(uuid.uuid4())
                
                # For batch processing, we just store the payload and text for now
                all_points.append({
                    "id": point_id,
                    "text_to_embed": chunk.page_content,
                    "payload": payload
                })

        print(f"Generated {len(all_points)} total chunks across {len(samples)} samples.")
        
        # Batch Embed and Upsert
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            
            texts = [item["text_to_embed"] for item in batch]
            print(f"Embedding batch {i} to {i+len(batch)}...")
            embeddings = self.embeddings.embed_documents(texts)
            
            points = [
                PointStruct(
                    id=item["id"],
                    vector=embeddings[idx],
                    payload=item["payload"]
                )
                for idx, item in enumerate(batch)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Upserted {len(points)} points to Qdrant.")
            
        print("Indexing Complete!")

    def get_langchain_retriever(self, target_dataset: str, k: int = 5):
        """
        Returns a LangChain VectorStoreRetriever uniquely configured to ONLY 
        query data belonging to `target_dataset`.
        
        This is the critical Multi-Tenancy bridge required for the RAG strategies.
        """
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # The Filter enforcing dataset segregation
        dataset_filter = Filter(
            must=[
                FieldCondition(
                    key="dataset",
                    match=MatchValue(value=target_dataset)
                )
            ]
        )
        
        # Re-wrap into LangChain's Qdrant abstraction
        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            content_payload_key="text"
        )
        
        return vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": dataset_filter
            }
        )
