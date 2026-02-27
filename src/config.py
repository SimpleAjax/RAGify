"""
RAGify Configuration Module

Loads configuration from environment variables with sensible defaults.
Copy .env.example to .env and customize for your setup.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Configuration class with environment variable loading."""
    
    # ==========================================================================
    # API Keys
    # ==========================================================================
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # ==========================================================================
    # Embedding Models (Local - FREE)
    # ==========================================================================
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    
    # ==========================================================================
    # RAGAS Evaluation Model
    # Default: claude-3-haiku (balanced - good judge, cheap)
    # ==========================================================================
    EVALUATION_MODEL: str = os.getenv("EVALUATION_MODEL", "openrouter/anthropic/claude-3-haiku")
    EVALUATION_API_BASE: str = os.getenv("EVALUATION_API_BASE", "https://openrouter.ai/api/v1")
    
    # ==========================================================================
    # RAG Strategy Models
    # ==========================================================================
    
    # Query Decomposition (structured JSON output required)
    # Default: gpt-4o-mini (best JSON adherence)
    DECOMPOSITION_MODEL: str = os.getenv("DECOMPOSITION_MODEL", "openrouter/openai/gpt-4o-mini")
    DECOMPOSITION_API_BASE: str = os.getenv("DECOMPOSITION_API_BASE", "https://openrouter.ai/api/v1")
    
    # Entity Extraction for GraphRAG (NER task)
    # Default: claude-3-haiku (good NER)
    ENTITY_EXTRACTION_MODEL: str = os.getenv("ENTITY_EXTRACTION_MODEL", "openrouter/anthropic/claude-3-haiku")
    ENTITY_EXTRACTION_API_BASE: str = os.getenv("ENTITY_EXTRACTION_API_BASE", "https://openrouter.ai/api/v1")
    
    # Answer Generation (NaiveRAG, GraphRAG final answer)
    # Default: gpt-4o-mini (good quality, low hallucination)
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "openrouter/openai/gpt-4o-mini")
    GENERATION_API_BASE: str = os.getenv("GENERATION_API_BASE", "https://openrouter.ai/api/v1")
    
    # Agentic RAG (tool calling + reasoning)
    # Default: claude-3-haiku (good tool use)
    AGENTIC_MODEL: str = os.getenv("AGENTIC_MODEL", "openrouter/anthropic/claude-3-haiku")
    AGENTIC_API_BASE: str = os.getenv("AGENTIC_API_BASE", "https://openrouter.ai/api/v1")
    
    # ==========================================================================
    # Alternative: Single Model for All RAG Tasks
    # If RAG_MODEL is set, it overrides individual strategy models
    # ==========================================================================
    RAG_MODEL: Optional[str] = os.getenv("RAG_MODEL")
    RAG_API_BASE: str = os.getenv("RAG_API_BASE", "https://openrouter.ai/api/v1")
    
    @classmethod
    def get_decomposition_config(cls) -> tuple[str, str]:
        """Get model and API base for decomposition."""
        if cls.RAG_MODEL:
            return cls.RAG_MODEL, cls.RAG_API_BASE
        return cls.DECOMPOSITION_MODEL, cls.DECOMPOSITION_API_BASE
    
    @classmethod
    def get_entity_extraction_config(cls) -> tuple[str, str]:
        """Get model and API base for entity extraction."""
        if cls.RAG_MODEL:
            return cls.RAG_MODEL, cls.RAG_API_BASE
        return cls.ENTITY_EXTRACTION_MODEL, cls.ENTITY_EXTRACTION_API_BASE
    
    @classmethod
    def get_generation_config(cls) -> tuple[str, str]:
        """Get model and API base for answer generation."""
        if cls.RAG_MODEL:
            return cls.RAG_MODEL, cls.RAG_API_BASE
        return cls.GENERATION_MODEL, cls.GENERATION_API_BASE
    
    @classmethod
    def get_agentic_config(cls) -> tuple[str, str]:
        """Get model and API base for agentic tasks."""
        if cls.RAG_MODEL:
            return cls.RAG_MODEL, cls.RAG_API_BASE
        return cls.AGENTIC_MODEL, cls.AGENTIC_API_BASE
    
    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "ragify_evaluator")
    
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # ==========================================================================
    # MLflow Configuration
    # ==========================================================================
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "RAGify_Evaluations")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    
    # ==========================================================================
    # Chunking Configuration
    # ==========================================================================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # ==========================================================================
    # Dataset Indexing Configuration
    # ==========================================================================
    # Sample sizes: use -1 for ALL samples, 0 to skip dataset, or positive number for subset
    HOTPOTQA_SAMPLE_SIZE: int = int(os.getenv("HOTPOTQA_SAMPLE_SIZE", "100"))
    TWOWIKI_SAMPLE_SIZE: int = int(os.getenv("TWOWIKI_SAMPLE_SIZE", "100"))
    MUSIQUE_SAMPLE_SIZE: int = int(os.getenv("MUSIQUE_SAMPLE_SIZE", "100"))
    MULTIHOP_RAG_SAMPLE_SIZE: int = int(os.getenv("MULTIHOP_RAG_SAMPLE_SIZE", "100"))
    
    # Dataset splits
    HOTPOTQA_SPLIT: str = os.getenv("HOTPOTQA_SPLIT", "validation")
    TWOWIKI_SPLIT: str = os.getenv("TWOWIKI_SPLIT", "train")
    MUSIQUE_SPLIT: str = os.getenv("MUSIQUE_SPLIT", "validation")
    MULTIHOP_RAG_SPLIT: str = os.getenv("MULTIHOP_RAG_SPLIT", "train")
    
    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """
        Get the primary API key to use.
        Prefers OPENROUTER_API_KEY, falls back to OPENAI_API_KEY.
        """
        return cls.OPENROUTER_API_KEY or cls.OPENAI_API_KEY
    
    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration and return list of missing/invalid settings.
        """
        errors = []
        
        if not cls.get_api_key():
            errors.append("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        
        # Check if using OpenRouter without API key
        models_to_check = [
            cls.EVALUATION_MODEL,
            cls.DECOMPOSITION_MODEL,
            cls.ENTITY_EXTRACTION_MODEL,
            cls.GENERATION_MODEL,
            cls.AGENTIC_MODEL,
        ]
        if cls.RAG_MODEL:
            models_to_check.append(cls.RAG_MODEL)
            
        uses_openrouter = any("openrouter" in m for m in models_to_check)
        
        if uses_openrouter and not cls.OPENROUTER_API_KEY:
            errors.append("Using OpenRouter models but OPENROUTER_API_KEY is not set.")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive values)."""
        print("=" * 60)
        print("RAGify Configuration")
        print("=" * 60)
        
        # API Keys (masked)
        openrouter_set = "[SET]" if cls.OPENROUTER_API_KEY else "[NOT SET]"
        openai_set = "[SET]" if cls.OPENAI_API_KEY else "[NOT SET]"
        print(f"\nAPI Keys:")
        print(f"  OPENROUTER_API_KEY: {openrouter_set}")
        print(f"  OPENAI_API_KEY: {openai_set}")
        
        # Embedding
        print(f"\nEmbedding Model (Local - FREE):")
        print(f"  EMBEDDING_MODEL_NAME: {cls.EMBEDDING_MODEL_NAME}")
        
        # Evaluation
        print(f"\nEvaluation Model:")
        print(f"  EVALUATION_MODEL: {cls.EVALUATION_MODEL}")
        print(f"  EVALUATION_API_BASE: {cls.EVALUATION_API_BASE}")
        
        # RAG Strategies
        if cls.RAG_MODEL:
            print(f"\nRAG Strategies (Using Single Model):")
            print(f"  RAG_MODEL: {cls.RAG_MODEL}")
            print(f"  RAG_API_BASE: {cls.RAG_API_BASE}")
        else:
            print(f"\nRAG Strategy Models:")
            print(f"  DECOMPOSITION_MODEL: {cls.DECOMPOSITION_MODEL}")
            print(f"  ENTITY_EXTRACTION_MODEL: {cls.ENTITY_EXTRACTION_MODEL}")
            print(f"  GENERATION_MODEL: {cls.GENERATION_MODEL}")
            print(f"  AGENTIC_MODEL: {cls.AGENTIC_MODEL}")
        
        # Database
        print(f"\nDatabase Configuration:")
        print(f"  QDRANT: {cls.QDRANT_HOST}:{cls.QDRANT_PORT}")
        print(f"  NEO4J: {cls.NEO4J_URI}")
        
        # Dataset Indexing
        print(f"\nDataset Indexing Configuration:")
        print(f"  HotpotQA: {cls._sample_size_str(cls.HOTPOTQA_SAMPLE_SIZE)} (split: {cls.HOTPOTQA_SPLIT})")
        print(f"  2WikiMultiHopQA: {cls._sample_size_str(cls.TWOWIKI_SAMPLE_SIZE)} (split: {cls.TWOWIKI_SPLIT})")
        print(f"  MuSiQue: {cls._sample_size_str(cls.MUSIQUE_SAMPLE_SIZE)} (split: {cls.MUSIQUE_SPLIT})")
        print(f"  MultiHop-RAG: {cls._sample_size_str(cls.MULTIHOP_RAG_SAMPLE_SIZE)} (split: {cls.MULTIHOP_RAG_SPLIT})")
        
        print("\n" + "=" * 60)
    
    @staticmethod
    def _sample_size_str(size: int) -> str:
        """Convert sample size to readable string."""
        if size == -1:
            return "ALL samples"
        elif size == 0:
            return "SKIPPED"
        else:
            return f"{size} samples"


# Create a singleton instance for easy import
config = Config()
