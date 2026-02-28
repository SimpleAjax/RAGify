"""
Tests for run_rag_evaluation.py script.

These tests mock expensive operations (LLM API calls, database connections)
to ensure the script logic works correctly without incurring costs.
"""

import pytest
import json
import os
import sys
from unittest.mock import MagicMock, patch, mock_open
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import UnifiedQASample


class TestDatasetLoader:
    """Tests for dataset loading functionality."""
    
    @patch('scripts.run_rag_evaluation.HotpotQALoader')
    def test_load_hotpotqa(self, mock_loader_class):
        """Test loading HotpotQA samples."""
        from scripts.run_rag_evaluation import load_test_samples
        
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Mock sample data
        mock_samples = [
            UnifiedQASample(
                sample_id="test1",
                dataset_name="HotpotQA",
                query="What is the capital of France?",
                ground_truth_answer="Paris",
                supporting_contexts=[{"title": "Doc1", "text": "Paris is the capital."}],
                corpus=[{"title": "Doc1", "text": "Paris is the capital of France."}],
                metadata={}
            ),
            UnifiedQASample(
                sample_id="test2",
                dataset_name="HotpotQA",
                query="Who wrote Romeo and Juliet?",
                ground_truth_answer="William Shakespeare",
                supporting_contexts=[],
                corpus=[],
                metadata={}
            )
        ]
        mock_loader.load.return_value = iter(mock_samples)
        
        # Test loading
        samples = load_test_samples("HotpotQA", max_samples=2)
        
        assert len(samples) == 2
        assert samples[0].query == "What is the capital of France?"
        assert samples[0].ground_truth_answer == "Paris"
        mock_loader_class.assert_called_once()
    
    @patch('scripts.run_rag_evaluation.HotpotQALoader')
    def test_load_limited_samples(self, mock_loader_class):
        """Test that max_samples limits the number of samples loaded."""
        from scripts.run_rag_evaluation import load_test_samples
        
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Create 10 mock samples
        mock_samples = [
            UnifiedQASample(
                sample_id=f"test{i}",
                dataset_name="HotpotQA",
                query=f"Question {i}?",
                ground_truth_answer=f"Answer {i}",
                supporting_contexts=[],
                corpus=[],
                metadata={}
            )
            for i in range(10)
        ]
        mock_loader.load.return_value = iter(mock_samples)
        
        # Load only 5
        samples = load_test_samples("HotpotQA", max_samples=5)
        
        assert len(samples) == 5


class TestRAGStrategyCreation:
    """Tests for RAG strategy creation."""
    
    def test_create_naive_rag(self):
        """Test creating NaiveRAG strategy."""
        from scripts.run_rag_evaluation import create_naive_rag
        
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_qdrant.get_langchain_retriever.return_value = mock_retriever
        
        strategy = create_naive_rag(mock_qdrant, "HotpotQA", mock_llm)
        
        assert strategy is not None
        mock_qdrant.get_langchain_retriever.assert_called_once_with(target_dataset="HotpotQA", k=5)
    
    def test_create_decomposition_rag(self):
        """Test creating DecompositionRAG strategy."""
        from scripts.run_rag_evaluation import create_decomposition_rag
        
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_qdrant.get_langchain_retriever.return_value = mock_retriever
        
        strategy = create_decomposition_rag(mock_qdrant, "HotpotQA", mock_llm)
        
        assert strategy is not None
        mock_qdrant.get_langchain_retriever.assert_called_once_with(target_dataset="HotpotQA", k=3)
    
    def test_create_graph_rag(self):
        """Test creating GraphRAG strategy."""
        from scripts.run_rag_evaluation import create_graph_rag
        
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        
        strategy = create_graph_rag(mock_neo4j, "HotpotQA", mock_llm)
        
        assert strategy is not None
        assert callable(strategy.graph_retriever)
    
    def test_create_agentic_rag(self):
        """Test creating AgenticRAG strategy."""
        from scripts.run_rag_evaluation import create_agentic_rag
        
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_qdrant.get_langchain_retriever.return_value = mock_retriever
        
        strategy = create_agentic_rag(mock_qdrant, "HotpotQA", mock_llm)
        
        assert strategy is not None
        assert len(strategy.tools) == 1


class TestStrategySelection:
    """Tests for strategy selection logic."""
    
    def test_get_naive_strategy(self):
        """Test getting NaiveRAG."""
        from scripts.run_rag_evaluation import get_rag_strategy
        
        mock_qdrant = MagicMock()
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        mock_qdrant.get_langchain_retriever.return_value = MagicMock()
        
        strategy = get_rag_strategy("naive", mock_qdrant, mock_neo4j, "HotpotQA", mock_llm)
        
        assert strategy is not None
        assert strategy.__class__.__name__ == "NaiveRAG"
    
    def test_get_graph_strategy(self):
        """Test getting GraphRAG."""
        from scripts.run_rag_evaluation import get_rag_strategy
        
        mock_qdrant = MagicMock()
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        
        strategy = get_rag_strategy("graph", mock_qdrant, mock_neo4j, "HotpotQA", mock_llm)
        
        assert strategy is not None
        assert strategy.__class__.__name__ == "GraphRAG"
    
    def test_get_agentic_strategy(self):
        """Test getting AgenticRAG."""
        from scripts.run_rag_evaluation import get_rag_strategy
        
        mock_qdrant = MagicMock()
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        mock_qdrant.get_langchain_retriever.return_value = MagicMock()
        
        strategy = get_rag_strategy("agentic", mock_qdrant, mock_neo4j, "HotpotQA", mock_llm)
        
        assert strategy is not None
        assert strategy.__class__.__name__ == "AgenticRAG"
    
    def test_get_decomposition_strategy(self):
        """Test getting DecompositionRAG."""
        from scripts.run_rag_evaluation import get_rag_strategy
        
        mock_qdrant = MagicMock()
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        mock_qdrant.get_langchain_retriever.return_value = MagicMock()
        
        strategy = get_rag_strategy("decomposition", mock_qdrant, mock_neo4j, "HotpotQA", mock_llm)
        
        assert strategy is not None
        assert strategy.__class__.__name__ == "DecompositionRAG"
    
    def test_invalid_strategy(self):
        """Test error on invalid strategy name."""
        from scripts.run_rag_evaluation import get_rag_strategy
        
        mock_qdrant = MagicMock()
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_rag_strategy("invalid", mock_qdrant, mock_neo4j, "HotpotQA", mock_llm)


class TestDatasetLoaderSelection:
    """Tests for dataset loader selection."""
    
    @patch('scripts.run_rag_evaluation.HotpotQALoader')
    def test_get_hotpotqa_loader(self, mock_loader):
        """Test getting HotpotQA loader."""
        from scripts.run_rag_evaluation import get_dataset_loader
        
        loader = get_dataset_loader("HotpotQA", split="validation")
        
        mock_loader.assert_called_once_with(split="validation")
        assert loader is not None
    
    @patch('scripts.run_rag_evaluation.TwoWikiMultiHopQALoader')
    def test_get_twowiki_loader(self, mock_loader):
        """Test getting 2Wiki loader."""
        from scripts.run_rag_evaluation import get_dataset_loader
        
        loader = get_dataset_loader("2WikiMultiHopQA")
        
        mock_loader.assert_called_once()
        assert loader is not None
    
    @patch('scripts.run_rag_evaluation.MuSiQueLoader')
    def test_get_musique_loader(self, mock_loader):
        """Test getting MuSiQue loader."""
        from scripts.run_rag_evaluation import get_dataset_loader
        
        loader = get_dataset_loader("MuSiQue")
        
        mock_loader.assert_called_once()
        assert loader is not None
    
    @patch('scripts.run_rag_evaluation.MultiHopRAGLoader')
    def test_get_multihoprag_loader(self, mock_loader):
        """Test getting MultiHopRAG loader."""
        from scripts.run_rag_evaluation import get_dataset_loader
        
        loader = get_dataset_loader("MultiHopRAG")
        
        mock_loader.assert_called_once()
        assert loader is not None
    
    def test_invalid_dataset(self):
        """Test error on invalid dataset name."""
        from scripts.run_rag_evaluation import get_dataset_loader
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_loader("InvalidDataset")


class TestRunEvaluationLogic:
    """Tests for the main run_evaluation function logic."""
    
    @patch('scripts.run_rag_evaluation.RagasEvaluator')
    @patch('scripts.run_rag_evaluation.ExperimentTracker')
    @patch('scripts.run_rag_evaluation.get_rag_strategy')
    @patch('scripts.run_rag_evaluation.ChatOpenAI')
    @patch('scripts.run_rag_evaluation.Neo4jManager')
    @patch('scripts.run_rag_evaluation.QdrantManager')
    @patch('scripts.run_rag_evaluation.load_test_samples')
    def test_run_evaluation_success(
        self, mock_load_samples, mock_qdrant, mock_neo4j, mock_chatopenai,
        mock_get_strategy, mock_tracker_class, mock_evaluator_class
    ):
        """Test successful evaluation run with mocked dependencies."""
        from scripts.run_rag_evaluation import run_evaluation
        
        # Mock samples
        mock_samples = [
            UnifiedQASample(
                sample_id="test1",
                dataset_name="HotpotQA",
                query="What is the capital of France?",
                ground_truth_answer="Paris",
                supporting_contexts=[],
                corpus=[],
                metadata={}
            )
        ]
        mock_load_samples.return_value = mock_samples
        
        # Mock RAG strategy
        mock_strategy = MagicMock()
        mock_strategy.retrieve_and_generate.return_value = {
            "query": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
            "retrieved_contexts": ["Paris is the capital."],
            "metadata": {"strategy": "NaiveRAG"}
        }
        mock_get_strategy.return_value = mock_strategy
        
        # Mock evaluator
        mock_evaluator = MagicMock()
        import pandas as pd
        mock_evaluator.evaluate_strategy.return_value = pd.DataFrame([
            {"context_precision": 1.0, "context_recall": 1.0}
        ])
        mock_evaluator_class.return_value = mock_evaluator
        
        # Mock tracker context manager
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_context = MagicMock()
        mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=mock_context)
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)
        
        # Run evaluation
        run_evaluation(
            strategy_name="naive",
            dataset_name="HotpotQA",
            model_name="test-model",
            api_base="https://test.api",
            max_samples=1,
            output_file="test_output.csv",
            run_name="test_run"
        )
        
        # Verify calls
        mock_load_samples.assert_called_once()
        mock_get_strategy.assert_called_once()
        mock_strategy.retrieve_and_generate.assert_called_once()
        mock_evaluator_class.assert_called_once()
        mock_tracker_class.assert_called_once()
    
    @patch('scripts.run_rag_evaluation.load_test_samples')
    def test_run_evaluation_no_samples(self, mock_load_samples):
        """Test handling when no samples are loaded."""
        from scripts.run_rag_evaluation import run_evaluation
        
        mock_load_samples.return_value = []
        
        # Should not raise error, just return early
        run_evaluation(
            strategy_name="naive",
            dataset_name="HotpotQA",
            model_name="test-model",
            api_base="https://test.api",
            max_samples=0,
            output_file="test_output.csv"
        )
        
        mock_load_samples.assert_called_once()
    
    @patch('scripts.run_rag_evaluation.QdrantManager')
    @patch('scripts.run_rag_evaluation.load_test_samples')
    def test_run_evaluation_qdrant_failure(self, mock_load_samples, mock_qdrant):
        """Test handling when Qdrant connection fails."""
        from scripts.run_rag_evaluation import run_evaluation
        
        mock_samples = [
            UnifiedQASample(
                sample_id="test1",
                dataset_name="HotpotQA",
                query="Test?",
                ground_truth_answer="Answer",
                supporting_contexts=[],
                corpus=[],
                metadata={}
            )
        ]
        mock_load_samples.return_value = mock_samples
        mock_qdrant.side_effect = Exception("Connection failed")
        
        # Should handle gracefully (naive strategy requires Qdrant)
        with patch('scripts.run_rag_evaluation.Neo4jManager') as mock_neo4j:
            mock_neo4j.side_effect = Exception("Neo4j also failed")
            # This should print error and return early
            run_evaluation(
                strategy_name="naive",
                dataset_name="HotpotQA",
                model_name="test-model",
                api_base="https://test.api",
                max_samples=1,
                output_file="test_output.csv"
            )


class TestDryRun:
    """Tests for dry-run mode (no API calls)."""
    
    @patch('scripts.run_rag_evaluation.RagasEvaluator')
    @patch('scripts.run_rag_evaluation.ExperimentTracker')
    @patch('scripts.run_rag_evaluation.get_rag_strategy')
    @patch('scripts.run_rag_evaluation.Neo4jManager')
    @patch('scripts.run_rag_evaluation.QdrantManager')
    @patch('scripts.run_rag_evaluation.load_test_samples')
    def test_dry_run_uses_fake_llm_and_skips_ragas(
        self, mock_load_samples, mock_qdrant, mock_neo4j,
        mock_get_strategy, mock_tracker_class, mock_evaluator_class
    ):
        """Test that dry-run mode uses FakeLLM and skips RAGAS evaluation."""
        from scripts.run_rag_evaluation import run_evaluation
        
        # Mock samples
        mock_samples = [
            UnifiedQASample(
                sample_id="test1",
                dataset_name="HotpotQA",
                query="Test?",
                ground_truth_answer="Answer",
                supporting_contexts=[],
                corpus=[],
                metadata={}
            )
        ]
        mock_load_samples.return_value = mock_samples
        
        # Mock strategy that returns predictable result
        mock_strategy = MagicMock()
        mock_strategy.retrieve_and_generate.return_value = {
            "query": "Test?",
            "answer": "[DRY RUN] Fake answer",
            "retrieved_contexts": ["Context"],
            "metadata": {}
        }
        mock_get_strategy.return_value = mock_strategy
        
        # Mock tracker
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_context = MagicMock()
        mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=mock_context)
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)
        
        # Run with dry_run=True
        run_evaluation(
            strategy_name="naive",
            dataset_name="HotpotQA",
            model_name="test-model",
            api_base="https://test.api",
            max_samples=1,
            output_file="test.csv",
            run_name="test",
            dry_run=True
        )
        
        # Verify strategy was called (which means FakeLLM was used)
        mock_get_strategy.assert_called_once()
        mock_strategy.retrieve_and_generate.assert_called_once()
        
        # Verify RagasEvaluator was NOT called (saving API costs)
        mock_evaluator_class.assert_not_called()


class TestMainFunction:
    """Tests for the main CLI function."""
    
    @patch('scripts.run_rag_evaluation.run_evaluation')
    @patch('scripts.run_rag_evaluation.config')
    def test_main_with_args(self, mock_config, mock_run_eval):
        """Test main function with CLI arguments."""
        from scripts.run_rag_evaluation import main
        
        mock_config.validate.return_value = []
        mock_config.GENERATION_MODEL = "default-model"
        mock_config.GENERATION_API_BASE = "https://default.api"
        
        test_args = [
            "run_rag_evaluation.py",
            "--strategy", "naive",
            "--dataset", "HotpotQA",
            "--samples", "50",
            "--model", "custom-model",
            "--output", "custom_output.csv",
            "--run-name", "custom_run"
        ]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_run_eval.assert_called_once()
        call_kwargs = mock_run_eval.call_args.kwargs
        assert call_kwargs['strategy_name'] == "naive"
        assert call_kwargs['dataset_name'] == "HotpotQA"
        assert call_kwargs['model_name'] == "custom-model"
        assert call_kwargs['max_samples'] == 50
        assert call_kwargs['output_file'] == "custom_output.csv"
        assert call_kwargs['run_name'] == "custom_run"
    
    @patch('scripts.run_rag_evaluation.config')
    def test_main_config_validation_error(self, mock_config):
        """Test main function exits on config validation error."""
        from scripts.run_rag_evaluation import main
        
        mock_config.validate.return_value = ["Missing API key"]
        
        test_args = [
            "run_rag_evaluation.py",
            "--strategy", "naive",
            "--dataset", "HotpotQA"
        ]
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('scripts.run_rag_evaluation.run_evaluation')
    @patch('scripts.run_rag_evaluation.config')
    def test_main_defaults(self, mock_config, mock_run_eval):
        """Test main function uses config defaults."""
        from scripts.run_rag_evaluation import main
        
        mock_config.validate.return_value = []
        mock_config.GENERATION_MODEL = "default-model"
        mock_config.GENERATION_API_BASE = "https://default.api"
        
        test_args = [
            "run_rag_evaluation.py",
            "--strategy", "graph",
            "--dataset", "2WikiMultiHopQA"
        ]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_run_eval.assert_called_once()
        call_kwargs = mock_run_eval.call_args.kwargs
        assert call_kwargs['strategy_name'] == "graph"
        assert call_kwargs['dataset_name'] == "2WikiMultiHopQA"
        assert call_kwargs['model_name'] == "default-model"  # From config
        assert call_kwargs['api_base'] == "https://default.api"  # From config
        assert call_kwargs['max_samples'] == 100  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
