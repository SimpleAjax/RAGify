import pytest
from unittest.mock import patch, MagicMock
from src.evaluation.evaluator import RagasEvaluator
import pandas as pd

def test_evaluator_initialization():
    """Test that the evaluator initializes with the LiteLLM wrapper."""
    with patch('src.evaluation.evaluator.ChatLiteLLM') as mock_llm:
        evaluator = RagasEvaluator(model_name="ollama/llama3", api_base="http://localhost:11434")
        assert evaluator.metrics is not None
        assert len(evaluator.metrics) == 4

def test_evaluator_missing_keys():
    """Test that missing keys in the eval_data dictionary raises ValueError."""
    with patch('src.evaluation.evaluator.ChatLiteLLM'):
        evaluator = RagasEvaluator()
        bad_data = [{"question": "Q", "answer": "A"}] # missing contexts and ground_truth
        with pytest.raises(ValueError, match="missing required keys"):
            evaluator.evaluate_strategy(bad_data)

@patch('src.evaluation.evaluator.evaluate')
def test_evaluator_execution(mock_evaluate):
    """Test that the evaluator correctly formats data and calls ragas."""
    # Mock the return of ragas evaluate
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = pd.DataFrame([{"context_precision": 0.9}])
    mock_evaluate.return_value = mock_result
    
    with patch('src.evaluation.evaluator.ChatLiteLLM'):
        evaluator = RagasEvaluator()
        good_data = [{
            "question": "Q1",
            "answer": "A1",
            "contexts": ["C1"],
            "ground_truth": "GT"
        }]
        
        df = evaluator.evaluate_strategy(good_data)
        assert mock_evaluate.called
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["context_precision"] == 0.9
