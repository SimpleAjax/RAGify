import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.evaluation.tracker import ExperimentTracker

def test_tracker_initialization():
    """Test that the tracker initializes correctly."""
    with patch('src.evaluation.tracker.mlflow.set_tracking_uri') as mock_set_uri, \
         patch('src.evaluation.tracker.mlflow.set_experiment') as mock_set_exp:
        
        tracker = ExperimentTracker(experiment_name="TestExp", tracking_uri="test_uri")
        
        mock_set_uri.assert_called_once_with("test_uri")
        mock_set_exp.assert_called_once_with("TestExp")

def test_log_parameters():
    """Test logging parameters."""
    with patch('src.evaluation.tracker.mlflow.set_tracking_uri'), \
         patch('src.evaluation.tracker.mlflow.set_experiment'), \
         patch('src.evaluation.tracker.mlflow.log_params') as mock_log_params:
         
        tracker = ExperimentTracker()
        tracker.log_parameters({"model": "gpt-4", "strategy": "naive"})
        mock_log_params.assert_called_once_with({"model": "gpt-4", "strategy": "naive"})

def test_log_metrics():
    """Test logging numeric metrics from a DataFrame."""
    with patch('src.evaluation.tracker.mlflow.set_tracking_uri'), \
         patch('src.evaluation.tracker.mlflow.set_experiment'), \
         patch('src.evaluation.tracker.mlflow.log_metrics') as mock_log_metrics:
         
        tracker = ExperimentTracker()
        df = pd.DataFrame({
            "context_precision": [0.8, 1.0],
            "context_recall": [0.5, 0.5],
            "non_numeric": ["A", "B"]
        })
        
        tracker.log_metrics(df)
        
        # Mean of precision is 0.9, recall is 0.5
        mock_log_metrics.assert_called_once_with({
            "context_precision": 0.9,
            "context_recall": 0.5
        })

def test_log_evaluation_artifact():
    """Test logging a dataframe as an artifact."""
    with patch('src.evaluation.tracker.mlflow.set_tracking_uri'), \
         patch('src.evaluation.tracker.mlflow.set_experiment'), \
         patch('src.evaluation.tracker.mlflow.log_artifact') as mock_log_artifact:
         
        tracker = ExperimentTracker()
        df = pd.DataFrame({"A": [1, 2]})
        
        tracker.log_evaluation_artifact(df, filename="test_res.csv")
        
        mock_log_artifact.assert_called_once()
