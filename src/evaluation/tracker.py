import os
import mlflow
import pandas as pd
from typing import Dict, Any, Optional

from src.config import config

class ExperimentTracker:
    def __init__(self, experiment_name: str = None, tracking_uri: str = None):
        """
        Initializes the MLflow Experiment Tracker.
        Args:
            experiment_name (str): Name of the experiment to log runs under.
                                  If None, uses MLFLOW_EXPERIMENT_NAME from config.
            tracking_uri (str): URI for the MLflow tracking server.
                               If None, uses MLFLOW_TRACKING_URI from config.
        """
        self.experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
        self.tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
        
        # Ensure we set the tracking URI before setting the experiment
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set the experiment, creating it if it doesn't already exist
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        """
        Starts an MLflow run. Should be used in a context manager.
        """
        return mlflow.start_run(run_name=run_name)

    def log_parameters(self, params: Dict[str, Any]):
        """
        Logs a dictionary of parameters to the current MLflow run.
        Useful for logging Strategy name, LLM model, chunk size, etc.
        """
        mlflow.log_params(params)

    def log_metrics(self, df_metrics: pd.DataFrame):
        """
        Logs aggregated metrics from the Ragas evaluation DataFrame.
        Automatically calculates the mean for all numeric columns.
        """
        # Select only numeric columns to average
        numeric_df = df_metrics.select_dtypes(include='number')
        if not numeric_df.empty:
            averages = numeric_df.mean().to_dict()
            mlflow.log_metrics(averages)

    def log_evaluation_artifact(self, df_results: pd.DataFrame, filename: str = "evaluation_results.csv"):
        """
        Saves the detailed DataFrame (including questions, generated answers, contexts, and row scores)
        as a CSV artifact attached to the current MLflow run.
        """
        # Save to a temporary file, log it as an artifact, then remove the temp file
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, filename)
            df_results.to_csv(temp_path, index=False)
            mlflow.log_artifact(temp_path)
