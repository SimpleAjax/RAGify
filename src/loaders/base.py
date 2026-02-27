from abc import ABC, abstractmethod
from typing import Iterator

from src.schema import UnifiedQASample

class AbstractDatasetLoader(ABC):
    """
    Base interface for all dataset loaders. 
    Each specific loader implementation (e.g., HotpotQA, MuSiQue) must implement the load() method 
    and yield UnifiedQASample objects.
    """
    
    def __init__(self, data_path: str = None, split: str = "validation"):
        """
        Initialize the AbstractDatasetLoader.
        
        Args:
            data_path (str, optional): Path to local data file, if not fetching via Hugging Face.
            split (str): The dataset split to load (e.g., 'train', 'validation', 'test').
        """
        self.data_path = data_path
        self.split = split
        
    @abstractmethod
    def load(self) -> Iterator[UnifiedQASample]:
        """
        Loads the dataset and yields standardized UnifiedQASample objects.
        
        Yields:
            UnifiedQASample: A mapped dataset sample.
        """
        pass
