"""
Base interface for mechanistic probes.

All probes inherit from this and implement the run() method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import json
import pickle

import torch
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader


class BaseProbe(ABC):
    """
    Abstract base class for mechanistic probes.
    
    Each probe:
    1. Takes a model checkpoint and data
    2. Runs some analysis (attention patterns, logit lens, patching, etc.)
    3. Returns results as a dictionary
    """
    
    name: str = "base_probe"
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize probe with optional config.
        
        Args:
            config: Probe-specific configuration
        """
        self.config = config or {}
        
    @abstractmethod
    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Run the probe on a model.
        
        Args:
            model: HookedTransformer to analyze
            dataloader: Data to use for analysis
            device: Device to run on
            
        Returns:
            Dictionary of results (must be JSON-serializable or pickleable)
        """
        raise NotImplementedError
        
    def save_results(self, results: Dict[str, Any], path: Path):
        """Save probe results to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try JSON first, fall back to pickle for tensors
        try:
            # Convert tensors to lists for JSON
            json_results = self._to_json_serializable(results)
            with open(path.with_suffix(".json"), "w") as f:
                json.dump(json_results, f, indent=2)
        except (TypeError, ValueError):
            # Fall back to pickle
            with open(path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(results, f)
                
    def _to_json_serializable(self, obj: Any) -> Any:
        """Convert tensors and numpy arrays to JSON-serializable types."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_json_serializable(v) for v in obj]
        else:
            return obj
            
    @classmethod
    def load_results(cls, path: Path) -> Dict[str, Any]:
        """Load probe results from disk."""
        json_path = path.with_suffix(".json")
        pkl_path = path.with_suffix(".pkl")
        
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
        elif pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"No results found at {path}")
