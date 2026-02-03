"""
Analysis runner: loads checkpoints and runs probes across training.

This is where the mechanistic story emerges.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from ..training.checkpoint import list_checkpoints, load_checkpoint
from ..probes import get_probe, list_probes, BaseProbe
from ..data import CharTokenizer, DisambiguationDataset, collate_fn


def run_probes_on_checkpoint(
    model: HookedTransformer,
    dataloader: DataLoader,
    probes: List[BaseProbe],
    device: str = "cuda",
) -> Dict[str, Dict[str, Any]]:
    """
    Run all probes on a single model checkpoint.
    
    Returns:
        Dict mapping probe names to their results
    """
    results = {}
    for probe in probes:
        results[probe.name] = probe.run(model, dataloader, device)
    return results


def run_analysis(
    experiment_dir: Path,
    dataset: DisambiguationDataset,
    tokenizer: CharTokenizer,
    model_factory,  # Callable that creates a fresh HookedTransformer
    probe_names: Optional[List[str]] = None,
    checkpoint_steps: Optional[List[int]] = None,
    batch_size: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run probes across all checkpoints for an experiment.
    
    Args:
        experiment_dir: Path to experiment output directory
        dataset: Dataset to use for probing
        tokenizer: Tokenizer
        model_factory: Function that creates a HookedTransformer
        probe_names: Which probes to run (default: all)
        checkpoint_steps: Which steps to analyze (default: all)
        batch_size: Batch size for probing
        device: Device to run on
        
    Returns:
        {
            "steps": [step1, step2, ...],
            "probe_results": {
                "probe_name": {
                    step1: {...results...},
                    step2: {...results...},
                    ...
                },
                ...
            }
        }
    """
    checkpoint_dir = experiment_dir / "checkpoints"
    
    # Get checkpoints
    all_steps = list_checkpoints(checkpoint_dir)
    if checkpoint_steps is not None:
        steps = [s for s in all_steps if s in checkpoint_steps]
    else:
        steps = all_steps
        
    if not steps:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    print(f"Found {len(steps)} checkpoints: {steps[:5]}...{steps[-5:]}")
    
    # Setup probes
    if probe_names is None:
        probe_names = list_probes()
    probes = [get_probe(name) for name in probe_names]
    
    # Setup dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Results container
    results = {
        "steps": steps,
        "probe_results": {probe.name: {} for probe in probes},
    }
    
    # Run probes on each checkpoint
    for step in tqdm(steps, desc="Analyzing checkpoints"):
        # Create fresh model and load checkpoint
        model = model_factory()
        load_checkpoint(model, None, checkpoint_dir, step)
        model.to(device)
        model.eval()
        
        # Run probes
        probe_results = run_probes_on_checkpoint(model, dataloader, probes, device)
        
        # Store results
        for probe_name, probe_result in probe_results.items():
            # Convert tensors to lists for JSON serialization
            serializable = {}
            for k, v in probe_result.items():
                if isinstance(v, torch.Tensor):
                    serializable[k] = v.tolist()
                else:
                    serializable[k] = v
            results["probe_results"][probe_name][step] = serializable
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save results
    results_path = experiment_dir / "probe_results" / "all_probes.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {results_path}")
    
    return results
