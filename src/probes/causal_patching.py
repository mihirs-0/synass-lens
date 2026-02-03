"""
Causal Patching probe.

This is the SMOKING GUN probe for z-dependence.

The idea:
1. Take two examples with SAME B but DIFFERENT z (and thus different A)
   Example 1: (B, z1) → A1
   Example 2: (B, z2) → A2

2. Run clean forward pass on example 1, get output A1

3. Run corrupted forward pass:
   - Replace z-position activations with those from example 2
   - If model uses z: output changes toward A2
   - If model ignores z: output stays A1

This directly measures CAUSAL z-dependence, not just correlation.

The metric: "z-dependence score"
- 0 = model completely ignores z (corrupting z has no effect)
- 1 = model fully depends on z (corrupting z completely changes output)
"""

from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
import random

from .base import BaseProbe


class CausalPatchingProbe(BaseProbe):
    """
    Probe that measures causal z-dependence via activation patching.
    """
    
    name = "causal_patching"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.n_examples = config.get("n_examples", 256) if config else 256
        
    def _find_paired_examples(
        self,
        dataloader: DataLoader,
    ) -> List[Tuple[Dict, Dict]]:
        """
        Find pairs of examples with same base string but different z.
        
        Returns list of (example1, example2) pairs.
        """
        # Collect all examples, grouped by base string
        base_to_examples = {}
        
        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size):
                base_string = batch["base_strings"][i]
                
                example = {
                    "input_ids": batch["input_ids"][i],
                    "labels": batch["labels"][i],
                    "z_positions": batch["z_positions"][i],
                    "z_end_positions": batch["z_end_positions"][i],
                    "target_start_positions": batch["target_start_positions"][i],
                    "target_end_positions": batch["target_end_positions"][i],
                    "base_string": base_string,
                }
                
                if base_string not in base_to_examples:
                    base_to_examples[base_string] = []
                base_to_examples[base_string].append(example)
        
        # Find bases with multiple examples (different z's)
        pairs = []
        for base_string, examples in base_to_examples.items():
            if len(examples) >= 2:
                # Create pairs
                for i in range(0, len(examples) - 1, 2):
                    pairs.append((examples[i], examples[i + 1]))
                    
        # Shuffle and limit
        random.shuffle(pairs)
        return pairs[:self.n_examples]
    
    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Run causal patching probe.
        
        Returns:
            {
                "z_dependence_score": float
                    0 = model ignores z, 1 = model fully uses z
                "patching_effect_by_layer": (n_layers,) array
                    Effect of patching z at each layer
                "n_pairs_tested": int
            }
        """
        model.eval()
        n_layers = model.cfg.n_layers
        
        # Find paired examples
        pairs = self._find_paired_examples(dataloader)
        
        if len(pairs) == 0:
            return {
                "z_dependence_score": 0.0,
                "patching_effect_by_layer": torch.zeros(n_layers).cpu(),
                "n_pairs_tested": 0,
                "error": "No paired examples found (K=1?)",
            }
        
        # Metrics
        patching_effects = torch.zeros(n_layers, device=device)
        total_z_dependence = 0.0
        n_valid = 0
        
        with torch.no_grad():
            for ex1, ex2 in tqdm(pairs, desc="Causal patching", leave=False):
                # Move to device
                input1 = ex1["input_ids"].unsqueeze(0).to(device)
                input2 = ex2["input_ids"].unsqueeze(0).to(device)
                
                z_start1 = ex1["z_positions"].item()
                z_end1 = ex1["z_end_positions"].item()
                target_start1 = ex1["target_start_positions"].item()
                
                z_start2 = ex2["z_positions"].item()
                z_end2 = ex2["z_end_positions"].item()
                
                # Get target tokens for both examples
                target1 = ex1["labels"][target_start1].item()
                target2 = ex2["labels"][ex2["target_start_positions"].item()].item()
                
                if target1 == -100 or target2 == -100:
                    continue
                    
                if target1 == target2:
                    # Same target, patching won't help
                    continue
                
                # Clean run on example 1
                clean_logits = model(input1)
                # Get logits for first target position (predicted from position before it)
                clean_pred = clean_logits[0, target_start1 - 1, :]  # (vocab,)
                clean_prob_target1 = torch.softmax(clean_pred, dim=-1)[target1].item()
                clean_prob_target2 = torch.softmax(clean_pred, dim=-1)[target2].item()
                
                # Get corrupted activations from example 2
                _, cache2 = model.run_with_cache(input2)
                
                # Patch at each layer and measure effect
                for layer in range(n_layers):
                    # Get clean cache for example 1
                    _, cache1 = model.run_with_cache(input1)
                    
                    # Create patching hook
                    def patch_hook(activation, hook, layer_idx=layer):
                        # activation shape: (batch, seq, d_model)
                        # Replace z positions with values from cache2
                        activation[:, z_start1:z_end1, :] = cache2["resid_post", layer_idx][:, z_start2:z_end2, :]
                        return activation
                    
                    # Run with patching
                    patched_logits = model.run_with_hooks(
                        input1,
                        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)],
                    )
                    
                    patched_pred = patched_logits[0, target_start1 - 1, :]
                    patched_prob_target1 = torch.softmax(patched_pred, dim=-1)[target1].item()
                    patched_prob_target2 = torch.softmax(patched_pred, dim=-1)[target2].item()
                    
                    # Effect: how much did probability shift from target1 to target2?
                    # Normalized effect: (clean_prob1 - patched_prob1) / clean_prob1
                    # Or: increase in target2 probability
                    effect = (patched_prob_target2 - clean_prob_target2)
                    patching_effects[layer] += effect
                
                # Overall z-dependence: does patching z change the output?
                # Use the layer with maximum effect
                max_effect = patching_effects.max().item() / (n_valid + 1)
                total_z_dependence += max_effect
                n_valid += 1
        
        if n_valid == 0:
            return {
                "z_dependence_score": 0.0,
                "patching_effect_by_layer": torch.zeros(n_layers).cpu(),
                "n_pairs_tested": 0,
                "error": "No valid pairs after filtering",
            }
        
        # Average
        patching_effects /= n_valid
        z_dependence_score = total_z_dependence / n_valid
        
        return {
            "z_dependence_score": z_dependence_score,
            "patching_effect_by_layer": patching_effects.cpu(),
            "n_pairs_tested": n_valid,
        }
