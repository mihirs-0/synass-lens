"""
Attention to z probe.

        Measures how much attention each head pays to the z (selector) tokens
        when predicting the target tokens.

This is THE key probe: if the model isn't attending to z, it can't
use z for disambiguation, explaining the lag.

What we measure:
- For each layer and head
- Attention weight from A positions → z positions
- Averaged across batch

Expected finding:
- Early in training: uniform/low attention to z
- Later in training: sharp increase in attention to z
"""

from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .base import BaseProbe


class AttentionToZProbe(BaseProbe):
    """
    Probe that measures attention from target positions to z positions.
    """
    
    name = "attention_to_z"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.aggregate = config.get("aggregate", True) if config else True
        self.first_token_only = config.get("first_token_only", False) if config else False
        
    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Run attention probe.
        
        Returns:
            {
                "attention_to_z": (n_layers, n_heads) array of attention weights,
                "attention_to_z_per_head": (n_layers, n_heads) detailed breakdown,
                "attention_entropy": (n_layers, n_heads) entropy of attention patterns,
            }
        """
        model.eval()
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        
        # Accumulators
        attn_to_z_sum = torch.zeros(n_layers, n_heads, device=device)
        attn_entropy_sum = torch.zeros(n_layers, n_heads, device=device)
        n_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Attention probe", leave=False):
                batch_size = batch["input_ids"].shape[0]
                input_ids = batch["input_ids"].to(device)
                z_positions = batch["z_positions"].to(device)  # (batch,)
                z_end_positions = batch["z_end_positions"].to(device)  # (batch,)
                target_start_positions = batch["target_start_positions"].to(device)  # (batch,)
                target_end_positions = batch["target_end_positions"].to(device)  # (batch,)
                
                # Run with cache to get attention patterns
                _, cache = model.run_with_cache(input_ids)
                
                # For each layer, extract attention patterns
                for layer in range(n_layers):
                    # Attention pattern: (batch, n_heads, seq_len, seq_len)
                    # attn[b, h, i, j] = attention from position i to position j
                    attn = cache["pattern", layer]
                    
                    # We want: attention from target positions to z positions
                    # This requires per-example indexing since positions vary
                    
                    for b in range(batch_size):
                        z_start = z_positions[b].item()
                        z_end = z_end_positions[b].item()
                        target_start = target_start_positions[b].item()
                        target_end = target_end_positions[b].item()
                        
                        if self.first_token_only:
                            target_positions = slice(target_start, target_start + 1)
                        else:
                            target_positions = slice(target_start, target_end)
                        
                        # Attention from target positions to z positions
                        # Shape: (n_heads, target_len, z_len)
                        attn_target_to_z = attn[b, :, target_positions, z_start:z_end]
                        
                        # Sum over source (target) and target (z) positions to get
                        # total attention mass from target → z for each head
                        # Then normalize by number of target positions
                        if self.first_token_only:
                            target_len = 1
                        else:
                            target_len = target_end - target_start
                        
                        # Average attention to z (per target position)
                        attn_to_z = attn_target_to_z.sum(dim=(1, 2)) / target_len  # (n_heads,)
                        attn_to_z_sum[layer] += attn_to_z
                        
                        # Entropy of attention pattern (from target positions)
                        # Higher entropy = more uniform attention
                        # Lower entropy = more focused attention
                        attn_from_target = attn[b, :, target_positions, :]  # (n_heads, target_len, seq_len)
                        # Average over target positions
                        attn_avg = attn_from_target.mean(dim=1)  # (n_heads, seq_len)
                        # Compute entropy
                        entropy = -(attn_avg * (attn_avg + 1e-10).log()).sum(dim=-1)  # (n_heads,)
                        attn_entropy_sum[layer] += entropy
                        
                        n_samples += 1
        
        # Average
        attn_to_z_avg = attn_to_z_sum / n_samples
        attn_entropy_avg = attn_entropy_sum / n_samples
        
        return {
            "attention_to_z": attn_to_z_avg.cpu(),
            "attention_entropy": attn_entropy_avg.cpu(),
            "n_samples": n_samples,
        }
