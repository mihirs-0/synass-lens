"""
Multihead specialization probe (Experiment 5, Probes 5A-5D).

Decomposes the Transformer's prediction into per-head and per-MLP-layer
contributions to understand the internal circuit that resolves the
disambiguation lag.

* 5A: Per-head attention weight from A-positions to z.
* 5B: Per-head value contribution (logit advantage for correct answer).
* 5C: Head activation timing (derived from 5A trajectory).
* 5D: MLP layer logit contribution.
"""

from typing import Any, Dict, List

import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from .base import BaseProbe


class MultiheadDecompositionProbe(BaseProbe):
    """Per-head attention + value + MLP decomposition at a single checkpoint."""

    name = "multihead_decomposition"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Run the full decomposition on *dataloader*.

        Returns:
            Dictionary with per-layer-per-head attention weights to z,
            logit advantage via z, and MLP logit contributions.
        """
        model.eval()
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        # Accumulators --------------------------------------------------
        attn_z = torch.zeros(n_layers, n_heads, device=device)
        attn_b = torch.zeros(n_layers, n_heads, device=device)
        head_advantage = torch.zeros(n_layers, n_heads, device=device)
        mlp_advantage = torch.zeros(n_layers, device=device)
        attn_layer_advantage = torch.zeros(n_layers, device=device)
        n_samples = 0

        W_U = model.W_U.detach()  # (d_model, d_vocab)

        with torch.no_grad():
            for batch in dataloader:
                batch_size = batch["input_ids"].shape[0]
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                z_positions = batch["z_positions"].to(device)
                z_end_positions = batch["z_end_positions"].to(device)
                target_start = batch["target_start_positions"].to(device)

                _, cache = model.run_with_cache(input_ids)

                for bi in range(batch_size):
                    z_s = z_positions[bi].item()
                    z_e = z_end_positions[bi].item()
                    t_pos = target_start[bi].item()
                    if t_pos <= 0 or z_s < 0:
                        continue

                    pred_pos = t_pos - 1
                    correct_token = labels[bi, t_pos].item()
                    if correct_token == -100:
                        continue

                    # B positions: tokens between BOS and first SEP
                    b_end = z_s - 1 if z_s > 1 else 1

                    for layer in range(n_layers):
                        attn_pattern = cache["pattern", layer]  # (B, H, seq, seq)

                        # --- 5A: attention weights to z and B ---
                        a_to_z = attn_pattern[bi, :, pred_pos, z_s:z_e].sum(dim=-1)  # (H,)
                        a_to_b = attn_pattern[bi, :, pred_pos, 1:b_end].sum(dim=-1)  # (H,)
                        attn_z[layer] += a_to_z
                        attn_b[layer] += a_to_b

                        # --- 5B: per-head logit advantage via z ---
                        result = cache["result", layer]  # (B, seq, H, d_head)
                        W_O = model.W_O[layer]  # (H, d_head, d_model)

                        for h in range(n_heads):
                            p_z = attn_pattern[bi, h, pred_pos, z_s:z_e].sum().item()
                            head_out = result[bi, pred_pos, h]  # (d_head,)
                            head_d = head_out @ W_O[h]  # (d_model,)
                            logit_contrib = head_d @ W_U  # (d_vocab,)

                            correct_logit = logit_contrib[correct_token].item()
                            mean_logit = logit_contrib.mean().item()
                            head_advantage[layer, h] += (correct_logit - mean_logit)

                        # --- 5D: MLP logit advantage ---
                        mlp_out = cache["mlp_out", layer]  # (B, seq, d_model)
                        mlp_d = mlp_out[bi, pred_pos]
                        mlp_logit = mlp_d @ W_U
                        mlp_advantage[layer] += (mlp_logit[correct_token] - mlp_logit.mean()).item()

                        # --- Attention sublayer total advantage ---
                        attn_out = cache["attn_out", layer]  # (B, seq, d_model)
                        attn_d = attn_out[bi, pred_pos]
                        attn_logit = attn_d @ W_U
                        attn_layer_advantage[layer] += (attn_logit[correct_token] - attn_logit.mean()).item()

                    n_samples += 1

        if n_samples > 0:
            attn_z /= n_samples
            attn_b /= n_samples
            head_advantage /= n_samples
            mlp_advantage /= n_samples
            attn_layer_advantage /= n_samples

        return {
            "attention_to_z": attn_z.cpu().tolist(),
            "attention_to_b": attn_b.cpu().tolist(),
            "head_logit_advantage": head_advantage.cpu().tolist(),
            "mlp_logit_advantage": mlp_advantage.cpu().tolist(),
            "attn_layer_logit_advantage": attn_layer_advantage.cpu().tolist(),
            "n_samples": n_samples,
        }
