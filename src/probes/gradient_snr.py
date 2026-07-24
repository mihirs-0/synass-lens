"""
Per-group gradient SNR probe (Experiment 2, Probe 2A).

Measures the signal-to-noise ratio of the gradient of the loss w.r.t. the
z-token embedding, decomposed by B-group.  This directly tests the
cancellation hypothesis: when z is shared across |B| groups the mean
gradient should be O(1/sqrt(|B|)) while the per-group variance stays O(1),
yielding SNR = O(1/sqrt(|B|)).

The probe is expensive (one forward+backward per example) so it should be
run only at selected checkpoint steps during the plateau.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from .base import BaseProbe
from ..data import MappingData, CharTokenizer


class GradientSNRProbe(BaseProbe):
    """Compute per-group gradient decomposition and SNR for the z-embedding."""

    name = "gradient_snr"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.n_sample_groups: int = config.get("n_sample_groups", 200) if config else 200

    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Not used directly — call :meth:`run_with_mappings` instead."""
        return {}

    def run_with_mappings(
        self,
        model: HookedTransformer,
        tokenizer: CharTokenizer,
        mapping_data: MappingData,
        device: str = "cuda",
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Compute gradient SNR decomposed by group.

        For each sampled group *b* and selector index *k*, computes the
        gradient of the first-target-token loss w.r.t. the z-token
        *embedding* (row of ``model.W_E``).  Aggregates across groups
        to get mean, variance, and SNR per selector.

        Returns dict with ``snr_per_k``, ``snr_mean``, ``mean_norm_per_k``,
        ``var_per_k``.
        """
        import random
        rng = random.Random(seed)

        model.eval()
        k = mapping_data.k
        base_strings = list(mapping_data.mappings.keys())
        n_sample = min(self.n_sample_groups, len(base_strings))
        sampled_bases = rng.sample(base_strings, n_sample)

        # For each selector index, collect per-group gradients
        # g_{b,k} = grad of loss w.r.t. W_E[z_token_id] for example (b, z_k)
        # We use the *embedding* gradient as a tractable proxy for v_z.
        grads_by_k: Dict[int, List[torch.Tensor]] = {ki: [] for ki in range(k)}

        for b_str in sampled_bases:
            entries = mapping_data.mappings[b_str]  # [(z_str, a_str), ...]
            for ki, (z_str, a_str) in enumerate(entries):
                enc = tokenizer.encode_sequence(b_str, z_str, a_str, task=mapping_data.task)
                input_ids = enc["input_ids"].unsqueeze(0).to(device)
                labels = enc["labels"].unsqueeze(0).to(device)
                target_start = enc["target_start_position"]

                model.zero_grad()
                logits = model(input_ids)

                pred_pos = target_start - 1
                if pred_pos < 0:
                    continue
                logits_first = logits[0, pred_pos]
                target_token = labels[0, target_start]
                loss = F.cross_entropy(logits_first.unsqueeze(0), target_token.unsqueeze(0))

                # Gradient w.r.t. the z-token embedding rows
                z_start = enc["z_position"]
                z_end = enc["z_end_position"]
                z_token_ids = input_ids[0, z_start:z_end]

                loss.backward()

                # Collect gradient for the z-token embedding
                W_E_grad = model.W_E.grad  # (vocab, d_model)
                if W_E_grad is None:
                    continue
                g = W_E_grad[z_token_ids].detach().clone().flatten()
                grads_by_k[ki].append(g)

                model.zero_grad()

        # Compute SNR per selector index
        snr_per_k: List[float] = []
        mean_norm_per_k: List[float] = []
        var_per_k: List[float] = []

        for ki in range(k):
            gs = grads_by_k[ki]
            if len(gs) < 2:
                snr_per_k.append(0.0)
                mean_norm_per_k.append(0.0)
                var_per_k.append(0.0)
                continue
            G = torch.stack(gs)  # (n_groups, dim)
            mean_g = G.mean(dim=0)
            diffs = G - mean_g.unsqueeze(0)
            var_scalar = (diffs.norm(dim=1) ** 2).mean().item()
            signal = mean_g.norm().item()
            snr = signal / (var_scalar ** 0.5 + 1e-12)

            snr_per_k.append(snr)
            mean_norm_per_k.append(signal)
            var_per_k.append(var_scalar)

        snr_mean = sum(snr_per_k) / max(len(snr_per_k), 1)

        return {
            "snr_per_k": snr_per_k,
            "snr_mean": snr_mean,
            "mean_norm_per_k": mean_norm_per_k,
            "var_per_k": var_per_k,
            "n_sample_groups": n_sample,
        }
