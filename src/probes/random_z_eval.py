"""
Random-z evaluation probe.

Measures how predictions change when z is replaced with another valid selector
from a different example in the same batch.
"""

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .base import BaseProbe


class RandomZEvalProbe(BaseProbe):
    """
    Probe that measures sensitivity to z by swapping z across examples.
    """

    name = "random_z_eval"

    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "target_prob_drop": float (mean drop in target prob),
                "argmax_change_rate": float (fraction where top-1 changes),
                "n_samples": int
            }
        """
        model.eval()
        total_prob_drop = 0.0
        total_argmax_change = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Random-z eval", leave=False):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                z_positions = batch["z_positions"].to(device)
                z_end_positions = batch["z_end_positions"].to(device)
                target_start_positions = batch["target_start_positions"].to(device)

                batch_size = input_ids.shape[0]
                if batch_size < 2:
                    continue

                # Swap z across examples within the batch
                perm = torch.randperm(batch_size, device=device)
                swapped_input = input_ids.clone()
                for i in range(batch_size):
                    j = perm[i].item()
                    if j == i:
                        j = (j + 1) % batch_size
                    z_start = z_positions[i].item()
                    z_end = z_end_positions[i].item()
                    z_start_j = z_positions[j].item()
                    z_end_j = z_end_positions[j].item()
                    z_len = z_end - z_start
                    z_len_j = z_end_j - z_start_j
                    if z_len != z_len_j:
                        continue
                    swapped_input[i, z_start:z_end] = input_ids[j, z_start_j:z_end_j]

                # Clean and swapped predictions
                clean_logits = model(input_ids)
                swapped_logits = model(swapped_input)

                for i in range(batch_size):
                    target_start = target_start_positions[i].item()
                    target_token = labels[i, target_start].item()
                    if target_token == -100:
                        continue
                    pred_pos = target_start - 1

                    clean_pred = clean_logits[i, pred_pos]
                    swapped_pred = swapped_logits[i, pred_pos]

                    clean_probs = torch.softmax(clean_pred, dim=-1)
                    swapped_probs = torch.softmax(swapped_pred, dim=-1)

                    clean_target_prob = clean_probs[target_token].item()
                    swapped_target_prob = swapped_probs[target_token].item()
                    total_prob_drop += (clean_target_prob - swapped_target_prob)

                    clean_top1 = clean_probs.argmax().item()
                    swapped_top1 = swapped_probs.argmax().item()
                    total_argmax_change += float(clean_top1 != swapped_top1)

                    n_samples += 1

        if n_samples == 0:
            return {
                "target_prob_drop": 0.0,
                "argmax_change_rate": 0.0,
                "n_samples": 0,
                "error": "No valid samples for random-z evaluation",
            }

        return {
            "target_prob_drop": total_prob_drop / n_samples,
            "argmax_change_rate": total_argmax_change / n_samples,
            "n_samples": n_samples,
        }
