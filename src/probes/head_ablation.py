"""
Per-head ablation probe for causal necessity/sufficiency testing.

For each attention head (L, H):
  - Zero ablation (necessity): zero out the head's output and measure loss increase
  - Mean ablation: replace with batch-mean activation
  - Sufficiency: ablate ALL heads except one, check if model still functions

This provides causal evidence for individual head contributions,
complementing the correlational attention-pattern analysis.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .base import BaseProbe


def _compute_first_target_loss(logits, batch):
    """Compute loss on first target token only."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss.item()


def _compute_z_shuffle_gap(model, batch, clean_loss, device):
    """Compute z-shuffle loss gap as a measure of z-usage."""
    input_ids = batch["input_ids"].clone()
    z_pos = batch["z_positions"]
    z_end = batch["z_end_positions"]
    bs = input_ids.shape[0]

    perm = torch.randperm(bs, device=device)
    for i in range(bs):
        if perm[i] == i:
            j = (i + 1) % bs
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()

    z_tokens = [input_ids[i, z_pos[i]:z_end[i]].clone() for i in range(bs)]
    for i in range(bs):
        input_ids[i, z_pos[i]:z_end[i]] = z_tokens[perm[i].item()]

    shuf_batch = {**batch, "input_ids": input_ids}
    logits = model(input_ids)
    shuf_loss = _compute_first_target_loss(logits, shuf_batch)
    return shuf_loss - clean_loss


class HeadAblationProbe(BaseProbe):
    """
    Probe that measures causal contribution of each attention head
    via zero and mean ablation at the attention output hook.
    """

    name = "head_ablation"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.n_batches = config.get("n_batches", 8) if config else 8
        self.ablation_types = config.get("ablation_types", ["zero", "mean"]) if config else ["zero", "mean"]

    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Run per-head ablation analysis.

        Returns dict with keys:
            clean_loss: float - baseline loss
            head_effects: dict mapping "{ablation_type}" to
                list of {layer, head, loss_delta, z_gap_delta} dicts
            sufficiency: list of {layer, head, loss_with_only_this_head}
        """
        model.eval()
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        # Collect batches
        batches = []
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batches.append(batch)
            if len(batches) >= self.n_batches:
                break

        if not batches:
            return {"error": "No data available"}

        # Compute clean baseline
        clean_losses = []
        clean_z_gaps = []
        with torch.no_grad():
            for batch in batches:
                logits = model(batch["input_ids"])
                cl = _compute_first_target_loss(logits, batch)
                clean_losses.append(cl)
                zg = _compute_z_shuffle_gap(model, batch, cl, device)
                clean_z_gaps.append(zg)

        clean_loss = sum(clean_losses) / len(clean_losses)
        clean_z_gap = sum(clean_z_gaps) / len(clean_z_gaps)

        # Compute mean activations per head for mean-ablation
        # Shape of hook_result: (batch, seq, n_heads, d_head)
        mean_activations = {}
        if "mean" in self.ablation_types:
            head_sums = {}
            head_counts = {}
            with torch.no_grad():
                for batch in batches:
                    _, cache = model.run_with_cache(batch["input_ids"])
                    for layer in range(n_layers):
                        key = f"blocks.{layer}.attn.hook_z"
                        act = cache[key]  # (batch, seq, n_heads, d_head)
                        for head in range(n_heads):
                            hkey = (layer, head)
                            head_act = act[:, :, head, :]  # (batch, seq, d_head)
                            if hkey not in head_sums:
                                head_sums[hkey] = torch.zeros_like(head_act[0].mean(dim=0))
                                head_counts[hkey] = 0
                            head_sums[hkey] += head_act.mean(dim=(0, 1))
                            head_counts[hkey] += 1
                    del cache

            for hkey in head_sums:
                mean_activations[hkey] = head_sums[hkey] / head_counts[hkey]

        # Per-head ablation
        results_by_type = {}
        for abl_type in self.ablation_types:
            head_effects = []
            for layer in range(n_layers):
                for head in range(n_heads):
                    ablated_losses = []
                    ablated_z_gaps = []

                    with torch.no_grad():
                        for batch in batches:
                            def make_hook(l, h, atype):
                                def hook_fn(activation, hook):
                                    # activation: (batch, seq, n_heads, d_head)
                                    if atype == "zero":
                                        activation[:, :, h, :] = 0.0
                                    elif atype == "mean":
                                        activation[:, :, h, :] = mean_activations[(l, h)]
                                    return activation
                                return hook_fn

                            hook_name = f"blocks.{layer}.attn.hook_z"
                            logits = model.run_with_hooks(
                                batch["input_ids"],
                                fwd_hooks=[(hook_name, make_hook(layer, head, abl_type))],
                            )
                            al = _compute_first_target_loss(logits, batch)
                            ablated_losses.append(al)

                            # z-gap under ablation
                            zg = _compute_z_shuffle_gap(
                                model, batch, al, device
                            )
                            ablated_z_gaps.append(zg)

                    avg_loss = sum(ablated_losses) / len(ablated_losses)
                    avg_z_gap = sum(ablated_z_gaps) / len(ablated_z_gaps)

                    head_effects.append({
                        "layer": layer,
                        "head": head,
                        "ablated_loss": avg_loss,
                        "loss_delta": avg_loss - clean_loss,
                        "ablated_z_gap": avg_z_gap,
                        "z_gap_delta": avg_z_gap - clean_z_gap,
                    })

            results_by_type[abl_type] = head_effects

        # Sufficiency test: ablate ALL heads except one
        sufficiency = []
        with torch.no_grad():
            for layer in range(n_layers):
                for head in range(n_heads):
                    suff_losses = []
                    for batch in batches:
                        hooks = []
                        for l2 in range(n_layers):
                            for h2 in range(n_heads):
                                if l2 == layer and h2 == head:
                                    continue  # keep this one
                                def make_zero_hook(h_idx):
                                    def hook_fn(activation, hook):
                                        activation[:, :, h_idx, :] = 0.0
                                        return activation
                                    return hook_fn
                                hooks.append(
                                    (f"blocks.{l2}.attn.hook_z",
                                     make_zero_hook(h2))
                                )

                        logits = model.run_with_hooks(
                            batch["input_ids"],
                            fwd_hooks=hooks,
                        )
                        sl = _compute_first_target_loss(logits, batch)
                        suff_losses.append(sl)

                    sufficiency.append({
                        "layer": layer,
                        "head": head,
                        "loss_only_this_head": sum(suff_losses) / len(suff_losses),
                    })

        return {
            "clean_loss": clean_loss,
            "clean_z_gap": clean_z_gap,
            "head_effects": results_by_type,
            "sufficiency": sufficiency,
            "n_layers": n_layers,
            "n_heads": n_heads,
        }
