"""
Isotropy verification probe (Experiments 2 and 7, Probes 2B / 7A-7D).

Analyses the geometry of the unembedding matrix W_U to check the isotropy
assumption from Proposition 2:

* 7A: Across-group cosine similarity of group centroids.
* 7B: Within-group pairwise cosine similarity (group cohesion).
* 7C: Effective rank of the centroid matrix (SVD).
* 7D: Selector-index alignment across groups.
"""

from typing import Any, Dict, List

import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from .base import BaseProbe
from ..data import MappingData, CharTokenizer


class IsotropyProbe(BaseProbe):
    """Verify isotropy of unembedding vectors across and within groups."""

    name = "isotropy"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Not used directly — call :meth:`run_with_mappings`."""
        return {}

    def run_with_mappings(
        self,
        model: HookedTransformer,
        tokenizer: CharTokenizer,
        mapping_data: MappingData,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Run all four isotropy sub-probes.

        Returns a dict with keys ``across_group_cosine``,
        ``within_group_cosine``, ``effective_rank``,
        ``selector_index_cosine``.
        """
        model.eval()

        # W_U shape: (d_model, d_vocab) — columns are decoding directions
        W_U = model.W_U.detach().to(device)  # (d_model, d_vocab)
        d_model = W_U.shape[0]

        # Build group -> list of A-token-ids
        k = mapping_data.k
        group_a_ids: Dict[str, List[int]] = {}
        for b_str, entries in mapping_data.mappings.items():
            ids = []
            for _, a_str in entries:
                a_tokens = tokenizer.encode(a_str)
                ids.append(a_tokens[0])  # first char token id
            group_a_ids[b_str] = ids

        groups = list(group_a_ids.keys())
        n_groups = len(groups)

        # ---- 7A: Across-group cosine similarity of centroids ----
        centroids = []
        for b_str in groups:
            ids = group_a_ids[b_str]
            vecs = W_U[:, ids]  # (d_model, K)
            centroid = vecs.mean(dim=1)
            centroids.append(centroid)
        C = torch.stack(centroids)  # (n_groups, d_model)
        C_normed = Fn.normalize(C, dim=1)

        # Pairwise cosine similarities (upper triangle)
        cos_matrix = C_normed @ C_normed.T
        n = cos_matrix.shape[0]
        mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
        pairwise_cos = cos_matrix[mask]
        across_mean = pairwise_cos.mean().item()
        across_std = pairwise_cos.std().item()

        # Sample for histogram (up to 10000 pairs)
        if pairwise_cos.numel() > 10000:
            indices = torch.randperm(pairwise_cos.numel(), device=device)[:10000]
            across_sample = pairwise_cos[indices].cpu().tolist()
        else:
            across_sample = pairwise_cos.cpu().tolist()

        # ---- 7B: Within-group pairwise cosine similarity ----
        within_means: List[float] = []
        for b_str in groups:
            ids = group_a_ids[b_str]
            if len(ids) < 2:
                continue
            vecs = Fn.normalize(W_U[:, ids].T, dim=1)  # (K, d_model)
            sim = vecs @ vecs.T
            k_local = sim.shape[0]
            wmask = torch.triu(torch.ones(k_local, k_local, device=device, dtype=torch.bool), diagonal=1)
            within_means.append(sim[wmask].mean().item())
        within_group_mean = sum(within_means) / max(len(within_means), 1)
        within_group_std = (
            (sum((w - within_group_mean) ** 2 for w in within_means) / max(len(within_means) - 1, 1)) ** 0.5
            if len(within_means) > 1 else 0.0
        )

        # ---- 7C: Effective rank of centroid matrix ----
        C_centered = C - C.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(C_centered, full_matrices=False)
        eff_rank = (S.sum() ** 2 / (S ** 2).sum()).item()

        # ---- 7D: Selector-index alignment across groups ----
        selector_cosines: List[float] = []
        for ki in range(k):
            ids_for_k = [group_a_ids[b_str][ki] for b_str in groups if ki < len(group_a_ids[b_str])]
            if len(ids_for_k) < 2:
                continue
            vecs = Fn.normalize(W_U[:, ids_for_k].T, dim=1)
            sim = vecs @ vecs.T
            n_k = sim.shape[0]
            smask = torch.triu(torch.ones(n_k, n_k, device=device, dtype=torch.bool), diagonal=1)
            selector_cosines.append(sim[smask].mean().item())
        selector_mean = sum(selector_cosines) / max(len(selector_cosines), 1)

        return {
            "across_group_cosine": {
                "mean": across_mean,
                "std": across_std,
                "sample": across_sample,
            },
            "within_group_cosine": {
                "mean": within_group_mean,
                "std": within_group_std,
            },
            "effective_rank": eff_rank,
            "d_model": d_model,
            "selector_index_cosine": {
                "mean": selector_mean,
                "per_k": selector_cosines,
            },
            "n_groups": n_groups,
            "k": k,
        }
