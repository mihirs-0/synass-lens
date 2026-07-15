#!/usr/bin/env python
"""
Run-2 core: conditioning-hierarchy utilities on top of scripts/relp_core.py.

Adds:
- PairBatch: run-1 EvalBatch (identical construction — same RNG path) augmented
  with per-position answer token ids, so counterfactual metrics exist at every
  A-position, not just position 1.
- Three counterfactual regimes sharing one generator: 'z' (same B, other z —
  run-1 pairs), 'B' (same z, other B), 'both' (both resampled). Each regime
  supplies a baseline over activations AND a per-position logit metric.
- Distribution utilities for E6/E7: per-position model distributions,
  JSD/KL, empirical shelves (q*_j, H(A_j|B,A_<j), H(A_j|A_<j)) computed from
  the actual generated data, never from formulas.

Position conventions (b_length=6, z_length=2, a_length=4):
A tokens sit at positions 11..14, predicted from read positions 10..13.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import generate_mappings
from scripts.relp_core import (
    EvalBatch, build_eval_batch, z_resample_baseline, b_resample_baseline,
    READ_POS, SEQ_LEN,
)

A_LEN = 4
A_POSITIONS = list(range(READ_POS + 1, READ_POS + 1 + A_LEN))   # 11..14
READ_POSITIONS = list(range(READ_POS, READ_POS + A_LEN))        # 10..13


# ---------------------------------------------------------------------------
# PairBatch
# ---------------------------------------------------------------------------

@dataclass
class PairBatch:
    base: EvalBatch
    ans_tids: torch.Tensor       # [n_b, K, A_LEN] answer token ids per position

    def __getattr__(self, name):
        return getattr(self.base, name)


def build_pair_batch(cfg, tokenizer, n_b_eval: int, seed: int = 1234,
                     mapping_data=None) -> PairBatch:
    """Same B-sample RNG path as run 1 (delegates to build_eval_batch), then
    reads answer tokens directly out of the encoded rows."""
    base = build_eval_batch(cfg, tokenizer, n_b_eval=n_b_eval, seed=seed,
                            mapping_data=mapping_data)
    n_b, k = base.n_b, base.k
    ans = base.input_ids[:, A_POSITIONS[0]:A_POSITIONS[-1] + 1].view(n_b, k, A_LEN)
    return PairBatch(base=base, ans_tids=ans.clone())


def to_device(pb: PairBatch, device: str) -> Tuple[torch.Tensor, PairBatch]:
    ids = pb.base.input_ids.to(device)
    for attr in ["b_idx", "z_idx", "cand_tids", "noncand_tids", "correct_tid"]:
        setattr(pb.base, attr, getattr(pb.base, attr).to(device))
    pb.ans_tids = pb.ans_tids.to(device)
    return ids, pb


# ---------------------------------------------------------------------------
# Regime machinery: baselines over activations + per-position logit metrics
# ---------------------------------------------------------------------------

def both_resample_baseline(act: torch.Tensor, n_b: int, k: int) -> torch.Tensor:
    """Mean activation over rows with B' != B and z' != z."""
    shaped = act.view(n_b, k, *act.shape[1:])
    tot = shaped.sum(dim=(0, 1), keepdim=True)
    row = shaped.sum(dim=1, keepdim=True)      # same B, all z
    col = shaped.sum(dim=0, keepdim=True)      # same z, all B
    base = (tot - row - col + shaped) / ((n_b - 1) * (k - 1))
    return base.reshape_as(act)


REGIME_BASELINES: Dict[str, Callable] = {
    "z": z_resample_baseline,
    "B": b_resample_baseline,
    "both": both_resample_baseline,
}


def counterfactual_answer_mean_logits(read_logits_j: torch.Tensor,
                                      ans_j: torch.Tensor,
                                      b_idx: torch.Tensor, z_idx: torch.Tensor,
                                      regime: str) -> torch.Tensor:
    """Mean logit (at this row's read position) of the answers that the
    counterfactual inputs would demand.

    read_logits_j: [N, V] logits at read position of A-position j
    ans_j: [n_b, K] answer token id at position j
    Returns [N].
    """
    n_b, K = ans_j.shape
    N = read_logits_j.shape[0]
    all_l = read_logits_j.gather(1, ans_j.flatten()[None, :].expand(N, -1))  # [N, n_b*K]
    all_l = all_l.view(N, n_b, K)
    rows = torch.arange(N, device=read_logits_j.device)
    self_l = all_l[rows, b_idx, z_idx]                       # [N]
    row_sum = all_l[rows, b_idx, :].sum(dim=1)               # same B, all z
    col_sum = all_l[rows, :, :].gather(
        2, z_idx[:, None, None].expand(N, n_b, 1)).squeeze(2).sum(dim=1)  # same z, all B
    tot = all_l.sum(dim=(1, 2))
    if regime == "z":
        return (row_sum - self_l) / (K - 1)
    if regime == "B":
        return (col_sum - self_l) / (n_b - 1)
    if regime == "both":
        return (tot - row_sum - col_sum + self_l) / ((n_b - 1) * (K - 1))
    raise ValueError(regime)


def regime_metrics(logits: torch.Tensor, pb: PairBatch) -> Dict[str, torch.Tensor]:
    """Per-example metrics for all regimes and A-positions.

    m_{regime}_pos{j} = logit(correct A_j | this input) - mean logit of the
    A_j the counterfactual inputs would demand, read at this input's teacher-
    forced position. m_z_pos1 is exactly run-1 m_diff (regression-tested).
    Returns dict of [N] tensors, plus ce/acc per position.
    """
    out = {}
    b_idx, z_idx = pb.b_idx, pb.z_idx
    N = logits.shape[0]
    rows = torch.arange(N, device=logits.device)
    for j in range(A_LEN):
        rl = logits[:, READ_POSITIONS[j], :]                       # [N, V]
        ans_j = pb.ans_tids[:, :, j]                               # [n_b, K]
        correct = ans_j[b_idx, z_idx]                              # [N]
        lc = rl[rows, correct]
        for regime in ("z", "B", "both"):
            cf = counterfactual_answer_mean_logits(rl, ans_j, b_idx, z_idx, regime)
            out[f"m_{regime}_pos{j+1}"] = lc - cf
        out[f"ce_pos{j+1}"] = F.cross_entropy(rl, correct, reduction="none")
        out[f"acc_pos{j+1}"] = (rl.argmax(-1) == correct).float()
    # pooled (mean over positions)
    for regime in ("z", "B", "both"):
        out[f"m_{regime}_pooled"] = torch.stack(
            [out[f"m_{regime}_pos{j+1}"] for j in range(A_LEN)]).mean(dim=0)
    out["ce_pooled"] = torch.stack([out[f"ce_pos{j+1}"] for j in range(A_LEN)]).mean(dim=0)
    # candidate-restricted CE at position 1 (run-1 continuity metric)
    rl1 = logits[:, READ_POS, :]
    cand = rl1.gather(1, pb.cand_tids[b_idx])                      # [N, K]
    self_c = cand.gather(1, z_idx[:, None]).squeeze(1)
    out["cand_ce_pos1"] = torch.logsumexp(cand, dim=1) - self_c
    return out


# differentiable scalar metric factories for RelP backward passes
def make_metric_fn(regime: str, pos: int = 1):
    def fn(logits, pb):
        return regime_metrics(logits, pb)[f"m_{regime}_pos{pos}"]
    return fn


def register_regime_metrics():
    """Expose regime metrics to run-1's run_relp (METRIC_FNS registry).
    PairBatch delegates attribute access to its EvalBatch, so run-1 code
    accepts it wherever it expects an EvalBatch."""
    from scripts import relp_core
    for regime in ("z", "B", "both"):
        for pos in range(1, A_LEN + 1):
            relp_core.METRIC_FNS[f"m_{regime}_pos{pos}"] = make_metric_fn(regime, pos)


# ---------------------------------------------------------------------------
# Distributions (E6 / E7)
# ---------------------------------------------------------------------------

def position_distributions(logits: torch.Tensor) -> torch.Tensor:
    """Softmax distributions at the 4 read positions: [N, A_LEN, V]."""
    return torch.stack([logits[:, rp, :].softmax(-1) for rp in READ_POSITIONS], dim=1)


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a * (torch.log(a + eps) - torch.log(b + eps))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mean_pairwise_jsd_grid(dists: torch.Tensor, n_b: int, k: int,
                           axis: str, max_pairs: int = 4000,
                           gen: Optional[torch.Generator] = None) -> float:
    """Mean pairwise JSD across inputs, per the requested axis.

    dists: [N, V] (one position). axis='B': pairs differ in B, same z (averaged
    over z). axis='z': pairs differ in z, same B. axis='all': any two inputs.
    Sampled pairs (uniform) capped at max_pairs.
    """
    d = dists.view(n_b, k, -1)
    if gen is None:
        gen = torch.Generator().manual_seed(0)
    vals = []
    for _ in range(max_pairs):
        if axis == "B":
            j = int(torch.randint(k, (1,), generator=gen))
            b1, b2 = torch.randint(n_b, (2,), generator=gen).tolist()
            if b1 == b2:
                continue
            vals.append(jsd(d[b1, j], d[b2, j]))
        elif axis == "z":
            b = int(torch.randint(n_b, (1,), generator=gen))
            j1, j2 = torch.randint(k, (2,), generator=gen).tolist()
            if j1 == j2:
                continue
            vals.append(jsd(d[b, j1], d[b, j2]))
        else:
            b1, b2 = torch.randint(n_b, (2,), generator=gen).tolist()
            j1, j2 = torch.randint(k, (2,), generator=gen).tolist()
            if b1 == b2 and j1 == j2:
                continue
            vals.append(jsd(d[b1, j1], d[b2, j2]))
    return float(torch.stack(vals).mean())


def build_pab(pb: PairBatch, V: int, device: str) -> torch.Tensor:
    """P(A_j | B, A_<j) per row, empirical over the row's B-candidates with
    matching prefix: [N, A_LEN, V]."""
    N = pb.base.input_ids.shape[0]
    K = pb.k
    ans = pb.ans_tids
    pab = torch.zeros(N, A_LEN, V, device=device)
    for i in range(N):
        b = int(pb.b_idx[i]); zi = int(pb.z_idx[i])
        my = ans[b, zi]
        for j in range(A_LEN):
            match = (ans[b, :, :j] == my[:j].unsqueeze(0)).all(dim=1) if j > 0 \
                else torch.ones(K, dtype=torch.bool, device=device)
            idx = ans[b, match, j]
            pab[i, j].scatter_add_(0, idx, torch.full((int(match.sum()),), 1.0, device=device))
        pab[i] = pab[i] / pab[i].sum(-1, keepdim=True)
    return pab


# ---------------------------------------------------------------------------
# Empirical shelves from the actual generated data (ground truth = data code)
# ---------------------------------------------------------------------------

def empirical_shelves(cfg, tokenizer) -> Dict:
    """All hypothesis shelf levels, computed from the full generated mapping
    data (all N x K examples, each appearing once per epoch — the training
    marginal).

    Returns per A-position j (1-indexed):
      q_star[j]: [V] empirical marginal over the token at position j
      H_qstar[j]: entropy of that marginal (constant-machine shelf, no prefix)
      H_given_prefix[j]: H(A_j | A_<j) (constant-in-(B,z) machine that reads
        the teacher-forced prefix)
      H_given_B[j]: H(A_j | B, A_<j) (hypothesis-M shelf)
      log_vocab, log_chars: uniform baselines.
    """
    md = generate_mappings(
        n_unique_b=cfg.data.n_unique_b, k=cfg.data.k,
        b_length=cfg.data.b_length, a_length=cfg.data.a_length,
        z_length=cfg.data.z_length, vocab_chars=cfg.data.vocab_chars,
        seed=cfg.experiment.seed, task=cfg.data.task,
        enforce_unique_a_first_char_per_b=bool(cfg.data.enforce_unique_a_first_char_per_b),
        disambiguation_prefix_length=int(cfg.data.disambiguation_prefix_length),
    )
    V = tokenizer.vocab_size
    a_len = cfg.data.a_length
    tid = lambda ch: tokenizer.token_to_id[ch]

    def entropy(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    q_star, H_qstar, H_prefix, H_B = {}, {}, {}, {}
    examples = [(b, a) for b, pairs in md.mappings.items() for (_z, a) in pairs]
    for j in range(a_len):
        counts = torch.zeros(V)
        for _b, a in examples:
            counts[tid(a[j])] += 1
        q_star[j + 1] = (counts / counts.sum())
        H_qstar[j + 1] = entropy(counts)
        # H(A_j | A_<j)
        groups: Dict[str, torch.Tensor] = {}
        for _b, a in examples:
            groups.setdefault(a[:j], torch.zeros(V))[tid(a[j])] += 1
        tot = sum(g.sum() for g in groups.values())
        H_prefix[j + 1] = float(sum(g.sum() / tot * entropy(g) for g in groups.values()))
        # H(A_j | B, A_<j)
        groupsB: Dict[Tuple[str, str], torch.Tensor] = {}
        for b, a in examples:
            groupsB.setdefault((b, a[:j]), torch.zeros(V))[tid(a[j])] += 1
        H_B[j + 1] = float(sum(g.sum() / tot * entropy(g) for g in groupsB.values()))

    return {
        "q_star": {j: q_star[j].tolist() for j in q_star},
        "H_qstar": H_qstar,
        "H_given_prefix": H_prefix,
        "H_given_B": H_B,
        "log_vocab": math.log(V),
        "log_chars": math.log(len(cfg.data.vocab_chars)),
        "log_K": math.log(cfg.data.k),
        "n_examples": len(examples),
    }
