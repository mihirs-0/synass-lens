#!/usr/bin/env python
"""
RelP (relevance patching) circuit attribution for MBC checkpoints.

Ports the method of "Language Model Circuits Are Sparse in the Neuron Basis"
(arXiv:2601.22594) onto the synass-lens training checkpoints:

- Nodes: MLP activations at ``blocks.{l}.mlp.hook_post`` (post-nonlinearity,
  pre-down-projection — the privileged basis), plus input embeddings and
  per-head attention outputs (``attn.hook_z``) for routing analysis.
- Attribution: one backward pass through a locally linearized model:
  * GELU frozen to a constant elementwise multiplier c = gelu(x)/x
    determined by the actual forward activations,
  * attention patterns detached (softmax frozen + Q/K stop-grad; gradient
    flows through the value/OV path only),
  * LayerNorm scale (std) detached.
  The MLP here is not gated, so the halving rule for elementwise products
  does not apply.
- Baselines: exact counterfactual pairs from the task. For (B, z_i) the
  z-resampled baseline is the mean activation over (B, z_j), j != i; the
  B-resampled baseline is the mean over (B', z_i), B' != B.

All functions assume the landauer_dense-style config: fixed-length sequences
<BOS> B(6) <SEP> z(2) <SEP> A(4) <EOS> (16 tokens), first-target-token
metrics read from position 10.
"""

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import generate_mappings
from src.model import create_model_from_config
from src.training.checkpoint import load_checkpoint
from scripts.experiment_helpers import make_config

# Sequence layout for b_length=6, z_length=2, a_length=4
POS_BOS = 0
POS_B = list(range(1, 7))
POS_SEP1 = 7
POS_Z = [8, 9]
POS_SEP2 = 10
READ_POS = 10  # position whose logits predict the first target token (pos 11)
SEQ_LEN = 16

POSITION_GROUPS = {
    "bos": [POS_BOS],
    "B": POS_B,
    "sep1": [POS_SEP1],
    "z": POS_Z,
    "readout": [POS_SEP2],  # last SEP; the position the prediction is read from
}


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Eval batch: all K z's per sampled B, row order b-major (row = b_i * K + j)
# ---------------------------------------------------------------------------

@dataclass
class EvalBatch:
    input_ids: torch.Tensor      # [N, 16] with N = n_b * K
    b_idx: torch.Tensor          # [N] index into sampled B list
    z_idx: torch.Tensor          # [N] z slot 0..K-1 (shared z ordering)
    cand_tids: torch.Tensor      # [n_b, K] token id of first char of A(B, z_j)
    noncand_tids: torch.Tensor   # [n_b, V_char - K] first chars NOT in B's candidate set
    correct_tid: torch.Tensor    # [N]
    n_b: int
    k: int
    b_strings: List[str]
    z_strings: List[str]


def build_eval_batch(cfg, tokenizer, n_b_eval: int, seed: int = 1234,
                     mapping_data=None) -> EvalBatch:
    md = mapping_data if mapping_data is not None else generate_mappings(
        n_unique_b=cfg.data.n_unique_b,
        k=cfg.data.k,
        b_length=cfg.data.b_length,
        a_length=cfg.data.a_length,
        z_length=cfg.data.z_length,
        vocab_chars=cfg.data.vocab_chars,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
        enforce_unique_a_first_char_per_b=bool(cfg.data.enforce_unique_a_first_char_per_b),
        disambiguation_prefix_length=int(cfg.data.disambiguation_prefix_length),
    )
    k = md.k
    all_b = sorted(md.mappings.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_b), generator=g).tolist()
    b_strings = [all_b[i] for i in perm[:n_b_eval]]

    # Shared z ordering: use the z list of the first B (z_sharing="shared")
    z_strings = [z for (z, _a) in md.mappings[b_strings[0]]]
    z_slot = {z: j for j, z in enumerate(z_strings)}

    char_tids = torch.tensor(
        [tokenizer.token_to_id[c] for c in cfg.data.vocab_chars], dtype=torch.long
    )

    rows, b_idx, z_idx, correct = [], [], [], []
    cand_tids = torch.zeros(n_b_eval, k, dtype=torch.long)
    noncand = []
    for bi, b in enumerate(b_strings):
        pairs = md.mappings[b]
        assert len(pairs) == k
        slot_a = {}
        for (z, a) in pairs:
            slot_a[z_slot[z]] = a
        cand_set = set()
        for j in range(k):
            a = slot_a[j]
            tid = tokenizer.token_to_id[a[0]]
            cand_tids[bi, j] = tid
            cand_set.add(tid)
        nc = [t for t in char_tids.tolist() if t not in cand_set]
        noncand.append(nc)
        for j in range(k):
            a = slot_a[j]
            enc = tokenizer.encode_sequence(b, z_strings[j], a, task=cfg.data.task)
            assert len(enc["input_ids"]) == SEQ_LEN
            assert enc["z_position"] == POS_Z[0]
            assert enc["target_start_position"] == READ_POS + 1
            rows.append(enc["input_ids"])
            b_idx.append(bi)
            z_idx.append(j)
            correct.append(cand_tids[bi, j].item())

    # noncand sets can vary in size when first chars are not unique per B
    # (e.g. K=100 with 2-char disambiguation prefixes); truncate to the common
    # minimum. K=10 unique-first-char runs are unaffected (constant 26).
    min_nc = min(len(nc) for nc in noncand)
    if min_nc == 0:
        noncand_t = torch.full((n_b_eval, 1), tokenizer.pad_token_id, dtype=torch.long)
        print("[WARN] some B has no non-candidate first chars; m_marg is invalid for this run")
    else:
        noncand_t = torch.stack([torch.tensor(nc[:min_nc], dtype=torch.long) for nc in noncand])

    return EvalBatch(
        input_ids=torch.stack(rows),
        b_idx=torch.tensor(b_idx, dtype=torch.long),
        z_idx=torch.tensor(z_idx, dtype=torch.long),
        cand_tids=cand_tids,
        noncand_tids=noncand_t,
        correct_tid=torch.tensor(correct, dtype=torch.long),
        n_b=n_b_eval,
        k=k,
        b_strings=b_strings,
        z_strings=z_strings,
    )


# ---------------------------------------------------------------------------
# Metrics (per-example, differentiable)
# ---------------------------------------------------------------------------

def metric_values(logits: torch.Tensor, batch: EvalBatch) -> Dict[str, torch.Tensor]:
    """Per-example metrics from full logits [N, seq, V].

    m_diff  = logit(correct A first char) - mean over other candidates
    m_marg  = mean over B's K candidates - mean over non-candidate chars
    m_plain = logit(correct A first char)
    ce_first, acc_first: bookkeeping (non-diff path fine).
    """
    read = logits[:, READ_POS, :]                                    # [N, V]
    cand = read.gather(1, batch.cand_tids[batch.b_idx])              # [N, K]
    lc = cand.gather(1, batch.z_idx[:, None]).squeeze(1)             # [N]
    k = batch.k
    other_mean = (cand.sum(dim=1) - lc) / (k - 1)
    m_diff = lc - other_mean
    noncand = read.gather(1, batch.noncand_tids[batch.b_idx])        # [N, V_char-K]
    m_marg = cand.mean(dim=1) - noncand.mean(dim=1)
    ce = F.cross_entropy(read, batch.correct_tid, reduction="none")
    acc = (read.argmax(dim=-1) == batch.correct_tid).float()
    return {"m_diff": m_diff, "m_marg": m_marg, "m_plain": lc, "ce_first": ce, "acc_first": acc}


METRIC_FNS: Dict[str, Callable] = {
    "m_diff": lambda logits, batch: metric_values(logits, batch)["m_diff"],
    "m_marg": lambda logits, batch: metric_values(logits, batch)["m_marg"],
    "m_plain": lambda logits, batch: metric_values(logits, batch)["m_plain"],
}


# ---------------------------------------------------------------------------
# Linearization (RelP backward rules)
# ---------------------------------------------------------------------------

def make_frozen_act(orig_act: Callable, zero_slope: float) -> Callable:
    """Elementwise nonlinearity y=f(x) -> y = c*x with c = (f(x)/x).detach().

    Forward value is unchanged (up to fp rounding); backward treats c as a
    constant multiplier. ``zero_slope`` = f'(0), used where |x| ~ 0.
    """
    def frozen(x: torch.Tensor) -> torch.Tensor:
        y = orig_act(x)
        safe = torch.where(x.abs() > 1e-6, x, torch.ones_like(x))
        c = torch.where(x.abs() > 1e-6, y / safe, torch.full_like(x, zero_slope))
        return c.detach() * x
    return frozen


def _detach_hook(t, hook):
    return t.detach()


@contextmanager
def linearized(model):
    """RelP linearization: frozen GELU, frozen attention pattern, frozen LN scale."""
    n_layers = model.cfg.n_layers
    orig_acts = []
    for l in range(n_layers):
        mlp = model.blocks[l].mlp
        orig_acts.append(mlp.act_fn)
        mlp.act_fn = make_frozen_act(mlp.act_fn, zero_slope=0.5)  # gelu'(0) = 0.5
    freeze_hooks = [(f"blocks.{l}.attn.hook_pattern", _detach_hook) for l in range(n_layers)]
    freeze_hooks += [(f"blocks.{l}.ln1.hook_scale", _detach_hook) for l in range(n_layers)]
    freeze_hooks += [(f"blocks.{l}.ln2.hook_scale", _detach_hook) for l in range(n_layers)]
    freeze_hooks += [("ln_final.hook_scale", _detach_hook)]
    try:
        with model.hooks(fwd_hooks=freeze_hooks):
            yield
    finally:
        for l in range(n_layers):
            model.blocks[l].mlp.act_fn = orig_acts[l]


# ---------------------------------------------------------------------------
# RelP run: forward (linearized) + one backward; collect acts and grads
# ---------------------------------------------------------------------------

def run_relp(
    model,
    input_ids: torch.Tensor,
    batch: EvalBatch,
    metric: str,
    linearize: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """Returns (acts, grads, per-example metric values).

    acts/grads keyed by 'mlp_post_{l}' [N, seq, d_mlp], 'attn_z_{l}'
    [N, seq, n_heads, d_head], 'embed' [N, seq, d_model]. grads are gradients
    of sum_i m_i, i.e. per-example gradients (examples are independent).
    """
    n_layers = model.cfg.n_layers
    names = {f"blocks.{l}.mlp.hook_post": f"mlp_post_{l}" for l in range(n_layers)}
    names.update({f"blocks.{l}.attn.hook_z": f"attn_z_{l}" for l in range(n_layers)})
    names["hook_embed"] = "embed"

    cache: Dict[str, torch.Tensor] = {}

    def save(t, hook):
        t.retain_grad()
        cache[names[hook.name]] = t
        return t

    save_hooks = [(name, save) for name in names]

    def _forward():
        with model.hooks(fwd_hooks=save_hooks):
            return model(input_ids)

    model.zero_grad(set_to_none=True)
    if linearize:
        with linearized(model):
            logits = _forward()
    else:
        logits = _forward()

    m = METRIC_FNS[metric](logits, batch)
    m.sum().backward()

    acts = {k: v.detach().clone() for k, v in cache.items()}
    grads = {k: (v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v))
             for k, v in cache.items()}
    model.zero_grad(set_to_none=True)
    return acts, grads, m.detach()


# ---------------------------------------------------------------------------
# Counterfactual baselines from the same batch (b-major layout)
# ---------------------------------------------------------------------------

def z_resample_baseline(act: torch.Tensor, n_b: int, k: int) -> torch.Tensor:
    """Per row (B, z_i): mean act over (B, z_j), j != i. act: [N, ...]."""
    shaped = act.view(n_b, k, *act.shape[1:])
    tot = shaped.sum(dim=1, keepdim=True)
    base = (tot - shaped) / (k - 1)
    return base.reshape_as(act)


def b_resample_baseline(act: torch.Tensor, n_b: int, k: int) -> torch.Tensor:
    """Per row (B, z_i): mean act over (B', z_i), B' != B."""
    shaped = act.view(n_b, k, *act.shape[1:])
    tot = shaped.sum(dim=0, keepdim=True)
    base = (tot - shaped) / (n_b - 1)
    return base.reshape_as(act)


BASELINE_FNS = {"z_resample": z_resample_baseline, "b_resample": b_resample_baseline}


# ---------------------------------------------------------------------------
# Ablation runner: patch selected mlp.hook_post entries to baseline values
# ---------------------------------------------------------------------------

def run_with_mlp_patch(
    model,
    input_ids: torch.Tensor,
    batch: EvalBatch,
    masks: Dict[int, torch.Tensor],       # layer -> bool [d_mlp] or [seq, d_mlp]; True = ablate
    baselines: Dict[int, torch.Tensor],   # layer -> [N, seq, d_mlp]
) -> Dict[str, torch.Tensor]:
    hooks = []
    for l, mask in masks.items():
        if mask is None or not bool(mask.any()):
            continue
        m = mask
        if m.dim() == 1:
            m = m[None, None, :]          # [d_mlp] -> all examples, all positions
        elif m.dim() == 2:
            m = m[None, :, :]             # [seq, d_mlp]
        elif m.dim() == 3 and m.shape[1] == 1:
            pass                          # [N, 1, d_mlp]: per-example neuron mask
        base = baselines[l]

        def make_hook(m=m, base=base):
            def h(post, hook):
                return torch.where(m, base, post)
            return h
        hooks.append((f"blocks.{l}.mlp.hook_post", make_hook()))

    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            logits = model(input_ids)
        return metric_values(logits, batch)


# ---------------------------------------------------------------------------
# Integrated Gradients comparator (same node set, raw gradients, no freezing)
# ---------------------------------------------------------------------------

def ig_attribution(
    model,
    input_ids: torch.Tensor,
    batch: EvalBatch,
    metric: str,
    clean_acts: Dict[str, torch.Tensor],
    base_acts: Dict[str, torch.Tensor],
    steps: int = 32,
) -> Dict[str, torch.Tensor]:
    """IG on the joint mlp.hook_post activation vector, straight-line path from
    the counterfactual baseline to the clean activations. All layers patched
    simultaneously; gradient of each node holds the other patched nodes fixed."""
    n_layers = model.cfg.n_layers
    total = {f"mlp_post_{l}": torch.zeros_like(clean_acts[f"mlp_post_{l}"]) for l in range(n_layers)}
    for s in range(steps):
        alpha = (s + 0.5) / steps
        interp = {}
        hooks = []
        for l in range(n_layers):
            key = f"mlp_post_{l}"
            t = (base_acts[key] + alpha * (clean_acts[key] - base_acts[key])).clone().requires_grad_(True)
            interp[key] = t

            def make_hook(t=t):
                def h(post, hook):
                    return t
                return h
            hooks.append((f"blocks.{l}.mlp.hook_post", make_hook()))
        model.zero_grad(set_to_none=True)
        with model.hooks(fwd_hooks=hooks):
            logits = model(input_ids)
        m = METRIC_FNS[metric](logits, batch)
        m.sum().backward()
        for key, t in interp.items():
            if t.grad is not None:
                total[key] += t.grad.detach()
    model.zero_grad(set_to_none=True)
    return {key: (clean_acts[key] - base_acts[key]) * (total[key] / steps) for key in total}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg, tokenizer, checkpoint_dir: Path, step: int, device: str):
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(device)
    load_checkpoint(model, None, checkpoint_dir, step=step)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def selftest_frozen_act(device: str = "cpu") -> Dict[str, float]:
    """Frozen GELU forward must equal plain GELU, incl. near zero; backward
    must be the constant multiplier gelu(x)/x."""
    x = torch.cat([
        torch.randn(1000) * 3,
        torch.linspace(-1e-5, 1e-5, 100),
    ]).to(device).requires_grad_(True)
    frozen = make_frozen_act(F.gelu, zero_slope=0.5)
    y = frozen(x)
    fwd_err = (y - F.gelu(x.detach())).abs().max().item()
    y.sum().backward()
    expected = torch.where(
        x.detach().abs() > 1e-6,
        F.gelu(x.detach()) / torch.where(x.detach().abs() > 1e-6, x.detach(), torch.ones_like(x)),
        torch.full_like(x.detach(), 0.5),
    )
    bwd_err = (x.grad - expected).abs().max().item()
    return {"fwd_err": fwd_err, "bwd_err": bwd_err}


def selftest_linear_toy(device: str = "cpu") -> Dict[str, float]:
    """Purely linear toy: metric = v . W2 @ act(W1 x) with act=identity.
    RelP relevance (grad x act) through the frozen wrapper must equal the
    analytic relevance (W2^T v) * h exactly."""
    torch.manual_seed(0)
    W1 = torch.randn(32, 16, device=device)
    W2 = torch.randn(8, 32, device=device)
    v = torch.randn(8, device=device)
    x = torch.randn(16, device=device, requires_grad=True)
    frozen_id = make_frozen_act(lambda t: t, zero_slope=1.0)
    h = frozen_id(W1 @ x)
    h.retain_grad()
    m = v @ (W2 @ h)
    m.backward()
    relevance = h.grad * h.detach()
    analytic = (W2.T @ v) * (W1 @ x)
    err = (relevance - analytic).abs().max().item()
    cons_err = abs(relevance.sum().item() - m.item())  # conservation: no bias terms
    return {"toy_err": err, "toy_conservation_err": cons_err}


def selftest_frozen_forward(model, input_ids: torch.Tensor) -> Dict[str, float]:
    """Linearization must not change the forward pass."""
    with torch.no_grad():
        clean = model(input_ids)
    with linearized(model):
        with torch.no_grad():
            frozen = model(input_ids)
    return {"frozen_forward_max_abs_diff": (clean - frozen).abs().max().item()}
