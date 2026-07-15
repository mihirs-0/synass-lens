"""
Stress test: does the early-plateau z-probe direction at target_start
align with the later causal-use direction?

Five analyses:
    1. Probe-subspace stability across phases (principal-angle cosines).
    2. Probe direction vs causal-difference direction (projection
       fractions).
    3. Amnesic erasure: project the probe subspace out of resid_post at
       layer L at target_start, measure candidate-set CE / delta_z; compare
       to a matched random-subspace control.
    4. Random-label probe baseline (shuffled labels): isolate probe-
       capacity contribution to the early decodability signal.
    5. Attention-fast vs MLP-slow: ablate L0 attention output (or L0 MLP
       output) at target_start, retrain probe on the ablated residuals
       and compare to baseline probe accuracy.

Cells: A, D primary; B, C secondary.

Layer convention: L=2 (the layer that maximized target_start CE recovery
in the position patching pass for A and D). All probes and erasures act
at `blocks.2.hook_resid_post`, position = target_start.

Outputs:
    eta_sweep/results/mech_interp_probe_causal_alignment.md
    eta_sweep/results/mech_interp_probe_causal_alignment.json
    eta_sweep/results/figures/mech_interp_probe_causal_alignment.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

ETA_SWEEP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import RESULTS_DIR  # noqa: E402
from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    CELLS, load_cell, make_model, select_phase_ckpts, _select_device,
)
from eta_sweep.analysis.mech_interp_patching import (  # noqa: E402
    _build_candidate_tensors, _build_pairs,
)


PROBE_LAYER = 2  # layer at which we read residuals for the probe.
HOOK_NAME = f"blocks.{PROBE_LAYER}.hook_resid_post"

# Position used for the linear z probe.
#
# The original "0.86 z-decodability at target_start" finding sat at the
# position whose residual produces the first-A logit, i.e. target_start − 1
# (the SEP token between z and A). In the standard transformer formulation,
# logits at position p predict the token at p+1, so the slot that emits
# A[0] is target_start − 1. We use this position for the probe so the
# stress test is aligned with the original claim's position. Causal
# patching is separately reported at both target_start and target_start − 1.
PROBE_POSITION_OFFSET = -1  # use position target_start + offset = ts - 1


# ---------------------------------------------------------------------------
# Residual collection
# ---------------------------------------------------------------------------

def collect_target_start_residuals(model, tokenizer, mapping_data,
                                   n_examples, device, ablation=None,
                                   seed=0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """For ~n_examples (B, z, A) examples, return:
        X: (N, d) target_start residuals at PROBE_LAYER's hook_resid_post
        y: (N,) global z-id (one of the K shared selectors)
        z_strings: list of K unique z strings, sorted
    `ablation` ∈ {None, 'l0_attn_zero', 'l0_mlp_zero'} optionally zeros
    the named L0 sublayer at target_start during the forward pass.
    """
    rng = random.Random(seed)
    z_pool = sorted({z for ms in mapping_data.mappings.values()
                     for z, _ in ms})
    z_to_id = {z: i for i, z in enumerate(z_pool)}

    bases = list(mapping_data.mappings.keys())
    rng.shuffle(bases)
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    n_per_base = max(1, n_examples // max(1, len(bases)) + 1)
    collected = 0
    for b in bases:
        ms = mapping_data.mappings[b]
        idxs = rng.sample(range(len(ms)), min(n_per_base, len(ms)))
        for i in idxs:
            z, a = ms[i]
            enc = tokenizer.encode_sequence(b, z, a, task="bz_to_a")
            ids = enc["input_ids"].unsqueeze(0).to(device)
            ts = int(enc["target_start_position"])
            probe_pos = ts + PROBE_POSITION_OFFSET

            if ablation == "l0_attn_zero":
                def hf_attn_zero(act, hook):
                    # zero L0 attention output contribution at probe_pos
                    act[:, probe_pos, :] = 0.0
                    return act
                hooks = [("blocks.0.hook_attn_out", hf_attn_zero)]
            elif ablation == "l0_mlp_zero":
                def hf_mlp_zero(act, hook):
                    act[:, probe_pos, :] = 0.0
                    return act
                hooks = [("blocks.0.hook_mlp_out", hf_mlp_zero)]
            else:
                hooks = []

            with torch.no_grad():
                if hooks:
                    cap = {}

                    def cache_hook(act, hook, cap_=cap):
                        cap_["resid"] = act.detach().clone()
                        return act
                    full_hooks = list(hooks) + [(HOOK_NAME, cache_hook)]
                    model.run_with_hooks(ids, fwd_hooks=full_hooks)
                    resid = cap["resid"][0, probe_pos].cpu().numpy()
                else:
                    _, cache = model.run_with_cache(ids)
                    resid = cache[("resid_post", PROBE_LAYER)][
                        0, probe_pos].cpu().numpy()

            X_rows.append(resid)
            y_rows.append(z_to_id[z])
            collected += 1
            if collected >= n_examples:
                break
        if collected >= n_examples:
            break

    return np.array(X_rows), np.array(y_rows, dtype=np.int64), z_pool


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_z_probe(X, y, n_train=256, n_test=128, seed=0, C=0.1,
                  shuffle_labels=False):
    """Returns dict with W (K, d), train_acc, test_acc, classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    if shuffle_labels:
        y = rng.permutation(y)
    perm = rng.permutation(X.shape[0])
    X = X[perm]; y = y[perm]
    n_train = min(n_train, X.shape[0] - 1)
    n_test = min(n_test, X.shape[0] - n_train)
    Xtr, Xte = X[:n_train], X[n_train:n_train + n_test]
    ytr, yte = y[:n_train], y[n_train:n_train + n_test]
    if len(np.unique(ytr)) < 2:
        return None
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500, C=C, solver="lbfgs",
                             n_jobs=1).fit(scaler.transform(Xtr), ytr)
    tr = float(clf.score(scaler.transform(Xtr), ytr))
    te = float(clf.score(scaler.transform(Xte), yte)) if n_test > 0 else float("nan")
    # Probe weights live in the standardized space; map back to raw
    # residual space by W_raw = W_std / std (per feature). For subspace
    # comparison we want directions in the residual space.
    W_std = clf.coef_  # (K_seen, d)
    # account for StandardScaler: standardized x = (x - mean) / std
    # so probe direction in raw space = W_std / std (broadcasted over features)
    std = scaler.scale_  # (d,)
    W_raw = W_std / std[None, :]
    # Center the rows so the subspace is {differences between class means}
    W_raw = W_raw - W_raw.mean(axis=0, keepdims=True)
    return {
        "W_raw": W_raw,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "train_acc": tr,
        "test_acc": te,
        "classes": clf.classes_.tolist(),
    }


# ---------------------------------------------------------------------------
# Subspace utilities
# ---------------------------------------------------------------------------

def topk_basis(W: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Top-k right singular directions of W (rows = vectors); shape (k, d)."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    if k is None:
        # keep components capturing 95% of variance
        cum = np.cumsum(s ** 2)
        total = cum[-1]
        k = int(np.searchsorted(cum, 0.95 * total) + 1)
    return Vt[:k], s[:k]


def principal_angle_cosines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (k, d), B: (m, d) — orthonormal rows. Returns cosines of
    principal angles, sorted descending."""
    # ensure orthonormal
    Qa, _ = np.linalg.qr(A.T)  # (d, k)
    Qb, _ = np.linalg.qr(B.T)  # (d, m)
    M = Qa.T @ Qb  # (k, m)
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, 0, 1)


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> Dict:
    """Pair of subspaces given by row-vectors. Returns mean / min principal-
    angle cosines and the full vector."""
    cosines = principal_angle_cosines(A, B)
    return {
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
        "cosines": cosines.tolist(),
    }


def projection_fraction(direction_set: np.ndarray,
                        subspace_basis: np.ndarray) -> float:
    """For each row v of `direction_set`, ||proj_subspace(v)||^2 / ||v||^2.
    Returns mean fraction across rows."""
    Q, _ = np.linalg.qr(subspace_basis.T)  # (d, k)
    P = Q @ Q.T  # projector onto subspace
    fracs = []
    for v in direction_set:
        n2 = float(np.dot(v, v))
        if n2 < 1e-12:
            continue
        pv = P @ v
        fracs.append(float(np.dot(pv, pv) / n2))
    return float(np.mean(fracs)) if fracs else float("nan")


# ---------------------------------------------------------------------------
# Causal-direction estimation
# ---------------------------------------------------------------------------

def causal_direction_set(model, tokenizer, mapping_data, n_pairs, device,
                         seed=0):
    """For ~n_pairs (B, z_clean, z_corrupt) triples, compute
    `resid_post[L][probe_pos]_clean - resid_post[L][probe_pos]_corrupt`
    AT BOTH probe_pos (ts + offset, the SEP slot used by the probe) AND
    target_start (ts, the slot the causal patching analysis localised).

    Returns dict with two (n_pairs, d) arrays:
        causal_at_probe_pos — same position as the probe
        causal_at_target_start — the slot patching identified as causal
    """
    pairs = _build_pairs(mapping_data, n_pairs, seed)
    rows_probe, rows_ts = [], []
    for p in pairs:
        enc_clean = tokenizer.encode_sequence(
            p["base"], p["z_clean"], p["a_clean"], task="bz_to_a")
        enc_corr = tokenizer.encode_sequence(
            p["base"], p["z_corrupt"], p["a_corrupt"], task="bz_to_a")
        ts = int(enc_clean["target_start_position"])
        probe_pos = ts + PROBE_POSITION_OFFSET
        ids_c = enc_clean["input_ids"].unsqueeze(0).to(device)
        ids_k = enc_corr["input_ids"].unsqueeze(0).to(device)
        with torch.no_grad():
            _, cache_c = model.run_with_cache(ids_c)
            _, cache_k = model.run_with_cache(ids_k)
        rc_p = cache_c[("resid_post", PROBE_LAYER)][0, probe_pos].cpu().numpy()
        rk_p = cache_k[("resid_post", PROBE_LAYER)][0, probe_pos].cpu().numpy()
        rc_t = cache_c[("resid_post", PROBE_LAYER)][0, ts].cpu().numpy()
        rk_t = cache_k[("resid_post", PROBE_LAYER)][0, ts].cpu().numpy()
        rows_probe.append(rc_p - rk_p)
        rows_ts.append(rc_t - rk_t)
    return {
        "at_probe_pos": np.array(rows_probe),
        "at_target_start": np.array(rows_ts),
    }


# ---------------------------------------------------------------------------
# Amnesic erasure
# ---------------------------------------------------------------------------

def candidate_loss_with_subspace_erasure(model, tokenizer, mapping_data,
                                         subspace_basis, device,
                                         n_pairs=24, seed=0):
    """Run candidate-set CE on (B, z_clean) targets while *projecting out*
    the given subspace from resid_post[PROBE_LAYER] at target_start.

    Also computes a delta_z proxy: candidate-set CE under z-shuffle, with
    the same erasure applied.

    Returns:
        L_clean_erased, L_shuf_erased, candidate_acc_clean_erased
    """
    Q = None
    if subspace_basis is not None and subspace_basis.size > 0:
        Q_, _ = np.linalg.qr(subspace_basis.T)  # (d, k)
        Q = torch.from_numpy(Q_.astype(np.float32)).to(device)

    pairs = _build_pairs(mapping_data, n_pairs, seed)
    losses_clean, losses_shuf, accs_clean = [], [], []

    def make_erasure_hook(ts):
        # Erase the probe subspace from both probe_pos (ts + offset) and
        # target_start (ts). probe_pos is where the probe was trained;
        # target_start is where causal patching localised the work.
        probe_pos = ts + PROBE_POSITION_OFFSET

        def hf(act, hook, ts_=ts, pp_=probe_pos, Q_=Q):
            if Q_ is None:
                return act
            v1 = act[:, pp_, :]
            act[:, pp_, :] = v1 - v1 @ Q_ @ Q_.T
            v2 = act[:, ts_, :]
            act[:, ts_, :] = v2 - v2 @ Q_ @ Q_.T
            return act
        return hf

    def score_under(model, ids, ts_starts, ts_ends, lbls, idx, hooks):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=hooks)
        K = ids.shape[0]
        seq_lp = torch.zeros(K, dtype=torch.float32, device=logits.device)
        for k in range(K):
            for pos in range(ts_starts[k], ts_ends[k] + 1):
                tgt = lbls[k, pos].item()
                if tgt == -100:
                    continue
                lp = F.log_softmax(logits[k, pos - 1], dim=-1)
                seq_lp[k] += lp[tgt].float()
        norm = seq_lp - torch.logsumexp(seq_lp, dim=0)
        cand_loss = float(-norm[idx].item())
        pred = int(torch.argmax(norm).item())
        return cand_loss, int(pred == idx)

    for p in pairs:
        ids_c, lbls_c, ts_c, te_c = _build_candidate_tensors(
            tokenizer, p["base"], p["z_clean"], p["candidates"], device)
        ids_k, lbls_k, ts_k, te_k = _build_candidate_tensors(
            tokenizer, p["base"], p["z_corrupt"], p["candidates"], device)
        hooks_c = [(HOOK_NAME, make_erasure_hook(ts_c[0]))]
        hooks_k = [(HOOK_NAME, make_erasure_hook(ts_k[0]))]
        Lc, ac = score_under(model, ids_c, ts_c, te_c, lbls_c,
                             p["idx_clean"], hooks_c)
        Lk, _ = score_under(model, ids_k, ts_k, te_k, lbls_k,
                            p["idx_clean"], hooks_k)
        losses_clean.append(Lc)
        losses_shuf.append(Lk)
        accs_clean.append(ac)

    return {
        "L_clean_erased": float(np.mean(losses_clean)),
        "L_shuf_erased": float(np.mean(losses_shuf)),
        "delta_z_erased": float(np.mean(losses_shuf) - np.mean(losses_clean)),
        "acc_clean_erased": float(np.mean(accs_clean)),
        "n_pairs": len(pairs),
    }


def random_subspace_basis(rank, d, seed=0):
    rng = np.random.RandomState(seed)
    M = rng.randn(rank, d).astype(np.float32)
    return M


# ---------------------------------------------------------------------------
# Driver per (cell, phase)
# ---------------------------------------------------------------------------

def run_phase(cs, cc, tokenizer, mapping_data, step, device,
              n_residuals=512, n_pairs_causal=128, n_pairs_erasure=24,
              probe_C=0.1):
    model = make_model(cc, tokenizer, step, device)

    # 1. baseline probe
    X, y, z_pool = collect_target_start_residuals(
        model, tokenizer, mapping_data, n_residuals, device,
        ablation=None, seed=0)
    probe = train_z_probe(X, y, n_train=min(256, X.shape[0] - 64),
                          n_test=64, seed=0, C=probe_C)
    if probe is None:
        del model
        return {"error": "probe train failed"}

    # 4. random-label control
    probe_random = train_z_probe(X, y, n_train=min(256, X.shape[0] - 64),
                                 n_test=64, seed=0, C=probe_C,
                                 shuffle_labels=True)

    # 5. ablation residuals
    X_attn, y_attn, _ = collect_target_start_residuals(
        model, tokenizer, mapping_data, n_residuals, device,
        ablation="l0_attn_zero", seed=0)
    probe_attn_abl = train_z_probe(X_attn, y_attn,
                                   n_train=min(256, X_attn.shape[0] - 64),
                                   n_test=64, seed=0, C=probe_C)
    X_mlp, y_mlp, _ = collect_target_start_residuals(
        model, tokenizer, mapping_data, n_residuals, device,
        ablation="l0_mlp_zero", seed=0)
    probe_mlp_abl = train_z_probe(X_mlp, y_mlp,
                                  n_train=min(256, X_mlp.shape[0] - 64),
                                  n_test=64, seed=0, C=probe_C)

    # 2. causal direction set (at probe_pos and at target_start)
    C_dirs_dict = causal_direction_set(model, tokenizer, mapping_data,
                                       n_pairs=n_pairs_causal, device=device,
                                       seed=0)

    # 3. erasure (only the trained-on-true-labels probe subspace)
    rank = probe["W_raw"].shape[0] - 1  # K-1 effective rank after centering
    rank = min(rank, X.shape[1] - 1)
    Vt_probe, _ = topk_basis(probe["W_raw"], k=rank)
    erase_baseline = candidate_loss_with_subspace_erasure(
        model, tokenizer, mapping_data, subspace_basis=None,
        device=device, n_pairs=n_pairs_erasure, seed=0)
    erase_probe = candidate_loss_with_subspace_erasure(
        model, tokenizer, mapping_data, subspace_basis=Vt_probe,
        device=device, n_pairs=n_pairs_erasure, seed=0)
    rand_basis = random_subspace_basis(rank=rank, d=X.shape[1], seed=0)
    erase_random = candidate_loss_with_subspace_erasure(
        model, tokenizer, mapping_data, subspace_basis=rand_basis,
        device=device, n_pairs=n_pairs_erasure, seed=0)

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "step": step,
        "n_residuals": int(X.shape[0]),
        "probe": {
            "train_acc": probe["train_acc"],
            "test_acc": probe["test_acc"],
            "n_classes": len(probe["classes"]),
            "W_raw": probe["W_raw"].tolist(),
        },
        "probe_random_label": {
            "train_acc": probe_random["train_acc"] if probe_random else float("nan"),
            "test_acc": probe_random["test_acc"] if probe_random else float("nan"),
        },
        "probe_l0_attn_ablated": {
            "train_acc": probe_attn_abl["train_acc"] if probe_attn_abl else float("nan"),
            "test_acc": probe_attn_abl["test_acc"] if probe_attn_abl else float("nan"),
        },
        "probe_l0_mlp_ablated": {
            "train_acc": probe_mlp_abl["train_acc"] if probe_mlp_abl else float("nan"),
            "test_acc": probe_mlp_abl["test_acc"] if probe_mlp_abl else float("nan"),
        },
        "causal_directions_at_probe_pos": C_dirs_dict["at_probe_pos"].tolist(),
        "causal_directions_at_target_start": C_dirs_dict["at_target_start"].tolist(),
        "erasure": {
            "baseline":      erase_baseline,
            "probe_subspace": erase_probe,
            "random_subspace": erase_random,
            "rank": int(rank),
        },
    }


# ---------------------------------------------------------------------------
# Cross-phase analysis
# ---------------------------------------------------------------------------

def analyse_cell(cell_data, label):
    """Given the per-phase results for one cell, compute cross-phase
    subspace similarities and probe-vs-causal projections."""
    phases = list(cell_data["phase_results"].keys())
    out = {"phases": phases}

    # For each phase, build the probe subspace basis (top components of W).
    bases = {}
    for ph in phases:
        pr = cell_data["phase_results"][ph]
        if "error" in pr or "probe" not in pr:
            continue
        W = np.array(pr["probe"]["W_raw"])  # (K-seen, d)
        rank = max(1, min(W.shape[0] - 1, W.shape[1] - 1))
        Vt, _ = topk_basis(W, k=rank)
        bases[ph] = Vt
    out["probe_subspace_rank"] = {ph: int(b.shape[0]) for ph, b in bases.items()}

    # 1. pairwise principal-angle similarity
    pair_sims = {}
    for i, p1 in enumerate(phases):
        for p2 in phases[i:]:
            if p1 not in bases or p2 not in bases:
                continue
            sim = subspace_similarity(bases[p1], bases[p2])
            pair_sims[f"{p1}__vs__{p2}"] = sim
    out["subspace_similarity"] = pair_sims

    # 2. probe vs causal subspace — at TWO causal positions
    causal_sub = {}
    for ph in phases:
        pr = cell_data["phase_results"][ph]
        if "error" in pr:
            continue
        for tag, key in [
            ("at_probe_pos", "causal_directions_at_probe_pos"),
            ("at_target_start", "causal_directions_at_target_start"),
        ]:
            C = np.array(pr[key])
            Cb, _ = topk_basis(C - C.mean(axis=0, keepdims=True),
                               k=min(C.shape[0], C.shape[1]) - 1)
            if ph in bases:
                r = bases[ph].shape[0]
                Cb = Cb[:r]
            causal_sub[(ph, tag)] = Cb

    proj_results = {}
    for ph_probe in bases:
        for (ph_causal, tag) in causal_sub:
            proj_p_on_c = projection_fraction(bases[ph_probe],
                                              causal_sub[(ph_causal, tag)])
            proj_c_on_p = projection_fraction(causal_sub[(ph_causal, tag)],
                                              bases[ph_probe])
            sim = subspace_similarity(bases[ph_probe],
                                      causal_sub[(ph_causal, tag)])
            proj_results[f"probe_{ph_probe}__causal_{ph_causal}_{tag}"] = {
                "proj_probe_on_causal": proj_p_on_c,
                "proj_causal_on_probe": proj_c_on_p,
                "subspace_mean_cosine": sim["mean_cosine"],
            }
    out["probe_vs_causal"] = proj_results

    # 3. Erasure summary
    erasure_summary = {}
    for ph in phases:
        pr = cell_data["phase_results"][ph]
        if "error" in pr:
            continue
        e = pr["erasure"]
        b = e["baseline"]
        es = e["probe_subspace"]
        er = e["random_subspace"]
        erasure_summary[ph] = {
            "rank": e["rank"],
            "L_clean_baseline": b["L_clean_erased"],
            "delta_z_baseline": b["delta_z_erased"],
            "L_clean_probe_erased": es["L_clean_erased"],
            "delta_z_probe_erased": es["delta_z_erased"],
            "L_clean_random_erased": er["L_clean_erased"],
            "delta_z_random_erased": er["delta_z_erased"],
            "delta_L_probe": es["L_clean_erased"] - b["L_clean_erased"],
            "delta_L_random": er["L_clean_erased"] - b["L_clean_erased"],
            "delta_dz_probe": es["delta_z_erased"] - b["delta_z_erased"],
            "delta_dz_random": er["delta_z_erased"] - b["delta_z_erased"],
        }
    out["erasure_summary"] = erasure_summary

    # 4 + 5: probe ablation summary
    probe_summary = {}
    for ph in phases:
        pr = cell_data["phase_results"][ph]
        if "error" in pr:
            continue
        probe_summary[ph] = {
            "real":         pr["probe"]["test_acc"],
            "random_label": pr["probe_random_label"]["test_acc"],
            "l0_attn_abl":  pr["probe_l0_attn_ablated"]["test_acc"],
            "l0_mlp_abl":   pr["probe_l0_mlp_ablated"]["test_acc"],
        }
    out["probe_summary"] = probe_summary

    return out


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def cell_verdict(label, cell_an):
    """Return a string verdict per cell."""
    phases = cell_an["phases"]
    if "early" not in phases:
        return f"{label}: insufficient phases."
    # gather metrics
    sim = cell_an["subspace_similarity"]
    early_post_key = None
    if "early__vs__post" in sim:
        early_post_key = "early__vs__post"
    early_post = sim.get(early_post_key, {}).get("mean_cosine", float("nan"))

    pvc = cell_an["probe_vs_causal"]
    # primary alignment test: early-probe → post-causal at the SAME position
    # used by the probe (probe_pos = ts - 1).
    early_to_late_causal = None
    last_causal = None
    for ph in ["post", "transition", "late"]:
        k = f"probe_early__causal_{ph}_at_probe_pos"
        if k in pvc:
            last_causal = ph
            early_to_late_causal = pvc[k]["proj_probe_on_causal"]
            break
    post_to_post_causal = pvc.get(
        "probe_post__causal_post_at_probe_pos", {}).get(
        "proj_probe_on_causal", float("nan"))
    # secondary cross-position test: probe (at probe_pos) vs causal at ts.
    early_to_late_causal_xpos = None
    for ph in ["post", "transition", "late"]:
        k = f"probe_early__causal_{ph}_at_target_start"
        if k in pvc:
            early_to_late_causal_xpos = pvc[k]["proj_probe_on_causal"]
            break

    er = cell_an["erasure_summary"]
    early_dl = er.get("early", {}).get("delta_L_probe", float("nan"))
    early_dl_rand = er.get("early", {}).get("delta_L_random", float("nan"))
    post_dl = er.get("post", {}).get("delta_L_probe", float("nan"))
    post_dl_rand = er.get("post", {}).get("delta_L_random", float("nan"))

    pr_sum = cell_an["probe_summary"]
    early_real = pr_sum.get("early", {}).get("real", float("nan"))
    early_rand = pr_sum.get("early", {}).get("random_label", float("nan"))
    early_attn = pr_sum.get("early", {}).get("l0_attn_abl", float("nan"))

    # Heuristic verdict:
    high_sim = early_post >= 0.5
    early_proj_aligned = (
        not math.isnan(early_to_late_causal) and early_to_late_causal >= 0.4
    )
    post_erasure_hurts = (
        not math.isnan(post_dl) and post_dl - post_dl_rand >= 0.10
    )
    early_erasure_neutral = (
        not math.isnan(early_dl) and abs(early_dl - early_dl_rand) < 0.10
    )
    probe_above_chance = (
        not math.isnan(early_real) and not math.isnan(early_rand)
        and early_real - early_rand >= 0.20
    )
    attn_drives_probe = (
        not math.isnan(early_attn) and not math.isnan(early_real)
        and early_real - early_attn >= 0.20
    )

    pluses = sum([high_sim, early_proj_aligned, post_erasure_hurts,
                  early_erasure_neutral, probe_above_chance])
    if pluses >= 4:
        verdict = "A: strong support for routed-before-readout"
    elif pluses <= 1:
        verdict = "B: linear-shadow-before-causal-basis"
    else:
        verdict = "C: mixed/regime-dependent"

    notes = []
    notes.append(f"early-vs-post probe subspace mean-cosine = {early_post:.2f}")
    e_str = (f"{early_to_late_causal:.2f}"
             if early_to_late_causal is not None else "n/a")
    ex_str = (f"{early_to_late_causal_xpos:.2f}"
              if early_to_late_causal_xpos is not None else "n/a")
    notes.append(
        f"early-probe→{last_causal}-causal projection at probe_pos "
        f"(ts−1) = {e_str}; cross-position to target_start (ts) = "
        f"{ex_str}; (post-probe→post-causal at probe_pos = "
        f"{post_to_post_causal:.2f})")
    notes.append(
        f"erasure ΔL_clean (post): probe = {post_dl:+.2f}, "
        f"random = {post_dl_rand:+.2f}; "
        f"(early): probe = {early_dl:+.2f}, random = {early_dl_rand:+.2f}")
    notes.append(
        f"early probe acc real = {early_real:.2f}, "
        f"random-label = {early_rand:.2f}, "
        f"L0 attn-ablated = {early_attn:.2f}, "
        f"attn drives probe = {attn_drives_probe}")
    return verdict, notes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    device = args.device or _select_device()
    print(f"device: {device}")

    # Schedule: A and D primary; B, C secondary.
    SCHEDULE = {
        "A_stable_K20": ["early", "transition", "post"],
        "D_K36":        ["early", "late", "transition", "post"],
        "B_critical_K20": ["early", "late", "transition", "post"],
        "C_stuck_K20":  ["early", "late", "transition", "post"],
    }
    label_to_cell = {cs.label: cs for cs in CELLS}
    out = {"by_cell": {}, "device": device,
           "PROBE_LAYER": PROBE_LAYER}
    json_path = RESULTS_DIR / "mech_interp_probe_causal_alignment.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    for label, phases in SCHEDULE.items():
        cs = label_to_cell[label]
        print(f"\n>>> {label}")
        cc, tokenizer, mapping_data, train_ds, rows = load_cell(cs, device)
        phase_map = select_phase_ckpts(cs, rows)
        cell_out = {"phases_map": phase_map, "phase_results": {}}
        for phase in phases:
            step = phase_map.get(phase)
            if step is None:
                continue
            t0 = time.time()
            try:
                pr = run_phase(
                    cs, cc, tokenizer, mapping_data, step, device,
                    n_residuals=args.n_residuals,
                    n_pairs_causal=args.n_pairs_causal,
                    n_pairs_erasure=args.n_pairs_erasure,
                    probe_C=args.probe_C,
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                pr = {"error": str(e)}
            cell_out["phase_results"][phase] = pr
            print(f"  {phase} step={step}  ({time.time()-t0:.1f}s)")
        # cross-phase analysis for this cell
        cell_out["analysis"] = analyse_cell(cell_out, label)
        out["by_cell"][label] = cell_out
        # save incrementally (drop the heavy fields for JSON)
        with open(json_path, "w") as f:
            json.dump(_strip_heavy(out), f, indent=2)

    # verdict
    out["verdicts"] = {}
    for label, cell in out["by_cell"].items():
        v, notes = cell_verdict(label, cell["analysis"])
        out["verdicts"][label] = {"verdict": v, "notes": notes}

    with open(json_path, "w") as f:
        json.dump(_strip_heavy(out), f, indent=2)
    write_summary_md(out)
    make_figure(out)


def _strip_heavy(out):
    """Drop W_raw and causal_directions tensors from the JSON to keep the
    file size manageable."""
    out2 = {**out, "by_cell": {}}
    for label, cell in out["by_cell"].items():
        cell2 = {**cell, "phase_results": {}}
        for ph, pr in cell["phase_results"].items():
            if "error" in pr:
                cell2["phase_results"][ph] = pr
                continue
            pr2 = {**pr}
            if "probe" in pr2:
                pr2["probe"] = {k: v for k, v in pr2["probe"].items()
                                if k != "W_raw"}
            for key in ["causal_directions_at_probe_pos",
                        "causal_directions_at_target_start"]:
                if key in pr2:
                    pr2[f"{key}_n"] = len(pr2[key])
                    del pr2[key]
            cell2["phase_results"][ph] = pr2
        out2["by_cell"][label] = cell2
    return out2


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def write_summary_md(out: Dict):
    path = RESULTS_DIR / "mech_interp_probe_causal_alignment.md"
    lines = []
    lines.append("# Probe–causal alignment stress test\n")
    lines.append(
        "Tests whether the early-plateau z-probe direction at the "
        "target_start residual aligns with the later causal-use "
        "direction. Probes operate at "
        f"`blocks.{PROBE_LAYER}.hook_resid_post`, position = target_start.\n"
    )
    lines.append(
        "Five analyses per cell × phase: (1) cross-phase probe-subspace "
        "stability, (2) probe-vs-causal subspace projection, (3) amnesic "
        "erasure of the probe subspace from the running residual stream "
        "vs random-subspace control, (4) random-label probe baseline, "
        "(5) L0 attention vs L0 MLP ablation effect on probe accuracy.\n"
    )

    for label, cell in out["by_cell"].items():
        an = cell["analysis"]
        lines.append(f"## {label}\n")
        # 1. probe stability
        lines.append("### Probe-subspace stability across phases (mean principal-angle cosine)\n")
        sim = an["subspace_similarity"]
        ph_keys = list(an["phases"])
        rows = ["| | " + " | ".join(ph_keys) + " |"]
        rows.append("|" + "---|" * (len(ph_keys) + 1))
        for p1 in ph_keys:
            row_vals = []
            for p2 in ph_keys:
                k1 = f"{p1}__vs__{p2}"
                k2 = f"{p2}__vs__{p1}"
                if k1 in sim:
                    row_vals.append(f"{sim[k1]['mean_cosine']:.2f}")
                elif k2 in sim:
                    row_vals.append(f"{sim[k2]['mean_cosine']:.2f}")
                else:
                    row_vals.append("—")
            rows.append(f"| {p1} | " + " | ".join(row_vals) + " |")
        lines.append("\n".join(rows))
        lines.append("")

        # 2. probe vs causal — split by causal-position
        lines.append("### Probe vs causal-difference subspace\n")
        lines.append(
            "Probe lives at `probe_pos = target_start − 1` (the slot whose "
            "residual produces the first-A logit). Causal differences "
            "are computed at two positions: same as probe (`at_probe_pos`) "
            "and at `target_start` itself (`at_target_start`, where "
            "earlier patching localised most causal mass).\n"
        )
        for tag in ["at_probe_pos", "at_target_start"]:
            lines.append(f"#### causal subspace {tag}\n")
            rows = ["| probe @ phase | causal @ phase | proj(probe→causal) | proj(causal→probe) | mean cosine |"]
            rows.append("|---|---|---|---|---|")
            for k, v in an["probe_vs_causal"].items():
                if not k.endswith(tag):
                    continue
                rest = k.replace("probe_", "")
                ph_p, rest2 = rest.split("__causal_")
                ph_c = rest2[:-len(tag) - 1]  # drop trailing "_<tag>"
                rows.append(
                    f"| {ph_p} | {ph_c} | "
                    f"{v['proj_probe_on_causal']:.2f} | "
                    f"{v['proj_causal_on_probe']:.2f} | "
                    f"{v['subspace_mean_cosine']:.2f} |"
                )
            lines.append("\n".join(rows))
            lines.append("")

        # 3. erasure
        lines.append("### Amnesic erasure (probe subspace vs random-subspace control)\n")
        rows = ["| phase | rank | L_baseline | L_probe_erased | L_random_erased | ΔL_probe | ΔL_random | dz_baseline | dz_probe_erased | dz_random_erased |"]
        rows.append("|" + "---|" * 11)
        for ph, e in an["erasure_summary"].items():
            rows.append(
                f"| {ph} | {e['rank']} | "
                f"{e['L_clean_baseline']:.3f} | "
                f"{e['L_clean_probe_erased']:.3f} | "
                f"{e['L_clean_random_erased']:.3f} | "
                f"{e['delta_L_probe']:+.3f} | "
                f"{e['delta_L_random']:+.3f} | "
                f"{e['delta_z_baseline']:+.3f} | "
                f"{e['delta_z_probe_erased']:+.3f} | "
                f"{e['delta_z_random_erased']:+.3f} |"
            )
        lines.append("\n".join(rows))
        lines.append("")

        # 4 + 5. probe summary
        lines.append("### Probe accuracy: real vs random-label vs L0-attn/MLP-ablated\n")
        rows = ["| phase | real | random-label | L0 attn-ablated | L0 mlp-ablated |"]
        rows.append("|---|---|---|---|---|")
        for ph, p in an["probe_summary"].items():
            rows.append(
                f"| {ph} | {p['real']:.2f} | {p['random_label']:.2f} | "
                f"{p['l0_attn_abl']:.2f} | {p['l0_mlp_abl']:.2f} |"
            )
        lines.append("\n".join(rows))
        lines.append("")

    # Verdicts
    lines.append("## Verdicts\n")
    for label, v in out["verdicts"].items():
        lines.append(f"### {label} → **{v['verdict']}**\n")
        for n in v["notes"]:
            lines.append(f"- {n}")
        lines.append("")
    lines.append(
        "Verdict legend: **A** — strong support for "
        "*routed-before-readout* (early probe subspace ≈ post probe "
        "subspace, early probe projects onto later causal direction, "
        "post-erasure hurts more than random, early-erasure does not). "
        "**B** — *linear-shadow-before-causal-basis* (probe decodes z "
        "early but the basis reorganises before causal use). **C** — "
        "mixed / regime-dependent.\n"
    )
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(out: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cells = list(out["by_cell"].keys())

    fig, axes = plt.subplots(4, len(cells), figsize=(4.6 * len(cells), 14),
                             squeeze=False)

    for ci, label in enumerate(cells):
        cell = out["by_cell"][label]
        an = cell["analysis"]
        phases = an["phases"]

        # Row 0: probe-subspace similarity matrix
        ax = axes[0][ci]
        n = len(phases)
        mat = np.full((n, n), np.nan)
        for i, p1 in enumerate(phases):
            for j, p2 in enumerate(phases):
                k1 = f"{p1}__vs__{p2}"
                k2 = f"{p2}__vs__{p1}"
                v = (an["subspace_similarity"].get(k1)
                     or an["subspace_similarity"].get(k2))
                if v:
                    mat[i, j] = v["mean_cosine"]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(phases, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(phases, fontsize=8)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=7)
        ax.set_title(f"{label}\nProbe subspace mean-cosine", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)

        # Row 1: probe→causal projection (at probe_pos = ts-1)
        ax = axes[1][ci]
        mat2 = np.full((n, n), np.nan)
        for i, pp in enumerate(phases):
            for j, pc in enumerate(phases):
                k = f"probe_{pp}__causal_{pc}_at_probe_pos"
                if k in an["probe_vs_causal"]:
                    mat2[i, j] = an["probe_vs_causal"][k][
                        "proj_probe_on_causal"]
        im = ax.imshow(mat2, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(phases, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(phases, fontsize=8)
        ax.set_xlabel("causal subspace @ phase", fontsize=8)
        ax.set_ylabel("probe subspace @ phase", fontsize=8)
        for i in range(n):
            for j in range(n):
                v = mat2[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=7)
        ax.set_title("proj(probe → causal)\nat probe_pos (ts−1)",
                     fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)

        # Row 2: erasure ΔL clean (probe vs random)
        ax = axes[2][ci]
        ph_list = list(an["erasure_summary"].keys())
        x = np.arange(len(ph_list))
        dL_probe = [an["erasure_summary"][p]["delta_L_probe"] for p in ph_list]
        dL_rand = [an["erasure_summary"][p]["delta_L_random"] for p in ph_list]
        ax.bar(x - 0.2, dL_probe, 0.4, label="probe subspace")
        ax.bar(x + 0.2, dL_rand, 0.4, label="random subspace",
               color="grey")
        ax.set_xticks(x); ax.set_xticklabels(ph_list, rotation=30,
                                             ha="right", fontsize=8)
        ax.set_ylabel("ΔL_clean (after erasure)", fontsize=8)
        ax.set_title("Erasure damage", fontsize=9)
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", lw=0.5)

        # Row 3: probe acc — real vs random-label vs L0-attn-abl vs L0-mlp-abl
        ax = axes[3][ci]
        ph_list = list(an["probe_summary"].keys())
        x = np.arange(len(ph_list))
        real = [an["probe_summary"][p]["real"] for p in ph_list]
        rand = [an["probe_summary"][p]["random_label"] for p in ph_list]
        attn = [an["probe_summary"][p]["l0_attn_abl"] for p in ph_list]
        mlp = [an["probe_summary"][p]["l0_mlp_abl"] for p in ph_list]
        w = 0.2
        ax.bar(x - 1.5 * w, real, w, label="real")
        ax.bar(x - 0.5 * w, rand, w, label="random label",
               color="grey")
        ax.bar(x + 0.5 * w, attn, w, label="L0 attn ablated",
               color="C3")
        ax.bar(x + 1.5 * w, mlp, w, label="L0 mlp ablated",
               color="C2")
        ax.set_xticks(x); ax.set_xticklabels(ph_list, rotation=30,
                                             ha="right", fontsize=8)
        ax.set_ylabel("probe test accuracy", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("Probe acc baselines", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Probe–causal alignment stress test — "
        "is early-plateau decodability the same basis as later causal use?",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "mech_interp_probe_causal_alignment.png",
                dpi=140)
    plt.close(fig)
    print(f"  wrote {fig_dir / 'mech_interp_probe_causal_alignment.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-residuals", type=int, default=512)
    p.add_argument("--n-pairs-causal", type=int, default=128)
    p.add_argument("--n-pairs-erasure", type=int, default=24)
    p.add_argument("--probe-C", type=float, default=0.1)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
