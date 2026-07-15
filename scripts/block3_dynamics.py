#!/usr/bin/env python
"""Block 3 — multi-diagnostic training dynamics with pre-committed onset rule.

For seed 42 (dense checkpoints), evaluate per-layer decodability across
training for three diagnostics that are cheap and non-redundant:
  (D1) logit lens P(correct)
  (D2) linear probe (logistic regression) accuracy on the correct first-target
  (D3) MLP probe (2-layer) accuracy

Onset rule (pre-committed):
  moderate: decodability >= 0.5 sustained for >= 10 consecutive eval steps
  strict:   decodability >= 0.75 sustained for >= 10 consecutive eval steps

Coupled-emergence supports iff, under BOTH thresholds, for each diagnostic:
  |onset(L2) - onset(L3)| <= tolerance
  AND |onset(L2) - onset(L3)| < |onset(L1) - onset(L2)|
  AND |onset(L2) - onset(L3)| < |onset(L1) - onset(L3)|

Tolerance = max(1% of 50000 = 500 steps, 200 steps) = 500 steps.
"""

import sys
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, DisambiguationDataset
from src.model import create_model_from_config
from src.training.checkpoint import load_checkpoint, list_checkpoints
from scripts.experiment_helpers import make_config

K = 10
LOG_K = math.log(K)
N_UNIQUE_B = 1000
N_LAYERS = 4
SEED = 42
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# Checkpoint sampling — dense around transition, sparser elsewhere
SAMPLE_STEPS = [
    100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1800, 1900, 2000,
    2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 6000, 7500,
    9000, 10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000,
]

N_EVAL = 512
BATCH_SIZE = 128

MODERATE_THRESHOLD = 0.5
STRICT_THRESHOLD = 0.75
SUSTAINED_WINDOW = 10   # consecutive eval steps
TOLERANCE_ABS = 500

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "block3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_cfg_tokenizer():
    cfg = make_config(experiment_name="b3_s42", k=K, seed=SEED,
                      n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    return cfg, tokenizer


def build_loader(mapping_data, tokenizer, n_eval=N_EVAL):
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:n_eval]
    ds.examples = ds.examples[:n_eval]
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )


def collect_acts_and_labels(model, loader):
    layer_acts = [[] for _ in range(N_LAYERS)]
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            tgt_starts = batch["target_start_positions"].to(DEVICE)
            bs = input_ids.shape[0]
            idx = torch.arange(bs, device=DEVICE)
            names = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]
            _, cache = model.run_with_cache(input_ids, names_filter=names)
            pred_pos = tgt_starts - 1
            for L in range(N_LAYERS):
                r = cache[f"blocks.{L}.hook_resid_post"][idx, pred_pos, :]
                layer_acts[L].append(r.cpu().numpy())
            correct_tok = labels[idx, tgt_starts]
            labels_all.append(correct_tok.cpu().numpy())
    X = [np.concatenate(layer_acts[L], axis=0) for L in range(N_LAYERS)]
    y = np.concatenate(labels_all, axis=0)
    return X, y


def logit_lens_per_layer(model, X):
    """P(correct) at each layer using cached activations already captured
    at the prediction position. X[L] is [N, d_model]."""
    # For efficiency we replicate logit-lens by pushing each stored residual
    # through ln_final + unembed.
    result = []
    for L in range(N_LAYERS):
        with torch.no_grad():
            r = torch.tensor(X[L], device=DEVICE)
            normed = model.ln_final(r)
            logits = model.unembed(normed)
            result.append(logits.cpu().numpy())
    return result


def train_eval_linear_probe(X, y, seed=0):
    """Leave-prompt-out: 80/20 split. Returns accuracy."""
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = perm[:split], perm[split:]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr_idx])
    Xte = scaler.transform(X[te_idx])
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="auto",
                             solver="liblinear")
    try:
        clf.fit(Xtr, y[tr_idx])
        return float(clf.score(Xte, y[te_idx]))
    except Exception:
        return 0.0


def train_eval_mlp_probe(X, y, seed=0, hidden=64, epochs=50, lr=1e-2):
    """Tiny 2-layer MLP classifier. Returns held-out accuracy."""
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = perm[:split], perm[split:]
    # Reindex labels to contiguous range
    unique_y = np.unique(y)
    lut = {int(v): i for i, v in enumerate(unique_y)}
    y_map = np.array([lut[int(v)] for v in y])
    n_classes = len(unique_y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr_idx])
    Xte = scaler.transform(X[te_idx])
    ytr = y_map[tr_idx]; yte = y_map[te_idx]

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes),
    ).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    yte_t = torch.tensor(yte, dtype=torch.long, device=DEVICE)
    for _ in range(epochs):
        net.train()
        logits = net(Xtr_t)
        loss = F.cross_entropy(logits, ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        preds = net(Xte_t).argmax(dim=-1)
        acc = (preds == yte_t).float().mean().item()
    return float(acc)


def evaluate_checkpoint(step, cfg, tokenizer, loader, ckpt_dir):
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, ckpt_dir, step=step)
    model.eval()

    X, y = collect_acts_and_labels(model, loader)
    # Logit lens P(correct)
    logits_per_layer = logit_lens_per_layer(model, X)
    d1_per_layer = []
    for L in range(N_LAYERS):
        # P(correct_first_target_token)
        log = logits_per_layer[L]
        probs = np.exp(log - log.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        p_correct = probs[np.arange(len(y)), y].mean()
        d1_per_layer.append(float(p_correct))

    # Linear probe accuracy
    d2_per_layer = [train_eval_linear_probe(X[L], y, seed=0) for L in range(N_LAYERS)]

    # MLP probe accuracy
    d3_per_layer = [train_eval_mlp_probe(X[L], y, seed=0) for L in range(N_LAYERS)]

    del model
    if DEVICE == "mps":
        try: torch.mps.empty_cache()
        except Exception: pass
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "step": step,
        "logit_lens": d1_per_layer,
        "linear_probe": d2_per_layer,
        "mlp_probe": d3_per_layer,
    }


def find_onset(series_per_layer, steps, threshold, window=SUSTAINED_WINDOW):
    """Return onset step: smallest step at which the diagnostic has been
    >= threshold for `window` consecutive eval steps (or the smallest window
    that's actually available if fewer than `window` points exist after onset).
    Returns None if no such window exists.
    """
    series = np.array(series_per_layer)  # length = len(steps)
    n = len(series)
    for i in range(n - window + 1):
        if np.all(series[i:i + window] >= threshold):
            return int(steps[i])
    # Looser fallback: if total samples < window, require all from i onward
    for i in range(n):
        if np.all(series[i:] >= threshold):
            return int(steps[i])
    return None


def onset_table(per_ckpt, diagnostic_key, steps):
    """diagnostic_key is 'logit_lens' | 'linear_probe' | 'mlp_probe'."""
    per_layer = [[c[diagnostic_key][L] for c in per_ckpt] for L in range(N_LAYERS)]
    out = {"moderate": {}, "strict": {}}
    for scheme, thr in [("moderate", MODERATE_THRESHOLD),
                         ("strict", STRICT_THRESHOLD)]:
        for L in range(N_LAYERS):
            onset = find_onset(per_layer[L], steps, thr)
            out[scheme][f"L{L}"] = onset
    return out


def coupled_emergence_verdict(onsets, max_step):
    """Given per-diagnostic onsets for both schemes, return per-diagnostic
    and overall verdict."""
    tolerance = max(TOLERANCE_ABS, int(0.01 * max_step))
    verdicts = {}
    for diag, d in onsets.items():
        diag_pass = True
        details = {}
        for scheme in ["moderate", "strict"]:
            o = d[scheme]
            l1, l2, l3 = o["L1"], o["L2"], o["L3"]
            if l2 is None or l3 is None:
                details[scheme] = "L2/L3 onset not reached"
                diag_pass = False
                continue
            gap_l2_l3 = abs(l2 - l3)
            gap_l1_l2 = abs(l1 - l2) if l1 is not None else float("inf")
            gap_l1_l3 = abs(l1 - l3) if l1 is not None else float("inf")
            within_tol = gap_l2_l3 <= tolerance
            closer_than_l1 = (gap_l2_l3 < gap_l1_l2) and (gap_l2_l3 < gap_l1_l3)
            details[scheme] = {
                "onset_L1": l1, "onset_L2": l2, "onset_L3": l3,
                "gap_L2_L3": gap_l2_l3,
                "gap_L1_L2": gap_l1_l2,
                "gap_L1_L3": gap_l1_l3,
                "within_tolerance": within_tol,
                "closer_than_L1": closer_than_l1,
                "scheme_passes": within_tol and closer_than_l1,
            }
            diag_pass = diag_pass and within_tol and closer_than_l1
        verdicts[diag] = {
            "schemes": details,
            "supports": diag_pass,
        }
    overall_pass = all(v["supports"] for v in verdicts.values())
    return verdicts, overall_pass, tolerance


def main():
    cfg, tokenizer = get_cfg_tokenizer()
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    loader = build_loader(mapping_data, tokenizer)

    ckpt_dir = Path("outputs") / "landauer_dense_k10" / "checkpoints"
    available = sorted(list_checkpoints(ckpt_dir))
    # Map requested SAMPLE_STEPS to nearest available checkpoint
    selected = []
    for s in SAMPLE_STEPS:
        nearest = min(available, key=lambda x: abs(x - s))
        if nearest not in selected:
            selected.append(nearest)
    selected = sorted(selected)
    print(f"Evaluating {len(selected)} checkpoints: {selected[:5]} ... {selected[-3:]}")

    per_ckpt = []
    for i, step in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] step {step}...", end="", flush=True)
        result = evaluate_checkpoint(step, cfg, tokenizer, loader, ckpt_dir)
        per_ckpt.append(result)
        ll = result["logit_lens"]
        lp = result["linear_probe"]
        mp = result["mlp_probe"]
        print(f"  lens=[{ll[0]:.2f} {ll[1]:.2f} {ll[2]:.2f} {ll[3]:.2f}]  "
              f"lin=[{lp[0]:.2f} {lp[1]:.2f} {lp[2]:.2f} {lp[3]:.2f}]  "
              f"mlp=[{mp[0]:.2f} {mp[1]:.2f} {mp[2]:.2f} {mp[3]:.2f}]")

    steps = [c["step"] for c in per_ckpt]
    onsets = {
        "logit_lens": onset_table(per_ckpt, "logit_lens", steps),
        "linear_probe": onset_table(per_ckpt, "linear_probe", steps),
        "mlp_probe": onset_table(per_ckpt, "mlp_probe", steps),
    }
    max_step = max(steps)
    verdicts, overall, tol = coupled_emergence_verdict(onsets, max_step)

    out = {
        "config": {
            "seed": SEED, "device": DEVICE, "n_checkpoints": len(selected),
            "moderate_threshold": MODERATE_THRESHOLD,
            "strict_threshold": STRICT_THRESHOLD,
            "sustained_window": SUSTAINED_WINDOW,
            "tolerance_steps": tol,
        },
        "checkpoints": per_ckpt,
        "onsets": onsets,
        "verdicts_per_diagnostic": verdicts,
        "overall_supports": overall,
    }
    path = OUT_DIR / "block3_dynamics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Block 3 verdict ===")
    for diag, v in verdicts.items():
        print(f"  {diag:14s} supports={v['supports']}")
        for scheme in ["moderate", "strict"]:
            s = v["schemes"][scheme]
            if isinstance(s, dict):
                print(f"    {scheme:8s}  L1={s['onset_L1']} L2={s['onset_L2']} L3={s['onset_L3']}  "
                      f"gap(L2,L3)={s['gap_L2_L3']} tol={tol}  passes={s['scheme_passes']}")
            else:
                print(f"    {scheme:8s}  {s}")
    print(f"\nOVERALL supports = {overall}  (tolerance = {tol} steps)")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
