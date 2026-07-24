#!/usr/bin/env python
"""Block 1 — stronger readouts comparison at final checkpoint.

For seed 42, K=10, final ckpt: compute three readouts per layer:
  (R1) logit lens P(correct)
  (R2) linear probe accuracy
  (R3) 2-layer MLP probe accuracy

Question: does late-layer decodability survive under stronger readouts?
If stronger readouts push "first decodable layer" shallower, the
ablation-sufficiency mismatch might be an artifact of a weak readout.
"""

import sys
import json
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

SEEDS = [42, 123, 456, 789]
K = 10
N_UNIQUE_B = 1000
N_LAYERS = 4
N_EVAL = 1024
BATCH_SIZE = 128
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "block1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_final(seed):
    cfg = make_config(experiment_name=f"b1_s{seed}", k=K, seed=seed,
                      n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=seed, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    ckpt_dir = Path("outputs") / ("landauer_dense_k10" if seed == 42
                                   else f"landauer_dense_k10_seed{seed}")
    ckpt_dir = ckpt_dir / "checkpoints"
    steps = sorted(list_checkpoints(ckpt_dir))
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, ckpt_dir, step=steps[-1])
    model.eval()
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=seed, task="bz_to_a",
    )
    n_take = min(N_EVAL, len(ds.tokenized))
    ds.tokenized = ds.tokenized[:n_take]
    ds.examples = ds.examples[:n_take]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )
    return model, loader, steps[-1]


def collect(model, loader):
    layer_acts = [[] for _ in range(N_LAYERS)]
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            tgt = batch["target_start_positions"].to(DEVICE)
            bs = ids.shape[0]; idx = torch.arange(bs, device=DEVICE)
            names = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]
            _, cache = model.run_with_cache(ids, names_filter=names)
            pred_pos = tgt - 1
            for L in range(N_LAYERS):
                r = cache[f"blocks.{L}.hook_resid_post"][idx, pred_pos, :]
                layer_acts[L].append(r.cpu().numpy())
            labels_all.append(lab[idx, tgt].cpu().numpy())
    X = [np.concatenate(a, axis=0) for a in layer_acts]
    y = np.concatenate(labels_all, axis=0)
    return X, y


def logit_lens_Pcorrect(model, X, y):
    out = []
    for L in range(N_LAYERS):
        with torch.no_grad():
            r = torch.tensor(X[L], device=DEVICE)
            logits = model.unembed(model.ln_final(r))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        out.append(float(probs[np.arange(len(y)), y].mean()))
    return out


def linear_probe_acc(X, y):
    out = []
    for L in range(N_LAYERS):
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(y))
        split = int(0.8 * len(y))
        tr, te = perm[:split], perm[split:]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[L][tr]); Xte = sc.transform(X[L][te])
        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        try:
            clf.fit(Xtr, y[tr])
            out.append(float(clf.score(Xte, y[te])))
        except Exception:
            out.append(0.0)
    return out


def mlp_probe_acc(X, y, hidden=64, epochs=80, lr=1e-2, seed=0):
    out = []
    unique_y = np.unique(y)
    lut = {int(v): i for i, v in enumerate(unique_y)}
    y_mapped = np.array([lut[int(v)] for v in y])
    n_classes = len(unique_y)
    for L in range(N_LAYERS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        split = int(0.8 * len(y))
        tr, te = perm[:split], perm[split:]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[L][tr]); Xte = sc.transform(X[L][te])
        torch.manual_seed(seed)
        net = nn.Sequential(
            nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        ).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
        ytr_t = torch.tensor(y_mapped[tr], dtype=torch.long, device=DEVICE)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
        yte_t = torch.tensor(y_mapped[te], dtype=torch.long, device=DEVICE)
        for _ in range(epochs):
            net.train()
            loss = F.cross_entropy(net(Xtr_t), ytr_t)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xte_t).argmax(dim=-1) == yte_t).float().mean().item()
        out.append(float(acc))
    return out


def first_decodable(series, threshold=0.5):
    for i, v in enumerate(series):
        if v >= threshold:
            return i
    return None


def main():
    per_seed = {}
    for s in SEEDS:
        try:
            model, loader, step = load_final(s)
        except Exception as e:
            print(f"[skip] seed {s}: {e}")
            continue
        print(f"\n=== seed {s} (step {step}) ===")
        X, y = collect(model, loader)
        r1 = logit_lens_Pcorrect(model, X, y)
        r2 = linear_probe_acc(X, y)
        r3 = mlp_probe_acc(X, y)
        per_seed[s] = {
            "final_step": step,
            "logit_lens": r1,
            "linear_probe": r2,
            "mlp_probe": r3,
            "first_decodable": {
                "logit_lens": first_decodable(r1),
                "linear_probe": first_decodable(r2),
                "mlp_probe": first_decodable(r3),
            },
        }
        print(f"  logit lens  : {[f'{v:.2f}' for v in r1]}  first≥0.5: L{per_seed[s]['first_decodable']['logit_lens']}")
        print(f"  linear probe: {[f'{v:.2f}' for v in r2]}  first≥0.5: L{per_seed[s]['first_decodable']['linear_probe']}")
        print(f"  MLP probe   : {[f'{v:.2f}' for v in r3]}  first≥0.5: L{per_seed[s]['first_decodable']['mlp_probe']}")
        del model
        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    # Aggregate across seeds
    agg = {}
    for key in ["logit_lens", "linear_probe", "mlp_probe"]:
        arr = np.array([per_seed[s][key] for s in per_seed])
        agg[key] = {
            "mean_per_layer": arr.mean(axis=0).tolist(),
            "std_per_layer": arr.std(axis=0, ddof=0).tolist(),
        }
    # First-decodable per readout across seeds
    fd = {key: [per_seed[s]["first_decodable"][key] for s in per_seed]
          for key in ["logit_lens", "linear_probe", "mlp_probe"]}

    # Verdict: stronger readouts shouldn't change the key early/late split.
    # If MLP probe makes L0/L1 decodable (first_decodable <= 1), then
    # the paper's "late decodability" claim is readout-dependent.
    mlp_first = [x for x in fd["mlp_probe"] if x is not None]
    lens_first = [x for x in fd["logit_lens"] if x is not None]
    if mlp_first and lens_first:
        onset_shift = np.mean(lens_first) - np.mean(mlp_first)
        verdict = ("supports" if abs(onset_shift) <= 0.5 else
                   "partially_supports" if abs(onset_shift) <= 1.0 else
                   "weakens")
    else:
        verdict = "inconclusive"

    out = {
        "config": {"seeds": SEEDS, "K": K, "n_eval": N_EVAL, "device": DEVICE},
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "aggregate": agg,
        "first_decodable_by_readout": fd,
        "onset_shift_lens_to_mlp": (
            float(onset_shift) if mlp_first and lens_first else None
        ),
        "verdict": verdict,
    }
    path = OUT_DIR / "block1_readouts.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Block 1 summary ===")
    for key, a in agg.items():
        m = [f"{x:.2f}" for x in a["mean_per_layer"]]
        print(f"  {key:12s} mean per layer: {m}")
    print(f"  First-decodable-layer (L where readout >= 0.5):")
    for key, fds in fd.items():
        print(f"    {key}: {fds}")
    print(f"  Verdict: {verdict}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
