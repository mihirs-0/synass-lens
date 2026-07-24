#!/usr/bin/env python
"""
Multi-seed attention-to-z trajectory using existing saddle_probe_ckpts
fine-cadence weight checkpoints (every 250 steps).

For each of 3 seeds, load all available checkpoints, apply AttentionToZProbe,
and plot the per-seed trajectory of mean attention-to-z (across all heads
and layers, averaged over a fixed eval batch).

Output:
  outputs/paper_figures/fig_attention_to_z_multiseed.{pdf,png}
  eta_sweep/results/multi_seed/attention_to_z_multiseed.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config  # noqa: E402
from src.model import create_model_from_config  # noqa: E402

from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import _select_device, _set_all_seeds, build_legacy_cfg


OUT = REPO_ROOT / "outputs" / "paper_figures"
MS_DIR = RESULTS_DIR / "multi_seed"
MS_DIR.mkdir(parents=True, exist_ok=True)


def attention_to_z_for_checkpoint(model, ckpt_path: Path, loader, device: str,
                                   z_positions=(8, 9), a_positions=(11, 12, 13, 14),
                                   n_batches: int = 4) -> dict:
    """Load checkpoint, run forward with attention hooks, compute mean attention
    from A positions to z positions, averaged across heads/layers and batch."""
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # We collect attention patterns via cache hooks
    per_head = np.zeros((n_layers, n_heads))
    n_seen = 0
    it = iter(loader)
    for b_idx in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_ids, names_filter=lambda n: "attn.hook_pattern" in n)
        for L in range(n_layers):
            patt = cache[f"blocks.{L}.attn.hook_pattern"]  # (batch, head, q, k)
            # Average attention from a_positions (queries) to z_positions (keys)
            atoz = patt[:, :, list(a_positions), :][:, :, :, list(z_positions)].mean(dim=(0, 2, 3))
            per_head[L] += atoz.cpu().numpy()
        n_seen += 1
    per_head /= max(n_seen, 1)
    return {
        "mean_attention_to_z": float(per_head.mean()),
        "max_attention_to_z": float(per_head.max()),
        "per_head": per_head.tolist(),
    }


def main():
    seeds = [0, 1, 2]
    device = _select_device()

    # Set up data + model
    cc = CellConfig(eta=0.001, k=10, seed=0)
    _set_all_seeds(0)
    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, _ = create_datasets_from_config(cfg, tokenizer)
    loader = DataLoader(train_dataset, batch_size=cc.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tokenizer).to(device)

    seeds_data = {}
    for seed in seeds:
        ckpt_dir = RESULTS_DIR / "saddle_probe_ckpts" / f"eta_0.001_K_10_seed_{seed}" / "checkpoints"
        if not ckpt_dir.exists():
            print(f"WARN: no fine-cadence checkpoints for seed {seed} at {ckpt_dir}")
            continue
        # Load all checkpoints
        ckpts = sorted(ckpt_dir.glob("model_step_*.pt"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        # Find τ from corresponding status.json
        status_f = RESULTS_DIR / "v_tracking" / f"eta_0.001_K_10_seed_{seed}" / "status.json"
        if status_f.exists():
            tau = json.loads(status_f.read_text()).get("transition_detected_step", 2500)
        else:
            tau = 2500

        print(f"Seed {seed}: {len(ckpts)} checkpoints, τ ≈ {tau}")
        per_seed_traj = []
        for ckpt in ckpts:
            step = int(ckpt.stem.split("_")[-1])
            if step > 3500:  # don't waste time post-transition
                continue
            res = attention_to_z_for_checkpoint(model, ckpt, loader, device,
                                                 n_batches=4)
            res["step"] = step
            res["step_over_tau"] = step / tau
            per_seed_traj.append(res)
            print(f"  step {step:5d} (step/τ={step/tau:.2f}): "
                  f"mean attention-to-z = {res['mean_attention_to_z']:.4f}, "
                  f"max = {res['max_attention_to_z']:.4f}")
        seeds_data[seed] = {"tau": tau, "trajectory": per_seed_traj}

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel a: trajectories vs step
    ax = axes[0]
    for seed, d in seeds_data.items():
        traj = d["trajectory"]
        if not traj:
            continue
        steps = np.array([t["step"] for t in traj])
        vals = np.array([t["mean_attention_to_z"] for t in traj])
        ax.plot(steps, vals, marker="o", lw=1.6, alpha=0.85, label=f"seed {seed}  (τ={d['tau']})")
        ax.axvline(d["tau"], color=f"C{seed}", linestyle=":", alpha=0.4, lw=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel("mean attention to z (across heads/layers, A→z)")
    ax.set_title("(a) Attention-to-z trajectory across 3 seeds")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, lw=0.4)

    # Panel b: trajectories vs step/τ
    ax = axes[1]
    for seed, d in seeds_data.items():
        traj = d["trajectory"]
        if not traj:
            continue
        sot = np.array([t["step_over_tau"] for t in traj])
        vals = np.array([t["mean_attention_to_z"] for t in traj])
        ax.plot(sot, vals, marker="o", lw=1.6, alpha=0.85, label=f"seed {seed}")
    ax.axvline(1.0, color="black", linestyle=":", lw=0.8, alpha=0.5, label=r"$\tau$")
    ax.set_xlabel(r"step / $\tau$")
    ax.set_ylabel("mean attention to z")
    ax.set_title("(b) Attention-to-z vs step/τ — alignment across seeds")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, lw=0.4)

    fig.suptitle("Multi-seed attention-to-z trajectory (3 seeds, K=10, η=10⁻³, AdamW)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig_attention_to_z_multiseed.pdf", dpi=200)
    fig.savefig(OUT / "fig_attention_to_z_multiseed.png", dpi=160)

    # Monotonicity check
    print("\n=== Monotonic growth check during plateau (step/τ in [0.2, 0.95]) ===")
    all_mono = True
    monotonic_seeds = []
    for seed, d in seeds_data.items():
        traj = d["trajectory"]
        plateau_pts = [t for t in traj if 0.2 <= t["step_over_tau"] <= 0.95]
        vals = [t["mean_attention_to_z"] for t in plateau_pts]
        is_mono = all(vals[i] <= vals[i+1] + 1e-4 for i in range(len(vals)-1))  # tolerance for tiny dips
        print(f"  seed {seed}: {len(vals)} plateau pts, range [{min(vals):.4f}, {max(vals):.4f}]  "
              f"{'✓ monotonic' if is_mono else '✗ NOT monotonic'}")
        if is_mono:
            monotonic_seeds.append(seed)
        else:
            all_mono = False

    out = {"seeds_data": {str(s): d for s, d in seeds_data.items()},
           "monotonic_seeds": monotonic_seeds, "all_monotonic": all_mono}
    (MS_DIR / "attention_to_z_multiseed.json").write_text(json.dumps(out, indent=2))
    print(f"\nMonotonic seeds: {monotonic_seeds} / {list(seeds_data.keys())}")
    print(f"Saved: {MS_DIR / 'attention_to_z_multiseed.json'}")
    print(f"Figure: {OUT / 'fig_attention_to_z_multiseed.pdf'}")


if __name__ == "__main__":
    main()
