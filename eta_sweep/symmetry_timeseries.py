#!/usr/bin/env python
"""Permutation-symmetry-breaking time series: does the MLP neuron population collapse to the
symmetric saddle during the plateau and differentiate (break symmetry) at the snap?

Metric: participation ratio PR of the MLP neuron-weight matrix (V = [W_in^T | W_out]), averaged
over layers. PR is SCALE-INVARIANT (= (sum s)^2 / sum s^2 of its singular values): low = neurons
collapsed/symmetric, high = differentiated/symmetry-broken. Tier-2 validated this separates
trapped (PR~82, collapsed) from converged (PR~202, broken).

Overlay per K with loss, KL_B, KL_z (the output-level translation-symmetry signals already validated),
and mark tau = loss-drop midpoint. Decisive: do PR-rise, KL-rise, and loss-drop coincide at tau?

(The attention-level delta_trans metric is dropped: it failed its positive control — confounded by
weight scale, and it measures 'attention reads z' not 'output uses z'. KL_z is the clean output signal.)
"""
import sys, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss
from eta_sweep.delta_b_diagnostic import roll_span, first_logp, mean_kl

CONFIGS = [(10, 0), (20, 0)]
OUT = REPO / "lesswrong_figures"


def participation_ratio(model):
    prs = []
    for l in range(model.cfg.n_layers):
        V = torch.cat([model.blocks[l].mlp.W_in.T, model.blocks[l].mlp.W_out], dim=1).detach().float().cpu()
        s = torch.linalg.svdvals(V)
        prs.append(float((s.sum() ** 2) / (s ** 2).sum()))
    return sum(prs) / len(prs)


def run(K, seed, device):
    _set_all_seeds(seed)
    cc = CellConfig(eta=0.001, k=K, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    ds, _, md = create_datasets_from_config(cfg, tok)
    loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tok).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

    N = 512
    exs = [ds[i] for i in range(N)]
    eb = collate_fn(exs); ids = eb["input_ids"]
    ex0 = exs[0]; zp, ze = ex0["z_position"], ex0["z_end_position"]; ts = ex0["target_start_position"]
    bz = dict(eb); bz["input_ids"] = roll_span(ids, zp, ze + 1)
    bb = dict(eb); bb["input_ids"] = roll_span(ids, 1, zp - 1)

    def ce(b):
        d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            l, _, _ = compute_loss(model, d)
        return float(l)

    def snap():
        model.eval()
        c = ce(eb); lc = first_logp(model, eb, device, ts)
        klz = mean_kl(lc, first_logp(model, bz, device, ts))
        klb = mean_kl(lc, first_logp(model, bb, device, ts))
        pr = participation_ratio(model)
        model.train()
        return c, klz, klb, pr

    maxstep = 5000 if K <= 10 else 9000
    sched = set(list(range(0, 3001, 100)) + list(range(3250, maxstep + 1, 250)))
    rec = {"step": [], "ce": [], "klz": [], "klb": [], "pr": []}

    def log(step):
        c, klz, klb, pr = snap()
        for k, v in zip(rec, (step, c, klz, klb, pr)):
            rec[k].append(v)

    log(0); step = 0
    while step < maxstep:
        for b in loader:
            if step >= maxstep:
                break
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            l, _, _ = compute_loss(model, b); l.backward(); opt.step(); step += 1
            if step in sched:
                log(step)
    # tau = first step where loss crosses below the plateau-midpoint
    plat = max(rec["ce"][1:4])
    mid = plat / 2
    tau = next((s for s, c in zip(rec["step"], rec["ce"]) if c < mid), rec["step"][-1])
    pr_plateau = min(rec["pr"][:len(rec["pr"]) // 2])   # collapse minimum in first half
    pr_final = rec["pr"][-1]
    pr_at_tau_rises = any(p > pr_plateau + 10 for s, p in zip(rec["step"], rec["pr"]) if s >= tau)
    print(f"K={K} s{seed}: tau(loss-mid)={tau}  PR plateau-min={pr_plateau:.0f}  PR final={pr_final:.0f}  "
          f"PR rises after tau={'yes' if pr_at_tau_rises else 'no'}")
    return dict(K=K, seed=seed, rec=rec, tau=tau, pr_plateau=pr_plateau, pr_final=pr_final)


def main():
    device = _select_device()
    print(f"device={device}  (permutation symmetry: PR of MLP neurons; attention delta_trans dropped)\n")
    res = [run(K, s, device) for K, s in CONFIGS]
    fig, axes = plt.subplots(2, len(res), figsize=(7.2 * len(res), 7.4), sharex="col")
    for j, r in enumerate(res):
        rc = r["rec"]; tau = r["tau"]
        ax = axes[0][j]
        ax.plot(rc["step"], rc["ce"], color="#1F77B4", lw=2.2, label="train CE")
        ax.set_ylim(0, 3.2); ax.set_ylabel("train CE (nats)", color="#1F77B4")
        ax.axvline(tau, color="#888", ls="--", lw=1.4)
        ax.text(tau, 3.05, f" τ={tau}", color="#555", fontsize=9, va="top")
        axr = ax.twinx()
        axr.plot(rc["step"], rc["pr"], color="#7B1FA2", lw=2.4, label="PR (MLP neurons)")
        axr.set_ylabel("participation ratio  (low=collapsed/symmetric)", color="#7B1FA2")
        axr.axhline(82, color="#D62728", ls=":", lw=1.2); axr.text(rc["step"][-1]*0.99, 86, "trapped PR≈82", ha="right", color="#D62728", fontsize=8.5)
        axr.axhline(202, color="#1D9E75", ls=":", lw=1.2); axr.text(rc["step"][-1]*0.99, 206, "converged PR≈202", ha="right", color="#147a59", fontsize=8.5)
        ax.set_title(f"K={r['K']} seed{r['seed']}  —  loss vs MLP participation ratio", fontsize=12)
        ax2 = axes[1][j]
        ax2.plot(rc["step"], rc["klz"], color="#E68613", lw=2, label="KL_z")
        ax2.plot(rc["step"], rc["klb"], color="#8C2D04", lw=2, label="KL_B")
        ax2.axvline(tau, color="#888", ls="--", lw=1.4)
        ax2.set_ylim(0, 1.6); ax2.set_xlabel("training step"); ax2.set_ylabel("output input-dependence (KL)")
        if j == 0:
            ax2.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Is the snap permutation-symmetry-breaking? MLP neurons collapse (low PR) on the plateau, "
                 "differentiate at τ — coincident with loss drop and KL rise", fontsize=12.5, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"symmetry_timeseries.{ext}", dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nsaved {OUT}/symmetry_timeseries.png / .pdf")


if __name__ == "__main__":
    main()
