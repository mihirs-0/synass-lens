#!/usr/bin/env python
"""Characterize the plateau on PLAIN full-vocab CE (candidate loss retired).

For each converging inverse run (B,z->A, eta=1e-3), log over training:
  * plain train CE  = full-vocabulary cross-entropy on the answer tokens (compute_loss; the headline curve)
  * KL_B, KL_z      = input-dependence diagnostics (same as the plateau timeseries), on the same checkpoints

and draw three reference floors computed ONCE:
  * uniform-over-vocab  = log(d_vocab)                          ("knows nothing, not even that answers are chars")
  * input-blind floor   = mean over target positions of H(empirical token marginal at that position),
                          computed directly from the data (best predictor that ignores B and z)
  * solved              = 0

Decisive question: does the plain-CE plateau equal the input-blind floor?  If yes, the plateau is the model
sitting at the best input-independent predictor -> a meaningful, non-circular floor and the honest replacement
for the log-K story.  candidate loss is NOT computed/logged/plotted anywhere here.
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

CONFIGS = [(10, 0), (10, 1), (20, 0)]
OUT = REPO / "lesswrong_figures"


def schedule(maxstep):
    s = list(range(0, 3001, 50)) + list(range(3250, maxstep + 1, 250))
    return set(s)


def input_blind_floor(ds, n):
    """Mean over target positions of the entropy of the empirical token marginal at that position."""
    exs = [ds[i] for i in range(min(n, len(ds)))]
    b = collate_fn(exs)
    ids, labels = b["input_ids"], b["labels"]
    positions = [q for q in range(labels.size(1)) if (labels[:, q] != -100).any()]
    Hs = []
    for q in positions:
        toks = ids[:, q]
        c = torch.bincount(toks).float(); p = c[c > 0] / c.sum()
        Hs.append(float(-(p * p.log()).sum()))
    return sum(Hs) / len(Hs), positions, Hs


def run(K, seed, device):
    _set_all_seeds(seed)
    cc = CellConfig(eta=0.001, k=K, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    train_ds, _, md = create_datasets_from_config(cfg, tok)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tok).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    vocab = int(model.cfg.d_vocab)
    blind, positions, Hs = input_blind_floor(train_ds, 8000)
    unif = math.log(vocab)

    N = 2000
    exs = [train_ds[i] for i in range(N)]
    eb = collate_fn(exs); ids = eb["input_ids"]
    ex0 = exs[0]; zp, ze = ex0["z_position"], ex0["z_end_position"]; ts = ex0["target_start_position"]
    bz = dict(eb); bz["input_ids"] = roll_span(ids, zp, ze + 1)
    bb = dict(eb); bb["input_ids"] = roll_span(ids, 1, zp - 1)

    def ce(batch):
        d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            l, _, _ = compute_loss(model, d)
        return float(l)

    def snap():
        model.eval()
        c = ce(eb)
        lc = first_logp(model, eb, device, ts)
        klz = mean_kl(lc, first_logp(model, bz, device, ts))
        klb = mean_kl(lc, first_logp(model, bb, device, ts))
        model.train()
        return c, klz, klb

    maxstep = 5000 if K <= 10 else 9000
    sched = schedule(maxstep)
    rec = {"step": [], "ce": [], "klz": [], "klb": []}

    def log(step):
        c, klz, klb = snap()
        rec["step"].append(step); rec["ce"].append(c); rec["klz"].append(klz); rec["klb"].append(klb)

    log(0); step = 0
    while step < maxstep:
        for batch in loader:
            if step >= maxstep:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            l, _, _ = compute_loss(model, batch); l.backward(); opt.step(); step += 1
            if step in sched:
                log(step)
    plat = [c for s, c in zip(rec["step"], rec["ce"]) if 150 <= s <= 1000]
    plateau_ce = sum(plat) / len(plat)
    print(f"K={K} seed{seed}: plateau plain-CE value = {plateau_ce:.2f}; input-blind floor = {blind:.2f}; "
          f"uniform-vocab = {unif:.2f}; match = {'yes' if abs(plateau_ce-blind) < 0.12 else 'NO'}")
    return dict(K=K, seed=seed, rec=rec, blind=blind, unif=unif, vocab=vocab, plateau_ce=plateau_ce)


def main():
    device = _select_device()
    print(f"device={device}  (candidate loss RETIRED — plain full-vocab CE only)\n")
    res = [run(K, s, device) for K, s in CONFIGS]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex="col")
    for j, r in enumerate(res):
        rc = r["rec"]; ax = axes[0][j]
        ax.plot(rc["step"], rc["ce"], color="#1F77B4", lw=2.2, label="plain train CE (full vocab)")
        ax.axhline(r["unif"], ls="--", lw=1.3, color="#9a9a9a")
        ax.axhline(r["blind"], ls="--", lw=1.6, color="#D62728")
        ax.axhline(0, ls="--", lw=1.3, color="#1D9E75")
        xr = max(rc["step"])
        ax.text(xr*0.98, r["unif"]+0.04, f"uniform-over-vocab  log V = {r['unif']:.2f}", ha="right", va="bottom", fontsize=9, color="#6a6a6a")
        ax.text(xr*0.98, r["blind"]+0.04, f"input-blind floor = {r['blind']:.2f}", ha="right", va="bottom", fontsize=9.5, color="#D62728", fontweight="bold")
        ax.text(xr*0.98, 0.06, "solved = 0", ha="right", va="bottom", fontsize=9, color="#147a59")
        m = abs(r["plateau_ce"]-r["blind"]) < 0.12
        ax.set_title(f"K={r['K']}  seed{r['seed']}   (plateau {r['plateau_ce']:.2f} {'=' if m else '≠'} blind {r['blind']:.2f})", fontsize=12)
        ax.set_ylim(0, 3.3); ax.grid(alpha=0.15)
        if j == 0:
            ax.set_ylabel("answer-region CE (nats)", fontsize=12); ax.legend(loc="center right", fontsize=9, frameon=False)
        ax2 = axes[1][j]
        ax2.plot(rc["step"], rc["klz"], color="#E68613", lw=2, label="KL_z (z-dependence)")
        ax2.plot(rc["step"], rc["klb"], color="#8C2D04", lw=2, label="KL_B (B-dependence)")
        ax2.set_ylim(0, 1.6); ax2.set_xlabel("training step", fontsize=12); ax2.grid(alpha=0.15)
        if j == 0:
            ax2.set_ylabel("input-dependence (KL, nats)", fontsize=12); ax2.legend(loc="upper left", fontsize=9, frameon=False)
        ax2.text(xr*0.97, 1.5, "≈0 through plateau,\nrises at the snap", ha="right", va="top", fontsize=9, color="#555")
    fig.suptitle("The plateau is the input-blind floor: plain CE parks at the best B,z-independent predictor "
                 "while KL_B, KL_z ≈ 0", fontsize=13.5, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"plateau_plaince.{ext}", dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nsaved {OUT}/plateau_plaince.png / .pdf")


if __name__ == "__main__":
    main()
