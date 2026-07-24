#!/usr/bin/env python
"""FALSIFIER: is the QK effective-rank refinement binding-specific or generic training?

Train the FORWARD task (A,z)->B (z redundant, NO binding to form) on the SAME data / architecture /
optimizer as the inverse, differing only in reading direction. Compare its QK effrank trajectory to the
inverse's (from the saved probe checkpoints).
  forward refines IDENTICALLY  -> refinement is generic attention structuring; "u_t = QK refinement" is DEAD.
  forward stays flat / refines much less -> refinement is binding-specific; the identification survives.
"""
import sys, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss
from eta_sweep.gate_mirror_runner import build           # build(cc, direction, tok)

SCHED = [0, 100, 200, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2400, 2800, 3500, 5000]


def eff_rank(M):
    s = torch.linalg.svdvals(M.float().cpu()); p = s / s.sum(); p = p[p > 1e-12]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def mean_qk(model):
    L, H = model.cfg.n_layers, model.cfg.n_heads
    return sum(eff_rank(model.W_Q[l, h] @ model.W_K[l, h].T) for l in range(L) for h in range(H)) / (L * H)


def main():
    device = _select_device()
    sched = set(SCHED)

    # ---------- FORWARD: train inline, record QK rank ----------
    _set_all_seeds(0)
    cc = CellConfig(eta=0.001, k=10, seed=0, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    train_ds, md = build(cc, "forward", tok)              # forward (A,z)->B
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tok).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    fsteps, floss, fqk = [0], [float("nan")], [mean_qk(model)]
    step = 0
    while step < 5000:
        for b in loader:
            if step >= 5000:
                break
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            l, _, _ = compute_loss(model, b); l.backward(); opt.step(); step += 1
            if step in sched:
                fsteps.append(step); floss.append(float(l)); fqk.append(mean_qk(model)); model.train()

    # ---------- INVERSE: QK rank from saved checkpoints ----------
    ckdir = RESULTS_DIR / "probe_ckpts" / "K10_s0"
    pr = json.load(open(ckdir / "probe_results.json"))
    inv = create_model_from_config(cfg, tok).to(device)
    isteps = pr["step"]; iqk = []
    for st in isteps:
        inv.load_state_dict(torch.load(ckdir / f"step{st}.pt", map_location=device)); iqk.append(mean_qk(inv))
    iloss = pr["loss"]

    # align on common steps
    common = [s for s in isteps if s in fsteps]
    fmap = {s: q for s, q in zip(fsteps, fqk)}; imap = {s: q for s, q in zip(isteps, iqk)}
    print(f"{'step':>5} {'fwd_QK':>7} {'inv_QK':>7} {'fwd-inv':>8}")
    for s in common:
        print(f"{s:>5} {fmap[s]:>7.2f} {imap[s]:>7.2f} {fmap[s]-imap[s]:>+8.2f}")
    f0, fF = fmap[common[0]], fmap[common[-1]]; i0, iF = imap[common[0]], imap[common[-1]]
    print(f"\nforward QK: init {f0:.2f} -> final {fF:.2f}  (Δ {fF-f0:+.2f})")
    print(f"inverse QK: init {i0:.2f} -> final {iF:.2f}  (Δ {iF-i0:+.2f})")
    fwd_drop, inv_drop = f0 - min(fmap[s] for s in common), i0 - min(imap[s] for s in common)
    print(f"forward refines by {fwd_drop:.2f}  vs  inverse {inv_drop:.2f}")
    if fwd_drop > 0.6 * inv_drop:
        print("VERDICT: forward refines comparably -> refinement is GENERIC attention structuring. "
              "'u_t = QK refinement' NOT supported; the ramp is unidentified.")
    else:
        print("VERDICT: forward refines much less -> refinement is BINDING-SPECIFIC; identification survives.")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(fsteps, fqk, color="#1D9E75", lw=2.4, marker="o", ms=3, label="QK effrank — FORWARD (no binding)")
    ax.plot(isteps, iqk, color="#D62728", lw=2.4, marker="s", ms=3, label="QK effrank — INVERSE (binding)")
    ax.set_ylabel("QK effective rank"); ax.set_xlabel("training step"); ax.set_ylim(24, 30.5)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Falsifier: does the QK rank refine the same WITHOUT a binding to form?")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "lesswrong_figures" / f"forward_qk_control.{ext}", dpi=140, bbox_inches="tight", facecolor="white")
    print("saved lesswrong_figures/forward_qk_control.png")


if __name__ == "__main__":
    main()
