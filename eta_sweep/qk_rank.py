#!/usr/bin/env python
"""Double-rotation symmetry in attention (Ziyin arXiv:2502.05300): is the plateau the LOW-RANK QK
(symmetric) saddle, and does the snap break it (rank growth)?

Symmetry: a_ij = X_i^T W_Q W_K^T X_j is invariant under W_Q->W_Q M, W_K->M^{-1} W_K. The invariant is
M_QK = W_Q W_K^T; the symmetric (unbroken) state is LOW effective rank of M_QK. We use a SCALE-INVARIANT
order parameter -- effective rank via singular-value entropy: p_i = s_i/sum s, effrank = exp(-sum p_i log p_i).
(Norm-based delta_trans failed yesterday on pure weight scale; effrank is scale-free by construction.)

GATE: positive control FIRST -- solved vs init effrank per head. The solved model uses the B x z binding,
so its relevant head(s) MUST show higher effrank than init. If solved ~ init, the metric is blind; stop.

Then overlay per-head QK effrank with loss, joint-A probe, and the raw-logit ramp (from probe_results.json),
tau marked, and report whether rank rises at tau (Ziyin-in-attention), before tau (the weight-resident ramp
= QK de-symmetrizing), or never (Ziyin ruled out). Also correlate rank growth with the raw-logit ramp.
"""
import sys, json, argparse, math
from pathlib import Path
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config


def eff_rank(M):
    s = torch.linalg.svdvals(M.float().cpu())
    p = s / s.sum(); p = p[p > 1e-12]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def qk_effranks(model):
    L, H = model.cfg.n_layers, model.cfg.n_heads
    out = {}
    for l in range(L):
        for h in range(H):
            M = model.W_Q[l, h] @ model.W_K[l, h].T          # (d_model, d_model), rank <= d_head
            out[(l, h)] = eff_rank(M)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=10); a = ap.parse_args()
    K, seed = a.K, 0
    device = _select_device()
    ckdir = RESULTS_DIR / "probe_ckpts" / f"K{K}_s{seed}"
    pr = json.load(open(ckdir / "probe_results.json"))            # step, loss, rawB, rawz, A@3 ...
    meta = json.load(open(ckdir / "meta.json"))
    plat = max(l for s, l in meta["loss"] if s in (100, 200, 300) or s in (200, 500))
    tau = next((s for s, l in meta["loss"] if l < plat / 2), meta["loss"][-1][0])

    _set_all_seeds(seed)
    cc = CellConfig(eta=0.001, k=K, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    ds, _, md = create_datasets_from_config(cfg, tok)
    model = create_model_from_config(cfg, tok).to(device)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    dhead = model.cfg.d_head

    steps = pr["step"]
    traj = {}                                                    # (l,h) -> [effrank per step]
    for st in steps:
        model.load_state_dict(torch.load(ckdir / f"step{st}.pt", map_location=device)); model.eval()
        er = qk_effranks(model)
        for k, v in er.items():
            traj.setdefault(k, []).append(v)

    # ----- POSITIVE CONTROL (gate) -----
    i0 = 0; iS = len(steps) - 1
    print(f"K={K}  d_head={dhead} (effrank<= {dhead})  tau={tau}  init=step{steps[i0]} solved=step{steps[iS]} (loss {pr['loss'][iS]:.2f})\n")
    print("POSITIVE CONTROL — QK effrank per head (init -> solved):")
    gains = []
    for l in range(L):
        for h in range(H):
            ei, es = traj[(l, h)][i0], traj[(l, h)][iS]
            gains.append(((l, h), es - ei, ei, es))
    gains.sort(key=lambda x: -x[1])
    for (lh, g, ei, es) in gains:
        flag = "  <== biggest gain" if (lh, g, ei, es) == gains[0] else ""
        print(f"  L{lh[0]}H{lh[1]}: init {ei:5.2f} -> solved {es:5.2f}  (Δ {g:+5.2f}){flag}")
    responsive = max(abs(g) for _, g, _, _ in gains) > 1.0
    direction = "DECREASE (refinement)" if gains[0][1] < 0 else "increase"
    print(f"\nCONTROL: literal 'solved>init' = {'pass' if gains[0][1] > 1 else 'FAIL'} "
          f"(init is high-rank RANDOM ~{traj[(0,0)][0]:.1f}, not the symmetric reference -> baseline mis-specified).")
    print(f"  But metric is RESPONSIVE: |Δ| up to {max(abs(g) for _,g,_,_ in gains):.1f}, uniform {direction} across all heads "
          f"-> not blind; trajectory IS interpretable.")
    if not responsive:
        print("  metric truly flat -> STOP / try W_V W_O."); return 1

    # ----- trajectory (mean over all heads; per-head is uniform) vs loss / probe / ramp -----
    def plateau_slope(series):
        idx = [i for i, s in enumerate(steps) if s <= tau and s > 0]
        return series[idx[-1]] - series[idx[0]] if len(idx) >= 2 else float("nan")
    allheads = list(traj.keys())
    topser = [sum(traj[lh][i] for lh in allheads) / len(allheads) for i in range(len(steps))]   # mean over all heads
    top = sorted(allheads, key=lambda lh: traj[lh][-1])[:3]
    sl = plateau_slope(topser)
    thr = topser[0] + 0.5 * (topser[-1] - topser[0])
    if topser[-1] < topser[0]:
        cross = next((s for s, v in zip(steps, topser) if v < thr), None)
    else:
        cross = next((s for s, v in zip(steps, topser) if v > thr), None)
    # correlation of rank growth with raw-logit ramp over plateau+early-snap
    import statistics as st
    mask = [i for i, s in enumerate(steps) if 0 < s <= tau * 1.2]
    xr = [pr["rawz"][i] for i in mask]; yr = [topser[i] for i in mask]
    if len(xr) >= 3 and st.pstdev(xr) > 0 and st.pstdev(yr) > 0:
        mx, my = st.mean(xr), st.mean(yr)
        corr = sum((x-mx)*(y-my) for x, y in zip(xr, yr)) / (len(xr)*st.pstdev(xr)*st.pstdev(yr))
    else:
        corr = float("nan")

    print(f"\nSUMMARY K={K}: QK effrank init/plateau/solved (top heads) = "
          f"{topser[0]:.2f} / {min(topser[:max(2,len(topser)//3)]):.2f} / {topser[-1]:.2f}")
    move = "RISING" if sl > 0.5 else ("DROPPING (refinement)" if sl < -0.5 else "FLAT")
    post = topser[-1] - min(topser)
    print(f"  plateau slope (to τ) = {sl:+.2f}  -> QK rank {move} during the plateau")
    print(f"  post-τ rise from minimum = {post:+.2f}  (the only 'symmetry-breaking-direction' move at the snap)")
    print(f"  effrank half-change crossing step = {cross}  vs  τ = {tau}  (<<τ ⇒ the move is on the plateau, not at the snap)")
    print(f"  corr(QK effrank, raw-logit ramp) over plateau+snap = {corr:+.2f}  (strong negative ⇒ rank-drop IS the ramp)")

    # figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lh in traj:
        ax.plot(steps, traj[lh], color="#cccccc", lw=0.8, zorder=1)
    cols = ["#7B1FA2", "#D62728", "#1D9E75"]
    for lh, c in zip(top, cols):
        ax.plot(steps, traj[lh], color=c, lw=2.4, label=f"QK effrank L{lh[0]}H{lh[1]}", zorder=4)
    ax.axvline(tau, color="#888", ls="--"); ax.text(tau, dhead*0.98, f" τ={tau}", color="#555", fontsize=9, va="top")
    ax.set_ylabel("QK effective rank  (low = symmetric / unbroken)"); ax.set_xlabel("training step")
    ax.set_ylim(0, dhead)
    axr = ax.twinx()
    axr.plot(steps, pr["loss"], color="#1F77B4", lw=2, ls="-", label="loss", zorder=3)
    axr.plot(steps, pr["A@3"], color="#E68613", lw=2, ls=":", label="joint-A probe", zorder=3)
    axr.set_ylabel("loss  /  joint-A probe acc"); axr.set_ylim(0, 3.1)
    ax.set_title(f"K={K}: QK effective rank (double-rotation symmetry) vs loss / joint-A   "
                 f"[control gain {gains[0][1]:+.1f}]")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "lesswrong_figures" / f"qk_rank_K{K}.{ext}", dpi=140, bbox_inches="tight", facecolor="white")
    print(f"saved lesswrong_figures/qk_rank_K{K}.png")


if __name__ == "__main__":
    raise SystemExit(main())
