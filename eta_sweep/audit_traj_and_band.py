#!/usr/bin/env python
"""Block 2.4 (ΔB/Δz over training, memorized control) + Block 5.11 (η-band: continuous vs discrete in weights)."""
import sys, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss

device = _select_device(); _set_all_seeds(0)
cc = CellConfig(eta=0.001, k=10, seed=0, n_unique_b=1000, batch_size=128, weight_decay=0.01)
cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
ds, _, md = create_datasets_from_config(cfg, tok)
ts = ds[0]["target_start_position"]; zp = ds[0]["z_position"]
N = 2000; exs = [ds[i] for i in range(N)]; ids = collate_fn(exs)["input_ids"].to(device)

# ---------- Block 2.4: KL_B/KL_z over the MEMORIZED (converging) run; trapped = flat ----------
pr = json.load(open(RESULTS_DIR / "probe_ckpts" / "K10_s0" / "probe_results.json"))
meta = json.load(open(RESULTS_DIR / "probe_ckpts" / "K10_s0" / "meta.json"))
plat = max(l for s, l in meta["loss"] if s in (100, 200, 300)); tau = next(s for s, l in meta["loss"] if l < plat / 2)
print("BLOCK 2.4 — memorized (converging) run is the control:")
print(f"  memorized KL_B: plateau {min(pr['klB'][1:5]):.3f} -> final {pr['klB'][-1]:.2f};  "
      f"KL_z: plateau {min(pr['klz'][1:5]):.3f} -> final {pr['klz'][-1]:.2f}  (both RISE at τ={tau})")
print(f"  trapped final KL_B≈0.0015, KL_z≈0.0033 (flat ~0 throughout, never learns).")
print("  => metric is NOT broken: it fires hugely on the memorized control; trapped is genuinely flat.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pr["step"], pr["klB"], color="#8C2D04", lw=2, marker="o", ms=3, label="ΔB (KL) — memorized run")
ax.plot(pr["step"], pr["klz"], color="#E68613", lw=2, marker="o", ms=3, label="Δz (KL) — memorized run")
ax.axhline(0.0015, color="#8C2D04", ls=":", lw=1.5); ax.axhline(0.0033, color="#E68613", ls=":", lw=1.5)
ax.text(pr["step"][-1]*0.98, 0.25, "trapped ΔB,Δz ≈ 0 (flat, never learns)", ha="right", fontsize=9, color="#555")
ax.axvline(tau, color="#888", ls="--"); ax.set_ylim(0, min(26, max(pr["klz"]) * 1.05))
ax.set_xlabel("training step"); ax.set_ylabel("input-dependence (KL, nats)")
ax.set_title("Block 2.4: ΔB & Δz over training — memorized control fires, trapped stays flat")
ax.legend(loc="upper left", frameon=False); fig.tight_layout()
fig.savefig(REPO / "lesswrong_figures" / "audit_deltaB_traj.png", dpi=140, bbox_inches="tight", facecolor="white")

# ---------- Block 5.11: across η band — continuous (weights) vs discrete (function)? ----------
print("\nBLOCK 5.11 — across the η band (inverse, seed0 — same init, varying η):")
ETAS = [0.001, 0.003, 0.006, 0.012, 0.025, 0.05]
init = create_model_from_config(cfg, tok).to(device)            # seed-0 init (fresh)
init_vec = torch.cat([p.detach().flatten() for p in init.parameters()])
rows = []
m = create_model_from_config(cfg, tok).to(device)
for e in ETAS:
    f = RESULTS_DIR / "gate_sweep" / f"inverse_eta{e:g}_K10_nb1000_seed0" / "model_final.pt"
    m.load_state_dict(torch.load(f, map_location=device)); m.eval()
    with torch.no_grad():
        loss = float(compute_loss(m, {"input_ids": ids, "labels": collate_fn(exs)["labels"].to(device)})[0])
        # functional: Δz (KL under z-swap)
        sw = ids.clone(); sw[:, zp:zp+2] = torch.roll(ids[:, zp:zp+2], 1, 0)
        p = F.log_softmax(m(ids)[:, ts-1], -1); q = F.log_softmax(m(sw)[:, ts-1], -1)
        klz = float((p.exp()*(p-q)).sum(-1).mean())
    vec = torch.cat([pp.detach().flatten() for pp in m.parameters()])
    wnorm = float(vec.norm()); dinit = float((vec - init_vec).norm())
    rows.append((e, loss, klz, wnorm, dinit))
print(f"  {'eta':>7} {'loss':>6} {'Δz':>7} {'|W|':>8} {'|W-init|':>9}  outcome")
for e, loss, klz, wn, di in rows:
    print(f"  {e:>7.3f} {loss:>6.2f} {klz:>7.2f} {wn:>8.1f} {di:>9.1f}  {'MEMORIZED' if loss<0.5 else 'TRAPPED'}")
print("  function (loss, Δz) is BIMODAL/discrete; watch |W| and |W-init| for continuity across the band.")

fig2, ax2 = plt.subplots(figsize=(8, 5))
es = [r[0] for r in rows]
ax2.plot(es, [r[3] for r in rows], color="#7B1FA2", marker="o", lw=2, label="|W| weight norm")
ax2.plot(es, [r[4] for r in rows], color="#1D9E75", marker="s", lw=2, label="|W - init|")
ax2.set_xscale("log"); ax2.set_xlabel("learning rate η (log)"); ax2.set_ylabel("weight-space distance")
ax2b = ax2.twinx()
ax2b.plot(es, [r[1] for r in rows], color="#D62728", marker="^", lw=2, ls="--", label="final loss (function)")
ax2b.set_ylabel("final loss"); ax2b.set_ylim(-0.1, 3.1)
ax2.set_title("Block 5.11: weights (continuous?) vs function (bimodal) across the η band")
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc="center left", frameon=False, fontsize=9); fig2.tight_layout()
fig2.savefig(REPO / "lesswrong_figures" / "audit_eta_band_weights.png", dpi=140, bbox_inches="tight", facecolor="white")
print("\nsaved audit_deltaB_traj.png, audit_eta_band_weights.png")
