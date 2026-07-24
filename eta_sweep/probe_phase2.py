#!/usr/bin/env python
"""Phase 2: precondition gate + linear probes through the plateau, on the Phase-1 checkpoints.

Precondition (gate): raw swap-logit norm ||logits(orig)-logits(swap)|| on answer positions, for B-swap
and z-swap, alongside KL_B/KL_z. If raw norm is pinned ~0 through the plateau, output flatness is genuine
and the probes are interpretable. If it is visibly rising while KL stays flat -> report and stop.

Probes (LINEAR only, held-out eval split): from frozen hidden state at position ts-1 (predicts A[0]),
at several residual-stream sites, train linear probes for:
  marginal-B (first B char) | marginal-z (which of K z) | joint-A (first answer token A[0], needs B x z).
A linear probe CANNOT manufacture the B x z interaction, so joint-A succeeds only if the interaction is
linearly present in the representation. H1: joint-A rises during the plateau (silent circuit). H2: joint-A
flat until tau (feature absent). Controls: solved-checkpoint positive control (all 3 must fire) and init floor.

Usage: python probe_phase2.py --K 10
"""
import sys, json, argparse, math
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

SITES = [1, 2, 3]               # resid_post layers to probe
NTR, NEV = 4000, 2000


def swap_logit_norm(model, ids, ans_pos, lo, hi):
    with torch.no_grad():
        base = model(ids)
        sw = ids.clone(); sw[:, lo:hi] = torch.roll(ids[:, lo:hi], 1, 0)
        alt = model(sw)
        d = (base[:, ans_pos, :] - alt[:, ans_pos, :])           # (N, |ans|, V)
        return d.flatten(1).norm(dim=1).mean().item()            # mean over batch of the logit-move norm


def kl_first(model, ids_a, ids_b, tsm1):
    with torch.no_grad():
        pa = F.log_softmax(model(ids_a)[:, tsm1, :], -1)
        pb = F.log_softmax(model(ids_b)[:, tsm1, :], -1)
        return (pa.exp() * (pa - pb)).sum(-1).mean().item()


def hiddens(model, ids, pos, layers):
    names = lambda n: any(n == f"blocks.{l}.hook_resid_post" for l in layers)
    with torch.no_grad():
        _, c = model.run_with_cache(ids, names_filter=names)
    return {l: c[f"blocks.{l}.hook_resid_post"][:, pos, :].float() for l in layers}


def train_probe(Htr, ytr, Hev, yev, C, device, steps=300):
    mu, sd = Htr.mean(0, keepdim=True), Htr.std(0, keepdim=True) + 1e-6
    Htr = ((Htr - mu) / sd).to(device); Hev = ((Hev - mu) / sd).to(device)
    ytr = ytr.to(device); yev = yev.to(device)
    W = torch.nn.Linear(Htr.size(1), C).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(W(Htr), ytr).backward(); opt.step()
    with torch.no_grad():
        return (W(Hev).argmax(-1) == yev).float().mean().item()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=10); a = ap.parse_args()
    K, seed = a.K, 0
    device = _select_device()
    ckdir = RESULTS_DIR / "probe_ckpts" / f"K{K}_s{seed}"
    meta = json.load(open(ckdir / "meta.json")); steps = meta["ckpts"]
    loss_by_step = {s: l for s, l in meta["loss"]}
    plat = max(loss_by_step.get(s, 0) for s in (100, 200, 300))
    tau = next((s for s, l in meta["loss"] if l < plat / 2), meta["loss"][-1][0])

    _set_all_seeds(seed)
    cc = CellConfig(eta=0.001, k=K, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    ds, _, md = create_datasets_from_config(cfg, tok)
    model = create_model_from_config(cfg, tok).to(device)
    inv_meta = []  # (b_idx, z_idx) per example, in dataset order
    # rebuild inv_meta from the dataset metadata
    z0 = ds[0]["z_position"]; ts = ds[0]["target_start_position"]; te = ds[0]["target_end_position"]
    tsm1 = ts - 1
    # eval batches
    tr = [ds[i] for i in range(NTR)]; ev = [ds[i] for i in range(NTR, NTR + NEV)]
    ids_tr = collate_fn(tr)["input_ids"].to(device); ids_ev = collate_fn(ev)["input_ids"].to(device)
    yB_tr, yB_ev = ids_tr[:, 1], ids_ev[:, 1]                        # marginal-B: first B char
    yA_tr, yA_ev = ids_tr[:, ts], ids_ev[:, ts]                      # joint-A: first answer token
    yz_tr = torch.tensor([tr[i]["z_index"] if "z_index" in tr[i] else 0 for i in range(NTR)])
    yz_ev = torch.tensor([ev[i]["z_index"] if "z_index" in ev[i] else 0 for i in range(NEV)])
    has_zidx = "z_index" in ds[0]
    if not has_zidx:   # fall back: z identity from the z tokens themselves
        zt_tr = collate_fn(tr)["input_ids"][:, z0:z0 + 2]; zt_ev = collate_fn(ev)["input_ids"][:, z0:z0 + 2]
        uniq = {tuple(t.tolist()): i for i, t in enumerate(torch.unique(zt_tr, dim=0))}
        yz_tr = torch.tensor([uniq.get(tuple(t.tolist()), 0) for t in zt_tr])
        yz_ev = torch.tensor([uniq.get(tuple(t.tolist()), 0) for t in zt_ev])
    Cz = int(max(yz_tr.max(), yz_ev.max()) + 1); V = int(model.cfg.d_vocab)
    ans_pos = list(range(ts - 1, te))                                # logit positions that predict the answer

    # swaps for precondition
    bz = ids_tr.clone(); bz[:, z0:z0 + 2] = torch.roll(ids_tr[:, z0:z0 + 2], 1, 0)
    bb = ids_tr.clone(); bb[:, 1:z0 - 1] = torch.roll(ids_tr[:, 1:z0 - 1], 1, 0)

    print(f"K={K} seed{seed}  tau(loss-mid)={tau}  sites(resid_post)={SITES}  probe pos={tsm1}  Cz={Cz}\n")
    rec = {"step": [], "loss": [], "rawB": [], "rawz": [], "klB": [], "klz": [],
           **{f"B@{l}": [] for l in SITES}, **{f"z@{l}": [] for l in SITES}, **{f"A@{l}": [] for l in SITES}}
    print(f"{'step':>5} {'loss':>6} {'rawB':>7} {'rawz':>7} {'KL_B':>6} {'KL_z':>6} | "
          + " ".join(f"A@{l}" for l in SITES) + " | " + " ".join(f"z@{l}" for l in SITES) + " | " + " ".join(f"B@{l}" for l in SITES))
    for s in steps:
        f = ckdir / f"step{s}.pt"
        if not f.exists():
            continue
        model.load_state_dict(torch.load(f, map_location=device)); model.eval()
        rb = swap_logit_norm(model, ids_tr, ans_pos, 1, z0 - 1)
        rz = swap_logit_norm(model, ids_tr, ans_pos, z0, z0 + 2)
        kB = kl_first(model, ids_tr, bb, tsm1); kz = kl_first(model, ids_tr, bz, tsm1)
        Htr = hiddens(model, ids_tr, tsm1, SITES); Hev = hiddens(model, ids_ev, tsm1, SITES)
        row = dict(step=s, loss=loss_by_step.get(s, float("nan")), rawB=rb, rawz=rz, klB=kB, klz=kz)
        for l in SITES:
            row[f"A@{l}"] = train_probe(Htr[l], yA_tr, Hev[l], yA_ev, V, device)
            row[f"z@{l}"] = train_probe(Htr[l], yz_tr, Hev[l], yz_ev, Cz, device)
            row[f"B@{l}"] = train_probe(Htr[l], yB_tr, Hev[l], yB_ev, V, device)
        for k in rec:
            rec[k].append(row[k])
        print(f"{s:>5} {row['loss']:>6.2f} {rb:>7.3f} {rz:>7.3f} {kB:>6.3f} {kz:>6.3f} | "
              + " ".join(f"{row[f'A@{l}']:.2f}" for l in SITES) + " | "
              + " ".join(f"{row[f'z@{l}']:.2f}" for l in SITES) + " | "
              + " ".join(f"{row[f'B@{l}']:.2f}" for l in SITES))

    json.dump(rec, open(ckdir / "probe_results.json", "w"))
    # pick best site by solved-checkpoint A accuracy
    best = max(SITES, key=lambda l: rec[f"A@{l}"][-1])
    print(f"\nPOSITIVE CONTROL (solved step {rec['step'][-1]}): "
          f"A@{best}={rec[f'A@{best}'][-1]:.2f}  z@{best}={rec[f'z@{best}'][-1]:.2f}  B@{best}={rec[f'B@{best}'][-1]:.2f}")
    print(f"INIT FLOOR (step 0):  A@{best}={rec[f'A@{best}'][0]:.2f}  z@{best}={rec[f'z@{best}'][0]:.2f}  B@{best}={rec[f'B@{best}'][0]:.2f}")

    # figure
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axT.plot(rec["step"], rec["loss"], color="#1F77B4", lw=2, label="loss")
    axT.set_ylabel("loss / output dependence"); axT.axvline(tau, color="#888", ls="--")
    axTr = axT.twinx()
    axTr.plot(rec["step"], rec["rawB"], color="#8C2D04", lw=1.6, label="raw ΔlogitB")
    axTr.plot(rec["step"], rec["rawz"], color="#E68613", lw=1.6, label="raw Δlogitz")
    axTr.set_ylabel("raw swap-logit norm")
    axT.set_title(f"K={K}: precondition (raw swap-logit) + loss   τ={tau}")
    for l, c in zip(SITES, ["#1D9E75", "#7B1FA2", "#D62728"]):
        axB.plot(rec["step"], rec[f"A@{l}"], color=c, lw=2.2, label=f"joint-A probe @L{l}")
        axB.plot(rec["step"], rec[f"z@{l}"], color=c, lw=1.2, ls="--", alpha=0.7)
        axB.plot(rec["step"], rec[f"B@{l}"], color=c, lw=1.0, ls=":", alpha=0.6)
    axB.axvline(tau, color="#888", ls="--"); axB.set_ylim(0, 1.02)
    axB.set_xlabel("training step"); axB.set_ylabel("held-out probe accuracy")
    axB.set_title("solid=joint-A  dashed=marginal-z  dotted=marginal-B  (per layer site)")
    axB.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "lesswrong_figures" / f"probe_K{K}.{ext}", dpi=140, bbox_inches="tight", facecolor="white")
    print(f"saved lesswrong_figures/probe_K{K}.png")

    # printout summary
    def slope(key):
        idx = [i for i, st in enumerate(rec["step"]) if st <= tau]
        return rec[key][idx[-1]] - rec[key][1] if len(idx) > 2 else float("nan")
    Across = next((st for st, v in zip(rec["step"], rec[f"A@{best}"]) if v > rec[f"A@{best}"][0] + 0.2), None)
    print(f"\nSUMMARY K={K}: raw swap-logit plateau norm ≈ {min(rec['rawB'][1:4]+rec['rawz'][1:4]):.3f}")
    print(f"  joint-A@{best} plateau slope (to τ) = {slope(f'A@{best}'):+.2f}  -> {'RISING (H1)' if slope(f'A@{best}')>0.15 else 'FLAT (H2)'}")
    print(f"  marginal-z@{best} plateau slope = {slope(f'z@{best}'):+.2f}   marginal-B@{best} plateau slope = {slope(f'B@{best}'):+.2f}")
    print(f"  joint-A crossing step = {Across}  vs  τ = {tau}")


if __name__ == "__main__":
    main()
