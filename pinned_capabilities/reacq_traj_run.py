"""Trajectory runner with DENSE checkpoint saving for the diagnostics trajectory
phases (1D escape directions, Phase 2 early/late v-only states, Phase 3
amplification-vs-first-passage). Reuses the verified reacq AdamW + routing
WITHOUT modifying any registered file. From the collapsed fork (slot_0). Shared
noise seeds across V/N/N_PM = common random numbers. Saves (theta,m,v,t_eff)
every CK steps to traj/{ARM}_sc{SCALE}/ckpt_{step}.pt.

usage: reacq_traj_run.py ARM T CK [SCALE]   ARM in {C0,V,N,N_PM}
"""
import sys, math, json
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch

ARM = sys.argv[1]; T = int(sys.argv[2]); CK = int(sys.argv[3])
SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
sys.argv = ["x", ARM, "collapsed", "0", "5", "mps", "6", str(SCALE)]   # module import (mps)
import pinned_capabilities.reacq_2x2_run as R
B1, B2, EPS, LR, WD = R.B1, R.B2, R.EPS, R.LR, R.WD
DEV = "mps"
OUT = R.REPO / f"pinned_capabilities/results/traj/{ARM}_sc{SCALE:.2f}"
OUT.mkdir(parents=True, exist_ok=True)


def save(step, params, m, v, t_eff, ngen, mbgen):
    torch.save({"step": step, "theta": [p.data.detach().cpu() for p in params],
                "m": [x.detach().cpu() for x in m], "v": [x.detach().cpu() for x in v], "t_eff": t_eff,
                "ngen": ngen.get_state(), "mbgen": mbgen.get_state()},
               OUT / f"ckpt_{step:06d}.pt")


def run():
    exp, params, m, v, t0, n, chunks = R.build()
    ngen = torch.Generator().manual_seed(1_000_000)           # SHARED across V/N/N_PM = CRN
    mbgen = torch.Generator().manual_seed(2_000_000)
    # resume from the latest trajectory checkpoint if present (lid-close safe)
    existing = sorted(OUT.glob("ckpt_*.pt"))
    start = 1
    if len(existing) > 1:                                     # >1 because ckpt_000000 always exists
        st = torch.load(existing[-1], map_location="cpu")
        for i, p in enumerate(params): p.data.copy_(st["theta"][i].to(DEV))
        m = [x.to(DEV) for x in st["m"]]; v = [x.to(DEV) for x in st["v"]]
        ngen.set_state(st["ngen"]); mbgen.set_state(st["mbgen"])
        start = st["step"] + 1
        writer = (OUT / "metrics.jsonl").open("a")
    else:
        save(0, params, m, v, t0, ngen, mbgen)                # theta0 = collapsed fork
        writer = (OUT / "metrics.jsonl").open("w")

    def mb_grad():
        idx = torch.randint(n, (128,), generator=mbgen)
        b = R.to_dev(R.collate_fn([exp.dataset[int(i)] for i in idx]))
        for p in params: p.grad = None
        R.compute_loss(exp.model, b)[0].backward()
        return [p.grad.detach().clone().float() for p in params]

    for step in range(start, T + 1):
        g = R.full_grad(exp, params, chunks); g2 = [x * x for x in g]
        if ARM == "C0":
            m_src, v_inc = g, g2
        else:
            ref = [(a - b) / math.sqrt(2) for a, b in zip(mb_grad(), mb_grad())]
            iso = [torch.randn(p.shape, generator=ngen).to(DEV) for p in params]
            sc = SCALE * R.flat_norm(ref) / (R.flat_norm(iso) + 1e-30)
            gpx = [a + sc * e for a, e in zip(g, iso)]; gpx2 = [x * x for x in gpx]
            m_src, v_inc = R.route_noisy(ARM, g, g2, gpx, gpx2)
        t_eff = t0 + step
        for i in range(len(m)):
            m[i].mul_(B1).add_(m_src[i], alpha=1 - B1); v[i].mul_(B2).add_(v_inc[i], alpha=1 - B2)
        u = R.bias_corrected_update(m, v, t_eff)
        for i, p in enumerate(params):
            p.data.mul_(1 - LR * WD).add_(u[i])
        if step % CK == 0:
            save(step, params, m, v, t_eff, ngen, mbgen)
        if step % 200 == 0:
            ce = R.ce_full(exp, chunks)
            writer.write(json.dumps({"step": step, "full_vocab_ce": ce}) + "\n"); writer.flush()
    writer.close()
    ck_steps = sorted(int(p.stem.split("_")[1]) for p in OUT.glob("ckpt_*.pt"))
    (OUT / "done.json").write_text(json.dumps({"arm": ARM, "scale": SCALE, "T": T, "CK": CK, "ckpts": ck_steps}))
    print(f"{ARM} sc{SCALE} traj done: {T} steps, {len(ck_steps)} ckpts")


if __name__ == "__main__":
    run()
