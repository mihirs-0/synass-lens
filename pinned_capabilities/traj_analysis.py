"""Trajectory analysis: Phase 1D (escape-direction stability), Phase 3
(amplification vs first-passage: s_t, a_t, P_T along each arm), Phase 2
(conditional update at early/late v-only states). Reads dense checkpoints from
results/traj/. Escape direction e from C0 clean pre-onset checkpoints (NOT the
solved-state displacement). CPU. usage: traj_analysis.py"""
import json, sys, math
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch
sys.argv = ["x", "V", "collapsed", "0", "5", "cpu", "1", "1.0"]
import pinned_capabilities.reacq_2x2_run as R

T = R.REPO / "pinned_capabilities/results/traj"
OUT = R.REPO / "pinned_capabilities/results/cond_update"
B1, B2, EPS, LR, WD = R.B1, R.B2, R.EPS, R.LR, R.WD
ARMS = {"C0": "C0_sc1.00", "V": "V_sc1.00", "N": "N_sc1.00", "N_PM": "N_PM_sc0.60"}


def flatten(theta): return torch.cat([x.reshape(-1) for x in theta])
def ckpts(arm): return sorted(T.joinpath(ARMS[arm]).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
def load(path): return torch.load(path, map_location="cpu")
def onset(arm):
    rows = [json.loads(l) for l in T.joinpath(ARMS[arm], "metrics.jsonl").open() if l.strip()]
    ce = [(r["step"], r["full_vocab_ce"]) for r in rows]
    for i, (s, c) in enumerate(ce):
        if c < 3.0 and all(cc < 3.0 for ss, cc in ce[i:] if ss <= s + 500):
            return s
    return None


def main():
    exp, params, m0, v0, t0, n, chunks = R.build()
    theta0 = load(ckpts("C0")[0])["theta"]; f0 = flatten(theta0)

    def set_theta(theta):
        for i, p in enumerate(params): p.data.copy_(theta[i])
    def grad_at(theta):
        set_theta(theta); return flatten(R.full_grad(exp, params, chunks))

    # ---------- Phase 1D: escape direction from C0 pre-onset ----------
    on_c0 = onset("C0")
    cks = ckpts("C0")
    steps = [int(p.stem.split("_")[1]) for p in cks]
    pre = [(s, p) for s, p in zip(steps, cks) if s > 0 and (on_c0 is None or s <= on_c0)]
    pre = pre[-8:]                                            # several immediately before onset
    e_dirs = []
    for s, p in pre:
        d = flatten(load(p)["theta"]) - f0
        e_dirs.append((s, d / (d.norm() + 1e-30)))
    import itertools
    cos_pairs = [float((a[1] * b[1]).sum()) for a, b in itertools.combinations(e_dirs, 2)]
    e = torch.stack([d for _, d in e_dirs]).mean(0); e = e / (e.norm() + 1e-30)   # stable mean direction
    p1d = {"C0_onset": on_c0, "pre_onset_steps": [s for s, _ in pre],
           "pairwise_cos_median": float(torch.tensor(cos_pairs).median()) if cos_pairs else None,
           "pairwise_cos_min": float(min(cos_pairs)) if cos_pairs else None}
    print("[1D] escape-dir:", json.dumps(p1d))

    # ---------- Phase 3: s_t, a_t, P_T along each arm ----------
    p3 = {}
    for arm in ["C0", "V", "N", "N_PM"]:
        rows = {r["step"]: r["full_vocab_ce"] for r in (json.loads(l) for l in T.joinpath(ARMS[arm], "metrics.jsonl").open() if l.strip())}
        prev = f0.clone(); P = 0.0; series = []
        for p in ckpts(arm):
            st = int(p.stem.split("_")[1])
            if st == 0: continue
            th = load(p)["theta"]; fth = flatten(th)
            g = grad_at(th)
            s_t = float((-g * e).sum())
            a_t = float((-g * e).sum() / (g.norm() * e.norm() + 1e-30))
            P += float(((fth - prev) * e).sum()); prev = fth
            series.append({"step": st, "s_t": s_t, "a_t": a_t, "P_T": P,
                           "ce": rows.get(st), "gnorm": float(g.norm())})
        p3[arm] = series
        pre_s = [x["s_t"] for x in series if x["ce"] is None or x["ce"] > 3.0]
        print(f"[3] {arm}: {len(series)} pts, pre-onset s_t range [{min(pre_s):.4f},{max(pre_s):.4f}]" if pre_s else f"[3] {arm}: {len(series)} pts")

    # ---------- Phase 2: conditional update at early/late v-only states ----------
    def cond_at(theta, mm, vv, teff, e, Kd=256):
        set_theta(theta)
        g = flatten(R.full_grad(exp, params, chunks))
        mf, vf = flatten(mm), flatten(vv); g2 = g * g
        c1, c2 = 1 - B1 ** teff, 1 - B2 ** teff
        mc = B1 * mf + (1 - B1) * g; vc = B2 * vf + (1 - B2) * g2
        uc = -LR * (mc / c1) / ((vc / c2).sqrt() + EPS); ucn = float(uc.norm()); eloc = uc / ucn
        mbg = torch.Generator().manual_seed(12345)
        def mb():
            idx = torch.randint(n, (128,), generator=mbg); b = R.to_dev(R.collate_fn([exp.dataset[int(i)] for i in idx]))
            for p in params: p.grad = None
            R.compute_loss(exp.model, b)[0].backward(); return flatten([p.grad.detach().clone().float() for p in params])
        xin = float(((mb() - mb()) / math.sqrt(2)).norm())
        ng = torch.Generator().manual_seed(777)
        suv = torch.zeros_like(g); sub = torch.zeros_like(g); s2v = s2b = 0.0
        for _ in range(Kd):
            iso = flatten([torch.randn(p.shape, generator=ng) for p in params]); iso *= xin / (iso.norm() + 1e-30)
            gpx = g + iso; gpx2 = gpx * gpx
            uv = -LR * ((B1 * mf + (1 - B1) * g) / c1) / (((B2 * vf + (1 - B2) * gpx2) / c2).sqrt() + EPS)
            ub = -LR * ((B1 * mf + (1 - B1) * gpx) / c1) / (((B2 * vf + (1 - B2) * gpx2) / c2).sqrt() + EPS)
            suv += uv; sub += ub; s2v += float((uv * uv).sum()); s2b += float((ub * ub).sum())
        muv, mub = suv / Kd, sub / Kd
        def summ(mu, s2): return {"mu_par": float((mu * eloc).sum()), "cos": float((mu * uc).sum() / (mu.norm() * ucn + 1e-30)),
                                  "mu_norm": float(mu.norm()), "V_perp": s2 / Kd - float((mu * mu).sum())}
        return {"v_only": summ(muv, s2v), "both": summ(mub, s2b), "uc_norm": ucn}

    on_v = onset("V"); vcks = ckpts("V"); vsteps = [int(p.stem.split("_")[1]) for p in vcks]
    early = next(p for s, p in zip(vsteps, vcks) if s >= 1000)
    late = [p for s, p in zip(vsteps, vcks) if on_v and s <= on_v - 100][-1] if on_v else vcks[-2]
    p2 = {}
    for tag, p in [("V_early", early), ("V_late", late)]:
        st = load(p); step = st["step"]
        p2[tag] = {"step": step, **cond_at(st["theta"], st["m"], st["v"], st["t_eff"], e)}
        print(f"[2] {tag} step {step}: dmu_par={p2[tag]['both']['mu_par']-p2[tag]['v_only']['mu_par']:.4f} "
              f"Vperp both/v={p2[tag]['both']['V_perp']/max(p2[tag]['v_only']['V_perp'],1e-9):.0f}x")

    rep = {"phase1D": p1d, "phase3": p3, "phase2_vstates": p2, "V_onset": on_v}
    (OUT / "traj_analysis.json").write_text(json.dumps(rep, indent=1))
    print("wrote traj_analysis.json")


if __name__ == "__main__":
    main()
