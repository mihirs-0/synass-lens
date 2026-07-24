"""Phase A gating compute (analysis of saved artifacts only; no new experiments).
A1: per-arm OWN-axis s_t and cumulative P_T (clean, v-only).
A2: trapped-state (both-channel) true-gradient escape-direction pull + ||g|| —
    burial (pull present) vs relocation (pull absent).
A3: q (noisy/clean update-norm ratio), rho (cosine), participation ratio of the
    clean update, and the Gamma=1+q^2-2q*rho check at the fork.
Reuses the verified reacq AdamW. MPS. Outputs results/cond_update/phase_a.json."""
import json, sys, math, itertools
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch
sys.argv = ["x", "V", "collapsed", "0", "5", "mps", "6", "1.0"]
import pinned_capabilities.reacq_2x2_run as R
DEV = "mps"
B1, B2, EPS, LR, WD = R.B1, R.B2, R.EPS, R.LR, R.WD
TR = R.REPO / "pinned_capabilities/results/traj"
OUT = R.REPO / "pinned_capabilities/results/cond_update"
ARMS = {"C0": "C0_sc1.00", "V": "V_sc1.00", "N": "N_sc1.00", "N_PM": "N_PM_sc0.60"}
ONSET = {"C0": 1600, "V": 3800}


def cks(arm): return sorted(TR.joinpath(ARMS[arm]).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
def theta_of(path): return [x.to(DEV) for x in torch.load(path, map_location="cpu")["theta"]]
def flat(t): return torch.cat([x.reshape(-1) for x in t])
def metrics(arm):
    return {r["step"]: r["full_vocab_ce"] for r in (json.loads(l) for l in TR.joinpath(ARMS[arm], "metrics.jsonl").open() if l.strip())}


def main():
    exp, params, m0, v0, t0, n, chunks = R.build()
    f0 = flat(theta_of(cks("C0")[0]))

    def set_theta(fth):
        i = 0
        for p in params:
            p.data.copy_(fth[i:i + p.numel()].view_as(p)); i += p.numel()
    def gflat(fth):
        set_theta(fth); return flat(R.full_grad(exp, params, chunks))

    # ---- own escape directions (each arm's own pre-onset displacement) ----
    def own_e(arm, window=400, cap=8):
        on = ONSET[arm]
        pre = [p for p in cks(arm) if 0 < int(p.stem.split("_")[1]) <= on and int(p.stem.split("_")[1]) >= on - window][-cap:]
        dirs = []
        for p in pre:
            d = flat(theta_of(p)) - f0; dirs.append(d / (d.norm() + 1e-30))
        cospw = [float((a * b).sum()) for a, b in itertools.combinations(dirs, 2)]
        e = torch.stack(dirs).mean(0); e = e / (e.norm() + 1e-30)
        return e, {"steps": [int(p.stem.split("_")[1]) for p in pre],
                   "pairwise_cos_median": float(torch.tensor(cospw).median()) if cospw else None}
    e_clean, i_c = own_e("C0"); e_v, i_v = own_e("V")

    # ---- A1: own-axis s_t, P_T ----
    A1 = {"e_stability": {"C0": i_c, "V": i_v}}
    for arm, e_own, on in [("C0", e_clean, 1600), ("V", e_v, 3800)]:
        ce = metrics(arm); prev = f0.clone(); P = 0.0; ser = []
        for p in cks(arm):
            st = int(p.stem.split("_")[1])
            if st == 0: continue
            fth = flat(theta_of(p)); g = gflat(fth)
            ser.append({"step": st, "s_t": float((-g * e_own).sum()),
                        "P_T": (P := P + float(((fth - prev) * e_own).sum())), "ce": ce.get(st)})
            prev = fth
        pre = [x["s_t"] for x in ser if (x["ce"] or 9) > 3.0]
        post = [x["s_t"] for x in ser if (x["ce"] or 9) <= 3.0]
        A1[arm] = {"series": ser, "pre_onset_s_t_first": pre[0] if pre else None,
                   "pre_onset_s_t_last": pre[-1] if pre else None,
                   "pre_onset_s_t_max": max(pre) if pre else None, "P_T_final": ser[-1]["P_T"] if ser else None,
                   "growth_ratio_last_over_first": (pre[-1] / pre[0]) if (pre and abs(pre[0]) > 1e-9) else None}

    # ---- A2: trapped-state pull (both-channel) onto clean & v-only escape dirs + ||g|| ----
    A2 = []
    for p in cks("N"):
        st = int(p.stem.split("_")[1])
        if st == 0 or st % 500 != 0: continue                      # subsample: several states
        fth = flat(theta_of(p)); g = gflat(fth)
        A2.append({"step": st, "pull_clean_e": float((-g * e_clean).sum()),
                   "pull_v_e": float((-g * e_v).sum()), "g_norm": float(g.norm()),
                   "ce": metrics("N").get(st)})
    # add reacq both-channel 16k state
    for seed in [0]:
        rp = R.REPO / f"pinned_capabilities/results/reacq_2x2/N_collapsed_seed{seed}/ckpt.pt"
        if rp.exists():
            st16 = torch.load(rp, map_location="cpu")
            fth = flat([x.to(DEV) for x in st16["theta"]]); g = gflat(fth)
            A2.append({"step": 16000, "seed": seed, "pull_clean_e": float((-g * e_clean).sum()),
                       "pull_v_e": float((-g * e_v).sum()), "g_norm": float(g.norm()), "ce": None})

    # ---- A3: q, rho, participation ratio, Gamma check at the fork ----
    set_theta(f0); g = flat(R.full_grad(exp, params, chunks)); g2 = g * g
    teff = t0 + 1; c1, c2 = 1 - B1 ** teff, 1 - B2 ** teff
    mf, vf = flat(m0), flat(v0)
    u_clean = -LR * ((B1 * mf + (1 - B1) * g) / c1) / (((B2 * vf + (1 - B2) * g2) / c2).sqrt() + EPS)
    ucn = float(u_clean.norm())
    # participation ratio of clean update
    PR = float((u_clean.pow(2).sum() ** 2) / (u_clean.pow(4).sum() + 1e-45))
    # many noise draws -> q, rho aggregate
    mbg = torch.Generator().manual_seed(12345)
    def mb():
        idx = torch.randint(n, (128,), generator=mbg); b = R.to_dev(R.collate_fn([exp.dataset[int(i)] for i in idx]))
        for p in params: p.grad = None
        R.compute_loss(exp.model, b)[0].backward(); return flat([p.grad.detach().clone().float() for p in params])
    xin = float(((mb() - mb()) / math.sqrt(2)).norm())
    ng = torch.Generator().manual_seed(777); qs = []; rhos = []
    for _ in range(64):
        iso = flat([torch.randn(p.shape, generator=ng).to(DEV) for p in params]); iso *= xin / (iso.norm() + 1e-30)
        gpx = g + iso; gpx2 = gpx * gpx
        ub = -LR * ((B1 * mf + (1 - B1) * gpx) / c1) / (((B2 * vf + (1 - B2) * gpx2) / c2).sqrt() + EPS)
        qs.append(float(ub.norm()) / ucn); rhos.append(float((ub * u_clean).sum() / (ub.norm() * ucn + 1e-30)))
    q = sum(qs) / len(qs); rho = sum(rhos) / len(rhos)
    a = xin / ucn
    A3 = {"q_mean": q, "rho_mean": rho, "a_amp_ratio": a, "scalar_pred_1_plus_a2": 1 + a * a,
          "Gamma_from_q_rho": 1 + q * q - 2 * q * rho, "participation_ratio_clean": PR,
          "PR_fraction_of_D": PR / f0.numel(), "uc_norm": ucn, "xi_norm": xin,
          "decomp_N_aggregate": {"Q_r": 6.2914, "A_align": -0.8714, "Gamma_running": 1.8939}}

    rep = {"A1_own_axis": A1, "A2_trapped_pull": A2, "A3_q_rho_PR": A3}
    (OUT / "phase_a.json").write_text(json.dumps(rep, indent=1))
    print("A1 own-axis:", json.dumps({k: {kk: A1[k][kk] for kk in ("pre_onset_s_t_first", "pre_onset_s_t_max", "P_T_final", "growth_ratio_last_over_first")} for k in ("C0", "V")}))
    print("A2 trapped pull (step: pull_clean_e, g_norm):", [(x["step"], round(x["pull_clean_e"], 5), round(x["g_norm"], 4)) for x in A2])
    print("A3:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in A3.items() if k != "decomp_N_aggregate"}))
    print("wrote phase_a.json")


if __name__ == "__main__":
    main()
