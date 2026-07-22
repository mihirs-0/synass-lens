"""Frozen-state conditional one-step AdamW update experiment (diagnostics 2026-07-22).
Central question: does first-moment noise merely add diffusion around the same
coherent update, or does it alter the CONDITIONAL MEAN update? Measures, at a
frozen state, the paired v-only vs both-channel one-step update distribution via
K draws with common random numbers. Reuses the verified reacq AdamW recursion.

Noise = matched isotropic xi (the registered arms' actual noise: ||xi|| =
||(g_B1-g_B2)/sqrt2|| at the state), paired (same xi_k) across arms. The full
gradient g is computed ONCE; the K draws are elementwise (cheap). Streaming
stats only (no dense covariance). Reference direction e = u_clean/||u_clean||.

usage: cond_update_probe.py [K] [state]   state: collapsed (default)
"""
import json, sys, math, time
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch
sys.argv_bak = sys.argv
sys.argv = ["x", "V", "collapsed", "0", "5", "cpu", "1", "1.0"]   # for module import
import pinned_capabilities.reacq_2x2_run as R

K = int(sys.argv_bak[1]) if len(sys.argv_bak) > 1 else 512
STATE = sys.argv_bak[2] if len(sys.argv_bak) > 2 else "collapsed"
OUT = R.REPO / "pinned_capabilities/results/cond_update"
OUT.mkdir(parents=True, exist_ok=True)
B1, B2, EPS, LR, WD = R.B1, R.B2, R.EPS, R.LR, R.WD


def flat(v): return torch.cat([x.reshape(-1) for x in v])
def dot(a, b): return float(sum((x * y).sum() for x, y in zip(a, b)))
def nrm(a): return math.sqrt(sum(float((x * x).sum()) for x in a))


def adam_update(m_list, v_list, t_eff):
    return R.bias_corrected_update(m_list, v_list, t_eff)


def main():
    exp, params, m, v, t0, n, chunks = R.build()
    t_eff = t0 + 1
    g = R.full_grad(exp, params, chunks)
    g2 = [x * x for x in g]

    # clean one-step update u_clean (reference direction e)
    m_c = [B1 * mi + (1 - B1) * gi for mi, gi in zip(m, g)]
    v_c = [B2 * vi + (1 - B2) * gi for vi, gi in zip(v, g2)]
    u_clean = adam_update(m_c, v_c, t_eff)
    uc_norm = nrm(u_clean)
    e = [x / uc_norm for x in u_clean]                    # unit escape-direction reference

    # matched isotropic noise magnitude at this state: ||(g_B1 - g_B2)/sqrt2||
    mbgen = torch.Generator().manual_seed(12345)

    def mb_grad():
        idx = torch.randint(n, (128,), generator=mbgen)
        batch = R.to_dev(R.collate_fn([exp.dataset[int(i)] for i in idx]))
        for p in params: p.grad = None
        loss, _, _ = R.compute_loss(exp.model, batch); loss.backward()
        return [p.grad.detach().clone().float() for p in params]
    ref = [(a - b) / math.sqrt(2) for a, b in zip(mb_grad(), mb_grad())]
    xi_norm = nrm(ref)
    D = sum(p.numel() for p in params)
    print(f"[state={STATE}] ||g||={nrm(g):.4f} ||u_clean||={uc_norm:.4e} ||xi||_matched={xi_norm:.4e} D={D}")

    ngen = torch.Generator().manual_seed(777)

    def one_draw():
        iso = [torch.randn(p.shape, generator=ngen) for p in params]
        sc = xi_norm / (nrm(iso) + 1e-30)
        xi = [sc * z for z in iso]
        gpx = [a + b for a, b in zip(g, xi)]
        gpx2 = [x * x for x in gpx]
        # v-only: clean m, noisy v ; both: noisy m, noisy v  (exact registered routing)
        mv = [B1 * mi + (1 - B1) * gi for mi, gi in zip(m, g)]
        vv = [B2 * vi + (1 - B2) * gg for vi, gg in zip(v, gpx2)]
        mb = [B1 * mi + (1 - B1) * gg for mi, gg in zip(m, gpx)]
        vb = [B2 * vi + (1 - B2) * gg for vi, gg in zip(v, gpx2)]
        return adam_update(mv, vv, t_eff), adam_update(mb, vb, t_eff)

    # Phase 0: runtime probe (32 draws)
    t0p = time.time()
    for _ in range(32): one_draw()
    spd = (time.time() - t0p) / 32
    print(f"[phase0] {spd*1000:.1f} ms/draw (2 arms) -> {K} draws ~ {spd*K:.1f}s ; 512x5 states ~ {spd*512*5/60:.1f} min")

    # Phase 2: K paired draws, streaming
    sum_uv = [torch.zeros_like(p) for p in params]
    sum_ub = [torch.zeros_like(p) for p in params]
    rows = []   # per-draw scalars for bootstrap
    s_uv2 = s_ub2 = 0.0
    for k in range(K):
        uv, ub = one_draw()
        for i in range(len(params)):
            sum_uv[i] += uv[i]; sum_ub[i] += ub[i]
        s_uv2 += nrm(uv) ** 2; s_ub2 += nrm(ub) ** 2
        rows.append({"uv_par": dot(uv, e), "ub_par": dot(ub, e),
                     "uv_n2": nrm(uv) ** 2, "ub_n2": nrm(ub) ** 2,
                     "uv_cos": dot(uv, u_clean) / (nrm(uv) * uc_norm + 1e-30),
                     "ub_cos": dot(ub, u_clean) / (nrm(ub) * uc_norm + 1e-30)})
    mu_v = [s / K for s in sum_uv]; mu_b = [s / K for s in sum_ub]

    def summ(mu, s_u2, key):
        mu_par = dot(mu, e); mu_n = nrm(mu)
        cos_mu = dot(mu, u_clean) / (mu_n * uc_norm + 1e-30)
        V_total = s_u2 / K - mu_n ** 2                      # E||u||^2 - ||mu||^2
        par = [r[key + "_par"] for r in rows]
        mpar = sum(par) / K
        V_par = sum((x - mpar) ** 2 for x in par) / K
        V_perp = V_total - V_par
        snr = abs(mu_par) / math.sqrt(V_par) if V_par > 0 else float("inf")
        return {"mu_par": mu_par, "mu_norm": mu_n, "cos_mu_uclean": cos_mu,
                "V_total": V_total, "V_par": V_par, "V_perp": V_perp, "SNR_par": snr}
    sv = summ(mu_v, s_uv2, "uv"); sb = summ(mu_b, s_ub2, "ub")

    # paired deltas + bootstrap CI (resample draws)
    bgen = torch.Generator().manual_seed(99)
    def boot(fn, B=2000):
        vals = []
        for _ in range(B):
            idx = torch.randint(K, (K,), generator=bgen).tolist()
            vals.append(fn([rows[i] for i in idx]))
        vals.sort(); return vals[int(0.025 * B)], vals[int(0.975 * B)]
    d_mupar = sb["mu_par"] - sv["mu_par"]
    ci_dpar = boot(lambda rs: sum(r["ub_par"] - r["uv_par"] for r in rs) / len(rs))
    d_cos = sb["cos_mu_uclean"] - sv["cos_mu_uclean"]
    ci_dcos = boot(lambda rs: sum(r["ub_cos"] for r in rs)/len(rs) - sum(r["uv_cos"] for r in rs)/len(rs))
    d_norm = sb["mu_norm"] - sv["mu_norm"]

    rep = {"state": STATE, "K": K, "t_eff": t_eff, "uc_norm": uc_norm, "xi_norm": xi_norm,
           "ms_per_draw": spd * 1000, "v_only": sv, "both": sb,
           "delta_mu_par": d_mupar, "delta_mu_par_CI": ci_dpar,
           "delta_cos": d_cos, "delta_cos_CI": ci_dcos, "delta_mu_norm": d_norm,
           "mu_par_ratio_both_over_v": sb["mu_par"] / sv["mu_par"] if sv["mu_par"] else None}
    (OUT / f"cond_update_{STATE}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (round(x, 6) if isinstance(x, float) else x) for k, x in rep.items() if k not in ("v_only", "both")}, indent=1))
    print("v_only:", {k: round(x, 6) for k, x in sv.items()})
    print("both:  ", {k: round(x, 6) for k, x in sb.items()})


if __name__ == "__main__":
    main()
