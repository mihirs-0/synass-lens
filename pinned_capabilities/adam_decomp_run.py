"""Registered Adam-channel decomposition (final pre-draft experiment).

Attributes noise-maintained trapping to a channel inside AdamW. Pre-registered
in ADAM_DECOMP_PREREG.md. Manual AdamW (unit-tested vs torch) with per-arm
gradient routing, pathwise clean shadow moments, and Gamma accounting.

Canonical noisy arm = matched ISOTROPIC noise xi_t = sigma * eps_t, sigma
frozen = collapsed-state per-coordinate minibatch-noise std (6.096e-5).

  C0    clean full gradient                          (must reacquire)
  N     g+xi into both moments                       (must trap)
  N_BC  m<-g+xi ; v<-(g+xi)^2 - sigma^2 , v-floored    (v-bias removed)
  V     m<-g   ; v<-(g+xi)^2                          (second-moment only)
  M     m<-g+xi; v<-g^2                               (first-moment only)
  P_AR  clean moments ; theta += u_clean + zeta_t     (post-Adam colored perturb)

usage: adam_decomp_run.py ARM STREAM [T] [threads]
"""
import json, sys, math
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
from dataclasses import replace
from pinned_capabilities.config import ProtocolConfig
from pinned_capabilities.experiment import MBCExperiment
from pinned_capabilities.gate0e_escape import load_snapshot_fresh_stream
from src.data import collate_fn
from src.training.trainer import compute_loss

ARM = sys.argv[1]
STREAM = int(sys.argv[2])
T = int(sys.argv[3]) if len(sys.argv) > 3 else 3500
SIGMA = 6.096e-05
LR, WD, B1, B2, EPS = 0.0125, 0.01, 0.9, 0.999, 1e-8
EVAL_EVERY, CHUNK = 50, 2500
VMIN_SNAP_STEPS = (100, 300, 500, 700, 900)
REPO = Path("/Users/mihir/synass-lens/synass-lens")
DECOMP = REPO / "pinned_capabilities/results/gate0e_adam_decomp"
COLLAPSED = REPO / "pinned_capabilities/results/gate0e_dev_curve_seed100/eta_0p0125/stream_00/checkpoints/slot_0.pt"


def build():
    cfg = ProtocolConfig()
    exp = MBCExperiment(replace(cfg.experiment, seed=100, device="cpu"), cfg.metric)
    load_snapshot_fresh_stream(COLLAPSED, model=exp.model, optimizer=exp.optimizer, map_location="cpu")
    params = [p for grp in exp.optimizer.param_groups for p in grp["params"]]
    S = exp.optimizer.state
    m = [S[p]["exp_avg"].clone() for p in params]
    v = [S[p]["exp_avg_sq"].clone() for p in params]
    step0 = S[params[0]]["step"]
    t0 = int(step0.item()) if torch.is_tensor(step0) else int(step0)
    return exp, params, m, v, t0


def full_grad(exp, params, chunks):
    exp.optimizer.zero_grad(set_to_none=True)
    for b in chunks:
        loss, _, _ = compute_loss(exp.model, b)
        (loss / len(chunks)).backward()
    return [p.grad.detach().clone() for p in params]


def bias_corrected_update(m, v, t_eff):
    c1, c2 = 1 - B1 ** t_eff, 1 - B2 ** t_eff
    return [-LR * (m[i] / c1) / ((v[i] / c2).sqrt() + EPS) for i in range(len(m))]


def flat_norm(lst):
    return math.sqrt(sum(float(x.pow(2).sum()) for x in lst))


def flat_dot(a, b):
    return sum(float((x * y).sum()) for x, y in zip(a, b))


def run():
    exp, params, m, v, t0 = build()
    n = len(exp.dataset)
    D = sum(p.numel() for p in params)
    chunks = [collate_fn([exp.dataset[i] for i in range(s, s + CHUNK)]) for s in range(0, n, CHUNK)]
    mbgen = torch.Generator().manual_seed(500_000 + 991 * sum(ord(c) for c in ARM) + STREAM)

    def mb_grad():
        idx = torch.randint(n, (128,), generator=mbgen)
        batch = collate_fn([exp.dataset[int(i)] for i in idx])
        exp.optimizer.zero_grad(set_to_none=True)
        loss, _, _ = compute_loss(exp.model, batch)
        loss.backward()
        return [p.grad.detach().clone() for p in params]
    m_sh = [x.clone() for x in m]
    v_sh = [x.clone() for x in v]
    v_min = None
    if ARM == "N_BC":
        v_min = json.loads((DECOMP / "v_min.json").read_text())["v_min"]
    a_sched = None
    if ARM == "P_AR":
        a_sched = json.loads((DECOMP / f"N_stream{STREAM:02d}/residual_norms.json").read_text())["r_norm"]
    egen = torch.Generator().manual_seed(700_000 + 991 * sum(ord(c) for c in ARM) + STREAM) if ARM != "C0" else None
    h = [torch.zeros_like(p) for p in params] if ARM == "P_AR" else None

    OUT = DECOMP / f"{ARM}_stream{STREAM:02d}"
    OUT.mkdir(parents=True, exist_ok=True)
    rows, r_norms, sum_r2, sum_ush2, vmin_pool, floor_binds = [], [], 0.0, 0.0, [], []
    writer = (OUT / "metrics.jsonl").open("w")

    for step in range(1, T + 1):
        exp.model.train()
        g = full_grad(exp, params, chunks)
        g2 = [x * x for x in g]
        t_eff = t0 + step
        # clean shadow moments (advance on clean g at current theta)
        for i in range(len(m_sh)):
            m_sh[i].mul_(B1).add_(g[i], alpha=1 - B1)
            v_sh[i].mul_(B2).add_(g2[i], alpha=1 - B2)
        u_sh = bias_corrected_update(m_sh, v_sh, t_eff)

        if ARM != "C0" and ARM != "P_AR":
            # per-step MATCHED isotropic noise (certified recipe): ||xi|| = ||(g_B1-g_B2)/sqrt2||
            ref = [(a - b) / math.sqrt(2) for a, b in zip(mb_grad(), mb_grad())]
            ref_norm = flat_norm(ref)
            iso = [torch.randn(p.shape, generator=egen) for p in params]
            scale = ref_norm / (flat_norm(iso) + 1e-30)
            xi = [scale * e for e in iso]
            sig2 = ref_norm * ref_norm / D  # per-coordinate variance of xi this step
            gpx = [a + b for a, b in zip(g, xi)]
            if ARM == "N":
                g_m, v_inc = gpx, [x * x for x in gpx]
            elif ARM == "N_BC":
                g_m, v_inc = gpx, [x * x - sig2 for x in gpx]
            elif ARM == "V":
                g_m, v_inc = g, [x * x for x in gpx]
            elif ARM == "M":
                g_m, v_inc = gpx, g2
            for i in range(len(m)):
                m[i].mul_(B1).add_(g_m[i], alpha=1 - B1)
                v_raw = v[i] * B2 + v_inc[i] * (1 - B2)
                if ARM == "N_BC":
                    floor_binds.append(float((v_raw < v_min).sum()) / v_raw.numel())
                    v_raw = v_raw.clamp(min=v_min)
                v[i] = v_raw
            u = bias_corrected_update(m, v, t_eff)
            r = [a - b for a, b in zip(u, u_sh)]
            for i, p in enumerate(params):
                p.data.mul_(1 - LR * WD).add_(u[i])
        elif ARM == "C0":
            for i in range(len(m)):
                m[i].mul_(B1).add_(g[i], alpha=1 - B1)
                v[i].mul_(B2).add_(g2[i], alpha=1 - B2)
            u = bias_corrected_update(m, v, t_eff)
            r = [a - b for a, b in zip(u, u_sh)]  # ~0
            for i, p in enumerate(params):
                p.data.mul_(1 - LR * WD).add_(u[i])
        else:  # P_AR
            for i, p in enumerate(params):
                h[i].mul_(B1).add_(math.sqrt(1 - B1 * B1) * torch.randn(p.shape, generator=egen))
            hnorm = flat_norm(h)
            a_t = a_sched[step - 1] if step - 1 < len(a_sched) else a_sched[-1]
            zeta = [(a_t / (hnorm + 1e-30)) * hi for hi in h]  # ||zeta|| = a_t
            r = zeta
            for i, p in enumerate(params):
                p.data.mul_(1 - LR * WD).add_(u_sh[i]).add_(zeta[i])

        qr = flat_norm(r); ush = flat_norm(u_sh)
        sum_r2 += qr * qr; sum_ush2 += ush * ush
        r_norms.append(qr)
        if ARM == "C0" and step in VMIN_SNAP_STEPS:
            vhat = torch.cat([(v_sh[i] / (1 - B2 ** t_eff)).reshape(-1) for i in range(len(v_sh))])
            vmin_pool.append(vhat[vhat > 0].clone())
            if step == VMIN_SNAP_STEPS[-1]:  # write v_min at step 900 so N_BC can launch mid-run
                pooled = torch.cat(vmin_pool)
                (DECOMP / "v_min.json").write_text(json.dumps(
                    {"v_min": float(torch.quantile(pooled, 0.001)), "n_pos": int(pooled.numel())}))
        if step % EVAL_EVERY == 0:
            ev = exp.evaluate()
            cos = flat_dot(r, u_sh) / (qr * ush) if qr > 0 and ush > 0 else 0.0
            rows.append({"step": step, "full_vocab_ce": ev["full_vocab_ce"], "c_int": ev["c_int"],
                         "Q_r": qr, "A_align": cos, "Gamma_running": sum_r2 / sum_ush2 if sum_ush2 else 0.0,
                         "max_floor_frac": max(floor_binds) if floor_binds else 0.0})
            writer.write(json.dumps(rows[-1]) + "\n"); writer.flush()
    writer.close()
    if not rows:
        print(json.dumps({"arm": ARM, "stream": STREAM, "note": "T<eval_interval; no rows"})); return

    def escape(rows):
        for i, rr in enumerate(rows):
            if rr["full_vocab_ce"] < 3.0:
                fut = [x for x in rows[i:] if x["step"] <= rr["step"] + 500]
                if len(fut) >= 11 and all(x["full_vocab_ce"] < 3.0 for x in fut):
                    return rr["step"]
        return None
    esc = escape(rows)
    summ = {"arm": ARM, "stream": STREAM, "T": T, "escape_step": esc, "censored": esc is None,
            "Gamma": sum_r2 / sum_ush2 if sum_ush2 else 0.0, "final_ce": rows[-1]["full_vocab_ce"],
            "final_c_int": rows[-1]["c_int"], "max_floor_frac": max(floor_binds) if floor_binds else 0.0,
            "degraded": bool(max(floor_binds) > 0.05) if floor_binds else False}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    (OUT / "residual_norms.json").write_text(json.dumps({"r_norm": r_norms}))
    if ARM == "C0" and vmin_pool:
        pooled = torch.cat(vmin_pool)
        (DECOMP / "v_min.json").write_text(json.dumps(
            {"v_min": float(torch.quantile(pooled, 0.001)), "n_pos": int(pooled.numel())}))
    print(json.dumps(summ))


if __name__ == "__main__":
    run()
