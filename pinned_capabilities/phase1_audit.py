"""Phase 1B (GD-regime) + Phase 1C (numerical audit) at a frozen state.
Gates Case D (numerical pathology) before interpreting the diffusion result.
Reuses the verified reacq build/AdamW. CPU. usage: phase1_audit.py"""
import json, sys, math
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch
sys.argv = ["x", "V", "collapsed", "0", "5", "cpu", "1", "1.0"]
import pinned_capabilities.reacq_2x2_run as R
import torch.nn.functional as F

OUT = R.REPO / "pinned_capabilities/results/cond_update"; OUT.mkdir(parents=True, exist_ok=True)
B1, B2, EPS = R.B1, R.B2, R.EPS


def flat(v): return torch.cat([x.reshape(-1) for x in v])


def main():
    exp, params, m, v, t0, n, chunks = R.build()
    g = R.full_grad(exp, params, chunks)
    gf = flat(g); D = gf.numel()
    mf, vf = flat(m), flat(v)

    # ---- Phase 1C: numerical audit ----
    exp.model.eval()
    probs_max, marg, ntok = [], [], 0
    with torch.no_grad():
        for b in chunks:
            logits = exp.model(b["input_ids"])
            sl = logits[:, :-1]; st = b["labels"][:, 1:]; mask = st != -100
            idx = mask.nonzero(as_tuple=True)
            lo = sl[idx[0], idx[1]]                       # (Ntok, vocab) at target positions
            tgt = st[idx[0], idx[1]]
            p = F.softmax(lo, dim=-1)
            probs_max.append(p.max(dim=-1).values)
            correct = lo.gather(1, tgt[:, None]).squeeze(1)
            lo2 = lo.clone(); lo2.scatter_(1, tgt[:, None], float("-inf"))
            marg.append(correct - lo2.max(dim=-1).values)
            ntok += tgt.numel()
    exp.model.train()
    pmax = torch.cat(probs_max); margin = torch.cat(marg)
    audit = {
        "D": D, "n_examples": n, "n_target_tokens": ntok,
        "grad_zero_frac": float((gf == 0).float().mean()),
        "grad_min_nonzero": float(gf[gf != 0].abs().min()) if (gf != 0).any() else 0.0,
        "grad_norm": float(gf.norm()), "param_norm": float(flat(params).norm()),
        "m_norm": float(mf.norm()), "v_norm": float(vf.norm()),
        "v_min": float(vf.min()), "v_max": float(vf.max()),
        "prob_max_median": float(pmax.median()), "prob_saturated_gt0.99_frac": float((pmax > 0.99).float().mean()),
        "prob_max_lt0.05_frac": float((pmax < 0.05).float().mean()),
        "logit_margin_median": float(margin.median()), "logit_margin_frac_positive": float((margin > 0).float().mean()),
        "nan_or_inf": bool(torch.isnan(gf).any() or torch.isinf(gf).any() or torch.isnan(vf).any() or torch.isnan(mf).any()),
        "loss_impl": "F.cross_entropy (softmax+NLL, fp32)",
    }

    # ---- Phase 1B: coordinatewise noise scale sigma_i (Welford over 32 minibatch grads) ----
    mbgen = torch.Generator().manual_seed(2024)
    def mb():
        idx = torch.randint(n, (128,), generator=mbgen)
        batch = R.to_dev(R.collate_fn([exp.dataset[int(i)] for i in idx]))
        for p in params: p.grad = None
        loss, _, _ = R.compute_loss(exp.model, batch); loss.backward()
        return flat([p.grad.detach().clone().float() for p in params])
    Kmb = 32; mean = torch.zeros(D); M2 = torch.zeros(D)
    for k in range(1, Kmb + 1):
        x = mb(); d = x - mean; mean += d / k; M2 += d * (x - mean)
    sigma = (M2 / (Kmb - 1)).sqrt()                      # per-coord noise std
    sig_pos = sigma[sigma > 0]

    # v-only update for one matched-noise draw, test u_v ~ g/(c*sigma)
    ng = torch.Generator().manual_seed(5)
    iso = flat([torch.randn(p.shape, generator=ng) for p in params]); iso *= (0.059089 / (iso.norm() + 1e-30))
    gpx = gf + iso
    t_eff = t0 + 1
    c1, c2 = 1 - B1 ** t_eff, 1 - B2 ** t_eff
    mv = B1 * mf + (1 - B1) * gf                         # clean m (v-only)
    vv = B2 * vf + (1 - B2) * gpx * gpx                  # noisy v
    u_v = -R.LR * (mv / c1) / ((vv / c2).sqrt() + EPS)
    vc = B2 * vf + (1 - B2) * gf * gf
    u_clean = -R.LR * (mv / c1) / ((vc / c2).sqrt() + EPS)
    pred = gf / sigma.clamp(min=sigma[sigma > 0].min())  # g/sigma (up to constant c)
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))
    gd = {
        "sigma_median": float(sig_pos.median()), "sigma_iqr": [float(sig_pos.quantile(.25)), float(sig_pos.quantile(.75))],
        "sigma_anisotropy_p95_over_p05": float(sig_pos.quantile(.95) / sig_pos.quantile(.05)),
        "corr(u_v, g)": corr(u_v, gf),
        "corr(u_v, g/sigma)": corr(u_v, pred),
        "corr(u_clean, g)": corr(u_clean, gf),
        "u_v_vs_g_over_sigma_slope": float((u_v * pred).sum() / (pred * pred).sum()),
        "clean_signlike_|cos(u_clean,sign g)|": abs(corr(u_clean, torch.sign(gf))),
        "v_only_signlike_|cos(u_v,sign g)|": abs(corr(u_v, torch.sign(gf))),
    }
    rep = {"state": "collapsed", "audit_1C": audit, "gd_regime_1B": gd}
    (OUT / "phase1_audit_collapsed.json").write_text(json.dumps(rep, indent=1))
    print("=== Phase 1C numerical audit ==="); print(json.dumps(audit, indent=1))
    print("=== Phase 1B GD-regime ==="); print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in gd.items()}, indent=1))


if __name__ == "__main__":
    main()
