#!/usr/bin/env python
"""EXP 1 — is the collapse partly a weight-decay (regularization) effect, not pure signal loss?
AdamW shrinks weights ~eta*lambda/step; high-LR runs eat more cumulative decay, and a prior-predicting
solution is cheap in weights while a lookup table is expensive. Test:
  (a) full N x eta sweep at lambda=0 vs matched lambda=0.01 -> does collapse survive at lambda=0?
  (b) iso-cumulative-decay: lambda(eta)=lam_ref*eta_ref/eta so eta*lambda*T is constant across eta.
  (c) norm traces (param/head-bias/head-w/tok-emb) at high vs low eta -> 'cheap prior vs expensive lookup'.
Collapse vanishes/weakens at lambda=0 -> decay is a driver (story changes). Survives -> decay not the cause."""
import json, os, math
import collapse_lib as L

Ns = [500, 1000, 2000, 4000, 8000]
SEEDS = 2; STEPS = 5000; M = 8000
vocab = L.NSPEC + L.ALPHA
ETA_REF, LAM_REF = 0.015, 0.01
path = "card_ckpts/exp1_wd.json"
res = json.load(open(path)) if os.path.exists(path) else {"sweep": {}, "iso": {}, "norms": {}}


def save(): json.dump(res, open(path, "w"), indent=2)


def cell(X, eta, wd, lam_per_step=None):
    fr = []
    for s in range(SEEDS):
        m, fl, kl = L.train(X, eta, s, STEPS, vocab, wd=wd, lam_per_step=lam_per_step)
        fr.append(L.classify(fl, kl))
    ntr = fr.count("trap"); npa = fr.count("partial")
    return (ntr + 0.5 * npa) / SEEDS, fr


def main():
    print(f"EXP1 weight decay. N={Ns} eta={L.ETA_GRID} seeds={SEEDS} steps={STEPS} M={M}\n")
    # (a) sweep at lambda=0 and lambda=0.01
    for wd in [0.0, 0.01]:
        key = f"wd{wd}"; res["sweep"].setdefault(key, {})
        print(f"--- lambda={wd} ---")
        for N in Ns:
            if str(N) in res["sweep"][key]:
                print(f"  N={N}: (cached) eta*={res['sweep'][key][str(N)]['eta_star']}"); continue
            X, *_ = L.build_task(N, M=M, seed=0); frac = {}
            for eta in L.ETA_GRID:
                f, fr = cell(X, eta, wd); frac[eta] = f
            es = L.eta_star(frac)
            res["sweep"][key][str(N)] = {"eta_star": es, "frac": frac}; save()
            fs = " ".join(f"{e:.3f}:{frac[e]:.2f}" for e in L.ETA_GRID)
            print(f"  N={N:>5}: eta*={('%.4f'%es) if es else 'n/a':>8}  | {fs}")
        print()

    # (b) iso-cumulative-decay at N=2000: lambda(eta) = LAM_REF*ETA_REF/eta  -> eta*lambda*T = const
    print(f"--- iso-cumulative-decay (eta*lambda*T const) at N=2000 ---")
    X, *_ = L.build_task(2000, M=M, seed=0); frac = {}
    for eta in L.ETA_GRID:
        lam = LAM_REF * ETA_REF / eta
        f, fr = cell(X, eta, None, lam_per_step=lam); frac[eta] = f
        print(f"  eta={eta:.3f} lambda={lam:.4f} (eta*lam={eta*lam:.5f}): frac_collapse={f:.2f} {fr}")
    res["iso"]["N2000"] = {"eta_star": L.eta_star(frac), "frac": frac,
                           "note": "lambda=LAM_REF*ETA_REF/eta so eta*lambda*T constant"}; save()
    print(f"  -> iso eta* = {res['iso']['N2000']['eta_star']}")

    # (c) norm traces: high vs low eta, lambda=0 vs 0.01, at N=2000
    print(f"\n--- norm traces at N=2000 (param/head_bias/head_w/tok_emb over training) ---")
    X, *_ = L.build_task(2000, M=M, seed=0)
    for wd in [0.0, 0.01]:
        for eta, tag in [(0.003, "low"), (0.06, "high")]:
            m, fl, kl, tr = L.train(X, eta, 0, STEPS, vocab, wd=wd, track_norms=True)
            res["norms"][f"wd{wd}_{tag}"] = {"eta": eta, "final_loss": round(fl, 3), "kl": round(kl, 3), "trace": tr}; save()
            last = tr[-1]
            print(f"  wd={wd} {tag}-eta({eta}): loss={fl:.3f} kl={kl:.3f} | "
                  f"final total={last['total']} head_bias={last['head_bias']} head_w={last['head_w']} tok_emb={last['tok_emb']}")
    save(); print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
