#!/usr/bin/env python
"""EXP 2 — confound factorial. 'How much to memorize' bundles: #distinct targets N, examples-per-target
epp, total examples M=N*epp, and target entropy. The original eta*(N) was measured at FIXED M=8000, so N
and epp co-varied -> confounded. Separate them:
  Arm A: N=1000 fixed, vary M/epp in {2000,4000,8000}  -> eta* vs epp (and M) AT FIXED N.
  Arm D: epp=4 fixed, vary N in {500,1000,2000} (M=2000,4000,8000) -> eta* vs N AT FIXED epp.
  Arm C: N=1000,M=8000 fixed, vary skew (entropy) -> eta* vs target entropy, mapping unchanged.
Verdict: eta* tracks N (A flat, D falls) -> 'number of distinct targets' survives + dilution holds.
         eta* tracks epp/M (A falls) or entropy (C moves) -> rewrite the claim around what it tracks.
Capacity-safe: all M<=8000 (d=64 memorizes ~8000)."""
import json, os, math
import collapse_lib as L

SEEDS = 2; STEPS = 5000; vocab = L.NSPEC + L.ALPHA
path = "card_ckpts/exp2_factorial.json"
res = json.load(open(path)) if os.path.exists(path) else {"A": {}, "D": {}, "C": {}}
def save(): json.dump(res, open(path, "w"), indent=2)


def eta_star_of(X):
    frac = {}
    for eta in L.ETA_GRID:
        fr = [L.classify(*L.train(X, eta, s, STEPS, vocab)[1:]) for s in range(SEEDS)]
        frac[eta] = (fr.count("trap") + 0.5 * fr.count("partial")) / SEEDS
    return L.eta_star(frac), frac


def entropy_ratio(probs):
    import math
    N = len(probs); h = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
    return h / math.log(N)


def main():
    print(f"EXP2 factorial. seeds={SEEDS} steps={STEPS} eta={L.ETA_GRID}\n")

    print("--- Arm A: N=1000 fixed, vary M/epp (eta* vs epp AT FIXED N) ---")
    for M in [2000, 4000, 8000]:
        k = str(M)
        if k in res["A"]: print(f"  M={M} epp={M//1000}: (cached) eta*={res['A'][k]['eta_star']}"); continue
        X, *_ = L.build_task(1000, M=M, seed=0); es, frac = eta_star_of(X)
        res["A"][k] = {"N": 1000, "M": M, "epp": M / 1000, "eta_star": es, "frac": frac}; save()
        print(f"  M={M:>5} epp={M/1000:>4.1f}: eta*={('%.4f'%es) if es else 'n/a':>8} | "
              + " ".join(f"{e:.3f}:{frac[e]:.2f}" for e in L.ETA_GRID))

    print("\n--- Arm D: epp=4 fixed, vary N (eta* vs N AT FIXED epp) ---")
    for N in [500, 1000, 2000]:
        k = str(N)
        if k in res["D"]: print(f"  N={N}: (cached) eta*={res['D'][k]['eta_star']}"); continue
        X, *_ = L.build_task(N, epp=4, seed=0); es, frac = eta_star_of(X)
        res["D"][k] = {"N": N, "M": 4 * N, "epp": 4, "eta_star": es, "frac": frac}; save()
        print(f"  N={N:>5} M={4*N:>5}: eta*={('%.4f'%es) if es else 'n/a':>8} | "
              + " ".join(f"{e:.3f}:{frac[e]:.2f}" for e in L.ETA_GRID))

    print("\n--- Arm C: N=1000,M=8000 fixed, vary entropy via skew (mapping unchanged) ---")
    for skew in [0.0, 4.0, 8.0]:
        k = f"skew{skew}"
        if k in res["C"]: print(f"  {k}: (cached) eta*={res['C'][k]['eta_star']}"); continue
        prior = "uniform" if skew == 0.0 else "skew"
        X, assign, outs, probs = L.build_task(1000, M=8000, prior=prior, skew=skew, seed=0)
        er = entropy_ratio(probs); es, frac = eta_star_of(X)
        res["C"][k] = {"skew": skew, "entropy_ratio": round(er, 3), "eta_star": es, "frac": frac}; save()
        print(f"  skew={skew:>4} H/Hmax={er:.3f}: eta*={('%.4f'%es) if es else 'n/a':>8} | "
              + " ".join(f"{e:.3f}:{frac[e]:.2f}" for e in L.ETA_GRID))

    print("\n=== SUMMARY ===")
    print("Arm A (N=1000, vary epp):  " + ", ".join(f"epp{res['A'][k]['epp']:.0f}->{res['A'][k]['eta_star']}" for k in sorted(res['A'], key=int)))
    print("Arm D (epp=4, vary N):     " + ", ".join(f"N{res['D'][k]['N']}->{res['D'][k]['eta_star']}" for k in sorted(res['D'], key=int)))
    print("Arm C (vary entropy):      " + ", ".join(f"{res['C'][k]['entropy_ratio']}->{res['C'][k]['eta_star']}" for k in res['C']))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
