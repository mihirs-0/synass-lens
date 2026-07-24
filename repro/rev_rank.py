#!/usr/bin/env python
"""REVIEW EXP C — spectrum/rank-controlled target family (deepest: one-mechanism vs two).
Confound: 'η* tracks N' might be 'η* tracks effective conditional rank', with N and rank co-varying.
Hold alphabet, positions, per-position marginals, M, N, examples fixed; independently vary the RANK of the
centered target-code matrix:
  random codewords (4 independent symbols)  -> HIGH rank
  compositional (u,v,u,v), u,v uniform      -> LOW rank, IDENTICAL per-position marginals
C1: fixed N=500, random vs compositional -> does η* move with rank at fixed N?
C2: vary N within each family -> does η* keep falling with N even where rank has saturated?
Verdict: η* moves with rank at fixed N (and tracks rank not N when they're decoupled) -> rank is the primitive,
         one-mechanism story (rote/compositional exponents fall out of rank). Else cardinality (N) survives.
"""
import json, os, math, random
import numpy as np
import torch, torch.nn.functional as F
import collapse_lib as L

device = L.device; vocab = L.NSPEC + L.ALPHA
NSPEC, ALPHA, LB, LA, BOS, SEP, EOS = L.NSPEC, L.ALPHA, L.LB, L.LA, L.BOS, L.SEP, L.EOS
CONTENT = list(range(NSPEC, NSPEC + ALPHA))
GRID = [3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2]
STEPS, SEEDS = 5000, 2


def targets_random(N, seed=0):
    rng = random.Random(seed); used = set(); outs = []
    while len(outs) < N:
        o = tuple(rng.choice(CONTENT) for _ in range(LA))
        if o not in used: used.add(o); outs.append(o)
    return outs

def targets_comp(N, seed=0):
    """Low-rank: o=(u,v,u,v). Distinct (u,v) pairs -> N<=26^2=676. Uniform per-position marginals."""
    assert N <= ALPHA * ALPHA, f"comp family maxes at {ALPHA*ALPHA} distinct"
    rng = random.Random(seed); pairs = [(u, v) for u in CONTENT for v in CONTENT]; rng.shuffle(pairs)
    return [(u, v, u, v) for (u, v) in pairs[:N]]

def code_stats(outs):
    """Centered one-hot code matrix (N x LA*ALPHA): full rank + effective rank + per-position marginal entropy."""
    Mtx = np.zeros((len(outs), LA * ALPHA))
    for i, o in enumerate(outs):
        for j, s in enumerate(o): Mtx[i, j * ALPHA + (s - NSPEC)] = 1.0
    Mc = Mtx - Mtx.mean(0)
    sv = np.linalg.svd(Mc, compute_uv=False); sv = sv[sv > 1e-9]
    rank = int(len(sv))
    p = sv**2 / (sv**2).sum(); eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    # mean per-position marginal entropy ratio (matched-ness check)
    hr = []
    for j in range(LA):
        cnt = np.bincount([o[j] - NSPEC for o in outs], minlength=ALPHA).astype(float); cnt /= cnt.sum()
        hr.append(-(cnt * np.log(cnt + 1e-12)).sum() / math.log(ALPHA))
    return rank, round(eff, 1), round(float(np.mean(hr)), 3)

def build_custom(outs, M, seed=0):
    rng = random.Random(seed + 1); N = len(outs)
    assign = [i % N for i in range(M)]; rng.shuffle(assign)
    used = set(); seqs = []
    for i in range(M):
        while True:
            b = tuple(rng.choice(CONTENT) for _ in range(LB))
            if b not in used: used.add(b); break
        seqs.append([BOS, *b, SEP, *outs[assign[i]], EOS])
    return torch.tensor(seqs)

def eta_star(X):
    frac = {}
    for eta in GRID:
        fr = [L.classify(*L.train(X, eta, s, STEPS, vocab)[1:]) for s in range(SEEDS)]
        frac[eta] = (fr.count("trap") + 0.5 * fr.count("partial")) / SEEDS
    return L.eta_star(frac), frac


def main():
    os.makedirs("card_ckpts", exist_ok=True); res = {"C1": {}, "C2": {}}
    print("=== EXP C rank-controlled targets ===\n")

    print("--- C1: fixed N=500, M=4000 (epp=8); random (high rank) vs compositional (low rank) ---")
    for kind, fn in [("random", targets_random), ("comp", targets_comp)]:
        outs = fn(500, seed=0); rank, eff, hr = code_stats(outs)
        X = build_custom(outs, 4000, seed=0); es, frac = eta_star(X)
        res["C1"][kind] = {"N": 500, "rank": rank, "eff_rank": eff, "perpos_H": hr, "eta_star": es, "frac": frac}
        print(f"  {kind:>6}: code rank={rank:>3} eff_rank={eff:>5} perpos_H/Hmax={hr} | "
              f"η*={('%.4f'%es) if es else 'n/a':>8}  " + " ".join(f"{e:.3f}:{frac[e]:.2f}" for e in GRID))
    r, c = res["C1"]["random"], res["C1"]["comp"]
    print(f"  → at FIXED N=500: rank {c['rank']}→{r['rank']} ({c['rank']*1.0/max(r['rank'],1):.0%}), "
          f"η* comp={c['eta_star']} vs random={r['eta_star']}")
    print(f"     {'η* MOVES with rank → rank matters' if c['eta_star'] and r['eta_star'] and abs(math.log(c['eta_star']/r['eta_star']))>0.3 else 'η* ~unchanged with rank at fixed N → N is the primitive, not rank'}\n")

    print("--- C2a: random family, vary N (rank saturates ~%d for N>~that) ---" % (LA*(ALPHA-1)))
    for N in [500, 2000]:
        outs = targets_random(N, seed=0); rank, eff, hr = code_stats(outs)
        X = build_custom(outs, 8000, seed=0); es, frac = eta_star(X)
        res["C2"][f"random_{N}"] = {"N": N, "rank": rank, "eff_rank": eff, "eta_star": es}
        print(f"  random N={N:>4}: rank={rank} eff={eff} η*={('%.4f'%es) if es else 'n/a'}")
    print("--- C2b: compositional family, vary N at ~saturated low rank ---")
    for N in [100, 300, 676]:
        outs = targets_comp(N, seed=0); rank, eff, hr = code_stats(outs)
        X = build_custom(outs, 8000, seed=0); es, frac = eta_star(X)
        res["C2"][f"comp_{N}"] = {"N": N, "rank": rank, "eff_rank": eff, "eta_star": es}
        print(f"  comp   N={N:>4}: rank={rank} eff={eff} η*={('%.4f'%es) if es else 'n/a'}")

    print("\n=== VERDICT ===")
    print("If η* fell with N in the original sweep while code rank SATURATES (~%d) for all those N," % (LA*(ALPHA-1)))
    print("then η* tracks N, not code rank. C1 tests rank at fixed N; C2 tests N at fixed/saturated rank.")
    json.dump(res, open("card_ckpts/rev_rank.json", "w"), indent=2)
    print("saved card_ckpts/rev_rank.json")


if __name__ == "__main__":
    main()
