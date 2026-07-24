#!/usr/bin/env python
"""
Analyze TIER 1b — K-vs-D confound. Classifies each run by Dz TREND (never a live
stuck flag), brackets eta* per cell, prints the two-arm adjudication table, and
emits the CV(tau)-vs-distance-from-boundary critical-slowing check.

  python eta_sweep/analyze_confound_tier1b.py
"""
from __future__ import annotations
import glob, json, math, statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ROOT = RESULTS / "tier1b_confound"

# eta* under each hypothesis (for the CV-distance x-axis fallback only)
def classify(logp: Path):
    """Return (klass, tau, dz_max). klass in transit|slow|trapped|ambiguous.
    transit/slow => below eta* (escapes / would escape);  trapped => above eta*."""
    L = [json.loads(x) for x in open(logp) if x.strip()]
    sp = logp.with_name("status.json")
    s = json.loads(sp.read_text()) if sp.exists() else {}
    if s.get("status") == "transitioned":
        return "transit", s.get("transition_detected_step"), max(r["delta_z"] for r in L)
    dz = [r["delta_z"] for r in L]
    if not dz:
        return "ambiguous", None, 0.0
    n = len(dz); dzmax = max(dz)
    late = st.mean(dz[int(0.8 * n):]) if n >= 5 else dz[-1]
    mid = st.mean(dz[int(0.4 * n):int(0.6 * n)]) if n >= 5 else dz[0]
    if dzmax > 0.15 and late > mid + 0.05:
        return "slow", None, dzmax        # Dz climbing -> below eta*
    if dzmax < 0.10:
        return "trapped", None, dzmax     # flat -> above eta*
    return "ambiguous", None, dzmax


def load():
    cells = {}  # (D,K,nb) -> {eta -> {seed -> (klass,tau,dzmax)}}
    for lp in glob.glob(str(ROOT / "*/eta_*/log.jsonl")):
        lp = Path(lp)
        celldir = lp.parts[-3]                # D{D}_K{K}_nb{nb}
        D = int(celldir.split("_")[0][1:]); K = int(celldir.split("_K")[1].split("_")[0]); nb = int(celldir.split("_nb")[1])
        rn = lp.parts[-2]
        eta = float(rn.split("_K_")[0].replace("eta_", "")); seed = int(rn.split("_seed_")[1])
        cells.setdefault((D, K, nb), {}).setdefault(eta, {})[seed] = classify(lp)
    return cells


def cell_verdict(per_eta):
    """Per eta -> 'below'/'above'/'mixed'; then bracket eta*."""
    rows = {}
    for eta in sorted(per_eta):
        ks = [v[0] for v in per_eta[eta].values()]
        below = sum(k in ("transit", "slow") for k in ks)
        above = sum(k == "trapped" for k in ks)
        amb = sum(k == "ambiguous" for k in ks)
        tag = "below" if below == len(ks) else "above" if above == len(ks) else f"mixed({below}b/{above}a/{amb}?)"
        rows[eta] = tag
    etas = sorted(rows)
    lo = max([e for e in etas if rows[e] == "below"], default=None)   # largest escaping eta
    hi = min([e for e in etas if rows[e] == "above"], default=None)   # smallest trapped eta
    return rows, lo, hi


def fmt_bracket(lo, hi, etas):
    if lo is None and hi is not None: return f"eta* < {hi:g}"
    if hi is None and lo is not None: return f"eta* > {lo:g}"
    if lo is None and hi is None:     return "indeterminate"
    return f"{lo:g} < eta* < {hi:g}"


def main():
    cells = load()
    if not cells:
        print("no tier1b results yet."); return
    print("="*78)
    print("TIER 1b  —  per-cell classification (b=below eta*, a=above/trapped)")
    print("="*78)
    brackets = {}
    for key in sorted(cells, key=lambda k: (k[1], k[0])):
        D, K, nb = key
        rows, lo, hi = cell_verdict(cells[key])
        brackets[key] = (lo, hi)
        cols = "  ".join(f"{e:g}:{rows[e]}" for e in sorted(rows))
        print(f"  D={D:<6} K={K:<3} nb={nb:<5}  {fmt_bracket(lo,hi,sorted(rows)):22s}  | {cols}")

    def b(K, nb):
        for (D, k, n), v in brackets.items():
            if k == K and n == nb: return v, D
        return (None, None), None

    print("\n" + "="*78)
    print("ADJUDICATION")
    print("="*78)
    fd = [(5, 2000), (10, 1000), (20, 500)]   # fixed-D arm (D=10k), vary K
    fk = [(10, 500), (10, 1000), (10, 2000)]  # fixed-K arm (K=10), vary D
    print("  fixed-D arm (D=10k, vary K):")
    for K, nb in fd:
        (lo, hi), D = b(K, nb); print(f"     K={K:<3} (nb={nb}): {fmt_bracket(lo,hi,[])}")
    print("  fixed-K arm (K=10, vary D):")
    for K, nb in fk:
        (lo, hi), D = b(K, nb); print(f"     D={D} (nb={nb}): {fmt_bracket(lo,hi,[])}")

    def moves(arm):
        mids = []
        for K, nb in arm:
            (lo, hi), _ = b(K, nb)
            if lo and hi: mids.append(math.sqrt(lo*hi))
            elif lo: mids.append(lo*1.3)
            elif hi: mids.append(hi/1.3)
        if len(mids) < 2: return None
        return max(mids)/min(mids)
    rD, rK = moves(fd), moves(fk)
    print(f"\n  fixed-D arm eta* spread (max/min of bracket midpoints): {rD:.2f}x" if rD else "\n  fixed-D spread: n/a")
    print(f"  fixed-K arm eta* spread: {rK:.2f}x" if rK else "  fixed-K spread: n/a")
    if rD and rK:
        TH = 1.5
        if rD < TH and rK >= TH: verdict = "D-LAW: eta* flat across K (fixed D), moves with D. K^-0.83 is a D-law."
        elif rD >= TH and rK < TH: verdict = "K-LAW: eta* moves with K (fixed D), flat across D. K-exponent survives."
        elif rD >= TH and rK >= TH: verdict = "BOTH matter: report partial-derivative structure, no single exponent."
        else: verdict = "NEITHER arm moves much: brackets too coarse — refine the decisive cells (K=5, D-arm)."
        print(f"\n  >>> {verdict}")

    # ---- CV(tau) vs distance-from-boundary (critical slowing) ----
    print("\n" + "="*78); print("CRITICAL-SLOWING CHECK  CV(tau) across seeds vs distance below eta*")
    pts = []
    for key in cells:
        (lo, hi) = brackets[key]
        edge = hi if hi else (lo if lo else None)
        if not edge: continue
        for eta, seeds in cells[key].items():
            taus = [v[1] for v in seeds.values() if v[0] == "transit" and v[1]]
            if len(taus) >= 2:
                cv = st.pstdev(taus)/st.mean(taus)
                dist = edge/eta   # >1; ->1 means at the boundary
                pts.append((dist, cv, key, eta))
    pts.sort()
    for dist, cv, key, eta in pts:
        print(f"   D={key[0]} K={key[1]} eta={eta:g}  dist(eta*/eta)={dist:.2f}  CV(tau)={cv:.3f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        if pts:
            xs=[1/p[0] for p in pts]; ys=[p[1] for p in pts]   # x=eta/eta* (->1 at boundary)
            plt.figure(figsize=(6,4)); plt.scatter(xs,ys)
            plt.xlabel("eta / eta*  (->1 approaches boundary)"); plt.ylabel("CV(tau) across seeds")
            plt.title("Critical slowing: seed CV of tau vs proximity to eta*")
            out=RESULTS/"tier1b_cv_vs_distance.png"; plt.tight_layout(); plt.savefig(out,dpi=130)
            print(f"\n  CV plot -> {out}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
