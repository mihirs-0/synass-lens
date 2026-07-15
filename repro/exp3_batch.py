#!/usr/bin/env python
"""EXP 3 — batch size: sharp test of the dilution mechanism.
Dilution = per-example gradients cancel when averaged. If the collapse is driven by minibatch averaging,
batch size moves eta* (bigger batch -> more averaging -> weaker surviving signal -> lower eta*). If instead
the cancellation is in the EXPECTED (full-batch) gradient -- as the SNR2-over-full-data measurement found --
then eta* is ~flat in batch and the collapse PERSISTS at full batch (deterministic, not noise-driven).
Sweep batch {64,256,1024,full} at FIXED N, M, steps. Measure eta* each. Full-batch = decisive.

NOTE (prior): SNR2 was measured over the full dataset and tracked eta*, so I EXPECT persistence at full
batch + weak batch-dependence (deterministic dilution). Reported against this prior either way."""
import json, os
import collapse_lib as L

N, M, STEPS, SEEDS = 1000, 4000, 3000, 2
BATCHES = [64, 256, 1024, "full"]   # 'full' = M
vocab = L.NSPEC + L.ALPHA
path = "card_ckpts/exp3_batch.json"
res = json.load(open(path)) if os.path.exists(path) else {"cells": {}}
def save(): json.dump(res, open(path, "w"), indent=2)


def main():
    print(f"EXP3 batch. N={N} M={M} steps={STEPS} seeds={SEEDS} batches={BATCHES} eta={L.ETA_GRID}")
    print("(fixed N,M,steps; only batch varies)\n")
    X, *_ = L.build_task(N, M=M, seed=0)
    for b in BATCHES:
        bk = str(b)
        if bk in res["cells"]:
            print(f"  batch={bk}: (cached) eta*={res['cells'][bk]['eta_star']}"); continue
        frac = {}
        for eta in L.ETA_GRID:
            fr = []
            for s in range(SEEDS):
                bb = M if b == "full" else b
                m, fl, kl = L.train(X, eta, s, STEPS, vocab, batch=bb)
                fr.append(L.classify(fl, kl))
            frac[eta] = (fr.count("trap") + 0.5 * fr.count("partial")) / SEEDS
        es = L.eta_star(frac)
        res["cells"][bk] = {"batch": bk, "eta_star": es, "frac": frac}; save()
        print(f"  batch={bk:>5}: eta*={('%.4f'%es) if es else 'n/a':>8} | "
              + " ".join(f"{e:.3f}:{frac[e]:.2f}" for e in L.ETA_GRID))
    print("\n=== SUMMARY: eta* vs batch ===")
    for b in BATCHES:
        c = res["cells"].get(str(b), {})
        print(f"  batch={str(b):>5}: eta*={c.get('eta_star')}")
    full = res["cells"].get("full", {})
    print(f"\nfull-batch collapse present? {'YES (deterministic, not noise-driven)' if full.get('eta_star') else 'check frac'}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
