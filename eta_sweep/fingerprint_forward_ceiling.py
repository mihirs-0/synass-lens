#!/usr/bin/env python
"""
Fingerprint the flat high-eta FORWARD cells: structured-trap vs chaotic-non-learning.
Does NOT use the inverse-built 'trapped' classifier. Computes, on logs/data on disk:
  (1) loss variance over last 5k steps  (structured trap ~ very low; chaotic ~ high)
  (2) analytic anchors for the FORWARD task (uniform / unigram / per-position) and
      whether the flat value matches a meaningful context-free predictor or just init
  (3) trajectory: loss@500 vs loss@last  (did it learn anything?)
  (4) per-position + exact-match acc at model_final.pt (if present)
  (5) inverse Delta_z fingerprint (comparison row)
"""
from __future__ import annotations
import glob, json, math, sys
import statistics as st
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
ROOT = Path("eta_sweep/results")


def loadlog(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def last5k_std(L):
    if len(L) < 3:
        return None, None, None
    last_step = L[-1]["step"]
    win = [r["train_loss"] for r in L if r["step"] >= last_step - 5000]
    if len(win) < 3:
        win = [r["train_loss"] for r in L[-50:]]
    return st.pstdev(win), st.mean(win), len(win)


def at_step(L, s):
    r = next((r for r in L if r["step"] >= s), None)
    return r["train_loss"] if r else None


def dz_max(L):
    dz = [r.get("delta_z") for r in L if r.get("delta_z") is not None]
    return max(dz) if dz else None


def anchors():
    """Analytic context-free losses for the FORWARD task (target = B + EOS)."""
    from eta_sweep.config import CellConfig
    from eta_sweep.run_single import build_legacy_cfg
    from src.data import create_tokenizer_from_config
    from eta_sweep.gate_mirror_runner import build
    cc = CellConfig(eta=6e-3, k=10, seed=0, n_unique_b=1000)
    tok = create_tokenizer_from_config(build_legacy_cfg(cc))
    V = tok.vocab_size
    train_ds, _ = build(cc, "forward", tok)
    pos = {}
    for i in range(min(3000, len(train_ds))):
        ex = train_ds[i]
        for p, l in enumerate(ex["labels"].tolist()):
            if l != -100:
                pos.setdefault(p, []).append(l)

    def H(toks):
        c = Counter(toks); n = len(toks)
        return -sum((v / n) * math.log(v / n) for v in c.values())
    per_pos = {p: H(t) for p, t in pos.items()}
    per_pos_avg = sum(per_pos.values()) / len(per_pos)
    unigram = H([t for ts in pos.values() for t in ts])
    return {
        "uniform_logV": math.log(V),
        "unigram": unigram,
        "per_position_avg": per_pos_avg,
        "per_pos_detail": {p: round(v, 3) for p, v in sorted(per_pos.items())},
        "n_scored": len(per_pos),
        "vocab": V,
    }


CELLS = [
    # (label, glob, category-hint)
    ("FWD eta2.5e-2 (flat?)", "gate_sweep/forward_eta0.025_*"),
    ("FWD eta5e-2  (flat?)",  "gate_sweep/forward_eta0.05_*"),
    ("FWD eta1.2e-2 (descending ref)", "gate_sweep/forward_eta0.012_*"),
    ("FWD eta6e-3 (converged ref)", "gate_sweep/forward_eta0.006_*"),
    ("INV eta6e-3 trapped (tier0_confirm)", "tier0_confirm/adamw_b1_0.9_wd0.01_eta0.006/*"),
    ("INV eta6e-3 trapped (gate)", "gate_mirror/inverse_eta0.006_*"),
]


def main():
    print("=== ANALYTIC ANCHORS (forward task: target = B(6 chars) + EOS) ===")
    A = anchors()
    print(f"  vocab={A['vocab']}  scored_positions={A['n_scored']}")
    print(f"  uniform   (log V)        = {A['uniform_logV']:.3f}")
    print(f"  unigram   (pooled freq)  = {A['unigram']:.3f}")
    print(f"  per-position avg         = {A['per_position_avg']:.3f}   <- best context-free predictor")
    print(f"  per-pos detail (pos:H)   = {A['per_pos_detail']}")
    print(f"  (inverse trap anchor = log K = 2.303 on the candidate metric, for comparison)\n")

    print("=== PER-CELL FINGERPRINTS ===")
    print(f"{'cell':40s} {'std(last5k)':>11} {'mean(last5k)':>12} {'loss@500':>9} {'loss@last':>9} {'dzmax':>6}")
    for label, g in CELLS:
        for d in sorted(glob.glob(str(ROOT / g))):
            lg = Path(d) / "log.jsonl"
            if not lg.exists():
                # tier0_confirm nests one deeper sometimes
                cand = list(Path(d).glob("log.jsonl")) or list(Path(d).glob("*/log.jsonl"))
                if not cand: continue
                lg = cand[0]
            L = loadlog(lg)
            if not L: continue
            sd, mn, nwin = last5k_std(L)
            seed = d.split("seed")[-1].split("/")[0] if "seed" in d else d.split("/")[-1]
            dz = dz_max(L)
            print(f"{label[:30]+' s'+seed:40s} {sd if sd is None else round(sd,4):>11} "
                  f"{mn if mn is None else round(mn,3):>12} {at_step(L,500):>9.3f} {L[-1]['train_loss']:>9.3f} "
                  f"{'' if dz is None else round(dz,3):>6}")


if __name__ == "__main__":
    main()
