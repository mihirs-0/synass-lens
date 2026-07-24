#!/usr/bin/env python
"""
TIER 1b — the corrected K-vs-D confound experiment (decision-grade).

Separates two variables the published eta*(K)=0.048*K^-0.83 sweep FUSED by holding
n_b=1000 (so K and D=K*n_b moved together; the repo's joint_nb_k_q_model.json
already concedes joint_identifiable=false). Two arms pivot on (K=10, n_b=1000, D=10k):

            n_b=2000   n_b=1000   n_b=500
  K=5        D=10k        -          -        <- fixed-D arm (vary K)
  K=10       D=20k      D=10k      D=5k       <- fixed-K arm (vary n_b)
  K=20         -          -        D=10k      <- fixed-D arm (vary K)

  fixed-D arm (all D=10k): (K=5,nb2000) (K=10,nb1000) (K=20,nb500)
  fixed-K arm (all K=10) : (K=10,nb500) (K=10,nb1000) (K=10,nb2000)
  5 distinct configs (pivot shared), 2 seeds, ~4 eta-levels each ~= 40 runs.

ADJUDICATION TABLE (write the outcome map BEFORE launch so it adjudicates, not rationalizes):
  +--------------------------------------------+-------------------------------------------------+
  | eta* flat across fixed-D, moves on fixed-K | It was D all along. gain~1/K story dies;        |
  |                                            | erosion~D wins. eta*(K)=K^-0.83 is a D-law.      |
  | eta* moves across fixed-D, flat on fixed-K | Real K-law. Narrowed K-exponent survives.       |
  | eta* moves on BOTH                         | Both matter; report partial-derivative          |
  |                                            | structure, do NOT claim a single exponent.      |
  +--------------------------------------------+-------------------------------------------------+

NON-NEGOTIABLE CONFIG (this is exactly what invalidated the first grid):
  * early-stuck DISABLED  (--stuck-patience 1e8); near eta* tau DIVERGES (critical slowing:
    historical K=10/eta3e-3 transitioned at tau=24000) -> a live stuck flag manufactures fakes.
  * budget 25k uniform across cells. We classify post-hoc on the Dz TREND, not a live loss flag,
    so a near-boundary SLOW run shows a climbing Dz within 25k (validated in Tier-0 confirm:
    eta3e-3 climbed to Dz~1.7 by 25k while eta6e-3 stayed flat ~0.02). Uniform budget keeps the
    cross-cell comparison unbiased; the trend (not the budget) prevents fake stucks.
  * classify: transitioned OR Dz climbing -> below eta*; Dz flat/declining -> trapped (above eta*).
  * eta* reported as a BRACKET per cell (e.g. 3e-3 < eta* < 6e-3). Never a point. Never "forever".

n_b threading VERIFIED end-to-end (D scales 5k/10k/20k with nb 500/1000/2000).

Usage:
  python eta_sweep/run_confound_tier1b.py --dry-run
  python eta_sweep/run_confound_tier1b.py --shard 0/6   (x6 for parallel workers)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ETA_SWEEP = Path(__file__).resolve().parent
REPO_ROOT = ETA_SWEEP.parent
RESULTS = ETA_SWEEP / "results"
PY = sys.executable
PROGRESS = RESULTS / "tier1b_progress.jsonl"

BUDGET = 25000
STUCK_OFF = 100_000_000   # effectively disables early-stuck

# (K, n_b, arm-label); D = K*n_b
CELLS = [
    (5, 2000, "fixedD"),    # D=10k
    (10, 1000, "pivot"),    # D=10k  (shared by both arms)
    (20, 500, "fixedD"),    # D=10k
    (10, 500, "fixedK"),    # D=5k
    (10, 2000, "fixedK"),   # D=20k
]

# Per-cell eta grids, each chosen to bracket the cell's plausible eta* under BOTH
# hypotheses (D-law ~4e-3 vs K-law: eta*(5)~1.2e-2, eta*(10)~4e-3, eta*(20)~3e-3).
ETA_GRID = {
    5:  [3e-3, 5e-3, 8e-3, 1.3e-2],     # straddles D-law(~4e-3) and K-law(~1.2e-2) -> most decisive cell
    10: [2e-3, 3e-3, 4.5e-3, 6.5e-3],   # SAME grid on all three K=10 cells -> clean fixed-K comparison
    20: [1.5e-3, 2.5e-3, 4e-3, 6e-3],   # K-law(~3e-3) vs D-law(~4e-3) are close; brackets both
}
SEEDS = [0, 1]


def _g(x): return f"{x:g}"


def jobs():
    js = []
    for (k, nb, arm) in CELLS:
        D = k * nb
        subdir = f"tier1b_confound/D{D}_K{k}_nb{nb}"
        for eta in ETA_GRID[k]:
            for seed in SEEDS:
                status = RESULTS / subdir / f"eta_{_g(eta)}_K_{k}_seed_{seed}" / "status.json"
                cmd = [PY, str(ETA_SWEEP / "run_single.py"),
                       "--eta", _g(eta), "--k", str(k), "--seed", str(seed),
                       "--n-unique-b", str(nb),
                       "--disambiguation-prefix-length", "1",
                       "--batch-size", "128", "--optimizer", "adamw", "--weight-decay", "0.01",
                       "--max-steps", str(BUDGET),
                       "--stuck-patience", str(STUCK_OFF),
                       "--output-subdir", subdir]
                js.append(dict(jid=f"D{D}_K{k}_nb{nb}_eta{_g(eta)}_s{seed}",
                               cmd=cmd, status=status,
                               meta=dict(k=k, nb=nb, D=D, arm=arm, eta=eta, seed=seed)))
    return js


def _status(p):
    try: return json.loads(p.read_text()).get("status", "?")
    except Exception: return "?"


def run_job(j, force=False):
    st = j["status"]
    if st.exists() and not force:
        s = _status(st)
        if s != "crashed":
            print(f"  SKIP {j['jid']} ({s})", flush=True); return
        print(f"  RETRY {j['jid']} (was crashed)", flush=True)
    st.parent.mkdir(parents=True, exist_ok=True)
    print(f"  RUN  {j['jid']}", flush=True)
    t0 = time.time()
    with open(st.parent / "orchestrator_run.log", "w") as lf:
        rc = subprocess.run(j["cmd"], stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)).returncode
    verdict = _status(st) if st.exists() else f"NO_STATUS(rc={rc})"
    with open(PROGRESS, "a") as pf:
        pf.write(json.dumps(dict(jid=j["jid"], rc=rc, status=verdict,
                                 wall_s=round(time.time()-t0, 1), **j["meta"])) + "\n")
    print(f"       -> {verdict}  {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default=None, help="I/N for parallel workers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    js = jobs()
    tag = ""
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        js = [j for idx, j in enumerate(js) if idx % n == i]
        tag = f" shard {i}/{n}"
    print(f"=== TIER 1b: {len(js)} jobs{tag}  (budget {BUDGET}, early-stuck OFF) ===", flush=True)
    for j in js:
        print(f"  {j['jid']:34s} [{j['meta']['arm']}]", flush=True)
        if a.dry_run:
            print(f"      {' '.join(j['cmd'])}")
    if a.dry_run:
        print("dry-run: nothing executed."); return
    n_ran = 0
    for j in js:
        if a.limit is not None and n_ran >= a.limit:
            print(f"limit {a.limit} reached"); break
        run_job(j, force=a.force); n_ran += 1
    print("done.", flush=True)


if __name__ == "__main__":
    main()
