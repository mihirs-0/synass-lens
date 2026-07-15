#!/usr/bin/env python
"""
B-TEST: does the inverse-trap boundary eta*(B) scale with batch size B?
  eta* ∝ B  => noise-set boundary (minibatch sampling noise; T_eff = eta/B)
  eta* flat => deterministic source (not sampling noise)
The last open mechanism question; arbitrates the gain-control / sigma-source story.

Grid: inverse task, B ∈ {32,128,512} × eta ∈ {1e-3,3e-3,8e-3,1.6e-2} × 2 seeds = 24.
  If eta*∝B: trap pattern shifts diagonally (B=32 traps low, B=512 converges high).
  If flat:   same trap/converge boundary (~5e-3) at every B.
Classify by the Dz fingerprint (trap = loss at marginal floor + Dz≈0; converged = Dz large).

  python eta_sweep/run_btest.py --dry-run
  python eta_sweep/run_btest.py --shard 0/6  (x6)
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ETA_SWEEP = Path(__file__).resolve().parent
REPO = ETA_SWEEP.parent
RESULTS = ETA_SWEEP / "results"
PY = sys.executable
RUNNER = str(ETA_SWEEP / "gate_mirror_runner.py")

BATCHES = [32, 128, 512]
ETAS = [1e-3, 3e-3, 8e-3, 1.6e-2]
SEEDS = [0, 1]


def _g(x): return f"{x:g}"


def jobs():
    js = []
    for B in BATCHES:
        for eta in ETAS:
            for s in SEEDS:
                sub = f"gate_btest/B{B}"
                cell = f"inverse_eta{_g(eta)}_K10_nb1000_seed{s}"
                cmd = [PY, RUNNER, "--direction", "inverse", "--eta", _g(eta),
                       "--k", "10", "--n-unique-b", "1000", "--seed", str(s),
                       "--batch-size", str(B), "--max-steps", "25000", "--output-subdir", sub]
                js.append(dict(jid=f"B{B}_eta{_g(eta)}_s{s}",
                               cmd=cmd, status=RESULTS / sub / cell / "status.json"))
    return js


def run_job(j, force=False):
    st = j["status"]
    if st.exists() and not force:
        print(f"  SKIP {j['jid']}", flush=True); return
    st.parent.mkdir(parents=True, exist_ok=True)
    print(f"  RUN  {j['jid']}", flush=True)
    t0 = time.time()
    with open(st.parent / "orchestrator_run.log", "w") as lf:
        subprocess.run(j["cmd"], stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO))
    v = "?"
    if st.exists():
        try: v = json.loads(st.read_text()).get("verdict", "?")
        except Exception: pass
    print(f"       -> {v}  {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    js = jobs()
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        js = [x for idx, x in enumerate(js) if idx % n == i]
    print(f"=== B-test: {len(js)} jobs ===", flush=True)
    for j in js:
        print(f"  {j['jid']}", flush=True)
        if a.dry_run:
            continue
    if a.dry_run:
        return
    for j in js:
        run_job(j, force=a.force)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
