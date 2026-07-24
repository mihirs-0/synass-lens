#!/usr/bin/env python
"""
Causal experiments for the (eta, K) phase diagram  —  Tier 0 + Tier 1.

Driven by an external diagnosis: the two "permanent-stuck" failure axes (wrong
optimizer at fixed eta; too-high eta with AdamW) may be one inequality
    Lambda  =  eta * (mu_eff - lambda_eff)  -  eta^2 * sigma^2 / 2
(escape iff Lambda > 0), and the published eta*(K) exponent may be confounded
with D = K * n_b because the boundary sweep held n_b = 1000 fixed.

------------------------------------------------------------------------------
TIER 0  —  RMSProp lambda falsifier
------------------------------------------------------------------------------
Does decoupled / zero weight decay gate the escape, and is the coupled-L2
stuck phase a *collapsed* boundary (stuck at every eta) or merely a *shifted*
one?  Branch protocol matched to the published 9-cell
(k=10, branch_step=1500, max_steps=6000); torch.optim.RMSprop(weight_decay=wd)
applies COUPLED L2, so wd=0.0 is the lambda=0 cell and wd=0.01 is coupled L2.

  cells:  branch_config=rmsprop, k=10, branch_step=1500, max_steps=6000
          wd  in {0.01, 0.0}      (0.01 reproduces 9-cell "rmsprop->stuck" ANCHOR;
                                    0.0  = no decay = lambda=0)
          eta in {1e-3, 3e-3, 6e-3}   (1e-3,3e-3 below eta*(10)~5e-3; 6e-3 above)
          seed in {0, 1}

  PRE-REGISTERED (unified inequality):
    wd=0.0 :  TRANSITION at low eta (1e-3, maybe 3e-3);  STUCK at high eta (6e-3).
    wd=0.01:  STUCK at ALL eta            (boundary collapsed to zero).
  FALSIFIERS:
    wd=0.0 stuck at every eta            -> "decay pins it" is WRONG (mu_eff<=0).
    wd=0.01 transitions at 1e-3          -> boundary SHIFTED, not collapsed.

------------------------------------------------------------------------------
TIER 1  —  fixed-D confound:  is eta*(K) a K-law or a D-law?
------------------------------------------------------------------------------
The published eta*(K)=0.048*K^-0.83 fit held n_b=1000, so K and D=K*n_b were
proportional (repo's own joint_nb_k_q_model.json: joint_identifiable=false).
Break the confound by locating eta* on two triplets:

  fixed-D triplet  (D=10000):  (k=5,nb=2000)  (k=10,nb=1000)  (k=20,nb=500)
  varying-D triplet (k=10):    (k=10,nb=500)=D5k  (k=10,nb=1000)=D10k  (k=10,nb=2000)=D20k
  eta grid: {3e-3, 5e-3, 8e-3, 1.2e-2, 1.8e-2}   (brackets eta* for K in [5,20])
  seeds: {0,1};  AdamW, wd=0.01, prefix_len=1 (all K<=36, single construction).

  PRE-REGISTERED:
    H1 (softmax gain ~ 1/K):  fixed-D triplet eta* DIFFERS by K (eta*(5)>eta*(10)>eta*(20));
                              varying-D triplet eta* SAME across n_b.
    H2 (trace erosion ~ D):   fixed-D triplet eta* SAME across K;
                              varying-D triplet eta* DIFFERS by D.
  (Opposite predictions on BOTH triplets -> clean adjudication.)

------------------------------------------------------------------------------
Serial, resumable (skips any cell whose status.json already exists), MPS-friendly.
Nothing trains until you pass --tier; --dry-run prints the matrix only.

  python eta_sweep/run_confound_experiments.py --tier 0 --dry-run
  python eta_sweep/run_confound_experiments.py --tier 0 --limit 1   # anchor smoke test
  python eta_sweep/run_confound_experiments.py --tier 0             # full Tier 0
  python eta_sweep/run_confound_experiments.py --tier 1             # full Tier 1
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
PROGRESS = RESULTS / "confound_experiments_progress.jsonl"


def _g(x: float) -> str:
    """Compact float formatting that matches config.py's f'{x:g}' run-name scheme."""
    return f"{x:g}"


# ----------------------------- Tier 0 -----------------------------
T0_K = 10
T0_BRANCH_STEP = 1500
T0_MAXSTEPS = 6000
T0_WDS = [0.01, 0.0]          # anchor (0.01, 1e-3) first
T0_ETAS = [1e-3, 3e-3, 6e-3]
T0_SEEDS = [0, 1]


def tier0_jobs() -> list[dict]:
    jobs = []
    for wd in T0_WDS:
        for eta in T0_ETAS:
            for seed in T0_SEEDS:
                subdir = f"tier0_lambda/wd{_g(wd)}_eta{_g(eta)}"
                status = RESULTS / subdir / f"rmsprop_seed{seed}" / "status.json"
                cmd = [PY, str(ETA_SWEEP / "run_optimizer_branch.py"),
                       "--branch-config", "rmsprop",
                       "--branch-step", str(T0_BRANCH_STEP),
                       "--max-steps", str(T0_MAXSTEPS),
                       "--eta", _g(eta), "--k", str(T0_K), "--seed", str(seed),
                       "--weight-decay", _g(wd),
                       "--output-subdir", subdir]
                jobs.append(dict(
                    jid=f"T0_rmsprop_wd{_g(wd)}_eta{_g(eta)}_s{seed}",
                    cmd=cmd, status=status,
                    meta=dict(tier=0, wd=wd, eta=eta, seed=seed, k=T0_K,
                              anchor=(wd == 0.01 and eta == 1e-3))))
    return jobs


# ----------------------------- Tier 1 -----------------------------
T1_ETAS = [3e-3, 5e-3, 8e-3, 1.2e-2, 1.8e-2]
T1_SEEDS = [0, 1]
T1_CELLS = [
    (5, 2000),    # D=10000   fixed-D triplet
    (10, 1000),   # D=10000   (shared between both triplets)
    (20, 500),    # D=10000
    (10, 500),    # D=5000    varying-D triplet
    (10, 2000),   # D=20000
]


def tier1_jobs() -> list[dict]:
    jobs = []
    for (k, nb) in T1_CELLS:
        D = k * nb
        subdir = f"tier1_confound/D{D}_K{k}_nb{nb}"
        for eta in T1_ETAS:
            for seed in T1_SEEDS:
                status = RESULTS / subdir / f"eta_{_g(eta)}_K_{k}_seed_{seed}" / "status.json"
                cmd = [PY, str(ETA_SWEEP / "run_single.py"),
                       "--eta", _g(eta), "--k", str(k), "--seed", str(seed),
                       "--n-unique-b", str(nb),
                       "--disambiguation-prefix-length", "1",
                       "--batch-size", "128",
                       "--optimizer", "adamw",
                       "--weight-decay", "0.01",
                       "--output-subdir", subdir]
                jobs.append(dict(
                    jid=f"T1_D{D}_K{k}_nb{nb}_eta{_g(eta)}_s{seed}",
                    cmd=cmd, status=status,
                    meta=dict(tier=1, k=k, nb=nb, D=D, eta=eta, seed=seed)))
    return jobs


def _read_status(status_path: Path) -> str:
    try:
        return json.loads(status_path.read_text()).get("status", "?")
    except Exception:
        return "?"


def run_job(job: dict, force: bool = False) -> tuple[str, str]:
    status = job["status"]
    if status.exists() and not force:
        s = _read_status(status)
        if s != "crashed":
            print(f"  SKIP {job['jid']} (exists: {s})", flush=True)
            return ("skipped", s)
        print(f"  RETRY {job['jid']} (previous: crashed)", flush=True)
    status.parent.mkdir(parents=True, exist_ok=True)
    log = status.parent / "orchestrator_run.log"
    print(f"  RUN  {job['jid']}", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run(job["cmd"], stdout=lf, stderr=subprocess.STDOUT,
                            cwd=str(REPO_ROOT)).returncode
    dt = time.time() - t0
    verdict = _read_status(status) if status.exists() else f"NO_STATUS(rc={rc})"
    rec = dict(jid=job["jid"], rc=rc, status=verdict, wall_s=round(dt, 1), **job["meta"])
    with open(PROGRESS, "a") as pf:
        pf.write(json.dumps(rec) + "\n")
    print(f"       -> rc={rc} status={verdict} wall={dt:.0f}s", flush=True)
    return ("ran", verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["0", "1"], help="which tier (omit for both)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run at most N jobs")
    ap.add_argument("--force", action="store_true", help="rerun even if status.json exists")
    ap.add_argument("--shard", type=str, default=None,
                    help="I/N: run only jobs whose index mod N == I (for parallel workers)")
    args = ap.parse_args()

    if args.tier == "0":
        jobs = tier0_jobs()
    elif args.tier == "1":
        jobs = tier1_jobs()
    else:
        jobs = tier0_jobs() + tier1_jobs()

    shard_tag = ""
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        jobs = [j for idx, j in enumerate(jobs) if idx % sn == si]
        shard_tag = f" shard {si}/{sn}"

    print(f"=== {len(jobs)} jobs (tier={args.tier or 'all'}{shard_tag}) ===", flush=True)
    for j in jobs:
        mark = "  [ANCHOR: expect stuck]" if j["meta"].get("anchor") else ""
        print(f"  {j['jid']}{mark}", flush=True)
        if args.dry_run:
            print(f"      cmd:    {' '.join(j['cmd'])}")
            print(f"      status: {j['status']}")
    if args.dry_run:
        print("dry-run: nothing executed.", flush=True)
        return

    n_ran = 0
    for j in jobs:
        if args.limit is not None and n_ran >= args.limit:
            print(f"limit {args.limit} reached; stopping.", flush=True)
            break
        run_job(j, force=args.force)
        n_ran += 1
    print("done.", flush=True)


if __name__ == "__main__":
    main()
