#!/usr/bin/env python
"""
Batch for the arrow note: SWEEP A (two-faced strip + forward ceiling control)
+ RESCUE (release-vs-scar) + optional REVERSE-ARM (hysteresis). Shards across workers.

SWEEP A: forward & inverse across eta. Forward goes UP TO DIVERGENCE (find the
  actual ceiling, not just clear 6e-3 -> distinguishes "no trap" from "no trap we
  looked for"). Inverse logs Delta_z (Tier-0 fingerprint on the trapped calls).
RESCUE: train inverse at eta=6e-3 to 25k (full trap), drop eta->1e-3, watch to 40k.
  ruin/active-stabilization => prompt escape (router re-leveled, never damaged);
  loss-of-plasticity        => stays trapped (25k high-eta scarred it). Never tested.
REVERSE-ARM (optional): converge inverse at eta=1e-3, then RAISE eta->1.2e-2 at 12k;
  survival = hysteresis (formed router owns a larger stability margin than its seed).

  python eta_sweep/run_gate_sweep.py --dry-run
  python eta_sweep/run_gate_sweep.py --shard 0/6   (x6)
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ETA_SWEEP = Path(__file__).resolve().parent
REPO = ETA_SWEEP.parent
RESULTS = ETA_SWEEP / "results"
PY = sys.executable
RUNNER = str(ETA_SWEEP / "gate_mirror_runner.py")

FWD_ETAS = [1e-3, 3e-3, 6e-3, 1.2e-2, 2.5e-2, 5e-2, 1e-1, 2e-1, 5e-1]  # up to divergence
INV_ETAS = [1e-3, 3e-3, 6e-3, 1.2e-2, 2.5e-2, 5e-2]
SEEDS = [0, 1]
K, NB = 10, 1000


def _g(x): return f"{x:g}"


def jobs():
    js = []

    def add(direction, eta, max_steps, subdir, drop_at=None, after=None, tag=""):
        cell = f"{direction}_eta{_g(eta)}_K{K}_nb{NB}_seed%d"
        for s in SEEDS:
            cmd = [PY, RUNNER, "--direction", direction, "--eta", _g(eta),
                   "--k", str(K), "--n-unique-b", str(NB), "--seed", str(s),
                   "--max-steps", str(max_steps), "--output-subdir", subdir]
            if drop_at is not None:
                cmd += ["--eta-drop-at", str(drop_at), "--eta-after", _g(after)]
            status = RESULTS / subdir / (cell % s) / "status.json"
            js.append(dict(jid=f"{tag}{direction}_eta{_g(eta)}_s{s}", cmd=cmd, status=status))

    # SWEEP A
    for eta in FWD_ETAS:
        add("forward", eta, 25000, "gate_sweep", tag="A:")
    for eta in INV_ETAS:
        add("inverse", eta, 25000, "gate_sweep", tag="A:")
    # RESCUE (drop eta after the trap forms)
    add("inverse", 6e-3, 40000, "gate_rescue", drop_at=25000, after=1e-3, tag="R:")
    # REVERSE-ARM (raise eta on a converged run)
    add("inverse", 1e-3, 30000, "gate_reverse", drop_at=12000, after=1.2e-2, tag="H:")
    return js


def run_job(j, force=False):
    st = j["status"]
    if st.exists() and not force:
        try:
            v = json.loads(st.read_text()).get("verdict", "?")
        except Exception:
            v = "?"
        print(f"  SKIP {j['jid']} ({v})", flush=True); return
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
    tag = ""
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        js = [x for idx, x in enumerate(js) if idx % n == i]
        tag = f" shard {i}/{n}"
    print(f"=== gate sweep: {len(js)} jobs{tag} ===", flush=True)
    for j in js:
        print(f"  {j['jid']}", flush=True)
        if a.dry_run:
            print(f"      {' '.join(j['cmd'])}")
    if a.dry_run:
        return
    for j in js:
        run_job(j, force=a.force)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
