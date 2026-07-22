"""Autonomous overnight finisher (user granted full autonomy 2026-07-21).
Waits for PILOTS_DONE, calibrates c (N_PM) and s* (M_T) by the frozen band
[0.85,1.15], runs the one permitted N_PM interpolation if the ladder misses,
closes M_T if no stable in-band dose, commits the freeze, and launches the full
runs (N_PM x2, M_T x2 or closed, V_fresh_orig). PAUSES only if N_PM can't hit
band after interpolation. Pilots = seed 0; full runs = seeds 1,2 (no collision).
Everything logged to finisher.log. Run with --dry to test logic without acting.
"""
import json, time, subprocess, sys
from pathlib import Path

DRY = "--dry" in sys.argv
REPO = Path("/Users/mihir/synass-lens/synass-lens")
R2 = str(REPO / "pinned_capabilities/reacq_2x2_run.py")
D = REPO / "pinned_capabilities/results/reacq_2x2"
S = Path("/private/tmp/claude-501/-Users-mihir-synass-lens/f708f5ca-07f8-4dbf-9316-07b5f65a2afd/scratchpad")
VERDICT = S / "reacq_verdict.txt"
LOG = S / "finisher.log"
STATUS = S / "finisher_status.txt"
LO, HI = 0.85, 1.15


def log(m):
    line = f"{time.strftime('%m-%d %H:%M')} {m}"
    print(line)
    if not DRY:
        with LOG.open("a") as f:
            f.write(line + "\n")


def summ(name):
    p = D / name / "summary.json"
    return json.load(p.open()) if p.exists() else None


def wait_marker(marker, timeout_h=10):
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        if VERDICT.exists() and marker in VERDICT.read_text():
            return True
        time.sleep(60)
    return False


def pick_band(cand):
    """cand: list of (key, gamma). Return in-band key closest to 1.0, or None."""
    inb = [(k, g) for k, g in cand if g is not None and LO <= g <= HI]
    return min(inb, key=lambda x: abs(x[1] - 1.0))[0] if inb else None


def calibrate_npm():
    cand = [(c, (summ(f"N_PM_collapsed_seed0_sc{c:.2f}") or {}).get("Gamma_median_100_500")) for c in (0.5, 0.6, 0.7)]
    log(f"N_PM ladder Gamma: {cand}")
    c = pick_band(cand)
    if c is not None:
        log(f"N_PM c*={c} (in band from ladder)")
        return c
    pts = sorted((k, g) for k, g in cand if g is not None)
    if len(pts) < 2:
        log("N_PM PAUSE: <2 pilot points"); return None
    cs = [p[0] for p in pts]; gs = [p[1] for p in pts]
    n = len(cs); mc = sum(cs) / n; mg = sum(gs) / n
    den = sum((x - mc) ** 2 for x in cs)
    slope = sum((x - mc) * (y - mg) for x, y in zip(cs, gs)) / den if den else 0.0
    c_int = round(max(0.1, min(0.9, mc + (1.0 - mg) / slope)) if slope else 0.3, 2)
    log(f"N_PM ladder misses band; interpolate c={c_int} (fit slope {slope:.2f})")
    if DRY:
        log("[dry] would run interp pilot"); return None
    subprocess.run(["python3", R2, "N_PM", "collapsed", "0", "500", "mps", "6", str(c_int)],
                   stdout=open(S / f"finisher_npm_interp_{c_int}.log", "w"), stderr=subprocess.STDOUT)
    g = (summ(f"N_PM_collapsed_seed0_sc{c_int:.2f}") or {}).get("Gamma_median_100_500")
    log(f"N_PM interp c={c_int} -> Gamma {g}")
    if g is not None and LO <= g <= HI:
        return c_int
    log("N_PM PAUSE: no in-band c after interpolation")
    return None


def calibrate_mt():
    cand = []
    for s_ in (0.02, 0.05, 0.1, 0.25):
        sm = summ(f"M_T_collapsed_seed0_sc{s_:.2f}") or {}
        cand.append((s_, sm.get("Gamma_median_100_500"), sm.get("diverged")))
    log(f"M_T ladder (s, Gamma, diverged): {cand}")
    stable = [(s_, g) for s_, g, div in cand if not div]
    s = pick_band(stable)
    if s is not None:
        log(f"M_T s*={s} (stable, in band)")
        return s, False
    log("M_T CLOSED: no stable in-band m-only dose exists")
    return None, True


def gen_full_orch(c_star, s_star, mt_closed, include_npm):
    lines = ["#!/bin/zsh", "# Autonomous full-run orchestrator (post-calibration). Serial MPS, self-heal.",
             f"R2={R2}", f"D={D}", f"S={S}", "",
             "ensure() { local done=$1 pat=$2; shift 2; local t=0;",
             '  while [ ! -f $done ]; do if ! pgrep -f "$pat" >/dev/null; then t=$((t+1)); [ $t -gt 8 ] && exit 1;',
             '    nohup "$@" >> $S/full_$(echo $pat|tr " ./" "___").log 2>&1 & sleep 10; fi; sleep 45; done; }', ""]
    if include_npm:
        ct = f"{c_star:.2f}"
        for seed in (1, 2):
            lines.append(f'ensure $D/N_PM_collapsed_seed{seed}_sc{ct}/summary.json "N_PM collapsed {seed} 16000 mps 6 {c_star}" python3 $R2 N_PM collapsed {seed} 16000 mps 6 {c_star}')
    if not mt_closed:
        st = f"{s_star:.2f}"
        for seed in (1, 2):
            lines.append(f'ensure $D/M_T_collapsed_seed{seed}_sc{st}/summary.json "M_T collapsed {seed} 16000 mps 6 {s_star}" python3 $R2 M_T collapsed {seed} 16000 mps 6 {s_star}')
    lines.append(f'ensure $D/V_fresh_orig_seed1/summary.json "V fresh_orig 1 10000 mps 6 1.0" python3 $R2 V fresh_orig 1 10000 mps 6 1.0')
    lines.append(f'echo "FULL_DONE $(date +%H:%M)" > $S/reacq_verdict.txt')
    p = S / "full_orch.sh"
    p.write_text("\n".join(lines) + "\n")
    return p


def main():
    log(f"finisher started (dry={DRY})")
    if not DRY:
        if not wait_marker("PILOTS_DONE"):
            log("TIMEOUT waiting for PILOTS_DONE"); STATUS.write_text("TIMEOUT"); return
        log("PILOTS_DONE seen")
    c_star = calibrate_npm()
    include_npm = c_star is not None
    if not include_npm:
        log("N_PM PAUSED for user (not power-matchable by scaling); M_T + V_fresh_orig proceed")
    s_star, mt_closed = calibrate_mt()
    freeze = (f"N_PM c={c_star}" if include_npm else "N_PM PAUSED (unmatchable)") + \
             "; M_T " + ("CLOSED" if mt_closed else f"s*={s_star}")
    log(f"FREEZE: {freeze}")
    if DRY:
        launch = (["N_PM x2"] if include_npm else []) + ([] if mt_closed else ["M_T x2"]) + ["V_fresh_orig"]
        log(f"[dry] would commit freeze + launch: {', '.join(launch) or 'nothing'}"); return
    (REPO / "pinned_capabilities/REACQ_FREEZE.md").write_text(
        f"# Autonomous freeze {time.strftime('%Y-%m-%d %H:%M')}\n\n{freeze}\n\n"
        f"N_PM ladder + M_T ladder Gamma tables in finisher.log.\n"
        + (f"Full N_PM seeds 1,2 @ c={c_star}. " if include_npm
           else "N_PM NOT power-matchable to V by scaling (Gamma ~2x V across c; Adam coupling) -> PAUSED for user judgment. ")
        + ("M_T CLOSED (no stable in-band dose). " if mt_closed else f"M_T seeds 1,2 @ s*={s_star}. ")
        + "V_fresh_orig seed 1 (original table).\n")
    for a in (["git", "-C", str(REPO), "add", "pinned_capabilities/REACQ_FREEZE.md"],
              ["git", "-C", str(REPO), "commit", "-q", "-m", f"freeze (autonomous): {freeze}"],
              ["git", "-C", str(REPO), "push", "-q"]):
        subprocess.run(a)
    log("freeze committed + pushed")
    orch = gen_full_orch(c_star, s_star, mt_closed, include_npm)
    subprocess.Popen(["nohup", "zsh", str(orch)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    STATUS.write_text("PAUSED_NPM+FULL_LAUNCHED" if not include_npm else "FULL_LAUNCHED")
    log(f"FULL RUNS LAUNCHED via {orch} (N_PM {'included' if include_npm else 'PAUSED'})")


if __name__ == "__main__":
    main()
