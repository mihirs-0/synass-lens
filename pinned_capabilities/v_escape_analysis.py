"""Deliverable: tau(s) trend + escape curves for the weak-noise V-arm package.
Produces v_escape_tau_table.csv and v_escape_trend.png. No manuscript prose."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/Users/mihir/synass-lens/synass-lens/pinned_capabilities/results/v_escape")
RUNS = [("s0.25_seed0", 0.25), ("s0.50_seed1", 0.50), ("s0.75_seed2", 0.75), ("s1.00_seed3", 1.00)]
CLEAN = 800  # clean full-gradient escape (C0), s->0 reference

def load(name):
    rows = [json.loads(l) for l in (D / name / "metrics.jsonl").open() if l.strip()]
    ce = sorted({r["step"]: r["full_vocab_ce"] for r in rows if r.get("full_vocab_ce") is not None}.items())
    summ = json.load((D / name / "summary.json").open())
    return ce, summ

data = {s: load(name) for name, s in RUNS}

# ---- tau table ----
rows_csv = ["s,tau_onset,tau_solve,capped,last_step,final_ce,noise_seed"]
for name, s in RUNS:
    _, sm = data[s]
    rows_csv.append(f'{s},{sm["tau_onset"]},{sm["tau_solve"]},{sm["capped"]},{sm["last_step"]},{sm["final_ce"]:.4f},{sm["noise_seed"]}')
(D / "v_escape_tau_table.csv").write_text("\n".join(rows_csv) + "\n")

# ---- fits on tau_onset(s) ----
pts = [(s, data[s][1]["tau_onset"]) for _, s in RUNS if data[s][1]["tau_onset"]]
xs = [math.log(s) for s, t in pts]; ys = [math.log(t) for s, t in pts]
n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
a = my - b * mx

# ---- figure: escape curves + tau(s) ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
blues = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]  # light->dark = more noise (sequential)
for (name, s), col in zip(RUNS, blues):
    ce, sm = data[s]
    axL.plot([st for st, _ in ce], [c for _, c in ce], color=col, lw=1.4, label=f"s={s:.2f} (τ={sm['tau_onset']})")
    if sm["tau_onset"]:
        axL.scatter([sm["tau_onset"]], [3.0], color=col, s=28, zorder=5)
axL.axhline(3.0, color="#cccccc", lw=1, ls="--", zorder=0)
axL.axvline(CLEAN, color="#e6550d", lw=1.2, ls=":", label=f"clean escape ({CLEAN})")
axL.text(CLEAN + 60, 0.4, "clean", color="#e6550d", fontsize=8)
axL.set_xlabel("step"); axL.set_ylabel("first-token full-vocab CE")
axL.set_title("V-arm escape curves (noise scaled by s)")
axL.legend(fontsize=8, frameon=False); axL.set_ylim(0, 3.8)

# tau(s): measured points + clean reference; annotate saturation
ss = [s for s, t in pts]; tt = [t for s, t in pts]
axR.scatter(ss, tt, color="#2171b5", s=60, zorder=5, label="measured τ_onset")
axR.scatter([0.0], [CLEAN], color="#e6550d", s=60, marker="D", zorder=5, label=f"clean (s→0) = {CLEAN}")
grid = [i / 100 for i in range(10, 101)]
axR.plot(grid, [math.exp(a + b * math.log(g)) for g in grid], color="#999999", lw=1, ls="--", label=f"log-log fit (b={b:.2f})")
for s, t in pts:
    axR.annotate(str(t), (s, t), textcoords="offset points", xytext=(6, 6), fontsize=8)
axR.set_xlabel("noise scale s"); axR.set_ylabel("τ_onset (step)")
axR.set_title("Escape time vs noise: SLOW, not trapped (saturates ~4600)")
axR.legend(fontsize=8, frameon=False); axR.set_xlim(-0.05, 1.1); axR.set_ylim(0, 5200)
fig.tight_layout()
fig.savefig(D / "v_escape_trend.png", dpi=130)
print("wrote", D / "v_escape_tau_table.csv")
print("wrote", D / "v_escape_trend.png")
print(f"fit: log tau = {a:.3f} + {b:.3f} log s   |   MEASURED tau(1.0)=4600 (no extrapolation needed)")
print("\ntau table:")
print("\n".join(rows_csv))
