"""H4 follow-up: moving-average m_t coherence test.

Tests the slow-drift interpretation from the appendix interpretation
paragraph: if m_t fluctuates around a slowly-drifting mean direction,
moving-average m_t over a 500- or 1000-step window should be coherent
across consecutive non-overlapping windows even when per-50-step m_t is not.

Inputs: H4 retrain ckpts in eta_sweep/results/full_optstate/eta_0.001_K_10_seed_0/
        160 opt-state ckpts at 50-step cadence through step 8000.
Outputs: figures + window-medians table; saved to analysis_outputs/v3_review_addons/.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RUN  = REPO / "eta_sweep" / "results" / "full_optstate" / "eta_0.001_K_10_seed_0"
OUT  = REPO / "analysis_outputs" / "v3_review_addons"
OUT.mkdir(parents=True, exist_ok=True)

PIPE_KEYS = ["blocks.0.attn.W_Q","blocks.0.attn.W_K","blocks.0.attn.W_V","blocks.0.attn.W_O",
             "blocks.0.mlp.W_in","blocks.0.mlp.W_out",
             "blocks.2.mlp.W_in","blocks.2.mlp.W_out",
             "blocks.3.attn.W_Q","blocks.3.attn.W_K","blocks.3.attn.W_V","blocks.3.attn.W_O"]
CTRL_KEYS = ["blocks.1.attn.W_Q","blocks.1.attn.W_K","blocks.1.attn.W_V","blocks.1.attn.W_O",
             "blocks.2.attn.W_Q","blocks.2.attn.W_K","blocks.2.attn.W_V","blocks.2.attn.W_O",
             "blocks.3.mlp.W_in","blocks.3.mlp.W_out","embed.W_E"]


def load_log(path):
    lines = [json.loads(l) for l in open(path)]
    return {"steps": np.array([r["step"] for r in lines]),
            "ftl":   np.array([r["first_target_loss"] for r in lines])}


def collect_m(state, keys):
    parts = []
    for k in keys:
        if k not in state: continue
        parts.append(state[k]["m"].float().numpy().flatten())
    return np.concatenate(parts) if parts else np.array([])


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    log = load_log(RUN / "log.jsonl")
    log_K = float(np.log(10))
    mask = log["ftl"] < 0.5 * log_K
    tau = int(log["steps"][np.argmax(mask)]) if mask.any() else None
    print(f"τ (50% of log K) = {tau}")

    opts = sorted((RUN / "checkpoints").glob("opt_step_*.pt"))
    steps = np.array([int(re.search(r"opt_step_(\d+)", p.name).group(1)) for p in opts])
    print(f"opt-state ckpts: {len(opts)}  step range: {steps.min()}..{steps.max()}  cadence: {int(np.diff(steps).min())}")

    print("loading m_t arrays...")
    states = []
    for i, p in enumerate(opts):
        if i % 20 == 0: print(f"  {i}/{len(opts)}")
        states.append(torch.load(p, map_location="cpu", weights_only=False))

    full_keys = list(states[0].keys())
    pipe_keys = [k for k in PIPE_KEYS if k in full_keys]
    ctrl_keys = [k for k in CTRL_KEYS if k in full_keys]

    # Stack m_t per ckpt for each subspace
    sets = {"pipeline": pipe_keys, "control": ctrl_keys, "full": full_keys}
    M = {name: np.stack([collect_m(s, keys) for s in states]) for name, keys in sets.items()}
    for n, m in M.items(): print(f"  {n}: m_t shape {m.shape}")

    # Compute moving-average m_t with multiple window sizes (in number of ckpts)
    # Cadence is 50 steps; window of 500 steps = 10 ckpts, 1000 steps = 20 ckpts
    cadence = 50
    windows_steps = [200, 500, 1000]  # smoothing windows in steps
    window_ckpts = [max(1, w // cadence) for w in windows_steps]
    print(f"\nsmoothing window sizes (in ckpts): {dict(zip(windows_steps, window_ckpts))}")

    def moving_avg(M, w):
        if w <= 1: return M
        ma = np.zeros_like(M)
        for i in range(M.shape[0]):
            lo = max(0, i - w//2); hi = min(M.shape[0], i + w//2 + 1)
            ma[i] = M[lo:hi].mean(0)
        return ma

    def cos_seq(MM):
        return np.array([cos(MM[t-1], MM[t]) for t in range(1, len(MM))])

    sim_steps = steps[1:]

    # baseline: w=1 (per-ckpt m_t)
    cos_per_ckpt = {n: cos_seq(M[n]) for n in M}

    # Smoothed
    cos_smooth = {}
    for w_steps, w_ckpts in zip(windows_steps, window_ckpts):
        for n, m in M.items():
            cos_smooth[(n, w_steps)] = cos_seq(moving_avg(m, w_ckpts))

    # Window medians
    def med(a, lo, hi):
        m = (sim_steps > lo) & (sim_steps < hi)
        return float(np.median(a[m])) if m.any() else float("nan")
    win_defs = [
        ("plateau (0.4τ..0.9τ)", 0.4*tau, 0.9*tau),
        ("τ-window (0.95τ..1.3τ)", 0.95*tau, 1.3*tau),
        ("1.5τ..3τ", 1.5*tau, 3*tau),
    ]

    print(f"\n{'='*100}")
    print("Moving-average m_t cosine similarity — slow-drift interpretation test")
    print(f"{'='*100}")
    for subspace in ["pipeline", "control", "full"]:
        print(f"\n--- {subspace} subspace ---")
        print(f"{'window':<26s}  {'per-ckpt (50s)':>15s}", end="")
        for w in windows_steps: print(f"  {'avg ' + str(w) + 's':>11s}", end="")
        print()
        for name, lo, hi in win_defs:
            print(f"{name:<26s}  {med(cos_per_ckpt[subspace], lo, hi):>+15.3f}", end="")
            for w in windows_steps:
                print(f"  {med(cos_smooth[(subspace, w)], lo, hi):>+11.3f}", end="")
            print()

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for ax, subspace in zip(axes, ["pipeline", "control", "full"]):
        # Use rolling-median smoothing for visualization
        def rmsmooth(x, w=5):
            pad = w // 2
            return np.array([np.median(x[max(0, i-pad):i+pad+1]) for i in range(len(x))])

        ax.plot(sim_steps, rmsmooth(cos_per_ckpt[subspace], 5),
                color="tab:gray", lw=1.0, alpha=0.7, label="per-ckpt m_t (50-step)")
        colors = ["tab:orange", "tab:red", "tab:purple"]
        for w_steps, c in zip(windows_steps, colors):
            ax.plot(sim_steps, rmsmooth(cos_smooth[(subspace, w_steps)], 5),
                    color=c, lw=1.6, label=f"avg m_t over {w_steps}-step window")
        ax.axvline(tau, ls=":", color="red", lw=1, label=f"τ = {tau}")
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.set_ylabel(f"cos sim (rolling med, w=5 ckpts)")
        ax.set_title(f"{subspace} subspace: per-checkpoint vs.\\ moving-average m_t cosine")
        ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("Slow-drift interpretation test: does smoothing m_t over wider windows recover coherence?")
    plt.savefig(OUT / "fig_H4_moving_avg.pdf"); plt.savefig(OUT / "fig_H4_moving_avg.png", dpi=150)

    # Verdict
    print(f"\n=== Slow-drift verdict ===")
    print(f"  At the τ-window, pipeline subspace:")
    pipe_50  = med(cos_per_ckpt["pipeline"], 0.95*tau, 1.3*tau)
    pipe_200 = med(cos_smooth[("pipeline", 200)], 0.95*tau, 1.3*tau)
    pipe_500 = med(cos_smooth[("pipeline", 500)], 0.95*tau, 1.3*tau)
    pipe_1k  = med(cos_smooth[("pipeline", 1000)], 0.95*tau, 1.3*tau)
    print(f"    per-ckpt (50-step) cos sim:    {pipe_50:+.3f}")
    print(f"    avg over 200-step window:      {pipe_200:+.3f}")
    print(f"    avg over 500-step window:      {pipe_500:+.3f}")
    print(f"    avg over 1000-step window:     {pipe_1k:+.3f}")

    if pipe_1k > 0.5 and pipe_50 < 0.1:
        v = "CONFIRMED: smoothing m_t over a wide window recovers coherence at τ. Slow-drift interpretation supported. The mean gradient direction is approximately stable through τ; per-50-step fluctuations dominate the unsmoothed cosine."
    elif pipe_1k > pipe_50 + 0.3:
        v = f"PARTIAL: smoothing improves cos sim from {pipe_50:+.3f} to {pipe_1k:+.3f} (gain {pipe_1k - pipe_50:+.3f}). Some slow drift exists but it does not dominate."
    elif abs(pipe_1k - pipe_50) < 0.1:
        v = "REJECTED: smoothing does not recover coherence. The gradient flow really is reorganizing rapidly through the τ-window, not just fluctuating around a stable mean."
    else:
        v = f"WEAK: smoothing gain at τ-window is {pipe_1k - pipe_50:+.3f}, between rejection and partial."
    print(f"\n  Verdict: {v}")

    np.savez(OUT / "h4_moving_avg_results.npz",
             steps=sim_steps,
             cos_per_ckpt={n: cos_per_ckpt[n] for n in cos_per_ckpt},
             cos_smooth={(f"{n}_{w}"): cos_smooth[(n, w)] for n, w in cos_smooth},
             tau=tau)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
