"""H4 analysis: cos sim of m_t vs cos sim of m_t/√v_t across consecutive
checkpoints.  Run after run_full_optstate.py finishes.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RUN  = REPO / "eta_sweep" / "results" / "full_optstate" / "eta_0.001_K_10_seed_0"
OUT  = REPO / "analysis_outputs" / "v3_review_addons"
OUT.mkdir(parents=True, exist_ok=True)

# Module-group keys (same as Analysis 4)
PIPE_KEYS = ["blocks.0.attn.W_Q","blocks.0.attn.W_K","blocks.0.attn.W_V","blocks.0.attn.W_O",
             "blocks.0.mlp.W_in","blocks.0.mlp.W_out",
             "blocks.2.mlp.W_in","blocks.2.mlp.W_out",
             "blocks.3.attn.W_Q","blocks.3.attn.W_K","blocks.3.attn.W_V","blocks.3.attn.W_O"]
CTRL_KEYS = ["blocks.1.attn.W_Q","blocks.1.attn.W_K","blocks.1.attn.W_V","blocks.1.attn.W_O",
             "blocks.2.attn.W_Q","blocks.2.attn.W_K","blocks.2.attn.W_V","blocks.2.attn.W_O",
             "blocks.3.mlp.W_in","blocks.3.mlp.W_out","embed.W_E"]
EPS = 1e-8


def load_log(path):
    lines = [json.loads(l) for l in open(path)]
    return {"steps": np.array([r["step"] for r in lines]),
            "ftl":   np.array([r["first_target_loss"] for r in lines]),
            "loss":  np.array([r["train_loss"] for r in lines])}


def collect_vec(state, keys, kind):
    """kind in {'m','v','m_over_sqrt_v','update'} — flatten + concat over the given keys."""
    parts = []
    for k in keys:
        if k not in state: continue
        m = state[k]["m"].float().numpy().flatten()
        v = state[k]["v"].float().numpy().flatten()
        if kind == "m":              parts.append(m)
        elif kind == "v":            parts.append(v)
        elif kind == "m_over_sqrt_v": parts.append(m / (np.sqrt(v) + EPS))
    return np.concatenate(parts) if parts else np.array([])


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    log = load_log(RUN / "log.jsonl")
    log_K = float(np.log(10))
    mask = log["ftl"] < 0.5 * log_K
    tau = int(log["steps"][np.argmax(mask)]) if mask.any() else None
    print(f"τ (50% of log K) = {tau}")
    if tau is None:
        print("ERROR: τ not found; check that the run reached transition.")
        return

    opts = sorted((RUN / "checkpoints").glob("opt_step_*.pt"))
    print(f"opt-state checkpoints: {len(opts)}")
    steps = np.array([int(re.search(r"opt_step_(\d+)", p.name).group(1)) for p in opts])

    # Load opt states (this is the heavy bit — a few GB total)
    print("loading opt states...")
    states = []
    for i, p in enumerate(opts):
        if i % 20 == 0: print(f"  {i}/{len(opts)}")
        states.append(torch.load(p, map_location="cpu", weights_only=False))

    full_keys = list(states[0].keys())
    pipe_keys = [k for k in PIPE_KEYS if k in full_keys]
    ctrl_keys = [k for k in CTRL_KEYS if k in full_keys]
    print(f"pipe keys: {len(pipe_keys)}/{len(PIPE_KEYS)}; ctrl keys: {len(ctrl_keys)}/{len(CTRL_KEYS)}")

    # Compute consecutive-checkpoint cos sims for m, m/√v, on pipeline + control + full
    def cos_seq(keyset, kind):
        vs = [collect_vec(s, keyset, kind) for s in states]
        sims = []
        for t in range(1, len(vs)):
            sims.append(cos(vs[t-1], vs[t]))
        return np.array(sims)

    print("\ncomputing cos sims...")
    sets = {"pipeline": pipe_keys, "control": ctrl_keys, "full": full_keys}
    cos_m = {name: cos_seq(keys, "m") for name, keys in sets.items()}
    cos_msv = {name: cos_seq(keys, "m_over_sqrt_v") for name, keys in sets.items()}
    sim_steps = steps[1:]

    # Window medians
    def med(a, lo, hi):
        m = (sim_steps > lo) & (sim_steps < hi)
        return float(np.median(a[m])) if m.any() else float("nan")
    windows = [
        ("plateau (0.4τ..0.9τ)", 0.4*tau, 0.9*tau),
        ("τ-window (0.95τ..1.3τ)", 0.95*tau, 1.3*tau),
        ("1.5τ..3τ", 1.5*tau, 3*tau),
    ]

    print(f"\n{'='*88}")
    print(f"COSINE SIMILARITY: m_t vs.\\ m_t/√v_t (consecutive opt-state checkpoints)")
    print(f"{'='*88}")
    print(f"\n{'window':<26s}  {'m (pipe)':>10s} {'m/√v (pipe)':>12s} | {'m (ctrl)':>10s} {'m/√v (ctrl)':>12s} | {'m (full)':>10s} {'m/√v (full)':>12s}")
    print("-" * 100)
    for name, lo, hi in windows:
        row = [med(cos_m["pipeline"], lo, hi), med(cos_msv["pipeline"], lo, hi),
               med(cos_m["control"], lo, hi),  med(cos_msv["control"], lo, hi),
               med(cos_m["full"], lo, hi),     med(cos_msv["full"], lo, hi)]
        print(f"{name:<26s}  " + " ".join(f"{v:>+10.3f} {' '*1}" for v in row[::2]))
        print(f"{'   m/√v→':<26s}  " + " ".join(f"{'         '+' '} {v:>+10.3f} " for v in row[1::2]))

    # Cleaner table format
    print("\n--- clean table ---")
    print(f"{'window':<26s}  {'pipe m':>8s}  {'pipe m/√v':>10s}  {'ctrl m':>8s}  {'ctrl m/√v':>10s}  {'full m':>8s}  {'full m/√v':>10s}")
    for name, lo, hi in windows:
        print(f"{name:<26s}  "
              f"{med(cos_m['pipeline'], lo, hi):>+8.3f}  "
              f"{med(cos_msv['pipeline'], lo, hi):>+10.3f}  "
              f"{med(cos_m['control'],  lo, hi):>+8.3f}  "
              f"{med(cos_msv['control'],  lo, hi):>+10.3f}  "
              f"{med(cos_m['full'],     lo, hi):>+8.3f}  "
              f"{med(cos_msv['full'],     lo, hi):>+10.3f}")

    # Plot
    def smooth(x, w=5):
        pad = w // 2
        return np.array([np.median(x[max(0,i-pad):i+pad+1]) for i in range(len(x))])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True, constrained_layout=True)
    for ax, kind, label, ydict in [(axes[0], "m", "first-moment estimate $m_t$", cos_m),
                                    (axes[1], "m_over_sqrt_v", "preconditioned update $m_t / \\sqrt{v_t}$", cos_msv)]:
        ax.plot(sim_steps, smooth(ydict["pipeline"], 5), color="tab:red", lw=2.0, label="pipeline subspace")
        ax.plot(sim_steps, smooth(ydict["control"],  5), color="tab:blue", lw=1.6, label="control subspace", alpha=0.8)
        ax.plot(sim_steps, smooth(ydict["full"],     5), color="black", ls="--", lw=1.4, label="full network", alpha=0.7)
        ax.axvline(tau, ls=":", color="red", lw=1, label=f"τ = {tau}")
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.set_ylabel(f"cos sim of {label}")
        ax.set_title(f"H4: cosine similarity of {label} across consecutive ckpts")
        ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    axes[-1].set_xlabel("step")
    fig.suptitle("H4 test: does AdamW's preconditioner cause the τ-window decorrelation?")
    plt.savefig(OUT / "fig_H4_m_vs_msv_cossim.pdf"); plt.savefig(OUT / "fig_H4_m_vs_msv_cossim.png", dpi=150)

    # Verdict
    tw_m_pipe = med(cos_m["pipeline"],   0.95*tau, 1.3*tau)
    tw_msv_pipe = med(cos_msv["pipeline"], 0.95*tau, 1.3*tau)
    tw_m_full = med(cos_m["full"], 0.95*tau, 1.3*tau)
    tw_msv_full = med(cos_msv["full"], 0.95*tau, 1.3*tau)
    print(f"\n=== H4 verdict ===")
    print(f"  At τ-window:")
    print(f"    pipeline cos(m):           {tw_m_pipe:+.3f}")
    print(f"    pipeline cos(m/√v):        {tw_msv_pipe:+.3f}")
    print(f"    full     cos(m):           {tw_m_full:+.3f}")
    print(f"    full     cos(m/√v):        {tw_msv_full:+.3f}")
    if tw_m_pipe > 0.5 and tw_msv_pipe < 0.2:
        v = "CONFIRMED: m_t is coherent at τ; the preconditioner is the source of decorrelation."
    elif tw_m_pipe > tw_msv_pipe + 0.3:
        v = "PARTIAL: m_t is more coherent than m/√v but neither is strongly coherent (>0.5)."
    elif abs(tw_m_pipe - tw_msv_pipe) < 0.1:
        v = "REJECTED: m_t and m/√v have indistinguishable τ-window cos sim. The preconditioner is NOT the source of decorrelation; the EMA-smoothed gradient itself is decorrelating."
    else:
        v = f"WEAK: m vs m/√v gap at τ-window is {tw_m_pipe-tw_msv_pipe:+.3f}."
    print(f"\n  Verdict: {v}")

    np.savez(OUT / "h4_results.npz", steps=sim_steps, cos_m=cos_m, cos_msv=cos_msv, tau=tau)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
