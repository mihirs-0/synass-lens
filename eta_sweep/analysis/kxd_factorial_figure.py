#!/usr/bin/env python
"""Figure for K×D factorial appendix: per-K exponents with bootstrap CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "paper_figures"

ANALYSIS = json.loads(
    (REPO_ROOT / "eta_sweep" / "results" / "kxd_factorial_analysis.json").read_text()
)

per_K = ANALYSIS["per_K_fits"]
Ks = sorted([int(k) for k in per_K.keys()])
slopes = [per_K[str(K)]["slope"] for K in Ks]
ci_lo = [per_K[str(K)]["ci_lo"] for K in Ks]
ci_hi = [per_K[str(K)]["ci_hi"] for K in Ks]
err_lo = [s - lo for s, lo in zip(slopes, ci_lo)]
err_hi = [hi - s for s, hi in zip(slopes, ci_hi)]

global_fit = ANALYSIS["global_fit"]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.errorbar(Ks, slopes, yerr=[err_lo, err_hi], fmt="o", color="C0",
            capsize=3, lw=1.4, markersize=6, label="per-K fit ± 95% CI")
ax.axhline(global_fit["slope"], color="C3", linestyle="--", lw=1.2,
           label=f"global pooled δ = {global_fit['slope']:.3f}")
ax.fill_between([min(Ks)-1, max(Ks)+2],
                global_fit["ci_lo"], global_fit["ci_hi"],
                color="C3", alpha=0.12, label=f"global 95% CI [{global_fit['ci_lo']:.3f}, {global_fit['ci_hi']:.3f}]")
ax.axhline(1.0, color="grey", linestyle=":", lw=0.9, alpha=0.7, label="linear (δ=1.0)")
ax.set_xlim(min(Ks)-1, max(Ks)+2)
ax.set_xlabel("K (fiber size)")
ax.set_ylabel(r"$\tau \propto D^\delta$ exponent (30% threshold)")
ax.set_title("Per-K exponents from the K×D factorial (4 D values per K, 3 seeds)")
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
ax.grid(True, alpha=0.3, linewidth=0.4)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_kxd_factorial.pdf", dpi=200)
fig.savefig(OUT_DIR / "fig_kxd_factorial.png", dpi=160)
print(f"  → {OUT_DIR / 'fig_kxd_factorial.pdf'}")
print(f"  → {OUT_DIR / 'fig_kxd_factorial.png'}")
