#!/usr/bin/env python
"""Generate a clean training curve figure for the HiLD paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load K=10 training history
with open("outputs/landauer_dense_k10/training_history.json") as f:
    hist = json.load(f)

steps = np.array(hist["steps"])
loss = np.array(hist["first_target_loss"])

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))

ax.plot(steps, loss, color="#1f77b4", linewidth=1.2, alpha=0.8)

# Annotate plateau
log_k = np.log(10)
ax.axhline(log_k, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(12000, log_k + 0.15, r"$\log K = 2.30$", fontsize=9, color="gray", ha="center")

# Annotate tau
tau = 1850
ax.axvline(tau, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.7)
ax.text(tau + 500, 1.8, r"$\tau$", fontsize=11, color="#d62728")

# Annotate phases
ax.annotate("Marginal phase\n" + r"$P(A|B)$", xy=(8000, 2.2), fontsize=8,
            ha="center", color="#555555", fontstyle="italic")
ax.annotate("Conditional phase\n" + r"$P(A|B,z)$", xy=(8000, 0.3), fontsize=8,
            ha="center", color="#555555", fontstyle="italic")

ax.set_xlabel("Training step", fontsize=10)
ax.set_ylabel("Loss (nats)", fontsize=10)
ax.set_xlim(0, 15000)
ax.set_ylim(-0.1, 3.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig("hild_paper/fig_training_curve.png", dpi=300, bbox_inches="tight")
print("Saved hild_paper/fig_training_curve.png")
