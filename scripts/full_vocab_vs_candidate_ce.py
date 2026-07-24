"""Compare full-vocab CE vs candidate-restricted CE at canonical K=10
checkpoints. Verifies the analytic-vs-modeling decomposition the paper claims.

Output: prints a table for direct inclusion in the paper.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LANDAUER = REPO_ROOT / "outputs" / "landauer_dense_k10"
log_k = math.log(10)

th = json.load(open(LANDAUER / "training_history.json"))
steps = th["steps"]
train_loss = th["train_loss"]      # full-vocab CE per token (mean over A seq)
cand_loss = th["candidate_loss"]   # full-A-sequence candidate CE
ftl = th["first_target_loss"]       # first-target-token full-vocab CE


def find_step_idx(target):
    return min(range(len(steps)), key=lambda i: abs(steps[i] - target))


# Sample at meaningful phases
phases = [
    ("init", 100),
    ("plateau-early", 500),
    ("plateau-mid", 1500),
    ("near-transition", 2200),
    ("transition", 2500),
    ("post", 5000),
    ("late", 25000),
]

print(f"K=10 (log K = {log_k:.4f})")
print(f"{'phase':<18} {'step':>6} {'train_loss':>10} {'first_tok':>10} "
      f"{'cand_full':>10} {'cand/logK':>10} {'first_tok/logK':>14}")
print("-" * 90)
for name, t in phases:
    i = find_step_idx(t)
    s = steps[i]
    tl = train_loss[i]
    cl = cand_loss[i]
    f = ftl[i]
    print(f"{name:<18} {s:>6} {tl:>10.3f} {f:>10.3f} {cl:>10.3f} "
          f"{cl/log_k:>10.3f} {f/log_k:>14.3f}")

# Aggregate plateau range (steps 500-1500)
plateau_idx = [i for i, s in enumerate(steps) if 500 <= s <= 1500]
print()
print(f"Plateau (steps 500-1500, n={len(plateau_idx)}):")
print(f"  candidate CE / log K     : {np.mean([cand_loss[i]/log_k for i in plateau_idx]):.3f} ± {np.std([cand_loss[i]/log_k for i in plateau_idx]):.3f}")
print(f"  first-target CE / log K  : {np.mean([ftl[i]/log_k for i in plateau_idx]):.3f} ± {np.std([ftl[i]/log_k for i in plateau_idx]):.3f}")
print(f"  full-vocab train CE / log K: {np.mean([train_loss[i]/log_k for i in plateau_idx]):.3f} ± {np.std([train_loss[i]/log_k for i in plateau_idx]):.3f}")
