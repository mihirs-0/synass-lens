"""Shuffled-label probe baseline: rerun the linear z-probe on the same
activation tensor but with shuffled z labels. Reports the chance level for
the linear probe in the same training/test regime.

Output: eta_sweep/results/shuffled_label_probe_baseline.json.
"""
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from eta_sweep.config import RESULTS_DIR  # noqa: E402
from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    CellSpec, load_cell, make_model, _select_device,
    build_eval_batch, collect_residuals, get_probe_labels, linear_probes,
)


def main():
    device = _select_device()
    print(f"device: {device}")

    # Use canonical Cell A seed 0 at plateau and transition
    cs = CellSpec("CellA_seed0", 0.001, 20, 0)
    cc, tokenizer, mapping_data, train_ds, _ = load_cell(cs, device)
    eval_batch, eval_idx = build_eval_batch(train_ds, 192, seed=0,
                                             device=device)
    labels_clean = get_probe_labels(eval_batch, train_ds, mapping_data,
                                     eval_idx)["z_id"][:192]

    rng = np.random.default_rng(42)
    labels_shuffled = labels_clean.copy()
    rng.shuffle(labels_shuffled)

    out = {"K": cs.k, "n_classes": int(np.max(labels_clean)) + 1,
           "chance": 1.0 / cs.k, "checkpoints": {}}

    for step in [2000, 4000]:
        model = make_model(cc, tokenizer, step, device)
        feats = collect_residuals(model, eval_batch, device)
        arr = feats["resid_pre_target_start"]   # (B, n_layers+1, d)
        out["checkpoints"][step] = {
            "clean_label_test_acc": [],
            "shuffled_label_test_acc": [],
        }
        for L in range(arr.shape[1]):
            X = arr[:, L, :]
            r_clean = linear_probes(X, labels_clean, n_train=128, n_test=48,
                                    seed=0, C=0.1)
            r_shuf = linear_probes(X, labels_shuffled,
                                    n_train=128, n_test=48, seed=0, C=0.1)
            out["checkpoints"][step]["clean_label_test_acc"].append(
                r_clean["test_acc"])
            out["checkpoints"][step]["shuffled_label_test_acc"].append(
                r_shuf["test_acc"])
        del model
        import torch
        if device == "mps":
            torch.mps.empty_cache()
        print(f"step={step}:")
        print(f"  clean   : {out['checkpoints'][step]['clean_label_test_acc']}")
        print(f"  shuffled: {out['checkpoints'][step]['shuffled_label_test_acc']}")

    out_path = RESULTS_DIR / "shuffled_label_probe_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
