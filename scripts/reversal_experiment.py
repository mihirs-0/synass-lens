#!/usr/bin/env python
"""
Suite 1: The Reversal Curse in the Wind Tunnel.

Tests directional asymmetry by training on the same data in 4 configurations:
  Task F:  A -> B  (forward, trivial — should converge fast)
  Task Bz: (B,z) -> A  (backward with z, the standard disambiguation task)
  Task B:  B -> A  (backward without z, impossible below log K)
  Transfer: Train F, then switch to Bz (sequential, not parallelizable)

All tasks at each K use the SAME surjective function f.
Single seed. Aggressive early stopping. Parallel GPU execution.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    make_config, run_exists, save_mapping, load_mapping,
    run_single_experiment, run_parallel, generate_mappings,
)
from omegaconf import OmegaConf

K_VALUES = [5, 10, 20, 36]
SEED = 42
LR = 1e-3
OUTPUT_DIR = "outputs"
MAX_WORKERS = 6


def run_suite1():
    """Run all Suite 1 experiments with parallel execution."""
    print("=" * 60)
    print("SUITE 1: THE REVERSAL CURSE IN THE WIND TUNNEL")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Generate/load all mappings first
    # ----------------------------------------------------------------
    mappings = {}
    for K in K_VALUES:
        mapping_path = Path(OUTPUT_DIR) / f"reversal_mapping_K{K}.json"
        if not mapping_path.exists():
            print(f"  Generating mapping for K={K}...")
            mapping = generate_mappings(
                n_unique_b=1000, k=K, b_length=6, a_length=4,
                z_length=2, vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
                seed=SEED, task="bz_to_a",
            )
            save_mapping(mapping, str(mapping_path))
        mappings[K] = str(mapping_path)

    # ----------------------------------------------------------------
    # Step 2: Collect all independent jobs (F, Bz, B across all K)
    # ----------------------------------------------------------------
    jobs = []

    for K in K_VALUES:
        mp = mappings[K]

        # Task F: A→B — converges fast, 20K is plenty with early stopping
        name_f = f"reversal_F_K{K}_s{SEED}"
        if not run_exists(name_f, min_steps=1, output_dir=OUTPUT_DIR):
            cfg = make_config(
                experiment_name=name_f, task="az_to_b", k=K, seed=SEED,
                lr=LR, max_steps=20000, eval_every=50,
                checkpoint_every=20000, early_stop_frac=0.05,
            )
            jobs.append({"cfg_dict": OmegaConf.to_container(cfg),
                         "mapping_path": mp, "output_dir": OUTPUT_DIR,
                         "name": name_f})
        else:
            print(f"  [SKIP] {name_f}")

        # Task Bz: (B,z)→A — the main disambiguation task
        name_bz = f"reversal_Bz_K{K}_s{SEED}"
        bz_steps = 200000 if K >= 20 else 100000
        if not run_exists(name_bz, min_steps=1, output_dir=OUTPUT_DIR):
            cfg = make_config(
                experiment_name=name_bz, task="bz_to_a", k=K, seed=SEED,
                lr=LR, max_steps=bz_steps, eval_every=200,
                checkpoint_every=max(bz_steps // 10, 5000),
                early_stop_frac=0.05,
            )
            jobs.append({"cfg_dict": OmegaConf.to_container(cfg),
                         "mapping_path": mp, "output_dir": OUTPUT_DIR,
                         "name": name_bz})
        else:
            print(f"  [SKIP] {name_bz}")

        # Task B: B→A no z — plateaus at log(K), 20K steps is enough to confirm
        name_b = f"reversal_B_K{K}_s{SEED}"
        if not run_exists(name_b, min_steps=1, output_dir=OUTPUT_DIR):
            cfg = make_config(
                experiment_name=name_b, task="b_to_a", k=K, seed=SEED,
                lr=LR, max_steps=20000, eval_every=50,
                checkpoint_every=20000,
            )
            jobs.append({"cfg_dict": OmegaConf.to_container(cfg),
                         "mapping_path": mp, "output_dir": OUTPUT_DIR,
                         "name": name_b})
        else:
            print(f"  [SKIP] {name_b}")

    # ----------------------------------------------------------------
    # Step 3: Run all independent jobs in parallel
    # ----------------------------------------------------------------
    print(f"\n  Independent tasks: {len(jobs)} jobs")
    run_parallel(jobs, max_workers=MAX_WORKERS, label="Suite1-parallel")

    # ----------------------------------------------------------------
    # Step 4: Transfer tasks (sequential: Phase1 then Phase2)
    # These cannot be parallelized because Phase2 needs Phase1's model.
    # But we run all 4 K-values' transfers in parallel with each other.
    # ----------------------------------------------------------------
    transfer_jobs = []
    for K in K_VALUES:
        name_tr = f"reversal_Transfer_K{K}_s{SEED}"
        if run_exists(name_tr, min_steps=1, output_dir=OUTPUT_DIR):
            print(f"  [SKIP] {name_tr}")
            continue

        print(f"  [RUN] Transfer K={K}: Phase1 (A→B, 20K) → Phase2 (Bz→A)")
        mp = mappings[K]
        mapping = load_mapping(mp)

        # Phase 1: Forward
        name_p1 = f"reversal_Transfer_P1_K{K}_s{SEED}"
        cfg_p1 = make_config(
            experiment_name=name_p1, task="az_to_b", k=K, seed=SEED,
            lr=LR, max_steps=20000, eval_every=100,
            checkpoint_every=20000, early_stop_frac=0.05,
        )
        try:
            model_p1, _, _, _ = run_single_experiment(
                cfg_p1, mapping_data=mapping, output_dir=OUTPUT_DIR,
            )
        except Exception as e:
            print(f"    Phase1 FAILED K={K}: {e}")
            continue

        # Phase 2: Backward with z, reusing model
        tr_steps = 150000 if K >= 20 else 80000
        cfg_p2 = make_config(
            experiment_name=name_tr, task="bz_to_a", k=K, seed=SEED,
            lr=LR, max_steps=tr_steps, eval_every=200,
            checkpoint_every=max(tr_steps // 10, 5000),
            early_stop_frac=0.05,
        )
        try:
            run_single_experiment(
                cfg_p2, mapping_data=mapping, output_dir=OUTPUT_DIR,
                model=model_p1,
            )
            print(f"    Transfer K={K} done", flush=True)
        except Exception as e:
            print(f"    Phase2 FAILED K={K}: {e}")

    print(f"\n{'='*60}")
    print("SUITE 1 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_suite1()
