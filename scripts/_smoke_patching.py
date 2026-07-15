"""Smoke test: load seed 1 at step 4000 and run patching only.
Goal: estimate wall-clock per phase before launching full driver.
"""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eta_sweep.analysis.mech_interp_mbc import (
    CellSpec, load_cell, make_model, activation_patching, _select_device,
)

device = _select_device()
print(f"device: {device}")

cs = CellSpec("CellA_seed1_smoke", 0.001, 20, 1, "smoke")
t0 = time.time()
cc, tok, mapping_data, train_ds, rows = load_cell(cs, device)
print(f"load_cell: {time.time()-t0:.1f}s")

t0 = time.time()
model = make_model(cc, tok, 4000, device)
print(f"make_model(step=4000): {time.time()-t0:.1f}s")

t0 = time.time()
result = activation_patching(model, tok, mapping_data, device,
                             n_pairs=32, seed=0)
print(f"activation_patching: {time.time()-t0:.1f}s")
print(f"  recovery_per_layer: {result.get('recovery_per_layer')}")
print(f"  n_pairs: {result.get('n_pairs')}")
print(f"  p_clean: {result.get('p_target_clean_mean'):.3f}")
print(f"  p_shuf:  {result.get('p_target_shuf_mean'):.3f}")
