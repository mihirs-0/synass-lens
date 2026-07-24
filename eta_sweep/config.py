"""
η-Sweep configuration.

Central place for the sweep grid, hyperparameters, and paths.  Imported by
run_single.py, launch_sweep.py, and the analysis scripts so that all
components see an identical grid.

NOTE on hyperparameter choice
-----------------------------
The existing three data points for c (9.91, 4.29, 110.2 at η = 3e-4, 1e-3,
3e-3) were produced from the landauer_dense baseline:

    scheduler:    constant
    warmup_steps: 0
    grad_clip:    None

Q = Σₜ η · ‖∇L(θₜ)‖² is defined with a constant step-size η; any cosine
decay contaminates that integral.  To keep the new six-η sweep directly
comparable to the existing three points we match the landauer_dense
baseline exactly.  Cosine + warmup are parametrised here so they can be
flipped on later, but the default is the comparable regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

ETA_VALUES: List[float] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
K_VALUES: List[int] = [10, 15, 20, 25, 36]
SEEDS: List[int] = [0]  # start single-seed; extend to [0,1,2] for 3-seed replication


def sweep_cells(
    etas: Optional[List[float]] = None,
    ks: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Tuple[float, int, int]]:
    """Materialise the (η, K, seed) cells to run.

    Default order:  middle η values first (where we already know the
    phenomenology exists), then outward to the edges.  This catches
    systemic bugs early.
    """
    etas = ETA_VALUES if etas is None else etas
    ks = K_VALUES if ks is None else ks
    seeds = SEEDS if seeds is None else seeds

    preferred_eta_order = [1e-3, 3e-3, 3e-4, 1e-2, 1e-4, 3e-2]
    etas_sorted = sorted(etas, key=lambda e: preferred_eta_order.index(e)
                         if e in preferred_eta_order else 999)

    cells = []
    for eta in etas_sorted:
        for k in ks:
            for seed in seeds:
                cells.append((eta, k, seed))
    return cells


# ---------------------------------------------------------------------------
# Per-cell hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class CellConfig:
    """Hyperparameters for a single (η, K, seed) run."""

    eta: float
    k: int
    seed: int

    # Data
    n_unique_b: int = 1000
    b_length: int = 6
    a_length: int = 4
    z_length: int = 2
    vocab_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789"
    enforce_unique_a_first_char_per_b: bool = True
    disambiguation_prefix_length: int = 1
    split_by_base: bool = True
    probe_fraction: float = 0.0
    task: str = "bz_to_a"

    # Model
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_head: int = 32
    d_mlp: int = 512
    act_fn: str = "gelu"

    # Optimiser / scheduler — match landauer_dense baseline for Q comparability
    batch_size: int = 128
    weight_decay: float = 0.01
    scheduler: str = "constant"
    warmup_steps: int = 0
    grad_clip: Optional[float] = None

    # Optimiser choice.  "adamw" (default) reproduces the rest of the sweep;
    # "sgd" is for the SGD control experiment.  momentum applies to "sgd" only.
    optimizer: str = "adamw"
    momentum: float = 0.0

    # Logging cadence
    # First 5k: log every 100 steps.  After: log every 500 steps.
    checkpoint_every_early: int = 100
    checkpoint_every_late: int = 500
    switch_step: int = 5000

    # Candidate eval
    candidate_eval_n: int = 32

    # Held-out gradient batch
    held_out_grad_batch_size: int = 256

    # Early stopping + stuck / diverged budgets
    min_steps: int = 2000

    # Scientific thresholds used by the live detector.  All match the
    # analysis thresholds used post-hoc.
    plateau_tol_frac: float = 0.05      # within 5% of log K ⇒ plateau
    delta_z_plateau: float = 0.05       # Δz < 0.05 nats ⇒ plateau
    transition_frac: float = 0.3        # cand_loss < 0.3 · log K ⇒ transitioned
    transition_patience: int = 500      # consecutive steps required for transition
    stuck_patience: int = 2000          # consecutive plateau steps ⇒ stuck
    diverge_ratio: float = 1.5          # cand_loss > 1.5 · log K ⇒ diverged
    diverge_patience: int = 500

    # Checkpoint save (model weights) — cheap, infrequent
    weight_checkpoint_every: int = 2000

    @property
    def max_steps(self) -> int:
        """η-dependent step budget."""
        if self.eta >= 1e-2:
            return 10_000
        if self.eta >= 1e-3:
            return 30_000
        if self.eta >= 3e-4:
            return 50_000
        return 100_000  # η = 1e-4

    @property
    def run_name(self) -> str:
        # Matches the output folder under results/
        return f"eta_{self.eta:g}_K_{self.k}_seed_{self.seed}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ETA_SWEEP_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = ETA_SWEEP_ROOT.parent
RESULTS_DIR: Path = ETA_SWEEP_ROOT / "results"
AGGREGATED_PARQUET: Path = RESULTS_DIR / "aggregated_runs.parquet"
PER_ETA_FITS_JSON: Path = RESULTS_DIR / "per_eta_fits.json"
META_FIT_JSON: Path = RESULTS_DIR / "meta_fit.json"


def run_dir(eta: float, k: int, seed: int) -> Path:
    return RESULTS_DIR / f"eta_{eta:g}_K_{k}_seed_{seed}"
