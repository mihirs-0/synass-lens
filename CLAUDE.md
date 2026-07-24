# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research framework investigating **Late Disambiguation Lag** — why transformers exhibit learning lag when disambiguating between multiple targets using a selector variable. Uses TransformerLens to implement mechanistic probes that reveal when and how models learn to exploit disambiguating information.

**Core task:** Train transformers on synthetic mappings `(B, z) → A` where B is ambiguous (maps to K different A values) and z is a selector that makes the mapping one-to-one. The "lag" is the delay before the model learns to use z.

## Commands

All commands run from the `synass-lens/` project root.

### Install
```bash
pip install -r requirements.txt
```

### Train a model
```bash
# Use a predefined experiment config
python scripts/train.py --config-name=k10_n1000

# Override parameters inline
python scripts/train.py experiment.name=my_exp data.k=50 data.n_unique_b=500

# Different task direction
python scripts/train.py --config-path ../configs --config-name base experiment.name=az_to_b_k10 data.task=az_to_b data.k=10
```

Hydra resolves configs from `configs/experiments/` (81 experiment YAML files that inherit from `configs/base.yaml`).

### Run analysis (probes + figures)
```bash
python scripts/analyze.py --experiment k10_n1000
python scripts/analyze.py --experiment k10_n1000 --probes attention_to_z logit_lens
python scripts/analyze.py --experiment k10_n1000 --figures-only
```

### Specialized analysis scripts
```bash
python scripts/analyze_temperature_sweep.py --output-dir outputs
python scripts/analyze_landauer_dense.py --experiment <name>
python scripts/analyze_phase_boundary.py --output-dir outputs
python scripts/analyze_k_sweep.py --output-dir outputs
```

### No formal test suite or linter configured
Validation is done through analysis scripts and probe diagnostics.

## Architecture

### Data flow
```
Hydra Config (YAML) → Tokenizer → Dataset (MappingData) → DataLoader → HookedTransformer → Training Loop → Checkpoints → Probes → Visualization
```

Training and analysis are decoupled: `train.py` saves checkpoints, `analyze.py` loads them post-hoc to run probes and generate figures.

### Source layout (`src/`)

- **`data/`** — Synthetic data generation (`dataset.py`) and character-level tokenizer (`tokenizer.py`). Generates B strings (6-char), z selectors (2-char), and A targets (4-char). Factory functions: `create_tokenizer_from_config()`, `create_datasets_from_config()`.
- **`model/`** — Model factory (`hooked_transformer.py`). Supports three architectures dispatched via `cfg.model.architecture`: HookedTransformer (default), GatedMLP, RNN. Factory: `create_model_from_config()`.
- **`training/`** — Training loop (`trainer.py`) with AdamW + cosine warmup, z-shuffle diagnostic, and checkpointing (`checkpoint.py`). Supports optional callbacks (`on_after_backward`, `on_after_step`, `on_checkpoint`).
- **`probes/`** — Mechanistic analysis probes. All inherit from `BaseProbe` and are registered in `PROBE_REGISTRY` (registry pattern). Key probes: `AttentionToZProbe`, `LogitLensProbe`, `CausalPatchingProbe`, `RandomZEvalProbe`, `GradientSNRProbe`, `IsotropyProbe`, `MultiheadDecompositionProbe`.
- **`analysis/`** — Probe orchestration (`run_probes.py`) and figure generation (`visualize.py`). Outputs: `dashboard.png`, `attention_to_z.png`, `logit_lens.png`, `z_dependence.png`.

### Key conventions

- **Hydra config inheritance:** All experiments override `configs/base.yaml`. Experiment configs live in `configs/experiments/`.
- **sys.path hack:** Scripts add the project root to `sys.path` to import from `src/`.
- **Probes are first-token-only by default:** Theoretical benchmarks (log-K floor) apply to the first target token, so probes use `first_token_only=true` to avoid averaging away the ambiguity signal.
- **Z-reshuffle diagnostic:** Training logs a z-shuffle loss (swaps z across batch, keeps B and A fixed) to detect when the model starts using z. Read-only, no gradients.
- **n_pairs_effective normalization:** Number of unique B strings is held constant across K values (e.g., 1000 B's whether K=1 or K=100), so total examples scale as K × n_unique_b.
- **Device auto-selection:** cuda > mps > cpu.
- **Output structure:** `outputs/{experiment_name}/` contains `checkpoints/`, `probe_results/`, `figures/`, and `config.yaml`.
