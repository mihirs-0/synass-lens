# Late Disambiguation Lag: Mechanistic Analysis

A TransformerLens-based framework for investigating why transformers exhibit learning lag when using disambiguating information.

## The Phenomenon

When training transformers on mappings (B, z) → A where:
- B alone is ambiguous (maps to K different A's)
- z is a selector that makes the mapping one-to-one

We observe a **lag** before the model learns to use z, even though:
- The information is present from the start
- The solution is information-theoretically trivial
- K=1 baseline learns quickly

## Key Question

**Why does SGD take so long to exploit a perfectly disambiguating variable, even when the solution is information-theoretically trivial?**

## Mechanistic Probes

This framework implements three probes to answer this:

### 1. Attention to z (`attention_to_z`)
- Measures how much attention flows from A positions to z positions
- **Expected finding**: Attention to z starts low/uniform, then increases sharply at a "transition point"

### 2. Logit Lens (`logit_lens`)
- Projects intermediate layer representations to vocabulary space
- Shows when each layer "knows" the correct answer
- **Expected finding**: Correct probability increases at deeper layers first, propagates to shallower layers

### 3. Causal Patching (`causal_patching`)
- The **smoking gun** test for z-dependence
- Corrupts z activations and measures effect on output
- **Expected finding**: Early in training, corrupting z has no effect; late in training, it changes the output

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a model
```bash
# K=10 experiment (main experiment)
python scripts/train.py --config-name=k10_n1000

# Az->B baseline (task direction swap)
python scripts/train.py --config-path ../configs --config-name base experiment.name=az_to_b_k10 data.task=az_to_b data.k=10

# Or with custom parameters
python scripts/train.py experiment.name=my_exp data.k=50 data.n_unique_b=500
```

### 2. Run analysis
```bash
python scripts/analyze.py --experiment k10_n1000
```

### 3. View results
Check `outputs/k10_n1000/figures/` for:
- `dashboard.png` - Combined view of all metrics
- `attention_to_z.png` - Attention patterns over training
- `logit_lens.png` - Layer-wise correct probability
- `z_dependence.png` - Causal z-dependence score

## Running on RunPod

For parallel experiments across K values:

```bash
# Generate commands
python scripts/launch_runpod.py --generate

# Output:
# K=1:   python scripts/train.py experiment.name=k1_n1000 data.k=1 ...
# K=10:  python scripts/train.py experiment.name=k10_n1000 data.k=10 ...
# K=50:  python scripts/train.py experiment.name=k50_n1000 data.k=50 ...
# K=100: python scripts/train.py experiment.name=k100_n1000 data.k=100 ...
```

Run each on a separate GPU instance.

## Project Structure

```
late-disambiguation-lag/
├── configs/
│   ├── base.yaml                 # Default hyperparameters
│   └── experiments/              # Per-experiment configs
├── src/
│   ├── data/
│   │   ├── dataset.py            # Synthetic data generation
│   │   └── tokenizer.py          # Character-level tokenizer
│   ├── model/
│   │   └── hooked_transformer.py # HookedTransformer setup
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   └── checkpoint.py         # Checkpoint management
│   ├── probes/
│   │   ├── attention_to_z.py     # Attention probe
│   │   ├── logit_lens.py         # Logit lens probe
│   │   └── causal_patching.py    # Causal intervention probe
│   └── analysis/
│       ├── run_probes.py         # Run probes across checkpoints
│       └── visualize.py          # Generate figures
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── analyze.py                # Analysis entry point
│   └── launch_runpod.py          # Multi-GPU launcher
└── outputs/                      # Results (gitignored)
```

## Key Design Decisions

### Task direction (Bz→A vs Az→B)
- `data.task=bz_to_a` (default): B is ambiguous and z disambiguates which A is correct (K targets per base).
- `data.task=az_to_b`: A is the base and B is the target. z is redundant here; K unique A strings map to the same B (K A per B).
- `data.task=b_to_a`: B is ambiguous and z is omitted; irreducible loss should approach `log K`.
- `data.task=a_to_b`: A is the base and z is omitted; mapping is deterministic.

### Controlling first-character bias (B→A sanity check)
For `b_to_a`, the irreducible loss bound assumes the **first character** of the K valid A's is uniform.  
To enforce this, set:
`data.enforce_unique_a_first_char_per_b=true`

### Probes are optional
By default `data.probe_fraction=0.0`, so no probe set is created.  
To enable probes/analysis, set `data.probe_fraction>0`.

If you enable probes, consider keeping all examples for a base together:
`data.split_by_base=true`

If `probe_fraction=0.0`, `scripts/analyze.py` will run probes on the **training**
dataset instead (no held-out validation data).

### First-token-only ethos
Our theoretical benchmarks (e.g. the log‑K floor) apply to the **first target token**.
To keep the plots aligned with that, probes default to **first-token-only**:
`probes.attention_to_z.first_token_only=true` and `probes.logit_lens.first_token_only=true`.
This avoids averaging away the ambiguity signal across later (easy) target tokens.

### Z-reshuffle (z-shuffle) diagnostic
During training we log a **z-reshuffle** loss that swaps z tokens across the batch
while keeping B and A fixed. This preserves the base ambiguity structure but
breaks the correct selector, giving a clean test of whether the model is using z.

Interpretation:
- If shuffled loss ≈ clean loss, the model is effectively ignoring z.
- If shuffled loss spikes, the model is relying on z to pick the correct A.
- The timing of the spike marks when z-dependence emerges.

This diagnostic is read-only (no gradients) and uses the **first target token**
loss to align with the log‑K theory baseline.

### n_pairs_effective normalization
We control the number of **unique B strings** across experiments, not total examples:

| K | Total Examples | Unique B's | n_pairs_effective |
|---|----------------|------------|-------------------|
| 1 | 1000           | 1000       | 1000              |
| 10| 10000          | 1000       | 1000              |
| 100| 100000        | 1000       | 1000              |

This ensures fair comparison: all models see the same "vocabulary" of B patterns.

### Checkpointing strategy
- Save every 200 steps (configurable)
- Probes run post-hoc on checkpoints
- Training and analysis are separate scripts

### Probe implementation
- All probes inherit from `BaseProbe`
- Results are JSON-serializable
- Easy to add new probes

## Expected Results

The "Late Disambiguation Lag" should manifest as:

1. **Learning curve**: Plateau-then-jump pattern (vs smooth learning for K=1)
2. **Attention to z**: Near-zero early, then sharp increase
3. **Logit lens**: Correct probability emerges at deep layers first
4. **Causal patching**: z-dependence score jumps from 0 to ~1 at transition
5. **Random-z eval**: swapping z should sharply degrade predictions only after z is being used

The transition point should correlate across all metrics, revealing the **mechanistic moment** when the model "discovers" z.

## Citation

If you use this framework, please cite:
```
@misc{late-disambiguation-lag,
  author = {Mihir Sahasrabudhe},
  title = {Late Disambiguation Lag: A Mechanistic Analysis},
  year = {2026},
  publisher = {GitHub},
}
```
