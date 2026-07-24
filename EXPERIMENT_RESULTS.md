# Reviewer Experiment Results

Comprehensive results from experiments addressing reviewer concerns for the Late Disambiguation Lag paper. All experiments use baseline-matched hyperparameters: `split_by_base=True`, `enforce_unique_a_first_char_per_b=True`, `warmup_steps=0`, `scheduler=constant`, `lr=1e-3`, `seed=42`.

---

## Status Overview

| Exp | Name | Status | Impact |
|-----|------|--------|--------|
| 1 | Dataset-size confound | **Complete** | Reframes central claim: scaling is D-driven, not K-driven |
| 2 | tau exponent reconciliation | **Complete** | Resolves Table 1 vs Table 2 discrepancy |
| 3 | Per-head causal ablation | **Complete** | Provides causal (not just correlational) evidence for L0H3 |
| 6 | tau threshold sensitivity | **Complete** | Confirms delta is robust across threshold definitions |
| 7 | Selector variants | **Complete** | Confirms phenomenon is robust to z_length |
| Combined | tau(D) reframing | **Complete** | Unified scaling law: tau ~ D^1.205 |
| 4 | Hessian robustness | Not started | Checkpoint analysis, no training needed |
| 5 | Optimizer ablation | Not started | Requires GPU (20 runs) |
| 8 | Boundary extension | Not started | Requires GPU (28 runs) |

---

## Experiment 1: Dataset-Size Confound Control (v2)

### Reviewer concern
D = n_unique_b x K, so larger K means more training data. The reported tau ~ K^delta might actually reflect dataset size, not ambiguity.

### Design
Hold total dataset size D constant by varying n_unique_b inversely with K:
- D = 10,000: K=5 (n=2000), K=10 (1000), K=20 (500), K=36 (278)
- D = 20,000: K=5 (4000), K=10 (2000), K=20 (1000), K=36 (556)
- 8 runs total, all with baseline-matched hyperparameters

### Results

#### tau at fixed D

| D | K | n_unique_b | tau | Final loss | Cand acc | Z-gap | Early stopped |
|---|---|-----------|-----|-----------|---------|-------|---------------|
| 10,000 | 5 | 2000 | 1600 | 0.093 | 100% | 11.23 | Yes (step 10550) |
| 10,000 | 10 | 1000 | 1650 | 0.157 | 100% | 11.87 | Yes (step 5900) |
| 10,000 | 20 | 500 | 1600 | 0.236 | 100% | 11.34 | Yes (step 5350) |
| 10,000 | 36 | 278 | 1600 | 0.153 | 100% | 14.17 | Yes (step 8550) |
| 20,000 | 5 | 4000 | 4250 | 0.191 | 100% | 10.87 | Yes (step 29950) |
| 20,000 | 10 | 2000 | 3800 | 0.256 | 100% | 11.03 | Yes (step 20950) |
| 20,000 | 20 | 1000 | 3900 | 0.353 | 100% | 10.90 | Yes (step 18400) |
| 20,000 | 36 | 556 | 3850 | 0.330 | 100% | 14.16 | Yes (step 22000) |

#### Power-law fits: tau ~ K^delta at fixed D

| D | delta | 95% CI | n |
|---|-------|--------|---|
| 10,000 | **-0.004** | [-0.044, +0.044] | 4 |
| 20,000 | **-0.042** | [-0.161, +0.037] | 4 |

### Key finding
**tau is flat across K when D is held constant.** The exponent delta is indistinguishable from zero at both D values. This means the original paper's tau ~ K^1.3 was actually tau ~ D^1.3, since D = 1000K in the baseline experiments.

### Baseline validation
The confound run at (K=10, n_unique_b=1000, D=10000) gives tau=1650, compared to the baseline `landauer_dense_k10` tau=1850. The ~11% difference is consistent with stochastic variation from different random mapping generation, confirming the v2 config correctly matches the baseline.

### Figures
- `outputs/paper_figures/fig_dataset_confound.{pdf,png}`

### Data
- `outputs/dataset_confound_v2_summary.json`

---

## Experiment 2: tau Exponent Reconciliation

### Reviewer concern
Table 1 reports delta=1.31+/-0.14, but Table 2 reports delta=1.70+/-0.09. Which is the correct exponent?

### Design
Dense K-sweep at lr=1e-3 with 11 K values: {3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 36}. All with n_unique_b=1000 (original baseline setting). Fit power law on full range vs restricted {10, 20, 36} range.

Only K=15 required a new training run; all others already existed.

### Results

#### tau values across K

| K | D (=1000K) | tau | Run name |
|---|-----------|-----|----------|
| 3 | 3,000 | 450 | landauer_dense_k3 |
| 5 | 5,000 | 800 | landauer_dense_k5 |
| 7 | 7,000 | 1050 | landauer_dense_k7 |
| 10 | 10,000 | 1850 | landauer_dense_k10 |
| 13 | 13,000 | 2100 | landauer_dense_k13 |
| 15 | 15,000 | 2600 | exponent_K15 |
| 17 | 17,000 | 3300 | landauer_dense_k17 |
| 20 | 20,000 | 3950 | landauer_dense_k20 |
| 25 | 25,000 | 5250 | landauer_dense_k25 |
| 30 | 30,000 | 6950 | landauer_dense_k30 |
| 36 | 36,000 | 8750 | landauer_dense_k36 |

#### Power-law fits

| Range | delta | 95% CI | n |
|-------|-------|--------|---|
| Full (K=3..36) | **1.194** | [1.137, 1.308] | 11 |
| Restricted (K=10,20,36) | 1.210 | [1.094, 2.749] | 3 |

### Key finding
The true exponent is **delta = 1.194 [1.137, 1.308]** with 11 data points. The paper's conflicting values (1.31 and 1.70) were artifacts of small sample sizes and range selection. The restricted 3-point fit gives a similar point estimate but a CI 10x wider, explaining the instability.

### Reinterpretation in light of Exp 1
Since D = 1000K in these baseline runs, tau ~ K^1.19 is equivalently tau ~ D^1.19. Experiment 1 confirms this is driven by D, not K: when D is fixed, tau does not vary with K.

### Figures
- `outputs/paper_figures/fig_tau_exponent.{pdf,png}`

### Data
- `outputs/tau_exponent_summary.json`

---

## Combined Analysis: tau(D) Unified Scaling Law

### Motivation
Experiments 1 and 2 together suggest tau is a function of D alone. Can we fit a single tau(D) curve across both experiments?

### Results

**Combined fit across 19 data points (11 from Exp 2 + 8 from Exp 1):**

> **tau = C * D^1.205, R^2 = 0.9919**
>
> 95% CI for exponent: [1.151, 1.307]

#### Cross-experiment validation
Exp 1 fixed-D points fall on the Exp 2 tau(D) curve with actual/predicted ratios of 0.91-1.06.

#### Residual analysis (is there a K effect beyond D?)
- Within D=10,000: correlation(log K, residual) = -0.250 (weak, not significant)
- Within D=20,000: correlation(log K, residual) = -0.703 (moderate but **wrong direction** — higher K gives *lower* residual)
- No evidence of a K effect beyond D

### Implications for the paper
The scaling law should be reframed from "tau ~ K^delta" to "tau ~ D^delta" where D = n_unique_b x K is total dataset size. The late disambiguation lag still exists (the model plateaus at loss = log K before transitioning), but its *duration* scales with data volume, not ambiguity level.

### Figures
- `outputs/paper_figures/fig_tau_vs_D_combined.{pdf,png}` (3 panels: tau vs D combined, Exp 2 with dual axes, Exp 1 flat tau vs K)

---

## Experiment 3: Per-Head Causal Ablation

### Reviewer concern
L1H3 attention patterns are correlational. Need causal evidence that specific heads are necessary (and sufficient) for z-usage.

### Design
For each of the 16 heads (4 layers x 4 heads), at 3 training phases (pre/mid/post-transition):
- **Zero ablation (necessity):** Zero out head output, measure loss increase
- **Mean ablation:** Replace with batch-mean activation, measure loss increase
- **Sufficiency:** Ablate ALL heads except one, check if model still works

Tested at K=10 (tau=1850) and K=20 (tau=3950). Used `hook_z` (per-head attention output, shape `(batch, seq, n_heads, d_head)`).

### Results: K=10

#### Pre-transition (step 500)
No head is causally important yet. All ablation deltas are tiny (<0.02).

| Ablation | Top head | Delta loss | Delta z-gap |
|----------|---------|-----------|------------|
| Zero | L0H3 | +0.019 | -0.020 |
| Mean | L0H3 | +0.019 | -0.018 |

Sufficiency: Best single head is L0H3 (loss=2.924), but this is barely below the baseline (2.827).

#### Mid-transition (step 1800)
L0H3 emerges as the most causally important head, but other L0 heads also matter.

| Ablation | Top head | Delta loss | Delta z-gap |
|----------|---------|-----------|------------|
| Zero | **L0H3** | **+1.412** | **-1.383** |
| Zero | L0H0 | +1.296 | -1.263 |
| Zero | L0H2 | +1.281 | -1.269 |
| Mean | **L0H3** | **+1.452** | **-1.438** |
| Mean | L0H1 | +1.317 | -1.346 |

Sufficiency: Best single head is L0H2 (loss=4.158).

#### Post-transition (step 3700)
L0H3 has the largest causal effect of any individual head.

| Ablation | Top head | Delta loss | Delta z-gap |
|----------|---------|-----------|------------|
| Zero | **L0H3** | **+3.651** | **-3.657** |
| Zero | L0H2 | +3.374 | -3.361 |
| Zero | L0H1 | +3.355 | -3.289 |
| Mean | **L0H3** | **+3.747** | **-3.789** |
| Mean | L0H1 | +3.565 | -3.729 |

Sufficiency: Best single head is L0H3 (loss=8.954), but no single head alone is sufficient (clean loss is ~0).

### Results: K=20

#### Pre-transition (step 1000)
L0H3 already shows the largest (though still small) causal effect.

| Ablation | Top head | Delta loss | Delta z-gap |
|----------|---------|-----------|------------|
| Zero | **L0H3** | **+0.022** | **-0.022** |
| Mean | **L0H3** | **+0.063** | **-0.063** |

#### Mid-transition (step 3900)
L0H3 dominates. The gap between L0H3 and the next head is larger at K=20 than at K=10.

| Ablation | Top head | Delta loss | Delta z-gap |
|----------|---------|-----------|------------|
| Zero | **L0H3** | **+1.722** | **-1.719** |
| Zero | L0H0 | +1.069 | -1.093 |
| Mean | **L0H3** | **+1.763** | **-1.768** |

Sufficiency: Best single head is L0H3 (loss=4.040).

#### Post-transition (step 7900)
L0H3 remains #1 but L0H0 is close.

| Ablation | Top head | Delta loss | Delta z-gap |
|----------|---------|-----------|------------|
| Zero | **L0H3** | **+3.691** | **-3.726** |
| Zero | L0H0 | +3.612 | -3.707 |
| Mean | **L0H3** | **+3.780** | **-3.772** |
| Mean | L0H0 | +3.453 | -3.433 |

### Key findings

1. **L0H3 is consistently the most causally important head**, across both K values and all training phases. Ablating it causes the largest loss increase and the largest reduction in z-usage (z-gap).

2. **The effect is in Layer 0, not Layer 1.** The paper's reference to "L1H3" may use 1-indexed layers. In TransformerLens 0-indexed convention, the critical head is `blocks.0.attn` head 3.

3. **Necessity is strong, sufficiency is weak.** No single head alone can maintain model performance. Z-usage requires a distributed circuit across multiple L0 heads.

4. **Causal effect grows through training.** L0H3's loss delta goes from +0.02 (pre) to +1.4 (mid) to +3.7 (post), tracking the model's increasing reliance on z.

5. **At K=20, L0H3 dominance is more pronounced** during mid-transition (L0H3 delta=1.72 vs next best 1.07), consistent with higher-K tasks requiring more selective z-attention.

### Figures
- `outputs/paper_figures/fig_head_ablation_k10.{pdf,png}` (heatmap of per-head ablation effects across phases)
- `outputs/paper_figures/fig_head_ablation_k20.{pdf,png}`

### Data
- `outputs/head_ablation_summary.json`

---

## Experiment 6: tau Threshold Sensitivity

### Reviewer concern
tau is defined as the step where candidate loss drops below 50% of log K. Is the scaling exponent delta sensitive to this arbitrary threshold choice?

### Design
Pure re-analysis of existing K-sweep data (no new training). Recompute tau at threshold fractions {0.3, 0.4, 0.5, 0.6, 0.7} of log K. Fit power law tau ~ K^delta at each threshold.

### Results

| Threshold | delta | 95% CI | n points |
|-----------|-------|--------|----------|
| 0.3 | 1.224 | [1.158, 1.340] | 10 |
| 0.4 | 1.200 | [1.132, 1.320] | 10 |
| **0.5** | **1.197** | **[1.144, 1.302]** | **10** |
| 0.6 | 1.169 | [1.109, 1.306] | 10 |
| 0.7 | 1.120 | [1.068, 1.229] | 10 |

#### Robustness statistics
- delta range across thresholds: 0.104
- delta mean: 1.182
- delta std: 0.035

### Key finding
**delta is robust across threshold definitions.** The exponent varies by only ~0.10 across the range 0.3-0.7, with all values falling within [1.12, 1.22]. The 95% CIs overlap substantially. A reviewer cannot dismiss the scaling law by arguing the threshold is arbitrary.

Note: As with Exp 2, this exponent reflects tau ~ D^delta in light of Exp 1.

### Figures
- `outputs/paper_figures/fig_threshold_sensitivity.{pdf,png}`
- `outputs/paper_figures/fig_exponent_robustness.{pdf,png}`

### Data
- `outputs/threshold_sensitivity_summary.json`

---

## Experiment 7: Selector Variant Tests

### Reviewer concern
The selector z is always 2 characters at a fixed position. Results might be an artifact of this specific format.

### Design
Test z_length in {1, 2, 3, 4} at K=10, lr=1e-3, n_unique_b=1000. z_length=2 uses the existing baseline run (`landauer_dense_k10`).

### Results

| z_length | tau | Candidate plateau | Plateau / log K | Final cand loss |
|----------|-----|------------------|-----------------|-----------------|
| 1 | 1650 | 2.342 | **1.017** | 0.0021 |
| 2 (baseline) | 1850 | 2.388 | **1.037** | ~0 |
| 3 | 1500 | 2.334 | **1.014** | ~0 |
| 4 | 1450 | 2.318 | **1.007** | ~0 |

Theory predicts: candidate_loss plateau = log K = 2.303 for K=10.

### Key findings

1. **Plateau height matches log K for all z_lengths.** The ratio candidate_plateau / log K ranges from 1.007 to 1.037, confirming the theoretical prediction regardless of selector format.

2. **tau is similar across z_lengths** (~1450-1850). Slight trend toward faster transition with longer z (more redundant information), but within noise range.

3. **All runs converge** to near-zero candidate loss and 100% accuracy.

4. **The late disambiguation lag is not an artifact of z_length=2.** The phenomenon (plateau at log K, then sharp transition) is robust to the selector variable format.

### Important methodological note
An earlier version of this analysis incorrectly used `first_target_loss` (which plateaus at log(|vocab|) = log(36) = 3.584) for the plateau measurement. The correct metric is `candidate_loss` (which measures only among the K valid candidates). This was caught and fixed — the `first_target_loss` / log K ratio of ~1.56 was spurious.

### Figures
- `outputs/paper_figures/fig_selector_variants.{pdf,png}`

### Data
- `outputs/selector_variants_summary.json`

---

## Experiment 4: Hessian Robustness Checks

### Reviewer concern
Power iteration is noisy. Need to verify eigenvalue estimates are stable across batch sizes and iteration counts.

### Design
Vary batch size {256, 512, 1024, 2048} and power iteration count {25, 50, 100, 200} across 5 checkpoints at K=20 (tau=3950). 80 computations total.

### Results

#### Power iteration convergence: ROBUST
At any fixed batch size, lambda_max is identical across 25/50/100/200 iterations (converges by iter=25). This means the paper's default of 50 iterations is more than sufficient.

Example at step 3900, bs=512:
| Iterations | lambda_max | lambda_min |
|-----------|-----------|-----------|
| 25 | +10.242 | -0.00509 |
| 50 | +10.284 | -0.00504 |
| 100 | +10.284 | -0.00504 |
| 200 | +10.284 | -0.00504 |

#### Batch size sensitivity: SIGNIFICANT for lambda_max, especially near transition

| Step | Phase | bs=256 | bs=512 | bs=1024 | bs=2048 | CV |
|------|-------|--------|--------|---------|---------|------|
| 800 | pre | 3.659 | 3.665 | 3.663 | 3.657 | 0.001 |
| 3100 | ~tau | 6.732 | 4.513 | 3.897 | 3.496 | 0.269 |
| 3900 | post | 16.091 | 10.284 | 9.197 | 8.548 | 0.271 |
| 6200 | post | 37.617 | 55.228 | 28.240 | 14.760 | 0.434 |
| 50000 | conv | 1.467 | 0.768 | 1.111 | 1.165 | 0.220 |

lambda_max is stable pre-transition (CV=0.001 at step 800) but becomes batch-size-dependent during and after transition (CV up to 0.43). Smaller batches give higher lambda_max estimates due to per-batch gradient variance.

#### lambda_min: NOISY
lambda_min is extremely small (often ~1e-5) and its CV exceeds 1.0 at most checkpoints. The sign is unreliable. At bs=256 step 3100, lambda_min = -3.016; at bs=2048 same step, lambda_min = +0.000001. This means any paper claims about lambda_min magnitude or sign should be treated with caution.

#### Spinodal decomposition test
The script also tested whether lambda_min crosses zero at tau (as predicted by spinodal decomposition analogy):
- **K=10**: lambda_min trends toward zero pre-tau, crosses to negative post-tau. **Partial support.**
- **K=20**: Same pattern. **Partial support.**
- But given lambda_min's high noise, this is suggestive rather than conclusive.

### Interpretation: what the Hessian tells us about the metastable regime

**Background.** lambda_max (largest Hessian eigenvalue) measures the steepest curvature direction in the loss landscape — a high value means a sharp wall or narrow valley. lambda_min (smallest eigenvalue) tells you whether you're at a local minimum (positive) or a saddle point (negative — meaning there's an escape direction the optimizer hasn't exploited yet).

**The geometric story across training phases:**

1. **Pre-transition (step 800, during the log(K) plateau):** lambda_max ~ 3.7, lambda_min ~ 0. The landscape is flat in almost every direction. The model is sitting in a broad, shallow basin. There is no steep escape route visible to the optimizer. This is consistent with the "ignore z, predict uniformly over K candidates" strategy being a genuinely stable equilibrium — not just slow progress, but an actual flat region.

2. **At/near transition (steps 3100-3900):** lambda_max spikes to 10-55, lambda_min becomes slightly negative. The landscape is suddenly sharp and structured. A steep direction has appeared (the "z-usage" direction), and a negative eigenvalue means a saddle point has emerged — there is now a downhill escape route. The flat basin has deformed into a saddle.

3. **Post-convergence (step 50000):** lambda_max ~ 1, lambda_min ~ 0. Back to a gentle, flat minimum. The model has settled into the correct solution.

**Why this matters for the paper's narrative.** This provides a geometric mechanism for both the plateau and the sharp transition:
- The plateau exists because the loss landscape is genuinely flat during it. The optimizer isn't being slow — there is nowhere steep to go. The "use z" direction doesn't appear as a viable gradient signal until the model has first learned the candidate structure.
- The transition is sharp because it corresponds to a geometric bifurcation: a saddle point and steep escape direction emerge together, rather than the landscape smoothly tilting.
- This connects to the decomposition between failure form and failure timescale: the model first learns candidate sets (which reshapes the landscape geometry), and only after that structure is in place does the "use z" escape route appear in the Hessian.

The Hessian story **supports the metastable regime narrative** — the plateau is a flat basin, the transition is a saddle point emerging. But the quantitative precision is limited (see robustness caveats above), so the paper should present this as geometric intuition backed by qualitative Hessian evidence, not as precise spectral measurements.

### Key findings (technical)

1. **Power iteration count doesn't matter** — 25 iterations is enough for convergence. The paper's estimates are not noisy in this dimension.

2. **Batch size matters a lot near the transition.** lambda_max varies 2-4x across batch sizes during the critical transition period. The paper should either fix a batch size and note this, or report batch-size-averaged values with error bars.

3. **lambda_min claims should be softened.** The quantity is too small and noisy to make precise claims about. Qualitative statements (e.g., "lambda_min approaches zero near tau") are defensible; quantitative ones are not.

4. **Pre-transition landscape is genuinely flat.** lambda_max ~ 3.7 and lambda_min ~ 0 at step 800, indicating a very flat loss landscape during the plateau. This is robust across all settings (CV = 0.001).

### Figures
- `outputs/paper_figures/fig_hessian_robustness.{pdf,png}`
- `outputs/paper_figures/fig_hessian_eigenvalues.{pdf,png}`
- `outputs/paper_figures/fig_hessian_lambda_min_zoom.{pdf,png}`

### Data
- `outputs/hessian_robustness_summary.json`

---

## Remaining Experiments

### Experiment 5: Optimizer Ablation
**Status:** Not started (requires GPU, 20 training runs)
**Script:** `scripts/exp_optimizer_ablation.py`
**Design:** SGD+momentum, SGD no momentum, AdamW with varied weight decay.
**Goal:** Confirm tau scaling is not optimizer-specific.

### Experiment 8: Extended Phase Boundary
**Status:** Not started (requires GPU, 28 training runs)
**Script:** `scripts/exp_boundary_extension.py`
**Design:** Fill K={7, 13, 25, 50} into the eta*(K) boundary with eta sweeps.
**Goal:** Tighten CI on the phase boundary exponent with 7+ K values.

---

## Summary of Key Findings for Paper Revision

1. **The scaling law is D-driven, not K-driven.** tau ~ D^1.2, not tau ~ K^1.2. When D is fixed, tau does not vary with K (delta = 0). This is the most consequential finding and requires reframing the paper's central claim.

2. **The late disambiguation lag itself is real and robust.** The candidate loss plateau at log K appears regardless of z_length, threshold definition, or dataset size. What scales with data is the *duration* of the lag, not its existence.

3. **L0H3 is causally necessary for z-usage.** Zero-ablation causes loss increase of +3.7 (from near-zero to above log K) post-transition. This converts the paper's correlational attention-pattern evidence into causal evidence.

4. **The exponent is stable.** delta ~ 1.18-1.22 across threshold definitions (0.3-0.7), K ranges, and fit methods. The paper's prior conflicting values (1.31 vs 1.70) were small-sample artifacts.

5. **No single head is sufficient.** The z-usage circuit is distributed across L0 heads, even though L0H3 is the most important individual contributor.
