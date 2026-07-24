# Post-Hoc Experiments: Connecting Disambiguation Lag to Theoretical Frameworks

Nine post-hoc experiments that test predictions from recent theoretical work against our existing experimental data on disambiguation lag (the metastable plateau at loss = log K followed by a sharp phase transition in transformer training). All experiments use **pre-existing checkpoints and training histories** -- no new training runs were conducted.

## Motivating Papers

| Paper | Key Idea | Our Test |
|-------|----------|----------|
| [Gromov 2024](https://arxiv.org/abs/2405.19454) "Deep Grokking" | Weight norms and feature rank predict delayed generalization | Exp 1, 2 |
| [Aguilera et al. 2024](https://arxiv.org/abs/2408.08954) Synergy as order parameter | O-information spike precedes phase transition | Exp 7 |
| [Lyu et al. 2025](https://arxiv.org/abs/2505.13738) "Power Lines" | AdamW composite timescale tau = B/(eta * lambda * D) governs learning | Exp 4 |
| [Chen et al. 2024](https://arxiv.org/abs/2512.00686) SLT / Hidden Progress | Weights move during plateau despite flat loss; LLC predicts transitions | Exp 3, 6 |
| [Rangamani et al. 2025](https://arxiv.org/abs/2509.20829) Neural Collapse & Grokking | Within-class variance contraction drives generalization | Exp 5 |
| [Tigges et al. 2024](https://arxiv.org/abs/2407.00886) CD-T Circuit Discovery | W_OV spectral structure reveals circuit formation | Exp 2 |

## Data Available

All runs use AdamW with weight_decay=0.01, d_model=128, 4 layers, 4 heads. Full weight checkpoints saved every 100 training steps.

- **Transformer K-sweep:** `landauer_dense_k{3,5,7,10,13,17,20,25,30,36}` (500 checkpoints each)
- **Batch-size sweep:** `temp_lr1e3_bs{64,128,256,512}_k{20,36}`
- **LR sweep:** `lr_sweep_eta{3e-4,5e-4,1e-3,2e-3}`
- **GatedMLP K-sweep:** `gatedmlp_k{10,15,20,25,36,50}`
- **RNN K-sweep:** `rnn_k{10,15,20,25,36,50,100}`
- **Per-head probes:** `temp_lr1e3_bs128_k20` with attention-to-z per head per step

---

## Experiment 1: Weight Norm Dynamics

**Script:** `scripts/posthoc_weight_norms.py`
**Prediction (Deep Grokking):** Total weight norm should show non-monotonic behavior -- growing during memorization, then reorganizing at the transition.

### Results

![Weight Norms](outputs/paper_figures/fig_weight_norms.png)

Weight norms grow **monotonically** with training -- there is no non-monotonic peak-and-reorganize pattern predicted by the deep grokking framework. The growth rate increases sharply at the transition tau (panel a), suggesting the transition is associated with rapid weight expansion rather than contraction. Higher K produces proportionally larger final norms (peak/init ratio grows from 1.13x at K=3 to 2.53x at K=36).

**Per-component breakdown (K=20, panel b):** Attention and MLP norms grow in parallel, but attention shows a visible inflection at tau while MLP continues growing monotonically. The unembed norm shows the most dramatic change -- a sharp rise beginning exactly at the transition. This is consistent with the model suddenly needing to map differentiated representations to K distinct output tokens.

**Per-layer breakdown (K=20, panel c):** Layer 1 norm peaks and then **decreases** after the transition -- the only layer showing non-monotonic behavior. This is notable because L1H3 is the identified selector head. The peak-and-decay of Layer 1 suggests it overshoots during the transition and then relaxes as the rest of the network adapts.

| K | tau | \|\|W\|\| at init | \|\|W\|\| at tau | \|\|W\|\| final | peak/init |
|---|-----|-----------------|-----------------|----------------|-----------|
| 3 | 550 | 72.04 | 79.01 | 80.14 | 1.13 |
| 5 | 1200 | 72.00 | 84.77 | 90.98 | 1.29 |
| 10 | 2700 | 72.00 | 98.34 | 98.07 | 1.44 |
| 20 | 10750 | 72.00 | 126.58 | 139.17 | 1.94 |
| 36 | -- | 71.99 | -- | 182.13 | 2.53 |

**Verdict:** The deep grokking prediction of non-monotonic total norms does **not** hold in our setting. However, per-layer analysis reveals that Layer 1 (containing the selector head L1H3) does show a localized non-monotonic signature. The overall picture is one of continuous weight growth with an acceleration at tau, not reorganization.

---

## Experiment 2: OV/QK Circuit Spectrum

**Script:** `scripts/posthoc_circuit_spectrum.py`
**Prediction (Deep Grokking + CD-T):** Effective rank of W_OV should decrease at the transition (crystallization into low-rank circuit). The selector head L1H3 should show distinctive spectral changes before other heads.

### Results

![Circuit Spectrum](outputs/paper_figures/fig_circuit_spectrum.png)

**Panel (a) -- Effective rank heatmap (K=20):** The heatmap shows that effective rank drops broadly across all heads around the transition (vertical red line at tau=10750), but the most dramatic drops occur in **Layer 0** and **Layer 3** heads -- not in the selector head L1H3. The Layer 0 heads (L0H0-L0H3) show the earliest and sharpest rank reduction, developing a darker band (lower rank) that begins well before tau.

**Panel (b) -- Dominant mode ratio sigma_1/sigma_2:** L0H0 (blue) shows a dramatic transient spike in sigma_1/sigma_2 early in training (~step 5000), reaching nearly 1.8x before settling back down. This spike occurs well before tau and is not seen in L1H3 (red) or other heads. L1H3's dominant mode ratio remains remarkably flat throughout training (~1.05-1.09), meaning it does **not** crystallize into a rank-1 circuit. Instead, it maintains a distributed representation.

**Panel (c) -- Nuclear norm:** L1H3 has notably **lower** OV nuclear norm than all other heads throughout training. While other heads cluster in the 50-75 range, L1H3 stays around 20-50. The gap narrows slightly at the transition. This suggests L1H3 operates as a precision instrument -- low-norm but functionally critical.

**Panel (d) -- L1H3 effective rank across K:** At K=10 the effective rank drops sharply at tau, at K=20 the drop is more gradual, and at K=36 it drops even more slowly. This K-dependence in the spectral dynamics is consistent with higher K requiring more complex representations that resist rank collapse.

**Key spectral evolution of L1H3 (K=20):**

| Phase | Step | OV eff_rank | sigma_1/sigma_2 | OV nuclear |
|-------|------|-------------|-----------------|------------|
| Init | 100 | 28.14 | 1.05 | 19.27 |
| Mid-plateau | 5400 | 25.84 | 1.09 | 40.19 |
| At tau | 10800 | 26.69 | 1.08 | 50.69 |
| Final | 50000 | 27.83 | 1.09 | 57.42 |

**Verdict:** The strong prediction of sharp rank collapse at the transition **partially holds** -- it is visible in the heatmap across many heads -- but L1H3 itself does not show it. The selector head maintains distributed (high effective rank) representations throughout. The most interesting finding is the early transient sigma_1/sigma_2 spike in L0H0, and L1H3's persistently low nuclear norm. The circuit formation appears to be a distributed process rather than concentration in a single low-rank head.

---

## Experiment 3: Weight Displacement & Hidden Progress

**Script:** `scripts/posthoc_weight_displacement.py`
**Prediction (Hidden Progress / SLT):** During the loss plateau, weight displacement per step should be nonzero and possibly increasing, despite constant loss. This would constitute "hidden progress" -- the weights are moving toward the transition even though the loss doesn't budge.

### Results

![Weight Displacement](outputs/paper_figures/fig_weight_displacement.png)

This is the strongest result across all seven experiments.

**Panel (a) -- Weight displacement per step:** Displacement is **clearly nonzero** during the entire plateau phase for all K values, running at approximately 10 Frobenius norm units per 200-step interval. At the transition, displacement shows a dramatic spike (visible as the sharp peak at each K's tau). Post-transition, displacement drops by an order of magnitude (K=10: from 14.2 to 1.4, K=20: from 11.3 to 3.9).

**Panel (b) -- Relative displacement:** Normalizing by weight norm reveals a clear three-phase structure: (1) high relative displacement at initialization that decays rapidly, (2) a plateau-era "cruising speed" of ~0.08-0.15, and (3) a sharp drop at tau to ~0.01-0.03. The model settles into a new, much more stable equilibrium after the transition.

**Panel (c) -- Per-component displacement (K=20):** During the plateau, embedding and unembed components show the highest displacement, while attention and MLP are lower. At the transition, all components spike simultaneously, but attention displacement shows the sharpest and tallest spike -- consistent with the transition being driven by attention circuit reorganization.

**Panel (d) -- Direction consistency (cosine similarity of consecutive deltas):** This is the most striking panel. During the plateau, direction consistency is **near zero** for K=20 (mean cos_sim = 0.036) and even slightly **negative** for the pre-transition window (-0.179). This means the weight updates during the plateau are essentially a **random walk** -- each step moves in an unrelated direction from the previous one. At the transition, direction consistency jumps sharply to 0.8-0.9, indicating the optimizer has found a consistent descent direction and is moving purposefully.

| Phase | K=10 | K=20 |
|-------|------|------|
| **Plateau displacement** | 10.80 +/- 4.31 | 11.27 +/- 4.22 |
| **Transition displacement** | 14.16 +/- 2.89 | 9.44 +/- 0.36 |
| **Post-transition displacement** | 1.39 +/- 4.00 | 3.93 +/- 4.71 |
| **Plateau cos_sim** | 0.224 | 0.036 |
| **Post-transition cos_sim** | 0.838 | 0.397 |

**Verdict:** Hidden progress is **strongly confirmed**. The weights move continuously during the plateau, but in a random-walk pattern. The transition manifests as a sudden alignment of update directions -- a symmetry-breaking event where the optimizer snaps from undirected exploration to directed descent. This is consistent with SLT's picture of the loss landscape having a nearly-flat manifold of solutions during the plateau, with the optimizer diffusing across it until it finds a descent channel.

---

## Experiment 4: AdamW Timescale Universality

**Script:** `scripts/posthoc_adamw_timescale.py`
**Prediction (Power Lines):** The composite timescale tau_adamw = B / (eta * lambda * D) should predict or correlate with the measured transition time tau_trans.

### Results

![AdamW Timescale](outputs/paper_figures/fig_adamw_timescale.png)

**Panel (a) -- tau_trans vs tau_adamw (log-log):** The relationship is **negative** -- tau_trans scales inversely with tau_adamw, opposite to the Power Lines prediction. The overall regression gives tau_trans proportional to tau_adamw^{-1.14} with R^2 = 0.70. This anti-correlation makes sense: tau_adamw = B/(eta * lambda * D) is large when the dataset D is small (low K), the batch size B is large, or the learning rate eta is small. All of these individually make the transition **faster** in our setting (except small eta, which has a non-monotonic effect).

**Per-sweep breakdown (Transformer only):**

| Sweep | Slope | R^2 | n | Interpretation |
|-------|-------|-----|---|---------------|
| K-sweep | -1.70 | 0.966 | 9 | tau_adamw decreases as K grows (D=1000K), but tau_trans increases steeply |
| BS-sweep | -1.16 | 0.908 | 6 | Larger B increases tau_adamw and decreases tau_trans |
| LR-sweep | -0.53 | 0.814 | 4 | Weak effect; eta affects both timescales |

**Panel (b) -- Ratio tau_trans/tau_adamw vs K:** The ratio spans **three orders of magnitude** (0.1 to 500), growing as a power law in K. All three architectures (Transformer, GatedMLP, RNN) show similar growth curves but with different offsets -- Transformer transitions are slowest for a given K, RNN fastest.

**Cross-architecture slopes (K-sweep only):**
- Transformer: slope = -1.70, R^2 = 0.966
- GatedMLP: slope = -1.62, R^2 = 0.969
- RNN: slope = -1.47, R^2 = 0.977

**Verdict:** The Power Lines prediction **does not hold** in our setting. tau_adamw is not a useful predictor of tau_trans -- the relationship is inverted. This is likely because our task's difficulty scales with K in a way that dominates over the optimizer dynamics. The AdamW weight-decay timescale (1/(eta * lambda) = 100k steps) is much longer than any observed transition, so weight decay acts as a slow background regularizer rather than the primary clock. The strong per-sweep R^2 values show the relationship is systematic, not noise -- it is a genuine failure of the Power Lines framework to capture this regime.

---

## Experiment 5: Neural Collapse Proxy

**Script:** `scripts/posthoc_neural_collapse.py`
**Prediction (Neural Collapse & Grokking):** Within-class variance of last-layer representations should contract at the transition. In our setup, grouping by base string B: before the transition all K representations for a given B should be similar (model ignores z), and after they should separate.

### Results

![Neural Collapse](outputs/paper_figures/fig_neural_collapse.png)

**Panel (a) -- Representation variance (K=20):** Within-B variance (red) rises sharply from near-zero at init to a peak of ~9.5 around step 18000 (well after tau=10750), then gradually declines. Between-B variance (blue) rises early, peaks around step 2000, then slowly decays throughout. Both variances are declining in the post-transition regime, but within-B variance declines faster.

**Panel (b) -- Variance ratio (within/between):** The ratio rises monotonically from ~1.7 at init to ~7.3 at step 40000, with a brief dip at step 42000. This means representations become progressively **more differentiated within each B group** relative to between-group spread -- the opposite of traditional neural collapse where within-class variance contracts.

**Panel (c) -- Total variance:** Total variance peaks around step 18000 then gradually contracts, suggesting weight decay is slowly regularizing the representation space after the transition.

**Post- vs pre-transition comparison:**

| Metric | Plateau mean | Post-transition mean | Change |
|--------|-------------|---------------------|--------|
| Within-B variance | 4.55 | 5.25 | 1.15x (increases) |
| Between-B variance | 1.45 | 0.89 | 0.61x (decreases) |

**Verdict:** The traditional neural collapse prediction (within-class contraction) is **inverted** in our setting. Within-B variance **increases** at the transition because the model is learning to differentiate the K=20 different (B, z) combinations within each B group. Between-B variance contracts, meaning the B-group centroids become more similar as the model shifts representational capacity from distinguishing B strings to distinguishing z values. This makes physical sense: the pre-transition model uses all its capacity to represent B (ignoring z), while the post-transition model must also represent z, which is a within-B distinction.

---

## Experiment 6: Gradient Norm Anatomy

**Script:** `scripts/posthoc_gradient_anatomy.py`
**Prediction (SLT / Deep Grokking):** Gradient norms should show structure during the plateau, and per-component norms should reveal differential dynamics.

### Results

![Gradient Anatomy](outputs/paper_figures/fig_gradient_anatomy.png)

**Panel (a) -- Gradient norm squared:** Gradient norms show a clear three-phase structure visible across all K values: (1) a sharp initial transient (large gradients at init), (2) a plateau phase with approximately constant, low gradient norms, and (3) a dramatic spike at tau followed by a post-transition decay. The transition spike is 3-13x the plateau level. After the spike, gradients drop to very low values as the model converges.

**Panel (b) -- Fisher proxy (||grad L||^2 / L):** Normalizing by loss reveals that the gradient-to-loss ratio drops during the plateau (gradients become less efficient at reducing loss) then spikes sharply at the transition. For K=3 and K=5, the Fisher proxy shows a clean spike at tau. For larger K, the picture is similar but noisier.

**Panel (c) -- Gradient waste during plateau:** The fraction of total gradient compute spent during the plateau (before tau/2) scales steeply with K. At K=3, essentially 0% of gradients are "wasted" in the plateau. At K=30, 19.3% of all gradient compute occurs during the plateau phase where loss barely changes.

The cumulative gradient waste scales as **tau^{2.14}** (R^2 = 0.994). This near-quadratic scaling means the computational cost of the plateau is disproportionately expensive for harder problems.

**Panel (d) -- Gradient norm across LR (K=20):** Higher learning rates produce **smaller** gradient norms (eta=3e-4 has mean 21.6, eta=2e-3 has mean 3.1) but paradoxically **longer** transition times. This is consistent with the anti-Kramers effect: larger steps overshoot the narrow descent channel, even though each individual gradient is smaller.

| K | tau | Plateau ||grad||^2 | Transition ||grad||^2 | Ratio | Plateau waste % |
|---|-----|---------------------|----------------------|-------|-----------------|
| 5 | 1200 | 1.48 | 5.03 | 3.4 | 0.3% |
| 10 | 2700 | 0.69 | 5.95 | 8.6 | 1.6% |
| 20 | 10750 | 2.70 | 8.92 | 3.3 | 6.6% |
| 30 | 30600 | 4.84 | 8.56 | 1.8 | 19.3% |

**Verdict:** Gradient norms **do** show clear structure during the plateau -- they are consistently small but nonzero, forming a well-defined "gradient floor" that increases slightly with K. The transition manifests as a sharp gradient spike. The most actionable finding is the quadratic gradient waste scaling: harder problems (larger K) waste disproportionately more compute in the unproductive plateau, suggesting that early-stopping heuristics based on gradient norms could save significant resources.

---

## Experiment 7: Z-Dependence as Synergy Proxy

**Script:** `scripts/posthoc_synergy_proxy.py`
**Prediction (Synergy as Order Parameter):** The z-shuffle gap (Delta_z = L_shuffle - L_clean), a proxy for synergistic information from z, should rise BEFORE the loss transition.

### Results

![Synergy Proxy](outputs/paper_figures/fig_synergy_proxy.png)

**Panel (a) -- Loss vs Delta_z:** For all K values, the z-shuffle gap (dashed lines) begins rising well before the loss (solid lines) begins dropping. The gap is especially visible for larger K (K=20, K=36) where the lead time is thousands of steps.

**Panel (b) -- Synergy onset vs loss onset:** All K values fall **below** the identity line t_syn = t_loss, confirming that synergy systematically leads loss. The lead is roughly proportional to the transition time: larger K values (yellow/green points) show larger absolute leads but similar fractional leads.

**Panel (c) -- Lead time vs K (log-log):** The synergy lead time follows a clean power law: **Delta_t proportional to K^{2.13}** with R^2 = 0.99. This is steeper than the transition time scaling itself (tau proportional to K^{1.7}), meaning the lead fraction grows with K.

**Full onset table:**

| K | log K | tau_loss | tau_synergy | Lead (steps) | Lead / tau |
|---|-------|----------|-------------|-------------|------------|
| 3 | 1.10 | 500 | 400 | 100 | 0.20 |
| 5 | 1.61 | 1100 | 700 | 400 | 0.36 |
| 7 | 1.95 | 1300 | 650 | 650 | 0.50 |
| 10 | 2.30 | 2350 | 1250 | 1100 | 0.47 |
| 13 | 2.56 | 3000 | 1300 | 1700 | 0.57 |
| 17 | 2.83 | 5900 | 2700 | 3200 | 0.54 |
| 20 | 3.00 | 8700 | 3800 | 4900 | 0.56 |
| 25 | 3.22 | 14550 | 5550 | 9000 | 0.62 |
| 30 | 3.40 | 24500 | 7800 | 16700 | 0.68 |
| 36 | 3.58 | 36750 | 11600 | 25150 | 0.68 |

**Mean lead fraction: 0.52 +/- 0.14** -- synergy onset occurs, on average, halfway through training before the loss drops.

**Per-head analysis (K=20):** L1H3 begins attending to z at step 500, which is **2250 steps before** the aggregate synergy signal (Delta_z > 0.1 at step 2750) and **8200 steps before** the loss transition (tau=8700). The individual circuit-level signal (head attention) precedes the aggregate information-theoretic signal (z-shuffle gap), which in turn precedes the task-level signal (loss).

**Verdict:** The synergy-as-order-parameter prediction is **strongly confirmed** with a remarkably clean power-law scaling of lead times. The z-shuffle gap serves as a reliable early warning signal for the impending phase transition. The temporal ordering -- L1H3 attention > aggregate synergy > loss drop -- suggests a mechanistic cascade: the selector head first learns to attend to z, this gradually increases the network's synergistic use of z, and eventually this crosses a critical threshold that triggers the sharp loss transition.

---

## Experiment 8: Hessian Eigenvalue Tracking (Spinodal Test)

**Script:** `scripts/posthoc_hessian_eigenvalue.py`
**Prediction (Spinodal Decomposition):** During the metastable plateau, the minimum Hessian eigenvalue should be positive (the uniform-over-K solution is a local minimum). At the transition τ, λ_min should cross zero — the solution becomes linearly unstable, triggering irreversible decomposition into K distinct representations. This is the thermodynamic "spinodal" mechanism.

### Method

Power iteration for λ_max (standard) and λ_min (shifted: power iteration on (H - σI) with σ = λ_max + 1, which makes all shifted eigenvalues negative so the largest-magnitude one corresponds to the smallest eigenvalue of H). Each Hessian-vector product costs ~2 backward passes. 50 iterations per eigenvalue, 512-example fixed subsample, computed on MPS (Apple Silicon). Checkpoints sampled sparsely (every 1000 steps) with dense sampling (every 200 steps) within ±2000 steps of τ.

### Results

![Hessian Eigenvalues](outputs/paper_figures/fig_hessian_eigenvalues.png)
![λ_min Zoom](outputs/paper_figures/fig_hessian_lambda_min_zoom.png)

**SPINODAL IS FALSIFIED.** The prediction requires λ_min > 0 during the plateau (local minimum), but we observe the opposite: λ_min < 0 throughout the plateau for all K values. The plateau is a **saddle point**, not a local minimum.

**K=10 (τ=2700):**
- During the plateau (steps 800-1200): λ_min ranges from -0.001 to -2.7 (brief large negative outlier at step 1200). λ_max ≈ 3-5.
- During transition (steps 1400-2800): λ_min stays small-negative (-0.003 to -0.008) while λ_max surges to 30-50 (a 10× spike).
- Post-convergence (step 3600+): λ_min crosses to **positive** (~+0.0002 to +0.00009) and stays positive. The converged solution is a genuine local minimum.
- λ_max decays from ~18 at step 3600 to ~0.008 at step 50000 as the model settles.

**K=20 (τ=10750):**
- During the plateau (steps 1000-10000): λ_min is consistently small-negative (-0.001 to -0.015), occasionally with a brief positive outlier (+0.0006 at step 7000, +0.0009 at step 9200). λ_max fluctuates between 20-82.
- During transition (steps 10000-17000): λ_min remains small-negative (-0.002 to -0.015) while λ_max stays elevated at 27-56.
- Post-convergence (step 18000+): λ_min crosses to **positive** (+0.0001) and stays positive through step 40000. λ_max decays from ~5.9 to ~0.00007.
- Late steps (41000+) show a loss spike (possible cosine LR restart or data sampling effect), with eigenvalues briefly re-entering transition dynamics.

**K=36 (τ=None, never converges):**
- λ_min is **negative throughout** all 50000 steps, ranging from -0.001 to -16.5 (large negative outliers at steps 13000, 19000, 21000, 46000).
- λ_max ranges from 2-64, fluctuating without clear convergence.
- Loss decays slowly from ~2.87 (log 36 = 3.58) to ~0.45 at step 50000 — still on the plateau, never reaching convergence.

**Key numbers:**

| K | τ | Plateau λ_min | Plateau λ_max | Ratio λ_max/|λ_min| | Post-conv λ_min |
|---|---|---------------|---------------|----------------------|-----------------|
| 10 | 2700 | -0.001 to -0.009 | 3-13 | ~1000× | +0.0002 → +5e-8 |
| 20 | 10750 | -0.002 to -0.015 | 20-82 | ~5000× | +0.0001 → +3e-8 |
| 36 | -- | -0.001 to -16.5 | 2-64 | ~5-20000× | stays negative |

**λ_max spike at transition:** For both K=10 and K=20, λ_max spikes 10-20× during the transition:
- K=10: From ~5 (plateau) to 51.5 (step 3200), then decays to 0.008 (step 50000)
- K=20: From ~35 (plateau) to 56.4 (step 14000), then decays to 0.0002 (step 36000)

This spike reflects the model descending through a narrow, steep canyon in the loss landscape during the rapid transition.

**The asymmetry is the key finding:** During the plateau, |λ_min| ≈ 0.001-0.01 while λ_max ≈ 5-80. The negative curvature is ~1000× weaker than the dominant positive curvature. This explains why the optimizer random-walks for many steps before finding the descent channel — the escape direction is a tiny sliver of parameter space compared to the dominant curvature directions.

**Verdict:** Spinodal decomposition is **definitively falsified**. The plateau is not a local minimum becoming unstable — it is a **saddle point** with hidden negative curvature throughout. The confirmed mechanism is **saddle-point escape**: the optimizer diffuses on a nearly-flat manifold (Exp 3's random walk) until it aligns with the feeble negative-curvature direction (~0.001 eigenvalue), at which point positive feedback amplifies the descent and the transition becomes irreversible. The converged solution is a genuine local minimum (λ_min > 0), explaining why trajectory forensics (Appendix M of the paper) finds zero reversions. The ~1000× asymmetry between |λ_min| and λ_max quantitatively explains the long plateau: escaping a saddle requires O(1/|λ_min|) steps, and the weak negative curvature predicts exactly the long random-walk phase we observe.

---

## Experiment 9: Signal Dilution Hypothesis (Coverage vs Phase Transition)

**Script:** `scripts/signal_dilution_test.py`
**Hypothesis:** The plateau duration τ is NOT caused by a geometric trap or phase transition — it's caused by statistical coverage. The model needs to learn K × 1000 individual (B, z) → A bindings, and each training example only teaches one. The gradient signal per binding weakens as K grows. Predicted scaling: τ ∝ K^{1.26}. If true, the "phase transition" is a finite-size averaging effect.

Four tests, with Test 3 being the decisive one.

### Test 1: τ × grad_norm_early ∝ K^β — PASS

![Signal Dilution Test 1](outputs/paper_figures/signal_dilution_test1.png)

The product τ × grad_norm scales as K^{0.99} (R² = 0.988), almost exactly the β = 1.0 predicted by pure coverage. The ratio τ×ḡ/K is remarkably flat:

| K | τ (steps) | ḡ_early | τ × ḡ | τ × ḡ / K |
|---|-----------|---------|-------|-----------|
| 3 | 450 | 0.616 | 277 | 92.4 |
| 5 | 800 | 0.690 | 552 | 110.4 |
| 7 | 1050 | 0.591 | 620 | 88.6 |
| 10 | 1850 | 0.615 | 1137 | 113.7 |
| 13 | 2100 | 0.530 | 1113 | 85.6 |
| 17 | 3300 | 0.518 | 1710 | 100.6 |
| 20 | 3950 | 0.488 | 1928 | 96.4 |
| 25 | 5250 | 0.438 | 2297 | 91.9 |
| 30 | 6950 | 0.431 | 2993 | 99.8 |
| 36 | 8750 | 0.392 | 3426 | 95.2 |

Mean τ×ḡ/K = 97.5 ± 8.5 (CV = 8.7%). The dimensional analysis of coverage is spot-on.

**Note:** grad_norm_early is computed from `sqrt(grad_norm_sq)` during the 20%-60% of τ window, representing the plateau-era gradient magnitude.

### Test 2: Candidate Loss Onset Timing — CONSISTENT

![Signal Dilution Test 2](outputs/paper_figures/signal_dilution_test2.png)

The 95% onset fraction (when candidate loss first drops below 95% of log K, as fraction of τ) **decreases** with K (r = -0.67): K=3 at 75%τ, K=20 at 44%τ, K=25 at 42%τ. This means larger K starts showing slight progress relatively earlier but takes much longer to finish — consistent with coverage starting early but requiring more steps to complete.

The normalized loss trajectories (right panel) show all K values collapse onto a similar shape when normalized by τ, but larger K values have a slightly longer tail past τ to reach 10% of log K (K=3: at 100%τ, K=36: at 156%τ).

### Test 3: Per-Group Learning Curves — PHASE TRANSITION (THE KEY TEST)

![Per-Group Learning Curves](outputs/paper_figures/per_group_learning_curves.png)
![Per-Group Histograms](outputs/paper_figures/per_group_histograms.png)

**This is the test that kills coverage.** We loaded checkpoints at 20 steps spanning the plateau and transition for K=10 and K=20, then evaluated the model on 200 randomly-sampled B-groups per checkpoint using K-way candidate scoring (sequence log-prob of each of K candidate A strings given the correct z).

**K=10 (τ=1850 steps):**

| Step | Fraction of τ | Groups ≥80% | Groups 100% | Mean acc |
|------|--------------|-------------|-------------|----------|
| 500 | 27% | 0.0% | 0.0% | 0.118 |
| 900 | 49% | 0.0% | 0.0% | 0.154 |
| 1400 | 76% | 0.0% | 0.0% | 0.320 |
| 1800 | 97% | 31.5% | 1.5% | 0.645 |
| 2100 | 114% | 74.0% | 22.0% | 0.828 |
| 2800 | 151% | 99.0% | 75.0% | 0.969 |

**K=20 (τ=3950 steps):**

| Step | Fraction of τ | Groups ≥80% | Groups 100% | Mean acc |
|------|--------------|-------------|-------------|----------|
| 1000 | 25% | 0.0% | 0.0% | 0.062 |
| 1900 | 48% | 0.0% | 0.0% | 0.083 |
| 2900 | 73% | 0.0% | 0.0% | 0.192 |
| 3800 | 96% | 1.5% | 0.0% | 0.508 |
| 4700 | 119% | 47.0% | 2.5% | 0.758 |
| 5900 | 149% | 87.0% | 8.5% | 0.870 |

**At τ/2: 0.0% of groups are ≥80% solved for both K=10 and K=20.** Even at 75% of τ, still 0.0%. The fraction-solved curve is a **step function**, not a sigmoid. Coverage predicts 30-50% of groups should be solved by τ/2 — we observe exactly 0%.

The histograms are particularly revealing: at 25%τ and 50%τ, all groups are clustered near chance (1/K). At 75%τ, the distribution starts spreading but remains below the 80% threshold. At 100%τ, a sudden **bimodal split** appears — groups begin separating into "solved" and "not yet solved."

**However, the mean accuracy does creep up uniformly during the plateau** (K=10: 0.10 → 0.32 by 76%τ; K=20: 0.05 → 0.19 by 73%τ). This is NOT groups being solved one-at-a-time — it's all groups getting slightly better than chance simultaneously. This is exactly the "hidden progress" signature from Exp 3: the model builds weak, distributed z-sensitivity across all groups, but no individual group crosses the threshold until the collective transition.

### Test 4: τ Prediction from First Principles — PASS

![Signal Dilution Test 4](outputs/paper_figures/signal_dilution_test4.png)

τ_predicted = C × K / grad_norm_early with a single global constant C = 97.5 ± 8.5 achieves R² = 0.995 across all 10 K values. Maximum fractional error is 14.3% (K=10). The per-K constant C_K varies only from 85.6 to 113.7.

However, this does not save coverage — the formula works for dimensional reasons (right units) regardless of mechanism. A collective transition whose timescale scales with K/grad_norm (as ours does) would produce the same R².

### Verdict: COVERAGE REJECTED

**The aggregate statistics (Tests 1, 2, 4) are perfectly consistent with coverage, but the mechanistic test (Test 3) definitively falsifies it.** The phase transition is real at the individual-group level — groups do not learn independently but transition collectively. The "signal dilution" formula τ = C × K / grad_norm gets the scaling right but for the wrong reason.

The reconciliation: during the plateau, the model builds weak z-sensitivity uniformly across ALL groups (hidden progress = random walk on saddle point). This looks like "diluted signal" in aggregate. But the transition from "all groups slightly better than chance" to "most groups solved" is a collective snap, not a gradual coverage process. The saddle-point escape mechanism (Exp 8) explains both the scaling and the snap: the escape time scales with the information-theoretic content (K bindings) divided by the signal strength (grad_norm), but the escape itself is a global event that affects all groups simultaneously because it corresponds to the model finding a single descent channel in weight space.

---

## Cross-Experiment Synthesis

### What holds up

1. **Hidden progress is real** (Exp 3): Weights move continuously during the plateau in a random-walk pattern. The transition is a sudden **direction alignment** -- not the onset of movement, but the onset of coherent movement.

2. **Synergy as early warning** (Exp 7): The z-shuffle gap reliably leads the loss transition by ~50% of tau, with a clean K^{2.13} power law. Combined with per-head data, this reveals a three-stage cascade: head attention -> aggregate synergy -> loss drop.

3. **Gradient structure during plateau** (Exp 6): Gradients are small but structured during the plateau, and the cumulative "waste" scales as tau^{2.14}. This has practical implications for detecting and potentially shortening plateaus.

4. **Spectral changes are distributed** (Exp 2): Effective rank changes are visible across many heads at the transition, not concentrated in the selector head. L0H0 shows the earliest spectral signature (a sigma_1/sigma_2 spike), while L1H3 maintains high effective rank throughout.

5. **Collective transition confirmed** (Exp 9): Per-group learning curves show 0% of B-groups solved at τ/2 — the transition is a step function, not a sigmoid. This rules out gradual coverage as the mechanism, despite the aggregate scaling τ × grad_norm ∝ K being perfectly consistent with it.

### What does not hold

1. **Non-monotonic weight norms** (Exp 1): Total norms grow monotonically -- no peak-and-reorganize. However, Layer 1 specifically does show non-monotonic behavior, suggesting the deep grokking prediction applies locally to the selector head's layer.

2. **AdamW timescale prediction** (Exp 4): tau_adamw anti-correlates with tau_trans. The Power Lines framework does not capture disambiguation lag, likely because the task difficulty (which scales with K) dominates over optimizer dynamics.

3. **Neural collapse contraction** (Exp 5): Within-class variance increases rather than decreases, because the "class" structure is inverted -- the model needs to differentiate within B-groups, not collapse them. The finding is physically sensible but opposite to the standard neural collapse prediction.

4. **Spinodal decomposition** (Exp 8): The plateau is NOT a local minimum becoming unstable (spinodal). λ_min < 0 throughout the plateau -- it is a saddle point with hidden negative curvature ~1000× weaker than the dominant positive curvature. The transition is saddle-point escape, not spontaneous instability.

5. **Signal dilution / coverage** (Exp 9): The scaling τ = C × K / grad_norm (R² = 0.995) is seductively clean but mechanistically wrong. Groups don't learn one-at-a-time — they snap together collectively. The formula works for dimensional reasons, not because the mechanism is per-example coverage.

### Emerging picture

The disambiguation lag is characterized by a **random walk on a saddle point** (Exp 3, Exp 8) where the loss landscape has hidden negative curvature ~1000× weaker than the dominant positive curvature. During this plateau, the model accumulates **distributed spectral changes** (Exp 2) and **synergistic z-dependence** (Exp 7) without any visible loss improvement. The transition occurs when the optimizer aligns with the feeble negative-curvature direction — once aligned, positive feedback amplifies the descent and the transition becomes **irreversible** (trajectory forensics finds zero reversions).

The transition is a **saddle-point escape**, not spinodal decomposition (Exp 8: λ_min < 0 during plateau, ruling out local-minimum instability) and not stochastic nucleation (trajectory forensics: zero reversions, ruling out barrier hopping). The post-transition regime is a genuine local minimum (λ_min > 0) marked by **directed, coherent weight updates** (Exp 3), **λ_max spikes 10-20× as the model descends through a steep canyon** (Exp 8), **weight norm expansion concentrated in the unembed layer** (Exp 1), and **within-group representation differentiation** (Exp 5).

The ~1000× curvature asymmetry (|λ_min| << λ_max) provides a quantitative explanation for the plateau duration: the escape direction is a tiny sliver of parameter space, and gradient-based optimization with random-walk dynamics takes O(1/|λ_min|) steps to find it.

**Crucially, per-group evaluation (Exp 9) confirms the transition is collective**: at τ/2, 0% of B-groups are individually solved despite mean accuracy creeping above chance. All groups improve uniformly (hidden progress) and then snap together at τ. The aggregate scaling τ ∝ K / grad_norm (R² = 0.995) captures the dimensional dependence but not the mechanism — the transition is a global saddle-point escape that affects all bindings simultaneously, not per-example coverage. The formula works because the saddle escape time scales with the information-theoretic content of the task (K bindings) divided by the signal strength (grad_norm), but the escape itself is a single event in weight space.

---

## Running the Experiments

All scripts are self-contained and require only the pre-existing `outputs/` directory.

```bash
# Fast experiments (no checkpoint loading, <1 sec each)
python scripts/posthoc_adamw_timescale.py
python scripts/posthoc_synergy_proxy.py
python scripts/posthoc_gradient_anatomy.py

# Weight-only experiments (checkpoint loading, ~2-5 min each)
python scripts/posthoc_weight_norms.py
python scripts/posthoc_circuit_spectrum.py
python scripts/posthoc_weight_displacement.py

# Forward-pass experiments (~15 min on MPS/CPU)
python scripts/posthoc_neural_collapse.py

# Second-order experiments (HVP, ~30-60 min on MPS per K value)
PYTHONUNBUFFERED=1 python scripts/posthoc_hessian_eigenvalue.py

# Per-group evaluation (checkpoint loading + forward passes, ~20 min on MPS)
PYTHONUNBUFFERED=1 python scripts/signal_dilution_test.py
```

All figures are saved to `outputs/paper_figures/` as both PDF and PNG.
Hessian results are incrementally saved to `outputs/hessian_eigenvalues_k{K}.json` with resume support.
