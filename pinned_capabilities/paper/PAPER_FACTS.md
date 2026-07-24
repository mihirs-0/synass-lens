# PAPER_FACTS — single source of truth for the noise-routing draft (2026-07-23)

Every number here is verified from saved artifacts. Section-drafting agents MUST
draw only from this file + the language law. Do not invent numbers or citations.

## The three contributions (order: lead N2, N1 methods payload, N3 spine)
- **N2 (lead — the dissociation):** gradient noise can deny recovery without
  weakening the drift toward it. From a collapsed checkpoint, at the probed
  states the escape-direction *pull* (conditional-mean update / true gradient)
  is intact while transverse diffusion runs 4–7 orders of magnitude above it.
  A measured counterexample to "noise = escape agent."
- **N1 (methods payload — pinning):** where noise is routed pins the realized
  update-space noise power; the injected magnitude is an x-axis the model never
  feels near small-gradient states. Corollary (method): gradient-space power
  matching is not an adequate control — report realized update-space power.
- **N3 (spine — trust):** frozen (θ,m,v) fork; frozen criteria; ≥-horizon
  censoring; Γ shadow accounting; manual-AdamW==framework replay verification
  (rel 1.3e-17); float32==float64.

## Core numbers

### The fork (Fig 1)
- Identical optimizer-complete state (θ,m,v,step counter, bias-correction).
- Clean full-gradient replay re-solves in **~800 steps** (CPU decomposition;
  on MPS the metastable escape is slower — device-sensitive, note it; the
  RELATIVE comparison across arms uses one harness).
- Measured-noise replay (both-channel, matched to minibatch magnitude): **never
  re-solves within 16,000 steps, both seeds censored** (≥20× slowdown, a LOWER
  BOUND). final CE 3.47 / 3.54 (floor ≈ 3.58 = ln 36).
- Minibatch and v-only also shown on the fork panel.

### Realized-power pinning table (Fig 2) — realized Γ vs injected dial
Γ = Σ‖u−u_sh‖²/Σ‖u_sh‖² (shadow-update power ratio), windowed-median over steps
100–500.
| routing | injected dial | realized Γ |
|---|---|---|
| v-only (clean m, noisy v) | s=1.0 / 2 / 4 | 0.98 / 0.975 / 0.984  (**≈1, dial-invariant**) |
| both-channel | c=0.5 / 0.6 / 0.7 / 1.0 | 1.99 / 1.86 / 2.15 / 1.89 (**≈2, dial-invariant**) |
| m-only, clean v | s=0.02 / 0.05 / 0.1 / 0.25 | 251 / 2,618 / 7,773 / 1.0e7 (**≥250, exploding**) |
- **Outcome-invariance:** both-channel at reduced amplitude c=0.6 (Γ≈1.86) still
  censors ≥16k — the trap tracks pinned felt-noise, not injected amplitude.
- **Γ≈2 anomaly (A3, main text as measurement):** both-channel measured Γ=1.89
  vs scalar prediction 1+a²=1.006 (a=0.079). From logs: q=‖u_both‖/‖u_clean‖=0.70
  (both-channel update SMALLER than clean), ρ=cos(u_both,u_clean)=−0.28
  (anti-correlated), ν=‖r‖/‖u_clean‖=1.38 (large residual), A_align=cos(r,u_clean)
  =−0.87. Γ=1+q²−2qρ=1.894 (=measured). **Appendix (exploratory):** sparsity
  account — clean update participation ratio 2,576 = 0.32% of D=807,720; noise
  dominates the ~99.7% near-zero coords where Adam per-coord normalization
  amplifies it; β₁ formula with independence assumptions stated.

### The dissociation measurement (Fig 3) — conditional one-step update
K=512 paired draws (common random numbers), v-only vs both-channel; escape
direction e = u_clean/‖u_clean‖. float32==float64 (bit-identical). E[u] estimated
by Monte-Carlo averaging of the draws (NOT by-construction) — MC errors reported.
| quantity | v-only | both-channel |
|---|---|---|
| μ∥ = ⟨μ_u,e⟩ | 0.498 ± 1.1e-4 | 0.502 ± 1.2e-3 |
| Δμ∥ | — | +0.0038, bootstrap 95% CI [0.0015, 0.0062] |
| ‖μ_u‖ | 0.534 | 1.372 |
| cos(μ_u, u_clean) | 0.933 | 0.366 |
| V⊥ (transverse) | 0.021 | 817 (**×39,000**) |
| SNR∥ | 193 | 18.5 |
Along-trajectory (Phase 2, K=256, v-only states): μ∥ v-only == both (0.375/0.375
at step 1,000; 0.223/0.223 at 3,700); V⊥ ratio both/v = 4.3e6 (step 1,000) →
5.3e7 (step 3,700, pre-onset). Drift matched; diffusion 4–7 orders above.

### Mechanism (Fig 3, from Phase A)
- **A1 own-axis (amplification, onset clause):** each recovering arm on its OWN
  escape direction shows systematic pre-onset growth of s_t=⟨−g,e_own⟩ —
  clean ×145 (P_T 741), v-only ×106 (P_T 166). Escape directions internally
  stable (pairwise cos 0.97). NOTE: escape directions differ ACROSS arms
  (v-only recovers nearly orthogonal to clean; on the SHARED clean axis v-only's
  P_T is only 9.9 vs its own-axis 166) — a single held-out direction does not
  transfer; state this.
- **A2 burial (mechanism clause):** at trapped both-channel states the true
  gradient norm is nonzero and GROWS (‖g‖ 0.048 → 0.583 across 0.5k–16k) with a
  small consistently-signed escape-direction pull — NOT drift-dead. "Diffusion
  buries an intact pull" (targeted; not relocation to drift-dead regions).
- Combined: recovery = amplification of the escape-direction pull along the
  arm's own axis; trapping = the pull is intact but transverse diffusion
  prevents its amplification.

### The 2×2 (Fig 4) — onset step, η=0.0125, single seed per cell
|  | clean | v-noise (s=1.0) |
|---|---|---|
| collapsed | 800 | 4,600 |
| fresh (original table, A6 landed) | 2,200 | **8,400** |
- Fresh+clean learns at 0.0125 (onset 2,200) — the rate is not the obstruction.
- Fresh+v-noise on the ORIGINAL table = 8,400 (escaped, not censored) >
  collapsed+v-noise 4,600 → head start PRESERVED under noise on the original
  table; the sibling-table number (4,000) that suggested inversion was a table
  artifact. **Claim: "advantage eliminated (possibly inverted)," present
  targeted-erasure and common-onset-floor with EQUAL weight; NO standalone
  inversion claim.** Dose–response inset: v-only onset vs noise scale s
  (collapsed): s=0.25→3,400; 0.5→3,600; 0.75→4,600; 1.0→4,600 (saturating).

## Construction checks (label as such in-line, NOT findings)
v-inflation of the second moment; flat Γ(c) as algebra (Kingma–Ba scale
invariance); Γ_v≈1; m-only explosion (clean-v denominator); generic
noise-impedes-learning.

## Verified citations (all confirmed at primary source; use these exact IDs)
- Adam — Kingma & Ba, arXiv:1412.6980, ICLR 2015.
- Dissecting Adam (sign/variance) — Balles & Hennig, arXiv:1705.07774, ICML 2018.
- On the Convergence of Adam and Beyond (AMSGrad) — Reddi, Kale, Kumar, ICLR 2018 (arXiv:1904.09237).
- Noise Is Not the Main Factor…Sign Descent Might Be — Kunstner, Chen, Lavington, Schmidt, arXiv:2304.13960, ICLR 2023.
- DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction) — Tang, Shpilevskiy, Lécuyer, arXiv:2312.14334, AAAI 2024.
- Large LMs Can Be Strong DP Learners (DP-Adam fine-tuning) — Li, Tramèr, Liang, Hashimoto, arXiv:2110.05679, ICLR 2022.
- How SGD Selects the Global Minima…Dynamical Stability — Wu, Ma, E, NeurIPS 2018.
- The Anisotropic Noise in SGD — Zhu, Wu, Yu, Wu, Ma, arXiv:1803.00195, ICML 2019.
- Stochastic Collapse — Chen, Kunin, Yamamura, Ganguli, arXiv:2306.04251, NeurIPS 2023.
- Parameter Symmetry and Noise Equilibrium of SGD — Ziyin, Wang, Li, Wu, arXiv:2402.07193, NeurIPS 2024.
- How to Escape Saddle Points Efficiently — Jin, Ge, Netrapalli, Kakade, Jordan, arXiv:1703.00887, ICML 2017.
- Interplay of Optimization and Generalization (Fisher noise) — Wen et al., arXiv:1902.08234, AISTATS 2020.
- Edge of Stochastic Stability — Andreyev & Beneventano, arXiv:2412.20553, 2024.
- Late-Stage Generalization Collapse in Grokking (anti-grokking) — Prakash & Martin, arXiv:2602.02859, 2026.
- Loss of plasticity in deep continual learning — Dohare et al., Nature 2024.
- Position: A Science of AI Must Study Training Dynamics — Biderman et al., arXiv:2606.06533, ICML 2026.

## Corrections ledger (appendix, in full; framed as audit trail)
permanence retracted; erasure-to-scratch retracted; "v explains everything"
retracted (N_PM/M_T infeasible to power-match — Adam coupling); the crawl/
round-trip overcount retracted; ignition demoted by its own registered test;
the corridor/reservoir demoted; the registered horizon deviation (T 3500→2000);
the MPS-vs-CPU escape-time device sensitivity.
