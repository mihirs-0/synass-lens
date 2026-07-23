# AdamW moment-routing diagnostics — frozen-state first pass (2026-07-22)

Code: `pinned_capabilities/cond_update_probe.py`, `phase1_audit.py` (commit
424810d); trajectory runner `reacq_traj_run.py` (commit ff64969). Manual one-step
AdamW re-verified == framework (rel 1.3e-17). State: collapsed fork (dev
0.0125/stream_00 slot_0). All results float32; **float64 replay bit-identical.**

## Result (lead)

**At the collapsed fork, first-moment noise does NOT weaken the escape-direction
coherent drift. It adds ~40,000x transverse diffusion and bends the conditional
mean off-axis. The trapping is diffusion-controlled, not drift-controlled.**

## Central question

> Does first-moment noise merely add diffusion around the same coherent update,
> or does it also alter the conditional mean update?

**Both, and the split is the finding:** along the escape direction the coherent
drift is unchanged; in the full vector the conditional mean is genuinely altered;
and the dominant effect is a huge transverse diffusion.

## Conditional one-step update (K=512 paired draws, CRN, v-only vs both-channel)

| quantity | v-only | both-channel | Δ (both − v) / ratio |
|---|---|---|---|
| μ∥  = ⟨μ_u, e⟩ (e=u_clean/‖u_clean‖) | 0.498 | 0.502 | Δ +0.0038, CI [0.0015, 0.0062]; ratio **1.008** |
| ‖μ_u‖ | 0.534 | 1.372 | +0.838 |
| cos(μ_u, u_clean) | 0.933 | **0.366** | −0.567 |
| V∥ (longitudinal) | 7e-6 | 7e-4 | — |
| **V⊥ (transverse)** | 0.021 | **817** | **×38,992** |
| SNR∥ = |μ∥|/√V∥ | 193 | 18.5 | — |

‖u_clean‖ = 0.744 (both arms' μ∥ ≈ 0.50 ≈ 67% of the clean drift). Off-axis mean
component of the both-channel arm: √(‖μ‖²−μ∥²) ≈ 1.28 (larger than its on-axis
0.50), arising from **m–v coupling** (same ξ in numerator and denominator).

## Classification — primarily Case A, with a Case-B caveat

- **Case A (decisive):** μ∥ comparable (ratio 1.008), V⊥ ×39,000. *The
  trajectory-level recovery separation cannot be reduced to denominator
  attenuation or a smaller conditional mean update.* Supports directional-scatter
  / first-passage control of accessibility.
- **Case-B element (documented):** cos(μ_u, u_clean) drops 0.933→0.366 and ‖μ‖
  grows 2.6x — first-moment noise DOES alter the conditional mean off-axis via
  coupling. The scalar zero-mean-scatter model is not fully adequate.

## Phase 1B — scalar GD-regime model FALSIFIED

corr(u_v, g)=0.010, corr(u_v, g/σ)=0.001, corr(u_clean, g)=0.008. Updates are
**uncorrelated with the current gradient** — dominated by the accumulated
first-moment EMA m̂ (m_new≈0.9m+0.1g at t_eff=22,401), not gradient-following.
σ_i is **~50,000x anisotropic** (median σ=0). The "u ≈ g/(cσ) / inverse-noise-
scale preconditioned GD" *description of the update* is not supported and must be
dropped.

## Phase 1C — numerical audit; Case D FLAGGED but immaterial

grad_zero_frac 0.0049; grad_min_nonzero 5.2e-23; grad_norm 0.032; param_norm
254.9; v_min **0.0** (ε-limited coords); prob_saturated>0.99 **0.20**, prob<0.05
0.80; logit_margin_median −0.138 (22% positive); **no NaN/inf**; loss =
F.cross_entropy fp32. → Case D present (v=0, tiny grads, saturation) **but the
float64 replay is bit-identical**, so it does not affect the conditional-update
result.

## Registered language: survives / weakened / falsified

- **Survives, sharpened:** "the trap is v-damped m-scatter" → the m-scatter is
  **transverse to the escape axis**; equal escape-drift, ~40,000x transverse
  diffusion blocks first passage. One-lever/Γ story untouched.
- **Weakened:** μ_both ≈ μ_v ≈ g/(cσ) holds ONLY as the escape-direction
  projection (μ∥ matches to 0.8%), false for the full mean (cos 0.37 vs 0.93).
  Any "matched coherent drift" claim must carry "along the escape direction."
- **Falsified:** the g/(cσ) / preconditioned-GD *description* of the update
  (EMA-memory-dominated; σ 50,000x anisotropic).

## Compute / provenance

Phase 0: 14.9 ms/draw (2 arms, CPU) → K=512 ≈ 7.6 s; full frozen-state pass
~minutes. Trajectory phases (1D/2/3) running: C0(1300/50), V/N(5200/100),
N_PM(5200/100,c=0.6) from the collapsed fork, ~9.4 GPU-h — for the
amplification-vs-first-passage discrimination (Phase 3) and early/late v-only
conditional states (Phase 2).

## Caveats

n=1 state (collapsed fork) for the headline; trajectory phases add early/late
v-only + matched both-channel states. V⊥ magnitude precision-robust but its
concentration (few ε-limited coords vs broad) is a Phase-2/3 follow-up. Noise =
matched isotropic ξ (registered arms' actual noise); real-minibatch anisotropic
variant not yet run.
