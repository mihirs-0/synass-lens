# Adam-channel decomposition — RESULTS (sealed 2026-07-19 22:35)

Final pre-draft experiment. Pre-registered in ADAM_DECOMP_PREREG.md (+ two
logged pre-outcome deviations). Attributes the noise-maintained post-
interpolation trapping to a channel inside AdamW. **After this seal,
experimentation stops and drafting begins.**

## Headline

**Minibatch noise maintains the collapse through Adam's SECOND-MOMENT (v)
channel.** The one clean, power-matched, non-degraded channel attribution is V:
noise routed into v alone (clean first moment) reproduces the trap. The arms
that strip the noise's inflation of v (N_BC, M) both diverge — the inflation is
load-bearing. Direct parameter perturbation (P_AR) also traps but with 4–6×
excess relative power, so it is magnitude-confounded, not a clean attribution.

## Per-arm results (2 streams/arm; T=2000; matched per-step isotropic noise)

| arm | routing | outcome | final CE | Γ | window | verdict |
|-----|---------|---------|----------|-----|--------|---------|
| C0  | clean full grad | ESCAPE @800 | 0.88 | 0 | — | control re-solves ✓ |
| N   | m,v ← (g+ξ) | TRAP (censored) | 3.47 / 3.48 | 1.89 / 2.05 | ref | validity ✓ |
| **V** | m←g, v←(g+ξ)² | **TRAP (censored)** | 3.45 / 3.39 | **0.98 / 0.98** | **IN [0.66,5.91]** | **2nd-moment SUFFICIENT (clean)** |
| N_BC| m←g+ξ, v←(g+ξ)²−σ² | **DIVERGE** | NaN | — | — | degraded (floor F=0.77/0.80) |
| M   | m←g+ξ, v←g² | **DIVERGE** | 8.3e8 | — | — | degraded |
| P_AR| clean moments; θ+=u+ζ, ‖ζ‖=‖rᴺ‖ | TRAP (censored) | 3.61 / 3.51 | 11.2 / 7.68 | OUT (ratio 3.9–5.7) | magnitude-confounded |

Γ = Σ‖r‖²/Σ‖u_sh‖² (perturbation power / clean-shadow-update power). Reference
Γ_N = 1.97; power window = [Γ_N/3, 3Γ_N] = [0.66, 5.91]. Only V lands in-window.

## Which registered branch fired

From the frozen outcome map, the branch that MET its conditions:
- **"V also traps in-window → second-moment distortion independently
  sufficient."** ✓ (V censored, Γ_V/Γ_N = 0.50 ∈ [1/3, 3].)

Branches NOT met (recorded, not claimed):
- "v-bias necessary: N traps, N_BC escapes non-degraded" — N_BC DIVERGED
  (degraded, F=0.77–0.80), did not cleanly escape.
- "first-moment sufficient: M traps, V escapes" — M DIVERGED, V trapped
  (opposite of the pattern).
- "direct perturbation sufficient: P_AR traps, Γ in-window" — P_AR trapped but
  OUT of window (magnitude-confounded).
- "multiple sufficient (≥2 in-window)" — only V is in-window.

The divergent/confounded arms fall in the map's "else → raw mechanism table"
bucket (above). See N_BC_DIVERGENCE.md for the N_BC + M divergence analysis.

## Mechanism

At the collapsed state ‖noise‖ ≈ 2·‖g_full‖ and the per-coordinate clean g² is
SMALLER than the injected noise variance σ² (1.3e-9 < 3.7e-9). Injected into
Adam, the noise dominates the second moment: v ≈ E[(g+ξ)²] ≈ g² + σ² ≈ σ². This
noise-set v rescales the effective step LR/√v so the update stays bounded and
re-scatters the state across the input-blind manifold — the trap.

- **Isolate it (V):** noise only in v, clean m → traps, power-matched. The
  second-moment inflation alone is sufficient.
- **Remove it (N_BC, M):** N_BC de-biases v (subtracts σ²), M uses clean g².
  Both drop v to the tiny clean g² → effective step LR/√v blows up ~5000× →
  divergence. No floor rescues this without re-adding the σ²-scale v that the
  arm exists to remove. The inflation is load-bearing (necessity, degraded).
- **Bypass Adam (P_AR):** perturbing θ directly (clean moments) also traps, but
  needed 4–6× the relative power → cannot cleanly separate "direct perturbation
  works" from "it simply had more power."

## Defensible statement

Continued AdamW training after perfect interpolation of a finite random table
collapses the model to the input-blind marginal, and **minibatch gradient noise
causally maintains that collapse through Adam's second-moment normalization**:
routing the noise into v alone reproduces the trap at matched power (V), while
stripping the noise's inflation of v destabilizes the dynamics into divergence
(N_BC, M). First-moment noise is not independently stabilizing; direct
parameter perturbation is suggestive but power-confounded.

## Provenance

n=1 collapsed checkpoint (dev 0.0125/stream_00 step-16k). Manual AdamW unit-
tested vs torch (rel 9.65e-9). Deviations logged before any channel outcome:
(1) T 3500→2000 + 6-concurrent queue; (2) constant σ → per-step matched
isotropic noise (validity fix; standalone N re-confirmed trap to step 500);
replication 4→2 streams/arm. Results dirs under results/gate0e_adam_decomp/.
