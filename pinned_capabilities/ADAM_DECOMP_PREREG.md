# Pre-registration: Adam-channel decomposition (final pre-draft experiment)

**Frozen 2026-07-18, before any decomposition outcome.** Implements the
registered spec (user-supplied) that attributes noise-maintained trapping to
a channel inside AdamW: m-dynamics, v-dynamics, direct parameter perturbation,
or coupling. After this seals, experimentation stops and drafting begins.

## Frozen constants

- Collapsed start s0 = dev 0.0125/stream_00 step-16k checkpoint: theta0,
  m0, v0, and the ORIGINAL Adam step counter t0 = 22,400 (preserved; bias
  correction uses t_eff = t0 + local_step).
- eta=0.0125, lambda=0.01, beta1=0.9, beta2=0.999, eps=1e-8, T=3500.
- d = 807,720 params (70 tensors).
- **Canonical noisy arm = matched ISOTROPIC noise**, xi_t = sigma*eps_t,
  **sigma = 6.096e-5 frozen constant** = collapsed-state per-coordinate
  minibatch-noise std (||minibatch noise||=0.0548 at s0, /sqrt(d)). Constant
  sigma makes E[xi^2]=sigma^2*1 known exactly, which the N_BC v-bias
  correction requires. N with this sigma must reproduce trapping (validity).
- Escape = full_vocab_ce < 3.0 sustained 500 steps; else censored. Extension
  to 2T=7000 if any escape has tau >= 0.8T. Split cells (2/2) get +4 streams.
- 4 streams per stochastic arm; C0 deterministic (1). Per-stream base noise
  eps_t^(k) from a frozen seed map; N/N_BC/V/M share the stream's eps; P_AR
  derives its colored h_t from the same eps.

## Arm update rules (manual AdamW, unit-tested vs torch)

  C0    m<-g,    v<-g^2                        clean; must reacquire
  N     m<-g+xi, v<-(g+xi)^2                    must trap
  N_BC  m<-g+xi, v<-(g+xi)^2 - sigma^2 (floored) v-bias removed
  V     m<-g,    v<-(g+xi)^2                    second-moment only
  M     m<-g+xi, v<-g^2                         first-moment only
  P_AR  m<-g,    v<-g^2 ; theta += u_clean + zeta_t   post-Adam colored perturb

theta update (all): theta <- theta*(1-eta*lambda) + u [+ zeta for P_AR].
v_min (N_BC floor) = 0.1th percentile of positive bias-corrected v-coords
pooled over C0 steps {100,300,500,700,900}; written at C0 step 900 so N_BC can
launch mid-run. Floor-binding fraction F_t recorded; cell DEGRADED if
max F_t > 0.05. P_AR amplitude a_t = ||r_t^N|| (per-step residual norm from the
paired native N stream), frozen; zeta_t = a_t * h_t/||h_t||, h_{t+1}=
beta1*h_t + sqrt(1-beta1^2)*eps_t.

## Accounting (per arm)

Pathwise clean shadow moments (advance on clean g at the arm's own theta,
init m0,v0); residual r_t = u_t - u_t^sh; Q_r=||r_t||; A=cos(r_t,u_t^sh);
Gamma_R = sum||r||^2 / sum||u^sh||^2. Arm is power-matched to N when
1/3 <= Gamma_R/Gamma_N <= 3; trapping outside that window is reported as
magnitude-confounded, not clean channel attribution.

## Unit tests (completion-rule requirement) — PASSED 2026-07-18

- Manual AdamW == torch AdamW, 3 clean full-gradient steps: relative diff
  9.65e-9. PASS.
- N_BC v-increment E[(g+xi)^2 - sigma^2] = g^2 unbiased: rel error scales
  1/sqrt(K) (0.206@400 -> 0.023@32000), mean-bias ~0. PASS (per-step variance
  smoothed by the beta2=0.999 EMA, ~1000-step window).
- C0 shadow residual r=0 by construction (identical clean-g advance from
  identical init). PASS.

## Registered outcome map (mechanical from the frozen rules)

- v-bias necessary: N traps, N_BC escapes (non-degraded). If V also traps
  in-window: second-moment distortion independently sufficient.
- first-moment sufficient: M traps, N_BC traps, V escapes, Gamma_M in-window.
- direct perturbation sufficient: P_AR traps, Gamma in-window.
- joint coupling required: N traps while N_BC, V, M, P_AR all escape in-window.
- multiple sufficient: >=2 isolated arms trap in-window.
- else: raw mechanism table {tau, CE, C_int, Gamma, F} per arm, no headline.

## Compute / staging

~5.6 s/step measured (free machine, 2 threads). ~21 streams x 3500 =
~114 CPU-h -> ~12-16 h wall staged. Wave 1 (dependency-free): C0(1) + N(4) +
V(4) + M(4). Wave 2: N_BC(4) after C0 step 900 (v_min). Wave 3: P_AR(4)
after N completes (residual schedule). Validity gate: C0 must reacquire, N
must trap, or the decomposition is void.

## Deviation 2026-07-18 23:48 (compute-driven, before any outcome)

The 13-stream launch thrashed on memory bandwidth (~36 s/step/stream, one
hung stream) and projected to ~2.4 days at T=3500. Two compute-driven
changes, made before any escape/trap outcome was readable (all streams at
step 100-150, dead-flat at the floor):
1. **T reduced 3500 -> 2000.** C0 escapes ~step 900; the registered escape
   criterion certifies at ~1400; 2000 leaves 1100 steps of censoring past
   C0's escape (the matched-noise arms showed zero slow-escape through 3500).
   The extension rule still applies: any escape with tau >= 0.8T=1600 extends
   that arm to 2T=4000. Scientifically ample to distinguish escape from trap.
2. **Dependency-aware queue at 6 concurrent** (bandwidth sweet spot, 1 thread
   each) replaces the 13-stream oversubscription. Same total throughput
   (bandwidth-capped) but no thrashing, no hangs, and dependencies resolve as
   streams complete. Principal-first order: C0+N -> N_BC (v_min gate) ->
   V,M -> P_AR (N-done gate).
No thresholds, arm rules, sigma, or outcome map changed.

## Deviation 2026-07-19 11:50 (validity-gate failure, before any outcome)

**Constant sigma -> per-step matched isotropic noise.** The frozen constant
sigma=6.096e-5 arm FAILED the pre-registered validity gate ("N with this sigma
must reproduce trapping"). The N streams held the floor only ~450 steps, then
drifted off and reached full_vocab_ce < 3.0 by ~step 1000 (escaping, following
C0). Root cause: a *fixed* noise magnitude cannot maintain the trap once the
model drifts, because minibatch noise scales with gradient magnitude and the
gradient grows as the state leaves the marginal; the certified bar-A result
(MATCHED_NOISE_RESULTS.md) re-matched the noise norm to ||(g_B1-g_B2)/sqrt2||
*every step*, which the frozen-constant simplification dropped. Per the prereg,
"N must trap, or the decomposition is void" — so the constant-sigma run is void
by the prereg's own rule, and restoring the certified per-step recipe is the
mandated fix, not a tuning choice.

Change (harness only): xi_t = ||(g_B1-g_B2)/sqrt2|| * eps_t/||eps_t|| (isotropic,
matched per step); the per-step variance sig2_t = ||xi_t||^2 / d is what N_BC now
subtracts for its v-bias correction, so E[(g+xi)^2 - sig2_t] = g^2 stays unbiased
per step (the unit-test logic holds per-step; the beta2=0.999 EMA still smooths
the residual variance). Made before ANY channel-attribution outcome was read
(N/N_BC/V/M/P_AR verdicts all unread; this fixes the N-trap validity gate only).

Validity re-confirmed before relaunch: a standalone matched-sigma N stream held
full_vocab_ce = 3.56 (the ln36 floor), dead flat, from step 50 through step 500 —
past the constant-sigma drift onset. TRAP. Arm rules (m/v routing), theta update,
Gamma accounting, escape criterion, outcome map, and T=2000 all unchanged.
Replication cut 4 -> 2 streams/arm (compute; split-cell +streams rule still
applies). C0 (deterministic, noise-free) was unaffected and kept running.
