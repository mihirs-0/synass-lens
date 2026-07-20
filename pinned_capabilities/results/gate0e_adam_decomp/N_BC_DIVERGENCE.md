# N_BC arm: structural divergence (recorded 2026-07-19)

N_BC (m<-g+xi, v<-(g+xi)^2 - sigma^2, v-floored) DIVERGES to NaN by step ~50,
both streams. max_floor_frac = 0.766 (76% of coords pinned to the floor) ->
DEGRADED by the pre-registered rule (F_t > 0.05).

## Why it is structural, not a code bug
- Frozen floor v_min = 0.1th pct of positive clean bias-corrected v = 1.40e-16.
- At the collapsed state ||g_full|| = 0.032, so clean per-coord g^2 ~ 1.27e-9,
  which is BELOW the injected noise variance sigma^2 ~ 3.72e-9.
- N_BC removes the noise's contribution to v (subtracts sigma^2). What remains
  (clean g^2) is smaller than the noise floor, so v collapses toward v_min.
- Effective step-size LR/sqrt(v) then blows up ~5000x vs the trapping N arm
  (where v is noise-inflated to ~sigma^2). Weights -> inf -> NaN.
- No floor value fixes this: any floor large enough to stabilize (~sigma^2)
  re-introduces the noise-scale second moment that N_BC exists to remove,
  making the arm vacuous. The divergence IS the answer.

## Interpretation (goes in the raw mechanism table, not a clean headline)
Removing the noise's inflation of the second moment destabilizes the trap into
divergence -> the v-channel inflation is load-bearing for numerical stability of
the trapped state (necessity side, DEGRADED). The clean complementary test is
the V arm (m<-g clean, v<-(g+xi)^2): it KEEPS the noise only in v and TRAPS
(CE 3.58, flat) -> second-moment noise inflation is SUFFICIENT to maintain the
collapse. V (sufficiency, clean) carries the v-channel verdict; N_BC (necessity)
corroborates via degradation.

## Handling
Two NaN streams stopped (no information past divergence). Not retuned (floor
change would be post-hoc tuning of the key arm and cannot succeed structurally).
Reported as DEGRADED-divergent per the frozen outcome map's "else" bucket.

---

# M arm: SAME structural divergence (recorded 2026-07-19)

M (m<-g+xi, v<-g^2 clean) also DIVERGES: M_stream00 full_vocab_ce = 9.4e8 by
step 150. Same mechanism as N_BC: M keeps the second moment v CLEAN (= clean g^2,
tiny at collapse ~1.3e-9) while the first moment m carries the noise (~noise
scale). u = m_hat / sqrt(v_hat) then explodes. Any arm that does NOT let the
noise inflate v inherits this: at the collapsed state the clean gradient is
smaller than the noise, so a non-inflated v collapses and the effective step-size
blows up.

Pattern across the moment-decomposition arms:
  v noise-INFLATED  -> N (both moments noisy) TRAP ; V (v-only) TRAP
  v NOT inflated     -> N_BC (v de-biased) DIVERGE ; M (v clean) DIVERGE
=> The load-bearing channel is Adam's SECOND MOMENT: the noise maintains the trap
by inflating v, which rescales the step to stay in the stable range. V isolates
this cleanly (traps). M and N_BC diverge because stripping the v-inflation is
destabilizing, not because the first moment is irrelevant -- the decomposition
cannot cleanly isolate a "first-moment-only" trap at this collapsed state.
M1 not run to completion (structurally identical to M0). Both M streams excluded.
