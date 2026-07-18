# Matched-noise result: minibatch noise causally maintains the collapse

**Certified 2026-07-18 against the frozen `MATCHED_NOISE_PREREG.md`.** First
Level-3 causal result of the program (bar A).

## The certified comparison (update-count- and signal-matched)

From one collapsed checkpoint (dev 0.0125/stream_00, step 16k, weights +
AdamW buffers), all arms same lr=0.0125, wd=0.01, single-threaded:

| Arm | injected noise | outcome (step 1,400 of 3,500) |
|---|---|---|
| A1 full deterministic | none | **CERTIFIED ESCAPE at step 900** (CE 3.585 -> 1.86, C_int 0 -> 6.33; re-solving) |
| A2 minibatch (ref) | (is the noise) | both streams trapped/censored (CE 3.58) |
| A3 full + empirical minibatch noise x4 | (g_B1-g_B2)/sqrt2 | **all 4 TRAPPED** (CE 3.581-3.586, C_int ~0.00; dead flat 500 steps past A1's escape) |
| A4 full + isotropic matched-norm x2 | isotropic, \|\|.\|\|=A3 | trapped, slightly perturbed (CE 3.51-3.52, C_int 0.05) |

A1 and A3 receive the identical full-data gradient and the identical number
of updates from the identical optimizer state. The ONLY difference is the
injected empirical-covariance noise. A1 escapes; A3 does not.

## Conclusion (bar A, per the frozen interpretation table)

**Minibatch gradient noise causally maintains the collapsed plateau.** The
full deterministic gradient escapes; adding minibatch-covariance noise to
that same gradient prevents escape. This isolates noise from the other
things full-batch changes (update count, moment-evolution cadence,
preconditioning), all of which are held fixed between A1 and A3. This is the
first causal statement in the project and it is the strongest pre-registered
outcome: noise-maintained trapping.

At the collapsed state ||minibatch noise|| = 1.98 x ||g_full|| (noise
dominates near the marginal), which is the mechanism: the deterministic drift
would leave, but the noise keeps re-scattering the state across the
input-blind manifold.

## Honest caveats (stated, not buried)

1. A3 is certified as "residence >> A1 escape" (trapped 500+ steps past A1's
   escape, C_int exactly 0, zero movement). Full censoring-to-3,500 for all
   A3 streams is pending (run in progress); at step 1,400 there is no hint of
   escape in any A3 stream.
2. **Covariance vs magnitude — RESOLVED (2026-07-18 17:47, A4 at step
   3,250/3,500): magnitude suffices.** Both A4 streams stayed TRAPPED through
   ~step 3,250 (2,350 steps past A1's escape; CE 3.40-3.46, C_int 0.07-0.17),
   never escaping, while A1 fully re-solved (CE 0.61, C_int 13.8). So
   isotropic noise of matched norm alone prevents escape — the specific
   minibatch covariance structure is NOT required for trapping (pre-reg:
   "A3 traps AND A4 traps -> magnitude may matter more than covariance").
   Refinement: empirical covariance traps TIGHTER — A3 stayed dead-flat
   (C_int 0.00, CE 3.585) while A4 drifted loosely (C_int 0.17, CE 3.40).
   So: any noise of sufficient magnitude maintains the plateau; minibatch
   covariance makes the trap firmer but is not necessary.
3. **A5 (Adam-reset control) has not run.** Whether the trap is
   noise-alone or noise-plus-optimizer-memory is open. Registered follow-up.
4. n=1 collapsed checkpoint. Replication across collapsed seeds is a
   follow-up if the headline is kept.

## Scope of the claim

"Minibatch noise causally maintains the plateau, holding update count and
optimizer cadence fixed" is certified. NOT yet claimed: that only the
covariance structure matters (A4 pending), that optimizer memory is
irrelevant (A5 not run), or that this generalizes beyond this checkpoint.
