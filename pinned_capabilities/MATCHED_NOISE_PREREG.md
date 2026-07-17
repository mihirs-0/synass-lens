# Pre-registration: matched-noise experiment

**Frozen 2026-07-17, before any arm's escape time is known.** Purpose: is
minibatch gradient noise the *causal* maintainer of the collapsed plateau, or
is the full-batch-vs-minibatch difference explained by something else (update
count, Adam-moment evolution, preconditioning)? This is the project's first
attempt at a Level-3 causal statement (bar A).

## Design

One collapsed state: dev seed-100, eta=0.0125, stream_00, step-16,000
checkpoint (weights + AdamW m,v,step + data cursor; CE 3.585, C_int 0, EM 0;
SHA c0a527fd...). Every arm reloads this identical complete state, sets
lr=0.0125, keeps wd=0.01, runs T=3,500 optimizer steps single-threaded
(bit-reproducible; FP-threading noise measured at 1.9e-8 << injected noise).
Measured at the collapsed state: ||g_full||=0.032, ||minibatch noise||=0.064
(noise/full = 1.98) — g_full is small near the marginal, noise dominates.

| Arm | per-step gradient | streams | role |
|---|---|---|---|
| A1 full | g_full (mean over all 10,000) | 1 (deterministic) | escape reference (expect escape ~2,250) |
| A2 minibatch | g_B, size 128 | 2 | trapping reference |
| A3 full+emp | g_full + (g_B1 - g_B2)/sqrt2 | 4 | **surgical: full signal + empirical-covariance noise** |
| A4 full+iso | g_full + eps, \|\|eps\|\|=\|\|A3 noise\|\| | 3 | norm-only control (covariance destroyed) |
| A5 reset+emp | A3 with AdamW m,v,step zeroed at fork | 4 | optimizer-memory control |

A1 vs A3 is update-count-matched and signal-matched: the only difference is
the injected empirical-covariance noise. A3 vs A4 isolates covariance
structure from magnitude. A5 vs A3 isolates optimizer memory.

## Escape criterion (registered, reused from the escape-time analysis)

`T_escape` = first optimizer step with full_vocab_ce < 3.0 sustained for the
next 500 logged steps; else right-censored at T=3,500. Reported by optimizer
steps AND examples processed (arms differ in examples/step). Primary
comparison is the escape/residence distribution across streams, not final loss.

## Pre-registered interpretations (frozen before results)

| Observation | Interpretation |
|---|---|
| A1 escapes, A3 residence >> A1 (traps or much delayed) | **Noise causally maintains trapping.** Level-3, bar A. |
| A3 traps AND A4 traps | Noise **magnitude** suffices; covariance not required. |
| A3 traps AND A4 escapes (like A1) | **Covariance/direction structure** of minibatch noise matters. |
| Neither A3 nor A4 traps (both escape like A1) | Additive gradient noise does NOT explain the full/minibatch gap; pivot to update-count or moment-cadence. |
| A5 (Adam reset) escapes where A3 traps | **Optimizer memory** (m,v) is central to the trap. |
| A2 does not trap | Setup/implementation failure — halt and debug before interpreting A3. |
| Streams vary wildly within an arm | Insufficient replication; add streams before any mechanism claim. |

## Kill/exit

If A3 escapes indistinguishably from A1 across all 4 streams, the
noise-maintained-trapping hypothesis (bar A) is rejected for this system, and
the honest claim reverts to the MVP technical note. This experiment can end
the causal question in either direction; it is not "interesting either way"
in the loose sense — each outcome eliminates specific hypotheses.

## Not in this package (deliberate)

Age ladder (premature — establish the effect at one age first); float64
control (kill criterion 1, run only if a mechanism survives); structured-rule
task (scope test, not launched). The original-vs-permuted relearning
experiment is the authorized **second** package, after this one reports.
