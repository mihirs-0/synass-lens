# Reassessment: the object is post-interpolation optimization dynamics

**Frame (accepted).** The scientific object is no longer capability
persistence. It is: continued AdamW optimization after perfect interpolation
of a random associative mapping. Everything is judged against whether it
teaches something general about that object. Boring names only.

## Claim hierarchy — where we actually are

| Level | Statement | Status |
|---|---|---|
| 1 Descriptive | A transformer interpolates a random table and later loses in-support retrieval under continued training, converging to the label marginal. | **HAVE.** Solid (dev + partial wave, 1a/1b/1c). |
| 2 Comparative | Collapse probability/timing and residence depend reproducibly on lr / batch / weight decay / age, with effect sizes and uncertainty. | **IN PROGRESS.** The 3-seed wave is the evidence; 124/192 sealed. lr-dependence and batch-dependence (from expressed) coming; wd-dependence needs the lambda=0 cell (not yet launched). |
| 3 Causal | Minibatch noise / optimizer state / precision / normalization is *responsible* for the transition. | **BARELY STARTED.** One arm landed (below). The clinching interventions are not run. |
| 4 General | Same mechanism in structured-rule tasks / other architectures. | **NONE.** |

We have a Level 1 result and a Level-2 package finishing. We do not have a
Level 3 result, and my earlier "age-ordered residual structure" language was
a Level-3/memory claim asserted on Level-2 evidence — retracted to a
hypothesis pending the permuted control (see Package 2).

## The professor's publishable bars, audited

**A. Causal noise-maintained trapping** — PARTIAL.
- full batch does not collapse from solved (fullbatch_fork): HAVE.
- full batch *escapes from the collapsed state* (fullbatch_from_stuck,
  escape onset step 2,250): HAVE — this is the "full batch escapes reliably"
  arm and it points to noise-dependence.
- minibatch noise monotonically increases residence time: NOT RUN.
- **injecting matched noise into the full gradient restores trapping**: NOT
  RUN. This is the causal clincher; without it we have correlation.
- Adam-state reset does not explain the whole effect: NOT RUN.
- float64 loss control (kill criterion 1): NOT RUN.

**B. Behaviorally silent residual memory** — NOT SUPPORTED.
- collapsed -> relearn original table (recovery races): HAVE.
- **fresh permuted table, same marginal (the critical control)**: NOT RUN.
- fresh init baseline: HAVE.
- per-example / decoder probe for original-target identity: NOT RUN.
- Verdict: our relearning-speed observations cannot distinguish residual
  memory from generic plasticity/high-norm trainability until the permuted
  control exists. No memory claim is currently licensed.

**C. Aging of memory traces** — NOT SUPPORTED. Requires B plus a systematic
collapse-age ladder. We have only informal young-vs-16k contrast.

**D. Memorization-vs-rule contrast** — NONE. No structured task run.

## Kill criteria — live status

1. float64/simple-optimizer removes it -> NOT TESTED.
2. batch effect vanishes after matching optimizer state AND examples
   processed -> our batch cells match *steps*, not examples; untested as a
   kill.
3. no original-vs-permuted advantage -> the permuted control is missing, so
   the latent-memory language stays parked.
4. structured == random -> untested.
5. checkpoint-specific after the seed wave -> the wave (3 seeds) addresses
   exactly this; finishing it is what retires or confirms this criterion.

## Is the gate wave needed? — decision

Split the wave into what it serves under the reframe:

- **Primary 3-seed escape curves + the lambda=0 cell: KEEP, FINISH.** These
  are the Level 2 comparative evidence (lr- and wd-dependence of collapse
  with cross-seed uncertainty) and they directly retire kill criterion 5.
  ~65% done; finishing is cheap and is the MVP's quantitative backbone.
- **Gate 0-E null-predictor verdict + b512 batch discriminator: VESTIGIAL.**
  They answer the dead object's methods question (does a checkpoint-local
  eigenvalue forecast the boundary). Compute the verdict for free at seal as
  a one-line methods note; do not treat it as a deliverable or wait on it.
- **b32 cells: LOW VALUE, droppable.** The batch question that matters under
  the reframe is Package 1's batch ladder *from the collapsed state*, not
  escape curves from the expressed state. Freeing these slots for the
  decisive packages is the better trade.

**The wave does not reach Level 3 and does not touch the memory question.**
It is necessary for Level 2 and nothing above it. The next compute after it
must be the two decisive packages, not another wave and not the structured
task.

## Next compute, in priority order (decisions attached)

1. **Matched-noise injection from the exact collapsed checkpoint** (Package 1
   clincher). Same weights + optimizer state; full gradient; full gradient +
   noise matched to the B=128 minibatch covariance; Adam-reset arm; float64
   arm.
   - restores trapping -> Level 3, bar A (noise-maintained).
   - does not -> reject the simple noise account; if only Adam-moment
     evolution matters, pivot to adaptive-state dynamics.
   - Highest value: we already have the full-batch-escapes half; this is the
     other half of the only causal statement within reach.
2. **Original-vs-permuted relearning** (Package 2 clincher). From matched
   collapsed checkpoints: relearn original table; relearn a fresh permutation
   with the same marginal; fresh init; across 2-3 collapse ages.
   - original faster than permuted -> bar B (residual memory); age decay ->
     bar C.
   - equal -> generic plasticity; drop all memory language.
3. Structured-rule scope test (bar D) only if 1 or 2 lands.

## Honest MVP, if nothing above succeeds

We characterize a post-interpolation instability in a small transformer
trained to memorize a random associative mapping: continued AdamW training
rapidly eliminates in-support retrieval and drives outputs to the empirical
label marginal; the collapsed state has enlarged weight norms, reduced
LayerNorm gains, an expansive local map, and a batch-dependent escape
profile; full behavioral reacquisition is rare and gradual. That is a
technical note. It becomes a contribution only if experiment 1 or 2 lands.
