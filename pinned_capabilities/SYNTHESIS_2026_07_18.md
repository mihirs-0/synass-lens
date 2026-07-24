# Synthesis — post-interpolation collapse program (2026-07-15 to 07-18)

## Object

Reframed 07-17 from "capability persistence" to: **continued AdamW training
after a small transformer has perfectly interpolated a finite random
associative table**. The task (10,000 (B,z)->A pairs, each answer a random
string unique to one example; z indexes a shared slot, A values independent
per B) affords no rule and no out-of-support generalization — held-out is
chance by construction (verified: 10,000 pairs / 10,000 unique targets).
"Solved" = memorization of the table; all evaluation is in-support
(probe_fraction=0). Every claim is now an optimizer-dynamics claim, not a
capability-loss claim. Field label: training-dynamics stability/control.

## The phenomenon (Level 1-2, quantitative)

Fork a memorized checkpoint (weights + AdamW buffers + RNG), vary only future
data order, hold 16,000 steps at a fixed rate. Outcome:

- **Collapse is universal and fast.** At every rate that destroys, the model
  leaves the solved state within ~100 steps and sits at the input-blind
  answer marginal (per-position entropy 3.58 nats ~ ln 36; KL 0.005 to the
  data marginal vs 0.108 to uniform; zero input information at the output).
- **A sharp, seed-stable permanence boundary.** Across three independently
  trained checkpoints, eta50 = 0.00359 / 0.00432 / 0.00413 (CV ~9%). Below
  ~0.004 the model re-learns; the collapse *sticks* only above it.
- **Collapse is a WINDOW, not a monotone threshold.** ~100% clean
  collapse-to-marginal in eta in [0.008, 0.032]; below ~0.004 it re-learns;
  at 0.05 it mostly DIVERGES (7/8, 4/8, 4/8 diverged) rather than collapsing.
- **Not weight-decay driven.** lambda=0: still 4/4 collapse at 0.008/0.0125/
  0.02; decay only modulates the boundary near threshold (0.005: 4/8->1/4).
  Kills the Ersoy L2-metastability reframe.
- **Fresh-init control:** fresh models at 0.005/0.0125 stay on the plateau
  where forked ones do too; the boundary is the task's max learnable rate,
  not a property of the collapsed state alone. (Young collapsed states escape
  faster than fresh init; aged ones slower — recorded as hypothesis, NOT a
  memory claim, pending the permuted control.)

## Anatomy of the collapsed state (descriptive)

- **1a weight norms:** stuck/solved global L2 = 2.45x (weights GREW; decay
  shrinkage e^-2=0.135x refuted). LayerNorm gains 0.45x (halved).
- **1b local stability:** augmented AdamW spectral radius at the stuck state
  1.35-8.15 across rates (1.93 at its own rate) vs solved 1.05-1.31 -
  super-unit, certified, yet the state never moves. The most immovable state
  carries the largest multipliers; checkpoint-local linearization is
  category-wrong for trapping here (noise-maintained stationary distribution,
  not a locally stable fixed point).
- **1c output:** exactly the data marginal, per example, zero input info.

## What is necessary (causal, established this slew)

- **The kick needs BOTH noise and an elevated rate.** Full-batch fork at
  0.0125 from solved: NO collapse. Fork at the training rate 0.001 (minibatch)
  x2: NO collapse. Minibatch fork at >=0.003: 8/8 collapse by ~100 steps. So
  collapse requires minibatch noise AND rate above a threshold between 0.001
  and 0.003; it is neither data-order shock nor deterministic step-size
  instability.
- **THE causal result (bar A, certified against frozen pre-registration).**
  From one collapsed checkpoint, 5 arms, identical lr/wd/optimizer-state,
  update-count- and signal-matched:
    A1 full deterministic gradient -> CERTIFIED ESCAPE at step 900, fully
       re-solves (final C_int 18.9).
    A2 minibatch -> trapped (reference).
    A3 full gradient + empirical minibatch-covariance noise (g_B1-g_B2)/sqrt2,
       x4 -> ALL TRAPPED to 3,500, C_int 0.00 (dead flat).
    A4 full gradient + isotropic matched-norm noise, x2 -> TRAPPED to 3,500.
  A1 and A3 differ ONLY by the injected noise (same full signal, same update
  count, same optimizer state). A1 leaves; A3 does not. **Minibatch noise
  causally maintains the collapse.** Mechanism: at the collapsed state
  ||noise|| = 1.98x ||g_full|| — the deterministic drift would leave, the
  noise re-scatters the state across the input-blind manifold.
  **Covariance vs magnitude RESOLVED: magnitude suffices** (isotropic noise
  also traps); empirical covariance traps tighter (A3 dead-flat vs A4 loose
  drift) but is not required.

## Epistemic corrections made this slew (uncertainty shrank)

- **Round-trip overcount RETRACTED.** An interim heuristic (final C_int>8.3)
  conflated near-expressed with re-solved; recomputed via the frozen
  is_expressed (EM>=0.90): of 115 certified-suppression streams only 3 ever
  returned to full expression, all unsustained, on 2 of 4 seeds. "Recovery"
  is graded re-learning (info -> interaction -> behavior, EM lags), not a
  distinct spontaneous-return phenomenon. Verified the error never entered
  pipeline code.
- **Task-structure concern CONCEDED.** The task is memorization; "capability/
  solved/relearning" overclaimed and are relabeled.
- **Spontaneous recovery / held capability / reversibility / optimizer-
  memory-capability:** all previously retracted; this slew tightened rather
  than rescued.

## Claim ladder — current standing

- Level 1 descriptive: HAVE (collapse to marginal).
- Level 2 comparative: HAVE (seed-stable boundary, window, wd-independence,
  effect sizes across 3 seeds).
- Level 3 causal: HAVE bar A (noise-maintained trapping) + magnitude-suffices.
- Level 4 general: NONE.

## Open / not done

- A5 Adam-reset control (noise-alone vs noise+optimizer-memory): NOT RUN.
- Package 2 original-vs-permuted relearning (bar B, table-specific residual
  memory): NOT RUN — the random table's unique test; the "age-ordered
  residual structure" hypothesis stands or falls here.
- n=1 collapsed checkpoint for the causal result; replication across seeds
  pending if the headline is kept.
- Structured-rule scope test (bar D): NOT RUN.

## Defensible statement today

Continued AdamW training drives a small transformer from perfect
interpolation of a finite random table to the input-blind marginal predictor;
entry is noise-gated and rate-gated (~100 steps, boundary eta ~0.004, seed-
stable, weight-decay-independent); the collapsed state has enlarged weights,
halved LN gains, and a super-unit-but-stationary local map; and **minibatch
gradient noise causally maintains the collapse — remove it (full gradient)
and the model re-solves; add matched-magnitude noise and it stays trapped.**
Whether table-specific memory survives the collapse, and whether optimizer
memory is co-responsible, are the two open causal questions.
