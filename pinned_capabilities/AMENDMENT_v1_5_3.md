# Amendment v1.5.3 — record corrections and discriminator hardening (frozen, still blind)

**Provenance.** Written after a second external-review pass, while the dev
curve stood at 10 of 56 streams with no aggregate written. No dev statistic,
gate fate, batch cell, or lambda=0 outcome had been computed or read. After
this amendment the protocol is declared frozen against further discretionary
change: only demonstrated implementation defects may be repaired, and each
repair must be logged as such.

## 1. Literature record correction (supersedes v1.5.2 section 1 in part)

Two statements frozen in v1.5.2 were factually false and are corrected here
after direct primary-source verification:

1. **Khanh is real, including its section 8.**
   *Cross-Trajectory Chimera Interventions Reveal Dissociable Roles of
   Weight Magnitude and Direction in Grokking*, Truong Xuan Khanh,
   arXiv:2607.06628, submitted 2026-07-07. Verified by reading the PDF:
   section 8 compares three continuation-optimizer conditions (freshly
   reset, recipient moments, donor moments); the modular-addition threshold
   gap is 0.34 under reset versus 0.25 under recipient and donor moments;
   per-pair thresholds shift 0.03-0.04 (about twice the bisection
   resolution) under moment transplantation; the paper concludes circuit
   identity is "a property of theta, with optimizer moments contributing a
   secondary, non-decisive perturbation." The v1.5.2 claim that these
   numbers were "retracted as fabricated" was itself the error: the peer
   thread's retraction was a false confession about real numbers.
2. **Feature Lottery is real.** *Feature Lottery? A Bifurcation Theory of
   Concept Emergence*, arXiv:2605.24057, submitted 2026-05-22. Verified at
   the abstract: representation onset as a supercritical pitchfork
   bifurcation driven by the loss Hessian; a label-free beta/beta_c phase
   coordinate spanning Pythia, CIFAR SSL, and modular-arithmetic grokking;
   explicit use as an early-warning instrument. This is adjacent prior art
   both for grokking metastability and for this program's Gate 3
   early-warning ambitions, whose bar it raises.

**Method lesson, registered:** two failed keyword searches were treated as
evidence of nonexistence. The standard from now on is direct arXiv-ID
resolution before any citation is declared unverifiable, and no citation may
be called fabricated without a failed direct-ID lookup.

**Novelty consequence.** Khanh section 8 establishes that Adam moments are
secondary and non-decisive for circuit-identity transfer in a full-batch,
effectively absorbing grokking regime. The open question this program tests
is therefore narrower and sharper than v1.5.2 stated: whether structured
optimizer state moves the *committor* of a reversible capability transition
near a stochastic separatrix — a regime where a secondary perturbation can
legitimately change fate — beyond what step scale, queued displacement, and
local reconstruction explain. If Gate 1 finds moments secondary here too,
the honest description is a cross-regime extension of Khanh's negative, and
it will be reported as such.

## 2. Diffusion statistic upgraded (implements the reviewer's correction)

The v1.5.2 statistic was the trace of a lag-zero update covariance; under
Adam's serially correlated updates its batch scaling need not match the
scaling of the effective diffusion that governs displacement and escape.

Registered replacement, computed inside the same refresh segments:

1. The lag-zero quantity is renamed `update_variance_power` and demoted to
   a reported proxy.
2. **Block-displacement diffusion.** For block lengths L in {1, 4, 16, 64},
   over non-overlapping blocks of consecutive steps,
   `D_L = Var(sum of Delta-theta over a block) / L`. The full ladder is
   reported (its L-dependence exposes correlation time). The decision
   statistic is `ln D_64`, requiring at least 8 complete blocks.
3. **Uncertainty gate.** Each cell stores per-block scalars sufficient for
   a leave-one-block-out jackknife SE of `ln D_64`. The pooled batch slope
   `d ln D_64 / d ln B` is resampled parametrically (2,000 draws, fixed
   seed) from the per-cell jackknife SEs; a noise direction is declared
   only when the 95% interval lies wholly beyond +/-0.15. Uncertainty can
   turn `decisive` into `non_discriminating`; it can never create
   decisiveness.
4. **v-direction magnitude floor.** A v-conditioned direction exists only
   when every seed's predicted shift satisfies
   `|ln(eta50_pred(B)/eta50_pred(128))| >= ln(1.10)` with consistent,
   antisymmetric signs; smaller predicted shifts are numerical noise and
   yield no direction.
5. The global statistic is an **empirical global-noise proxy**: variance
   projected onto preregistered coordinates (the C_int gradient, the
   dominant augmented mode, per-group power) may be reported as
   diagnostics but is never decision-bearing at Gate 0-E.

All other discriminator semantics (decisive requires opposing non-flat
directions; non-discriminating makes null_wins unsatisfiable) are unchanged
from v1.5.2.

## 3. Gate 1 clarifications (registered before any Gate 1 run)

1. **The net-kick control's inference is one-way.** Reproduction of the
   fate shift by directly applying Delta-theta_K with reset moments kills
   the strong non-weight-memory claim. Non-reproduction does NOT vindicate
   it, because one immediate net vector is not a path of moment-driven
   updates through changing geometry.
2. **Reset-at-theta_K persistence family** (added): run the transplanted
   moments for K steps; at the resulting theta_K, either reset all moments
   or retain them, and continue identically. Persistence of the fate
   difference after reset means the first K steps wrote it into weights;
   disappearance means non-weight state remains causally active beyond K.
   Run on the same K ladder {1, 5, 25, 125}; this is the clean version of
   the washout reading.
3. **Dual analysis, never collapsed.** The surgery reports both the
   within-weight calibrated analysis (each weight source near its own
   sensitivity point; maximizes power for the committor difference dq(W))
   and a common-rate factorial (all four weight-by-moment cells under one
   shared challenge rate or small common grid; preserves the interaction's
   interpretation). The categorical verdict may cite either only with its
   confound stated; the two analyses answer different questions and are
   reported side by side.

## 4. Affirmative-mechanism language

"Arrhenius-like" remains the strongest permitted phrase until noise and
deterministic drift have been varied independently. The registered
affirmative-evidence set for any activated-escape claim is: approximately
exponential dwell-time tails; mutually coherent forward and reverse rates;
rate response to independently injected or varied noise amplitude; and
stability of the inferred barrier under fixed drift and decay (the
orthogonalized eta_g/rho design). A good Arrhenius fit alone is
compatibility, not identification.

## 5. Amendment freeze

v1.5.3 closes the discretionary amendment window. Subsequent changes
require a demonstrated implementation defect (code behaving contrary to
this frozen text) and are logged as defect repairs, not design changes.
