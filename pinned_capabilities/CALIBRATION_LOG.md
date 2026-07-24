# Calibration log

These runs validate instruments and estimate compute. They are not Gate 0 or
Gate 1 evidence and cannot enter confirmatory intervals.

## 2026-07-15 — metric algebra

The initially proposed log-softmax interaction score failed its synthetic
additive-shortcut test because the condition-dependent log partition creates a
spurious interaction. The primary score was replaced by union-centered logits,
which cancel additive `B`-only and `z`-only score functions algebraically. The
test now fails on any regression to log-softmax centering.

## 2026-07-15 — reference smoke

A 1-layer, width-32, `K=3`, 24-`B` system reached the fixed solved endpoint in
140 and 280 steps for two seeds. Its empirical input-blind answer-token loss
was about 3.35 nats, versus `log K = 1.10`; this confirmed that the two
quantities cannot be substituted. The smoke also exposed zero empirical SD in
full-sequence chance accuracy, motivating the absolute 90% expressed-state
floor in protocol v1.2.1.

## 2026-07-15 — erasure smoke

From the tiny solved checkpoint, fixed rates 0.03 through 0.20 retained 100%
expression for the 100-step smoke hold. At 0.5, exact match collapsed but
full-vocabulary CE rose to about 101 nats while `C_int` wandered into its
order-zero interval. Protocol v1.2.3 consequently requires joint return to the
interaction and empirical `q*` bands and treats high-loss branches as
divergence. The tiny system had no valid retained-to-erased bracket.

## 2026-07-15 — production reference timing

The production `K=10`, 1,000-`B`, 4-layer seed-100 calibration reached the
fixed solved endpoint at step 4,400 and completed its 2,000-step hold at step
6,400. Its solved interaction summary was 16.639. This bounded a ten-seed CPU
reference ensemble at roughly 60–90 minutes and justified launching it.

## 2026-07-15 — frozen ten-seed behavioral ruler

Reference seeds 110–119 all reached the dual-probe endpoint and completed the
registered 2,000-step post-solution hold. Endpoint steps ranged from 3,300 to
4,700 and final steps from 5,300 to 6,700. An independent artifact audit
reconstructed every endpoint from the raw 50-step logs, reproduced every
final-window median, checked snapshot metadata, and exactly regenerated the
aggregate bands:

| quantity | frozen value |
|---|---:|
| order-zero `C_int` mean | -0.0009412 |
| order-zero `C_int` SD | 0.0064312 |
| solved `C_int` median | 16.6531944 |
| chance exact-match mean | 0.0 |
| chance exact-match SD | 0.0 |
| empirical `q*` loss mean | 3.5818329 |
| empirical `q*` loss SD | 0.0002120 |

The absolute 90% expressed-state floor is therefore active. The flat-loss
radius is set by its 2% relative floor (0.0716367 nats), rather than by three
reference SDs (0.0006361 nats). No gate seed or gate outcome entered these
quantities.

## 2026-07-15 — augmented-map numerical calibration

An unbalanced Arnoldi solve on the tiny system returned an apparent multiplier
near 69 with a 2.6% eigenpair residual. It was rejected. RMS balancing of the
`(theta,m,v)` coordinate blocks is a similarity transform and therefore leaves
eigenvalues unchanged while improving conditioning. The balanced solve returned
`|lambda|=1.077` with residual 0.080%.

On the production solved seed-100 checkpoint, a reduced-cost rate scan gave:

| learning rate | augmented spectral radius | relative residual |
|---:|---:|---:|
| 0.0001 | 1.0021 | 0.00000093 |
| 0.001 | 1.0745 | 0.000056 |
| 0.01 | 1.0485 | 0.000585 |
| 0.1 | 1.5014 | 0.000074 |

The instantaneous augmented map is slightly expanding even where the observed
longer trajectory is stable, and the estimate is not strictly monotone over
this coarse grid. No unit-multiplier crossing is inferred from calibration.
Gate 0 must compare registered empirical boundaries against the complete local
predictor rather than assuming that `rho(J)=1` is fate-equivalent.

## 2026-07-15 — Gate 0 candidate-grid and input freeze

Before opening the seed-100 fate scan, protocol v1.4.0 froze the candidate
rates `(0.003, 0.006, 0.012, 0.025, 0.05)`, batch size 128, and a hard
calibration rule: the grid must contain a strict nondivergent
retained-to-erased pair or the program stops before seed 0.
The execution uses two spawned workers; worker count is deliberately excluded
from the scientific manifest because every process owns a disjoint,
deterministic rate cell.

The consuming manifests bind the ten-seed reference ensemble SHA-256
`8ab7fd81cbe0ec039152ee9414fb18435a17938297fbd0a983875f6a6400da0d` and its
source-manifest configuration hash
`c30b2ebb1ca78f705a3e4c28e7af0089f6b41864fdec195d5c2cb9f26860fcaf`. The
seed-100 solved snapshot is bound as
`5f1756fdc8a836d18cfdb7f75474771df07e2478dd05b5ea2e9edc205305702f` at step
6,400 with matched seed metadata; its source manifest is bound as
`783a89147ee8cf3ff0fabdb1d6dbdb0f6a62985ea2c385a4b942db3c47653837`. That
calibration snapshot predates
embedded experiment-config metadata, so its config check is explicitly
unavailable. Its source manifest and raw training record confirm the same
production architecture, batch size 128, learning rate 0.001, and exact
training implementation. This exception is calibration-only: all new official
starting snapshots embed and validate their complete experiment config.

The closed-form depth-2 scalar positive control was rerun under v1.4.0. Its
nontrivial multiplier was stable at rates 0.5 and 0.9, exactly `-1` at the
analytic critical rate 1.0, and unstable at 1.1 and 1.5. The unit-circle
classifier therefore recovered the analytic boundary with no numerical
misclassification.

## 2026-07-15 — v1.4.1 pre-outcome adversarial audit

Before the seed-100 boundary scan, an independent read-only audit successfully
forged the v1.4.0 calibration, first-cell `collect`, and Gate 0 `continue`
artifacts, and showed that an empty local aggregate could satisfy the CLI
prerequisite. A second audit showed that edited aggregate rows and resumed
metrics histories were not revalidated at the verdict boundary. No calibration
or gate fate had been run, so these were protocol defects rather than outcome-
conditioned changes.

Version 1.4.1 now re-derives every authorization from bound raw inputs, matches
scan aggregates to sealed child cells, recomputes local certification, binds
transactional checkpoints to exact metrics prefixes, and derives terminal
registry completion. The v1.4.0 scalar-only positive-control artifact is no
longer eligible. Its v1.4.1 replacement must additionally pass the production
augmented-Adam HVP Jacobian versus centered-finite-difference check and the
eigenpair residual check before seed 100 is opened.

## 2026-07-15 — seed-100 erasure calibration stops Gate 0

The frozen v1.4.1 seed-100/batch-128 erasure scan completed all five 8,000-step
holds. Ordered outcomes were:

```text
eta:      0.003      0.006       0.012       0.025    0.05
outcome:  retained   unresolved  unresolved  erased   unresolved
```

The late suppressed-band entry steps at 0.012 and 0.05 were 5,450 and 8,000,
outside the frozen 2,000-step erasure-entry horizon; 0.025 entered at step 100.
Consequently the scan contains no adjacent `retained -> erased` pair. The
content-validated calibration result is therefore `stop_before_gate0`: the
local calibration scan, official seeds 0--4, and Gate 1 are not eligible.
This is a Gate 0 design failure, not evidence for reduction or non-reduction.

The aggregate scan SHA-256 is
`e7c6e49efe6bb6e5da1ef20ffae4a97fd1d05af81adb3dcc0b892f8a9d72d305`;
its manifest SHA-256 is
`953611a6c2f777b946fa781f5d6e8c29e288c6d9ac8fc2675f16d6e242f9d04b`.
Real-artifact validation exposed two administrative validator assumptions:
the branch binds the parsed reference-band content hash rather than the source
JSON byte hash, and unresolved branches may record a late entry. The fixes
only make the already frozen outcome executable. A no-local-scan precheck path
was added so the terminal stop can be written without spending ineligible
local-measurement compute; no threshold, grid, label, or decision rule changed.
The resulting executable precheck artifact SHA-256 is
`e1690a07f891918586ca51e6e3cc79955d5a5cc8a73c61f267b285a26d76cd5c`;
its manifest SHA-256 is
`67b666e89dc90d8921ca6af308b802d5d4391ec6e4f1eeced0c0c6554f8121b0`
and records source commit `b186ef0`.

## 2026-07-15 — Gate 0-E runner smoke (amendment v1.5)

On a 1-layer width-32 system, two escape streams from one snapshot at the
same rate produced different trajectories under fresh data-order seeds; a
from-scratch rerun of one stream was bit-identical to its original metrics
log; and a stream killed at branch step 10,000 of 12,000 resumed to a
continuous complete log. Infrastructure validation only.

## 2026-07-15 — Gate 0-E positive-control calibration

The first control regime (32 streams, 50,000-step horizon, rates 0.006 to
0.012) produced a monotone escape curve but Arrhenius R^2 of 0.60: with a
short horizon most fit cells sat at low escape fractions where the
registered conditional-median estimator is strongly censoring-biased. The
regime was recalibrated to 192 streams and a 250,000-step horizon with rates
0.0055 to 0.0095, placing fit cells at high-but-partial fractions. A first
pass (without 0.0055) gave 3 fit points and R^2 0.9998; the registered
default adds 0.0055 for 4 fit points. Final artifact: monotone curve
(fractions 0.64 to 1.0), Arrhenius R^2 0.968, observed slope 1.07x the
semi-analytic Kramers slope. `passed: true`. Control artifact SHA-256
`5c5bb6e55aebb3f19302ae667010aa8a3a75ebd119a99da9d82d75a1d4cf2749`. The
pass rule (monotone plus R^2 >= 0.9) was never altered; only the simulated
regime moved. A numpy-bool serialization defect in the logistic separation
flag was fixed before any neural curve ran.
