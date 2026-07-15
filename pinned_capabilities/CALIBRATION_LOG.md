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
