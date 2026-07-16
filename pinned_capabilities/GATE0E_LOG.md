# Gate 0-E run log

Chronological record of Branch A execution. Artifacts live outside git and
are bound here by SHA-256. No decision rule appears here; rules live in the
frozen amendments.

## 2026-07-16 06:40 — dev curve sealed (seed 100, batch 128)

56/56 streams, 16,000-step holds. `curve.json` SHA-256 recorded below.

| eta | erased | retained | unresolved | diverged | P (primary) | P (sensitivity) | median tau |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 0 | 8 | 0 | 0 | 0.000 | 0.000 | — |
| 0.005 | 1 | 1 | 6 | 0 | 0.125 | 0.875 | 100 |
| 0.008 | 6 | 0 | 2 | 0 | 0.750 | 1.000 | 100 |
| 0.0125 | 8 | 0 | 0 | 0 | 1.000 | 1.000 | 100 |
| 0.02 | 8 | 0 | 0 | 0 | 1.000 | 1.000 | 100 |
| 0.032 | 8 | 0 | 0 | 0 | 1.000 | 1.000 | 200 |
| 0.05 | 6 | 0 | 0 | 2 | 1.000 | 1.000 | 850 |

- Curve VALID (7 eligible rates; 0.05 at 25% divergence, below the 50%
  unstable bar). Monotone, no flags.
- eta50 = 0.00672, stream-bootstrap 95% CI [0.00559, 0.00753]
  (1,190/2,000 valid replicates). Logistic slope 7.17 in ln eta.
- **Batch-test rates (frozen rule, dev P nearest 0.25/0.75): 0.005 and
  0.008.**
- The transition zone is narrow (about 1.6x in eta) but genuinely mixed
  inside: at 0.005 the eight fates split 1 erased / 1 retained / 6
  unresolved. The primary-versus-sensitivity gap concentrates exactly there
  (0.125 vs 0.875), so the registered unresolved convention is
  decision-relevant and the sensitivity table ships with every report.
- Erasure is immediate-or-never within the hold: median tau is about 100
  steps at every erasing rate up to 0.02 (200 at 0.032; 850 at 0.05 where
  the hot shelf delays joint certification). The Arrhenius secondary is
  vacuous here (2 usable points). Fate appears to be decided within the
  first few hundred steps — a first-passage structure closer to
  kick-decides-it than to slow diffusive barrier crossing.

**Dwell audit, 56 trajectories:** recrossings occur ONLY at eta <= 0.005 —
every stream at 0.003 and 0.005 shows 1-2 expressed/suppressed round trips
(transient collapses and recoveries at rates the curve calls retained),
while all 40 streams at eta >= 0.008 are strictly one-way. Graze fraction
0.92. Basin language remains unearned at the behavioral-coordinate level
(interpretation registry rules stay in force); the reacquisition phenomenon
is ubiquitous below the boundary and absent above it.

**Dev null scan (seed 100, B=128):** all 7 rates certified; augmented
radius monotone 1.053 -> 1.307 across the grid. Moment refresh preserved
the solved state (theta drift 1.1%, loss ~3e-4). Diffusion ladder
D_1 2.5e-6, D_4 8.5e-6, D_16 2.1e-5, D_64 2.0e-5 — positive update
correlation saturating near L=16-64, validating the block-diffusion design;
15 decision blocks.

Execution note: the first pool was stopped by the runtime at ~00:50 after
about 4 hours (background-task lifetime); detached workers survived and the
relaunch resumed from transactional checkpoints with zero scientific loss.
Long stages now run detached.

Artifacts (SHA-256):
- `7d3c5d40c7f9fbc25ba45d9e54057cc60d86ec261d6a2084ac396d74b2ab5d00` pinned_capabilities/results/gate0e_dev_curve_seed100/curve.json
- `203f7ec880d734038eec62da844e914140313cfc57761cbfdee131291dc47a92` pinned_capabilities/results/gate0e_dwell_audit_dev/audit.json
- `8ebe040989ca7666a5c0883f2f2c4bed9bf2807430f17d551ffa74f685451869` pinned_capabilities/results/gate0e_null_seed100_b128/null_scan.json

## 2026-07-16 09:45 — predictions frozen; discriminator NON-DISCRIMINATING

c* = 1.0838 (dev radius at dev eta50). Out-of-sample null predictions,
sealed before any gate stream: seed 0 -> eta50 0.0118, seed 1 -> 0.0160,
seed 2 -> 0.0118 (all at B=128). Batch conditions: B=32 crossings at
0.0037/0.0033 (seeds 0/1); B=512 right-censored (radius never reaches c*
below 0.05).

The measured update diffusion falls steeply with batch (pooled
d ln D/d ln B = -1.93, 95% interval [-2.02, -1.85]) -> noise-theory
direction UP. The v-conditioned crossings ALSO rise with batch, and the
B=512 censoring voids the v-direction under the frozen rule. Status:
**non_discriminating** -> per v1.5.2, null_wins is unsatisfiable. Gate 0-E
now decides only between null_loses and ambiguous, via the seed ratio
checks. Note the two theories' batch directions agree in sign here -
exactly the degeneracy the external review predicted; the frozen machinery
declared it rather than us.

Context for the ratio checks: dev observed eta50 was 0.0067; the sealed
gate predictions sit 1.8-2.4x above that. If gate seeds resemble dev, the
null misses near or beyond the 2x line; if their boundaries genuinely sit
higher, it hits the 1.5x band. Live either way.

Artifact: `ede7474ae6f8ca17b94014698de757ace98ce8e2dc8a741e2df448415dcf17ea`
pinned_capabilities/results/gate0e_predictions/predictions.json

## 2026-07-16 10:20 — pre-outcome measurement note on the null predictor

Written and committed while all gate curves are mid-flight (no gate
aggregate exists). Verified from the sealed tables; no frozen rule changes.

**1. The predictor is ill-conditioned at the precision the rule demands.**
Near the crossing region, dR/d ln(eta) = 0.039-0.085 across the four
scans, so the 1.5x match band requires radius precision 0.016-0.035.
Observed instrument noise is the same size: seed 0's table is non-monotone
at the bottom (1.0319 -> 1.0090 -> 1.0611), a dip of 0.023. The ratio
checks therefore carry limited evidential weight IN EITHER DIRECTION: a
hit or a miss at the 1.5-2x scale is within the instrument's noise band.

**2. A uniform downward offset separates gate tables from dev.** The
gate-seed mean radius sits below dev's at every rate (-0.024 to -0.094),
which is what pushes every sealed prediction 1.76-2.38x above dev's
observed eta50. Hypothesis: snapshot AGE. Dev's c* was calibrated on the
legacy step-6,400 calibration snapshot; the gate states are step-8,000 by
the frozen v1.4.1 recipe. More settling, lower local curvature, lower
radius. Testable prediction: a seed-100 state prepared at step 8,000
should show radii near the GATE tables, not near dev's 6,400 table. The
test (dev-side, exploratory, protocol-neutral) is running now; its result
will be appended below before any gate curve seals.

**3. Outcome map under the frozen thresholds.** If the gate seeds' true
boundaries match dev's (0.0067), the misses are 1.76x / 2.38x / 1.76x -
only seed 1 exceeds 2x, so miss_count = 1 and the verdict is AMBIGUOUS,
not null_loses. A null_loses verdict requires two seeds observed at or
below roughly 0.0059 / 0.0080 / 0.0059. The frozen rule is thus buffered
against the central artifact scenario, but a null_loses reached via ~2x
misses would still ride partly on the offset and conditioning above, and
will be reported with that caveat attached.

**4. Gate 0-E is no longer a kill test, and we say so now.** With the
discriminator non-discriminating, null_wins is unsatisfiable; the gate
decides only null_loses versus ambiguous, through a channel whose
resolution is comparable to its noise. Commitment, recorded pre-outcome:
the ratio outcomes will be reported on their merits regardless of label -
if the sealed predictions land inside 1.5x, that is a predictive success
of the local theory and will be written as one, even though the frozen
rule can only print "ambiguous". Gate 0-E's honest product is a
methods-level result about a cheap checkpoint-local predictor; the
project's weight rests on the reversibility phenomenology and Gate 1.

**5. The diffusion slope (-1.93) is anomalous under both theories** (SGD
noise predicts -1; batch-invariant Adam predicts ~0) and is flagged
unreliable for any future temperature or Arrhenius use. Suspected
mechanism: incomplete second-moment re-equilibration - the refresh length
equals one beta_2 timescale (1,000 steps), so v is only ~63% adapted to
the condition's batch size when diffusion is measured, inflating the
apparent batch dependence. Verdict-irrelevant (the discriminator is
already void), but it must be resolved before D enters any positive
mechanism claim.

## 2026-07-16 12:05 — age test REFUTES the age-confound hypothesis

Seed-100 re-prepared at step 8,000 (matching the gate recipe) and re-scanned:
radii shift by -0.005 to -0.030 versus its 6,400-step table (mean -0.010,
sign mixed, +0.013 at 0.005). The gate-versus-dev offset (-0.024 to -0.094,
mean -0.047) is therefore predominantly seed-level, not snapshot age. c*
transfer carries at most a small age component. Consequence, recorded before
any gate curve seals: the sealed predictions' elevation above dev's boundary
is either genuine seed physics (gate boundaries truly higher - a predictor
hit if the curves land near 0.011-0.016) or seed-level scatter in an
instrument whose noise equals its required resolution (a miss if the curves
land dev-like near 0.0067). The measurement note's item 2 hypothesis is
closed as refuted; item 1 (conditioning) stands. Artifact:
results/gate0e_null_seed100_age8000/null_scan.json.
- `ac46dcd6ca40d2c8eef1c2a9107b4b262851c6f51b64f34714a2f6b50ef39e76` pinned_capabilities/results/gate0e_null_seed100_age8000/null_scan.json
