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

## 2026-07-16 12:40 — figure-driven reinterpretation: no spontaneous return

Plotting the 40 dev trajectories (loss vs steps) showed what the frozen
labels cannot: EVERY fork at EVERY rate collapses to the input-blind floor
at ~step 100, including eta=0.003 ("8/8 retained" = collapsed then fully
re-learned). Nothing spontaneously came back: the fork knocks the model to
the plateau and everything afterward is plateau-escape learning. Computed
escape times (CE committed below 3.0): 0.003 -> {50, 1200-1550};
0.005 -> {2800-3850, 5700}; 0.008 -> {5950, 10050, 15950} (3/8);
>= 0.0125 -> none in 8/8 x 4 rates. Escape time grows with eta and goes
infinite between 0.008 and 0.0125.

DEAD, and recorded as such: pinned / held / latent / reversible /
"optimizer-maintained suppression" / "spontaneous certified recovery."
"Two collapsed models, different futures" survives only in the mundane
reading (one rate is learnable, one is not). The scaffold requires a v3
after the controls below.

FAVORED HYPOTHESIS (legacy prior): the fork threshold is the task's
maximum learnable rate. Legacy gate_rescue inverse runs - fresh models at
eta=0.006 - sat at the plateau for 25,000 steps and learned only after
dropping to 0.001; legacy near-boundary escape times diverged (~24k at
K=10, eta=3e-3, old pipeline). Deciding controls, launched today:

1. Fresh-init cell (new pipeline): seeds 300 at eta in {0.003, 0.005,
   0.0125}, 25k steps. Fresh(0.0125) learning while forked(0.0125) never
   escapes = post-fork state worse than random init (a real result).
   Fresh matching forked escape times = full deflation to the MBC phase
   diagram.
2. Recovery races (running): forked-suppressed 16k states dropped to
   0.001, against the fresh-init 0.001 reference band (3,300-4,700 steps)
   - the same-rate comparison at the original training rate.
3. Queued: forks at eta=0.001 (does the fork-shock collapse occur at the
   training rate itself?).

The Gate 0-E machinery, verdict, and wave are unchanged - the frozen
pipeline measures what it measures; the INTERPRETATION of "erasure
boundary" is now "maximum rate at which the post-fork plateau is
escapable," pending the controls.

## 2026-07-16 13:50 — Task-1 checkpoint measurements + first control results

**1a. Weight norms (L2), stuck eta=0.0125 @16k vs solved vs init.** Global:
init 71.9, solved 103.9, stuck 254.9 (stuck/solved = 2.45x, stuck/init =
3.55x). Per group (stuck/solved): embeddings 4.62x, attention 2.92x, MLP
2.43x, unembedding 2.23x, layer_norms **0.45x**. The pure-decay shrinkage
prediction (e^-2 = 0.135x) is refuted: weights GREW except LN gains, which
halved.

**1c. Output distribution at the stuck state.** Per-example entropy
3.5794-3.5805 at every answer position (mean-dist 3.5801-3.5811), floor
3.5818, ln 36 = 3.5835. KL to data marginal 0.005-0.006 vs KL to uniform
0.108: every input maps to the same data-marginal distribution;
zero input information at the output. Replicates legacy Exp4b in this
pipeline.

**1b.** Stuck-state augmented radius scan running (certified path, no
refresh); table to be appended.

**Recovery races (n=2, exploratory).** Post-fork collapsed states (16k
steps at 0.0125) dropped to eta=0.001 for 12,000 steps: NO onset
(final C_int 0.077 / 0.007, EM 0.0). Fresh inits at 0.001 solve in
3,300-4,700 steps (10/10). At n=2: the aged collapsed state is a worse
starting point than random initialization at the original training rate.

**Fresh-init interim (6,600/25,000 steps).** All three rates still on the
plateau (CE 3.56-3.585, C_int ~0), including 0.003 — while FORKED runs at
0.003 escaped at ~1,300 steps. Two-sided so far: young collapsed states
(fork+~100 steps) escape faster than fresh inits at the same rate; aged
collapsed states (16k at 0.0125) escape slower than fresh inits at 0.001.
Consistent with plateau entrenchment growing with dwell time (legacy:
"deepened prior delays"). Interim - full horizons pending.

## 2026-07-16 14:35 — controls, second readout

**Full-batch fork at 0.0125 (exact full-dataset gradient): NO collapse
through step 200** (CE 0.000, C_int 17.1->17.4, EM 1.00) - versus 8/8
minibatch collapses by ~step 100 at the same rate from the same snapshot.
The fork collapse is noise-driven, not a deterministic step-size
instability. Follow-up queued: full-batch continuation FROM the stuck
state (noise-maintained vs geometric trapping).

**Fresh-init (n=1/rate, interim ~20k/25k):** eta=0.003 escaped at 10,450
(forked escaped ~1,300: 8x faster); eta=0.005 STUCK at 19,900 (forked
escaped ~3,400); eta=0.0125 stuck. Fresh learnability boundary lies
between 0.003 and 0.005 (legacy eta* ~ 5e-3 consistent). Combined with
the races: young post-fork collapsed states are BETTER starting points
than random init (escape at rates fresh cannot); 16k-aged collapsed
states are WORSE (no onset at 0.001 in 12k). Residual structure exists in
the young collapsed state and decays with high-eta dwell - measurable as
escape advantage, not as spontaneous return.

**1b note:** stuck-state radius scan died (BrokenProcessPool under load
35); relaunching single-worker when fresh-init slots free.

## 2026-07-16 18:05 — kick anatomy complete

**Fork at eta=0.001 (training rate), 2 streams, 8,000 steps: NO collapse**
(max CE 0.0004; C_int grew 17.9 -> 21.3). **Full-batch fork at 0.0125,
capped at 550 steps: NO collapse** (max CE 0.0003) - 5x past the minibatch
collapse window. Together with 8/8 minibatch collapses at every rate
>= 0.003: the fork collapse requires BOTH minibatch noise AND a rate above
a threshold between 0.001 and 0.003. It is not data-order shock and not a
deterministic step-size instability. Fresh-init finals: 0.003 escapes at
10,450 (forked: ~1,300), 0.005 and 0.0125 never in 25k. Remaining in
flight: full-batch continuation FROM the stuck state (noise-maintained vs
geometric trap) and the stuck-state radius scan (1b, relaunched
in-process after two pool crashes under memory pressure).

## 2026-07-16 19:00 — 1b: the stuck state is locally UNSTABLE on paper

Augmented AdamW spectral radius AT the stuck eta=0.0125 checkpoint (its own
theta, m, v; certified at all 7 rates), against the solved state:

| rate | solved | stuck |
|---:|---:|---:|
| 0.003 | 1.053 | 1.353 |
| 0.005 | 1.062 | 1.495 |
| 0.008 | 1.097 | 1.680 |
| 0.0125 | 1.124 | 1.926 |
| 0.02 | 1.139 | 2.297 |
| 0.032 | 1.220 | 4.397 |
| 0.05 | 1.307 | 8.149 |

The "parked => radius < 1" hypothesis is refuted in the strongest direction:
the state that never moves for 16,000 steps carries the largest certified
multipliers measured in this project (1.93 at its own rate). The
deterministic checkpoint-local linearization is uninformative about
trapping here - the plateau is a noise-maintained stationary distribution
or nonlinearly re-attracted orbit, not a locally stable fixed point.
Numbers only; mechanism deferred to the full-batch-from-stuck run (in
flight: no escape through 350 steps, CE pinned at 3.580).

## 2026-07-17 13:40 — RETRACTION: "round trips" recounted rigorously

My 2026-07-17 interim claim ("~9 certified round trips, replicated on every
seed") used a heuristic (outcome==erased AND final_c_int>8.3) that skipped
the exact-match floor and conflated the near-expressed cluster with genuine
re-expression. Recomputed via the frozen is_expressed and
sustained_suppression_entry functions over 158 completed streams
(dev + seeds 0/1/2), figure results/figures/roundtrip_candidates.png:

- 115 streams reached CERTIFIED suppression (2000-step joint in-band window).
- Of those, streams returning to full expressed (C_int>=8.33 AND EM>=0.90
  AND dz>0) after the window: **3** — dev/0.005/s03, s1/0.003/s01,
  s1/0.003/s06. Seeds 0 and 2: ZERO.
- 8 near-expressed (C_int/loss recover, EM stalls 0.18-0.88, never re-solve).
- 104 permanent.

The plots show the mechanism: gray (info) and blue (C_int) lead, orange
(EM) lags and usually stalls below 0.90. Recovery is a slow sigmoidal
learning curve over thousands of steps, NOT a switch. Of the 3 that reach
EM>=0.90: s1/0.003 pair re-solve to 0.94-0.99 but flicker at a noisy
near-boundary operating point (strict all-rows-final-1000 sustained =
False); dev only grazes 0.90 at step ~15000, final EM 0.81.

Conclusion (corrects the interim, strengthens the deflation): no distinct
"spontaneous return." The return leg IS re-learning - graded, behavior-last,
completing to full re-solve in 3/158 streams. "Certified round trip" as a
headline phenomenon does not survive. PAPER_SCAFFOLD §4.3 ("recovery is
rate-dependent", "certified round trip") requires rewrite to: below the
permanence boundary the collapsed model RE-LEARNS; full behavioral
re-solving within 16k steps is rare and seed-specific; the common
sub-boundary behavior is partial (info+interaction) recovery without EM
re-solving.

## 2026-07-18 01:15 — lambda=0 cell sealed: collapse is NOT weight-decay-driven

Seed 1, four middle rates, 4 streams each, weight_decay=0:
  eta=0.005: 1/4 erased (medtau 13,650 - very late)
  eta=0.008: 4/4 erased (medtau 200)
  eta=0.0125: 4/4 erased (medtau 150)
  eta=0.02:  4/4 erased (medtau 300)
vs seed-1 default wd=0.01: 0.005 4/8, 0.008 8/8, 0.0125 8/8, 0.02 5/8.

Reading (Level 2): collapse occurs at 100% with ZERO weight decay for
eta in {0.008, 0.0125, 0.02}, so the phenomenon is not a reparametrized
L2/Ersoy weight-decay metastability - it survives lambda=0. Weight decay
modulates the boundary near threshold (at 0.005, removing it drops collapse
4/8 -> 1/4 and delays the survivor to tau 13,650). Retires kill-criterion 2
in the good direction. Caveat: n=4 (wd0) vs n=8 (default), so the
boundary-rate contrast is noisy; the high-rate 100%-vs-100% is clean.

## 2026-07-18 08:40 — WAVE SEALED (Level 2 complete): seed-stable collapse boundary

3-seed collapse curves (erased/8; +Nd = diverged, excluded):
  eta    seed0     seed1     seed2    mean P(collapse)
  0.003  2/8       2/8       1/8      0.21
  0.005  7/8       4/8       6/8      0.71
  0.008  8/8       8/8       8/8      1.00
  0.0125 8/8       8/8       8/8      1.00
  0.02   6/8+2d    5/8+3d    6/8+2d   0.71
  0.032  8/8       8/8       8/8      1.00
  0.05   1/8+7d    4/8+4d    4/8+4d   0.38

eta50 per seed (sealed, valid, monotone): 0.00359 / 0.00432 / 0.00413
(tight; CV ~9%). Collapse is a WINDOW: below ~0.004 the model re-learns
(permanence boundary); 0.008-0.032 gives ~100% clean collapse-to-marginal;
0.05 mostly DIVERGES (7/8, 4/8, 4/8) rather than collapsing. The dip at
0.02/0.05 in P(collapse) is divergence, not survival.

Level-2 verdicts:
- Collapse replicates across 3 independently trained checkpoints; eta50
  seed-stable at ~0.004. Kill-criterion 5 (checkpoint-specific) RETIRED good.
- Gate 0-E null (VESTIGIAL under reframe): observed eta50 ~0.004 vs sealed
  predictions 0.0118/0.0160/0.0118 -> misses by 3.3x/3.7x/2.9x, all >2x, on
  all 3 seeds. null_loses (and the batch discriminator was already
  non-discriminating, so null_wins was unsatisfiable). Reported as "a cheap
  checkpoint-local predictor failed out of sample," not a mechanism.

Level-2 map is done. Mechanism (Level 3) is the matched-noise experiment,
now running.
