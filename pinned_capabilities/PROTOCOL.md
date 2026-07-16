# Pinned capabilities: bistability, memory, and early warning in neural-network training

**Prospective protocol v1.4.1 — pilot-informed, not a pristine preregistration**

This document freezes all new decisions before the new gate suite is run. It
is informed by existing MBC experiments, including the order-0 `q*` result,
the high-to-low and low-to-high learning-rate switches, and the optimizer and
batch-size sweeps. Those experiments are pilots, never test-set evidence for
the claims below. Every subsequent change must be dated in the deviations
appendix. Test predictions and analysis code are committed before oracle
outcomes are opened.

## 1. Thesis

A conditional capability in a small neural network can occupy a **pinned
state**: behaviorally absent, stable for many updates, and maintained by the
training dynamics rather than described adequately as ordinary slow learning.
The program tests four separable signatures:

1. **Non-reduction:** the boundary is not fully explained by a standard local
   optimizer-stability calculation.
2. **Hysteresis:** expressed and suppressed states coexist at the same fixed
   training rule, with history determining the realized state.
3. **Memory and escape:** weights, optimizer state, or both carry the state,
   and finite module-specific interventions can cross its basin boundary.
4. **Early warning:** under a fixed near-boundary rule, state fluctuations
   anticipate spontaneous transitions better than amplitude or loss trends.

Every signature has an independent kill condition. Failure never gets renamed
as support for a weaker signature.

## 2. Scope and non-claims

The primary system is MBC conditional binding. Transfer is attempted only on
small tasks with a counterfactual behavioral assay. There is no claim about
frontier-model pretraining, abstract reasoning, or arbitrary capabilities.
Localization means that changing a module family's training rule changes
fate; it does not mean that the module is the capability's circuit.

The field label is **training-dynamics stability and control**, not
mechanistic interpretability.

## 3. Measurements

### 3.1 Behavioral scores

The task maps `(B,z)` to `A(B,z)`. The visible shelf previously called
"log K" is `log K` only under candidate-restricted first-token cross entropy.
The full-vocabulary order-0 reference is the empirically computed constant
machine `q*`, whose loss is `H(q*)` position by position. All state and
flat-loss tests use the full-vocabulary `q*` reference. Candidate-restricted
`log K` is retained only as a continuity diagnostic.

The primary capability metric is a 2x2 interaction contrast. For target
`A=A(B,z)` and counterfactual `B' != B`, `z' != z`, define

```text
C_int = E[s(A|B,z) - s(A|B,z') - s(A|B',z) + s(A|B',z')].
```

Each target logit is centered by the mean logit over the de-duplicated union
of the four quartet targets. This linear centering is primary because it makes
additive B-only and z-only score functions cancel algebraically; log-softmax
normalization does not have that property and is reported only as a secondary
metric. The implementation is unit-tested on synthetic additive and joint
score functions.

Secondary metrics are candidate-normalized `delta_z`, full-sequence held-out
exact match, full-vocabulary cross entropy, candidate-restricted cross
entropy, and a redundant-direction control. A branch counts as capability
expression only when `C_int`, exact match, and `delta_z` agree. Agreement is
frozen as `delta_z > 0`: `C_int` and exact match carry the magnitude criteria,
while the sign check rejects a perverse interaction that favors the wrong
condition. Improvement in `delta_z` without `C_int` is labeled shortcut
acquisition.

Two fixed, disjoint counterfactual probe batches are logged every 50 steps.
Correlated movement across batches estimates state fluctuation; uncorrelated
movement estimates evaluation noise.

### 3.2 Frozen state labels

Ten reference seeds, disjoint from gate seeds, estimate the order-0 `C_int`
mean and SD, solved `C_int`, chance exact-match mean and SD, and the empirical
`q*` loss band.

Raw-logit `C_int` is not bounded and therefore has no intrinsic solved scale.
A reference seed enters the solved ensemble only after both probes reach at
least 99% full-sequence exact match and at most 0.05 nats of full-vocabulary
answer-token cross entropy. Training then continues for a fixed 2,000-step
hold; that seed's solved `C_int` is the median over the final 1,000 steps. A
seed that does not reach the endpoint within its registered budget is a failed
reference, not a low solved value. Gates do not start unless all ten reference
seeds succeed.

- **Suppressed:** `C_int` remains inside order-0 mean plus or minus 3 SD for
  at least 2,000 consecutive steps.
- **Expressed:** `C_int >= 0.5 * C_int_solved` and exact match exceeds both
  chance by at least 5 SD and an absolute 90% floor. The absolute floor keeps
  a zero-success chance ensemble from making the behavioral criterion vacuous.
- **Transition:** first crossing of `0.2 * C_int_solved` with no return to the
  order-0 band during the next 1,000 steps.
- **Flat loss:** full-vocabulary loss stays inside the empirical `q*` band.

The `q*` band radius is the larger of three reference-seed SDs and 2% of the
mean `q*` answer-token loss. The relative floor prevents near-identical
analytic entropy values across randomly generated datasets from imposing an
unphysical sub-0.001-nat tolerance on a trained plateau.

Reference generation is frozen in its own manifest. Gate outcomes cannot
update these bands.

## 4. Gate 0 — reduction to known local stability

### 4.1 Question

Does the learning-rate boundary reduce to the local stability of the optimizer
state map? If it does, the larger program stops. A capability-resolved version
of edge of stability is correct but not the proposed contribution.

### 4.2 Two directional boundaries

The original 8k bisection would misclassify slow acquisition as stability.
Gate 0 therefore keeps the directions separate:

- **Erasure boundary:** from a common expressed checkpoint, the smallest
  learning rate that returns the model to both the suppressed `C_int` band and
  the empirical `q*` loss band within 2,000 steps and keeps both there for the
  remainder of an 8,000-step hold. Divergent high-rate states do not count as
  erasure and cannot serve as the upper endpoint of a bisection bracket.
- **Acquisition boundary:** from a common suppressed checkpoint, the largest
  learning rate that reaches the registered transition within 40,000 steps.

The short erasure boundary is primary for Gate 0. The acquisition boundary is
reported as censored time-to-event data rather than converting every timeout
to a stable state.

Gate 0 uses seeds 0--4, disjoint from reference seeds 110--119. The learning-
rate grid is calibrated only on the non-gate seed 100, then frozen before any
seed 0--4 boundary is opened. Batch comparisons are paired within seed.

The calibration candidate grid is `(0.003, 0.006, 0.012, 0.025, 0.05)` at
batch size 128. From the completed seed-100 solved checkpoint, the suite runs
both the complete 8,000-step erasure scan and the certified local-stability
scan at all five rates. Calibration succeeds only if the empirical scan has a
strict adjacent retained-to-erased pair, all five predictor cells are
certified, and neither endpoint is divergent or unresolved. The
smallest eligible retained-to-erased pair is the calibration bracket. The
five-rate candidate grid then becomes the frozen gate grid without alteration.
If calibration fails, the program stops before any gate seed is opened; no
adaptive grid search is allowed.

The executable erasure precheck runs before the expensive local scan. A
missing strict bracket writes `stop_before_gate0` immediately; a passing
precheck only authorizes the local calibration measurement and is never itself
an artifact that official seeds can consume.

Calibration is closed by an executable adjudication artifact, not a notebook
judgment. It binds the positive-control result, both seed-100 scans, their
manifests, and the calibration snapshot. Official seed 0--4 commands reject a
missing, stopped, stale, or control-mismatched artifact.

The first registered cell is seed 0 at batch size 128. Every expressed Gate 0
start is trained at learning rate 0.001 and saved at the common step 8,000 only
after expression holds throughout the final 2,000 steps. Its complete local
scan is written before its empirical fate scan starts. Acquisition starts from
a separate, jointly suppressed step-8,000 checkpoint trained at the smallest
erasing calibration rate. Failure to construct either starting state without
divergence stops the suite rather than changing its age or learning rate.
The matching complete and certified local scan is a required manifest input to
every empirical fate scan, enforcing prediction before outcome. If the first
cell does not provide immediate non-reduction evidence, its partial analysis
must return `collect` before any other gate cell or acquisition start can run.

### 4.3 Local stability calculation

`H` is the Hessian of the **training loss**, never the Hessian of `C_int`.
The task gradient `g = grad C_int` selects the behavioral direction only.

For a fixed diagonal Adam preconditioner `D`, the symmetric local curvature in
whitened coordinates is

```text
lambda_C = (g^T D H D g) / (g^T D g).
```

The suite reports this approximation, the largest eigenvalue of
`D^(1/2) H D^(1/2)`, and the dominant eigenvalues of the augmented
`(theta,m,v)` one-step Adam map. The augmented Jacobian is applied matrix-free
using exact Hessian-vector products and is checked against centered finite
differences on a tractable system. Its spectral radius is the primary local
predictor because momentum and second-moment feedback change Adam's stability
constant. The positive-control artifact has two independent parts: the
depth-two scalar-GD system must recover its closed-form unit-circle boundary,
and the production augmented-Adam HVP operator on a tractable two-parameter
quadratic must agree with a centered finite-difference Jacobian while its
reported eigenpairs satisfy the residual tolerance. This calibrates both the
decision convention and the numerical pipeline; it does not pretend that
conditional binding is a linear task.

The capability direction uses a fixed differentiable assay with 32 `B` values
and 256 registered quartets, averaged over answer positions. This assay is
smaller than the two state-label probes solely to make repeated
Hessian-vector products tractable; it never supplies a state label or gate
outcome. The training-loss Hessian uses the exact next minibatch from the saved
cursor and restores the cursor without advancing it.

The augmented eigensolver applies an eigenvalue-preserving similarity
transform that RMS-balances weight, first-moment, and second-moment blocks.
Every reported eigenpair includes a scale-normalized residual and is eligible
for Gate 0 only when that residual is at most 1%.

The local prediction itself is the ordinary discrete-map criterion, not a fit
to gate outcomes: `rho(J) < 1` predicts persistence of the starting state and
`rho(J) > 1` predicts departure. Adjacent certified rates that change side of
the unit circle define the predicted boundary interval. No crossing, when
every rate lies on the same side of the unit circle, produces the corresponding
one-sided censored interval. Because that interval has no geometric center, it
is diagnostic only: it can neither establish reduction nor open Gate 1.
Multiple reversals, a change in the wrong
direction, an exactly unit multiplier, or an uncertified cell do not get
repaired after seeing fate; they make the local predictor fail that registered
cell. The deep-linear control must recover its analytic unit-circle boundary
before any neural prediction is eligible.

The official finite-grid boundary estimate is the geometric center of its
strict adjacent interval. Reduction and non-reduction factors compare those
centers; interval endpoints and widths remain reported resolution diagnostics.
This convention is applied identically to empirical and predicted scans and is
validated on the positive control. The 12-step bisection command is a secondary
post-gate refinement only. It cannot enter the registered verdict, replace a
coarse cell, or rescue an invalid scan.

### 4.4 Batch-size test

A raw boundary shift with batch size is insufficient because batch size also
changes Adam's measured `v` and hence local preconditioned curvature. The
registered stochastic signature is a residual batch effect after conditioning
on the measured augmented multiplier and curvature. Batch sizes span at least
16x, with independently regenerated optimizer-state measurements at every
batch size. Confidence intervals resample the five seed clusters, never
individual branches.

The primary exact-16x contrast is 128 to 2,048 and the secondary contrast is
32 to 512. Both are required in both directions for a reduction verdict. The
conditioned response is the paired within-seed difference of
`log(empirical boundary) - log(predicted boundary)`; its 95% interval exactly
enumerates the `5^5` seed-cluster bootstrap resamples.

### 4.5 Decision

Continue to Gate 1 if either condition holds:

1. Either observed directional boundary differs from the
   positive-control-calibrated augmented stability prediction by more than 2x
   in at least one registered configuration; or
2. a 16x batch change moves the boundary by more than 25% after conditioning
   on the measured local predictor, with the paired confidence interval
   excluding zero.

Kill as reduction if both directional boundaries track the calibrated
augmented predictor within 1.5x across every seed and configuration and both
registered batch-effect intervals lie wholly inside the symmetric 1.25x
multiplicative-equivalence region. Exit product: a short capability-resolved
optimizer-stability note.

This is also a total investment rule. During staged collection, a certified
non-reduction result opens Gate 1 immediately; otherwise the action is to
collect the remaining registered Gate 0 cells. Once the fixed registry is
complete, the 1.5x--2x evidence band, a missing or censored boundary, an
uncertified predictor, incomplete seed coverage, or an invalid exact-16x
comparison stops the suite without opening Gate 1. Such a stop is reported as
a failed Gate 0 design, not as evidence for reduction or non-reduction. The
intermediate `collect` action is never a final outcome: the finite registry
must end in `continue` or `stop`.
Every Gate 1 runner requires the content-bound Gate 0 report with action
`continue`; a `collect`, `stop`, or missing report cannot open the next gate.

The verdict command accepts only paths to the raw empirical and local scan
aggregates. It re-derives both boundary intervals under the rules above,
computes bounded-cell residuals from geometric interval centers, and writes
the evidence decision plus action. It validates each aggregate against its
sealed child cells, rechecks local eigenpair count/residual/radius consistency,
and replays the content-bound calibration, local-before-fate, and first-cell
authorization chain. Registry completion is inferred from the exact frozen
cell set; no `finalize` flag can prolong or prematurely close the gate.
Hand-entered boundary values or gate reports are not eligible.

## 5. Gate 1 — hysteresis and memory

### 5.1 Hysteresis sweep

Starting suppressed at high learning rate, decrease learning rate in geometric
steps (12 levels per decade). At each level, dwell at least 5,000 steps. A
level is provisionally equilibrated only when both probe batches have absolute
`C_int` slope below one plateau SD per 1,000 steps over the final 2,000 steps.
The maximum base dwell is 20,000 steps; right-censored levels remain
non-equilibrated and cannot define a boundary.

Record `eta_on` when the down-sweep reaches the expressed state. Sweep upward
with the same schedule and record `eta_off` when expression collapses. Run two
cycles for five seeds. The primary band statistic is

```text
rho = eta_off / eta_on.
```

On three seeds, repeat using three times the accepted dwell at every level.
Inside the putative band, hold the rule fixed for 10,000 steps and exclude
slingshot-like loss spikes or cyclic weight-norm growth.

Pass when `rho >= 1.25`, the seed-clustered 95% interval excludes one, at
least four of five seeds show the loop, longer dwell shrinks width by less
than 25%, cycle two shrinks it by less than 30%, and the slingshot exclusions
pass. If the loop vanishes with dwell, phase and bistability language are
killed.

### 5.2 Strong reversibility

Low-high-low runs must distinguish recovery from relearning. Compare:

1. first acquisition from initialization;
2. never-solved high-to-low acquisition;
3. expressed-to-high collapse followed by low-rate recovery;
4. the same trajectory with Adam state reset at each switch.

Fast recovery is a secondary result, not required for hysteresis. The strong
"memory survives behavioral collapse" claim requires recovery in at most 25%
of first-acquisition time and faster recovery than the never-solved control in
at least four of five seeds. Otherwise the behavior is reversible but retained
capability memory is not claimed.

### 5.3 Memory-location surgery

At matched step and learning rate inside the band, cross expressed and
suppressed sources in a 2x2 design:

```text
weights source x optimizer-state source.
```

Optimizer state includes `m`, `v`, per-parameter step counters, scheduler
state, gradient-scaler state when present, and the saved data/RNG cursor.
Weight source determines memory if it determines fate in at least four of five
seeds regardless of optimizer source; optimizer source analogously; otherwise
the result is joint or unresolved. A norm-matched Gaussian weight perturbation
controls for nonspecific kicks.

### 5.4 Mechanism controls

Pilot results already show deterministic high-learning-rate suppression can
survive very large batches, so noise necessity is not preregistered. Instead:

- full batch tests whether the band remains finite without minibatch noise;
- a batch ladder measures how noise shifts `eta_on`, `eta_off`, and width;
- Adam-to-SGD-with-momentum swaps use effective per-module steps matched from
  Adam's current preconditioner;
- weight decay is crossed independently because it can create slingshot-like
  cycling.

The result is classified as noise-modulated, optimizer-state-maintained, or
geometry-maintained only after these interventions.

## 6. Gate 2 — module escape thresholds

Gate 2 runs only after Gate 1 passes. In a bistable system, subthreshold
responses relax toward zero, so the instrument is a basin-crossing threshold,
not a graded tangent response.

For each module family, reduce its learning rate by a fixed factor of 0.1 for
`H` steps, restore the base rule, and hold for a registered challenge period.
An adaptive staircase uses `H` in `{250,500,1000,2000,4000,8000}`. The escape
threshold `E_m` is the pulse duration at which escape probability crosses
50%. Candidate families are embeddings, attention, MLPs, layer norms,
unembedding, all parameters, and a parameter-count-matched random subset.

The frozen selection rule is `argmin_m E_m`. Oracle fate is measured by
maintaining each intervention to a 40,000-step cap.

Twelve development anchors select challenge duration and staircase details.
At least 36 untouched anchors use held-out complete combinations of task,
data, model shape, batch, learning rate, and seed. Discovery language is
allowed only if oracle winners are heterogeneous enough that the most-frequent
development winner is not competitive; otherwise the result is validation of
a consistent intervention, not blind localization.

Baselines include global low rate, most-frequent development winner, module
gradient norm, projected instantaneous capability change, update norm, linear
probe progress, preconditioned curvature, greedy short-horizon response,
one-step and 50-step pulses, restricted local learning coefficient, random
module, and update-movement-matched variants.

Pass only if top-1 selection exceeds the best baseline by at least 20 points
with a paired anchor-clustered 95% interval excluding zero, selected median
time is within 25% of oracle, total measurement-plus-training compute beats
global low rate, and accuracy drops by at most ten points on fresh independent
noise streams. Any tying baseline kills the instrument claim.

Common-random-number pairs are averaged over at least four independently drawn
streams to estimate an expected causal effect. Unpaired streams are the
external robustness test; a single matched future is never treated as phase
prediction.

## 7. Gate 3 — fixed-rule early warning

Early-warning statistics are valid only under a fixed control rule. Gate 3
therefore excludes learning-rate sweeps, pulse windows, and forced switches.
It uses dedicated constant-rule holds near both sides of the registered band
and any spontaneous fixed-rule transitions from Gate 2 controls.

Rolling variance and lag-one autocorrelation are computed on detrended
`C_int` separately for both fixed probe batches. A signal counts only when its
trend agrees across batches. The detector predicts transition within 1,000 and
3,000 steps and is compared with `C_int` amplitude trend and loss trend.

Pass requires AUROC at least 0.75 at both horizons and improvement of at least
0.10 over amplitude trend with trajectory-clustered intervals excluding zero.
At least 30 transition events are required; repeated events from one trajectory
remain one cluster. Failure is reported as evidence for noise-triggered
transitions without detectable critical slowing.

## 8. Outcome tree

- Gate 0 kills: capability-resolved local stability note; stop.
- Gate 0 is incomplete or ambiguous: stop without a mechanism claim; do not
  spend Gate 1 compute.
- Gate 1 passes alone: a paper only if the full memory-location surgery and
  deterministic/noise mechanism classification are clean; hysteresis alone is
  prior art after Ersoy and Wiesner (2026).
- Gates 1 and 2 pass: predictive module escape instrument.
- Gate 3 additionally passes: early warning and an empirical taxonomy of
  progressing, pinned, and unreachable absence.
- Gate 2 loses to a cheap baseline: publish Gate 1 and the negative instrument
  result without relabeling it as success.

## 9. Deviations appendix

### 2026-07-15 — version 1.2.0, before gate outcomes

The solved `C_int` ensemble was given an explicit behavioral endpoint and
fixed hold because a raw-logit contrast otherwise has arbitrary scale. The
full-vocabulary `q*` comparison is computed over answer tokens (excluding the
deterministic EOS token) to match the logged behavioral cross entropy. No gate
outcome had been generated when this clarification was made.

### 2026-07-15 — version 1.2.1, reference-smoke correction

The small reference smoke produced zero full-sequence successes in every
order-zero model, making empirical chance mean plus five SD equal zero. An
absolute 90% exact-match floor was added to the expressed-state definition.
The smoke is infrastructure validation only; no Gate 0 or Gate 1 outcome had
been run or inspected.

### 2026-07-15 — version 1.2.2, augmented-map implementation clarification

The originally proposed finite-difference augmented multiplier was replaced
by dominant eigenvalues of the exact matrix-free one-step Jacobian, with a
finite-difference equality test on a small system. Spectral radius is invariant
to rescaling the weight and moment coordinates; a raw directional response
norm is not. This changes the numerical method, not the Gate 0 decision rule.

### 2026-07-15 — version 1.2.3, erasure-smoke correction

An infrastructure erasure smoke showed that `C_int` alone can re-enter its
order-zero interval while full-vocabulary loss diverges. Gate 0 erasure now
requires sustained membership in both the `C_int` and empirical `q*` loss
bands, and divergent branches are explicitly excluded from bisection. No gate
outcome or boundary had been generated.

### 2026-07-15 — version 1.2.4, local-probe implementation freeze

The differentiable capability-direction probe was fixed at 32 `B` values and
256 quartets before any neural local-stability measurement. The full state
probes remain unchanged and cannot be replaced by this smaller assay.

### 2026-07-15 — version 1.3.0, field-novelty threshold

A literature refresh identified Ersoy and Wiesner (arXiv:2606.17120, June
2026), which already demonstrates neural-training hysteresis, metastable
states, and Arrhenius noise-driven escape in deep linear networks. The outcome
tree now states that hysteresis alone is insufficient: Gate 1 must also locate
training-state memory and classify deterministic versus noise-maintained
pinning. This raises the publication bar before any Gate 0 or Gate 1 outcome.
The ten-seed behavioral reference ensemble had started under v1.2.4; because
v1.3.0 changes no reference measurement, threshold, seed, or training rule,
that content-addressed ensemble remains eligible for subsequent gates.

### 2026-07-15 — version 1.3.1, q-star band calibration

Production reference calibrations showed that the analytic `q*` entropy has
extremely small between-dataset SD, much smaller than ordinary trained-plateau
fluctuation. A preregistered 2% relative radius was added as a floor under the
three-SD band before any suppressed checkpoint or gate outcome. Reference
generation itself is unchanged, so the running v1.2.4 ensemble remains
eligible.

### 2026-07-15 — version 1.3.2, intervention-novelty clarification

A further prior-art pass identified latent intervention evidence for hidden
capabilities (Park et al., arXiv:2406.19370) and established layer-wise or
selective-module training methods (LeRaC and Modular Adaptive Training). Gate
2 therefore cannot claim either hidden capability or module-specific learning
rates. Its contribution requires prospective behavioral selection to beat the
registered cheap baselines and cause a durable post-restoration fate change.
This clarification changes no measurement, threshold, seed, or training rule;
the running v1.2.4 reference ensemble remains eligible.

### 2026-07-15 — version 1.3.3, strict erasure brackets

The coarse erasure scan now distinguishes retained, erased, divergent, and
unresolved branches. Only an adjacent retained-to-erased pair is eligible for
bisection, and any divergent or unresolved midpoint invalidates that path
rather than being silently treated as a retaining endpoint. This fixes an
implementation ambiguity before any Gate 0 boundary result. Reference
generation is unchanged, so the running v1.2.4 ensemble remains eligible.

### 2026-07-15 — version 1.3.4, Gate 0 seed freeze

Gate 0 previously required a paired confidence interval for the residual batch
effect without naming its sampling units. Seeds 0--4 are now frozen as Gate 0
replicates, disjoint from reference seeds 110--119; the rate grid is calibrated
only on seed 100. Batch comparisons are paired within seed and intervals
resample whole seed clusters. No Gate 0 outcome had been run or inspected.
Reference measurements are unchanged, so the running v1.2.4 ensemble remains
eligible.

### 2026-07-15 — version 1.4.0, decision-complete Gate 0 freeze

A pre-outcome audit found that the prior Gate 0 prose did not freeze the rate
grid, first cell, state age, acquisition start, exact 16x contrast, or the
action taken in the 1.5x--2x evidence band. It also promised `delta_z`
agreement without enforcing it and bound inputs only by mutable paths. Version
1.4.0 freezes the five-rate seed-100 calibration, seed-0/batch-128 first cell,
step-8,000 starting states, exact contrasts, unit-circle predictor, sign-only
`delta_z` agreement, deterministic cluster bootstrap, and the total
continue-or-stop investment rule. It also makes the positive control and
seed-100 adjudication executable prerequisites, enforces local-before-fate and
first-cell ordering, and uses geometric centers for the symmetric frozen-grid
comparison while reserving bisection for post-gate refinement. Every reference
and snapshot input is now
content-hashed in its consuming manifest, with snapshot seed/config metadata
validated when available. The complete ten-seed reference ruler had finished,
but no seed-100 boundary calibration or seed 0--4 gate fate had been run or
inspected. These changes therefore alter no gate outcome.

### 2026-07-15 — version 1.4.1, artifact-chain and terminal-rule hardening

A second pre-outcome adversarial audit showed that the v1.4.0 CLI could trust
handwritten calibration, first-cell, or continuation JSON; verdict analysis
did not revalidate sealed child cells; resume histories were not content-bound;
and terminal completion depended on an optional `finalize` flag. Version 1.4.1
closes those routes. Calibration and Gate 0 authorizations are freshly
re-derived from their manifests and raw inputs at every use. Scan aggregates
must match self-sealed child results whose controls and source hashes match the
manifest; local certification is recomputed from eigenpair fields. Empirical
and state-preparation progress now binds the exact metrics prefix and
checkpoint bytes. Official snapshots require matched seed and experiment
metadata, except for the recorded seed-100 legacy calibration snapshot.
Censored boundaries cannot open Gate 1, registry completion is derived, and
single-rate exploratory commands cannot inspect official fate before the
registered order. The positive control now also exercises the production
augmented-Adam HVP/eigensolver against centered finite differences. No seed-100
boundary scan or seed 0--4 gate fate had been run or inspected, so the audit
changed no scientific outcome.

### 2026-07-15 — version 1.5.0, conditional Gate 0 reformulation

The v1.4.1 seed-100 calibration ended in `stop_before_gate0`: the frozen grid
contained no strict adjacent retained-to-erased pair. Amendment v1.5
([`AMENDMENT_v1_5.md`](AMENDMENT_v1_5.md)) was committed before any autopsy of
the three sealed unresolved cells. It freezes: a five-label autopsy classifier
with fixed precedence; a mechanical label-to-branch mapping (Branch A
escape-curve reformulation, Branch B deterministic parametric fix, Branch C
intermediate-state addendum); the complete Branch A design (grid, streams,
holds, dev/gate split, one-constant null with a v-conditioned batch-shift sign
discriminator, pass/kill rules, and the Gate 1 handoff at η₅₀). All v1.4.1
sealing, manifest, and re-derivation machinery remains in force. No autopsy
label, branch activation, or new run outcome existed when this amendment was
committed.
