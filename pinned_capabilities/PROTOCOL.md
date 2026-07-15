# Pinned capabilities: bistability, memory, and early warning in neural-network training

**Prospective protocol v1.2.1 — pilot-informed, not a pristine preregistration**

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
expression only when `C_int`, exact match, and `delta_z` agree. Improvement in
`delta_z` without `C_int` is labeled shortcut acquisition.

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
  learning rate that returns the model to the suppressed band within 2,000
  steps and keeps it there for the remainder of an 8,000-step hold.
- **Acquisition boundary:** from a common suppressed checkpoint, the largest
  learning rate that reaches the registered transition within 40,000 steps.

The short erasure boundary is primary for Gate 0. The acquisition boundary is
reported as censored time-to-event data rather than converting every timeout
to a stable state.

### 4.3 Local stability calculation

`H` is the Hessian of the **training loss**, never the Hessian of `C_int`.
The task gradient `g = grad C_int` selects the behavioral direction only.

For a fixed diagonal Adam preconditioner `D`, the symmetric local curvature in
whitened coordinates is

```text
lambda_C = (g^T D H D g) / (g^T D g).
```

The suite reports this approximation, the largest eigenvalue of
`D^(1/2) H D^(1/2)`, and a finite-difference multiplier of the augmented
`(theta,m,v)` one-step Adam map. The augmented multiplier is the primary local
predictor because momentum and second-moment feedback change Adam's stability
constant. The deep-linear positive control calibrates the numerical pipeline;
it does not pretend that conditional binding is a linear task.

### 4.4 Batch-size test

A raw boundary shift with batch size is insufficient because batch size also
changes Adam's measured `v` and hence local preconditioned curvature. The
registered stochastic signature is a residual batch effect after conditioning
on the measured augmented multiplier and curvature. Batch sizes span at least
16x, with independently regenerated optimizer-state measurements at every
batch size.

### 4.5 Decision

Continue if either condition holds:

1. The observed erasure boundary differs from the positive-control-calibrated
   augmented stability prediction by more than 2x in at least one registered
   configuration; or
2. a 16x batch change moves the boundary by more than 25% after conditioning
   on the measured local predictor, with the paired confidence interval
   excluding zero.

Kill if both directional boundaries track the calibrated augmented predictor
within 1.5x across every configuration and the residual batch effect is below
25%. Exit product: a short capability-resolved optimizer-stability note.

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
- Gate 1 passes alone: hysteresis and memory-location paper.
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
