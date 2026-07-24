# Capability suppression is usually immediate and often reversible
### The learning rate controls persistence, not destruction

**Scaffold v2 — corrected to the two-state picture (2026-07-16).** Changes
from v1: the marginal/shortcut villain is deleted (no such solution exists in
this task); the floor is the measured input-blind optimum over the full
vocabulary, not log K; "immediate" carries its measured exception; every
DOUBT line now has an ANSWER line naming the experiment that resolves it and
its status. Rules unchanged: write only what you can point at in a log file;
box what you can't.

Data in hand: one model, 7 rates x 8 copies, 16,000 steps = 56 runs.
`[PENDING-SAT]` = the 232-stream replication (3 fresh models + batch +
lambda=0 cells), sealing ~Saturday. `[RUNNING]` = launched today.

---

## Abstract — write LAST; the shape

1. A capability in a trained transformer can be destroyed by continued
   training at a higher learning rate.
2. Destruction is usually not gradual: median time to certified loss is
   ~100 of 16,000 steps at every destroying rate (rare late events near the
   boundary; say them).
3. Below a sharp threshold, destroyed capabilities spontaneously return
   under the same fixed rule; above it, none return (0 in 40 runs).
4. The learning rate therefore controls whether suppression persists, not
   whether it occurs.
5. Consequently two suppressed models can be identical in loss, capability
   measure, and behavior — both emit the input-blind distribution — and
   have opposite futures.
6. We measure the boundary (eta ~ 0.0067), characterize recovery, and test
   one cheap local-stability explanation, which our instrument cannot
   resolve at the required precision.

**DOUBT:** Sentence 5 is the paper. Defensible with what you have?
**ANSWER:** Population version — yes, today: suppressed states at 0.005 and
0.0125 are indistinguishable on every logged quantity (both at the measured
floor, C_int ~ 0, EM ~ 0) and their return rates differ (8/8 region-returns
vs 0/8). The sharp within-rate version (same rate, same look, different
fates) is the 0.005 cell itself: 1 certified destruction, 1 retained, 6
near-expressed, from one file. The committor pilot (registered, post-wave)
upgrades this to matched-checkpoint resolution. Write the population
version; box the committor version.

---

## 1. Introduction

Assumption attacked: a network is its weights; absence of a capability means
"not in there." Why it's hard to test: you need a capability reading that
cannot be faked and many reruns of the same starting state — hence a toy.
What you did: built that; found destruction is a kick (usually ~100 steps),
not erosion, and below a threshold the capability climbs back on its own.

Contributions (4): the measure and its guarantee (§2); the dose-response
curve with sharp threshold (§4.1); the finding — destruction usually
immediate everywhere, persistence rate-dependent (§4.2-4.3); a
pre-registered null test reported with its resolution limit (§5).

**DOUBT:** Contribution 3 stated as *your* finding in this system?
**ANSWER:** Yes — "in this system" appears in the sentence. Generality is
Saturday's replication plus nothing; write no more than that.

---

## 2. Setup

### 2.1 Task — the two-state picture (corrected)
Model sees B and z, must produce A = A(B,z), K-fold ambiguous in B alone.
**There is no intermediate solution.** The task affords exactly two stable
behaviors: the full joint binding, or the input-blind answer distribution.
No "uses B, ignores z" stage exists behaviorally — the June audit showed the
supposed marginal shelf is crossed without pausing, and the solution
crystallizes whole. The collapsed state's loss floor is the **measured
input-blind optimum over the answer alphabet**: 3.5818 +/- 0.0002 nats
(ten-model reference), numerically ~ ln 36 = 3.5835 because answer
characters are near-uniform by construction; where data marginal and uniform
differ, the state tracks the data marginal. It is NOT log K = 2.30, which is
a candidate-restricted diagnostic only. Get this right or a referee will.

### 2.2 The measure — a guarantee, not a detector
    C_int = s(A|B,z) - s(A|B,z') - s(A|B',z) + s(A|B',z')
on centered logits over balanced quartets. Any B-only or z-only score
function cancels algebraically (softmax versions do not — the log-partition
is not additive). Given 2.1, this is not catching a live shortcut; it is the
*guarantee* that "expressed" means joint binding, stated in one Methods
paragraph. It is not redundant with exact match: near the boundary they
decouple (C_int 14.6-16.5 with EM 0.85-0.89 in six 0.005 runs).

Secondary: exact match, Delta_z, answer-token CE against the measured floor.

### 2.3 States
Expressed: >=99% sequence accuracy, CE <= 0.05 (solved reference); the
frozen expressed label also uses the 0.90 EM floor — name it, and say why
(chance EM is 0; a chance+SD rule degenerates). Suppressed: jointly inside
the interaction band AND the measured-floor CE band for 2,000 consecutive
steps, both probes. Bands from the 10-model reference ensemble.

**DOUBT:** Would the result change at a 1,000- or 4,000-step dwell?
**ANSWER:** Instrumented — the dwell audit reports all fate classifications
at {1k, 2k, 4k}; on the 56 runs the choice matters exactly where the state
runs hot (0.05: certified at 1k/2k, not at 4k) and nowhere else. Reruns
automatically over the 232-run wave. `[PENDING-SAT]` — cite the table.

---

## 3. Method
One boring paragraph: solved model; freeze complete state (weights, Adam
moments, RNG); fork 8 copies differing only in future data order; 16,000
steps at one fixed rate; seven rates 0.003-0.05. Everything frozen before
data existed — grid, copies, hold, labels, rules. Cite commits (amendment
v1.5 chain; predictions sealed pre-outcome at `6617a45`).

---

## 4. Results

### 4.1 Dose-response
Figure 1: P(destroyed) vs eta, log-x. 0/8, 1/8, 6/8, 8/8, 8/8, 8/8, 6/8+2div.
Monotone; threshold 0.0067, CI [0.0056, 0.0075]. Two honesty sentences
in-line: (a) six 0.005 runs are *unresolved* — they end behaviorally
expressed (C_int 14.6-16.5) under the 0.90 EM floor; the frozen convention
counts them not-destroyed, which the raw traces show is the CORRECT side,
and the sensitivity table ships anyway. (b) The two 0.05 divergences are
excluded and labeled.

**DOUBT:** CI width at n=8 per rate.
**ANSWER:** Say "CI spans 1.35x" out loud. Resolution: `[PENDING-SAT]`
adds 24 runs per rate across three fresh models.

### 4.2 Destruction is *usually* immediate
Figure 2: time-to-destruction against the 16,000-step axis.
Median ~100 steps at every destroying rate through 0.02 (200 at 0.032, 850
at 0.05 — the state runs above the floor band there; say it). **The
exception you must print:** 2 of 44 destructions entered late — 1,250
(0.05) and 7,750 (0.008, near-boundary). So: fate is usually sealed in the
first ~0.6% of the hold; near the boundary it can hang in the balance for
half the run.

**DOUBT:** "Doesn't change afterward" — checked on retained runs too?
**ANSWER:** Checked today across all 56: no retained run ever certifies
suppression late (their band-touches never sustain), and the two late
destructions above are the only fate changes after step 1,000. The
universal claim is dead; the median claim with stated exceptions survives.

### 4.3 THE result: recovery is rate-dependent
Figure 3, two panels, same axes. Left (0.005): C_int drops to 0 at step
100, sits ~6,600 steps, climbs back, ends at 15.0. Right (0.0125): drops at
100, stays for 15,900 steps. Same starting file.

Numbers: at eta <= 0.005 every run leaves and re-enters the expressed
region 1-2 times; at eta >= 0.008, zero returns in 40 runs. The sentence:

> The learning rate does not control whether the capability is destroyed —
> it is destroyed at every rate we tested, usually within ~100 steps. It
> controls whether destruction persists.

The caveat, in this section: 92% of band-crossings are grazes. Exactly one
run (0.005/03) held certified suppression (entered 100, >= 2,000-step dwell)
and still recovered — onset 6,700, rebinding fast after onset (half-solved
by 8,200), EM >= 0.90 at 15,000. The certified-sustained round trip is n=1.

**DOUBT (the real one):** If the replication shows flickering but no
further certified round trips, what's the paper?
**ANSWER, written before Saturday:** Still a paper, smaller: (i) the
permanence boundary stands on 96 runs (destruction universal, return
strictly below threshold); (ii) same-floor different-futures stands at
population level; (iii) "spontaneous certified recovery" demotes from a
property of the regime to an observed-once event with a measured flicker
precursor everywhere below threshold. What dies is the word "often" in the
title — it becomes "sometimes". Additional resolution beyond the wave: a
dedicated long-hold cell at 0.005-0.0067 (dev seed, 32k steps, n>=8),
queued when wave capacity frees, converts flicker-vs-round-trip into a
rate with error bars. `[PENDING-SAT + queued cell]`

### 4.4 Same floor, different futures (corrected framing)
Both suppressed states emit the same thing: the input-blind answer
distribution at the measured floor. The loss DOES register collapse — it
climbs from ~0 to 3.58; write that, never "the loss stays flat." What the
loss cannot do is distinguish a model that will recover from one that never
will: after collapse, output carries zero information about the input in
both, and there is no behavioral remnant to rebuild from — recovery starts
from behavioral silence. The one number that separates the futures is the
learning rate, which is not in the file.

**"Held" vs "different futures" — the language decision:** on current
evidence, write DIFFERENT FUTURES only. In the one certified round trip,
recovery (onset 6,700; EM90 at 15,000) was *slower* end-to-end than
from-scratch acquisition (3,300-4,700 steps) — though at 5x the from-scratch
rate, so confounded. The deciding experiment is `[RUNNING]`: suppressed
16,000-step states dropped to the original 0.001, timed to expressed,
against the reference acquisition band. Faster = "held" returns to the
table with evidence; comparable/slower = the paper says reacquisition, and
says it plainly.

---

## 5. Is it just the stability limit? (methods posture)
The null: dominant eigenvalue of the one-step AdamW map in (theta, m, v),
one dev-fit constant, sealed out-of-sample predictions for three fresh
models (0.0118 / 0.0160 / 0.0118), committed before any gate stream ran.
Pre-outcome conditioning analysis (logged): dR/dln eta = 0.04-0.085 near
the crossing, so the 1.5x band needs radius precision 0.016-0.035, and the
instrument's own scatter is ~0.023 (one table non-monotone). Required
precision = noise. Also logged pre-outcome: gate tables sit uniformly below
dev's; suspected snapshot-age confound (6,400 vs 8,000) — direct age test
`[RUNNING]`, result to be appended before the curves seal. Conclusion
template regardless of Saturday: "a cheap checkpoint-local predictor did
not resolve the boundary at this precision" — never "local stability
fails." The batch discriminator declared itself non-discriminating (both
theories' directions agree; B=512 censored), so this section CANNOT stop
the paper's phenomenology and says so.

**DOUBT:** Reporting because informative, or because you ran it?
**ANSWER:** Informative twice over: it disciplines the mechanism talk
(nothing cheap explains the boundary yet) and it documents an instrument
limit others will hit. But it is a methods section, not a finding — and
the diffusion slope (-1.93) stays OUT of the paper (flagged anomalous;
suspected refresh-length artifact).

---

## 6. What this doesn't show
- One task, one architecture, toy scale; nothing about pretraining.
- Needs a counterfactual assay; "reasoning" in the abstract doesn't qualify.
- Where the recovering/non-recovering difference lives is NOT shown — Gate 1
  (moment surgery with queued-kick, washout, u-matched-shuffle,
  reset-at-theta_K controls) is the next paper.
- Weight decay is live: AdamW shrinkage scales with eta, so an eta-sweep is
  partly a decay sweep, and the 2026 deep-linear metastability line explains
  similar phenomena that way. `[PENDING-SAT: lambda=0 cell]`
- Certified round trip n=1. Flicker ubiquity n=16 runs.
- Sixth item, stated: the reference bands and the boundary were measured on
  the SAME dev model that supplied the round trip; the wave's fresh seeds
  are the first fully out-of-sample phenomenology. `[PENDING-SAT]`

---

## 7. Discussion — three paragraphs
The reframe: expected a kill switch, found a healing rate. What "absent"
means: an evaluation reading "absent" cannot distinguish not-there from
not-up-right-now; here one number distinguishes them and it is not in the
checkpoint. The open question, as a question: two identical-looking
suppressed states, different futures — the difference lives somewhere in
the training state; the optimizer's discarded running averages are the
suspect; testing that requires transplant surgery with a control for the
shove; next paper.

---

## Related work (concessions unchanged from v1)
Saxe / Cohen / Lewkowycz (the null's lineage — you report an instrument
limit, not a defeat); Ersoy-class L2 metastability (lambda=0 cell is the
discriminator); Feature Lottery (close setting, concede); Barak (mirror
image); Nanda (retrospective, circuit-bound; yours behavioral); Sevetlidis
& Pavlidis (the standing call); Khanh (closest transplant prior — their
moments-secondary is in an absorbing full-batch regime; yours is a
reversible coin-flip regime; one sentence, no more). Field label:
training-dynamics stability and control. Not interpretability.

---

## Write order
4.3 -> 4.2 -> 4.1 -> 2 -> 6 -> 3 -> 5 -> 7 -> 1 -> abstract. Start with
Figure 3. If 4.3 survives your own hand, the rest is downstream.

## Experiment ledger for the boxes
| Box | Experiment | Status |
|---|---|---|
| Abstract S5 sharp version | committor pilot, matched suppressed checkpoints | registered; post-wave |
| 2.3 dwell sensitivity | window-ladder audit over 232 runs | auto at wave seal |
| 4.1 CI | 3-model replication | sealing ~Sat |
| 4.2 late entries | full-log sweep | DONE today (2/44; max 7,750) |
| 4.3 round-trip rate | wave 0.005 cells + long-hold dev cell (32k, n>=8) | Sat + queued |
| 4.4 held-vs-futures | recovery race at 0.001 vs reference band | RUNNING today |
| 5 predictor | conditioning note + age test | logged; age test RUNNING |
| 6 decay confound | lambda=0 cell | queued in wave |
| 7 where it lives | Gate 1 surgery + controls | next paper |
