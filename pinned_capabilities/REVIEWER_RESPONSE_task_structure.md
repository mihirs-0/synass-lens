# Response: what was learned before collapse (task-structure concern)

**Verdict: the concern is correct and we concede it.** The task is a finite
random association table, not a generalizable rule. "Capability", "solved
task", "information loss", and "relearning" as previously used overclaim.
The corrected claim is an optimizer-dynamics result about a memorized
mapping — which is where this project's evidence already sits. Nothing below
requires new heavy compute; the answers are in the generator code, one label
measurement, and the project's own prior finding.

## The data-generating process (§1), verified from `src/data/dataset.py`

- Inputs: B (6 random chars, globally unique, 1,000 of them); z (2-char
  selector; K=10 z-tokens **shared globally** across all B in the primary
  config). A (4 random chars).
- For each B, `generate_random_string` draws K unique A strings into an
  ordered list `a_list[B]`. Example `(B, z_i) -> a_list[B][i]`.
- The only reusable structure is the z->slot indexing (z_i selects position
  i, shared across B). **The A values are independent random strings.** There
  is no function A = f(B, z).
- Measured on the seed-100 mapping: **10,000 (B,z) pairs, 10,000 unique A
  strings — every target appears in exactly one example.** A held-out
  `(B, z_i)` therefore has a target that occurs in zero other examples and is
  unpredictable at chance. This is a proof, not an experiment.

## What examples the model has seen (§2)

Current pinned-capabilities experiments use `probe_fraction=0.0`,
`split="train"`: **all 10,000 pairs are training data.** The probes sample B
from that same support, so `full_vocab_ce`, exact match, and C_int are all
evaluated **in-support**. There is no held-out set in these experiments.
("full vocabulary" = cross-entropy over the 36-token output alphabet, versus
the candidate-restricted diagnostic — NOT "all inputs".)

## Rule evidence and what "solved" means (§3, §4)

There is no rule to learn, so there is no rule evidence, by design. The
project's own audits state it: `paper_audit_2026-05-01.md` — "3% held-out
generalization -> lookup table, not program";
`phase_diagram_open_question_QUERY.md` — "The endpoint is a lookup table.
Held-out (B,z) accuracy is at/below chance... This is intentional."
**"Solved" means perfect interpolation/memorization of the finite
10,000-entry table.** It is not generalization and not algorithmic
understanding. Corrected terminology: *interpolating solution* /
*memorized mapping*.

## What disappears during collapse (§5)

This is **not ungrokking**: there was no generalization to lose (held-out is
chance before, during, and after — flat, by construction). What collapses is
the in-support memorized input->output mapping. The end state is the
input-blind label marginal: measured (task 1c) as per-example output entropy
3.579-3.581 with KL 0.005 to the data marginal versus 0.108 to uniform —
i.e. "all inputs map to the label marginal; complete loss of expressed input
dependence on the evaluated support", the fourth of the reviewer's scenarios.

## The label marginal (§6), measured on seed 100

Per answer position: 36 tokens, entropy 3.582-3.582 nats vs uniform
ln 36 = 3.5835; token frequencies 243-327 per 10,000 (uniform would be 278).
**Near-uniform, mildly imbalanced.** The marginal is computed over the
training support (= full support here, since all pairs are trained). Because
labels are near-balanced, the marginal predictor is essentially the
maximum-entropy constant, and — per the reviewer — the notable object is the
**trajectory from perfect fit to that constant and its dynamical
persistence**, not the marginal's existence. That trajectory and its
optimizer-controlled persistence is exactly what the study measures.

## What C_int certifies (§8)

C_int certifies that the model's outputs use B and z **jointly** in the
correct 2x2 algebraic form (additive B-only and z-only score functions
cancel), **on the evaluated support**. It does NOT certify a correct rule
(there is none) or generalization. On this task "correct" means "reproduces
the memorized table": untrained ~ 0, partial memorization intermediate, full
memorization ~ solved (16.65). It does not predict held-out performance
(always chance). It is a **within-support joint-binding statistic**, not a
capability or generalization certificate. This is consistent with the
corrected round-trip recount: C_int recovering while EM stalls = partial
re-memorization of joint structure without full table recovery.

## The claim that survives under the weakest interpretation (§10)

> Continued AdamW training can drive a small transformer from perfect
> interpolation of a finite synthetic mapping to the optimal input-blind
> predictor. Entry to that state is noise-gated and rapid; the state is
> maintained differently under stochastic versus full-batch updates; whether
> the mapping is subsequently re-interpolated is controlled by the learning
> rate (a sharp, seed-stable permanence boundary); and a recently collapsed
> state retains re-interpolation scaffolding that decays with dwell time.

Every surviving result — permanence boundary, noise-gated kick anatomy,
full-batch vs minibatch difference, escape-time ladder, age-ordered residual
structure, the stuck-state measurements (1a/1b/1c) — is an optimizer- and
representation-dynamics fact that needs no "capability" framing. The
external significance is about optimization after interpolation, not about
capability loss in a rule-learning system.

## The one control that would license a stronger claim (§7), not yet run

To separate "optimizer dynamics after interpolation" (task-structure
independent) from "loss of a learned computation" (task-structure dependent),
the informative comparison is a **structured-rule task with genuine held-out
generalization** (e.g. modular addition), matched on model, example count,
and marginal, run through the same fork/collapse protocol — reporting
train-support vs held-out separately through collapse. A random-association
control is nearly a no-op here because our task already is one. Prediction:
if collapse is pure post-interpolation optimizer dynamics, the structured
task collapses similarly on its train support; whether its **held-out**
(generalization) curve collapses at the same rate, earlier, or differently is
the discriminating measurement. Estimated cost ~ one solved-checkpoint train
+ a 7-rate x 8-stream fork on the structured task (~half the current wave).
Not launched; offered as the upgrade path.

## Consequence for the write-up

`PAPER_SCAFFOLD.md` must relabel throughout: "capability" -> "memorized
mapping / interpolating solution"; "solved task" -> "perfectly interpolated
the finite table"; "information loss" -> "collapse to the label marginal";
"relearning" -> "re-interpolation"; and §2.1 must state the DGP, the
per-example-unique targets, the in-support evaluation, and that held-out is
chance by construction. The phenomenon is unchanged; its interpretation is
narrowed to what the evaluations support.
