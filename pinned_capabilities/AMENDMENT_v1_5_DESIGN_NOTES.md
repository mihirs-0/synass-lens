# Amendment v1.5 Branch A design notes (frozen before dev-curve unblinding)

These resolutions complete the amendment's Branch A text where it is silent or
ambiguous. They are committed while the seed-100 dev curve is still running
and before any of its numbers have been read. None of them may change after
dev or gate outcomes are visible; any later change goes to the deviations
appendix with its trigger.

## 1. Gate-seed indexing

The amendment says both "the first three frozen gate seeds" and "gate seeds
1–3". The registered Gate 0 seeds are (0, 1, 2, 3, 4). The ordinal reading is
frozen: "gate seed k" means the k-th registered seed. Therefore:

- Primary curves: seeds 0, 1, 2.
- Batch cells: seeds 0, 1.
- Weight-decay (lambda = 0) cell: seed 0.

## 2. Gate-seed starting states

"Untouched until predictions are committed" refers to fate measurement.
Expressed starting states for seeds 0-2 must exist before the null's
predictions can be computed from their own snapshots, exactly as the dev
c\* requires the dev snapshot. Therefore: `prepare-expressed` runs for seeds
0-2 under the v1.4.1 recipe (learning rate 0.001, saved at common step 8,000,
expression holding through the final 2,000 steps) are authorized after this
note, but no escape stream, local-stability measurement at challenge rates,
or any fate-revealing run touches a gate seed until the prediction artifact
is committed. State preparation reveals only that a seed solves at the
preparation rate, which the v1.4.1 protocol already required.

The v1.4.1 `prepare-expressed` gating demanded a passed calibration artifact
for official seeds; that artifact can no longer exist (`stop_before_gate0`).
The v1.5 replacement authorization for state preparation only is: the
committed autopsy artifact with `primary_branch = "A"` plus the passing
positive-control artifact, both content-bound in the preparation manifest.
The dev seed 100 keeps its recorded legacy snapshot; it is never re-prepared.

## 3. Batch cells share the common snapshot

Batch cells start from the same expressed snapshot as the primary curves
(prepared at the production batch size 128); only the hold's batch size (32
or 512) differs, consistent with "production-B0 streams reuse the primary
curves". Batch size is part of the frozen stream-seed derivation, so batch
cells draw fresh data orders by construction.

## 4. The v-conditioned local predictor

"Computed with the empirically realized second moments at each condition" is
implemented as a **moment refresh**: from the condition's starting snapshot,
train exactly 1,000 steps at the preparation rate 0.001 and the condition's
batch size (fresh data-order seed derived as
`gate0e_refresh|seed=<seed>|batch=<batch>`), then measure the augmented-Adam
spectral radius at every challenge rate from the refreshed state
(theta, m, v), with the training-loss Hessian minibatch drawn at the
condition's batch size from the refreshed cursor. The refresh length is one
second-moment timescale (1/(1-beta2)). The refresh is applied uniformly to
every condition, including the dev c\* measurement and the production batch
size, so no condition is privileged. At the preparation rate the state is a
solved optimum; the 1,000-step drift in theta is order the training-loss
gradient there and is recorded in the artifact.

## 5. c\* and predicted eta50

c\* is |lambda_aug| log-log interpolated at the dev eta50 on the dev curve's
refreshed-rate scan. Each gate seed's predicted eta50 is where its own
refreshed |lambda_aug|(eta) crosses c\*, log-log interpolated between adjacent
certified rates; no crossing inside the grid yields a one-sided censored
prediction, which cannot satisfy the null-wins criterion (it can only produce
`null loses` via the >2x rule when the observed eta50 sits at least 2x inside
the uncrossed side, or `ambiguous` otherwise).

## 6. Batch-shift discriminator statistic

**Superseded pre-outcome by v1.5.1 (2026-07-15, deviations appendix).** The
original statistic here — bootstrap CIs on per-seed
`log eta50(B) - log eta50(128)` — proved structurally underpowered on
synthetic verdict tests (degenerate resamples, undefined under saturation)
before any batch cell or dev statistic existed. The registered statistic is
now the pooled Haldane-Anscombe log-odds shift of the erasure fraction at
the fixed contrast rates, averaged over seeds 0-1, with a full-validity
stream-level bootstrap CI; its predicted sign is the negative of the
predicted `log eta50` shift sign from the refreshed local scans. The
matches/opposite decision semantics are unchanged from the amendment;
descriptive eta50 shifts are reported alongside.

## 7. Registered estimator details

- Median tau is the conditional median of observed escape times in a cell
  (validated in the positive control at 1.07x the semi-analytic slope).
- The eta50 bootstrap resamples stream labels within each rate, 2,000
  replicates, seed 20260715.
- Divergence-marked rates never enter fits; the primary fraction counts
  unresolved as not-erased and the sensitivity table flips only that
  convention.

## 8. Compute record

The dev curve (56 streams of 16,000 steps, batch 128, CPU) was launched with
5 workers x 2 torch threads on a 10-core M4 immediately after this note's
content was fixed; its manifest binds the autopsy, control, snapshot, and
reference artifacts.
