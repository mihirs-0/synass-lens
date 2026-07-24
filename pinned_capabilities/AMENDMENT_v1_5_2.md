# Amendment v1.5.2 — external-review adoptions (frozen before dev-curve unblinding)

> **Correction notice (v1.5.3).** Two literature statements in section 1
> below — that "Khanh" was retracted as fabricated and that "Feature
> Lottery" could not be found — were factually wrong. Both papers are real
> (arXiv:2607.06628 and arXiv:2605.24057) and were verified at the primary
> source while still blind. See `AMENDMENT_v1_5_3.md` section 1. The text
> below is preserved unedited as the record of the error.

**Provenance.** An external adversarial review was received on 2026-07-15
while the seed-100 dev escape curve was still executing. At the moment this
amendment was written, 5 of 56 dev streams had completed, `curve.json` did
not exist, and no dev statistic, gate-seed fate, batch cell, or lambda=0
outcome had been computed or read. The review was written against a
secondhand *brief* produced outside this repository, not against the code,
so every claim was first audited against the actual implementation.

## 1. Ground-truth audit of the review

**Hits the actual protocol (adopted below):**

1. *Batch-size sign discriminator imports the SGD temperature law.* True of
   the frozen amendment text itself (section 2.3: "effective temperature
   proportional to eta/B... moves up, approximately linearly in B"). In Adam,
   the noise scale of `m/sqrt(v)` can be nearly batch-invariant because both
   the minibatch noise and `sqrt(v)` scale as `1/sqrt(B)` at a
   noise-dominated optimum; published SDE analyses give Adam square-root
   rather than linear batch rules. The escape-side sign is therefore an
   empirical property of the regime, not a theorem. Adopted: section 2.
2. *Gate 0-E's mechanistic inference overclaims.* The certified spectral
   radius of one deterministic augmented map, linearized at one draw from a
   stationary distribution, is one member of a family (mean map, mean-square,
   random-product Lyapunov, finite-horizon transient growth). Adopted:
   verdict language demotion, section 3.
3. *Gate 1 surgery lacks the queued-displacement control and effect-size
   estimands.* Correct; Gate 1 had not yet been specified beyond v1.4.1.
   Adopted: section 4.
4. *lambda=0 alone cannot attribute causation to decay.* Correct. Adopted:
   section 5.
5. *Novelty sentence must narrow.* Correct; one cited precedent verified
   real. Adopted: section 6.

**Hits only the peer brief, not this repository:**

- "Norm-matched Gaussian v is out-of-distribution": the registered v1.4.1
  control is a norm-matched Gaussian **weight** perturbation
  (PROTOCOL section 5.3), not Gaussian moments. The review's shuffle-family
  upgrade is still adopted for the moments arms (section 4).
- Architecture description (one layer, d_model 256): the production system
  is 4 layers, d_model 128, as frozen in `MBCExperimentConfig`.
- "Khanh" and its section-8 numbers: no artifact in this repository
  references such work; the peer thread itself retracted the numbers as
  fabricated. Nothing in this program may rely on it without independent
  verification.

**Already satisfied by the suite (no change needed):**

- Stream-level bootstrap only; time steps are never resampled.
- Age-matched step counters, scheduler, scaler, RNG, and data cursor travel
  as one optimizer-state source (v1.4.1 snapshot ontology, PROTOCOL 5.3).
- The per-stream `erased` label is already a fixed-horizon first-passage
  event (first sustained window anywhere, irreversible), not a terminal
  classification; `retained` additionally requires never-suppressed.
- The verdict already refuses to stop on ambiguity.

## 2. Batch discriminator: demotion and measured-diffusion registration

The escape-side sketch "eta50 moves up approximately linearly in B" is
struck. The registered replacement:

1. **Measured update diffusion.** Every moment-refresh segment now records
   the per-step parameter-update statistics: mean step vector power
   (drift), per-step variance power (diffusion), and step count, at the
   condition's batch size and the fixed refresh rate. For each gate seed
   with conditions at B in {32, 128, 512}, the pooled log-log slope
   `s = d ln D / d ln B` is computed from these records (D = per-step
   update variance power).
2. **Noise-theory predicted sign** of `d eta50 / d B` is `-sign(s)`:
   diffusion falling with B predicts the fixed-horizon eta50 rises with B;
   `|s| < 0.15` is declared FLAT.
3. **Discriminator status**, sealed into the prediction artifact before any
   gate curve runs: `decisive` iff the v-conditioned predicted sign exists
   (crossings at all three batches) and differs from a non-flat
   noise-predicted sign; otherwise `non_discriminating`.
4. **Verdict tightening:** when the status is `non_discriminating`, the
   batch leg cannot support the null and **null_wins is unsatisfiable**;
   the outcome can only be `null_loses` or `ambiguous`. Matches/opposite
   are still computed and reported. This strictly hardens the stop
   condition ("stopping requires the null to affirmatively win") and can
   never rescue the null, so it is a permissible pre-outcome change.

The v-conditioned computation, the pooled log-odds v1.5.1 sign statistic,
and the ratio checks are unchanged.

## 3. Verdict claim language (demotion)

- `null_loses` licenses exactly: "the registered one-checkpoint
  deterministic local predictor (certified augmented spectral radius with
  one dev-fit constant) failed out of sample." It never licenses "local
  stability fails" or "the transition is noise-activated."
- Positive mechanism claims (`noise_activated_regime`) come only from the
  registered Arrhenius secondary and, once available, the measured-diffusion
  scaling — affirmative evidence, not the null's failure.
- These strings are embedded in the verdict artifact.

## 4. Gate 1 registered additions (before any Gate 1 run exists)

The following are registered as Gate 1 controls and estimands now, while
every gate-seed fate is unknown:

1. **Queued-displacement control.** For each transplanted optimizer state,
   compute the early weight displacement `dtheta_K` it produces under a
   common gradient stream, for K in {1, 5, 25, 125}; a control arm starts
   from reset moments plus `dtheta_K` added directly to the weights. If the
   direct kick reproduces the fate shift, the moments were a delivery
   vehicle, and the non-weight-memory claim dies.
2. **Clamped-weight washout.** Hold weights fixed while transplanted
   moments evolve on the identical gradient stream for K steps, release,
   and plot transplant effect against K on the same ladder.
3. **Effective-update controls.** The moment-randomization family acts on
   `u = m/(sqrt(v)+eps)`: paired within-layer (m_i, v_i) shuffle,
   independent shuffles, sign-shuffled m with v fixed, and a control
   matched on the layerwise distribution of u. Norm-only matching is
   insufficient. The v1.4.1 Gaussian weight kick remains as the
   nonspecific-perturbation control.
4. **Local-reconstruction control.** Moments synthesized from a short
   gradient window at the recipient weights; if they reproduce the donor
   effect, the information was local, not historical.
5. **Estimands.** The categorical 4/5-seed rule remains the preregistered
   headline for integrity, and the registered co-primary effect estimands
   are the same-weight committor differences
   `dq(W) = q(W, M_expressed) - q(W, M_suppressed)` at each weight source
   plus their interaction, with stream-level bootstrap intervals.
6. **Per-weight-source challenge calibration.** The challenge rate is
   calibrated so each weight source's baseline committor is estimated at
   its own operating point; eta50 from expressed snapshots is not assumed
   to be the sensitivity point for suppressed weights. Both transition
   directions are characterized (forward and reverse curves).
7. **Core-set sensitivity.** Every fate classification is reported at
   sustained-window lengths {1,000, 2,000, 4,000}; the registered rules
   stay at 2,000, and a conclusion visible only at exactly 2,000 is
   flagged, not defended.

## 5. Weight-decay follow-up upgraded

If `erasure_is_wd_mediated` fires, the mandatory follow-up is the
orthogonalized design — decoupled shrinkage `rho` and gradient step
`eta_g` varied independently (three families: vary `eta_g` at fixed `rho`,
vary `rho` at fixed `eta_g`, conventional diagonal) — replacing the
fixed-eta-lambda cell. Interpretation language is fixed now: a lambda=0
cell that never erases shows decay is necessary for the landscape; it does
not show decay caused the transitions.

## 6. Novelty sentence and citations

The program's novelty sentence is replaced by the reviewer's narrowed
version:

> Prior work establishes optimizer-state-dependent continuation dynamics,
> training-path hysteresis, and metastable feature acquisition separately.
> The missing causal experiment is whether structured optimizer state, at
> fixed weights, predicts the reversible fate of a specific latent
> capability beyond effects attributable to step scale, queued updates,
> and local preconditioning.

Verified and added to the novelty audit: *The Viscosity of Logic*
(arXiv:2601.17260, 2026-01-24) — beta-swept DPO with capability-resolved
probes and explicit training-path hysteresis. Not found in two targeted
searches and therefore recorded as unverified: "Feature Lottery"
(metastable plateaus, one-layer transformer, AdamW); it may exist under
another name and must be re-searched at write-up.

## 7. Gate-seed familiarity assessment

Legacy experiments used seeds 0 and 1 extensively, which raises a peeking
concern for the frozen gate seeds (0-4). Assessment: the pinned-capability
suite derives its mapping with `seed_mapping = 1_000_003 * seed + k` and
its own dataset, probe, and training pipeline, so no legacy run observed
any registered Gate 0-E cell's dataset, let alone its fate; familiarity is
with the phenomenon, which applies equally to any seed. The v1.3.4 seed
freeze therefore stands. Recorded rather than repaired.

## 8. Explicitly not changed

The frozen grid, 8 streams, T_hold 16,000, band definitions, stream labels
and their first-passage semantics, divergence rule, c* mechanics, the
1.5x/2x ratio thresholds, the v1.5.1 log-odds statistic, and the order of
operations all stand. No dev, gate, batch, or lambda=0 number was read in
the making of this amendment.
