# Interpretation registry (non-decision-bearing)

**Status.** Registered 2026-07-16 from the external triage received the same
day, while the dev curve was still running (15/56 streams, no aggregate).
This document changes NO gate rule, threshold, or registered cell — the
v1.5.3 amendment freeze is intact. It binds claim language to evidence
tiers and schedules contingent interpretation tests that run outside the
decision pipeline.

## 1. The claim ladder

Each rung requires strictly more evidence; the program's gates only ever
purchase the first rung:

1. **Causal sufficiency** (Gate 1's claim): at identical weights, a
   controlled optimizer-state intervention changes future capability fate.
   Requires: the registered Gate 1 surgery plus its v1.5.2/v1.5.3 controls.
   Nothing in this registry is needed for this rung.
2. **Reachable-history effect**: optimizer states produced by valid
   gradient histories at those exact weights can alter fate. Requires
   test 3B below. Run only if Gate 1 finds a structured-moment effect.
3. **Natural prevalence**: ordinary training actually produces same-weight
   states with different fates. Not claimable from this program's designs;
   at most discussed as an open question.
4. **Latent capability retention**: the capability remained present during
   behavioral suppression. Requires tests 2A and 2B below. Without them the
   permitted phrase is *reacquisition*, never *retention*, *latent*, or
   *pinned*.

## 2. Language rules in force now

- "Expressed and suppressed **behavioral regions**", not "two basins
  separated along C_int", until the committor test (1B) shows that the
  behavioral coordinates locate the transition. Basin/committor language
  may describe the *hypothesis*, never the *finding*, until then.
- Escape-curve results are described operationally: fixed-horizon
  first-passage to the sustained-suppression criterion. (Artifact field
  names keep their frozen identifiers.)
- The round-trip observation is "behavioral suppression followed by
  reacquisition."

## 3. Test 1A — dwell/recrossing audit (RUN; first result below)

`gate0e_dwell_audit.py` (module + tests committed with this registry)
labels every logged row S (jointly inside both suppressed bands on both
probes), E (registered expressed criteria), or B (between); measures
episode dwells (completed and end-censored separately), transition versus
graze spans, recrossings, and first sustained-suppression entries at
window lengths {1,000, 2,000, 4,000}.

**First result — the five sealed seed-100 calibration cells** (artifact
`results/gate0e_dwell_audit_calibration/audit.json`, SHA-256
`22e6d7ef6ddd5390e430b756e830806e7017187f7a45d5abbe0c894ef22d3b0d`):

- Row-level region membership is noisy: graze fraction 0.875; completed
  (non-censored) dwell median only 75 steps against a transition-span
  median of 2,650; the meaningful long dwells are end-censored (median
  1,450, max 7,900).
- Even the retained eta=0.003 trajectory shows transient full-collapse
  excursions into the joint suppressed region (episodes E-B-S-B-E-B-E-B-E)
  that never sustain 1,000 steps — spike-like collapses and recoveries
  inside a "retained" run.
- The eta=0.05 cell alternates in and out of the suppressed region nine
  times; its sustained label exists at the 1,000- and 2,000-step windows
  (entry 850) but NOT at 4,000. The eta=0.012 cell sustains from step 100
  at every window. Window choice is load-bearing exactly where the state
  runs hot.

**Reading, recorded before any gate outcome exists:** at 50-step
resolution the behavioral coordinates do not, by themselves, exhibit
clean two-basin timescale separation; the sustained-window core-sets are
doing real work, and basin language is not currently earned at the
behavioral-coordinate level. This is consistent with (and mildly supports)
the program's own hypothesis that the full training state, not C_int,
carries the dynamical state — but that is exactly what test 1B must show
before the write-up may say it. The audit reruns automatically on the
complete dev and gate curves once they unseal (56 and 168 trajectories of
16,000 steps), including a hysteresis-smoothed episode variant to separate
measurement blips from genuine region changes.

## 4. Test 1B — committor-shooting pilot (after transitions replicate)

Only after the gate curves demonstrate the transition phenomenon at scale:
roughly eight matched checkpoint pairs near the apparent transition region
(matched on C_int, CE, delta_z; differing in arrival history), eight
paired future noise streams each, stopped at expressed core, suppressed
core, or horizon. Similar committors across matched pairs earn coordinate
language; history-dependent committors support full-state language; all
near 0/1 means the transition region was not sampled. Failure renames
things; it kills nothing. Requires new short runs plus checkpoint
retention around transitions — scheduled with the Gate 1 wave, never
inserted into the running dev/gate scans.

## 5. Tests 2A/2B — retention versus fast relearning (contingent)

Run only if retention-tier language is wanted for the paper:

- **2A frozen-body recovery:** expressed, suppressed-after-expression, and
  age/loss-matched never-expressed checkpoints; freeze the body, reset and
  retrain only the readout on a small balanced conditional set; compare
  examples-to-recovery across classes and readout initializations.
- **2B standardized relearning race:** suppressed-after-expression versus
  never-expressed, identical optimizer reset, low rate, balanced data,
  paired minibatch streams; compare full relearning curves and
  first-passage times to frozen milestones, not endpoint success.
- Even jointly positive results license "prior acquisition leaves a
  recoverability advantage," not "the capability was continuously
  present." A probe-only result licenses nothing (decodable is not
  causal).

## 6. Tests 3A/3B — chimera plausibility and reachability (contingent)

- **3A (registered as Gate 1 reported diagnostics, zero extra compute):**
  every surgery arm logs, over its first steps: total and layerwise
  ||m/sqrt(v)||, cosine of m/sqrt(v) with the current gradient, first-step
  and early-window update norms, first-step loss and C_int jumps, and the
  distance of these summaries from the natural parents' range. A crossed
  state whose early updates sit far outside the natural envelope makes a
  positive transplant easy to dismiss; inside the envelope rules out the
  crudest pathology. Reported, never decision-bearing.
- **3B reachable same-weight conditioning (only after a structured-moment
  effect survives Gate 1 controls):** clones of one near-transition
  checkpoint run at learning rate zero so weights stay exactly fixed while
  m, v, and step counters evolve under different genuine minibatch
  histories of equal length (tied to the beta_2 timescale); then the
  challenge rate is restored on paired future streams and the committor
  compared across conditioning histories. Staged: one conditioning length,
  several histories first; expand only if between-history variation
  appears.

## 7. Explicitly skipped (recorded so they are not re-litigated)

- No search for naturally co-occurring near-identical weight vectors
  (symmetries and flat directions make Euclidean proximity uninterpretable,
  and residual differences confound the continuation).
- No probe-based retention claims.
- None of the above is required for Gate 1's rung-1 claim, and none of it
  delays Gate 0-E.
