# Amendment v1.5 — Gate 0 Reformulation (conditional, frozen pre-autopsy)

**Status.** Written after `stop_before_gate0` fired on the seed-100 calibration scan (v1.4.1, HEAD 929eca1). This amendment is committed **before** the autopsy classifier is run on the sealed cells. Branch activation below is mechanical from the autopsy labels — no judgment call occurs after unblinding. All v1.4.1 machinery (sealing, manifests, re-derived reports, forged-artifact rejection, dev/gate seed separation) remains in force unchanged.

**Why a reformulation and not a retry.** The bracket rule ("adjacent retained→erased") silently assumed the null hypothesis's ontology: a deterministic instability has a sharp threshold. A noise-activated transition produces exactly the observed pattern — clean retention far below, clean erasure far above, a broad zone where a single finite trajectory cannot be classified. The amendment therefore replaces the *object* Gate 0 measures (a boundary rate) with one that is well-defined in both worlds (an escape curve), rather than stretching the old object until it fits.

---

## 1. Autopsy of the sealed seed-100 cells (frozen classifier)

Applied to the three unresolved cells (η = 0.006, 0.012, 0.05) using only their sealed logs. Rules apply in this precedence order; the first that fires is the label. "Quarters" are quarters of the 8,000-step hold; all conditions must hold on **both** probes.

1. **Near-divergence.** Any logged answer-CE exceeding the q\* band by the registered divergence multiplier (v1.4.1 config value) at any point → `near_divergent`.
2. **Deadline artifact.** Jointly inside the suppressed C_int band and the q\* CE band for the final 2,000 consecutive logged steps (i.e., it *did* settle into suppression; only the 2,000-step onset deadline censored it) → `deadline_artifact`.
3. **Slow erasure in progress.** Mean C_int over the final quarter < 0.5 × mean over the first quarter, and the final quarter still above the suppressed band's upper edge (descending, not settled) → `slow_erasure`.
4. **Intermediate settling.** |C_int slope over final quarter| below the registered equilibration tolerance, final level strictly between the suppressed band's upper edge and 0.5 × the expressed reference, CE inside the q\* band → `intermediate` (possible third state).
5. **Stochastic wandering.** None of the above → `stochastic`.

**Branch selection (frozen mapping):**

- Any cell labeled `stochastic` → **Branch A** (escape-curve reformulation).
- All three cells in {`slow_erasure`, `deadline_artifact`} and none `stochastic` → **Branch B** (deterministic parametric fix).
- Any cell labeled `intermediate` → **Branch C** activates *in addition to* A or B (C alone never substitutes; if all three are `intermediate`, run C then A).
- A `near_divergent` label at 0.05 selects no branch by itself; it marks the grid's upper edge as unstable and caps future grids at 0.032.

The autopsy report (labels + the log excerpts that fired each rule) is committed as an appendix to this amendment before any new run launches.

---

## 2. Branch A — Gate 0-E: the escape curve (primary redesign)

### 2.1 Object of measurement

From the frozen expressed snapshot of a seed (weights + full optimizer state + RNG), for each learning rate η in the registered grid, run **n independent noise streams** (fresh data-order RNG, identical snapshot) for a hold of **T_hold** steps. Per-stream labels:

- **erased**: reaches *sustained suppression* — jointly inside the suppressed C_int band and the q\* CE band on both probes for 2,000 consecutive logged steps. τ = first step of that window. The old 2,000-step onset deadline is **removed**; the divergence guard below replaces its function.
- **diverged**: any logged CE beyond the q\* band × divergence multiplier before sustained suppression. Excluded from erased/retained counts; a rate with ≥ 50% diverged streams is marked `unstable` and excluded from curve fitting.
- **retained**: inside the expressed criteria at end of hold, never sustained-suppressed.
- **unresolved**: none of the above. Counts as **not-erased** in the primary fraction (declared conservative convention); treated as right-censored at T_hold in the secondary survival analysis. A sensitivity table with the opposite convention is reported.

Primary objects: the empirical erasure curve **P̂(erase | η; T_hold)** per seed, and per-stream **τ**.

### 2.2 Frozen design

- **Grid**: η ∈ {0.003, 0.005, 0.008, 0.0125, 0.02, 0.032, 0.05} (≈ ×1.6 geometric; anchors the calibrated retained and erased rates).
- **Streams**: n = 8 per rate.
- **Hold**: T_hold = 16,000 steps. Single pre-declared escalation: if any autopsy `slow_erasure` cell's final-quarter slope, linearly extrapolated, reaches the suppressed band later than step 14,000, T_hold = 24,000 instead. No other value is permitted.
- **Curve validity**: ≥ 4 non-`unstable` rates with ≥ 6 classified (non-diverged) streams each. If violated, the grid shifts down by ×0.5 exactly once; a second failure is a registered `stop_gate0e_no_regime` outcome.
- **Dev/test split**: seed 100 remains dev — its full curve is run first and used for (i) the calibration constant c\* (§2.3) and (ii) selecting the two batch-test rates (the grid rates whose dev P̂ is nearest 0.25 and 0.75). Official curves run on the **first three frozen gate seeds**, untouched until predictions are committed.

### 2.3 The null's prediction (winnable, one constant)

The certified augmented-AdamW multiplier |λ_aug|(η), computed with the **empirically realized second moments** at each condition, is the local theory's instrument. Because stable training sits above unit multiplier generically, the null is given one dev-fit constant: **c\*** = |λ_aug| evaluated at the dev curve's η₅₀ on seed 100. The null then predicts, out of sample, that each gate seed's η₅₀ lies where its own v-conditioned |λ_aug|(η) crosses c\*.

**The directional discriminator (registered in advance).** The two theories predict *opposite signs* for the batch-size shift of η₅₀. At an expressed optimum the true gradient is near zero, so v is noise-dominated and shrinks with batch size: larger B → smaller v → larger effective step at fixed η → the v-conditioned local boundary moves **down** with B (≈ η₅₀ ∝ B^(−1/2)). Noise-activated escape has effective temperature ∝ η/B, so the fixed-horizon 50% point moves **up**, ≈ linearly in B. The sign of dη₅₀/dB is therefore decisive even at coarse resolution. (The registered null number is the actual v-conditioned computation at each B, not this scaling sketch.)

### 2.4 Registered cells

1. **Primary curves**: 7 rates × 8 streams on dev seed 100 and on gate seeds 1–3.
2. **Batch cells**: at the two dev-selected rates, batch sizes {B₀/4, B₀, 4B₀} (B₀ = production), 6 streams each, gate seeds 1–2. Production-B₀ streams reuse the primary curves.
3. **Weight-decay cell (Ersoy confound, pulled forward)**: gate seed 1, the four middle grid rates, 4 streams each, **λ = 0**, same T_hold. If matched λ>0 cells erase while all 16 λ=0 streams retain, register `erasure_is_wd_mediated`: this does not stop the program but triggers a mandatory framing review plus a fixed-η·λ follow-up cell before Gate 1 — because in that world the phenomenon is an L2 phase transition reparametrized, and every later claim must be written against that mechanism.
4. **Positive control**: the existing deep-linear pipeline extended with SGD noise in a regime where escape theory is semi-analytic. Pass: monotone P̂ curve and Arrhenius fit R² ≥ 0.9. Validates the survival-statistics path before any neural verdict is trusted.

### 2.5 Pass / kill (frozen)

- **Null wins → program stops.** All three gate seeds: |log(η₅₀^obs / η₅₀^pred)| ≤ log 1.5, **and** the observed batch shift of η₅₀ matches the v-conditioned direction with a bootstrap CI excluding the opposite sign. Exit product: *"The capability erasure boundary is locally predictable from one checkpoint measurement."*
- **Null loses → proceed to Gate 1, claim earned.** ≥ 2 of 3 gate seeds off by > 2×, **or** the batch-shift sign is opposite to the v-conditioned prediction with CI excluding agreement.
- **Ambiguous → proceed to Gate 1, claim demoted.** Any other pattern. The surgery runs regardless; the paper may then say only "local stability was not established as sufficient," never "local stability fails." Stopping requires the null to affirmatively win — ambiguity never stops the program.
- **Arrhenius secondary (labeled finding, not a gate)**: pooled median ln τ vs B₀/η across ≥ 5 cells with 0 < P̂ < 1: linear R² ≥ 0.8 registers `noise_activated_regime`, the quantitative bridge to arXiv 2606.17120.

### 2.6 Statistics

η₅₀ per seed by binomial-likelihood logistic fit in log η; CIs by stream-level bootstrap. Monotonicity of P̂(η) is checked, not enforced — a violation beyond noise is an anomaly flag that pauses interpretation. Unit of analysis for the null test: seed. Divergence-marked rates never enter fits.

### 2.7 Handoff to Gate 1

The escape curve becomes the standing ruler: the memory factorial's challenge rate is set to each gate seed's η₅₀ estimate — the rate where fate is a fair coin — so any systematic fate shift from transplanted weights or moments has maximum detectability. This is registered now so the surgery's operating point cannot be chosen after seeing surgery pilots.

### 2.8 Cost (CPU, ~0.09 s/step)

Dev curve 56 runs, gate curves 168, batch cells 48, λ=0 cell 16 ≈ **288 runs × ~24 min ≈ 115 h serial; ~1.3 days at 4 workers**. Streams are embarrassingly parallel; on a single modest GPU this is hours. If T_hold escalates to 24,000, multiply by 1.5.

---

## 3. Branch B — deterministic parametric fix (only if zero `stochastic` labels)

Grid unchanged plus fillers at 0.009 and 0.018; single stream per rate; T_hold = 24,000; onset deadline removed, divergence guard retained. The original adjacent-bracket rule and bisection then resume, and the original Gate 0 pass/kill conditions apply unchanged. If this branch *also* ends without a bracket, Branch A activates automatically — a deterministic fix that fails twice is treated as evidence the transition is not deterministic.

---

## 4. Branch C — intermediate-state addendum (additive)

For any `intermediate` cell: (i) resume the sealed run 8,000 further steps — does the level persist? (ii) full behavioral panel at the settled state (EM, Δz, C_int, per-position CE) to identify the partial solution; (iii) 3 fresh streams at that rate — attractor or fluke? A reproducible intermediate attractor is registered as a new object (`partial_phase`) and folded into the taxonomy; in Gate 0-E it counts as not-erased and is flagged. C never blocks A or B.

---

## 5. Order of operations

1. Commit this amendment (v1.5.0). 2. Run the autopsy classifier; commit its report. 3. Activate the branch mechanically; record activation. 4. Branch A: positive control → dev curve → freeze c\* and batch-test rates → commit gate-seed predictions → run gate curves + batch + λ=0 cells → verdict by §2.5. 5. Gate 1 proceeds on `null loses` or `ambiguous`; stops only on `null wins`.

## 6. Deviations

(empty — v1.5.0)
