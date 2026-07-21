# Amendment: two confound-killing controls — N_PM and M_T (2026-07-21)

Frozen BEFORE any full run. Closes two defects an external review found in the
"durable trapping requires both moment channels" headline: (1) the both-channel
arm N carries ~2x the realized update-noise power of v-only V (Gamma 1.97 vs
0.98) -> trapping might be power, not routing; (2) the m-only arm M diverged ->
the interaction rests on one controlled arm and one crash. Two new arms, then
the experimental phase closes permanently (stop rule below).

## Revision 2026-07-21 (stop rule + total run count UNCHANGED)
1. **Reorder after N1** (do not preempt N1): resume -> 7 pilots -> freeze commit
   (c, s*) -> N_PM x2 -> M_T x2 -> V_fresh replication. Short/decision-relevant
   before 16k marathons; resume protects the 4,600 sustained-escape anchor;
   pilots gate the confound-killers.
2. **N2 CANCELLED, replaced by a V_fresh replication on the ORIGINAL table.**
   The full-noise censor was seen twice (N0 + old real-minibatch); a third is
   redundant. The fragile cell is fresh+noise=4,000 (n=1, on a seed-300 sibling
   table). New run: INIT=fresh_orig (fresh random model, new seed, on the
   collapsed model's seed-100 table), V arm, s=1.0, eta=0.0125, 10k cap, frozen
   criteria. Removes the table confound and de-risks the thesis cell at once.
   (Net run count: -N2 +V_fresh_repl = 0.)
3. M_T selector -> Gamma-band match (see M_T section).
4. 2x2 writeup language registered: REACQ_2X2_WRITEUP_LANGUAGE.md.

Harness `reacq_2x2_run.py` extended (arms N_PM, M_T; SCALE dose; mandatory Gamma
accounting via pathwise clean shadow moments, sealed definition
Gamma=sum||u-u_sh||^2/sum||u_sh||^2). Routing unit tests `test_reacq_arms.py`
GREEN (per-channel routing, scaled-sigma injection, manual AdamW==torch 1.3e-17).

## Common setup (both arms)
- Start: same collapsed checkpoint (theta0,m0,v0,t0=22400) as all decomposition arms.
- eta=0.0125, full-data gradient, all criteria/cadence frozen at existing values.
  Onset = frozen interaction-contrast + sustained-escape. Non-escape = right-
  censored, reported as >= horizon, never "permanent". Horizon 16,000.
- CRN: base noise streams reused by seed index (ngen/mbgen seeded by arm+seed).
- Per stream record: CE, C_int(exact_match proxy), m/v norms, realized Gamma.
  Gamma reporting MANDATORY; an arm whose realized Gamma misses its band is
  MISCALIBRATED, not trapped/escaped.

## Arm N_PM — power-matched both-channel noise
Construction: identical to N (noise into m and v) with noise scaled by c<1:
g̃ = g + c·xi (SCALE=c). Same routing as N; only magnitude changes.
Calibration (pilot, pre-registration only): 500-step pilots at c in {0.5,0.6,0.7},
one stream each (seed 0); median instantaneous Gamma over steps 100-500
(Gamma_median_100_500). Choose c with median Gamma in [0.85, 1.15]. If none
lands, interpolate ONE additional c and repeat once (one recalibration total,
logged). Freeze c. Full runs: 2 seeds at frozen c, 16k cap.
FROZEN c = <PENDING PILOT>.
Pre-committed readings:
- both streams censored@16k, Gamma in band -> power confound dead; at matched
  power v-only escapes 4,600 while both-channel stays trapped >=16k. Strongest
  supportable insufficiency headline.
- escape tau>9,200 (>2x v-only) -> interaction real but graded; report tau w/ Gamma;
  headline softens to "m-channel substantially extends v-channel retardation at
  matched power".
- escape tau ~4,600 (+-20%) -> full-strength durable trap was a POWER effect;
  anti-DP insufficiency NOT supported at matched power; paper center reverts to
  saturating v-channel retardation + noise-gated 2x2. Written at equal prominence.

## Arm M_T — tamed m-only dose ladder
Construction: m <- g + s·xi ; v <- clean g^2 (unchanged from M spec). SCALE=s.
Stability probes (pilot): single-stream 2,000-step probes at
**s in {0.02, 0.05, 0.1, 0.25}** (0.5 dropped — smoke test Gamma~6e4 at s=0.25
already rules it out). Divergence (registered): CE>8.0 any eval, any NaN/Inf in
params/moments, or update norm >100x clean-arm median at same step index.
**AMENDED SELECTOR 2026-07-21 (prompted by the smoke test: clean v amplifies even
small s into huge realized Gamma; "largest stable dose" would run at a
magnitude-confounded power the gate cannot interpret — the fate of the old
parameter-noise arm).** Divergence is a CONSTRAINT (must complete 2,000 steps),
not the selector. Select s* = the STABLE dose whose windowed-median realized
Gamma (steps 100-500, Gamma_median_100_500) falls in [0.85, 1.15] — the SAME band
as N_PM, making V / N_PM / M_T a like-for-like triple (one shaking power, three
routings). If no dose is both stable and in-band, CLOSE the arm and report
"no stable in-band m-only dose exists" as the arm's finding (a publishable
property of the channel, not a failure). FROZEN s* = <PENDING PILOT>.
Full runs: 2 seeds at s*, 16k cap. Report realized Gamma prominently (clean small
v -> even small s can realize large Gamma; Gamma makes this comparable).
Pre-committed readings:
- s* censored@16k, Gamma <= ~2 -> m-channel independently sufficient for trapping
  at bounded power; interaction headline revised: either channel can trap; v's
  distinctive contribution is the saturating/bounded form of its retardation.
- s* escapes tau within ~1.5x of clean (800) -> bounded m-noise nearly benign; the
  durable trap needs the coupled config (v-inflated damping of full m-noise).
  Strongest support for the coupling claim.
- s* escapes slower, dose-dependent -> report m dose-response beside v dose-response
  as a paired figure; channel asymmetry (saturating vs non-saturating) is the finding.

## Validity gates
Valid only if: routing tests pass; every stream's realized Gamma reported; N_PM's
Gamma in [0.85,1.15] on the FULL runs (not just pilot); no criterion/horizon/
hyperparameter differs from the frozen package. A validity failure triggers
debugging + logged note, never reinterpretation.

## Deliverables
1. This amendment commit (+ frozen c, s* in a follow-up before full-run launch).
2. Calibration memo: pilot Gamma tables for both arms.
3. Per-run trajectories, Gamma table, survival table updated.
4. Mechanism cell selected mechanically from the pre-committed readings — one
   paragraph, no post-hoc language.
5. Confirmation N1/N2, fresh 2x2, resume completed as scoped, in the same table.

## Stop rule
After these two arms + the already-queued runs: NO further experiments under any
outcome. Unfavorable cells written at full prominence. Next artifact = paper draft.
