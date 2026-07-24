# Amendment v1.5 autopsy report (appendix)

**Sequence.** The amendment was committed at `5a64670`; the classifier and its
19 synthetic tests were committed at `ffcc26b`; only then was the classifier
run on the sealed cells. The full machine-readable artifact, with per-rule
diagnostics for every cell and content bindings for every input, is
`results/gate0_autopsy_v1_5/autopsy.json`, SHA-256
`6a88bca70e9c7ef9ceae13f5a0c512e058a636e92832d3094d8119929a509b9c`.

**Inputs (content-bound in the artifact).** Sealed aggregate scan
`e7c6e49efe6bb6e5…` (the v1.4.1 seed-100 erasure scan), frozen reference
ensemble `8ab7fd81cbe0ec03…`, and the three sealed cell logs
(η=0.006 metrics `7681adf5…`, η=0.012 metrics `d7e305e3…`, η=0.05 metrics
`84b84aa7…`). Bands: suppressed C_int [−0.02023, 0.01835], q\* CE
[3.5102, 3.65347], divergence threshold 5.0 × q\* mean = 17.909.

## Labels

| η | label | firing rule |
|---:|---|---|
| 0.006 | `stochastic` | none of rules 1–4 |
| 0.012 | `deadline_artifact` | rule 2 |
| 0.05 | `stochastic` | none of rules 1–4 |

## Per-cell evidence

### η = 0.006 — `stochastic`

The trajectory visited **both basins inside one hold**. It was already
suppressed at the first logged row (step 50: C_int 0.0154/0.0067, CE
4.98/4.88 — CE still outside the q\* band), hovered at the band edge through
step 2,000 (C_int 0.0199/0.0227, marginally above the 0.01835 upper edge), and
then **re-acquired**: C_int 2.74 at step 6,050 rising to 8.32/8.18 at step
8,000 with CE falling to 0.97/0.92. Rule checks: near-divergence never fired
(max CE ≪ 17.9); the final window is far outside both bands (rule 2); the
final quarter mean (5.58/5.53) is not below half the first quarter mean
(0.0104/0.0107) — the trajectory *rose* (rule 3); the final-quarter slope is
+2.94/+2.91 per 1,000 steps against a 0.0064 tolerance (rule 4).

This is the amendment's predicted signature verbatim: a single finite
trajectory at an intermediate rate cannot be classified because fate there is
genuinely stochastic — this one collapsed immediately and then escaped back.

### η = 0.012 — `deadline_artifact`

Jointly inside both bands for the entire final 2,000 steps on both probes
(41/41 rows; window endpoints: step 6,050 C_int −0.0001/0.0023, step 8,000
C_int 0.0020/0.0016 with CE 3.5825/3.5831). It settled into full joint
suppression at ~step 5,450 — the run erased; only the removed 2,000-step
onset deadline censored it.

### η = 0.05 — `stochastic`

C_int is dead suppressed for the whole hold (first-quarter mean 0.0004/−0.0005,
final-quarter mean 0.0004/−0.0002), and the early CE excursion (11.83/11.90 at
step 50) decayed back — never crossing the 17.9 divergence threshold, so
near-divergence correctly did not fire and **no grid cap applies**. Rule 2
failed on exactly 4 of 41 final-window rows (steps 7,600–7,950) where CE
drifted to 3.653–3.692, a few hundredths of a nat above the 3.65347 band edge,
while C_int stayed suppressed. Rule 4 failed only its level test: the final
level (0.0004) is *below* the suppressed band's upper edge, not between the
bands. In substance the state is erased but runs slightly hotter than the
reference q\* band; under the frozen joint criterion that is not certifiable
suppression, hence `stochastic`.

## Mechanical branch activation

Per the frozen mapping: at least one cell is `stochastic` → **Branch A
(Gate 0-E escape curve) activates**. No cell is `intermediate` → Branch C does
not attach. The 0.05 cell is not `near_divergent` → the full 7-rate grid
{0.003, 0.005, 0.008, 0.0125, 0.02, 0.032, 0.05} stands. No cell is
`slow_erasure` → no extrapolated crossing exceeds step 14,000 → **T_hold =
16,000** (no escalation).

## Notes recorded at activation (no rule changed)

1. The η=0.05 evidence warns that high-rate suppressed states can hover just
   above the q\* CE band. Branch A's frozen `unresolved → not-erased` primary
   convention plus the required opposite-convention sensitivity table already
   bound the effect of this; no threshold is altered.
2. The η=0.006 collapse-then-reacquisition trajectory means Branch A streams
   at low-to-middle rates may re-express after transient suppression. The
   Branch A per-stream labels already handle this: sustained suppression at
   any time marks `erased` with its entry step τ; re-expression afterward does
   not un-mark it. This reading of "reaches sustained suppression" (first
   sustained window anywhere in the hold, irreversibly labeling the stream) is
   recorded here, before any Branch A run, as the operative interpretation.
3. Interpretation 8 of the classifier docstring (completion of unenumerated
   label mixtures) was not exercised: the realized labels fall under the
   amendment's explicitly enumerated "any stochastic → A" clause.
