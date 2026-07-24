# Closing package (frozen 2026-07-22) — FINAL runs, then stop rule fires permanently

After item 7, the next artifact is the paper draft regardless of outcomes.
Two 16k runs + five cheap items. No interpolation (supersedes the finisher's
c=0.1 plan). N2 cancelled. M_T closed. No tie-breakers.

## 1. In flight — DONE
- N1: censored at 16,000 (trapping headline now n=2). ✓
- s=1.0 V resume: from step-5,300 ckpt -> met sustained-escape + exact-match
  (tau_solve=7,000, final CE 0.0023, 100% retrieval). Did NOT relapse. ✓

## 2. V Gamma-pinning pilots: s=2, s=4, 500 steps each (collapsed, v-only)
REGISTERED PREDICTION (committed before launch): windowed-median Gamma ~ 1.0,
flat in s. Purpose: complete the pinning table from the HIGH side (N's Gamma
can't be pushed down to 1; this shows V's can't be pushed up to 2). GATE: if
either realized Gamma_median_100_500 rises materially above ~1.3, HALT and flag
the user before items 3-4 run — the ill-posedness argument would need rewording.

## 3. One N_PM full run: c=0.6, 16k, frozen criteria (collapsed, seed 1)
Purpose: outcome-invariance to the dial. Pre-committed readings:
- censored >=16k -> the trap tracks the pinned felt-noise level, not the injected
  amplitude -> supports the one-lever claim.
- escapes -> outcome depends on c even though Gamma doesn't (~1.86 at c=0.6);
  complicates the pinning story; written at FULL prominence.

## 4. V_fresh replication: new seed, ORIGINAL table, s=1.0, eta=0.0125, 10k
INIT=fresh_orig (fresh model, seed 1, on the collapsed model's seed-100 table).
Purpose: 2nd sighting of the thesis cell (~4,000) + remove the table variable.
Whatever it lands, the 2x2 language stays as registered (REACQ_2X2_WRITEUP_
LANGUAGE.md): "advantage eliminated (possibly inverted)," both mechanisms equal
weight, no standalone inversion claim.

## 5. M_T CLOSED — no full arms
Registered finding: no stable in-band m-only dose exists; realized Gamma pinned
~250 by clean-v amplification (s=0.02 -> Gamma 251). The pilot ladder table +
that sentence is the arm's complete contribution.

## 6. No other runs
Not N2, not a fine s-sweep, not tie-breakers (e.g. scaling V up to Gamma~2 —
noted as the manipulation that WOULD cleanly separate power from routing, but
OUT of scope by the stop rule). New needs get flagged in writing, wait for draft.

## 7. Final deliverables (then draft)
(a) Gamma-pinning table: routing -> pinned realized power (V~1 across s; N~2
    across c=0.5-1.0; m-only~250), s=2/4 folded in.
(b) survival table, every non-escape as >= horizon.
(c) mechanism paragraph (mechanical from registered language): in AdamW, noise
    routing and effective noise power are ONE lever, because the second moment is
    simultaneously noise sink and step-size regulator; the trap is v-damped
    m-scatter.
(d) one-page run ledger: arm, seed, horizon, outcome, deviation — for the appendix.
