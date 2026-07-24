# Registered writeup language for the reacquisition 2x2 (frozen 2026-07-21)

Frozen BEFORE the fresh+noise replication (item 2) lands, to stop the writeup
from hardening around the exciting version. The four numbers (all onset step,
eta=0.0125, n=1 per cell):

|            | clean | v-noise |
|------------|-------|---------|
| collapsed  | 800   | 4,600   |
| fresh      | 2,200 | 4,000   |

## Claims PERMITTED
- The head start is real: the collapsed model re-learns faster than a fresh one
  from clean gradients, 800 vs 2,200 (clean-vs-clean, n=1). fresh+clean DOES
  learn at 0.0125 (onset 2,200) — the rate is not the obstruction.
- v-noise slows the collapsed model more than the fresh one: 5.75x (800->4,600)
  vs 1.8x (2,200->4,000). n=1 per cell; SIBLING-TABLE CAVEAT (collapsed on the
  seed-100 table, fresh on a seed-300 sibling) until the item-2 replication on
  the original table lands.
- Under v-noise the trained model's re-learning advantage is "eliminated
  (possibly inverted)."

## Claims NOT permitted
- A standalone inversion finding (collapsed 4,600 > fresh 4,000). The 15% gap is
  smaller than this system's ~20% seed-to-seed wobble and sits across two
  tables; it can flip on rerun. Do NOT headline it.
- "Noise specifically targets the trained model's advantage" as CAUSAL language.

## Mandatory alternative reading (EQUAL weight)
Full-strength v-noise may simply impose a common onset floor (~4,000-4,600)
regardless of starting state. This fits the SAME four numbers as the erasure
story: both starts converge to ~the same noisy-onset time. **This 2x2 cannot
distinguish "noise erases the head start" from "a universal noise speed-limit."**
Any writeup of these cells must state this explicitly and give the speed-limit
rival equal prominence. The erasure reading is licensed only if a later
manipulation breaks the tie (not available here).

## Why registered now
The inversion line is the most quotable in the update — precisely the failure
mode behind the three earlier retractions (round-trips, held-capability,
spontaneous-recovery). Freezing the permitted/forbidden/alternative language
before the replication keeps the honest framing regardless of what item 2
returns.
