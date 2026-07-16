# Gate 0-E run log

Chronological record of Branch A execution. Artifacts live outside git and
are bound here by SHA-256. No decision rule appears here; rules live in the
frozen amendments.

## 2026-07-16 06:40 — dev curve sealed (seed 100, batch 128)

56/56 streams, 16,000-step holds. `curve.json` SHA-256 recorded below.

| eta | erased | retained | unresolved | diverged | P (primary) | P (sensitivity) | median tau |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 0 | 8 | 0 | 0 | 0.000 | 0.000 | — |
| 0.005 | 1 | 1 | 6 | 0 | 0.125 | 0.875 | 100 |
| 0.008 | 6 | 0 | 2 | 0 | 0.750 | 1.000 | 100 |
| 0.0125 | 8 | 0 | 0 | 0 | 1.000 | 1.000 | 100 |
| 0.02 | 8 | 0 | 0 | 0 | 1.000 | 1.000 | 100 |
| 0.032 | 8 | 0 | 0 | 0 | 1.000 | 1.000 | 200 |
| 0.05 | 6 | 0 | 0 | 2 | 1.000 | 1.000 | 850 |

- Curve VALID (7 eligible rates; 0.05 at 25% divergence, below the 50%
  unstable bar). Monotone, no flags.
- eta50 = 0.00672, stream-bootstrap 95% CI [0.00559, 0.00753]
  (1,190/2,000 valid replicates). Logistic slope 7.17 in ln eta.
- **Batch-test rates (frozen rule, dev P nearest 0.25/0.75): 0.005 and
  0.008.**
- The transition zone is narrow (about 1.6x in eta) but genuinely mixed
  inside: at 0.005 the eight fates split 1 erased / 1 retained / 6
  unresolved. The primary-versus-sensitivity gap concentrates exactly there
  (0.125 vs 0.875), so the registered unresolved convention is
  decision-relevant and the sensitivity table ships with every report.
- Erasure is immediate-or-never within the hold: median tau is about 100
  steps at every erasing rate up to 0.02 (200 at 0.032; 850 at 0.05 where
  the hot shelf delays joint certification). The Arrhenius secondary is
  vacuous here (2 usable points). Fate appears to be decided within the
  first few hundred steps — a first-passage structure closer to
  kick-decides-it than to slow diffusive barrier crossing.

**Dwell audit, 56 trajectories:** recrossings occur ONLY at eta <= 0.005 —
every stream at 0.003 and 0.005 shows 1-2 expressed/suppressed round trips
(transient collapses and recoveries at rates the curve calls retained),
while all 40 streams at eta >= 0.008 are strictly one-way. Graze fraction
0.92. Basin language remains unearned at the behavioral-coordinate level
(interpretation registry rules stay in force); the reacquisition phenomenon
is ubiquitous below the boundary and absent above it.

**Dev null scan (seed 100, B=128):** all 7 rates certified; augmented
radius monotone 1.053 -> 1.307 across the grid. Moment refresh preserved
the solved state (theta drift 1.1%, loss ~3e-4). Diffusion ladder
D_1 2.5e-6, D_4 8.5e-6, D_16 2.1e-5, D_64 2.0e-5 — positive update
correlation saturating near L=16-64, validating the block-diffusion design;
15 decision blocks.

Execution note: the first pool was stopped by the runtime at ~00:50 after
about 4 hours (background-task lifetime); detached workers survived and the
relaunch resumed from transactional checkpoints with zero scientific loss.
Long stages now run detached.

Artifacts (SHA-256):
- `7d3c5d40c7f9fbc25ba45d9e54057cc60d86ec261d6a2084ac396d74b2ab5d00` pinned_capabilities/results/gate0e_dev_curve_seed100/curve.json
- `203f7ec880d734038eec62da844e914140313cfc57761cbfdee131291dc47a92` pinned_capabilities/results/gate0e_dwell_audit_dev/audit.json
- `8ebe040989ca7666a5c0883f2f2c4bed9bf2807430f17d551ffa74f685451869` pinned_capabilities/results/gate0e_null_seed100_b128/null_scan.json
