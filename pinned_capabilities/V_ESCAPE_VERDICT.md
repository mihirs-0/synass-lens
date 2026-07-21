# Verdict memo — weak-noise V-arm escape trend (2026-07-20)

Amendment `V_ESCAPE_PREREG.md`, commits abf7a140 (prereg) .. 0af67b9 (fast-stop).
Runs on MPS from the collapsed state s0 (dev 0.0125/stream_00 step-16k), V arm
(m<-g clean, v<-(g+s*xi)^2, matched isotropic xi), one seed per s.

## Result

| s (noise) | τ_onset | note |
|-----------|---------|------|
| clean (→0) | 800 | C0 reference (re-solves) |
| 0.25 | 3,400 | |
| 0.50 | 3,600 | |
| 0.75 | 4,600 | |
| **1.00** | **4,600** | **MEASURED, full strength** |

τ_onset = first step with first-token full-vocab CE < 3.0 sustained ≥ 500 steps.

## The single sentence the evidence supports

**Full-strength second-moment noise does not trap the model permanently — it
retards reacquisition by ~5.75× (escape onset step 800 → 4,600); the sealed
decomposition's "trapped" verdict was an under-observation artifact of its
2,000-step horizon.**

## Trapped vs. slow: SLOW, NOT TRAPPED

The registered decomposition (ADAM_DECOMP_RESULTS.md) ran the full-strength V
and N arms to step 2,000, saw them flat at the ln36 floor, and called them
"trapped/censored." Run to 5,300 steps here, the same V recipe at s=1.0
**escapes at onset 4,600** (CE crosses 3.0 at 4,600 and falls to 1.36). Escape
begins 2.6× past where trapping was declared. One escape falsifies permanence:
the state is metastable with a long dwell, not a permanent trap.

## What survives, what reframes

- **Mechanism claim SURVIVES:** noise routed through Adam's second moment (v) is
  still what causes the delay — remove it (clean C0) and escape is 800; add it
  (V, s=1.0) and escape is 4,600. The v-channel is load-bearing for the
  *retardation*. N_BC/M divergence unchanged.
- **"Trap/permanent" REFRAMES to "large finite retardation."** The headline
  becomes a measured slowdown (~5.75×), not a permanent capability loss —
  exactly the audit's pre-committed Row-1=SLOW outcome.
- **Dose-response SATURATES:** τ climbs 800→3,400 (huge jump at first noise),
  then 3,400→3,600→4,600→4,600 — nearly flat for s≥0.75. Escape time is set
  by an intrinsic reorganization cost once noise is present, not by noise
  magnitude. Log-log slope b=0.24 (NOT the b≈1 the linear-slowdown hypothesis
  predicted → escape is not simply 1/SNR).
- **Kick-back mechanism (Phase-2 F4):** all runs escape via oscillation — noise
  spikes repeatedly knock CE back above 3.0 (e.g. s=0.25 step 3900: 98.9% →
  step 4000: 13%) before stabilizing. KICK-BACK, not smooth freezing.

## Decision table (rows this package can fill)

| # | Question | Observed | Verdict |
|---|----------|----------|---------|
| 1 | Trapped vs slow | s=1.0 escapes at onset 4,600 (< 5,300 run) | **SLOW, NOT TRAPPED** |
| 4 | Mechanism (kick-back vs freezing) | oscillatory escape, noise-spike kick-backs | **KICK-BACK** |
| 6 | SNR/1-over-SNR scaling | log-log b=0.24, saturating; not b≈1 | **INCONSISTENT w/ 1/SNR** (retardation is not linear-slowdown) |

Rows 2 (scalar-LR control), 3 (fork replication seeds×ages), 5 (N→M
dose-response) not in this scoped package.

## Caveats (honest)

- **n=1 seed at s=1.0.** One escape is sufficient to falsify "permanent trap,"
  but the exact τ_onset=4,600 needs replication before it's a quantitative law.
- τ_solve (full 99.9% retrieval) not reached under the fast-stop (onset+200);
  τ_onset is the registered endpoint and is what the trend uses. The secondary
  τ_solve column is noisy.
- Escapes are oscillatory; τ_onset (CE<3.0 sustained 500) is robust to the
  kick-backs by construction, but single-seed onset has ~±200-step grid noise
  (s=0.75 and s=1.0 both round to 4,600).

## Artifacts
- `results/v_escape/v_escape_tau_table.csv` — per-run τ table
- `results/v_escape/v_escape_trend.png` — escape curves + τ(s)
- `results/v_escape/s*/metrics.jsonl` — per-run trajectories (CE, accuracy)
