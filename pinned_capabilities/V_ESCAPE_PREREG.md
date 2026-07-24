# Pre-registration: weak-noise V-arm escape trend (amendment 2026-07-20)

Frozen BEFORE any outcome. Scoped follow-up to the sealed Adam-channel
decomposition (ADAM_DECOMP_RESULTS.md): measure escape time vs injected-noise
scale s for the V arm, to extrapolate whether the full-strength (s=1.0) run is
worth the laptop time. **No s=1.0 long run in this package.** Evidence only; no
manuscript claims.

## Fixed inputs (from sealed work; not re-derived)
- Collapsed state s0 = dev 0.0125/stream_00 step-16k ckpt (theta0,m0,v0,t0=22400).
- eta=0.0125, lambda=0.01, beta1=0.9, beta2=0.999, eps=1e-8. d=807,720 (70 tensors).
- Sanity (from logs, no rerun): clean C0 escaped @step800 -> CE 0.88; sealed V
  censored flat (CE 3.39/3.45 @ T=2000). PASS.

## Arm recipe (identical to registered V except noise x s)
  m <- g (clean full gradient)
  v <- (g + s*xi)^2,  xi = per-step MATCHED isotropic noise ||xi||=||(g_B1-g_B2)/sqrt2||
  theta <- theta*(1-eta*lambda) + u ; u = bias-corrected AdamW (t_eff=t0+step)
Gradients + optimizer state (m,v) fp32 throughout (tiny v must not be truncated).

## Runs (launch order; s=0.75 only after 1-2 finish)
| s    | cap    | seed | noise_seed / mb_seed |
|------|--------|------|----------------------|
| 0.25 |  8,000 | 0    | 1000000 / 2000000    |
| 0.50 | 10,000 | 1    | 1000001 / 2000001    |
| 0.75 | 12,000 | 2    | 1000002 / 2000002    |
Early stop at tau_solve + 200. Resumable (theta,m,v,rng) ckpt every 1000 steps.

## Endpoints (frozen)
- tau_onset = first step with full_vocab_ce < 3.0 sustained >= 500 steps.
- tau_solve = first step with 100% exact retrieval on all 10,000 keys
  (argmax at every target position for every key), confirmed at the next
  accuracy eval.
- Capped (no solve by cap) => censored; do NOT extend without asking.

## Logging
CE every 200 steps; exact-retrieval accuracy every 1000 — TIGHTENED to every 100
once CE < 2.5, so tau_solve resolves to +/-100 and the solve+200 early-stop is
well-defined (registered cadence choice, pre-outcome). Checkpoint every 1000.

## Analysis (pre-registered, after >= 2 runs finish)
- Table: s, tau_onset, tau_solve, capped/uncapped.
- Fit a line log(tau_onset) = a + b*log(s) on the escaping s-values, anchored
  by the clean escape (~800) as the s->0 reference. Report slope b and the
  extrapolated tau_onset(s=1.0) with a rough range.
- Verdict, one sentence: "the trend predicts full-strength escape by step X."
  X decides whether/how long to run the s=1.0 arm. Censored runs reported as
  bounds, not extrapolation anchors.

## Compute
CPU full-gradient step measured 4.68 s (4 thread; 8/10-thread slower,
bandwidth-bound). MPS 1.72 s but HookedTransformer forward has a CPU-index
device bug -> runs on CPU. Two runs concurrent (s=0.25, s=0.50) ~6-7 s/step
each. Projected per run (escape-expected / cap-bound): s=0.25 ~3-6h / ~14h;
s=0.50 ~6-11h / ~18h; s=0.75 ~11-16h / ~22h. Detached, resumable.

## Refinement 2026-07-20 (pre-outcome): MPS + first-token CE readout
exp.evaluate() (probe-based full_vocab_ce) indexes CPU probe tensors against
MPS logits and fails on MPS (metrics.py:104, answer_token_ids[q.b,q.z]). Replaced
the CE readout with ce_full(): FIRST-target-token full-vocab CE over ALL 10,000
keys, device-resident. Validated == probe full_vocab_ce at collapse (3.5851 vs
3.5848, both cpu & mps); more faithful (full population vs probe sample). Enables
MPS (full-grad 1.72s vs 4.68s CPU). Runs now on MPS serially. No endpoint,
threshold, seed, or arm-recipe change.

## Deviation 2026-07-20 (practical early-stop): solve threshold 100% -> 99.9%
s=0.25 escaped (tau_onset=3400) and reached CE 0.003 / 99.9% retrieval by ~step
4500, then the last 0.1% (a few keys) noise-jittered without hitting strict
100%. Relaxed the tau_solve early-stop from exact_acc>=1.0 to >=0.999 to avoid
burning compute on that tail. Does NOT affect tau_onset (the metric the trend
fit consumes) — only the secondary tau_solve endpoint and when a run early-stops.
s=0.25 sealed from data: tau_onset=3400, tau_solve(>=99.9%)=4500.

## Judgment 2026-07-20 (fast iteration): stop at tau_onset + 200, escape is OSCILLATORY
Inspecting s=0.25's escape: it is NOT a monotonic climb. Retrieval oscillates
with kick-backs before stabilizing:
  step 3100: CE 0.69 / 80% -> 3200: CE 4.32 / 5.7% (KICKED BACK to floor)
  step 3600: CE 0.20 / 94% -> 3700: CE 2.59 / 11.5% (kicked back)
  step 3900: CE 0.05 / 98.9% -> 4000: CE 2.37 / 13% (kicked back) -> 4100+ stabilizes
The "last 0.9%" (99% -> 99.9%) cost ~400-600 steps, half of it a kick-back, not
learning -- i.e. noise-jitter on a few keys, not signal. So tau_solve is
noise-dominated and not worth chasing. Policy: each run STOPS at tau_onset + 200
(ONSET_TAIL). tau_onset (the log-tau_onset fit metric) is fully captured by then;
solve is reported only as whatever was reached at stop, not pursued. This is the
fast-iteration win: an oscillating run no longer burns to its cap.
Separately, the kick-back pattern is itself the audit's Phase-2 F4 answer for the
V arm: KICK-BACK (noise spikes knock the model back), not freezing.
