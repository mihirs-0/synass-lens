# Pre-registration: reacquisition 2x2 + full-jitter replication (2026-07-21)

Frozen BEFORE any outcome. Post-audit package for the capability-persistence
note. Resolves (a) whether the ~5.75x noise-induced slowdown is the whole
full-jitter story or just the v-channel isolation, and (b) whether the collapsed
model's re-learning advantage is a real head-start that noise erases, or an
artifact of comparing across learning rates. Harness `reacq_2x2_run.py`
(generalizes the sealed V-arm harness; unit-equivalent core, smoke-tested
collapsed CE 3.585 / fresh CE 3.965 / all three arms step clean, no NaN).

## The 2x2 (all eta=0.0125, full-gradient) + N replication

|            | clean (C0)          | v-noise (V)          | both-moments (N)          |
|------------|---------------------|----------------------|---------------------------|
| collapsed  | 800  (HAVE)         | 4,600 (HAVE, n=1)    | **RUN: 3 seeds, cap 16k** |
| fresh      | **RUN: 1, cap 10k** | **RUN: 1, cap 10k**  | -                         |

Why fresh+clean is load-bearing: without it, a stuck fresh+v-noise is
uninterpretable (did v-noise defeat it, or can 0.0125 not train from scratch
even clean?). Nothing on disk has it — the fresh-init exploratory runs were
MINIBATCH (both channels, magnitude set by large fresh gradients), and the only
clean-full-batch check was from a SOLVED state, not fresh init.

## Arm recipes (per-step MATCHED isotropic noise, S=1.0)

xi_t isotropic, ||xi_t|| = ||(g_B1 - g_B2)/sqrt2|| at the CURRENT state (so the
v-noise magnitude is native to each init's own minibatch jitter — this is the
fair per-state matched-magnitude v-only channel, and directly answers the
magnitude-mismatch objection to reusing the native-minibatch fresh data).
  C0: m<-g,      v<-g^2
  V : m<-g,      v<-(g+xi)^2
  N : m<-g+xi,   v<-(g+xi)^2
theta <- theta*(1-eta*lambda) + u ; u bias-corrected AdamW (t_eff=t0+step).
collapsed: t0=22400, m0/v0 from s0. fresh: t0=0, m=v=0, model+table seed=300
(SAME weights+table for its clean & v cells; only the noise stream differs).

## Endpoints (frozen) + PRE-COMMITTED CENSORING

- tau_onset = first step CE(first-token,10k keys) < 3.0 sustained >= 500 steps.
- tau_solve = first step >= 0.999 exact retrieval on all 10k, confirmed next eval.
- **Run to CAP. If no tau_onset by CAP -> censored; report as a BOUND
  (escape > CAP). NO ad-hoc extension.** (The real-minibatch aging run showed no
  escape through ~16k, so N may not land near 4,600 — the 16k cap and this
  censoring rule are set now, before seeing N.)
- Fast-stop at tau_onset+200 only if onset is reached (captures the metric;
  escapes are oscillatory so solve is noise-dominated and not chased).

## The 5 runs

1-3. N, collapsed, seeds {0,1,2}, cap 16,000.
4.   C0, fresh, seed 0, cap 10,000.
5.   V,  fresh, seed 0, cap 10,000.

## Pre-committed interpretations (fixed before outcomes)

**N replication (honest full-jitter number):**
- N onset ~ 4,600 (= V) across seeds -> the v-channel reproduces the ENTIRE
  full-jitter slowdown; report slowdown 800->~4,600 (~5.75x) with the seed error
  bar; the mechanism headline (second-moment channel) carries the whole effect.
- N onset >> 4,600 -> the m-moment adds slowdown beyond the v-channel; report
  both and soften the "v explains everything" claim.
- N censored at 16k (>=1 seed) -> report escape > 16k as a bound for that seed;
  the slowdown is >= 20x and only lower-bounded; add to the survival table.

**Fresh 2x2 (head-start vs rate artifact):**
- fresh+clean escapes (<=10k) AND fresh+v-noise stuck/much-slower -> v-noise
  specifically obstructs fresh learning; the collapsed head-start is real and
  only PARTIALLY erased by noise (collapsed+v still 4,600 << fresh+v). Thesis:
  "noise erodes but does not eliminate the trained advantage."
- fresh+clean ALSO stuck (<=10k, no onset) -> 0.0125 cannot train from scratch
  even clean; the fresh-vs-collapsed gap is a learning-RATE effect, not a noise-
  erased head-start; the "noise erases the advantage" framing is BARRED. Report
  plainly: collapsed learns at a rate fresh cannot, clean or noisy.
- both fresh cells escape at similar times -> no head-start distinction at this
  rate; report.

## Free items (no new long runs)
- Resume the sealed s=1.0 V ckpt (step 5300) to the frozen sustained solve
  criterion; show it does not fall back through the oscillations.
- Deviation paragraph: the earlier full-jitter run stopped at 2,000 per the
  registered T 3500->2000 compute deviation (ADAM_DECOMP_PREREG.md), NOT an
  outcome-driven stop.
- Add the ~16k real-minibatch aging row (recovery races: no onset through ~16k
  at the training rate) to the survival table as an independent reference.

## Compute
MPS ~1.7 s/step (HookedTransformer forward CPU-index bug -> no concurrent MPS;
runs serial). Realistic (N escapes ~5k, fresh ~10k): ~17 h. Worst case (all
censored to cap): ~32 h. Detached, resumable, fp32 optimizer state. Fresh runs
may be offloaded to CPU (~4.7 s/step) to parallelize if wall-clock demands.
