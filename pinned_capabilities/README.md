# Pinned capabilities experiment suite

This package implements the staged falsification program in
[`PROTOCOL.md`](PROTOCOL.md). It is isolated from the older `eta_sweep` scripts
so that prospective decisions, pilot evidence, and legacy analyses cannot be
silently mixed.

[`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) records the field-level novelty bar as
of 2026-07-15. In particular, hysteresis and noise-driven metastable escape in
deep linear networks are prior art; the suite must add nonlinear behavioral
state, complete memory localization, mechanism discrimination, or useful
training-time control.

[`CALIBRATION_LOG.md`](CALIBRATION_LOG.md) records instrument failures,
corrections, compute timing, and non-oracle numerical checks. Calibration
results are explicitly excluded from gate confidence intervals.

The suite starts from behavioral measurements, not a parameter-space
"capability direction." Its primary metric is a union-centered 2x2
interaction contrast that cancels additive B-only and z-only shortcuts.

Administrative commands:

```bash
python -m pinned_capabilities show-config
python -m pinned_capabilities freeze pinned_capabilities/results/manifest.json
python -m pinned_capabilities deep-linear --output pinned_capabilities/results/gate0_positive_control
python -m pinned_capabilities gate0-calibrate --help
python -m pinned_capabilities smoke
python -m pinned_capabilities reference-smoke
python -m pinned_capabilities reference --seeds 110 111 112 113 114 115 116 117 118 119
python -m pinned_capabilities prepare-suppressed --help
python -m pinned_capabilities prepare-expressed --help
python -m pinned_capabilities hysteresis --help
python -m pinned_capabilities local-stability --help
python -m pinned_capabilities local-stability-scan --help
python -m pinned_capabilities erasure-scan --help
python -m pinned_capabilities erasure-boundary --help
python -m pinned_capabilities acquisition --help
python -m pinned_capabilities acquisition-scan --help
python -m pinned_capabilities gate0-analyze --help
python -m pinned_capabilities memory-factorial --help
python -m pinned_capabilities transition-time --help
python -m unittest discover -s pinned_capabilities/tests -v
```

The three multi-rate Gate 0 scans accept `--workers 2` to run independent rate
cells in spawned processes. Worker count is an execution-only control: it is
not part of the frozen scientific manifest, completed cells are reused when a
scan is resumed, and aggregate output remains ordered by learning rate.

The deep-linear command is Gate 0's dual positive control: it recovers the
closed-form scalar-GD boundary and independently compares the production
augmented-Adam HVP Jacobian with centered finite differences on a tractable
six-dimensional optimizer state. The smoke
command constructs a fresh-seeded TransformerLens model, creates the two
disjoint counterfactual probes, evaluates the empirical input-blind `q*`, and
runs the shared deterministic training path end to end.

`reference-smoke` additionally exercises the solved-endpoint hold, seed-level
aggregation, and complete solved snapshots on a small system. Its manifest is
explicitly labeled as infrastructure validation, not gate evidence.

Every experiment cell receives an immutable JSON manifest containing the
complete configuration, its SHA-256 digest, and the source commit. Results are
never written into source directories and are ignored through the repository's
existing `results` convention once the gate runners are added.
Consuming manifests also bind reference, snapshot, raw-scan, and source-
manifest bytes. Scan aggregates must equal their child result files; every
child result and its control/source provenance is self-sealed and revalidated
at calibration and verdict time. New official state snapshots embed their full
experiment configuration so seed and batch identity is mandatory before a
branch runs. The sole exception is the explicitly recorded legacy seed-100
calibration snapshot.

`gate0-analyze` takes a small JSON index whose cells name a seed, batch size,
direction, empirical `scan.json`, and local-stability `scan.json`. It derives
the registered intervals from those raw files, computes the paired residuals,
and returns both a scientific evidence label and an investment action:
`continue` on registered non-reduction, `collect` while the finite registry is
incomplete, and `stop` when the complete registry does not justify Gate 1.
Registry completion is derived from the evidence file; there is no discretionary
finalization switch. Hand-entered boundary estimates are rejected by the official CLI.

```json
{
  "cells": [{
    "seed": 0,
    "batch_size": 128,
    "direction": "erasure",
    "empirical_scan": "pinned_capabilities/results/gate0_seed0_batch128_erasure/scan.json",
    "local_scan": "pinned_capabilities/results/gate0_seed0_batch128_local/scan.json"
  }]
}
```

The enforced Gate 0 order is: write the positive control; run the seed-100
erasure scan; run `gate0-calibrate` without `--local-scan` to close a missing
bracket immediately; only after a passing precheck run the seed-100 local scan
and create `gate0-calibrate`'s passing artifact; prepare the
seed-0/batch-128 expressed state; complete its local scan; run its erasure
scan; and apply `gate0-analyze`. A `continue` report opens Gate 1.
A `collect` report authorizes the remaining registered Gate 0 cells. Official
fate scans reject a missing matching local scan, and Gate 1 commands reject
anything except a content-bound `continue` report.

Implementation order is deliberately strict:

1. Metrics, reference bands, state labels, and manifests.
2. Gate 0: reduction to standard optimizer stability.
3. Gate 1: hysteresis and memory-location surgery.
4. Gate 2: module escape thresholds, only after Gate 1 passes.
5. Gate 3: fixed-rule early warning, using logs collected from the beginning.

Gate 1 uses fixed-rule dwell records. A level is boundary-eligible only after
both disjoint probes meet the registered 2,000-step slope tolerance. The
snapshot crossing utility treats Adam moments, step counters, scheduler,
gradient scaler, RNG, and deterministic data cursor as one optimizer-state
source, so the memory-location factorial does not silently mix histories.
Each factorial arm checkpoints transactionally every 1,000 steps and resumes
from its last complete checkpoint; completed arms are never rerun.
The 40,000-step transition-timing arms use the same exact-resume discipline,
including optimizer-reset state and the sustained-expression clock.

Gate 0 erasure branches likewise reload one complete expressed snapshot. A
branch counts as erased only after sustained joint return to the order-zero
interaction band and the empirical input-blind loss band. High-rate divergence
is a separate outcome and invalidates any bisection path that encounters it.
The resumable coarse scan reports a bracket only when adjacent tested rates are
strictly classified as retained and erased; unresolved branches are never
silently promoted to retained endpoints.
The registered verdict uses the geometric center of this frozen-grid interval.
`erasure-boundary` performs optional post-gate refinement, requires a rederived
Gate 0 `continue` artifact, and cannot replace or repair a registered scan.
Single-rate local and acquisition commands are exploratory and reject official
seed 0--4 work so they cannot leak outcomes around the registered scan order.
Each 8,000-step erasure branch also checkpoints transactionally, so scan-level
resumption never restarts a partially completed fate trajectory.
The 40,000-step acquisition branches use the same transactional checkpointing
and multi-rate reuse. Their aggregate preserves transition events, censored
timeouts, and divergence as separate outcomes; it never turns a timeout into a
stable-state label.

All state-preparation, Gate 0, hysteresis, memory-surgery, and transition-time
commands accept `--batch-size`. The override is embedded in the immutable
manifest so the registered `(32, 128, 512, 2048)` mechanism cells cannot be
confused with post-hoc reruns.
State preparation itself checkpoints every 1,000 steps with alternating
transactional slots, preserving the exact optimizer, RNG, and data cursor on
resume. Each progress record binds its checkpoint and exact metrics prefix;
completed results bind all scientific inputs and the final snapshot bytes.
