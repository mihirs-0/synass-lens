# Pinned capabilities experiment suite

This package implements the staged falsification program in
[`PROTOCOL.md`](PROTOCOL.md). It is isolated from the older `eta_sweep` scripts
so that prospective decisions, pilot evidence, and legacy analyses cannot be
silently mixed.

The suite starts from behavioral measurements, not a parameter-space
"capability direction." Its primary metric is a union-centered 2x2
interaction contrast that cancels additive B-only and z-only shortcuts.

Administrative commands:

```bash
python -m pinned_capabilities show-config
python -m pinned_capabilities freeze pinned_capabilities/results/manifest.json
python -m pinned_capabilities deep-linear
python -m pinned_capabilities smoke
python -m pinned_capabilities reference-smoke
python -m unittest discover -s pinned_capabilities/tests -v
```

The deep-linear command is Gate 0's closed-form positive control. The smoke
command constructs a fresh-seeded TransformerLens model, creates the two
disjoint counterfactual probes, evaluates the empirical input-blind `q*`, and
runs the shared deterministic training path end to end.

`reference-smoke` additionally exercises the solved-endpoint hold, seed-level
aggregation, and complete solved snapshots on a small system. Its manifest is
explicitly labeled as infrastructure validation, not gate evidence.

Every experiment cell will receive an immutable JSON manifest containing the
complete configuration, its SHA-256 digest, and the source commit. Results are
never written into source directories and are ignored through the repository's
existing `results` convention once the gate runners are added.

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

Gate 0 erasure branches likewise reload one complete expressed snapshot. A
branch counts as erased only after sustained joint return to the order-zero
interaction band and the empirical input-blind loss band. High-rate divergence
is a separate outcome and is invalid as a bisection endpoint.
