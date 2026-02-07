# Candidate-Set Normalized Evaluation (logK Exposure)

## Why this exists
The original `first_target_loss` metric plateaus at approximately `log(36) ≈ 3.58`
because the model is predicting a character from a 36‑symbol alphabet. That
entropy floor depends on the *token vocabulary size*, not on the *number of
candidates* `K`. During the disambiguation lag phase, this makes it impossible
to tell whether the model is treating `K=10`, `K=20`, or `K=36` candidates
equally — they all look like `log(36)`.

This change introduces a candidate‑set normalized loss whose floor is exactly
`log(K)` when the model ignores `z`, and goes to `0` when the model uses `z`.
That directly exposes the logK limit.

## Core idea
For each `(B, z)` pair, score **all K candidate A strings** that correspond to
the same `B` in the mapping data. For each candidate `A^(k)`:

1. Build the full sequence `<BOS> B <SEP> z <SEP> A <EOS>`.
2. Compute the **sequence log‑probability** of the target tokens
   (all `A` tokens **plus EOS**) by summing log‑probs:
   ```
   s_k = sum_{pos in target} log p(token_pos | prefix)
   ```
3. Renormalize across the K candidates:
   ```
   log_normalizer = logsumexp([s_0, ..., s_{K-1}])
   normalized_k = s_k - log_normalizer
   ```

Then:
```
candidate_loss = -normalized_{correct}
candidate_accuracy = argmax(normalized) == correct_index
```

**Key invariant:** If the model cannot use `z`, all candidates are equally
likely and the loss is `log(K)`. If the model uses `z`, the correct candidate
dominates and the loss → 0.

## Implementation details (what matters)
- **No padding or attention masks:** all sequences are length 16 for the fixed
  `(b, z, a)` lengths.
- **Correct indexing:** logits at position `pos-1` predict token at position
  `pos`. We sum over `target_start ... target_end` (inclusive of EOS).
- **Renormalization in float64:** we aggregate sequence log‑probs in the model’s
  dtype, then convert the per‑candidate totals to float64 **on CPU** for
  `logsumexp` stability (and MPS compatibility).

Relevant code:
- `src/analysis/candidate_eval.py::score_candidate_sequences`
- `src/analysis/candidate_eval.py::run_candidate_eval`

## z‑usage metrics and binding onset
We reuse existing trainer utilities:
- `compute_z_usage_metrics` calls `compute_loss` on clean data and on a
  batch with `z` shuffled across examples, then computes:
  ```
  z_gap = loss_z_shuffled - loss_clean
  ```
- `detect_binding_onset` finds the first step where `z_gap` exceeds a threshold
  for N consecutive checkpoints.

## Outputs
`scripts/run_candidate_eval.py` writes:
`outputs/{experiment}/candidate_eval_results.json` with:
- `candidate_loss`, `candidate_accuracy`, `candidate_top3_accuracy`
- `loss_clean`, `loss_z_shuffled`, `z_gap`
- `mean_correct_log_prob`, `mean_incorrect_log_prob`
- `binding_onset_step` and config
- `candidate_loss_plateau_avg` and `candidate_loss_post_binding_avg`

## Plotting
`scripts/plot_candidate_eval.py` generates:
- `candidate_eval_dashboard.png` (2×2 dashboard)
- `candidate_eval_z_usage.png` (z‑usage + first‑target loss)
- `candidate_eval_candidate.png` (candidate loss + accuracy)

The **key plot** is candidate loss:
- Plateau at `log(K)` during the lag phase.
- Sharp drop when binding emerges.
- Convergence to ~0 after disambiguation.

## Tests (what each test ensures)
`scripts/test_candidate_eval.py` includes:
- **Test A: Logprob Consistency** — confirms sequence scoring is indexed
  correctly. This is the critical correctness test.
- **Test B: Candidate Loss Floor** — checks the loss is within a wide band
  around `log(K)` at random init (random biases can shift it, so tolerance is
  intentionally broad).
- **Test C: Candidate Set Integrity** — ensures candidates come from the same
  `B` mapping entry.
- **Test D: Sequence Length Uniformity** — validates the no‑padding assumption.
- **Test E: EOS Token Handling** — validates label and EOS positions.

## How to run
Example (K=20):
```
python scripts/run_candidate_eval.py --experiment temp_lr1e3_bs128_k20 --n-examples 32 --every-n 2
python scripts/plot_candidate_eval.py --experiment temp_lr1e3_bs128_k20
```

This will generate the dashboard figures under:
`outputs/{experiment}/figures/`

