"""
Option-(b) sequence reversal for the dataset-reversal experiment.

Given a tokenized example from DisambiguationDataset (created by the
existing src/data pipeline, untouched), this module produces an
equivalent example where the SEMANTIC CONTENT between BOS and EOS is
reversed, but BOS and EOS remain at the natural first/last positions.

Original (bz_to_a, example shapes for B=6, z=2, A=4):
    positions:   0    1..6   7    8,9   10   11..14  15
    input_ids: [BOS, b1..b6, SEP, z1,z2, SEP, a1..a4, EOS]
    labels:    [-100, -100×6, -100, -100×2, -100, a1..a4, EOS]

After option-(b) reversal (reverse indices 1..-2 inclusive):
    positions:   0    1..4    5    6,7   8    9..14   15
    input_ids: [BOS, a4..a1, SEP, z2,z1, SEP, b6..b1, EOS]
    labels:    [-100, a4..a1, -100, -100, -100, -100×6, EOS]

Target positions for the cross-entropy loss now live at indices 1..4
(what used to be positions 11..14), plus EOS at the end — same
count (5 scored positions) as the forward task.

The z_position / z_end_position / target_start / target_end fields
in the tokenized dict are recomputed for the reversed layout so
downstream code (shuffle_z_in_batch, candidate_eval, etc.) still
points at the right spans.
"""

from __future__ import annotations

from typing import Dict

import torch


def reverse_tokenized(item: Dict) -> Dict:
    """Return a new tokenized dict with option-(b) reversal applied.

    The input dict is the per-example format produced by
    `DisambiguationDataset._precompute_tokens()`:
      {
        "input_ids": torch.long[L],
        "labels":    torch.long[L],
        "z_position":        int,
        "z_end_position":    int,
        "target_start_position": int,
        "target_end_position":   int,
        "b", "z", "a", "base_string": str,
      }

    We reverse slice [1:L-1] of input_ids (and rebuild labels the
    same way) and shift the span markers to the new coordinates.
    """
    input_ids = item["input_ids"]
    labels = item["labels"]
    L = int(input_ids.shape[0])

    if L < 3:
        # Degenerate — just return unchanged.
        return dict(item)

    # Reverse the middle (indices 1..L-2 inclusive, i.e. slice 1:-1).
    new_input_ids = input_ids.clone()
    new_labels = labels.clone()
    new_input_ids[1:-1] = input_ids[1:-1].flip(0)
    new_labels[1:-1] = labels[1:-1].flip(0)

    # Remap span markers.
    # Any original position p in [1..L-2] becomes new position (L-1) - p.
    def _map(p: int) -> int:
        if p < 0:
            return p  # -1 sentinel for tasks without z
        if p == 0 or p == L - 1:
            return p  # boundary tokens untouched
        return (L - 1) - p

    old_z_start = int(item.get("z_position", -1))
    old_z_end = int(item.get("z_end_position", -1))
    old_t_start = int(item.get("target_start_position", 0))
    old_t_end = int(item.get("target_end_position", 0))

    if old_z_start >= 0 and old_z_end >= 0:
        # z occupies positions [old_z_start, old_z_end).  After reversal,
        # the first z token (originally at old_z_start) ends up at
        # position (L-1)-old_z_start, and the last z token (at old_z_end-1)
        # ends up at (L-1)-(old_z_end-1) = L-old_z_end.  So new span is
        # [L-old_z_end, L-old_z_start).
        new_z_start = L - old_z_end
        new_z_end = L - old_z_start
    else:
        new_z_start = old_z_start
        new_z_end = old_z_end

    # target span: [old_t_start, old_t_end) → [L - old_t_end, L - old_t_start).
    new_t_start = L - old_t_end
    new_t_end = L - old_t_start

    out = dict(item)
    out["input_ids"] = new_input_ids
    out["labels"] = new_labels
    out["z_position"] = new_z_start
    out["z_end_position"] = new_z_end
    out["target_start_position"] = new_t_start
    out["target_end_position"] = new_t_end
    return out


def reverse_dataset_inplace(dataset) -> None:
    """Apply reverse_tokenized to every pre-tokenized example in a
    DisambiguationDataset.  Modifies in place; the dataset's
    `self.tokenized` list is replaced with reversed copies.
    """
    if not hasattr(dataset, "tokenized"):
        raise ValueError(
            "dataset does not expose `.tokenized`; was it created by "
            "DisambiguationDataset from src.data.dataset?"
        )
    dataset.tokenized = [reverse_tokenized(item) for item in dataset.tokenized]


# ---------------------------------------------------------------------------
# Self-test: verify that the reversal preserves the right invariants.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run `python eta_sweep/reverse_dataset.py` to sanity-check the reversal
    # against a freshly-tokenized (B=6, z=2, A=4) example.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.tokenizer import CharTokenizer

    tok = CharTokenizer()
    ex = tok.encode_sequence("bbbbbb", "zz", "aaaa", task="bz_to_a")
    print("ORIG input_ids:", ex["input_ids"].tolist())
    print("ORIG labels:   ", ex["labels"].tolist())
    print("ORIG target span:", ex["target_start_position"], ex["target_end_position"])
    print("ORIG z span:", ex["z_position"], ex["z_end_position"])
    rev = reverse_tokenized(ex)
    print()
    print("REV  input_ids:", rev["input_ids"].tolist())
    print("REV  labels:   ", rev["labels"].tolist())
    print("REV  target span:", rev["target_start_position"], rev["target_end_position"])
    print("REV  z span:", rev["z_position"], rev["z_end_position"])

    # Invariants we care about:
    assert rev["input_ids"][0] == ex["input_ids"][0], "BOS must stay at position 0"
    assert rev["input_ids"][-1] == ex["input_ids"][-1], "EOS must stay at last position"
    # Target tokens must still be present at the new target span.
    original_targets = ex["input_ids"][ex["target_start_position"]:ex["target_end_position"]]
    new_targets_at_span = rev["input_ids"][rev["target_start_position"]:rev["target_end_position"]]
    assert torch.equal(original_targets.flip(0), new_targets_at_span), (
        f"expected target tokens to appear reversed at the new span; "
        f"got {new_targets_at_span} vs reversed {original_targets.flip(0)}"
    )
    # Labels must be -100 everywhere except the new target span + final EOS.
    L = int(rev["input_ids"].shape[0])
    for i in range(L):
        in_target = rev["target_start_position"] <= i < rev["target_end_position"]
        is_final = (i == L - 1)
        if in_target or is_final:
            assert rev["labels"][i] != -100, f"labels[{i}] should be scored"
        else:
            assert rev["labels"][i] == -100, f"labels[{i}] should be masked"
    print("\n[OK] self-test passed")
