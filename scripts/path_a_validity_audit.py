#!/usr/bin/env python
"""Path A validity audit — is the trained model using z, or filler, as the
lookup key?

Runs four checks on the short_s42 checkpoint (and any other completed
Path A seeds). Uses CPU to avoid MPS contention with the still-running
Path A training process.
"""

import sys
import json
import random
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings
from src.model import create_model_from_config
from src.training.checkpoint import load_checkpoint, list_checkpoints
from scripts.experiment_helpers import make_config
from scripts.path_a_transport import TransportDataset

DEVICE = "cpu"
K = 10
N_AUDIT = 1000
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "path_a_transport"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_for_run(experiment_name, seed, filler_head, filler_tail):
    cfg = make_config(experiment_name=experiment_name, task="bz_to_a",
                      k=K, seed=seed, n_unique_b=1000, max_steps=15000,
                      checkpoint_every=15000, eval_every=250,
                      early_stop_frac=0.005)
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=1000, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=seed, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    model = create_model_from_config(cfg, tokenizer).to(DEVICE)
    ckpt_dir = Path("outputs") / experiment_name / "checkpoints"
    steps = sorted(list_checkpoints(ckpt_dir))
    load_checkpoint(model, None, ckpt_dir, step=steps[-1])
    model.eval()
    ds = TransportDataset(mapping_data, tokenizer, filler_head, filler_tail,
                          seed, n_examples=N_AUDIT)
    return cfg, tokenizer, mapping_data, model, ds


def stacked_batch(items):
    return torch.stack([it["input_ids"] for it in items]), \
           torch.stack([it["labels"] for it in items]), \
           torch.tensor([it["target_start_position"] for it in items])


def predict_first_target(model, input_ids, target_starts, batch_size=64):
    """Return predicted token id for each example at position target_start-1."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            ids = input_ids[i:i+batch_size].to(DEVICE)
            ts = target_starts[i:i+batch_size].to(DEVICE)
            logits = model(ids)
            bi = torch.arange(ids.shape[0], device=DEVICE)
            pred = logits[bi, ts - 1, :].argmax(dim=-1)
            preds.append(pred.cpu())
    return torch.cat(preds)


def truth_first_target(labels, target_starts):
    truths = []
    for lab, ts in zip(labels, target_starts):
        truths.append(int(lab[int(ts)].item()))
    return torch.tensor(truths)


def check1_z_shuffle(ds, model, tokenizer, seed):
    """Swap z across examples within a batch, see if prediction changes."""
    items = [ds.tokenized[i] for i in range(len(ds))]
    input_ids, labels, ts = stacked_batch(items)
    orig_pred = predict_first_target(model, input_ids, ts)

    # Build z-swapped version: take each example's input, overwrite z with
    # another example's z (shifted by 1 in the list).
    swapped = input_ids.clone()
    # z occupies positions [z_position : z_position + z_length]
    z_positions = [it["z_position"] for it in items]
    z_lengths = [it["z_end_position"] - it["z_position"] for it in items]
    n = len(items)
    for i in range(n):
        j = (i + 1) % n  # donor
        zp_i, zl_i = z_positions[i], z_lengths[i]
        zp_j, zl_j = z_positions[j], z_lengths[j]
        # Only swap if same length (always true: z_length=2 for all)
        if zl_i == zl_j:
            swapped[i, zp_i:zp_i + zl_i] = input_ids[j, zp_j:zp_j + zl_j]
    swapped_pred = predict_first_target(model, swapped, ts)
    frac_changed = float((orig_pred != swapped_pred).float().mean())
    return {
        "fraction_changed_when_z_swapped": frac_changed,
        "n_examples": n,
        "orig_pred_accuracy": float((orig_pred == truth_first_target(labels, ts)).float().mean()),
    }


def check2_filler_shuffle(ds, model, tokenizer, seed, filler_head, filler_tail):
    """Swap filler across examples; does prediction change?"""
    items = [ds.tokenized[i] for i in range(len(ds))]
    input_ids, labels, ts = stacked_batch(items)
    orig_pred = predict_first_target(model, input_ids, ts)

    n = len(items)
    swapped = input_ids.clone()
    # Filler head occupies positions [1 : 1 + filler_head]
    # Filler tail occupies positions [z_end_position + 1 : z_end_position + 1 + filler_tail]
    for i in range(n):
        j = (i + 1) % n
        if filler_head > 0:
            swapped[i, 1:1 + filler_head] = input_ids[j, 1:1 + filler_head]
        if filler_tail > 0:
            ze_i = items[i]["z_end_position"]
            ze_j = items[j]["z_end_position"]
            # filler_tail starts right after SEP (which is at z_end_position),
            # so filler tail occupies [ze+1 : ze+1+filler_tail]
            swapped[i, ze_i + 1:ze_i + 1 + filler_tail] = \
                input_ids[j, ze_j + 1:ze_j + 1 + filler_tail]
    swapped_pred = predict_first_target(model, swapped, ts)
    frac_changed = float((orig_pred != swapped_pred).float().mean())
    return {
        "fraction_changed_when_filler_swapped": frac_changed,
        "n_examples": n,
    }


def check3_controlled_filler_candidate_acc(ds, model, tokenizer, mapping_data,
                                             seed, filler_head, filler_tail):
    """For each (B, z) pair in ds, build K=10 candidate continuations (one
    per A in mapping_data.mappings[B]) using the SAME filler, score each,
    check if the correct A gets highest total log-prob.
    """
    items = [ds.tokenized[i] for i in range(len(ds))]
    # Need the B string and A string for each item. Already stored.
    correct_count = 0
    total_count = 0
    batch_size = 32
    import torch.nn.functional as F
    with torch.no_grad():
        for i, it in enumerate(items):
            B = it["b"]; Z = it["z"]; A_true = it["a"]
            candidates = mapping_data.mappings.get(B, [])
            if len(candidates) < 2:
                continue
            A_list = [a for (_, a) in candidates]
            # Build K candidate sequences, SAME filler as the stored example.
            # Reconstruct the filler from the stored input_ids.
            ids = it["input_ids"].tolist()
            head = ids[1:1 + filler_head]
            ze = it["z_end_position"]
            tail = ids[ze + 1:ze + 1 + filler_tail]
            seqs = []
            for A_c in A_list:
                tokens = [tokenizer.bos_token_id] + head + tokenizer.encode(B) + [tokenizer.sep_token_id]
                tokens.extend(tokenizer.encode(Z))
                tokens.append(tokenizer.sep_token_id)
                tokens.extend(tail)
                target_start = len(tokens)
                target_tokens = tokenizer.encode(A_c)
                tokens.extend(target_tokens)
                tokens.append(tokenizer.eos_token_id)
                seqs.append((torch.tensor(tokens, dtype=torch.long), target_start,
                              target_tokens, A_c))
            # Score each candidate
            best_score = -float("inf")
            best_A = None
            cand_ids = torch.stack([s[0] for s in seqs]).to(DEVICE)
            # All same length because A length is fixed = 4 and filler same.
            logits = model(cand_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            for k, (ids_c, ts_c, target_tokens, A_c) in enumerate(seqs):
                # Sum log-prob of target tokens at positions [ts_c-1, ts_c, ts_c+1, ts_c+2] predicting tokens[ts_c..ts_c+3]
                lp_c = 0.0
                for p_pred, tok_id in enumerate(target_tokens):
                    lp_c += log_probs[k, ts_c - 1 + p_pred, tok_id].item()
                if lp_c > best_score:
                    best_score = lp_c
                    best_A = A_c
            total_count += 1
            if best_A == A_true:
                correct_count += 1
    return {
        "controlled_filler_candidate_accuracy": correct_count / max(total_count, 1),
        "n_examples": total_count,
    }


def check4_filler_corruption(ds, model, tokenizer, filler_head, filler_tail):
    """Replace filler with a fixed out-of-distribution token (PAD=0), compare
    held-out accuracy."""
    items = [ds.tokenized[i] for i in range(len(ds))]
    input_ids, labels, ts = stacked_batch(items)
    pred_intact = predict_first_target(model, input_ids, ts)
    truth = truth_first_target(labels, ts)
    acc_intact = float((pred_intact == truth).float().mean())

    corrupted = input_ids.clone()
    pad = tokenizer.pad_token_id
    n = len(items)
    for i in range(n):
        if filler_head > 0:
            corrupted[i, 1:1 + filler_head] = pad
        if filler_tail > 0:
            ze = items[i]["z_end_position"]
            corrupted[i, ze + 1:ze + 1 + filler_tail] = pad
    pred_corr = predict_first_target(model, corrupted, ts)
    acc_corr = float((pred_corr == truth).float().mean())
    return {
        "accuracy_intact": acc_intact,
        "accuracy_filler_corrupted": acc_corr,
        "ratio_corrupted_over_intact": acc_corr / max(acc_intact, 1e-9),
    }


def audit_one(experiment_name, seed, filler_head, filler_tail):
    print(f"\n=== auditing {experiment_name} (seed={seed}, filler=({filler_head},{filler_tail})) ===")
    cfg, tokenizer, mapping_data, model, ds = load_model_for_run(
        experiment_name, seed, filler_head, filler_tail)
    print(f"  dataset size: {len(ds)}")

    c1 = check1_z_shuffle(ds, model, tokenizer, seed)
    print(f"  [1] z-shuffle: P(pred changes | z swap) = {c1['fraction_changed_when_z_swapped']:.3f}  "
          f"(orig accuracy = {c1['orig_pred_accuracy']:.3f})")

    c2 = check2_filler_shuffle(ds, model, tokenizer, seed, filler_head, filler_tail)
    print(f"  [2] filler-shuffle: P(pred changes | filler swap) = {c2['fraction_changed_when_filler_swapped']:.3f}")

    c3 = check3_controlled_filler_candidate_acc(
        ds, model, tokenizer, mapping_data, seed, filler_head, filler_tail)
    print(f"  [3] controlled-filler candidate acc: {c3['controlled_filler_candidate_accuracy']:.3f}  "
          f"(n={c3['n_examples']})")

    c4 = check4_filler_corruption(ds, model, tokenizer, filler_head, filler_tail)
    print(f"  [4] filler corruption: intact={c4['accuracy_intact']:.3f}  "
          f"corrupted={c4['accuracy_filler_corrupted']:.3f}  "
          f"ratio={c4['ratio_corrupted_over_intact']:.3f}")

    return {"experiment": experiment_name, "seed": seed,
            "filler": (filler_head, filler_tail),
            "check1_z_shuffle": c1,
            "check2_filler_shuffle": c2,
            "check3_controlled_filler_cand": c3,
            "check4_filler_corruption": c4}


def verdict(a):
    c1 = a["check1_z_shuffle"]["fraction_changed_when_z_swapped"]
    c2 = a["check2_filler_shuffle"]["fraction_changed_when_filler_swapped"]
    c3 = a["check3_controlled_filler_cand"]["controlled_filler_candidate_accuracy"]
    c4 = a["check4_filler_corruption"]["ratio_corrupted_over_intact"]
    uses_z = (c1 > 0.7) and (c2 < 0.2) and (c3 > 0.7) and (c4 > 0.7)
    uses_filler = (c1 < 0.3) and (c2 > 0.5) and (c3 < 0.3) and (c4 < 0.3)
    if uses_z and not uses_filler:
        return "uses_z_properly"
    if uses_filler and not uses_z:
        return "uses_filler_as_key"
    return "mixed"


def main():
    # Find completed Path A checkpoints
    import re
    outputs = Path("/Users/mihir/synass-lens/synass-lens/outputs")
    runs = []
    for d in sorted(outputs.glob("path_a_*")):
        if not (d / "checkpoints").exists():
            continue
        ckpts = list((d / "checkpoints").iterdir())
        if not ckpts:
            continue
        m = re.match(r"path_a_(short|medium|long)_s(\d+)", d.name)
        if not m:
            continue
        variant, seed = m.group(1), int(m.group(2))
        fh, ft = {"short": (6, 0), "medium": (3, 3), "long": (0, 6)}[variant]
        runs.append((d.name, seed, fh, ft))

    print(f"Found {len(runs)} Path A runs with saved checkpoints:")
    for r in runs:
        print(f"  {r[0]}")

    if not runs:
        print("No completed Path A runs to audit. Exiting.")
        return

    audits = []
    for name, seed, fh, ft in runs:
        a = audit_one(name, seed, fh, ft)
        a["verdict"] = verdict(a)
        audits.append(a)

    # Write JSON + markdown
    with open(OUT_DIR / "validity_audit.json", "w") as f:
        json.dump(audits, f, indent=2, default=str)

    lines = ["# Path A validity audit\n"]
    lines.append("Pre-committed verdict rule:\n")
    lines.append("- `uses_z_properly`  iff  c1>0.7 AND c2<0.2 AND c3>0.7 AND c4>0.7")
    lines.append("- `uses_filler_as_key`  iff  c1<0.3 AND c2>0.5 AND c3<0.3 AND c4<0.3")
    lines.append("- `mixed`  otherwise\n")

    for a in audits:
        lines.append(f"\n## Section 1: {a['experiment']} (filler {a['filler']})\n")
        c1 = a["check1_z_shuffle"]
        c2 = a["check2_filler_shuffle"]
        c3 = a["check3_controlled_filler_cand"]
        c4 = a["check4_filler_corruption"]
        lines.append("### Check 1 — z-shuffle sensitivity\n")
        lines.append(f"- P(pred changes | z swapped) = **{c1['fraction_changed_when_z_swapped']:.3f}**  (n={c1['n_examples']})")
        lines.append(f"- Original prediction accuracy = {c1['orig_pred_accuracy']:.3f}")
        lines.append(f"- Under *uses z*: expect > 0.7. Under *uses filler*: expect < 0.3.\n")
        lines.append("### Check 2 — filler-shuffle sensitivity\n")
        lines.append(f"- P(pred changes | filler swapped) = **{c2['fraction_changed_when_filler_swapped']:.3f}**")
        lines.append(f"- Under *uses z*: expect < 0.2. Under *uses filler*: expect > 0.5.\n")
        lines.append("### Check 3 — controlled-filler candidate accuracy\n")
        lines.append(f"- accuracy on K=10 candidate selection with filler held fixed = **{c3['controlled_filler_candidate_accuracy']:.3f}**  (n={c3['n_examples']})")
        lines.append(f"- Under *uses z*: expect > 0.7. Under *uses filler*: expect < 0.3.\n")
        lines.append("### Check 4 — filler-corruption effect on accuracy\n")
        lines.append(f"- accuracy intact = {c4['accuracy_intact']:.3f}")
        lines.append(f"- accuracy with filler replaced by PAD = {c4['accuracy_filler_corrupted']:.3f}")
        lines.append(f"- ratio corrupted/intact = **{c4['ratio_corrupted_over_intact']:.3f}**")
        lines.append(f"- Under *uses z*: expect > 0.7. Under *uses filler*: expect < 0.3.\n")
        lines.append(f"## Section 2: Verdict for {a['experiment']}\n")
        lines.append(f"**`{a['verdict']}`**\n")

    lines.append("\n## Section 3: Implication for Path A\n")
    verdicts = [a["verdict"] for a in audits]
    if all(v == "uses_z_properly" for v in verdicts):
        lines.append("All audited Path A runs pass the z-use checks. Path A is mechanistically valid; the pre-committed area/Spearman monotonicity rule applies when the sweep completes.")
    elif all(v == "uses_filler_as_key" for v in verdicts):
        lines.append("All audited runs fail the z-use checks in the 'uses filler' direction. Path A is mechanistically **invalid**: the transport-distance manipulation does not isolate z-to-pred transport because the model is using filler as a lookup key. The experiment needs redesign.")
    else:
        lines.append("The audit is **mixed or inconclusive**. Individual per-run verdicts vary or fall into the 'mixed' bucket. Specific issues: " +
                      ", ".join(f"{a['experiment']}={a['verdict']}" for a in audits) +
                      ". Do not interpret Path A area/Spearman numbers until the audit is clean.")

    with open(OUT_DIR / "validity_audit.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n========== VALIDITY AUDIT SUMMARY ==========")
    for a in audits:
        print(f"  {a['experiment']}: verdict = {a['verdict']}")
        c1 = a['check1_z_shuffle']['fraction_changed_when_z_swapped']
        c2 = a['check2_filler_shuffle']['fraction_changed_when_filler_swapped']
        c3 = a['check3_controlled_filler_cand']['controlled_filler_candidate_accuracy']
        c4 = a['check4_filler_corruption']['ratio_corrupted_over_intact']
        print(f"    c1_zshuf={c1:.2f}  c2_fillshuf={c2:.2f}  "
              f"c3_ctrl_cand={c3:.2f}  c4_corr_ratio={c4:.2f}")
    final_verdict = verdict(audits[0]) if len(audits) == 1 else (
        verdicts[0] if all(v == verdicts[0] for v in verdicts) else "mixed")
    # Single printed line per spec
    if all(v == "uses_z_properly" for v in verdicts):
        print("Verdict: uses z. Path A manipulation is mechanistically valid.")
    elif all(v == "uses_filler_as_key" for v in verdicts):
        print("Verdict: uses filler. Path A manipulation is invalid; transport distance isn't isolated.")
    else:
        print("Verdict: mixed. Individual runs disagree or fall outside the clean buckets.")


if __name__ == "__main__":
    main()
