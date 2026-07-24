#!/usr/bin/env python
"""
One-shot generalization test: can the converged model predict held-out B-groups?

Generates 1000 B-groups (K=10), trains on 500, evaluates on both halves.
If held-out accuracy > 80%: the circuit is a program (generalizes).
If held-out accuracy < 20%: the circuit is a lookup table (does not generalize).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, MappingData, DisambiguationDataset
from src.model import create_model_from_config
from src.training import train
from scripts.experiment_helpers import make_config


def evaluate(model, loader, device):
    """Compute mean loss and accuracy on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            mask = (shift_labels != -100) & (shift_labels != 2)  # exclude pad + EOS
            if mask.sum() == 0:
                continue

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='sum')
            # count only non-ignored, non-EOS
            loss_mask = flat_labels != -100
            n_loss_tokens = loss_mask.sum().item()

            preds = flat_logits.argmax(dim=-1)
            flat_mask = mask.view(-1)
            correct = ((preds == flat_labels) & flat_mask).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_tokens += flat_mask.sum().item()
            n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


def main():
    # --- Config ---
    K = 10
    N_B_TOTAL = 1000
    N_B_TRAIN = 500
    SEED = 42
    LR = 1e-3
    BS = 128
    MAX_STEPS = 30000
    EARLY_STOP_LOSS = 0.1

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Generate full mapping with 1000 B-groups ---
    cfg = make_config(
        experiment_name="generalization_test",
        k=K, seed=SEED, lr=LR, bs=BS,
        max_steps=MAX_STEPS, n_unique_b=N_B_TOTAL,
        eval_every=100, checkpoint_every=99999,
        early_stop_frac=None,
    )
    tokenizer = create_tokenizer_from_config(cfg)

    full_mapping = generate_mappings(
        n_unique_b=N_B_TOTAL, k=K,
        b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True,
        disambiguation_prefix_length=1,
    )

    # --- Split B-groups: first 500 train, last 500 held-out ---
    all_b_strings = list(full_mapping.mappings.keys())
    train_bs = set(all_b_strings[:N_B_TRAIN])
    heldout_bs = set(all_b_strings[N_B_TRAIN:])
    print(f"Total B-groups: {len(all_b_strings)}")
    print(f"Train B-groups: {len(train_bs)}, Held-out B-groups: {len(heldout_bs)}")

    # Build separate MappingData for train and held-out
    train_mappings = {b: v for b, v in full_mapping.mappings.items() if b in train_bs}
    train_examples = [ex for ex in full_mapping.examples if ex["b"] in train_bs]
    train_md = MappingData(
        mappings=train_mappings, examples=train_examples,
        n_unique_b=N_B_TRAIN, n_unique_a=len(set(ex["a"] for ex in train_examples)),
        k=K, task="bz_to_a",
    )

    heldout_mappings = {b: v for b, v in full_mapping.mappings.items() if b in heldout_bs}
    heldout_examples = [ex for ex in full_mapping.examples if ex["b"] in heldout_bs]
    heldout_md = MappingData(
        mappings=heldout_mappings, examples=heldout_examples,
        n_unique_b=N_B_TOTAL - N_B_TRAIN,
        n_unique_a=len(set(ex["a"] for ex in heldout_examples)),
        k=K, task="bz_to_a",
    )

    print(f"Train examples: {len(train_examples)} (D={len(train_examples)})")
    print(f"Held-out examples: {len(heldout_examples)}")

    # --- Build datasets and loaders ---
    train_ds = DisambiguationDataset(
        mapping_data=train_md, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    # For eval, wrap both sets as "train" split with probe_fraction=0 so all examples are included
    train_eval_ds = DisambiguationDataset(
        mapping_data=train_md, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    heldout_eval_ds = DisambiguationDataset(
        mapping_data=heldout_md, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate_fn, num_workers=0)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=BS, shuffle=False, collate_fn=collate_fn, num_workers=0)
    heldout_eval_loader = DataLoader(heldout_eval_ds, batch_size=BS, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # --- Create model ---
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(device)

    # --- Train with manual loop (early stop on candidate_loss < 0.1) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f"\nTraining on {len(train_examples)} examples (500 B-groups x K={K})...")
    print(f"Early stop when loss < {EARLY_STOP_LOSS}")
    print()

    step = 0
    epoch = 0
    converged = False

    while step < MAX_STEPS and not converged:
        epoch += 1
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 200 == 0:
                print(f"  Step {step:>6d}  loss={loss.item():.4f}")

            if loss.item() < EARLY_STOP_LOSS:
                # Verify on full train set
                train_loss, train_acc = evaluate(model, train_eval_loader, device)
                if train_loss < EARLY_STOP_LOSS:
                    print(f"  Step {step:>6d}  CONVERGED (train_loss={train_loss:.4f})")
                    converged = True
                    break

            if step >= MAX_STEPS:
                break

    if not converged:
        print(f"  Reached max_steps={MAX_STEPS} without convergence")

    # --- Final evaluation ---
    print("\nEvaluating...")
    train_loss, train_acc = evaluate(model, train_eval_loader, device)
    heldout_loss, heldout_acc = evaluate(model, heldout_eval_loader, device)

    log_k = math.log(K)

    print()
    print("=" * 50)
    print("GENERALIZATION TEST")
    print("=" * 50)
    print(f"Training groups (500):  loss = {train_loss:.4f},  accuracy = {train_acc*100:.1f}%")
    print(f"Held-out groups (500):  loss = {heldout_loss:.4f},  accuracy = {heldout_acc*100:.1f}%")
    print(f"(log K = {log_k:.3f} for reference)")
    print()

    if heldout_acc > 0.80:
        print("VERDICT: GENERALIZES (the circuit is a program)")
    elif heldout_acc < 0.20:
        print("VERDICT: DOES NOT GENERALIZE (the circuit is a lookup table)")
    else:
        print(f"VERDICT: AMBIGUOUS ({heldout_acc*100:.1f}% -- between 20% and 80%)")


if __name__ == "__main__":
    main()
