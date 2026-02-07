#!/usr/bin/env python
"""
Unit tests for candidate-set normalized evaluation.
"""

import sys
from pathlib import Path
import argparse
import math
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, generate_mappings
from src.model import create_hooked_transformer
from src.analysis.candidate_eval import score_candidate_sequences, run_candidate_eval


def _build_test_config(k: int):
    return SimpleNamespace(
        experiment=SimpleNamespace(seed=42),
        data=SimpleNamespace(
            n_unique_b=50,
            k=k,
            b_length=6,
            a_length=4,
            z_length=2,
            vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
            task="bz_to_a",
        ),
        tokenizer=SimpleNamespace(
            pad_token="<PAD>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            sep_token="<SEP>",
        ),
    )


def test_logprob_consistency(tokenizer, mapping_data, model, seed: int = 123) -> bool:
    rng = random.Random(seed)
    examples = mapping_data.examples
    sample = rng.sample(examples, min(10, len(examples)))

    for ex in sample:
        b = ex["b"]
        z = ex["z"]
        a = ex["a"]

        encoded = tokenizer.encode_sequence(b, z, a, task="bz_to_a")
        input_ids = encoded["input_ids"].unsqueeze(0)
        labels = encoded["labels"]
        target_start = encoded["target_start_position"]
        target_end = encoded["target_end_position"]

        with torch.no_grad():
            logits = model(input_ids)

        direct_log_prob = torch.zeros((), dtype=torch.float64)
        for pos in range(target_start, target_end + 1):
            if labels[pos].item() == -100:
                continue
            log_probs = F.log_softmax(logits[0, pos - 1], dim=-1)
            direct_log_prob += log_probs[labels[pos].item()].double()

        mappings = mapping_data.mappings[b]
        correct_index = None
        for idx, (z_i, a_i) in enumerate(mappings):
            if z_i == z and a_i == a:
                correct_index = idx
                break
        if correct_index is None:
            print("FAILED: Could not find correct index for example.")
            return False

        candidate_a_strings = [a_i for _, a_i in mappings]
        scored = score_candidate_sequences(
            model=model,
            tokenizer=tokenizer,
            base_string=b,
            z_string=z,
            candidate_a_strings=candidate_a_strings,
            correct_index=correct_index,
            task="bz_to_a",
            device="cpu",
        )
        candidate_log_prob = scored["correct_log_prob"]

        if abs(direct_log_prob.item() - candidate_log_prob) >= 1e-4:
            print("FAILED: Logprob consistency mismatch.")
            return False

    return True


def test_candidate_loss_floor(tokenizer, mapping_data, model, k: int) -> bool:
    results = run_candidate_eval(
        model=model,
        tokenizer=tokenizer,
        mapping_data=mapping_data,
        n_examples=50,
        task="bz_to_a",
        device="cpu",
        seed=123,
    )
    mean_loss = results["candidate_loss"]
    mean_acc = results["candidate_accuracy"]
    log_k = math.log(k)

    loss_ok = (0.5 * log_k) <= mean_loss <= (1.5 * log_k)
    acc_ok = abs(mean_acc - (1.0 / k)) < 0.2

    if not loss_ok:
        print(
            f"FAILED: candidate_loss {mean_loss:.4f} outside "
            f"[0.5*log(K)={0.5*log_k:.4f}, 1.5*log(K)={1.5*log_k:.4f}]"
        )
    if not acc_ok:
        print(f"FAILED: candidate_accuracy {mean_acc:.4f} not near 1/K {1.0/k:.4f}")

    return loss_ok and acc_ok


def test_candidate_set_integrity(mapping_data, k: int, seed: int = 123) -> bool:
    rng = random.Random(seed)
    base_strings = list(mapping_data.mappings.keys())
    sample = rng.sample(base_strings, min(20, len(base_strings)))

    if not sample:
        print("FAILED: No base strings found.")
        return False

    global_z = [z for z, _ in mapping_data.mappings[sample[0]]]

    for b in sample:
        candidates = mapping_data.mappings[b]
        if len(candidates) != k:
            print("FAILED: Candidate set size mismatch.")
            return False
        a_strings = [a for _, a in candidates]
        if len(set(a_strings)) != len(a_strings):
            print("FAILED: Duplicate A strings in candidate set.")
            return False
        z_selectors = [z for z, _ in candidates]
        if z_selectors != global_z:
            print("FAILED: z selector list mismatch for base string.")
            return False

    return True


def test_sequence_length_uniformity(tokenizer, mapping_data, k: int, seed: int = 123) -> bool:
    rng = random.Random(seed)
    base_strings = list(mapping_data.mappings.keys())
    sample = rng.sample(base_strings, min(20, len(base_strings)))

    expected_len = 1 + 6 + 1 + 2 + 1 + 4 + 1
    for b in sample:
        lengths = []
        for z, a in mapping_data.mappings[b]:
            encoded = tokenizer.encode_sequence(b, z, a, task="bz_to_a")
            lengths.append(len(encoded["input_ids"]))
        if len(set(lengths)) != 1:
            print("FAILED: Candidate sequences have differing lengths.")
            return False
        if lengths[0] != expected_len:
            print(f"FAILED: Sequence length {lengths[0]} != expected {expected_len}")
            return False

    return True


def test_eos_token_handling(tokenizer, mapping_data) -> bool:
    example = mapping_data.examples[0]
    encoded = tokenizer.encode_sequence(example["b"], example["z"], example["a"], task="bz_to_a")
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    target_start = encoded["target_start_position"]
    target_end = encoded["target_end_position"]

    print("input_ids:", input_ids)
    print("labels:", labels)

    target_slice = labels[target_start:target_end]
    all_targets_real = torch.all(target_slice != -100).item()
    eos_ok = labels[target_end].item() == tokenizer.eos_token_id
    prefix_ok = torch.all(labels[:target_start] == -100).item()

    if not all_targets_real:
        print("FAILED: Target tokens contain -100 values.")
    if not eos_ok:
        print("FAILED: EOS token mismatch.")
    if not prefix_ok:
        print("FAILED: Prefix labels are not all -100.")

    return all_targets_real and eos_ok and prefix_ok


def main():
    parser = argparse.ArgumentParser(description="Test candidate evaluation implementation")
    parser.add_argument("--k", type=int, default=10, help="K value for test mappings")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cpu"

    cfg = _build_test_config(args.k)
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=cfg.data.n_unique_b,
        k=cfg.data.k,
        b_length=cfg.data.b_length,
        a_length=cfg.data.a_length,
        z_length=cfg.data.z_length,
        vocab_chars=cfg.data.vocab_chars,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
    )

    model = create_hooked_transformer(
        tokenizer=tokenizer,
        n_layers=2,
        n_heads=2,
        d_model=64,
        device=device,
    )
    model.eval()

    results = {
        "Test A: Logprob Consistency": test_logprob_consistency(tokenizer, mapping_data, model),
        "Test B: Candidate Loss Floor": test_candidate_loss_floor(tokenizer, mapping_data, model, args.k),
        "Test C: Candidate Set Integrity": test_candidate_set_integrity(mapping_data, args.k),
        "Test D: Sequence Length Uniformity": test_sequence_length_uniformity(tokenizer, mapping_data, args.k),
        "Test E: EOS Token Handling": test_eos_token_handling(tokenizer, mapping_data),
    }

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("FAILURES DETECTED - DO NOT TRUST PLOTS")
        sys.exit(1)


if __name__ == "__main__":
    main()
