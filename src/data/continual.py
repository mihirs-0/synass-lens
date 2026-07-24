"""
Data modification functions for continual learning experiments.

Provides tools to systematically modify existing B→A mappings:
- reassign_mappings: shuffle z→A assignments for a fraction of B groups
- expand_k: add new candidates to each B group
- contract_k: remove candidates from each B group
- compute_mapping_divergence: theoretical prediction for minimum dissipation
"""

import math
import random
from typing import Dict, List, Set, Tuple

from .dataset import MappingData, generate_random_string


def reassign_mappings(
    mappings: Dict[str, List[Tuple[str, str]]],
    fraction: float,
    seed: int = 137,
) -> Tuple[Dict[str, List[Tuple[str, str]]], Set[str]]:
    """
    Reassign z→A mappings for a fraction of B groups.

    For each selected B group, shuffle which A target goes with which z value.
    The candidate set {A_1, ..., A_K} stays the same — only the z→A pairing
    changes.

    Args:
        mappings: dict {B: [(z_1, A_1), (z_2, A_2), ...]}
        fraction: float in [0, 1], fraction of B groups to reassign
        seed: random seed for reproducibility

    Returns:
        new_mappings: dict with same structure, reassigned groups shuffled
        reassigned_bs: set of B strings that were reassigned
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    rng = random.Random(seed)
    all_bs = sorted(mappings.keys())
    shuffled_bs = all_bs[:]
    rng.shuffle(shuffled_bs)
    n_reassign = int(len(all_bs) * fraction)
    reassigned_bs = set(shuffled_bs[:n_reassign])

    new_mappings: Dict[str, List[Tuple[str, str]]] = {}
    for b, pairs in mappings.items():
        if b in reassigned_bs:
            zs = [z for z, _ in pairs]
            a_targets = [a for _, a in pairs]
            rng.shuffle(a_targets)
            new_mappings[b] = list(zip(zs, a_targets))
        else:
            new_mappings[b] = list(pairs)

    return new_mappings, reassigned_bs


def expand_k(
    mappings: Dict[str, List[Tuple[str, str]]],
    new_k: int,
    z_length: int,
    a_length: int,
    vocab_chars: str,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Add new candidates to each B group, expanding from current K to new_k.

    For each B group, generates (new_k - current_k) additional (z, A) pairs.
    New z tokens and A strings are guaranteed unique across the entire mapping.

    Args:
        mappings: existing mappings {B: [(z, A), ...]}
        new_k: target number of candidates per B group
        z_length: length of z selector strings
        a_length: length of A target strings
        vocab_chars: character vocabulary
        seed: random seed

    Returns:
        expanded mappings with new_k candidates per B group
    """
    rng = random.Random(seed)

    current_k = len(next(iter(mappings.values())))
    if new_k <= current_k:
        raise ValueError(f"new_k ({new_k}) must be > current K ({current_k})")

    used_z: Set[str] = set()
    used_a: Set[str] = set()
    for pairs in mappings.values():
        for z, a in pairs:
            used_z.add(z)
            used_a.add(a)

    new_mappings: Dict[str, List[Tuple[str, str]]] = {}
    n_extra = new_k - current_k

    for b, pairs in mappings.items():
        new_pairs = list(pairs)
        for _ in range(n_extra):
            z = generate_random_string(z_length, vocab_chars, rng)
            while z in used_z:
                z = generate_random_string(z_length, vocab_chars, rng)
            used_z.add(z)

            a = generate_random_string(a_length, vocab_chars, rng)
            while a in used_a:
                a = generate_random_string(a_length, vocab_chars, rng)
            used_a.add(a)

            new_pairs.append((z, a))
        new_mappings[b] = new_pairs

    return new_mappings


def contract_k(
    mappings: Dict[str, List[Tuple[str, str]]],
    new_k: int,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Remove candidates from each B group, contracting to new_k.

    Keeps the first new_k candidates (deterministic ordering from original
    generation), shuffled by seed for which ones to keep.

    Args:
        mappings: existing mappings
        new_k: target number of candidates (must be < current K)
        seed: random seed

    Returns:
        contracted mappings with new_k candidates per B group
    """
    rng = random.Random(seed)

    current_k = len(next(iter(mappings.values())))
    if new_k >= current_k:
        raise ValueError(f"new_k ({new_k}) must be < current K ({current_k})")

    new_mappings: Dict[str, List[Tuple[str, str]]] = {}
    for b, pairs in mappings.items():
        indices = list(range(current_k))
        rng_local = random.Random(rng.randint(0, 2**31))
        rng_local.shuffle(indices)
        kept = sorted(indices[:new_k])
        new_mappings[b] = [pairs[i] for i in kept]

    return new_mappings


def compute_mapping_divergence(
    old_mappings: Dict[str, List[Tuple[str, str]]],
    new_mappings: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, object]:
    """
    Compute divergence statistics between old and new mappings.

    For each B group, counts how many (z → A) assignments changed.
    Provides both raw counts and theoretical KL predictions.

    Returns:
        dict with:
            total_pairs: total number of (B, z) pairs
            changed_pairs: number of pairs where A target changed
            fraction_changed: changed_pairs / total_pairs
            n_groups_changed: number of B groups with any change
            per_group_changes: dict {B: n_changed_pairs}
            effective_kl: estimated KL = n_groups_changed * log(K)
            per_pair_kl: estimated KL = changed_pairs * log(K) / K
    """
    total_pairs = 0
    changed_pairs = 0
    n_groups_changed = 0
    per_group_changes: Dict[str, int] = {}

    k = len(next(iter(old_mappings.values())))

    for b in old_mappings:
        old_pairs = old_mappings[b]
        new_pairs = new_mappings[b]
        n_changed = sum(
            1 for (_, a_old), (_, a_new) in zip(old_pairs, new_pairs)
            if a_old != a_new
        )
        total_pairs += len(old_pairs)
        changed_pairs += n_changed
        if n_changed > 0:
            n_groups_changed += 1
            per_group_changes[b] = n_changed

    log_k = math.log(k) if k > 1 else 0.0

    return {
        "total_pairs": total_pairs,
        "changed_pairs": changed_pairs,
        "fraction_changed": changed_pairs / total_pairs if total_pairs > 0 else 0.0,
        "n_groups_changed": n_groups_changed,
        "n_groups_total": len(old_mappings),
        "K": k,
        "log_K": log_k,
        "per_group_changes": per_group_changes,
        "effective_kl_per_group": n_groups_changed * log_k,
        "effective_kl_per_pair": changed_pairs * log_k / k if k > 0 else 0.0,
    }


def mappings_to_examples(
    mappings: Dict[str, List[Tuple[str, str]]],
) -> List[Dict[str, str]]:
    """Convert mappings dict to flat list of example dicts."""
    examples = []
    for b, pairs in mappings.items():
        for z, a in pairs:
            examples.append({"b": b, "z": z, "a": a})
    return examples


def verify_reassignment(
    old_mappings: Dict[str, List[Tuple[str, str]]],
    new_mappings: Dict[str, List[Tuple[str, str]]],
    reassigned_bs: Set[str],
    fraction: float,
):
    """
    Verify correctness of reassignment.

    Checks:
    1. Unchanged B groups have identical mappings
    2. Reassigned B groups have same candidate sets but (mostly) different pairings
    3. Actual change fraction is close to requested fraction

    Raises AssertionError on failure. Prints summary statistics.
    """
    for b in old_mappings:
        if b not in reassigned_bs:
            assert old_mappings[b] == new_mappings[b], (
                f"Unchanged B group {b} was modified!"
            )

    for b in reassigned_bs:
        old_as = set(a for _, a in old_mappings[b])
        new_as = set(a for _, a in new_mappings[b])
        assert old_as == new_as, (
            f"Candidate set changed for B={b}! "
            f"Old: {old_as}, New: {new_as}"
        )

    divergence = compute_mapping_divergence(old_mappings, new_mappings)
    total = divergence["total_pairs"]
    changed = divergence["changed_pairs"]

    print(f"Reassignment verification:")
    print(f"  Requested fraction: {fraction:.3f}")
    print(f"  B groups reassigned: {len(reassigned_bs)}/{len(old_mappings)}")
    print(f"  Actual changed pairs: {changed}/{total} = {changed / total:.3f}")
    print(f"  Groups with changes: {divergence['n_groups_changed']}")
