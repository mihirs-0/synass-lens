"""
Synthetic dataset for Late Disambiguation Lag experiments.

Generates disambiguation mappings where:
- base strings map to K different targets (bz_to_a / b_to_a)
- or K different bases map to the same target (az_to_b / a_to_b)
- task direction is configurable (Bz->A, Az->B, B->A, A->B)

Key insight: We control n_pairs_effective (number of unique B's) across
experiments, not total number of examples. This ensures fair comparison.
"""

import math
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass
class MappingData:
    """Container for generated mappings and examples."""
    # base_string -> [(z_string, target_string), ...]
    mappings: Dict[str, List[Tuple[str, str]]]
    # Flat list of all examples
    examples: List[Dict[str, str]]
    # Metadata
    n_unique_b: int
    n_unique_a: int
    k: int
    task: str
    
    
def generate_random_string(length: int, chars: str, rng: random.Random) -> str:
    """Generate a random string of given length from character set."""
    return "".join(rng.choices(chars, k=length))


def _generate_z_selectors(
    k: int,
    z_length: int,
    vocab_chars: str,
    rng: random.Random,
    n_sets: int = 1,
) -> List[List[str]]:
    """Generate ``n_sets`` distinct sets of K unique z-selector strings.

    Returns a list of length ``n_sets``, each element a list of K strings.
    All strings across all sets are globally unique.

    Auto-increases ``z_length`` if the requested capacity (n_sets * k)
    exceeds len(vocab_chars) ** z_length.
    """
    total_needed = n_sets * k
    capacity = len(vocab_chars) ** z_length
    effective_z_length = z_length
    while capacity < total_needed * 2:  # 2× headroom to avoid slow rejection sampling
        effective_z_length += 1
        capacity = len(vocab_chars) ** effective_z_length
    if effective_z_length != z_length:
        print(f"[INFO] z_length auto-increased {z_length} → {effective_z_length} "
              f"(need {total_needed} unique z-tokens, alphabet size {len(vocab_chars)})",
              flush=True)

    used: set = set()
    all_sets: List[List[str]] = []
    for _ in range(n_sets):
        zs: List[str] = []
        for _ in range(k):
            z = generate_random_string(effective_z_length, vocab_chars, rng)
            while z in used:
                z = generate_random_string(effective_z_length, vocab_chars, rng)
            used.add(z)
            zs.append(z)
        all_sets.append(zs)
    return all_sets


def generate_mappings(
    n_unique_b: int,
    k: int,
    b_length: int,
    a_length: int,
    z_length: int,
    vocab_chars: str,
    seed: int = 42,
    task: str = "bz_to_a",
    enforce_unique_a_first_char_per_b: bool = False,
    disambiguation_prefix_length: int = 1,
    z_sharing: str = "shared",
    n_supergroups: int = 1,
) -> MappingData:
    """
    Generate mappings for the selected task.

    Args:
        z_sharing: How z-selectors are shared across B-groups.
            * ``"shared"`` (default): one global set of K z-tokens.
            * ``"private"``: each B-group gets its own K unique z-tokens.
            * ``"supergroup"``: groups are partitioned into ``n_supergroups``
              super-groups that each share a set of K z-tokens.
        n_supergroups: Number of super-groups when ``z_sharing="supergroup"``.
            ``n_supergroups=1`` is equivalent to ``"shared"``;
            ``n_supergroups=n_unique_b`` is equivalent to ``"private"``.

    Tasks:
      - bz_to_a: base is B, target is A (B, z) -> A (K targets per base)
      - az_to_b: base is A, target is B (A, z) -> B (K bases per target; z redundant)
      - b_to_a: base is B, target is A (B) -> A (K targets per base; no z)
      - a_to_b: base is A, target is B (A) -> B (K bases per target; no z)
    """
    if task not in {"bz_to_a", "az_to_b", "b_to_a", "a_to_b"}:
        raise ValueError(f"Unknown task: {task}")
    if z_sharing not in {"shared", "private", "supergroup"}:
        raise ValueError(f"Unknown z_sharing mode: {z_sharing}")

    rng = random.Random(seed)

    # Track used strings to ensure uniqueness where intended
    used_b: set = set()
    used_a: set = set()

    # ---- Generate z-selector sets based on sharing mode ----
    if z_sharing == "shared":
        z_sets = _generate_z_selectors(k, z_length, vocab_chars, rng, n_sets=1)
    elif z_sharing == "private":
        z_sets = _generate_z_selectors(k, z_length, vocab_chars, rng, n_sets=n_unique_b)
    elif z_sharing == "supergroup":
        if n_supergroups < 1 or n_supergroups > n_unique_b:
            raise ValueError(f"n_supergroups must be in [1, {n_unique_b}], got {n_supergroups}")
        z_sets = _generate_z_selectors(k, z_length, vocab_chars, rng, n_sets=n_supergroups)

    def _z_selectors_for_group(group_idx: int) -> List[str]:
        if z_sharing == "shared":
            return z_sets[0]
        elif z_sharing == "private":
            return z_sets[group_idx]
        else:
            sg_idx = group_idx % n_supergroups
            return z_sets[sg_idx]
    
    mappings: Dict[str, List[Tuple[str, str]]] = {}
    examples: List[Dict[str, str]] = []
    
    if task in {"bz_to_a", "b_to_a"}:
        prefix_len = disambiguation_prefix_length if enforce_unique_a_first_char_per_b else 0

        if enforce_unique_a_first_char_per_b:
            if prefix_len == 1:
                max_unique = len(vocab_chars)
            elif prefix_len == 2:
                max_unique = len(vocab_chars) ** 2
            else:
                raise ValueError(f"disambiguation_prefix_length must be 1 or 2, got {prefix_len}")
            if k > max_unique:
                raise ValueError(
                    f"Cannot enforce unique A {prefix_len}-char prefixes per B: "
                    f"k={k} exceeds max unique prefixes ({max_unique})."
                )

        # Base strings are B, targets are A
        for group_idx in range(n_unique_b):
            b = generate_random_string(b_length, vocab_chars, rng)
            while b in used_b:
                b = generate_random_string(b_length, vocab_chars, rng)
            used_b.add(b)

            z_selectors = _z_selectors_for_group(group_idx)

            # Generate K unique A's for this B (global uniqueness)
            a_list = []
            if enforce_unique_a_first_char_per_b and prefix_len == 1:
                first_chars = rng.sample(list(vocab_chars), k)
                for first_char in first_chars:
                    if a_length == 1:
                        a = first_char
                    else:
                        suffix = generate_random_string(a_length - 1, vocab_chars, rng)
                        a = first_char + suffix
                    attempts = 0
                    while a in used_a:
                        attempts += 1
                        if attempts > 1000:
                            raise ValueError(
                                "Failed to generate unique A strings. "
                                "Try increasing a_length or vocab size."
                            )
                        if a_length == 1:
                            a = first_char
                        else:
                            suffix = generate_random_string(a_length - 1, vocab_chars, rng)
                            a = first_char + suffix
                    used_a.add(a)
                    a_list.append(a)
            elif enforce_unique_a_first_char_per_b and prefix_len == 2:
                all_prefixes = [c1 + c2 for c1 in vocab_chars for c2 in vocab_chars]
                selected_prefixes = rng.sample(all_prefixes, k)
                for prefix in selected_prefixes:
                    if a_length <= 2:
                        a = prefix[:a_length]
                    else:
                        suffix = generate_random_string(a_length - 2, vocab_chars, rng)
                        a = prefix + suffix
                    attempts = 0
                    while a in used_a:
                        attempts += 1
                        if attempts > 1000:
                            raise ValueError(
                                "Failed to generate unique A strings. "
                                "Try increasing a_length or vocab size."
                            )
                        suffix = generate_random_string(a_length - 2, vocab_chars, rng)
                        a = prefix + suffix
                    used_a.add(a)
                    a_list.append(a)
            else:
                for _ in range(k):
                    a = generate_random_string(a_length, vocab_chars, rng)
                    while a in used_a:
                        a = generate_random_string(a_length, vocab_chars, rng)
                    used_a.add(a)
                    a_list.append(a)

            mappings[b] = [(z_selectors[i], a_list[i]) for i in range(k)]
            for i in range(k):
                examples.append({"b": b, "z": z_selectors[i], "a": a_list[i]})
        
        n_unique_a = len(used_a)
    else:
        # Base strings are A, targets are B (z is redundant)
        for group_idx in range(n_unique_b):
            b = generate_random_string(b_length, vocab_chars, rng)
            while b in used_b:
                b = generate_random_string(b_length, vocab_chars, rng)
            used_b.add(b)

            z_selectors = _z_selectors_for_group(group_idx)

            a_list = []
            for _ in range(k):
                a = generate_random_string(a_length, vocab_chars, rng)
                while a in used_a:
                    a = generate_random_string(a_length, vocab_chars, rng)
                used_a.add(a)
                a_list.append(a)

            mappings[b] = [(z_selectors[i], a_list[i]) for i in range(k)]
            for i in range(k):
                examples.append({"b": b, "z": z_selectors[i], "a": a_list[i]})
        
        n_unique_a = len(used_a)
    
    return MappingData(
        mappings=mappings,
        examples=examples,
        n_unique_b=n_unique_b,
        n_unique_a=n_unique_a,
        k=k,
        task=task,
    )


class DisambiguationDataset(Dataset):
    """
    PyTorch Dataset for disambiguation tasks.
    
    Handles tokenization and returns ready-to-use tensors.
    """
    
    def __init__(
        self,
        mapping_data: MappingData,
        tokenizer: CharTokenizer,
        split: str = "train",
        probe_fraction: float = 0.1,
        seed: int = 42,
        task: str = "bz_to_a",
        split_by_base: bool = False,
        label_noise_prob: float = 0.0,
    ):
        self.mapping_data = mapping_data
        self.tokenizer = tokenizer
        self.split = split
        self.task = task
        self.label_noise_prob = label_noise_prob
        self._seed = seed
        
        # Split examples
        rng = random.Random(seed)
        examples = mapping_data.examples.copy()
        
        if split_by_base:
            base_key = "b" if self.task in {"bz_to_a", "b_to_a"} else "a"
            base_to_examples: Dict[str, List[Dict[str, str]]] = {}
            for ex in examples:
                base_to_examples.setdefault(ex[base_key], []).append(ex)
            
            bases = list(base_to_examples.keys())
            rng.shuffle(bases)
            n_probe_bases = int(len(bases) * probe_fraction)
            probe_bases = set(bases[:n_probe_bases])
            
            if split == "train":
                self.examples = [
                    ex for b in bases if b not in probe_bases for ex in base_to_examples[b]
                ]
            else:
                self.examples = [
                    ex for b in bases if b in probe_bases for ex in base_to_examples[b]
                ]
        else:
            rng.shuffle(examples)
            n_probe = int(len(examples) * probe_fraction)
            if split == "train":
                self.examples = examples[n_probe:]
            else:
                self.examples = examples[:n_probe]
            
        # Pre-tokenize all examples
        self._precompute_tokens()
        
    def _precompute_tokens(self):
        """Pre-tokenize all examples for faster training.
        
        If label_noise_prob > 0 and this is the training split, randomly
        replaces the target A string with a wrong candidate (from the same
        B group) for a fraction of examples. This implements Ziyin's label
        noise experiment. The stored raw strings (tok["a"]) always reflect
        the CLEAN ground truth for probing/analysis purposes.
        """
        self.tokenized = []
        noise_rng = random.Random(self._seed + 99999)
        noise_count = 0
        
        for ex in self.examples:
            b_str, z_str, a_str = ex["b"], ex["z"], ex["a"]
            
            # Label noise: replace target A with wrong candidate (training only)
            if (self.label_noise_prob > 0
                    and self.split == "train"
                    and self.task in ("bz_to_a", "b_to_a")
                    and noise_rng.random() < self.label_noise_prob):
                candidates = self.mapping_data.mappings[b_str]
                wrong_a_list = [a for (_, a) in candidates if a != a_str]
                if wrong_a_list:
                    a_str = noise_rng.choice(wrong_a_list)
                    noise_count += 1
            
            tok = self.tokenizer.encode_sequence(b_str, z_str, a_str, task=self.task)
            # Store raw (CLEAN) example for probing — always ground truth
            tok["b"] = ex["b"]
            tok["z"] = ex["z"]
            tok["a"] = ex["a"]
            tok["base_string"] = ex["b"] if self.task in {"bz_to_a", "b_to_a"} else ex["a"]
            self.tokenized.append(tok)
        
        if noise_count > 0:
            total = len(self.examples)
            print(f"  [Label noise] {noise_count}/{total} training examples "
                  f"({100 * noise_count / total:.1f}%) have corrupted targets")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.tokenized[idx]
    

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Handles variable-length sequences with padding.
    """
    # Find max length in batch
    max_len = max(len(item["input_ids"]) for item in batch)
    
    # Get pad token from first item (hacky but works)
    # Assume pad_token_id = 0
    pad_id = 0
    
    batch_input_ids = []
    batch_labels = []
    batch_z_positions = []
    batch_z_end_positions = []
    batch_target_starts = []
    batch_target_ends = []
    batch_base_strings = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        
        # Pad input_ids
        input_ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_id, dtype=torch.long)
        ])
        batch_input_ids.append(input_ids)
        
        # Pad labels with -100
        labels = torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        batch_labels.append(labels)
        
        # Positions (no padding needed, these are scalars)
        batch_z_positions.append(item["z_position"])
        batch_z_end_positions.append(item["z_end_position"])
        batch_target_starts.append(item["target_start_position"])
        batch_target_ends.append(item["target_end_position"])
        batch_base_strings.append(item["base_string"])
    
    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "z_positions": torch.tensor(batch_z_positions, dtype=torch.long),
        "z_end_positions": torch.tensor(batch_z_end_positions, dtype=torch.long),
        "target_start_positions": torch.tensor(batch_target_starts, dtype=torch.long),
        "target_end_positions": torch.tensor(batch_target_ends, dtype=torch.long),
        "base_strings": batch_base_strings,
    }


def create_datasets_from_config(cfg, tokenizer: CharTokenizer) -> Tuple[DisambiguationDataset, DisambiguationDataset, MappingData]:
    """
    Factory function to create train/probe datasets from Hydra config.
    
    Returns:
        (train_dataset, probe_dataset, mapping_data)
    """
    # Generate mappings
    mapping_data = generate_mappings(
        n_unique_b=cfg.data.n_unique_b,
        k=cfg.data.k,
        b_length=cfg.data.b_length,
        a_length=cfg.data.a_length,
        z_length=cfg.data.z_length,
        vocab_chars=cfg.data.vocab_chars,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
        enforce_unique_a_first_char_per_b=getattr(cfg.data, "enforce_unique_a_first_char_per_b", False),
        disambiguation_prefix_length=int(getattr(cfg.data, "disambiguation_prefix_length", 1)),
        z_sharing=getattr(cfg.data, "z_sharing", "shared"),
        n_supergroups=int(getattr(cfg.data, "n_supergroups", 1)),
    )
    
    # Label noise: only applied to training data, never to probe/eval
    label_noise_prob = float(getattr(cfg.data, "label_noise_prob", 0.0))
    
    train_dataset = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split="train",
        probe_fraction=cfg.data.probe_fraction,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
        split_by_base=getattr(cfg.data, "split_by_base", False),
        label_noise_prob=label_noise_prob,
    )
    
    probe_dataset = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split="probe",
        probe_fraction=cfg.data.probe_fraction,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
        split_by_base=getattr(cfg.data, "split_by_base", False),
        label_noise_prob=0.0,  # eval is ALWAYS clean
    )
    
    return train_dataset, probe_dataset, mapping_data
