"""
Synthetic dataset for Late Disambiguation Lag experiments.

Generates disambiguation mappings where:
- base strings map to K different targets (bz_to_a / b_to_a)
- or K different bases map to the same target (az_to_b / a_to_b)
- task direction is configurable (Bz->A, Az->B, B->A, A->B)

Key insight: We control n_pairs_effective (number of unique B's) across
experiments, not total number of examples. This ensures fair comparison.
"""

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


def generate_mappings(
    n_unique_b: int,
    k: int,
    b_length: int,
    a_length: int,
    z_length: int,
    vocab_chars: str,
    seed: int = 42,
    task: str = "bz_to_a",
) -> MappingData:
    """
    Generate mappings for the selected task.
    
    Tasks:
      - bz_to_a: base is B, target is A (B, z) -> A (K targets per base)
      - az_to_b: base is A, target is B (A, z) -> B (K bases per target; z redundant)
      - b_to_a: base is B, target is A (B) -> A (K targets per base; no z)
      - a_to_b: base is A, target is B (A) -> B (K bases per target; no z)
    """
    if task not in {"bz_to_a", "az_to_b", "b_to_a", "a_to_b"}:
        raise ValueError(f"Unknown task: {task}")
    
    rng = random.Random(seed)
    
    # Track used strings to ensure uniqueness where intended
    used_b: set = set()
    used_a: set = set()
    
    # Generate z selectors (reused across all base strings)
    z_selectors = []
    for _ in range(k):
        z = generate_random_string(z_length, vocab_chars, rng)
        while z in z_selectors:
            z = generate_random_string(z_length, vocab_chars, rng)
        z_selectors.append(z)
    
    mappings: Dict[str, List[Tuple[str, str]]] = {}
    examples: List[Dict[str, str]] = []
    
    if task in {"bz_to_a", "b_to_a"}:
        # Base strings are B, targets are A
        for _ in range(n_unique_b):
            b = generate_random_string(b_length, vocab_chars, rng)
            while b in used_b:
                b = generate_random_string(b_length, vocab_chars, rng)
            used_b.add(b)
            
            # Generate K unique A's for this B (global uniqueness)
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
    else:
        # Base strings are A, targets are B (z is redundant)
        for _ in range(n_unique_b):
            # Generate unique B
            b = generate_random_string(b_length, vocab_chars, rng)
            while b in used_b:
                b = generate_random_string(b_length, vocab_chars, rng)
            used_b.add(b)
            
            # Generate K unique A's for this B (global uniqueness)
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
    ):
        self.mapping_data = mapping_data
        self.tokenizer = tokenizer
        self.split = split
        self.task = task
        
        # Split examples
        rng = random.Random(seed)
        examples = mapping_data.examples.copy()
        rng.shuffle(examples)
        
        n_probe = int(len(examples) * probe_fraction)
        if split == "train":
            self.examples = examples[n_probe:]
        else:
            self.examples = examples[:n_probe]
            
        # Pre-tokenize all examples
        self._precompute_tokens()
        
    def _precompute_tokens(self):
        """Pre-tokenize all examples for faster training."""
        self.tokenized = []
        for ex in self.examples:
            tok = self.tokenizer.encode_sequence(ex["b"], ex["z"], ex["a"], task=self.task)
            # Store raw example too for probing
            tok["b"] = ex["b"]
            tok["z"] = ex["z"]
            tok["a"] = ex["a"]
            tok["base_string"] = ex["b"] if self.task in {"bz_to_a", "b_to_a"} else ex["a"]
            self.tokenized.append(tok)
    
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
        (train_dataset, probe_dataset)
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
    )
    
    train_dataset = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split="train",
        probe_fraction=cfg.data.probe_fraction,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
    )
    
    probe_dataset = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split="probe",
        probe_fraction=cfg.data.probe_fraction,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
    )
    
    return train_dataset, probe_dataset, mapping_data
