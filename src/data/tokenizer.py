"""
Character-level tokenizer for Late Disambiguation Lag experiments.

Handles the mapping between character sequences and token IDs.
Designed to work seamlessly with HookedTransformer.
"""

from typing import List, Dict, Optional
import torch


class CharTokenizer:
    """
    Simple character-level tokenizer with special tokens.
    
    Token layout:
        0: <PAD>
        1: <BOS>
        2: <EOS>
        3: <SEP>
        4+: vocabulary characters
    """
    
    def __init__(
        self,
        vocab_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789",
        pad_token: str = "<PAD>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        sep_token: str = "<SEP>",
    ):
        self.vocab_chars = vocab_chars
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        
        # Build token -> id mapping
        self.special_tokens = [pad_token, bos_token, eos_token, sep_token]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        # Add vocab characters
        for i, char in enumerate(vocab_chars):
            idx = len(self.special_tokens) + i
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            
        # Convenience attributes
        self.pad_token_id = self.token_to_id[pad_token]
        self.bos_token_id = self.token_to_id[bos_token]
        self.eos_token_id = self.token_to_id[eos_token]
        self.sep_token_id = self.token_to_id[sep_token]
        self.vocab_size = len(self.token_to_id)
        
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode a string to token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                raise ValueError(f"Unknown character: {char}")
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to string."""
        chars = []
        for id in ids:
            token = self.id_to_token[id]
            if skip_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)
    
    def encode_sequence(
        self,
        b: str,
        z: str,
        a: str,
        task: str = "bz_to_a",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a (B, z, A) example for training.
        
        Tasks:
            - "bz_to_a": <BOS> B <SEP> z <SEP> A <EOS>
            - "az_to_b": <BOS> A <SEP> z <SEP> B <EOS>
        
        Returns dict with:
            - input_ids: full sequence
            - labels: -100 for input positions, token IDs for target positions
            - z_position: index of first z token (for probing)
            - target_start_position: index of first target token
        """
        if task not in {"bz_to_a", "az_to_b"}:
            raise ValueError(f"Unknown task: {task}")
        
        if task == "bz_to_a":
            left, right = b, a
        else:
            left, right = a, b
        
        # Build sequence
        tokens = [self.bos_token_id]
        tokens.extend(self.encode(left))
        tokens.append(self.sep_token_id)
        z_position = len(tokens)  # First z token position
        tokens.extend(self.encode(z))
        tokens.append(self.sep_token_id)
        target_start = len(tokens)  # First target token position
        target_tokens = self.encode(right)
        tokens.extend(target_tokens)
        tokens.append(self.eos_token_id)
        
        # Labels: -100 for all positions except target (and EOS)
        labels = [-100] * target_start + target_tokens + [self.eos_token_id]
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "z_position": z_position,
            "z_end_position": z_position + len(z),
            "target_start_position": target_start,
            "target_end_position": target_start + len(right),
        }
    
    def batch_encode(
        self,
        examples: List[Dict[str, str]],  # List of {"b": ..., "z": ..., "a": ...}
        task: str = "bz_to_a",
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple examples with padding.
        
        Returns batched tensors ready for model input.
        """
        encoded = [self.encode_sequence(ex["b"], ex["z"], ex["a"], task=task) for ex in examples]
        
        # Find max length
        max_len = max(len(e["input_ids"]) for e in encoded)
        
        # Pad everything
        batch_input_ids = []
        batch_labels = []
        batch_z_positions = []
        batch_target_starts = []
        
        for e in encoded:
            seq_len = len(e["input_ids"])
            pad_len = max_len - seq_len
            
            batch_input_ids.append(
                torch.cat([e["input_ids"], torch.full((pad_len,), self.pad_token_id)])
            )
            batch_labels.append(
                torch.cat([e["labels"], torch.full((pad_len,), -100)])
            )
            batch_z_positions.append(e["z_position"])
            batch_target_starts.append(e["target_start_position"])
            
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "z_positions": torch.tensor(batch_z_positions),
            "target_start_positions": torch.tensor(batch_target_starts),
        }


def create_tokenizer_from_config(cfg) -> CharTokenizer:
    """Factory function to create tokenizer from Hydra config."""
    return CharTokenizer(
        vocab_chars=cfg.data.vocab_chars,
        pad_token=cfg.tokenizer.pad_token,
        bos_token=cfg.tokenizer.bos_token,
        eos_token=cfg.tokenizer.eos_token,
        sep_token=cfg.tokenizer.sep_token,
    )
