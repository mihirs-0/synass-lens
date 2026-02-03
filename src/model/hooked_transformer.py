"""
HookedTransformer configuration and factory for training from scratch.

TransformerLens is typically used for analyzing pretrained models,
but supports training from scratch with some setup.
"""

from typing import Optional
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from ..data.tokenizer import CharTokenizer


def create_hooked_transformer(
    tokenizer: CharTokenizer,
    n_layers: int = 4,
    n_heads: int = 4,
    d_model: int = 128,
    d_head: Optional[int] = None,
    d_mlp: Optional[int] = None,
    act_fn: str = "gelu",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> HookedTransformer:
    """
    Create a HookedTransformer configured for training from scratch.
    
    Args:
        tokenizer: Our character-level tokenizer
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        d_model: Model dimension (residual stream width)
        d_head: Head dimension (default: d_model // n_heads)
        d_mlp: MLP hidden dimension (default: 4 * d_model)
        act_fn: Activation function
        device: Device to place model on
        
    Returns:
        HookedTransformer ready for training
    """
    if d_head is None:
        d_head = d_model // n_heads
    if d_mlp is None:
        d_mlp = 4 * d_model
        
    # HookedTransformerConfig for training from scratch
    cfg = HookedTransformerConfig(
        # Architecture
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        d_mlp=d_mlp,
        d_vocab=tokenizer.vocab_size,
        
        # Context length - set based on our sequence format
        # <BOS> B(6) <SEP> z(2-3) <SEP> A(4) <EOS> = ~15-16 tokens max
        n_ctx=32,  # Some headroom
        
        # Activation function
        act_fn=act_fn,
        
        # Positional embeddings
        positional_embedding_type="standard",
        
        # No tied embeddings (simpler for training)
        # Actually, TransformerLens handles this internally
        
        # Normalization
        normalization_type="LN",  # LayerNorm
        
        # Attention
        attn_only=False,  # Include MLPs
        
        # Device
        device=device,
        
        # Random init
        seed=42,
        
        # Important: tell TransformerLens this is for training
        # (affects some internal behaviors)
        init_weights=True,
    )
    
    model = HookedTransformer(cfg)
    
    return model


def create_model_from_config(cfg, tokenizer: CharTokenizer) -> HookedTransformer:
    """Factory function to create model from Hydra config."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return create_hooked_transformer(
        tokenizer=tokenizer,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model,
        d_head=cfg.model.d_head,
        d_mlp=cfg.model.d_mlp,
        act_fn=cfg.model.act_fn,
        device=device,
    )
