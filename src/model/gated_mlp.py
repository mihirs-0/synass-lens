"""
Gated MLP for disambiguation: B and z pathways interact multiplicatively.

Architecture:
    h_b = MLP_b(mean_pool(embed(B tokens)))
    h_z = MLP_z(mean_pool(embed(z tokens)))
    h_gate = LayerNorm(h_b ⊙ h_z)       — multiplicative gating
    logits[pos] = MLP_out([h_gate; embed(input[pos])])

This architecture can represent the (B × z) interaction needed for
disambiguation but has NO attention, NO recurrence, NO sequence mixing
beyond the explicit multiplicative gate.

B and z regions are detected dynamically from SEP token positions in
the input, so no hard-coded offsets are needed.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


class GatedMLP(nn.Module):
    """
    Feedforward network with multiplicative gating between B and z pathways.

    Parameters
    ----------
    vocab_size : int
        Number of tokens (including special tokens).
    d_model : int
        Embedding / pathway dimension.
    d_hidden : int
        Hidden dimension inside each MLP sub-network.
    device : str
        Target device string (set on ``self.cfg.device`` for checkpoint compat).
    """

    def __init__(self, vocab_size: int, d_model: int = 128, d_hidden: int = 512,
                 device: str = "cpu"):
        super().__init__()
        self.cfg = SimpleNamespace(
            d_vocab=vocab_size,
            d_model=d_model,
            d_hidden=d_hidden,
            device=device,
        )

        # Shared token embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # B pathway
        self.mlp_b = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

        # z pathway
        self.mlp_z = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

        # Layer norms for stability
        self.ln_b = nn.LayerNorm(d_model)
        self.ln_z = nn.LayerNorm(d_model)
        self.ln_gate = nn.LayerNorm(d_model)

        # Output pathway: gated representation + current token → next-token logits
        self.mlp_out = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, vocab_size),
        )

    # ------------------------------------------------------------------
    # Region detection
    # ------------------------------------------------------------------
    def _detect_regions(self, input_ids: torch.Tensor):
        """Return boolean masks for B and z token positions.

        Assumes sequence format: <BOS> B ... <SEP> z ... <SEP> A ... <EOS>
        """
        SEP_ID = 3
        BOS_ID = 1

        sep_mask = (input_ids == SEP_ID)
        sep_cumsum = sep_mask.cumsum(dim=1)

        # B region: between BOS and first SEP (cumsum == 0, excluding BOS)
        b_mask = (sep_cumsum == 0) & (input_ids != BOS_ID)
        # z region: between first SEP and second SEP (cumsum == 1, excluding SEP)
        z_mask = (sep_cumsum == 1) & (~sep_mask)

        return b_mask, z_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len) LongTensor

        Returns
        -------
        logits : (batch, seq_len, vocab_size) FloatTensor
        """
        batch_size, seq_len = input_ids.shape

        emb = self.embed(input_ids)  # (batch, seq, d_model)

        b_mask, z_mask = self._detect_regions(input_ids)

        # Mean-pool B embeddings
        b_mask_f = b_mask.float().unsqueeze(-1)  # (batch, seq, 1)
        h_b = (emb * b_mask_f).sum(dim=1) / b_mask_f.sum(dim=1).clamp(min=1)
        h_b = self.ln_b(self.mlp_b(h_b))  # (batch, d_model)

        # Mean-pool z embeddings
        z_mask_f = z_mask.float().unsqueeze(-1)
        h_z = (emb * z_mask_f).sum(dim=1) / z_mask_f.sum(dim=1).clamp(min=1)
        h_z = self.ln_z(self.mlp_z(h_z))  # (batch, d_model)

        # Multiplicative gating
        h_gate = self.ln_gate(h_b * h_z)  # (batch, d_model)

        # Broadcast gated representation to every sequence position and
        # concatenate with the per-position token embedding.
        h_gate_exp = h_gate.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([h_gate_exp, emb], dim=-1)  # (batch, seq, 2·d_model)

        logits = self.mlp_out(combined)  # (batch, seq, vocab_size)
        return logits


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def create_gated_mlp_from_config(cfg, tokenizer) -> GatedMLP:
    """Create a GatedMLP from a Hydra-style config namespace."""
    device = _select_device()
    d_model = getattr(cfg.model, "d_model", 128)
    d_hidden = getattr(cfg.model, "d_hidden", 512)

    model = GatedMLP(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        d_hidden=d_hidden,
        device=device,
    )
    model = model.to(device)
    model.cfg.device = device
    return model
