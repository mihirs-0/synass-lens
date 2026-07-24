"""
LSTM-based sequence model for disambiguation.

Processes <BOS> B <SEP> z <SEP> A <EOS> left-to-right. The hidden state
after processing B and z encodes the multiplicative interaction needed
for target prediction through the nonlinear recurrent dynamics.

Unlike the Gated MLP, the LSTM never explicitly gates the two pathways —
the interaction emerges from the nonlinear hidden-state dynamics as B
tokens are processed first and z tokens modulate the already-formed
representation.
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


class SequenceRNN(nn.Module):
    """
    Left-to-right LSTM language model.

    Parameters
    ----------
    vocab_size : int
        Number of tokens (including special tokens).
    d_model : int
        Token embedding dimension.
    d_hidden : int
        LSTM hidden state dimension.
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (only when n_layers > 1).
    device : str
        Target device string.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 d_hidden: int = 256, n_layers: int = 2,
                 dropout: float = 0.0, device: str = "cpu"):
        super().__init__()
        self.cfg = SimpleNamespace(
            d_vocab=vocab_size,
            d_model=d_model,
            d_hidden=d_hidden,
            n_layers=n_layers,
            device=device,
        )

        self.embed = nn.Embedding(vocab_size, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.ln = nn.LayerNorm(d_hidden)

        self.output_proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, vocab_size),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len) LongTensor

        Returns
        -------
        logits : (batch, seq_len, vocab_size) FloatTensor
        """
        emb = self.embed(input_ids)         # (batch, seq, d_model)
        rnn_out, _ = self.lstm(emb)          # (batch, seq, d_hidden)
        rnn_out = self.ln(rnn_out)
        logits = self.output_proj(rnn_out)   # (batch, seq, vocab_size)
        return logits


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def create_rnn_from_config(cfg, tokenizer) -> SequenceRNN:
    """Create a SequenceRNN from a Hydra-style config namespace."""
    device = _select_device()
    d_model = getattr(cfg.model, "d_model", 128)
    d_hidden = getattr(cfg.model, "d_hidden", 256)
    n_layers = getattr(cfg.model, "n_rnn_layers", 2)

    model = SequenceRNN(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        d_hidden=d_hidden,
        n_layers=n_layers,
        device=device,
    )
    model = model.to(device)
    model.cfg.device = device
    return model
