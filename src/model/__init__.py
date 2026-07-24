from .hooked_transformer import create_hooked_transformer
from .hooked_transformer import create_model_from_config as _create_transformer_from_config


def create_model_from_config(cfg, tokenizer):
    """Unified model factory that dispatches on ``cfg.model.architecture``.

    Falls back to ``'transformer'`` when the field is absent, preserving
    backward compatibility with existing experiment configs.
    """
    arch = getattr(getattr(cfg, "model", None), "architecture", "transformer")

    if arch == "transformer":
        return _create_transformer_from_config(cfg, tokenizer)
    elif arch == "gated_mlp":
        from .gated_mlp import create_gated_mlp_from_config
        return create_gated_mlp_from_config(cfg, tokenizer)
    elif arch == "rnn":
        from .rnn_model import create_rnn_from_config
        return create_rnn_from_config(cfg, tokenizer)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


__all__ = [
    "create_hooked_transformer",
    "create_model_from_config",
]
