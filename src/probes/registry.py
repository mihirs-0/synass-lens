"""
Probe registry for easy access and configuration.
"""

from typing import Dict, Type, Any, List
from .base import BaseProbe
from .attention_to_z import AttentionToZProbe
from .logit_lens import LogitLensProbe
from .causal_patching import CausalPatchingProbe
from .random_z_eval import RandomZEvalProbe


# Registry of all available probes
PROBE_REGISTRY: Dict[str, Type[BaseProbe]] = {
    "attention_to_z": AttentionToZProbe,
    "logit_lens": LogitLensProbe,
    "causal_patching": CausalPatchingProbe,
    "random_z_eval": RandomZEvalProbe,
}


def get_probe(name: str, config: Dict[str, Any] = None) -> BaseProbe:
    """
    Get a probe instance by name.
    
    Args:
        name: Probe name (must be in PROBE_REGISTRY)
        config: Optional probe-specific configuration
        
    Returns:
        Instantiated probe
    """
    if name not in PROBE_REGISTRY:
        raise ValueError(f"Unknown probe: {name}. Available: {list(PROBE_REGISTRY.keys())}")
    
    return PROBE_REGISTRY[name](config)


def get_all_probes(probe_configs: Dict[str, Dict[str, Any]] = None) -> List[BaseProbe]:
    """
    Get instances of all registered probes.
    
    Args:
        probe_configs: Dict mapping probe names to their configs
        
    Returns:
        List of all probe instances
    """
    if probe_configs is None:
        probe_configs = {}
        
    return [
        get_probe(name, probe_configs.get(name, {}))
        for name in PROBE_REGISTRY.keys()
    ]


def list_probes() -> List[str]:
    """List all available probe names."""
    return list(PROBE_REGISTRY.keys())
