"""Network factory for SocialJax.

This module provides a factory function for creating network instances
from registered network classes. It also defines preset network configurations
for common use cases.
"""

from typing import Any, Dict, Optional
from flax import linen as nn

from socialjax.networks.registry import get_network_class, list_networks

# Network configuration presets for different scales
NETWORK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small": {
        "hidden_size": 64,
        "num_layers": 2,
        "channel_sizes": [16, 32],
        "kernel_sizes": [3, 3],
    },
    "medium": {
        "hidden_size": 128,
        "num_layers": 3,
        "channel_sizes": [32, 64, 64],
        "kernel_sizes": [3, 3, 3],
    },
    "large": {
        "hidden_size": 256,
        "num_layers": 4,
        "channel_sizes": [32, 64, 128, 128],
        "kernel_sizes": [3, 3, 3, 3],
    },
}

def create_network(name: str, action_dim: int, config_preset: Optional[str] = None, **kwargs) -> nn.Module:
    """Create a network instance by name."""
    network_class = get_network_class(name)
    if config_preset is not None:
        if config_preset not in NETWORK_CONFIGS:
            raise ValueError(f"Unknown config preset '{config_preset}'. Available: {list(NETWORK_CONFIGS.keys())}")
        merged_kwargs = {**NETWORK_CONFIGS[config_preset], **kwargs}
    else:
        merged_kwargs = kwargs
    return network_class(action_dim=action_dim, **merged_kwargs)

def get_config_preset(name: str) -> Dict[str, Any]:
    """Get a network configuration preset by name."""
    if name not in NETWORK_CONFIGS:
        raise ValueError(f"Unknown config preset '{name}'. Available: {list(NETWORK_CONFIGS.keys())}")
    return NETWORK_CONFIGS[name].copy()

def list_config_presets() -> list:
    """List all available configuration preset names."""
    return sorted(NETWORK_CONFIGS.keys())
