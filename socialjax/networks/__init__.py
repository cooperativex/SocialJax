"""SocialJax networks module.

This module provides neural network architectures for multi-agent reinforcement
learning algorithms. It includes:
- A registry system for network classes
- A factory function for creating network instances
- Preset configurations for common architectures

Example usage:
    from socialjax.networks import create_network, register_network, NETWORK_CONFIGS

    # Register a custom network
    @register_network("my_network")
    class MyNetwork(nn.Module):
        action_dim: int
        ...

    # Create a network instance
    network = create_network("my_network", action_dim=4)

    # Use a preset configuration
    network = create_network("my_network", action_dim=4, config_preset="medium")
"""

from socialjax.networks.registry import (
    register_network,
    get_network_class,
    list_networks,
    unregister_network,
    is_network_registered,
    clear_registry,
    NetworkAlreadyRegisteredError,
    NetworkNotFoundError,
)

from socialjax.networks.factory import (
    create_network,
    get_config_preset,
    list_config_presets,
    NETWORK_CONFIGS,
)

from socialjax.networks.cnn import (
    CNNSmall,
    CNNActorCritic,
    CNNSmallEncoder,
    CNNImpala,
)

from socialjax.networks.mlp import (
    MLPSmall,
    MLPActorCritic,
    MLPEncoder,
    MLPLargeActorCritic,
)

__all__ = [
    # Registry functions
    "register_network",
    "get_network_class",
    "list_networks",
    "unregister_network",
    "is_network_registered",
    "clear_registry",
    # Exceptions
    "NetworkAlreadyRegisteredError",
    "NetworkNotFoundError",
    # Factory functions
    "create_network",
    "get_config_preset",
    "list_config_presets",
    # Configuration
    "NETWORK_CONFIGS",
    # CNN Networks
    "CNNSmall",
    "CNNActorCritic",
    "CNNSmallEncoder",
    "CNNImpala",
    # MLP Networks
    "MLPSmall",
    "MLPActorCritic",
    "MLPEncoder",
    "MLPLargeActorCritic",
]
