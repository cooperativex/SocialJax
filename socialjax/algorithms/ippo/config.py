"""Configuration for IPPO (Independent Proximal Policy Optimization) algorithm.

This module defines the default hyperparameters and configuration options
for the IPPO algorithm used in multi-agent reinforcement learning.
"""

from typing import Dict, Any

# Default IPPO configuration
IPPO_DEFAULT_CONFIG: Dict[str, Any] = {
    # Training hyperparameters
    "LR": 2.5e-4,  # Learning rate
    "ANNEAL_LR": True,  # Whether to anneal learning rate
    "GAMMA": 0.99,  # Discount factor
    "GAE_LAMBDA": 0.95,  # GAE lambda parameter

    # PPO specific hyperparameters
    "CLIP_EPS": 0.2,  # PPO clipping parameter
    "VF_COEF": 0.5,  # Value function coefficient
    "ENT_COEF": 0.01,  # Entropy coefficient
    "MAX_GRAD_NORM": 0.5,  # Maximum gradient norm for clipping

    # Training schedule
    "UPDATE_EPOCHS": 4,  # Number of epochs per update
    "NUM_MINIBATCHES": 4,  # Number of minibatches
    "NUM_STEPS": 128,  # Number of steps per rollout

    # Network architecture
    "ACTIVATION": "relu",  # Activation function ("relu" or "tanh")
    "HIDDEN_SIZE": 64,  # Hidden layer size

    # Multi-agent specific
    "PARAMETER_SHARING": True,  # Whether to share parameters between agents
}


def get_ippo_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get IPPO configuration with optional overrides.

    Args:
        overrides: Dictionary of configuration overrides.

    Returns:
        Complete IPPO configuration dictionary.
    """
    config = IPPO_DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
