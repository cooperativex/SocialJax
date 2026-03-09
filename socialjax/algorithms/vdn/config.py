"""Configuration for VDN (Value Decomposition Networks) algorithm.

This module defines the default hyperparameters and configuration options
for the VDN algorithm used in cooperative multi-agent reinforcement learning.

VDN uses value decomposition where the total team Q-value is the sum of
individual agent Q-values: Q_tot = sum(Q_i)
"""

from typing import Dict, Any

# Default VDN configuration
VDN_DEFAULT_CONFIG: Dict[str, Any] = {
    # Training hyperparameters
    "LR": 0.001,  # Learning rate
    "GAMMA": 0.99,  # Discount factor
    "MAX_GRAD_NORM": 10.0,  # Maximum gradient norm for clipping

    # Exploration
    "EPS_START": 1.0,  # Initial exploration epsilon
    "EPS_FINISH": 0.05,  # Final exploration epsilon
    "EPS_DECAY": 0.5,  # Fraction of training to decay epsilon

    # Target network
    "TARGET_UPDATE_INTERVAL": 200,  # Steps between target network updates
    "TAU": 1.0,  # Soft update coefficient (1.0 = hard update)

    # Experience replay
    "BUFFER_SIZE": 5000,  # Replay buffer size
    "BUFFER_BATCH_SIZE": 32,  # Batch size for training
    "LEARNING_STARTS": 1000,  # Steps before learning starts

    # Training schedule
    "NUM_EPOCHS": 4,  # Number of epochs per update
    "NUM_STEPS": 128,  # Number of steps per rollout
    "NUM_ENVS": 8,  # Number of parallel environments

    # Network architecture
    "ACTIVATION": "relu",  # Activation function ("relu" or "tanh")
    "HIDDEN_SIZE": 64,  # Hidden layer size

    # Multi-agent specific
    "PARAMETER_SHARING": True,  # Whether to share parameters between agents

    # Reward shaping (optional, environment-dependent)
    "REW_SHAPING_HORIZON": 100000,  # Steps to anneal shaped rewards
}


def get_vdn_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get VDN configuration with optional overrides.

    Args:
        overrides: Dictionary of configuration overrides.

    Returns:
        Complete VDN configuration dictionary.
    """
    config = VDN_DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
