"""Configuration for MAPPO (Multi-Agent Proximal Policy Optimization) algorithm.

This module defines the default hyperparameters and configuration options
for the MAPPO algorithm used in multi-agent reinforcement learning.

MAPPO uses centralized training with decentralized execution (CTDE):
- The critic has access to global information (all agent observations)
- The actor only has access to local observations (for decentralized execution)
"""

from typing import Dict, Any

# Default MAPPO configuration
MAPPO_DEFAULT_CONFIG: Dict[str, Any] = {
    # Training hyperparameters
    "LR": 2.5e-4,  # Learning rate (shared for actor and critic)
    "LR_ACTOR": None,  # Separate actor learning rate (uses LR if None)
    "LR_CRITIC": None,  # Separate critic learning rate (uses LR if None)
    "ANNEAL_LR": True,  # Whether to anneal learning rate
    "GAMMA": 0.99,  # Discount factor
    "GAE_LAMBDA": 0.95,  # GAE lambda parameter

    # PPO specific hyperparameters
    "CLIP_EPS": 0.2,  # PPO clipping parameter
    "SCALE_CLIP_EPS": True,  # Scale clip_eps by num_agents
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

    # MAPPO specific
    "USE_CENTRALIZED_VALUE": True,  # Use centralized value function
    "POPULATE_CRITIC_VALUE": True,  # Replicate critic value for all agents
}


def get_mappo_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get MAPPO configuration with optional overrides.

    Args:
        overrides: Dictionary of configuration overrides.

    Returns:
        Complete MAPPO configuration dictionary.
    """
    config = MAPPO_DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)

    # Set default learning rates if not specified
    if config["LR_ACTOR"] is None:
        config["LR_ACTOR"] = config["LR"]
    if config["LR_CRITIC"] is None:
        config["LR_CRITIC"] = config["LR"]

    return config
