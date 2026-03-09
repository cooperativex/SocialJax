"""Configuration for SVO (Social Value Orientation) algorithm.

This module defines the default hyperparameters and configuration options
for the SVO algorithm used in multi-agent reinforcement learning with
social preferences.
"""

from typing import Dict, Any
import math

# Default SVO configuration
SVO_DEFAULT_CONFIG: Dict[str, Any] = {
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

    # SVO-specific parameters
    "SVO_ANGLE": 45.0,  # SVO angle in degrees (0=selfish, 45=cooperative, 90=altruistic)
    "USE_FAIRNESS_REWARD": True,  # Whether to use fairness-aware reward shaping
    "FAIRNESS_WEIGHT": 0.1,  # Weight for fairness component in reward
    "REW_SHAPING_HORIZON": 1000000,  # Horizon for annealing reward shaping
}


def get_svo_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get SVO configuration with optional overrides.

    Args:
        overrides: Dictionary of configuration overrides.

    Returns:
        Complete SVO configuration dictionary.
    """
    config = SVO_DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def svo_angle_to_radians(angle_degrees: float) -> float:
    """Convert SVO angle from degrees to radians.

    Args:
        angle_degrees: SVO angle in degrees (0-90)

    Returns:
        Angle in radians
    """
    return math.radians(angle_degrees)


def get_svo_weights(angle_degrees: float) -> tuple:
    """Get the self and other weights for SVO reward transformation.

    The SVO reward transformation is:
        r_svo = w_self * r_self + w_other * r_other

    Where:
        w_self = cos(angle)
        w_other = sin(angle)

    Typical SVO angles:
        0 degrees: Purely selfish (w_self=1, w_other=0)
        45 degrees: Cooperative (w_self=0.707, w_other=0.707)
        90 degrees: Purely altruistic (w_self=0, w_other=1)

    Args:
        angle_degrees: SVO angle in degrees (0-90)

    Returns:
        Tuple of (self_weight, other_weight)
    """
    angle_rad = svo_angle_to_radians(angle_degrees)
    return math.cos(angle_rad), math.sin(angle_rad)
