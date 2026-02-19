"""SVO (Social Value Orientation) algorithm module.

This module provides the SVO algorithm for multi-agent reinforcement learning
with social preferences. SVO allows agents to balance self-interest with
collective welfare based on an angle parameter.
"""

from socialjax.algorithms.svo.config import (
    SVO_DEFAULT_CONFIG,
    get_svo_config,
    svo_angle_to_radians,
    get_svo_weights,
)
from socialjax.algorithms.svo.network import (
    SVOCNN,
    SVOActorCritic,
)
from socialjax.algorithms.svo.algorithm import (
    Transition,
    SVOAlgorithmState,
    SVOAlgorithm,
    compute_svo_reward,
    compute_batch_svo_reward,
)

__all__ = [
    # Config
    "SVO_DEFAULT_CONFIG",
    "get_svo_config",
    "svo_angle_to_radians",
    "get_svo_weights",
    # Network
    "SVOCNN",
    "SVOActorCritic",
    # Algorithm
    "Transition",
    "SVOAlgorithmState",
    "SVOAlgorithm",
    "compute_svo_reward",
    "compute_batch_svo_reward",
]
