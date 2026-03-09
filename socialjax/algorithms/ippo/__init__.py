"""IPPO (Independent Proximal Policy Optimization) algorithm module.

This module provides the IPPO implementation for multi-agent reinforcement learning.
IPPO trains each agent independently using PPO with shared parameters.

Usage:
    >>> from socialjax.algorithms.ippo import IPPOAlgorithm, get_ippo_config
    >>> config = get_ippo_config({"LR": 0.0003})
    >>> algo = IPPOAlgorithm(observation_space, action_space, config)
"""

from socialjax.algorithms.ippo.config import (
    IPPO_DEFAULT_CONFIG,
    get_ippo_config,
)
from socialjax.algorithms.ippo.network import (
    IPPOActorCritic,
    IPPOCNN,
)
from socialjax.algorithms.ippo.algorithm import (
    IPPOAlgorithm,
    IPPOAlgorithmState,
    Transition,
)

__all__ = [
    # Configuration
    "IPPO_DEFAULT_CONFIG",
    "get_ippo_config",
    # Networks
    "IPPOActorCritic",
    "IPPOCNN",
    # Algorithm
    "IPPOAlgorithm",
    "IPPOAlgorithmState",
    "Transition",
]
