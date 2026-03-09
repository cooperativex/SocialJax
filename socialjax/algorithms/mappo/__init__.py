"""MAPPO (Multi-Agent Proximal Policy Optimization) algorithm module.

This module provides the MAPPO implementation for multi-agent reinforcement learning.
MAPPO uses centralized training with decentralized execution (CTDE):
- Centralized critic receives observations from all agents
- Decentralized actor only receives local observations for execution

Usage:
    >>> from socialjax.algorithms.mappo import MAPPOAlgorithm, get_mappo_config
    >>> config = get_mappo_config({"LR": 0.0003})
    >>> algo = MAPPOAlgorithm(observation_space, action_space, config, num_agents=5)
"""

from socialjax.algorithms.mappo.config import (
    MAPPO_DEFAULT_CONFIG,
    get_mappo_config,
)
from socialjax.algorithms.mappo.network import (
    MAPPOActor,
    MAPPOActorCNN,
    MAPPOActorCritic,
    MAPPOCritic,
    MAPPOCriticCNN,
)
from socialjax.algorithms.mappo.algorithm import (
    MAPPOAlgorithm,
    MAPPOAlgorithmState,
    Transition,
)

__all__ = [
    # Configuration
    "MAPPO_DEFAULT_CONFIG",
    "get_mappo_config",
    # Networks
    "MAPPOActor",
    "MAPPOActorCNN",
    "MAPPOActorCritic",
    "MAPPOCritic",
    "MAPPOCriticCNN",
    # Algorithm
    "MAPPOAlgorithm",
    "MAPPOAlgorithmState",
    "Transition",
]
