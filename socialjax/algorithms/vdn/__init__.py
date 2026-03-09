"""VDN (Value Decomposition Networks) algorithm module.

This module provides the VDN algorithm for cooperative multi-agent
reinforcement learning, implementing value decomposition for
centralized training with decentralized execution.

VDN decomposes the team Q-value into individual agent Q-values:
    Q_tot(s, a) = sum_i Q_i(s_i, a_i)

Main components:
    - VDNAlgorithm: Main algorithm class
    - VDNQNetwork: Q-network for value estimation
    - VDNAlgorithmState: State container for training
    - VDN_DEFAULT_CONFIG: Default hyperparameters
"""

from socialjax.algorithms.vdn.config import (
    VDN_DEFAULT_CONFIG,
    get_vdn_config,
)

from socialjax.algorithms.vdn.network import (
    VDNCNN,
    VDNQNetwork,
    compute_q_tot,
    compute_vdn_target,
)

from socialjax.algorithms.vdn.algorithm import (
    VDNAlgorithm,
    VDNAlgorithmState,
    VDNTransition,
)

__all__ = [
    # Config
    "VDN_DEFAULT_CONFIG",
    "get_vdn_config",
    # Networks
    "VDNCNN",
    "VDNQNetwork",
    "compute_q_tot",
    "compute_vdn_target",
    # Algorithm
    "VDNAlgorithm",
    "VDNAlgorithmState",
    "VDNTransition",
]
