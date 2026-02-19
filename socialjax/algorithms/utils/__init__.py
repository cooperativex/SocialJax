"""SocialJax algorithm utilities module.

This module provides shared utilities for multi-agent reinforcement learning
algorithms, including:

- **GAE (Generalized Advantage Estimation)**: Advantage computation for policy gradients
- **PPO Loss**: Clipped surrogate objective for PPO/IPPO/MAPPO
- **Value Decomposition**: VDN/QMIX utilities for cooperative MARL

All utilities are JAX-JIT compatible and designed for high-performance training.

Example:
    >>> from socialjax.algorithms.utils import compute_gae, compute_ppo_loss
    >>> from socialjax.algorithms.utils import vdn_decomposition, GAETransition
    >>>
    >>> # Compute GAE for a trajectory
    >>> traj = GAETransition(done=dones, value=values, reward=rewards)
    >>> advantages, targets = compute_gae(traj, last_value)
    >>>
    >>> # Compute VDN decomposition
    >>> output = vdn_decomposition(q_values, actions)
    >>> q_tot = output.q_tot

Modules:
    gae: Generalized Advantage Estimation utilities
    ppo_update: PPO loss computation utilities
    value_decomposition: VDN/QMIX value decomposition utilities
"""

# GAE utilities
from socialjax.algorithms.utils.gae import (
    compute_gae,
    compute_gae_batched,
    normalize_advantages,
    compute_returns,
    GAETransition,
    TransitionProtocol,
)

# PPO update utilities
from socialjax.algorithms.utils.ppo_update import (
    compute_policy_loss,
    compute_value_loss,
    compute_entropy_bonus,
    compute_ppo_loss,
    create_ppo_update_fn,
    PPOLossComponents,
)

# Value decomposition utilities
from socialjax.algorithms.utils.value_decomposition import (
    vdn_decomposition,
    vdn_target,
    qmix_mixing_network,
    compute_td_loss,
    epsilon_greedy_action,
    soft_target_update,
    hard_target_update,
    create_vdn_loss_fn,
    ValueDecompositionOutput,
)

__all__ = [
    # GAE
    "compute_gae",
    "compute_gae_batched",
    "normalize_advantages",
    "compute_returns",
    "GAETransition",
    "TransitionProtocol",
    # PPO
    "compute_policy_loss",
    "compute_value_loss",
    "compute_entropy_bonus",
    "compute_ppo_loss",
    "create_ppo_update_fn",
    "PPOLossComponents",
    # Value decomposition
    "vdn_decomposition",
    "vdn_target",
    "qmix_mixing_network",
    "compute_td_loss",
    "epsilon_greedy_action",
    "soft_target_update",
    "hard_target_update",
    "create_vdn_loss_fn",
    "ValueDecompositionOutput",
]
