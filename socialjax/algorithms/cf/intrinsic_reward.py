"""
Module: Intrinsic Reward Construction
Equation: Eq.10 from paper

Constructs intrinsic reward from counterfactual regret.
The intrinsic reward is the negative of regret, which encourages
agents to choose prosocial actions.

r_t^{i,in} = -Regret_t^i

Where:
- Regret_t^i is the counterfactual regret from Eq.9
- When agent chooses the optimal prosocial action, regret = 0, so intrinsic = 0
- When agent chooses a suboptimal action, regret > 0, so intrinsic < 0

The intrinsic reward serves as a learning signal to guide agents toward
actions that maximize collective welfare.

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial


def compute_intrinsic_reward(
    regret: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute intrinsic reward from counterfactual regret.

    Implements Eq.10: r_t^{i,in} = -Regret_t^i

    The intrinsic reward is the negative of regret. This encourages agents
    to minimize regret (i.e., choose actions that maximize collective welfare).

    Properties:
    - When current action is optimal for prosocial behavior, regret = 0, so intrinsic = 0
    - When better actions exist (current is suboptimal), regret > 0, so intrinsic < 0
    - Maximum intrinsic reward is 0 (achieved when choosing optimal prosocial action)
    - Minimum intrinsic reward is unbounded below (more regret = more negative intrinsic)

    Args:
        regret: Counterfactual regret for each agent
                Shape: [batch, num_agents]
                Must be >= 0 (from compute_counterfactual_regret)

    Returns:
        intrinsic_reward: Intrinsic reward for each agent
                         Shape: [batch, num_agents]
                         Always <= 0

    Example:
        >>> regret = jnp.array([[0.0, 0.5, 1.0]])  # Agent 0 optimal, Agent 2 suboptimal
        >>> intrinsic = compute_intrinsic_reward(regret)
        >>> # intrinsic = [[0.0, -0.5, -1.0]]
    """
    intrinsic_reward = -regret
    return intrinsic_reward


def compute_intrinsic_reward_from_cf(
    collective_cf_rewards: jnp.ndarray,
    actual_collective: jnp.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convenience function to compute intrinsic reward directly from CF rewards.

    Combines regret computation (Eq.9) and intrinsic reward computation (Eq.10)
    into a single function.

    Args:
        collective_cf_rewards: Collective counterfactual rewards
                               Shape: [num_agents, action_dim, batch]
        actual_collective: Actual collective rewards
                          Shape: [batch, num_agents]
        epsilon: Tolerance for floating point errors

    Returns:
        regret: Counterfactual regret [batch, num_agents]
        intrinsic_reward: Intrinsic reward [batch, num_agents]

    Note:
        This function is provided for convenience. For efficiency when you need
        both regret and intrinsic reward, use this function. If you only need
        one or the other, use the individual functions.
    """
    # Import here to avoid circular dependency
    from .regret import compute_counterfactual_regret

    # Compute regret first (Eq.9)
    regret = compute_counterfactual_regret(
        collective_cf_rewards, actual_collective, epsilon
    )

    # Then compute intrinsic reward (Eq.10)
    intrinsic_reward = compute_intrinsic_reward(regret)

    return regret, intrinsic_reward


def compute_scaled_intrinsic_reward(
    regret: jnp.ndarray,
    alpha: float = 1.0,
) -> jnp.ndarray:
    """
    Compute scaled intrinsic reward.

    This is a convenience function that computes intrinsic reward and scales
    it by a coefficient. The scaled intrinsic reward can be directly added
    to the extrinsic reward for the shaped reward (Eq.11).

    r^{in,scaled} = alpha * (-Regret)

    Args:
        regret: Counterfactual regret [batch, num_agents]
        alpha: Scaling coefficient (default 1.0)
               - Higher alpha = agent cares more about collective welfare
               - Paper suggests alpha ≈ N-1 (number of other agents)

    Returns:
        scaled_intrinsic: Scaled intrinsic reward [batch, num_agents]

    Note:
        This is equivalent to: alpha * compute_intrinsic_reward(regret)
    """
    return alpha * compute_intrinsic_reward(regret)


def get_intrinsic_reward_statistics(
    intrinsic_reward: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute statistics over intrinsic reward values.

    Useful for logging and monitoring during training.

    Args:
        intrinsic_reward: Intrinsic rewards [batch, num_agents]

    Returns:
        mean_intrinsic: Mean intrinsic reward per agent [num_agents]
                       (should be <= 0)
        min_intrinsic: Minimum (most negative) intrinsic reward per agent [num_agents]
                      (indicates worst suboptimal choices)
        zero_intrinsic_ratio: Ratio of zero-intrinsic cases per agent [num_agents]
                            (indicates how often agents choose optimal prosocial actions)
    """
    # Mean intrinsic reward per agent
    mean_intrinsic = jnp.mean(intrinsic_reward, axis=0)  # [num_agents]

    # Minimum (most negative) intrinsic reward per agent
    min_intrinsic = jnp.min(intrinsic_reward, axis=0)  # [num_agents]

    # Ratio of zero intrinsic reward (agents choosing optimally)
    # Since intrinsic = -regret, zero intrinsic means zero regret
    zero_intrinsic_ratio = jnp.mean((intrinsic_reward > -1e-6).astype(jnp.float32), axis=0)

    return mean_intrinsic, min_intrinsic, zero_intrinsic_ratio


def compute_intrinsic_reward_gradient(
    regret: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the gradient of intrinsic reward with respect to regret.

    Since r^{in} = -Regret, the gradient is simply -1.

    This function is provided for documentation and testing purposes.
    In practice, JAX autodiff handles this automatically.

    Args:
        regret: Counterfactual regret [batch, num_agents]

    Returns:
        gradient: Gradient of intrinsic reward w.r.t. regret
                 Shape: [batch, num_agents]
                 All values are -1.0
    """
    return -jnp.ones_like(regret)
