"""
Module: Counterfactual Regret Calculation
Equation: Eq.9 from paper

Computes counterfactual regret which measures how much better
the collective reward could have been if the agent had chosen
a different action.

Regret_t^i = max_{a^{cf}}[R^{-i,cf}] - R^{-i}

Where:
- R^{-i,cf} is the collective counterfactual reward (sum of other agents' rewards)
  for each possible counterfactual action
- R^{-i} is the actual collective reward (sum of other agents' actual rewards)
- The max is taken over all possible counterfactual actions

Regret is always >= 0 because we compare to the best possible action.

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial


def compute_counterfactual_regret(
    collective_cf_rewards: jnp.ndarray,
    actual_collective: jnp.ndarray,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """
    Compute counterfactual regret for each agent.

    Implements Eq.9: Regret_t^i = max_{a^{cf}}[R^{-i,cf}] - R^{-i}

    Regret measures how much better the collective reward for other agents
    could have been if agent i had chosen a different (more prosocial) action.
    Regret is always non-negative since we compare to the maximum.

    Args:
        collective_cf_rewards: Collective counterfactual rewards for each agent
                               Shape: [num_agents, action_dim, batch]
                               where collective_cf_rewards[i, a, b] is the sum of
                               other agents' rewards when agent i takes action a
                               (from compute_collective_cf_reward)
        actual_collective: Actual collective rewards for each agent
                          Shape: [batch, num_agents]
                          where actual_collective[b, i] is the sum of other agents'
                          actual rewards (from compute_actual_collective_reward)
        epsilon: Small tolerance for floating point errors when ensuring non-negativity

    Returns:
        regret: Counterfactual regret for each agent
                Shape: [batch, num_agents]
                Always >= 0 (with tolerance for floating point errors)

    Notes:
        - When current action is optimal for prosocial behavior, regret = 0
        - When better actions exist, regret > 0
        - Higher regret means more suboptimal the current action is for the collective
    """
    # Find the maximum collective CF reward over all counterfactual actions
    # collective_cf_rewards: [num_agents, action_dim, batch]
    # max_cf: [num_agents, batch]
    max_cf_rewards = jnp.max(collective_cf_rewards, axis=1)

    # Transpose to [batch, num_agents] to match actual_collective shape
    max_cf_rewards = max_cf_rewards.T  # [batch, num_agents]

    # Compute regret: max - actual
    # If actual is already optimal, regret = 0
    # If better action exists, regret > 0
    regret = max_cf_rewards - actual_collective

    # Ensure non-negative (handle floating point errors)
    # Theoretically regret >= 0, but numerical errors might cause small negatives
    regret = jnp.maximum(regret, -epsilon)

    return regret


def compute_regret_with_best_action(
    collective_cf_rewards: jnp.ndarray,
    actual_collective: jnp.ndarray,
    actual_actions: jnp.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute counterfactual regret and identify the best prosocial action.

    In addition to computing regret, this function also returns the index
    of the action that maximizes collective counterfactual reward.

    Args:
        collective_cf_rewards: Collective counterfactual rewards
                               Shape: [num_agents, action_dim, batch]
        actual_collective: Actual collective rewards
                          Shape: [batch, num_agents]
        actual_actions: Current actions taken by each agent
                       Shape: [batch, num_agents]
        epsilon: Tolerance for floating point errors

    Returns:
        regret: Counterfactual regret for each agent [batch, num_agents]
        best_actions: The best prosocial action for each agent [batch, num_agents]
    """
    # Find argmax over counterfactual actions
    # collective_cf_rewards: [num_agents, action_dim, batch]
    # best_actions: [num_agents, batch]
    best_actions = jnp.argmax(collective_cf_rewards, axis=1)

    # Transpose to [batch, num_agents]
    best_actions = best_actions.T

    # Compute regret
    regret = compute_counterfactual_regret(
        collective_cf_rewards, actual_collective, epsilon
    )

    return regret, best_actions


def compute_normalized_regret(
    collective_cf_rewards: jnp.ndarray,
    actual_collective: jnp.ndarray,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """
    Compute normalized counterfactual regret in range [0, 1].

    Normalized regret = Regret / max_cf_reward

    This is useful when comparing regret across different scales
    or when using regret as a feature in downstream tasks.

    Args:
        collective_cf_rewards: Collective counterfactual rewards
                               Shape: [num_agents, action_dim, batch]
        actual_collective: Actual collective rewards
                          Shape: [batch, num_agents]
        epsilon: Tolerance for floating point errors and division

    Returns:
        normalized_regret: Normalized regret in [0, 1] [batch, num_agents]
    """
    # Compute raw regret
    regret = compute_counterfactual_regret(
        collective_cf_rewards, actual_collective, epsilon
    )

    # Get max CF rewards for normalization
    max_cf_rewards = jnp.max(collective_cf_rewards, axis=1).T  # [batch, num_agents]

    # Normalize (avoid division by zero)
    normalized_regret = regret / jnp.maximum(jnp.abs(max_cf_rewards), epsilon)

    # Clamp to [0, 1] to handle edge cases
    normalized_regret = jnp.clip(normalized_regret, 0.0, 1.0)

    return normalized_regret


def get_regret_statistics(
    regret: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute statistics over regret values.

    Useful for logging and monitoring during training.

    Args:
        regret: Counterfactual regret [batch, num_agents]

    Returns:
        mean_regret: Mean regret per agent [num_agents]
        max_regret: Maximum regret per agent [num_agents]
        zero_regret_ratio: Ratio of zero-regret cases per agent [num_agents]
                          (indicates how often agents choose optimal prosocial actions)
    """
    # Mean regret per agent
    mean_regret = jnp.mean(regret, axis=0)  # [num_agents]

    # Max regret per agent
    max_regret = jnp.max(regret, axis=0)  # [num_agents]

    # Ratio of zero regret (agents choosing optimally)
    zero_regret_ratio = jnp.mean((regret < 1e-6).astype(jnp.float32), axis=0)

    return mean_regret, max_regret, zero_regret_ratio
