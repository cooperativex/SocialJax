"""
Module: Counterfactual Reward Generation
Equations: Eq.7, Eq.8 from paper

Generates counterfactual rewards by:
1. Enumerating all possible actions for a given agent
2. Keeping other agents' actions fixed
3. Using the generative model to predict rewards for each counterfactual action

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial


def enumerate_counterfactual_actions(
    actual_actions: jnp.ndarray,
    agent_id: int,
    action_dim: int,
) -> jnp.ndarray:
    """
    Enumerate all possible counterfactual actions for a specific agent.

    For each possible action a_cf in [0, action_dim), creates a modified
    action vector where the target agent's action is replaced with a_cf,
    while all other agents' actions remain unchanged.

    Args:
        actual_actions: Current joint actions [batch, num_agents]
        agent_id: The agent for which to enumerate counterfactual actions
        action_dim: Number of discrete actions available

    Returns:
        cf_actions: Counterfactual action tensors [action_dim, batch, num_agents]
                    where cf_actions[a] is the joint action with agent_id taking action a
    """
    batch_size = actual_actions.shape[0]
    num_agents = actual_actions.shape[1]

    # Create array of all possible actions [action_dim]
    all_actions = jnp.arange(action_dim, dtype=actual_actions.dtype)

    # Expand actual_actions to [action_dim, batch, num_agents]
    # by repeating for each counterfactual action
    cf_actions = jnp.tile(actual_actions[jnp.newaxis, :, :], (action_dim, 1, 1))

    # Replace the agent_id column with each counterfactual action
    # cf_actions[a, :, agent_id] = a for all a
    agent_actions = jnp.tile(all_actions[:, jnp.newaxis], (1, batch_size))
    cf_actions = cf_actions.at[:, :, agent_id].set(agent_actions)

    return cf_actions


def enumerate_all_agents_counterfactual_actions(
    actual_actions: jnp.ndarray,
    action_dim: int,
) -> jnp.ndarray:
    """
    Enumerate counterfactual actions for all agents.

    For each agent, enumerates all possible actions while keeping others fixed.
    This is useful for batch processing all agents at once.

    Args:
        actual_actions: Current joint actions [batch, num_agents]
        action_dim: Number of discrete actions available

    Returns:
        cf_actions: Counterfactual action tensors
                    [num_agents, action_dim, batch, num_agents]
                    where cf_actions[i, a] is joint action with agent i taking action a
    """
    batch_size = actual_actions.shape[0]
    num_agents = actual_actions.shape[1]

    # Create output array [num_agents, action_dim, batch, num_agents]
    cf_actions = jnp.zeros((num_agents, action_dim, batch_size, num_agents),
                           dtype=actual_actions.dtype)

    # For each agent, enumerate their counterfactual actions
    for agent_id in range(num_agents):
        agent_cf = enumerate_counterfactual_actions(actual_actions, agent_id, action_dim)
        cf_actions = cf_actions.at[agent_id].set(agent_cf)

    return cf_actions


def generate_counterfactual_rewards_single_agent(
    reward_model_apply,
    params: dict,
    obs: jnp.ndarray,
    agent_id: int,
    action_dim: int,
    actual_actions: jnp.ndarray,
) -> jnp.ndarray:
    """
    Generate counterfactual rewards for a single agent.

    Implements Eq.7: r_t^{i,cf} = Φ_m^i(o_t, a_t^{i,cf}, a_t^{-i})

    For each possible action a_cf, predicts what rewards all agents would receive
    if agent_id took action a_cf while all other agents' actions remain unchanged.

    Args:
        reward_model_apply: The apply function of the RewardModel (bound method)
        params: Model parameters
        obs: Joint observations [batch, num_agents, H, W, C]
        agent_id: The agent for which to generate counterfactuals
        action_dim: Number of discrete actions
        actual_actions: Current joint actions [batch, num_agents]

    Returns:
        cf_rewards: Predicted rewards for each counterfactual action
                    [action_dim, batch, num_agents]
                    where cf_rewards[a, b, i] is predicted reward for agent i
                    in batch b when agent_id takes action a
    """
    batch_size = obs.shape[0]
    num_agents = obs.shape[1]

    # Get counterfactual action combinations [action_dim, batch, num_agents]
    cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id, action_dim)

    # Reshape obs to [action_dim, batch, num_agents, H, W, C]
    obs_expanded = jnp.tile(obs[jnp.newaxis, :, :], (action_dim, 1, 1, 1, 1, 1))

    # Reshape for batch processing: [action_dim * batch, num_agents, H, W, C]
    obs_flat = obs_expanded.reshape(-1, *obs.shape[1:])  # Keep num_agents dimension
    actions_flat = cf_actions.reshape(-1, num_agents)

    # Get predictions for all counterfactual actions at once
    cf_rewards_flat = reward_model_apply(params, obs_flat, actions_flat)

    # Reshape back to [action_dim, batch, num_agents]
    cf_rewards = cf_rewards_flat.reshape(action_dim, batch_size, num_agents)

    return cf_rewards


def generate_counterfactual_rewards_vmap(
    reward_model_apply,
    params: dict,
    action_dim: int,
    obs: jnp.ndarray,
    actual_actions: jnp.ndarray,
) -> jnp.ndarray:
    """
    Generate counterfactual rewards for all agents using vmap.

    This version uses jax.vmap to efficiently compute counterfactual rewards
    for all agents in parallel.

    Args:
        reward_model_apply: The apply function of the RewardModel
        params: Model parameters
        action_dim: Number of discrete actions
        obs: Joint observations [batch, num_agents, H, W, C]
        actual_actions: Current joint actions [batch, num_agents]

    Returns:
        cf_rewards: Predicted rewards for each counterfactual action
                    [num_agents, action_dim, batch, num_agents]
                    where cf_rewards[i, a, b, j] is predicted reward for agent j
                    in batch b when agent i takes action a
    """
    batch_size = obs.shape[0]
    num_agents = obs.shape[1]

    def compute_for_agent(agent_id):
        """Compute counterfactual rewards for a single agent."""
        return generate_counterfactual_rewards_single_agent(
            reward_model_apply,
            params,
            obs,
            agent_id,
            action_dim,
            actual_actions,
        )

    # Use vmap to compute for all agents in parallel
    # agent_id ranges from 0 to num_agents-1
    agent_ids = jnp.arange(num_agents)
    cf_rewards = jax.vmap(compute_for_agent)(agent_ids)

    return cf_rewards  # [num_agents, action_dim, batch, num_agents]


def generate_counterfactual_rewards(
    reward_model_apply,
    params: dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    action_dim: int,
    agent_ids: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Generate counterfactual rewards for specified agents.

    This is the main entry point for counterfactual reward generation.
    Implements Eq.7: r_t^{i,cf} = Φ_m^i(o_t, a_t^{i,cf}, a_t^{-i})

    Args:
        reward_model_apply: The apply function of the RewardModel
        params: Model parameters
        obs: Joint observations [batch, num_agents, H, W, C]
        actions: Current joint actions [batch, num_agents]
        action_dim: Number of discrete actions
        agent_ids: Agents to compute counterfactuals for [num_target_agents]
                   If None, computes for all agents

    Returns:
        cf_rewards: Predicted counterfactual rewards
                    If agent_ids is None:
                        [num_agents, action_dim, batch, num_agents]
                    If agent_ids is provided:
                        [num_target_agents, action_dim, batch, num_agents]
                    where cf_rewards[i, a, b, j] is predicted reward for agent j
                    in batch b when agent i takes action a
    """
    num_agents = obs.shape[1]

    if agent_ids is None:
        # Compute for all agents using vmap
        return generate_counterfactual_rewards_vmap(
            reward_model_apply, params, action_dim, obs, actions
        )
    else:
        # Compute for specific agents
        def compute_for_agent(agent_id):
            return generate_counterfactual_rewards_single_agent(
                reward_model_apply,
                params,
                obs,
                agent_id,
                action_dim,
                actions,
            )

        cf_rewards = jax.vmap(compute_for_agent)(agent_ids)
        return cf_rewards


def compute_collective_cf_reward(
    cf_rewards: jnp.ndarray,
    exclude_self: bool = True,
) -> jnp.ndarray:
    """
    Compute collective counterfactual rewards for each agent.

    Implements Eq.8: R^{-i,cf}_t = Σ_{j≠i} r_{j,t}^{i,cf}

    For each counterfactual action, sums the rewards of all other agents
    (excluding the ego agent's own reward).

    Args:
        cf_rewards: Counterfactual rewards [num_agents, action_dim, batch, num_agents]
                    where cf_rewards[i, a, b, j] is reward for agent j when
                    agent i takes counterfactual action a
        exclude_self: If True, exclude the ego agent's own reward from the sum

    Returns:
        collective_cf_rewards: Sum of other agents' rewards
                               [num_agents, action_dim, batch]
                               where collective_cf_rewards[i, a, b] is the
                               collective reward for agent i's counterfactual action a
    """
    if exclude_self:
        # Create mask to exclude ego agent
        num_agents = cf_rewards.shape[0]
        action_dim = cf_rewards.shape[1]
        batch_size = cf_rewards.shape[2]

        # mask[i, j] = 1 if i != j, else 0
        mask = 1.0 - jnp.eye(num_agents)  # [num_agents, num_agents]

        # Reshape mask for broadcasting: [num_agents, 1, 1, num_agents]
        mask = mask[:, jnp.newaxis, jnp.newaxis, :]

        # Apply mask and sum over other agents
        masked_rewards = cf_rewards * mask
        collective_cf_rewards = jnp.sum(masked_rewards, axis=-1)
    else:
        # Sum all agents' rewards
        collective_cf_rewards = jnp.sum(cf_rewards, axis=-1)

    return collective_cf_rewards  # [num_agents, action_dim, batch]


def compute_actual_collective_reward(
    rewards: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the actual collective reward for each agent.

    R^{-i} = Σ_{j≠i} r_j (sum of all other agents' rewards)

    Args:
        rewards: Actual rewards [batch, num_agents]

    Returns:
        collective_rewards: Collective reward for each agent [batch, num_agents]
                            where collective_rewards[b, i] = sum of rewards
                            for all agents except i in batch b
    """
    # Total reward per batch
    total_reward = jnp.sum(rewards, axis=-1, keepdims=True)  # [batch, 1]

    # Subtract own reward to get collective reward
    collective_rewards = total_reward - rewards  # [batch, num_agents]

    return collective_rewards


# Convenience function combining counterfactual generation and collective computation
def get_counterfactual_analysis(
    reward_model_apply,
    params: dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    action_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform complete counterfactual analysis for a batch.

    Combines:
    1. Counterfactual reward generation (Eq.7)
    2. Collective counterfactual reward computation (Eq.8)
    3. Actual collective reward computation

    Args:
        reward_model_apply: The apply function of the RewardModel
        params: Model parameters
        obs: Joint observations [batch, num_agents, H, W, C]
        actions: Current joint actions [batch, num_agents]
        rewards: Actual rewards [batch, num_agents]
        action_dim: Number of discrete actions

    Returns:
        cf_rewards: Counterfactual rewards [num_agents, action_dim, batch, num_agents]
        collective_cf_rewards: Collective CF rewards [num_agents, action_dim, batch]
        actual_collective: Actual collective rewards [batch, num_agents]
    """
    # Generate counterfactual rewards (Eq.7)
    cf_rewards = generate_counterfactual_rewards_vmap(
        reward_model_apply, params, action_dim, obs, actions
    )

    # Compute collective counterfactual rewards (Eq.8)
    collective_cf_rewards = compute_collective_cf_reward(cf_rewards, exclude_self=True)

    # Compute actual collective rewards
    actual_collective = compute_actual_collective_reward(rewards)

    return cf_rewards, collective_cf_rewards, actual_collective
