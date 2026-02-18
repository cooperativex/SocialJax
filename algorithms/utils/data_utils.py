"""
Shared data manipulation utilities for multi-agent reinforcement learning.

This module contains helper functions for batching/unbatching agent data,
which are used across different MARL algorithms (IPPO, MAPPO, SVO, etc.).
"""

import jax.numpy as jnp
from typing import Dict, List


def batchify(x: dict, agent_list: List, num_actors: int) -> jnp.ndarray:
    """
    Batch multi-agent data from dict format to array format.

    Used for batching observations/rewards when agents are indexed by integers
    in a 2D array (e.g., reward[env_idx, agent_idx]).

    Args:
        x: Dictionary with shape [num_envs, num_agents, ...] accessed as x[:, agent_idx]
        agent_list: List of agent indices
        num_actors: Total number of actors (num_envs * num_agents for parameter sharing,
                    or num_envs for independent learning)

    Returns:
        Batched array of shape [num_actors, ...]

    Example:
        >>> rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 2 envs, 2 agents
        >>> agent_list = [0, 1]
        >>> batched = batchify(rewards, agent_list, num_actors=4)
        >>> # Result shape: [4,] = [env0_agent0, env0_agent1, env1_agent0, env1_agent1]
    """
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_dict(x: Dict[str, jnp.ndarray], agent_list: List, num_actors: int) -> jnp.ndarray:
    """
    Batch multi-agent data from dict-of-arrays to single array.

    Used when agent data is stored in a dictionary with string keys (e.g., {"0": obs0, "1": obs1}).

    Args:
        x: Dictionary mapping agent_id (as string) to data arrays
        agent_list: List of agent indices
        num_actors: Total number of actors

    Returns:
        Batched array of shape [num_actors, ...]

    Example:
        >>> done = {"0": jnp.array([False, False]), "1": jnp.array([False, True])}
        >>> batched = batchify_dict(done, agent_list=[0, 1], num_actors=4)
    """
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_numpy(x: dict, agent_list: List, num_actors: int) -> jnp.ndarray:
    """
    Alternative batching function for numpy-indexed multi-agent data.

    This is functionally identical to `batchify` but kept as a separate function
    for backward compatibility with MAPPO code.

    Args:
        x: Dictionary with shape [num_envs, num_agents, ...]
        agent_list: List of agent indices
        num_actors: Total number of actors

    Returns:
        Batched array of shape [num_actors, ...]
    """
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(
    x: jnp.ndarray,
    agent_list: List,
    num_envs: int,
    num_actors: int
) -> Dict[str, jnp.ndarray]:
    """
    Unbatch flattened actor data back to per-agent dictionary format.

    This reverses the batching operation, converting a flat array of actor data
    back into a dictionary mapping agent IDs to their respective data.

    Args:
        x: Batched array of shape [num_actors, ...]
        agent_list: List of agent IDs (can be integers or strings)
        num_envs: Number of parallel environments
        num_actors: Total number of actors (should equal len(agent_list) for parameter sharing)

    Returns:
        Dictionary mapping agent_id -> array of shape [num_envs, ...]

    Example:
        >>> actions = jnp.array([0, 1, 2, 3])  # 4 actors (2 envs × 2 agents)
        >>> unbatched = unbatchify(actions, agent_list=["0", "1"], num_envs=2, num_actors=2)
        >>> # Result: {"0": [0, 2], "1": [1, 3]}
    """
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
