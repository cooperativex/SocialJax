"""Shared dataclasses/NamedTuples used across algorithms."""

import jax.numpy as jnp
from typing import NamedTuple


class Transition(NamedTuple):
    """Standard PPO transition (IPPO / SVO / TRANSFER)."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class MAPPOTransition(NamedTuple):
    """Centralized-critic transition for MAPPO (adds global_done & world_state)."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


class IRATTransition(NamedTuple):
    """IRAT dual-policy transition with separate individual/team heads."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    # Individual policy
    ind_action: jnp.ndarray
    ind_value: jnp.ndarray
    ind_log_prob: jnp.ndarray
    # Team policy
    team_action: jnp.ndarray
    team_value: jnp.ndarray
    team_log_prob: jnp.ndarray
    # Rewards
    ind_reward: jnp.ndarray
    team_reward: jnp.ndarray
    # Observations
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
