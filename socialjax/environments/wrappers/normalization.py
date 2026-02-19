"""Normalization wrapper for SocialJax environments.

This module provides wrappers that normalize observations and rewards
using running statistics (mean and standard deviation). This helps
stabilize training by keeping values in a consistent range.
"""

from typing import Dict, Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from functools import partial
from flax import struct
import numpy as np

from socialjax.environments.multi_agent_env import MultiAgentEnv, State
from socialjax.environments.wrappers.base_wrapper import BaseWrapper


@struct.dataclass
class RunningMeanStd:
    """Running statistics for normalization.

    Uses Welford's online algorithm to compute running mean and variance.

    Attributes:
        mean: Running mean
        var: Running variance
        count: Number of samples seen
    """
    mean: chex.Array
    var: chex.Array
    count: chex.Array

    @classmethod
    def create(cls, shape: Tuple[int, ...]) -> "RunningMeanStd":
        """Create a new RunningMeanStd with zeros.

        Args:
            shape: Shape of the statistics

        Returns:
            New RunningMeanStd instance
        """
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=jnp.zeros((), dtype=jnp.float32),
        )


def update_running_stats(
    stats: RunningMeanStd,
    batch: chex.Array,
    epsilon: float = 1e-8,
) -> RunningMeanStd:
    """Update running statistics with a new batch of data.

    Uses a batch version of Welford's algorithm for numerical stability.

    Args:
        stats: Current running statistics
        batch: New batch of data to incorporate
        epsilon: Small value to prevent division by zero

    Returns:
        Updated running statistics
    """
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    delta = batch_mean - stats.mean
    total_count = stats.count + batch_count

    new_mean = stats.mean + delta * batch_count / total_count
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * stats.count * batch_count / total_count
    new_var = M2 / jnp.maximum(total_count, epsilon)
    new_count = total_count

    return RunningMeanStd(
        mean=new_mean,
        var=jnp.maximum(new_var, epsilon),
        count=new_count,
    )


def normalize(
    x: chex.Array,
    stats: RunningMeanStd,
    clip: float = 10.0,
    epsilon: float = 1e-8,
) -> chex.Array:
    """Normalize values using running statistics.

    Args:
        x: Values to normalize
        stats: Running statistics
        clip: Maximum absolute value for clipping
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized values
    """
    normalized = (x - stats.mean) / jnp.sqrt(stats.var + epsilon)
    return jnp.clip(normalized, -clip, clip)


@struct.dataclass
class NormalizationState(State):
    """State for NormalizationWrapper.

    Attributes:
        done: Whether episode is done
        step: Current step count
        env_state: Wrapped environment state
        obs_stats: Running statistics for observations
        reward_stats: Running statistics for rewards
    """
    done: chex.Array = None  # type: ignore
    step: int = 0
    env_state: State = None  # type: ignore
    obs_stats: RunningMeanStd = None  # type: ignore
    reward_stats: RunningMeanStd = None  # type: ignore


class NormalizationWrapper(BaseWrapper):
    """Wrapper that normalizes observations and rewards.

    This wrapper maintains running statistics (mean and variance) for
    observations and rewards, and normalizes them using these statistics.
    This helps stabilize training by keeping values in a consistent range.

    Normalization is done per-agent, with shared statistics across agents
    by default. The wrapper updates statistics during step() calls.

    Example:
        >>> from socialjax import make
        >>> from socialjax.environments.wrappers import NormalizationWrapper
        >>> env = make('coin_game', num_agents=5)
        >>> env = NormalizationWrapper(env, normalize_obs=True, normalize_reward=True)
        >>> obs, state = env.reset(key)
        >>> # obs are now normalized

    Attributes:
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        clip_obs: Maximum absolute value for normalized observations
        clip_reward: Maximum absolute value for normalized rewards
        gamma: Discount factor for reward normalization
        epsilon: Small value for numerical stability
        _obs_shape: Shape of observations
        _training: Whether the wrapper is in training mode
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """Initialize the normalization wrapper.

        Args:
            env: Environment to wrap
            normalize_obs: Whether to normalize observations
            normalize_reward: Whether to normalize rewards
            clip_obs: Maximum absolute value for normalized observations
            clip_reward: Maximum absolute value for normalized rewards
            gamma: Discount factor for reward normalization
            epsilon: Small value for numerical stability
        """
        super().__init__(env)
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self._training = True

        # Get observation shape from the first agent
        first_agent = f"agent_{0}"
        if first_agent in env.observation_spaces:
            obs_space = env.observation_spaces[first_agent]
            if hasattr(obs_space, 'shape'):
                self._obs_shape = obs_space.shape
            else:
                self._obs_shape = ()
        else:
            self._obs_shape = ()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], NormalizationState]:
        """Reset the environment and initialize normalization statistics.

        Args:
            key: PRNG key

        Returns:
            Tuple of (normalized observations, state)
        """
        obs, env_state = self.env.reset(key)

        # Initialize statistics
        obs_stats = RunningMeanStd.create(self._obs_shape)
        reward_stats = RunningMeanStd.create(())

        state = NormalizationState(
            done=jnp.array(False),
            step=0,
            env_state=env_state,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
        )

        # Normalize initial observations
        if self.normalize_obs:
            obs = self._normalize_obs_batch(obs, obs_stats)

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: NormalizationState,
        actions: Dict[str, chex.Array],
        timestep: int = 0,
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], NormalizationState, Dict[str, float], Dict[str, bool], Dict]:
        """Step the environment with normalization.

        Args:
            key: PRNG key
            state: Current normalization state
            actions: Actions for each agent
            timestep: Current timestep
            reset_state: Optional state to reset to

        Returns:
            Tuple of (normalized obs, state, normalized rewards, dones, infos)
        """
        # Step the wrapped environment
        obs, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions, timestep, reset_state
        )

        # Update and apply observation normalization
        if self.normalize_obs and self._training:
            # Stack observations for batch update
            obs_batch = jnp.stack(list(obs.values()), axis=0)
            new_obs_stats = update_running_stats(state.obs_stats, obs_batch, self.epsilon)
        else:
            new_obs_stats = state.obs_stats

        # Update and apply reward normalization
        if self.normalize_reward and self._training:
            # Stack rewards for batch update
            reward_batch = jnp.stack(list(rewards.values()), axis=0)
            new_reward_stats = update_running_stats(state.reward_stats, reward_batch, self.epsilon)
        else:
            new_reward_stats = state.reward_stats

        # Create new state
        new_state = NormalizationState(
            done=dones.get("__all__", jnp.array(False)),
            step=state.step + 1,
            env_state=env_state,
            obs_stats=new_obs_stats,
            reward_stats=new_reward_stats,
        )

        # Normalize observations
        if self.normalize_obs:
            obs = self._normalize_obs_batch(obs, new_obs_stats)

        # Normalize rewards
        if self.normalize_reward:
            rewards = self._normalize_reward_batch(rewards, new_reward_stats)

        return obs, new_state, rewards, dones, infos

    def _normalize_obs_batch(
        self,
        obs: Dict[str, chex.Array],
        stats: RunningMeanStd,
    ) -> Dict[str, chex.Array]:
        """Normalize a batch of observations.

        Args:
            obs: Dictionary of observations
            stats: Running statistics

        Returns:
            Dictionary of normalized observations
        """
        return {
            agent: normalize(o, stats, self.clip_obs, self.epsilon)
            for agent, o in obs.items()
        }

    def _normalize_reward_batch(
        self,
        rewards: Dict[str, chex.Array],
        stats: RunningMeanStd,
    ) -> Dict[str, chex.Array]:
        """Normalize a batch of rewards.

        Args:
            rewards: Dictionary of rewards
            stats: Running statistics

        Returns:
            Dictionary of normalized rewards
        """
        return {
            agent: normalize(r, stats, self.clip_reward, self.epsilon)
            for agent, r in rewards.items()
        }

    def get_obs(self, state: NormalizationState) -> Dict[str, chex.Array]:
        """Get observations from state.

        Args:
            state: Normalization state

        Returns:
            Dictionary of normalized observations
        """
        obs = self.env.get_obs(state.env_state)
        if self.normalize_obs:
            obs = self._normalize_obs_batch(obs, state.obs_stats)
        return obs

    def set_training(self, training: bool) -> None:
        """Set whether the wrapper should update statistics.

        Args:
            training: If True, statistics are updated during steps
        """
        self._training = training

    def get_stats(self, state: NormalizationState) -> Dict[str, RunningMeanStd]:
        """Get the current normalization statistics.

        Args:
            state: Current wrapper state

        Returns:
            Dictionary with 'obs' and 'reward' statistics
        """
        return {
            'obs': state.obs_stats,
            'reward': state.reward_stats,
        }
