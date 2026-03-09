"""Rollout buffer for on-policy reinforcement learning algorithms.

This module provides RolloutBuffer, a buffer implementation for storing
complete trajectories collected during rollouts. Used by on-policy algorithms
like IPPO, MAPPO, and SVO.

The buffer stores:
- Observations
- Actions
- Rewards
- Done flags
- Log probabilities (for policy gradient methods)
- Value estimates (for GAE computation)

Example:
    >>> from socialjax.buffers import RolloutBuffer
    >>> buffer = RolloutBuffer(
    ...     buffer_size=128,
    ...     num_envs=8,
    ...     obs_shape=(15, 15, 3),
    ...     action_dim=8,
    ... )
    >>> # During rollout
    >>> for step in range(128):
    ...     buffer.add(obs, action, reward, done, log_prob, value)
    >>> # After rollout, get all data for update
    >>> batch = buffer.get()
    >>> loss = algorithm.update(batch)
    >>> buffer.clear()  # Reset for next rollout
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from socialjax.buffers.base_buffer import (
    BaseBuffer,
    BufferEmptyError,
    InsufficientDataError,
)


class RolloutBuffer(BaseBuffer):
    """Buffer for storing rollouts in on-policy algorithms.

    This buffer stores complete trajectories for on-policy learning where
    data is collected, used for an update, then discarded. It supports:
    - Multiple parallel environments
    - Efficient numpy array storage
    - JAX array conversion for training

    The buffer uses a circular design with a fixed size. After filling,
    calling get() returns all stored data, then clear() resets for the
    next rollout.

    Attributes:
        buffer_size: Maximum steps to store per environment
        num_envs: Number of parallel environments
        obs_shape: Shape of observations
        action_dim: Dimension of action space
        observations: Stored observation array (buffer_size, num_envs, *obs_shape)
        actions: Stored action array (buffer_size, num_envs)
        rewards: Stored reward array (buffer_size, num_envs)
        dones: Stored done flags (buffer_size, num_envs)
        log_probs: Stored log probabilities (buffer_size, num_envs)
        values: Stored value estimates (buffer_size, num_envs)
        advantages: Computed advantages (buffer_size, num_envs) - set externally
        returns: Computed returns (buffer_size, num_envs) - set externally
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        dtype: np.dtype = np.float32,
    ):
        """Initialize the rollout buffer.

        Args:
            buffer_size: Number of steps to store per environment
            num_envs: Number of parallel environments
            obs_shape: Shape of a single observation (e.g., (15, 15, 3) for CNN)
            action_dim: Dimension of action space (for discrete: num_actions)
            dtype: Data type for floating point arrays (default: float32)

        Raises:
            ValueError: If buffer_size or num_envs is not positive
        """
        super().__init__(buffer_size, obs_shape, action_dim, num_envs)
        self.dtype = dtype

        # Initialize storage arrays
        # Observations: (buffer_size, num_envs, *obs_shape)
        obs_full_shape = (buffer_size, num_envs) + tuple(obs_shape)
        self.observations = np.zeros(obs_full_shape, dtype=dtype)

        # Actions: (buffer_size, num_envs) - discrete actions
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int32)

        # Rewards: (buffer_size, num_envs)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=dtype)

        # Dones: (buffer_size, num_envs)
        self.dones = np.zeros((buffer_size, num_envs), dtype=dtype)

        # Log probabilities: (buffer_size, num_envs)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=dtype)

        # Values: (buffer_size, num_envs)
        self.values = np.zeros((buffer_size, num_envs), dtype=dtype)

        # Advantages and returns (computed externally, e.g., by GAE)
        self.advantages = np.zeros((buffer_size, num_envs), dtype=dtype)
        self.returns = np.zeros((buffer_size, num_envs), dtype=dtype)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Add a single step to the buffer.

        Args:
            obs: Observation array of shape (num_envs, *obs_shape)
            action: Action array of shape (num_envs,)
            reward: Reward array of shape (num_envs,)
            done: Done flag array of shape (num_envs,)
            log_prob: Log probability array of shape (num_envs,)
            value: Value estimate array of shape (num_envs,)

        Note:
            Automatically wraps around when buffer is full. Check self.full
            after adding to know if data has wrapped.
        """
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.dones[self._pos] = done
        self.log_probs[self._pos] = log_prob
        self.values[self._pos] = value

        self._pos = (self._pos + 1) % self.buffer_size
        if self._pos == 0:
            self._full = True

    def get(
        self,
        as_jax: bool = False,
        include_advantages: bool = True,
    ) -> Dict[str, Union[np.ndarray, "jnp.ndarray"]]:
        """Get all stored data from the buffer.

        Args:
            as_jax: If True, convert arrays to JAX arrays (default: False)
            include_advantages: If True, include advantages and returns (default: True)

        Returns:
            Dictionary containing:
                - observations: (n_steps, num_envs, *obs_shape)
                - actions: (n_steps, num_envs)
                - rewards: (n_steps, num_envs)
                - dones: (n_steps, num_envs)
                - log_probs: (n_steps, num_envs)
                - values: (n_steps, num_envs)
                - advantages: (n_steps, num_envs) [if include_advantages]
                - returns: (n_steps, num_envs) [if include_advantages]

        Raises:
            BufferEmptyError: If buffer is empty
        """
        if self.size == 0:
            raise BufferEmptyError("Cannot get data from empty buffer")

        # Get indices for stored data
        indices = np.arange(self._pos if not self._full else self.buffer_size)

        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "log_probs": self.log_probs[indices],
            "values": self.values[indices],
        }

        if include_advantages:
            data["advantages"] = self.advantages[indices]
            data["returns"] = self.returns[indices]

        # Convert to JAX arrays if requested
        if as_jax and JAX_AVAILABLE:
            data = {k: jnp.array(v) for k, v in data.items()}

        return data

    def get_batch(
        self,
        batch_size: int,
        as_jax: bool = False,
    ) -> Dict[str, Union[np.ndarray, "jnp.ndarray"]]:
        """Get a batch of data, flattening across steps and envs.

        This method is useful for mini-batch updates where you want to
        sample from all collected experience.

        Args:
            batch_size: Number of samples to return
            as_jax: If True, convert arrays to JAX arrays (default: False)

        Returns:
            Dictionary with flattened arrays of shape (batch_size, ...)

        Raises:
            BufferEmptyError: If buffer is empty
            InsufficientDataError: If buffer has less data than batch_size
        """
        if self.size == 0:
            raise BufferEmptyError("Cannot get batch from empty buffer")
        if not self.can_sample(batch_size):
            raise InsufficientDataError(
                f"Buffer has {self.size} samples, need {batch_size}"
            )

        # Get all data
        data = self.get(as_jax=False, include_advantages=True)

        # Flatten steps and envs: (steps, envs, ...) -> (steps*envs, ...)
        n_samples = self.size * self.num_envs
        flat_data = {}
        for key, arr in data.items():
            if key == "observations":
                # (steps, envs, *obs_shape) -> (steps*envs, *obs_shape)
                flat_data[key] = arr.reshape(n_samples, *self.obs_shape)
            else:
                # (steps, envs) -> (steps*envs,)
                flat_data[key] = arr.reshape(n_samples)

        # Sample random indices
        indices = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)

        batch = {k: v[indices] for k, v in flat_data.items()}

        if as_jax and JAX_AVAILABLE:
            batch = {k: jnp.array(v) for k, v in batch.items()}

        return batch

    def clear(self) -> None:
        """Reset the buffer to empty state.

        This resets the position counter and full flag, but does not
        zero out the arrays (for efficiency). New data will overwrite
        old data.
        """
        self._pos = 0
        self._full = False

    def reset_storage(self) -> None:
        """Fully reset all storage arrays to zeros.

        Unlike clear(), this zeros out all arrays. Use this if you
        need to ensure no old data remains in memory.
        """
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.log_probs.fill(0)
        self.values.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)
        self._pos = 0
        self._full = False

    def set_advantages(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        """Set computed advantages and returns (e.g., from GAE).

        Args:
            advantages: Advantage estimates of shape (n_steps, num_envs)
            returns: Return targets of shape (n_steps, num_envs)

        Raises:
            ValueError: If shapes don't match buffer size
        """
        n_steps = advantages.shape[0]
        if n_steps > self.buffer_size:
            raise ValueError(
                f"Advantages size {n_steps} exceeds buffer size {self.buffer_size}"
            )
        if advantages.shape[1] != self.num_envs:
            raise ValueError(
                f"Advantages envs {advantages.shape[1]} != buffer envs {self.num_envs}"
            )

        self.advantages[:n_steps] = advantages
        self.returns[:n_steps] = returns

    def get_flattened(
        self,
        as_jax: bool = False,
    ) -> Dict[str, Union[np.ndarray, "jnp.ndarray"]]:
        """Get all data flattened across steps and environments.

        This is useful for algorithms that need a single batch of
        all experience rather than separated by step.

        Args:
            as_jax: If True, convert arrays to JAX arrays (default: False)

        Returns:
            Dictionary with flattened arrays:
                - observations: (n_steps * num_envs, *obs_shape)
                - actions: (n_steps * num_envs,)
                - etc.
        """
        data = self.get(as_jax=False, include_advantages=True)

        n_samples = self.size * self.num_envs
        flat_data = {}

        for key, arr in data.items():
            if key == "observations":
                flat_data[key] = arr.reshape(n_samples, *self.obs_shape)
            else:
                flat_data[key] = arr.reshape(n_samples)

        if as_jax and JAX_AVAILABLE:
            flat_data = {k: jnp.array(v) for k, v in flat_data.items()}

        return flat_data

    def memory_size(self) -> int:
        """Calculate approximate memory usage in bytes.

        Returns:
            Approximate memory usage in bytes
        """
        # Each array's memory
        obs_size = self.observations.nbytes
        actions_size = self.actions.nbytes
        rewards_size = self.rewards.nbytes
        dones_size = self.dones.nbytes
        log_probs_size = self.log_probs.nbytes
        values_size = self.values.nbytes
        advantages_size = self.advantages.nbytes
        returns_size = self.returns.nbytes

        total = (
            obs_size + actions_size + rewards_size + dones_size +
            log_probs_size + values_size + advantages_size + returns_size
        )
        return total
