"""Replay buffer for off-policy reinforcement learning algorithms.

This module provides ReplayBuffer, a buffer implementation for storing
experiences with random sampling. Used by off-policy algorithms like
VDN, QMIX, DQN, and SAC.

The buffer supports:
- Random mini-batch sampling
- Circular buffer with automatic overflow handling
- Efficient numpy storage
- JAX array conversion for training

Example:
    >>> from socialjax.buffers import ReplayBuffer
    >>> buffer = ReplayBuffer(
    ...     buffer_size=10000,
    ...     obs_shape=(4,),
    ...     action_dim=2,
    ... )
    >>> # Collect experience
    >>> for step in range(1000):
    ...     buffer.add(obs, action, reward, next_obs, done)
    >>> # Sample for training
    >>> batch = buffer.sample(32)
    >>> loss = algorithm.update(batch)
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


class ReplayBuffer(BaseBuffer):
    """Buffer for storing experiences with random sampling.

    This buffer stores transition tuples (s, a, r, s', d) for off-policy
    learning where experiences can be sampled multiple times. It uses a
    circular buffer design to handle overflow gracefully.

    Attributes:
        buffer_size: Maximum number of transitions to store
        obs_shape: Shape of observations
        action_dim: Dimension of action space
        observations: Stored observations (buffer_size, *obs_shape)
        actions: Stored actions (buffer_size,) or (buffer_size, action_dim)
        rewards: Stored rewards (buffer_size,)
        next_observations: Stored next observations (buffer_size, *obs_shape)
        dones: Stored done flags (buffer_size,)

    Example:
        >>> buffer = ReplayBuffer(10000, obs_shape=(4,), action_dim=2)
        >>> buffer.add(obs, action, reward, next_obs, done)
        >>> if buffer.can_sample(32):
        ...     batch = buffer.sample(32)
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        num_envs: int = 1,
        dtype: np.dtype = np.float32,
        handle_timeout_termination: bool = True,
    ):
        """Initialize the replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            obs_shape: Shape of a single observation
            action_dim: Dimension of action space (for discrete: num_actions)
            num_envs: Number of parallel environments (default: 1)
            dtype: Data type for floating point arrays (default: float32)
            handle_timeout_termination: If True, treat timeouts as not done
                for bootstrapping purposes (default: True)

        Raises:
            ValueError: If buffer_size is not positive
        """
        super().__init__(buffer_size, obs_shape, action_dim, num_envs)
        self.dtype = dtype
        self.handle_timeout_termination = handle_timeout_termination

        # Initialize storage arrays
        # Observations: (buffer_size, *obs_shape)
        obs_full_shape = (buffer_size,) + tuple(obs_shape)
        self.observations = np.zeros(obs_full_shape, dtype=dtype)
        self.next_observations = np.zeros(obs_full_shape, dtype=dtype)

        # Actions: (buffer_size,) for discrete, (buffer_size, action_dim) for continuous
        # We use int32 for discrete actions
        self.actions = np.zeros(buffer_size, dtype=np.int32)

        # Rewards: (buffer_size,)
        self.rewards = np.zeros(buffer_size, dtype=dtype)

        # Dones: (buffer_size,)
        self.dones = np.zeros(buffer_size, dtype=dtype)

        # Timeouts (optional, for handling truncation)
        self.timeouts = np.zeros(buffer_size, dtype=bool)

        # For multi-agent support, track which agent the transition belongs to
        # This is optional and can be None for single-agent or shared buffers
        self.agent_ids: Optional[np.ndarray] = None

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        timeout: bool = False,
        agent_id: Optional[int] = None,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Observation array of shape obs_shape
            action: Action (int for discrete, array for continuous)
            reward: Reward value
            next_obs: Next observation array of shape obs_shape
            done: Whether episode ended
            timeout: Whether episode ended due to timeout (truncation)
            agent_id: Optional agent identifier for multi-agent scenarios

        Note:
            When handle_timeout_termination is True and timeout=True,
            the done flag is treated differently for bootstrapping.
        """
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.dones[self._pos] = done
        self.timeouts[self._pos] = timeout

        # Track agent ID if provided
        if agent_id is not None:
            if self.agent_ids is None:
                self.agent_ids = np.zeros(self.buffer_size, dtype=np.int32)
            self.agent_ids[self._pos] = agent_id

        self._pos = (self._pos + 1) % self.buffer_size
        if self._pos == 0:
            self._full = True

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
    ) -> None:
        """Add a batch of transitions to the buffer.

        Args:
            obs: Observation array of shape (batch, *obs_shape)
            actions: Action array of shape (batch,) or (batch, action_dim)
            rewards: Reward array of shape (batch,)
            next_obs: Next observation array of shape (batch, *obs_shape)
            dones: Done array of shape (batch,)
            timeouts: Optional timeout array of shape (batch,)

        Note:
            If the batch is larger than remaining capacity, older data
            will be overwritten in a circular fashion.
        """
        batch_size = obs.shape[0]

        for i in range(batch_size):
            timeout = timeouts[i] if timeouts is not None else False
            self.add(
                obs=obs[i],
                action=actions[i],
                reward=rewards[i],
                next_obs=next_obs[i],
                done=dones[i],
                timeout=timeout,
            )

    def sample(
        self,
        batch_size: int,
        as_jax: bool = False,
        include_timeouts: bool = False,
    ) -> Dict[str, Union[np.ndarray, "jnp.ndarray"]]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            as_jax: If True, convert arrays to JAX arrays (default: False)
            include_timeouts: If True, include timeout information (default: False)

        Returns:
            Dictionary containing:
                - observations: (batch_size, *obs_shape)
                - actions: (batch_size,)
                - rewards: (batch_size,)
                - next_observations: (batch_size, *obs_shape)
                - dones: (batch_size,)
                - timeouts: (batch_size,) [if include_timeouts]
                - agent_ids: (batch_size,) [if tracked]

        Raises:
            BufferEmptyError: If buffer is empty
            InsufficientDataError: If buffer has less data than batch_size
        """
        if self.size == 0:
            raise BufferEmptyError("Cannot sample from empty buffer")

        # Sample with replacement if not enough data
        replace = batch_size > self.size
        if not replace and not self.can_sample(batch_size):
            raise InsufficientDataError(
                f"Buffer has {self.size} samples, need {batch_size}"
            )

        # Sample random indices
        indices = np.random.choice(
            self.size, size=batch_size, replace=replace
        )

        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

        if include_timeouts:
            data["timeouts"] = self.timeouts[indices]

        if self.agent_ids is not None:
            data["agent_ids"] = self.agent_ids[indices]

        # Convert to JAX arrays if requested
        if as_jax and JAX_AVAILABLE:
            data = {k: jnp.array(v) for k, v in data.items()}

        return data

    def get(
        self,
        batch_size: Optional[int] = None,
        as_jax: bool = False,
    ) -> Dict[str, Union[np.ndarray, "jnp.ndarray"]]:
        """Get all data or a batch from the buffer.

        Args:
            batch_size: If provided, sample a batch. If None, return all data.
            as_jax: If True, convert arrays to JAX arrays (default: False)

        Returns:
            Dictionary of arrays

        Raises:
            BufferEmptyError: If buffer is empty
        """
        if batch_size is not None:
            return self.sample(batch_size, as_jax=as_jax)

        if self.size == 0:
            raise BufferEmptyError("Cannot get data from empty buffer")

        indices = np.arange(self.size)
        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

        if self.agent_ids is not None:
            data["agent_ids"] = self.agent_ids[indices]

        if as_jax and JAX_AVAILABLE:
            data = {k: jnp.array(v) for k, v in data.items()}

        return data

    def clear(self) -> None:
        """Reset the buffer to empty state.

        This resets the position counter and full flag, but does not
        zero out the arrays (for efficiency).
        """
        self._pos = 0
        self._full = False
        self.agent_ids = None

    def reset_storage(self) -> None:
        """Fully reset all storage arrays to zeros.

        Unlike clear(), this zeros out all arrays.
        """
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_observations.fill(0)
        self.dones.fill(0)
        self.timeouts.fill(False)
        if self.agent_ids is not None:
            self.agent_ids.fill(0)
        self._pos = 0
        self._full = False

    def get_recent(self, n: int) -> Dict[str, np.ndarray]:
        """Get the n most recent transitions.

        Args:
            n: Number of recent transitions to get

        Returns:
            Dictionary of arrays with the n most recent transitions

        Raises:
            BufferEmptyError: If buffer is empty
            InsufficientDataError: If buffer has less than n samples
        """
        if self.size == 0:
            raise BufferEmptyError("Cannot get recent from empty buffer")
        if not self.can_sample(n):
            raise InsufficientDataError(
                f"Buffer has {self.size} samples, need {n}"
            )

        # Get indices for most recent n items
        if self._full:
            # Buffer is full, last n items ending at pos-1
            end_idx = self._pos
            start_idx = (end_idx - n) % self.buffer_size
            if start_idx < end_idx:
                indices = np.arange(start_idx, end_idx)
            else:
                # Wrapped around
                indices = np.concatenate([
                    np.arange(start_idx, self.buffer_size),
                    np.arange(0, end_idx)
                ])
        else:
            # Buffer not full, last n items from pos-n to pos
            indices = np.arange(max(0, self._pos - n), self._pos)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def sample_with_next_values(
        self,
        batch_size: int,
        gamma: float = 0.99,
        as_jax: bool = False,
    ) -> Dict[str, Union[np.ndarray, "jnp.ndarray"]]:
        """Sample a batch with TD targets computed.

        This is a convenience method for computing TD targets directly
        in the buffer, useful for value-based methods.

        Args:
            batch_size: Number of transitions to sample
            gamma: Discount factor for TD target computation
            as_jax: If True, convert arrays to JAX arrays (default: False)

        Returns:
            Dictionary with additional 'td_target' key containing
            r + gamma * (1 - done) * V(s') or placeholder for V(s')
        """
        data = self.sample(batch_size, as_jax=False)

        # Note: TD target requires value estimates which are algorithm-specific
        # Here we compute the reward + bootstrap part
        # The caller should add the value estimate for next_obs
        bootstrap = (1 - data["dones"]).astype(self.dtype)
        if self.handle_timeout_termination:
            # For timeouts, we should still bootstrap
            bootstrap = bootstrap * (1 - data.get("timeouts", np.zeros(batch_size)))

        data["bootstrap"] = bootstrap
        data["gamma"] = gamma

        if as_jax and JAX_AVAILABLE:
            data = {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in data.items()}

        return data

    def memory_size(self) -> int:
        """Calculate approximate memory usage in bytes.

        Returns:
            Approximate memory usage in bytes
        """
        obs_size = self.observations.nbytes
        next_obs_size = self.next_observations.nbytes
        actions_size = self.actions.nbytes
        rewards_size = self.rewards.nbytes
        dones_size = self.dones.nbytes
        timeouts_size = self.timeouts.nbytes

        total = (
            obs_size + next_obs_size + actions_size + rewards_size +
            dones_size + timeouts_size
        )

        if self.agent_ids is not None:
            total += self.agent_ids.nbytes

        return total


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized experience replay (PER).

    This extends ReplayBuffer to support prioritized sampling where
    transitions are sampled with probability proportional to their
    TD error (or other priority measure).

    Uses proportional prioritization: p_i = |δ_i| + ε

    Attributes:
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        epsilon: Small constant to ensure non-zero priorities
        beta: Importance sampling exponent (annealed during training)

    Example:
        >>> buffer = PrioritizedReplayBuffer(10000, obs_shape=(4,), action_dim=2)
        >>> buffer.add(obs, action, reward, next_obs, done)
        >>> batch = buffer.sample(32)
        >>> # After computing TD errors:
        >>> buffer.update_priorities(indices, td_errors)
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        """Initialize the prioritized replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            obs_shape: Shape of a single observation
            action_dim: Dimension of action space
            alpha: Priority exponent (default: 0.6)
            beta: Importance sampling exponent (default: 0.4)
            epsilon: Small constant for priorities (default: 1e-6)
            **kwargs: Additional arguments passed to ReplayBuffer
        """
        super().__init__(buffer_size, obs_shape, action_dim, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # Initialize priorities with max priority for new transitions
        self._priorities = np.zeros(buffer_size, dtype=np.float32)
        self._max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        timeout: bool = False,
        priority: Optional[float] = None,
    ) -> None:
        """Add transition with optional priority.

        Args:
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done flag
            timeout: Timeout flag
            priority: Optional priority value (defaults to max priority)
        """
        # Set priority for new transition
        if priority is None:
            priority = self._max_priority

        self._priorities[self._pos] = priority

        super().add(obs, action, reward, next_obs, done, timeout)

    def sample(
        self,
        batch_size: int,
        as_jax: bool = False,
        include_weights: bool = True,
    ) -> Dict[str, Union[np.ndarray, "jnp.ndarray"]]:
        """Sample a prioritized batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            as_jax: If True, convert arrays to JAX arrays
            include_weights: If True, include importance sampling weights

        Returns:
            Dictionary with sampled data plus:
                - indices: Sampled indices (for updating priorities)
                - weights: Importance sampling weights (if include_weights)
        """
        if self.size == 0:
            raise BufferEmptyError("Cannot sample from empty buffer")

        # Compute sampling probabilities
        priorities = self._priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
            "indices": indices,
        }

        # Compute importance sampling weights
        if include_weights:
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / weights.max()  # Normalize
            data["weights"] = weights.astype(self.dtype)

        if as_jax and JAX_AVAILABLE:
            data = {k: jnp.array(v) for k, v in data.items()}

        return data

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically |TD_error|)
        """
        priorities = np.abs(priorities) + self.epsilon
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, priorities.max())

    def clear(self) -> None:
        """Reset buffer to empty state."""
        super().clear()
        self._priorities.fill(0)
        self._max_priority = 1.0

    def reset_storage(self) -> None:
        """Fully reset all storage arrays."""
        super().reset_storage()
        self._priorities.fill(0)
        self._max_priority = 1.0
