"""Base buffer interface for experience storage.

This module defines the abstract base class for all buffer types used in
reinforcement learning algorithms. Buffers store experience tuples for
training neural networks.

The buffer system supports two main paradigms:
- On-policy (RolloutBuffer): Stores complete trajectories, used once then cleared
- Off-policy (ReplayBuffer): Stores experiences for random sampling, persistent

Example:
    >>> from socialjax.buffers import RolloutBuffer, ReplayBuffer
    >>> # On-policy training
    >>> rollout = RolloutBuffer(buffer_size=128, num_envs=8, obs_shape=(4,), action_dim=2)
    >>> rollout.add(obs, action, reward, done, log_prob, value)
    >>> batch = rollout.get()
    >>> rollout.clear()
    >>> # Off-policy training
    >>> replay = ReplayBuffer(buffer_size=10000, obs_shape=(4,), action_dim=2)
    >>> replay.add(obs, action, reward, next_obs, done)
    >>> batch = replay.sample(batch_size=32)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np


class BaseBuffer(ABC):
    """Abstract base class for experience buffers.

    This class defines the interface that all buffer implementations must follow.
    Buffers store experience data for reinforcement learning algorithms.

    Subclasses must implement:
        - add(): Add experience to buffer
        - get(): Retrieve data from buffer
        - clear(): Reset buffer state
        - size property: Current number of stored items

    Attributes:
        buffer_size: Maximum capacity of the buffer
        obs_shape: Shape of observation arrays
        action_dim: Dimension of action space
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        num_envs: int = 1,
    ):
        """Initialize the base buffer.

        Args:
            buffer_size: Maximum number of steps to store
            obs_shape: Shape of a single observation
            action_dim: Dimension of the action space
            num_envs: Number of parallel environments (default: 1)

        Raises:
            ValueError: If buffer_size <= 0 or num_envs <= 0
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")

        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.num_envs = num_envs
        self._pos = 0
        self._full = False

    @property
    def size(self) -> int:
        """Get current number of stored items.

        Returns:
            Number of items currently stored in buffer
        """
        return self._pos if not self._full else self.buffer_size

    @property
    def full(self) -> bool:
        """Check if buffer is full.

        Returns:
            True if buffer has wrapped around at least once
        """
        return self._full

    @property
    def pos(self) -> int:
        """Get current write position.

        Returns:
            Current index where next item will be written
        """
        return self._pos

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add experience to the buffer.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Get data from the buffer.

        This method must be implemented by subclasses.

        Returns:
            Dictionary of arrays containing stored data
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Reset the buffer to empty state.

        This method must be implemented by subclasses.
        """
        pass

    def can_sample(self, n_samples: int) -> bool:
        """Check if buffer has enough data to sample.

        Args:
            n_samples: Number of samples requested

        Returns:
            True if buffer has at least n_samples items
        """
        return self.size >= n_samples

    def __len__(self) -> int:
        """Get buffer size using len().

        Returns:
            Current number of stored items
        """
        return self.size

    def __repr__(self) -> str:
        """Get string representation of buffer.

        Returns:
            String with buffer type and size information
        """
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}/{self.buffer_size}, "
            f"obs_shape={self.obs_shape}, "
            f"action_dim={self.action_dim})"
        )


class BufferError(Exception):
    """Exception raised for buffer-related errors.

    This exception is raised when buffer operations fail, such as:
    - Sampling from empty buffer
    - Buffer capacity exceeded
    - Invalid data shapes
    """
    pass


class BufferEmptyError(BufferError):
    """Exception raised when trying to get data from empty buffer."""
    pass


class BufferFullError(BufferError):
    """Exception raised when buffer is full and doesn't allow overflow."""
    pass


class InsufficientDataError(BufferError):
    """Exception raised when buffer doesn't have enough data for operation."""
    pass
