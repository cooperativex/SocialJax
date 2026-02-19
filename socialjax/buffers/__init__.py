"""SocialJax buffers module for experience storage.

This module provides buffer implementations for reinforcement learning:
- RolloutBuffer: On-policy buffer for storing complete trajectories
- ReplayBuffer: Off-policy buffer with random sampling
- PrioritizedReplayBuffer: Replay buffer with prioritized sampling

Example:
    >>> from socialjax.buffers import RolloutBuffer, ReplayBuffer
    >>>
    >>> # On-policy training (IPPO, MAPPO, SVO)
    >>> rollout = RolloutBuffer(
    ...     buffer_size=128,
    ...     num_envs=8,
    ...     obs_shape=(15, 15, 3),
    ...     action_dim=8,
    ... )
    >>> for step in range(128):
    ...     rollout.add(obs, action, reward, done, log_prob, value)
    >>> batch = rollout.get()
    >>> rollout.clear()
    >>>
    >>> # Off-policy training (VDN, DQN)
    >>> replay = ReplayBuffer(
    ...     buffer_size=10000,
    ...     obs_shape=(4,),
    ...     action_dim=2,
    ... )
    >>> for step in range(1000):
    ...     replay.add(obs, action, reward, next_obs, done)
    >>> batch = replay.sample(32)
"""

from socialjax.buffers.base_buffer import (
    BaseBuffer,
    BufferError,
    BufferEmptyError,
    BufferFullError,
    InsufficientDataError,
)

from socialjax.buffers.rollout_buffer import RolloutBuffer

from socialjax.buffers.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
)

__all__ = [
    # Base classes
    "BaseBuffer",
    # Exceptions
    "BufferError",
    "BufferEmptyError",
    "BufferFullError",
    "InsufficientDataError",
    # Buffer implementations
    "RolloutBuffer",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
]
