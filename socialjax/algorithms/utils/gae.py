"""Generalized Advantage Estimation (GAE) utilities.

This module provides JAX-JIT compatible implementations of GAE computation
for policy gradient algorithms like PPO, IPPO, and MAPPO.

Reference:
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    (Schulman et al., 2016)
"""

from typing import Tuple, NamedTuple, Optional, Protocol, Any
import jax
import jax.numpy as jnp


class TransitionProtocol(Protocol):
    """Protocol for transition objects used in GAE computation."""
    done: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray


class GAETransition(NamedTuple):
    """Standard transition format for GAE computation.

    Attributes:
        done: Episode termination flags (shape: [T] or [T, B])
        value: Value estimates (shape: [T] or [T, B])
        reward: Rewards received (shape: [T] or [T, B])
    """
    done: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray


@jax.jit
def compute_gae(
    traj_batch: TransitionProtocol,
    last_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE computes advantages using a biased, truncated estimator of the
    advantage function that balances bias and variance through the lambda
    parameter.

    The GAE formula is:
        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Args:
        traj_batch: Batch of transitions with done, value, and reward fields.
            Can be a NamedTuple or any object with these attributes.
        last_value: Value estimate for the final observation (after trajectory).
            Shape should match the value shape in traj_batch.
        gamma: Discount factor for future rewards. Default: 0.99
        gae_lambda: GAE lambda parameter controlling bias-variance tradeoff.
            lambda=0 gives low-variance, high-bias (TD(0)).
            lambda=1 gives high-variance, low-bias (Monte Carlo).
            Default: 0.95

    Returns:
        Tuple of (advantages, value_targets) where:
        - advantages: GAE advantage estimates, same shape as traj_batch.value
        - value_targets: Value function targets (advantages + values)

    Example:
        >>> from socialjax.algorithms.utils.gae import compute_gae, GAETransition
        >>> import jax.numpy as jnp
        >>> # Create sample trajectory (T=10 timesteps)
        >>> dones = jnp.zeros(10)
        >>> dones = dones.at[5].set(1.0)  # Episode ends at step 5
        >>> values = jnp.ones(10)
        >>> rewards = jnp.ones(10) * 0.1
        >>> traj = GAETransition(done=dones, value=values, reward=rewards)
        >>> last_value = jnp.array(1.0)
        >>> advantages, targets = compute_gae(traj, last_value, gamma=0.99, gae_lambda=0.95)
    """
    def _get_advantages(
        gae_and_next_value: Tuple[jnp.ndarray, jnp.ndarray],
        transition: TransitionProtocol,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Scan function for computing GAE recursively."""
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.value, transition.reward

        # TD residual: delta = r + gamma * V(s') - V(s)
        delta = reward + gamma * next_value * (1 - done) - value

        # GAE recursion: A = delta + gamma * lambda * (1 - done) * A
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return (gae, value), gae

    # Scan backwards through trajectory to compute advantages
    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        traj_batch,
        reverse=True,
        unroll=16,  # Unroll for performance
    )

    # Value targets = advantages + values
    return advantages, advantages + traj_batch.value


@jax.jit
def compute_gae_batched(
    dones: jnp.ndarray,
    values: jnp.ndarray,
    rewards: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE from separate arrays (batched version).

    This is a convenience function that accepts arrays directly instead of
    a transition object.

    Args:
        dones: Episode termination flags. Shape: [T] or [T, B]
        values: Value estimates. Shape: [T] or [T, B]
        rewards: Rewards received. Shape: [T] or [T, B]
        last_value: Value for final observation. Shape: [] or [B]
        gamma: Discount factor. Default: 0.99
        gae_lambda: GAE lambda parameter. Default: 0.95

    Returns:
        Tuple of (advantages, value_targets)

    Example:
        >>> import jax.numpy as jnp
        >>> T, B = 100, 8  # 100 timesteps, batch size 8
        >>> dones = jnp.zeros((T, B))
        >>> values = jnp.ones((T, B))
        >>> rewards = jnp.ones((T, B)) * 0.1
        >>> last_values = jnp.ones(B)
        >>> advantages, targets = compute_gae_batched(
        ...     dones, values, rewards, last_values, gamma=0.99, gae_lambda=0.95
        ... )
    """
    traj_batch = GAETransition(done=dones, value=values, reward=rewards)
    return compute_gae(traj_batch, last_value, gamma, gae_lambda)


@jax.jit
def normalize_advantages(advantages: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize advantages to have zero mean and unit variance.

    This is commonly done before using advantages in the PPO loss to
    stabilize training.

    Args:
        advantages: Advantage estimates. Shape: any
        eps: Small constant for numerical stability. Default: 1e-8

    Returns:
        Normalized advantages with zero mean and unit variance.

    Example:
        >>> import jax.numpy as jnp
        >>> advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized = normalize_advantages(advantages)
        >>> print(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


def compute_returns(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
) -> jnp.ndarray:
    """Compute discounted returns (Monte Carlo targets).

    This computes the full discounted return G_t = sum_{k=0}^{inf} gamma^k * r_{t+k}
    which is used as an alternative to GAE for value function training.

    Args:
        rewards: Rewards received. Shape: [T] or [T, B]
        dones: Episode termination flags. Shape: [T] or [T, B]
        gamma: Discount factor. Default: 0.99

    Returns:
        Discounted returns with same shape as rewards.

    Note:
        This is not JIT-compiled by default because it's typically used
        in non-performance-critical paths. Wrap in jax.jit if needed.
    """
    def _scan_fn(carry, transition):
        returns, = carry
        reward, done = transition
        returns = reward + gamma * returns * (1 - done)
        return (returns,), returns

    # Initialize returns to zero
    init_returns = jnp.zeros_like(rewards[0])

    # Scan backwards through rewards
    _, returns = jax.lax.scan(
        _scan_fn,
        (init_returns,),
        (rewards, dones),
        reverse=True,
    )

    return returns


# Convenience exports
__all__ = [
    "compute_gae",
    "compute_gae_batched",
    "normalize_advantages",
    "compute_returns",
    "GAETransition",
    "TransitionProtocol",
]
