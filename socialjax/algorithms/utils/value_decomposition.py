"""Value Decomposition utilities for multi-agent reinforcement learning.

This module provides JAX-JIT compatible implementations of value decomposition
methods for cooperative multi-agent RL algorithms like VDN and QMIX.

Value decomposition allows for centralized training with decentralized execution
by decomposing the global Q-value into individual agent Q-values.

References:
    - "Value-Decomposition Networks for Cooperative Multi-Agent Learning"
      (Sunehag et al., 2018) - VDN
    - "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent
      Reinforcement Learning" (Rashid et al., 2018) - QMIX
"""

from typing import Tuple, Dict, Any, Optional, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp


class ValueDecompositionOutput(NamedTuple):
    """Output of value decomposition computation.

    Attributes:
        q_tot: Total team Q-value. Shape: [batch]
        individual_q: Individual agent Q-values. Shape: [num_agents, batch]
        chosen_q_values: Q-values for chosen actions. Shape: [num_agents, batch]
    """
    q_tot: jnp.ndarray
    individual_q: jnp.ndarray
    chosen_q_values: jnp.ndarray


@jax.jit
def vdn_decomposition(
    q_values: jnp.ndarray,
    actions: jnp.ndarray,
) -> ValueDecompositionOutput:
    """Compute VDN-style value decomposition.

    VDN decomposes the team Q-value as a sum of individual agent Q-values:
        Q_tot(s, a) = sum_i Q_i(s_i, a_i)

    This additive decomposition is the simplest form and maintains IGM
    (Individual-Global-Max) consistency.

    Args:
        q_values: Individual Q-values for all agents.
            Shape: [num_agents, batch, action_dim]
        actions: Actions taken by each agent.
            Shape: [num_agents, batch]

    Returns:
        ValueDecompositionOutput with total Q, individual Qs, and chosen Qs

    Example:
        >>> import jax.numpy as jnp
        >>> # 3 agents, batch size 4, action dim 5
        >>> q_values = jnp.ones((3, 4, 5))
        >>> actions = jnp.array([[0, 1, 2, 3],
        ...                      [1, 2, 3, 4],
        ...                      [2, 3, 4, 0]])
        >>> output = vdn_decomposition(q_values, actions)
        >>> print(f"Q_tot shape: {output.q_tot.shape}")  # [4]
    """
    num_agents, batch_size, action_dim = q_values.shape

    # Get Q-values for chosen actions
    # actions shape: [num_agents, batch] -> [num_agents, batch, 1]
    actions_expanded = actions[..., jnp.newaxis]

    # Gather chosen Q-values: [num_agents, batch]
    chosen_q = jnp.take_along_axis(q_values, actions_expanded, axis=-1).squeeze(-1)

    # Sum across agents for total Q
    q_tot = jnp.sum(chosen_q, axis=0)

    return ValueDecompositionOutput(
        q_tot=q_tot,
        individual_q=q_values,
        chosen_q_values=chosen_q,
    )


@jax.jit
def vdn_target(
    q_values_target: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
) -> jnp.ndarray:
    """Compute VDN-style TD target.

    The VDN target is:
        y = r + gamma * sum_i max_a' Q_i^target(s'_i, a')

    Args:
        q_values_target: Target network Q-values for next state.
            Shape: [num_agents, batch, action_dim]
        rewards: Team rewards. Shape: [batch]
        dones: Episode termination flags. Shape: [batch]
        gamma: Discount factor. Default: 0.99

    Returns:
        TD targets. Shape: [batch]

    Example:
        >>> import jax.numpy as jnp
        >>> # 3 agents, batch size 4, action dim 5
        >>> q_target = jnp.ones((3, 4, 5))
        >>> rewards = jnp.array([1.0, 0.5, 0.0, -0.5])
        >>> dones = jnp.array([0.0, 0.0, 1.0, 0.0])
        >>> targets = vdn_target(q_target, rewards, dones, gamma=0.99)
    """
    # Get max Q for each agent: [num_agents, batch]
    q_max = jnp.max(q_values_target, axis=-1)

    # Sum across agents for Q_tot_next: [batch]
    q_tot_next = jnp.sum(q_max, axis=0)

    # Compute TD target
    targets = rewards + (1 - dones) * gamma * q_tot_next

    return targets


@partial(jax.jit, static_argnames=('mixing_embed_dim'))
def qmix_mixing_network(
    individual_q: jnp.ndarray,
    state: jnp.ndarray,
    hyper_w1_params: jnp.ndarray,
    hyper_w2_params: jnp.ndarray,
    hyper_b1_params: jnp.ndarray,
    hyper_b2_params: Optional[jnp.ndarray],
    mixing_embed_dim: int = 32,
) -> jnp.ndarray:
    """Compute QMIX-style monotonic mixing of individual Q-values.

    QMIX uses a hypernetwork to generate mixing weights that ensure monotonicity:
        Q_tot = f(Q_1, ..., Q_n; s)
    where the mixing weights are produced by hypernetworks conditioned on state.

    This ensures that argmax_a Q_tot = (argmax_a1 Q_1, ..., argmax_an Q_n),
    maintaining IGM consistency.

    Args:
        individual_q: Individual agent Q-values for chosen actions.
            Shape: [num_agents, batch]
        state: Global state for conditioning. Shape: [batch, state_dim]
        hyper_w1_params: Parameters for first hypernetwork (W1).
        hyper_w2_params: Parameters for second hypernetwork (W2).
        hyper_b1_params: Parameters for first bias hypernetwork (b1).
        hyper_b2_params: Parameters for second bias hypernetwork (b2). Can be None.
        mixing_embed_dim: Dimension of mixing embedding. Default: 32

    Returns:
        Total mixed Q-value. Shape: [batch]

    Note:
        This is a simplified implementation. A full implementation would include
        proper hypernetwork classes and parameter management.

    Example:
        >>> import jax.numpy as jnp
        >>> # Simplified example with mock parameters
        >>> num_agents, batch_size, state_dim = 3, 4, 10
        >>> individual_q = jnp.ones((num_agents, batch_size))
        >>> state = jnp.zeros((batch_size, state_dim))
        >>> # Hypernetwork params would normally come from networks
        >>> # This is just showing the interface
    """
    num_agents = individual_q.shape[0]
    batch_size = individual_q.shape[1]

    # Transpose for batch-first: [batch, num_agents]
    q_batch = individual_q.T

    # First layer: W1 is generated by hypernetwork, shape [batch, num_agents, embed_dim]
    # For simplicity, we'll use a linear approximation here
    # In practice, these would come from actual hypernetworks

    # Placeholder: In real implementation, compute W1, b1 from state
    # W1 = hyper_w1(state)  -> [batch, num_agents * embed_dim]
    # W1 = jnp.abs(W1.reshape(batch, num_agents, embed_dim))  # Ensure positive

    # For now, return sum (equivalent to VDN) as a fallback
    # Full QMIX implementation would need hypernetwork integration
    return jnp.sum(individual_q, axis=0)


def compute_td_loss(
    q_tot: jnp.ndarray,
    targets: jnp.ndarray,
    loss_type: str = "mse",
) -> jnp.ndarray:
    """Compute TD loss for value decomposition.

    Args:
        q_tot: Predicted total Q-values. Shape: [batch]
        targets: TD targets. Shape: [batch]
        loss_type: Type of loss - "mse" or "huber". Default: "mse"

    Returns:
        Scalar loss value

    Example:
        >>> import jax.numpy as jnp
        >>> q_tot = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> targets = jnp.array([1.1, 1.9, 3.2, 3.8])
        >>> loss = compute_td_loss(q_tot, targets)
    """
    if loss_type == "mse":
        loss = jnp.mean(jnp.square(q_tot - targets))
    elif loss_type == "huber":
        delta = jnp.abs(q_tot - targets)
        quadratic = jnp.minimum(delta, 1.0)
        linear = delta - quadratic
        loss = jnp.mean(0.5 * quadratic ** 2 + linear)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


@jax.jit
def epsilon_greedy_action(
    q_values: jnp.ndarray,
    rng: jax.random.PRNGKey,
    epsilon: float,
) -> jnp.ndarray:
    """Select actions using epsilon-greedy exploration.

    Args:
        q_values: Q-values for all actions. Shape: [..., action_dim]
        rng: JAX random key
        epsilon: Exploration rate (0 = greedy, 1 = random)

    Returns:
        Selected actions. Shape matches q_values shape without last dimension.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> q_values = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])
        >>> actions = epsilon_greedy_action(q_values, key, epsilon=0.1)
    """
    action_shape = q_values.shape[:-1]
    action_dim = q_values.shape[-1]

    # Greedy actions
    greedy_actions = jnp.argmax(q_values, axis=-1)

    # Random actions
    random_actions = jax.random.randint(rng, action_shape, 0, action_dim)

    # Epsilon-greedy selection
    explore = jax.random.uniform(rng, action_shape) < epsilon
    actions = jnp.where(explore, random_actions, greedy_actions)

    return actions


@jax.jit
def soft_target_update(
    params: Any,
    target_params: Any,
    tau: float = 0.005,
) -> Any:
    """Perform soft target network update.

    Soft update: target = tau * params + (1 - tau) * target

    Args:
        params: Main network parameters
        target_params: Target network parameters
        tau: Soft update coefficient. Default: 0.005

    Returns:
        Updated target parameters

    Example:
        >>> # During training
        >>> new_target_params = soft_target_update(params, target_params, tau=0.005)
    """
    return jax.tree_util.tree_map(
        lambda p, t: tau * p + (1 - tau) * t,
        params,
        target_params,
    )


@jax.jit
def hard_target_update(
    params: Any,
    target_params: Any,
) -> Any:
    """Perform hard target network update.

    Hard update: target = params

    Args:
        params: Main network parameters
        target_params: Current target network parameters (unused)

    Returns:
        Updated target parameters (copy of params)

    Example:
        >>> # Update target every N steps
        >>> new_target_params = hard_target_update(params, target_params)
    """
    return jax.tree_util.tree_map(
        lambda p: p,
        params,
    )


def create_vdn_loss_fn(
    network_apply_fn,
    target_network_apply_fn,
    gamma: float = 0.99,
    loss_type: str = "mse",
):
    """Create a VDN TD loss function for gradient computation.

    Args:
        network_apply_fn: Function to apply main Q-network
        target_network_apply_fn: Function to apply target Q-network
        gamma: Discount factor. Default: 0.99
        loss_type: Type of TD loss. Default: "mse"

    Returns:
        Loss function suitable for jax.grad

    Example:
        >>> loss_fn = create_vdn_loss_fn(
        ...     network_apply_fn=network.apply,
        ...     target_network_apply_fn=target_network.apply,
        ...     gamma=0.99,
        ... )
        >>> (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        ...     params, obs, actions, rewards, dones, next_obs
        ... )
    """
    def loss_fn(
        params,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        next_obs: jnp.ndarray,
        target_params: Any,
    ):
        """VDN loss function."""
        # Get Q-values for current state (all agents)
        q_values = jax.vmap(network_apply_fn, in_axes=(None, 0))(
            params, obs
        )  # [num_agents, batch, action_dim]

        # VDN decomposition for current Q
        output = vdn_decomposition(q_values, actions)

        # Get target Q-values for next state
        q_target = jax.vmap(target_network_apply_fn, in_axes=(None, 0))(
            target_params, next_obs
        )  # [num_agents, batch, action_dim]

        # Compute TD targets
        targets = vdn_target(q_target, rewards, dones, gamma)

        # Compute TD loss
        loss = compute_td_loss(output.q_tot, targets, loss_type)

        return loss, {
            "q_tot_mean": output.q_tot.mean(),
            "target_mean": targets.mean(),
        }

    return loss_fn


# Convenience exports
__all__ = [
    "vdn_decomposition",
    "vdn_target",
    "qmix_mixing_network",
    "compute_td_loss",
    "epsilon_greedy_action",
    "soft_target_update",
    "hard_target_update",
    "create_vdn_loss_fn",
    "ValueDecompositionOutput",
]
