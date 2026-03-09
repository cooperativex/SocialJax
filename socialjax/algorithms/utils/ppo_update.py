"""PPO (Proximal Policy Optimization) loss utilities.

This module provides JAX-JIT compatible implementations of PPO loss functions
for policy gradient algorithms like PPO, IPPO, and MAPPO.

The key components are:
- Clipped surrogate objective for policy updates
- Value function loss with optional clipping
- Entropy bonus for exploration

Reference:
    "Proximal Policy Optimization Algorithms"
    (Schulman et al., 2017)
"""

from typing import Tuple, Dict, Any, Optional, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
import distrax


class PPOLossComponents(NamedTuple):
    """Components of the PPO loss.

    Attributes:
        policy_loss: Clipped surrogate policy loss
        value_loss: Value function loss (MSE with optional clipping)
        entropy: Policy entropy (for exploration bonus)
        total_loss: Combined loss: policy_loss + vf_coef * value_loss - ent_coef * entropy
        clip_frac: Fraction of samples where clipping was applied
        approx_kl: Approximate KL divergence between old and new policy
    """
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy: jnp.ndarray
    total_loss: jnp.ndarray
    clip_frac: jnp.ndarray
    approx_kl: jnp.ndarray


@partial(jax.jit, static_argnames=('clip_eps', 'normalize_advantages'))
def compute_policy_loss(
    log_prob: jnp.ndarray,
    old_log_prob: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_eps: float = 0.2,
    normalize_advantages: bool = True,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the PPO clipped surrogate policy loss.

    The clipped surrogate objective is:
        L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
        where r_t = pi(a|s) / pi_old(a|s) is the probability ratio

    Args:
        log_prob: Current policy log probabilities. Shape: [batch] or [batch, ...]
        old_log_prob: Old policy log probabilities from data collection. Same shape.
        advantages: Advantage estimates. Same shape.
        clip_eps: PPO clipping parameter. Default: 0.2
        normalize_advantages: Whether to normalize advantages before use. Default: True
        eps: Small constant for numerical stability. Default: 1e-8

    Returns:
        Tuple of (policy_loss, clip_fraction, approx_kl) where:
        - policy_loss: Scalar policy loss (to be minimized)
        - clip_fraction: Fraction of samples where clipping was active
        - approx_kl: Approximate KL divergence (old_log_prob - log_prob).mean()

    Example:
        >>> import jax.numpy as jnp
        >>> import distrax
        >>> # Simulated outputs
        >>> log_prob = jnp.array([-0.5, -1.0, -0.8, -1.2])
        >>> old_log_prob = jnp.array([-0.4, -0.9, -1.0, -1.1])
        >>> advantages = jnp.array([1.0, -0.5, 0.2, 0.8])
        >>> loss, clip_frac, kl = compute_policy_loss(log_prob, old_log_prob, advantages)
    """
    # Normalize advantages
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

    # Compute probability ratio r = pi / pi_old = exp(log_pi - log_pi_old)
    ratio = jnp.exp(log_prob - old_log_prob)

    # Clipped surrogate objective
    ratio_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    # Compute both terms
    loss1 = ratio * advantages
    loss2 = ratio_clipped * advantages

    # Take minimum (pessimistic bound)
    policy_loss = -jnp.minimum(loss1, loss2).mean()

    # Compute clip fraction for monitoring
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(float))

    # Approximate KL divergence for monitoring
    approx_kl = (old_log_prob - log_prob).mean()

    return policy_loss, clip_frac, approx_kl


@partial(jax.jit, static_argnames=('clip_eps', 'use_clipping'))
def compute_value_loss(
    value: jnp.ndarray,
    old_value: jnp.ndarray,
    target: jnp.ndarray,
    clip_eps: float = 0.2,
    use_clipping: bool = True,
) -> jnp.ndarray:
    """Compute the value function loss with optional clipping.

    Value clipping can help prevent large value function updates that could
    destabilize training. The clipped value loss is:
        L^VF = 0.5 * E[max((V - target)^2, (V_clipped - target)^2)]
        where V_clipped = V_old + clip(V - V_old, -eps, eps)

    Args:
        value: Current value estimates. Shape: [batch] or [batch, ...]
        old_value: Value estimates from data collection. Same shape.
        target: Value targets (e.g., from GAE). Same shape.
        clip_eps: Value clipping parameter. Default: 0.2
        use_clipping: Whether to use value clipping. Default: True

    Returns:
        Scalar value loss (MSE, optionally with clipping)

    Example:
        >>> import jax.numpy as jnp
        >>> value = jnp.array([0.5, 0.8, 0.3, 0.9])
        >>> old_value = jnp.array([0.4, 0.7, 0.4, 0.85])
        >>> target = jnp.array([0.6, 0.75, 0.35, 0.95])
        >>> loss = compute_value_loss(value, old_value, target, clip_eps=0.2)
    """
    if use_clipping:
        # Clipped value prediction
        value_pred_clipped = old_value + jnp.clip(value - old_value, -clip_eps, clip_eps)

        # Compute both losses
        value_losses = jnp.square(value - target)
        value_losses_clipped = jnp.square(value_pred_clipped - target)

        # Take maximum (pessimistic bound)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        # Standard MSE loss
        value_loss = 0.5 * jnp.mean(jnp.square(value - target))

    return value_loss


@jax.jit
def compute_entropy_bonus(
    distribution: distrax.Distribution,
) -> jnp.ndarray:
    """Compute entropy bonus for exploration.

    Higher entropy encourages more exploration. The entropy bonus is typically
    added to the loss as: total_loss -= ent_coef * entropy

    Args:
        distribution: Distribution object (e.g., from distrax.Categorical)
            with an entropy() method.

    Returns:
        Scalar mean entropy.

    Example:
        >>> import jax.numpy as jnp
        >>> import distrax
        >>> logits = jnp.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> dist = distrax.Categorical(logits=logits)
        >>> entropy = compute_entropy_bonus(dist)
    """
    return distribution.entropy().mean()


def compute_ppo_loss(
    distribution: distrax.Distribution,
    value: jnp.ndarray,
    action: jnp.ndarray,
    old_log_prob: jnp.ndarray,
    old_value: jnp.ndarray,
    advantage: jnp.ndarray,
    target: jnp.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    normalize_advantages: bool = True,
    use_value_clipping: bool = True,
) -> PPOLossComponents:
    """Compute all components of the PPO loss.

    This is a convenience function that computes policy loss, value loss,
    and entropy bonus together.

    Args:
        distribution: Current policy distribution
        value: Current value estimates. Shape: [batch]
        action: Actions taken. Shape: [batch]
        old_log_prob: Log probabilities from data collection. Shape: [batch]
        old_value: Value estimates from data collection. Shape: [batch]
        advantage: Advantage estimates. Shape: [batch]
        target: Value targets. Shape: [batch]
        clip_eps: PPO clipping parameter. Default: 0.2
        vf_coef: Value function loss coefficient. Default: 0.5
        ent_coef: Entropy bonus coefficient. Default: 0.01
        normalize_advantages: Whether to normalize advantages. Default: True
        use_value_clipping: Whether to clip value updates. Default: True

    Returns:
        PPOLossComponents namedtuple with all loss components

    Example:
        >>> import jax.numpy as jnp
        >>> import distrax
        >>> # Simulated outputs
        >>> logits = jnp.array([[1.0, 2.0], [2.0, 1.0], [0.5, 1.5]])
        >>> dist = distrax.Categorical(logits=logits)
        >>> value = jnp.array([0.5, 0.8, 0.3])
        >>> action = jnp.array([1, 0, 1])
        >>> old_log_prob = jnp.array([-0.5, -0.4, -0.6])
        >>> old_value = jnp.array([0.4, 0.7, 0.4])
        >>> advantage = jnp.array([0.1, -0.2, 0.3])
        >>> target = jnp.array([0.6, 0.75, 0.5])
        >>> losses = compute_ppo_loss(dist, value, action, old_log_prob,
        ...                           old_value, advantage, target)
    """
    # Current log probability
    log_prob = distribution.log_prob(action)

    # Policy loss
    policy_loss, clip_frac, approx_kl = compute_policy_loss(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantage,
        clip_eps=clip_eps,
        normalize_advantages=normalize_advantages,
    )

    # Value loss
    value_loss = compute_value_loss(
        value=value,
        old_value=old_value,
        target=target,
        clip_eps=clip_eps,
        use_clipping=use_value_clipping,
    )

    # Entropy bonus
    entropy = compute_entropy_bonus(distribution)

    # Total loss
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    return PPOLossComponents(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        total_loss=total_loss,
        clip_frac=clip_frac,
        approx_kl=approx_kl,
    )


def create_ppo_update_fn(
    network_apply_fn,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    normalize_advantages: bool = True,
    use_value_clipping: bool = True,
):
    """Create a JIT-compiled PPO update function.

    This factory function creates an update function suitable for use
    with JAX's grad transformation.

    Args:
        network_apply_fn: Function that applies the network to get (dist, value)
        clip_eps: PPO clipping parameter. Default: 0.2
        vf_coef: Value function loss coefficient. Default: 0.5
        ent_coef: Entropy bonus coefficient. Default: 0.01
        normalize_advantages: Whether to normalize advantages. Default: True
        use_value_clipping: Whether to clip value updates. Default: True

    Returns:
        A function loss_fn(params, obs, action, old_log_prob, old_value, advantage, target)
        that returns (total_loss, PPOLossComponents)

    Example:
        >>> # Create loss function for your network
        >>> loss_fn = create_ppo_update_fn(
        ...     network_apply_fn=network.apply,
        ...     clip_eps=0.2,
        ...     vf_coef=0.5,
        ...     ent_coef=0.01,
        ... )
        >>> # Compute gradients
        >>> (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, ...)
    """
    def loss_fn(
        params,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        old_value: jnp.ndarray,
        advantage: jnp.ndarray,
        target: jnp.ndarray,
    ):
        """PPO loss function for gradient computation."""
        # Forward pass through network
        distribution, value = network_apply_fn(params, obs)

        # Compute loss components
        loss_components = compute_ppo_loss(
            distribution=distribution,
            value=value,
            action=action,
            old_log_prob=old_log_prob,
            old_value=old_value,
            advantage=advantage,
            target=target,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            normalize_advantages=normalize_advantages,
            use_value_clipping=use_value_clipping,
        )

        return loss_components.total_loss, loss_components

    return loss_fn


# Convenience exports
__all__ = [
    "compute_policy_loss",
    "compute_value_loss",
    "compute_entropy_bonus",
    "compute_ppo_loss",
    "create_ppo_update_fn",
    "PPOLossComponents",
]
