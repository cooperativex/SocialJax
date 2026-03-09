"""
Module: Reward Shaping
Equation: Eq.11 from paper

Combines extrinsic reward and intrinsic reward to create the shaped reward
that guides agents toward prosocial behavior while still optimizing for
individual rewards.

r̂_t^i = r_t^{i,ex} + α * r_t^{i,in}

Where:
- r_t^{i,ex} is the extrinsic (environment) reward
- r_t^{i,in} is the intrinsic reward (negative regret from Eq.10)
- α is the coefficient controlling how much the agent cares about others

The paper suggests α ≈ N-1 (number of other agents) to balance individual
and collective rewards.

Properties:
- When α = 0: Agent only optimizes extrinsic reward (selfish)
- When α > 0: Agent also considers prosocial behavior
- Higher α = more prosocial behavior

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial


# Default alpha values
DEFAULT_ALPHA = 1.0
SUGGESTED_ALPHA_N_MINUS_1 = True  # Use N-1 as default


def compute_shaped_reward(
    extrinsic_reward: jnp.ndarray,
    intrinsic_reward: jnp.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> jnp.ndarray:
    """
    Compute shaped reward by combining extrinsic and intrinsic rewards.

    Implements Eq.11: r̂_t^i = r_t^{i,ex} + α * r_t^{i,in}

    The shaped reward is what the policy is trained to maximize. It balances
    individual extrinsic rewards with intrinsic rewards that encourage
    prosocial behavior.

    Args:
        extrinsic_reward: Environment rewards for each agent
                         Shape: [batch, num_agents]
        intrinsic_reward: Intrinsic rewards (negative regret) for each agent
                         Shape: [batch, num_agents]
                         Should be <= 0 (from compute_intrinsic_reward)
        alpha: Coefficient controlling prosocial behavior (default 1.0)
               - α = 0: Only optimize extrinsic reward
               - α > 0: Also consider intrinsic reward
               - Paper suggests α ≈ N-1 (number of other agents)

    Returns:
        shaped_reward: Combined reward for each agent
                      Shape: [batch, num_agents]

    Example:
        >>> extrinsic = jnp.array([[1.0, 0.5, 0.0]])
        >>> intrinsic = jnp.array([[0.0, -0.5, -1.0]])  # negative regret
        >>> shaped = compute_shaped_reward(extrinsic, intrinsic, alpha=2.0)
        >>> # shaped = [[1.0, -0.5, -2.0]]
        >>> #          1.0 + 0*2,  0.5 + (-0.5)*2,  0.0 + (-1.0)*2

    Note:
        Since intrinsic_reward = -regret and regret >= 0:
        - intrinsic_reward <= 0
        - alpha * intrinsic_reward <= 0 (for alpha > 0)
        - shaped_reward <= extrinsic_reward (prosocial penalty)
    """
    shaped_reward = extrinsic_reward + alpha * intrinsic_reward
    return shaped_reward


def compute_shaped_reward_from_regret(
    extrinsic_reward: jnp.ndarray,
    regret: jnp.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> jnp.ndarray:
    """
    Convenience function to compute shaped reward directly from regret.

    Combines intrinsic reward computation (Eq.10) and shaped reward
    computation (Eq.11) into a single function.

    Args:
        extrinsic_reward: Environment rewards [batch, num_agents]
        regret: Counterfactual regret [batch, num_agents]
               Should be >= 0
        alpha: Coefficient for prosocial behavior

    Returns:
        shaped_reward: Combined reward [batch, num_agents]

    Note:
        This is equivalent to:
        intrinsic = -regret
        shaped = extrinsic + alpha * intrinsic
    """
    intrinsic_reward = -regret
    return compute_shaped_reward(extrinsic_reward, intrinsic_reward, alpha)


def compute_alpha_n_minus_1(
    num_agents: int,
) -> float:
    """
    Compute suggested alpha value based on number of agents.

    The paper suggests α ≈ N-1 (number of other agents) as a good default.
    This balances the scale of extrinsic reward with collective welfare.

    Args:
        num_agents: Number of agents in the environment

    Returns:
        alpha: Suggested coefficient value (N-1)

    Example:
        >>> compute_alpha_n_minus_1(3)  # 3 agents
        2.0
        >>> compute_alpha_n_minus_1(7)  # 7 agents
        6.0
    """
    return float(num_agents - 1)


def compute_shaped_reward_auto_alpha(
    extrinsic_reward: jnp.ndarray,
    intrinsic_reward: jnp.ndarray,
    num_agents: int,
) -> jnp.ndarray:
    """
    Compute shaped reward with automatic alpha = N-1.

    Uses the paper's suggested alpha value (number of other agents).

    Args:
        extrinsic_reward: Environment rewards [batch, num_agents]
        intrinsic_reward: Intrinsic rewards [batch, num_agents]
        num_agents: Number of agents (used to compute alpha = N-1)

    Returns:
        shaped_reward: Combined reward with alpha = N-1
    """
    alpha = compute_alpha_n_minus_1(num_agents)
    return compute_shaped_reward(extrinsic_reward, intrinsic_reward, alpha)


def normalize_shaped_reward(
    shaped_reward: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Normalize shaped rewards to have zero mean and unit variance.

    Optional normalization for training stability. This is useful when
    extrinsic and intrinsic rewards have very different scales.

    Args:
        shaped_reward: Shaped rewards [batch, num_agents]
        eps: Small constant for numerical stability

    Returns:
        normalized_reward: Normalized rewards [batch, num_agents]

    Note:
        Normalization is applied across the batch dimension, preserving
        the relative differences between agents.
    """
    mean = jnp.mean(shaped_reward, axis=0, keepdims=True)
    std = jnp.std(shaped_reward, axis=0, keepdims=True)
    normalized = (shaped_reward - mean) / (std + eps)
    return normalized


def compute_shaped_reward_normalized(
    extrinsic_reward: jnp.ndarray,
    intrinsic_reward: jnp.ndarray,
    alpha: float = DEFAULT_ALPHA,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Compute shaped reward with optional normalization.

    Combines reward shaping and normalization for convenience.

    Args:
        extrinsic_reward: Environment rewards [batch, num_agents]
        intrinsic_reward: Intrinsic rewards [batch, num_agents]
        alpha: Coefficient for prosocial behavior
        eps: Small constant for numerical stability

    Returns:
        normalized_shaped_reward: Normalized combined reward
    """
    shaped = compute_shaped_reward(extrinsic_reward, intrinsic_reward, alpha)
    return normalize_shaped_reward(shaped, eps)


def get_shaped_reward_statistics(
    shaped_reward: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute statistics over shaped reward values.

    Useful for logging and monitoring during training.

    Args:
        shaped_reward: Shaped rewards [batch, num_agents]

    Returns:
        mean_reward: Mean shaped reward per agent [num_agents]
        std_reward: Standard deviation per agent [num_agents]
        min_reward: Minimum shaped reward per agent [num_agents]
        max_reward: Maximum shaped reward per agent [num_agents]
    """
    mean_reward = jnp.mean(shaped_reward, axis=0)  # [num_agents]
    std_reward = jnp.std(shaped_reward, axis=0)  # [num_agents]
    min_reward = jnp.min(shaped_reward, axis=0)  # [num_agents]
    max_reward = jnp.max(shaped_reward, axis=0)  # [num_agents]

    return mean_reward, std_reward, min_reward, max_reward


def compute_shaped_reward_with_components(
    extrinsic_reward: jnp.ndarray,
    intrinsic_reward: jnp.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute shaped reward and return all components for analysis.

    Useful for debugging and understanding the contribution of each
    reward component.

    Args:
        extrinsic_reward: Environment rewards [batch, num_agents]
        intrinsic_reward: Intrinsic rewards [batch, num_agents]
        alpha: Coefficient for prosocial behavior

    Returns:
        shaped_reward: Combined reward [batch, num_agents]
        extrinsic_component: Extrinsic contribution [batch, num_agents]
        intrinsic_component: Scaled intrinsic contribution [batch, num_agents]
    """
    extrinsic_component = extrinsic_reward
    intrinsic_component = alpha * intrinsic_reward
    shaped_reward = extrinsic_component + intrinsic_component

    return shaped_reward, extrinsic_component, intrinsic_component


def verify_shaped_reward_properties(
    shaped_reward: jnp.ndarray,
    extrinsic_reward: jnp.ndarray,
    intrinsic_reward: jnp.ndarray,
    alpha: float,
) -> Tuple[bool, str]:
    """
    Verify that shaped reward satisfies expected properties.

    Args:
        shaped_reward: Computed shaped reward [batch, num_agents]
        extrinsic_reward: Original extrinsic rewards [batch, num_agents]
        intrinsic_reward: Original intrinsic rewards [batch, num_agents]
        alpha: Coefficient used

    Returns:
        is_valid: Whether properties are satisfied
        message: Description of any violations
    """
    # Check formula correctness
    expected = extrinsic_reward + alpha * intrinsic_reward
    formula_correct = jnp.allclose(shaped_reward, expected, atol=1e-6)

    if not formula_correct:
        return False, "Shaped reward does not match expected formula"

    # Check for NaN/Inf
    has_nan = jnp.any(jnp.isnan(shaped_reward))
    has_inf = jnp.any(jnp.isinf(shaped_reward))

    if has_nan:
        return False, "Shaped reward contains NaN values"
    if has_inf:
        return False, "Shaped reward contains Inf values"

    return True, "All properties satisfied"


def compute_shaped_reward_gradient(
    extrinsic_reward: jnp.ndarray,
    intrinsic_reward: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gradients of shaped reward with respect to inputs.

    Since r̂ = r^{ex} + α * r^{in}:
    - ∂r̂/∂r^{ex} = 1
    - ∂r̂/∂r^{in} = α

    This function is provided for documentation and testing purposes.
    In practice, JAX autodiff handles this automatically.

    Args:
        extrinsic_reward: Environment rewards [batch, num_agents]
        intrinsic_reward: Intrinsic rewards [batch, num_agents]

    Returns:
        grad_extrinsic: Gradient w.r.t. extrinsic reward (all 1s)
        grad_intrinsic: Gradient w.r.t. intrinsic reward (all 1s)
                       Note: alpha gradient is handled separately
    """
    grad_extrinsic = jnp.ones_like(extrinsic_reward)
    grad_intrinsic = jnp.ones_like(intrinsic_reward)
    return grad_extrinsic, grad_intrinsic


# JIT-compiled versions for performance
compute_shaped_reward_jit = jax.jit(compute_shaped_reward)
compute_shaped_reward_from_regret_jit = jax.jit(compute_shaped_reward_from_regret)
compute_shaped_reward_auto_alpha_jit = jax.jit(compute_shaped_reward_auto_alpha)
normalize_shaped_reward_jit = jax.jit(normalize_shaped_reward)
get_shaped_reward_statistics_jit = jax.jit(get_shaped_reward_statistics)
