"""
Module: Policy Learning (PPO with Shaped Reward)
Equation: Eq.12 from paper

This module implements the policy learning component of the Counterfactual Regret
algorithm. It uses Proximal Policy Optimization (PPO) to train agents using the
shaped reward that combines extrinsic rewards with intrinsic rewards (negative regret).

Key Components:
- ActorCritic: CNN-based network with actor and critic heads
- GAE: Generalized Advantage Estimation
- PPO Loss: Clipped surrogate objective
- Value Loss: MSE with clipping
- Entropy Bonus: Encourages exploration

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, Tuple, Optional, NamedTuple, Any
import numpy as np
import distrax
import optax
from functools import partial


# ============================================================================
# CNN Feature Extractor
# ============================================================================


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for visual observations.

    Architecture:
        - Conv2D layers with configurable features and kernels
        - Flatten and Dense projection to hidden dimension

    Attributes:
        cnn_features: Number of features in each CNN layer
        cnn_kernels: Kernel sizes for each CNN layer
        hidden_dim: Output hidden dimension
        activation: Activation function name ("relu" or "tanh")
    """
    cnn_features: Sequence[int] = (32, 32, 32)
    cnn_kernels: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Extract features from visual observations.

        Args:
            obs: Observation tensor [batch, height, width, channels]

        Returns:
            features: Extracted features [batch, hidden_dim]
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = obs
        for features, kernel in zip(self.cnn_features, self.cnn_kernels):
            x = nn.Conv(
                features=features,
                kernel_size=kernel,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation(x)

        # Flatten and project to hidden dimension
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        return x


# ============================================================================
# Actor-Critic Network
# ============================================================================


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for CF algorithm.

    This network uses a shared CNN backbone with separate actor and critic heads.
    The critic estimates the value function using the shaped reward.

    Architecture:
        - Shared CNN feature extractor
        - Actor head: Dense(64) -> Dense(action_dim) -> Categorical distribution
        - Critic head: Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        action_dim: Number of discrete actions
        cnn_features: Number of features in each CNN layer
        cnn_kernels: Kernel sizes for each CNN layer
        hidden_dim: Hidden dimension for MLP layers
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Tuple of (policy distribution, value estimate)
    """
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32)
    cnn_kernels: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        """
        Forward pass through actor-critic network.

        Args:
            obs: Observation tensor [batch, height, width, channels]

        Returns:
            pi: Categorical policy distribution
            value: Value estimate [batch]
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN feature extraction
        embedding = CNNFeatureExtractor(
            cnn_features=self.cnn_features,
            cnn_kernels=self.cnn_kernels,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
        )(obs)

        # Actor head
        actor_mean = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


# ============================================================================
# Transition Buffer
# ============================================================================


class Transition(NamedTuple):
    """Container for a single environment transition."""
    done: jnp.ndarray          # [num_steps, batch]
    action: jnp.ndarray        # [num_steps, batch]
    value: jnp.ndarray         # [num_steps, batch]
    reward: jnp.ndarray        # [num_steps, batch] (shaped reward)
    log_prob: jnp.ndarray      # [num_steps, batch]
    obs: jnp.ndarray           # [num_steps, batch, H, W, C]


# ============================================================================
# GAE (Generalized Advantage Estimation)
# ============================================================================


def compute_gae(
    traj_batch: Transition,
    last_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Implements the GAE algorithm from "High-Dimensional Continuous Control Using
    Generalized Advantage Estimation" (Schulman et al., 2016).

    GAE computes advantages as:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...

    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        traj_batch: Batch of transitions (done, action, value, reward, log_prob, obs)
        last_value: Value estimate for the final state [batch]
        gamma: Discount factor (default 0.99)
        gae_lambda: GAE lambda parameter (default 0.95)

    Returns:
        advantages: Computed advantages [num_steps, batch]
        targets: Value targets = advantages + values [num_steps, batch]
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.value, transition.reward

        # TD error: δ = r + γV(s') - V(s)
        delta = reward + gamma * next_value * (1 - done) - value

        # GAE: A = δ + γλ(1-done)A'
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return (gae, value), gae

    # Scan backwards through the trajectory
    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        traj_batch,
        reverse=True,
        unroll=1,
    )

    # Targets = advantages + values
    targets = advantages + traj_batch.value

    return advantages, targets


# ============================================================================
# PPO Loss Functions
# ============================================================================


def compute_ppo_loss(
    params: dict,
    apply_fn: callable,
    traj_batch: Transition,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Compute PPO loss with value function and entropy regularization.

    Implements Eq.12 from the paper using the standard PPO clipped surrogate
    objective with shaped rewards used in advantage calculation.

    The total loss is:
        L = L_actor + vf_coef * L_value - ent_coef * entropy

    Where:
        - L_actor: Clipped surrogate objective
        - L_value: Value function loss (MSE with clipping)
        - entropy: Policy entropy for exploration

    Args:
        params: Network parameters
        apply_fn: Network apply function
        traj_batch: Batch of transitions
        advantages: Computed GAE advantages
        targets: Value function targets
        clip_eps: PPO clipping parameter (default 0.2)
        vf_coef: Value function loss coefficient (default 0.5)
        ent_coef: Entropy bonus coefficient (default 0.01)

    Returns:
        total_loss: Combined loss scalar
        aux: Tuple of (value_loss, actor_loss, entropy)
    """
    # Flatten trajectory for network forward pass
    # traj_batch.obs shape: [num_steps, batch, H, W, C] -> [num_steps * batch, H, W, C]
    num_steps = traj_batch.obs.shape[0]
    batch_size = traj_batch.obs.shape[1]
    obs_flat = traj_batch.obs.reshape(-1, *traj_batch.obs.shape[2:])

    # Forward pass through network
    pi, value = apply_fn(params, obs_flat)
    log_prob = pi.log_prob(traj_batch.action.reshape(-1))

    # Reshape outputs back to [num_steps, batch]
    value = value.reshape(num_steps, batch_size)
    log_prob = log_prob.reshape(num_steps, batch_size)

    # ================================
    # Value Function Loss (with clipping)
    # ================================
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-clip_eps, clip_eps)

    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    # ================================
    # Actor Loss (PPO clipped objective)
    # ================================
    # Importance sampling ratio
    ratio = jnp.exp(log_prob - traj_batch.log_prob)

    # Normalize advantages (for stability)
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Clipped surrogate objective
    loss_actor1 = ratio * advantages_normalized
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_normalized
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

    # ================================
    # Entropy Bonus
    # ================================
    entropy = pi.entropy().mean()

    # ================================
    # Total Loss
    # ================================
    total_loss = (
        loss_actor
        + vf_coef * value_loss
        - ent_coef * entropy
    )

    return total_loss, (value_loss, loss_actor, entropy)


def compute_ppo_loss_with_shaped_reward(
    params: dict,
    apply_fn: callable,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    shaped_rewards: jnp.ndarray,
    dones: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    old_values: jnp.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Compute PPO loss directly from shaped rewards.

    This is a convenience function that combines GAE computation with PPO loss.
    Uses the shaped reward (extrinsic + alpha * intrinsic) for advantage calculation.

    Args:
        params: Network parameters
        apply_fn: Network apply function
        obs: Observations [num_steps, batch, H, W, C]
        actions: Actions [num_steps, batch]
        shaped_rewards: Shaped rewards [num_steps, batch]
        dones: Done flags [num_steps, batch]
        old_log_probs: Old policy log probabilities [num_steps, batch]
        old_values: Old value estimates [num_steps, batch]
        clip_eps: PPO clipping parameter
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        total_loss: Combined loss scalar
        aux: Tuple of (value_loss, actor_loss, entropy, last_value)
    """
    # Create transition batch
    traj_batch = Transition(
        done=dones,
        action=actions,
        value=old_values,
        reward=shaped_rewards,
        log_prob=old_log_probs,
        obs=obs,
    )

    # Get last value for GAE
    pi, last_value = apply_fn(params, obs[-1])

    # Compute GAE
    advantages, targets = compute_gae(traj_batch, last_value, gamma, gae_lambda)

    # Compute PPO loss
    total_loss, (value_loss, actor_loss, entropy) = compute_ppo_loss(
        params, apply_fn, traj_batch, advantages, targets,
        clip_eps, vf_coef, ent_coef
    )

    return total_loss, (value_loss, actor_loss, entropy, last_value)


# ============================================================================
# Gradient Clipping
# ============================================================================


def clip_gradients(
    grads: Any,
    max_grad_norm: float = 0.5,
) -> Tuple[Any, jnp.ndarray]:
    """
    Clip gradients by global norm.

    Args:
        grads: Gradient pytree
        max_grad_norm: Maximum gradient norm (default 0.5)

    Returns:
        clipped_grads: Clipped gradients
        grad_norm: Original gradient norm
    """
    grad_norm = optax.global_norm(grads)
    clipped_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(
            grad_norm > max_grad_norm,
            g * max_grad_norm / (grad_norm + 1e-8),
            g
        ),
        grads
    )
    return clipped_grads, grad_norm


# ============================================================================
# Training State Creation
# ============================================================================


def create_actor_critic_train_state(
    network: ActorCritic,
    rng: jax.random.PRNGKey,
    sample_obs: jnp.ndarray,
    learning_rate: float = 0.0003,
    max_grad_norm: float = 0.5,
    anneal_lr: bool = False,
    num_updates: int = 1000,
) -> Tuple[TrainState, jax.random.PRNGKey]:
    """
    Create training state for actor-critic network.

    Args:
        network: ActorCritic network instance
        rng: JAX random key
        sample_obs: Sample observation for initialization [batch, H, W, C]
        learning_rate: Learning rate for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        anneal_lr: Whether to use learning rate annealing
        num_updates: Number of updates for LR schedule (if annealing)

    Returns:
        train_state: Flax TrainState with model params and optimizer
        next_rng: Remaining random key
    """
    # Initialize model parameters
    init_rng, next_rng = jax.random.split(rng)
    params = network.init(init_rng, sample_obs)

    # Create optimizer
    if anneal_lr:
        # Linear learning rate schedule
        lr_schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=num_updates,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )

    # Create train state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    return train_state, next_rng


# ============================================================================
# Update Step Functions
# ============================================================================


@partial(jax.jit, static_argnames=('apply_fn', 'clip_eps', 'vf_coef', 'ent_coef'))
def ppo_update_step(
    train_state: TrainState,
    traj_batch: Transition,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    apply_fn: callable,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[TrainState, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Perform one PPO update step.

    Args:
        train_state: Current training state
        traj_batch: Batch of transitions
        advantages: GAE advantages
        targets: Value targets
        apply_fn: Network apply function
        clip_eps: PPO clipping parameter
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient

    Returns:
        new_train_state: Updated training state
        loss_info: Tuple of (total_loss, value_loss, actor_loss, entropy)
    """
    def loss_fn(params):
        return compute_ppo_loss(
            params, apply_fn, traj_batch, advantages, targets,
            clip_eps, vf_coef, ent_coef
        )

    (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grads)

    return new_train_state, (total_loss,) + aux


def ppo_update_epoch(
    train_state: TrainState,
    traj_batch: Transition,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    rng: jax.random.PRNGKey,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[TrainState, jax.random.PRNGKey, dict]:
    """
    Perform multiple PPO update epochs with minibatching.

    Args:
        train_state: Current training state
        traj_batch: Batch of transitions
        advantages: GAE advantages
        targets: Value targets
        rng: Random key for shuffling
        num_minibatches: Number of minibatches
        update_epochs: Number of update epochs
        clip_eps: PPO clipping parameter
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient

    Returns:
        new_train_state: Updated training state
        new_rng: Updated random key
        metrics: Dictionary of training metrics
    """
    batch_size = traj_batch.done.shape[0] * traj_batch.done.shape[1]
    minibatch_size = batch_size // num_minibatches

    def _update_epoch(carry, unused):
        ts, rng = carry

        # Shuffle batch
        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, batch_size)

        # Flatten and shuffle
        batch = (traj_batch, advantages, targets)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]),
            batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0),
            batch
        )

        # Split into minibatches
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [num_minibatches, minibatch_size] + list(x.shape[1:])),
            shuffled_batch
        )

        def _update_minibatch(ts, mb):
            mb_traj, mb_adv, mb_tgt = mb
            ts, loss_info = ppo_update_step(
                ts, mb_traj, mb_adv, mb_tgt,
                ts.apply_fn, clip_eps, vf_coef, ent_coef
            )
            return ts, loss_info

        ts, loss_info = jax.lax.scan(_update_minibatch, ts, minibatches)

        return (ts, rng), loss_info

    (new_train_state, new_rng), loss_info = jax.lax.scan(
        _update_epoch,
        (train_state, rng),
        None,
        update_epochs
    )

    # Aggregate metrics
    metrics = {
        'total_loss': loss_info[0].mean(),
        'value_loss': loss_info[1].mean(),
        'actor_loss': loss_info[2].mean(),
        'entropy': loss_info[3].mean(),
    }

    return new_train_state, new_rng, metrics


# ============================================================================
# Utility Functions
# ============================================================================


def get_action(
    params: dict,
    apply_fn: callable,
    obs: jnp.ndarray,
    rng: jax.random.PRNGKey,
    deterministic: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample action from policy.

    Args:
        params: Network parameters
        apply_fn: Network apply function
        obs: Observation [batch, H, W, C]
        rng: Random key
        deterministic: If True, return mode instead of sample

    Returns:
        action: Sampled action [batch]
        log_prob: Log probability of action [batch]
        value: Value estimate [batch]
    """
    pi, value = apply_fn(params, obs)

    if deterministic:
        action = pi.mode()
    else:
        action = pi.sample(seed=rng)

    log_prob = pi.log_prob(action)

    return action, log_prob, value


def get_value(
    params: dict,
    apply_fn: callable,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Get value estimate for observation.

    Args:
        params: Network parameters
        apply_fn: Network apply function
        obs: Observation [batch, H, W, C]

    Returns:
        value: Value estimate [batch]
    """
    _, value = apply_fn(params, obs)
    return value


# ============================================================================
# JIT-compiled Versions
# ============================================================================

# Note: Functions with apply_fn as argument cannot be directly JIT compiled
# because apply_fn is not traceable. Use the non-JIT versions or create
# JIT-compiled wrappers that capture the apply_fn in a closure.

compute_gae_jit = jax.jit(compute_gae)

# For PPO loss, create a factory that captures apply_fn
def make_compute_ppo_loss_jit(apply_fn, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """Create a JIT-compiled PPO loss function with captured apply_fn."""
    @jax.jit
    def loss_fn(params, traj_batch, advantages, targets):
        return compute_ppo_loss(
            params, apply_fn, traj_batch, advantages, targets,
            clip_eps, vf_coef, ent_coef
        )
    return loss_fn

# For get_action and get_value, create factory functions
def make_get_action_jit(apply_fn):
    """Create a JIT-compiled action function with captured apply_fn (stochastic only)."""
    @jax.jit
    def action_fn(params, obs, rng):
        return get_action(params, apply_fn, obs, rng, deterministic=False)
    return action_fn

def make_get_action_deterministic_jit(apply_fn):
    """Create a JIT-compiled deterministic action function with captured apply_fn."""
    @jax.jit
    def action_fn(params, obs):
        rng = jax.random.PRNGKey(0)  # Dummy key for deterministic mode
        return get_action(params, apply_fn, obs, rng, deterministic=True)
    return action_fn

def make_get_value_jit(apply_fn):
    """Create a JIT-compiled value function with captured apply_fn."""
    @jax.jit
    def value_fn(params, obs):
        return get_value(params, apply_fn, obs)
    return value_fn
