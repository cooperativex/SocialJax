"""
Module: Generative Model for Counterfactual Regret
Equation: Eq.6 from paper - L_m^i = E[||Φ_m^i(o_t, a_t) - r_t^ex||^2]

The generative model Φ_m predicts rewards for all agents given joint observations
and joint actions. This is the foundation for counterfactual reasoning.

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Optional, Tuple, Any
import numpy as np


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for visual observations.

    Extracts features from observations using a CNN architecture
    similar to standard RL visual encoders.
    """
    cnn_features: Sequence[int] = (32, 32, 32)
    cnn_kernels: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            obs: Observation tensor [batch, height, width, channels]
                 or [batch*num_agents, height, width, channels]

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


class RewardModel(nn.Module):
    """Generative Model Φ_m for reward prediction.

    Predicts rewards for all agents given joint observations and joint actions.
    Input: O^N × A^N → Output: R^N

    Architecture:
    1. CNN extracts features from each agent's observation
    2. Actions are one-hot encoded
    3. Features and actions are concatenated
    4. MLP outputs predicted rewards for all agents

    Attributes:
        num_agents: Number of agents in the environment
        action_dim: Dimension of discrete action space
        cnn_features: Number of features in each CNN layer
        cnn_kernels: Kernel sizes for each CNN layer
        hidden_dim: Hidden dimension for MLP layers
        activation: Activation function ("relu" or "tanh")
    """
    num_agents: int
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32)
    cnn_kernels: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict rewards for all agents.

        Args:
            obs: Joint observations [batch, num_agents, height, width, channels]
            actions: Joint actions [batch, num_agents] (discrete integer actions)

        Returns:
            predicted_rewards: Predicted rewards [batch, num_agents]
        """
        batch_size = obs.shape[0]
        num_agents = obs.shape[1]

        # Get activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        # Reshape observations for CNN: [batch*num_agents, H, W, C]
        obs_reshaped = obs.reshape(-1, *obs.shape[2:])

        # Extract features using CNN
        cnn = CNNFeatureExtractor(
            cnn_features=self.cnn_features,
            cnn_kernels=self.cnn_kernels,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
        )
        embeddings = cnn(obs_reshaped)  # [batch*num_agents, hidden_dim]

        # Reshape embeddings: [batch, num_agents * hidden_dim]
        embeddings = embeddings.reshape(batch_size, num_agents * self.hidden_dim)

        # One-hot encode actions: [batch, num_agents, action_dim] -> [batch, num_agents * action_dim]
        actions_onehot = nn.one_hot(actions, self.action_dim)
        actions_flat = actions_onehot.reshape(batch_size, -1)

        # Concatenate embeddings and actions
        x = jnp.concatenate([embeddings, actions_flat], axis=-1)

        # MLP to predict rewards
        x = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation_fn(x)
        x = nn.Dense(
            num_agents,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)

        return x  # [batch, num_agents]


def generative_model_loss(
    predicted_rewards: jnp.ndarray,
    actual_rewards: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    reduction: str = "mean",
) -> jnp.ndarray:
    """
    MSE Loss for training the generative model.

    Implements Eq.6: L_m^i = E[||Φ_m^i(o_t, a_t) - r_t^ex||^2]

    The loss is the mean squared error between predicted and actual rewards,
    averaged over all agents and batch elements.

    Args:
        predicted_rewards: Model predictions [batch, num_agents]
        actual_rewards: Ground truth rewards [batch, num_agents]
        mask: Optional mask for valid entries [batch, num_agents]
        reduction: "mean", "sum", or "none"

    Returns:
        loss: Scalar loss value (or [batch, num_agents] if reduction="none")
    """
    # Compute squared error
    squared_error = (predicted_rewards - actual_rewards) ** 2

    # Apply mask if provided
    if mask is not None:
        squared_error = squared_error * mask

    # Apply reduction
    if reduction == "none":
        return squared_error
    elif reduction == "sum":
        return jnp.sum(squared_error)
    else:  # "mean"
        if mask is not None:
            return jnp.sum(squared_error) / (jnp.sum(mask) + 1e-8)
        return jnp.mean(squared_error)


def compute_generative_model_loss(
    params: dict,
    reward_model: RewardModel,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the generative model loss for training.

    This is a functional version suitable for use with JAX transformations.

    Args:
        params: Model parameters
        reward_model: RewardModel instance
        obs: Joint observations [batch, num_agents, H, W, C]
        actions: Joint actions [batch, num_agents]
        rewards: Actual rewards [batch, num_agents]
        mask: Optional mask [batch, num_agents]

    Returns:
        loss: Scalar MSE loss
        predicted_rewards: Predicted rewards [batch, num_agents]
    """
    # Get predictions
    predicted_rewards = reward_model.apply(params, obs, actions)

    # Compute MSE loss using the function
    loss = generative_model_loss(predicted_rewards, rewards, mask, reduction="mean")

    return loss, predicted_rewards


def create_reward_model_train_state(
    reward_model: RewardModel,
    rng: jax.random.PRNGKey,
    sample_obs: jnp.ndarray,
    sample_actions: jnp.ndarray,
    learning_rate: float = 0.001,
) -> Tuple[Any, Any]:
    """
    Create training state for the reward model.

    Args:
        reward_model: RewardModel instance
        rng: JAX random key
        sample_obs: Sample observation for initialization [batch, num_agents, H, W, C]
        sample_actions: Sample actions for initialization [batch, num_agents]
        learning_rate: Learning rate for optimizer

    Returns:
        train_state: Flax TrainState with model params and optimizer
        init_rng: Remaining random key
    """
    from flax.training.train_state import TrainState
    import optax

    # Initialize model parameters
    init_rng, next_rng = jax.random.split(rng)
    params = reward_model.init(init_rng, sample_obs, sample_actions)

    # Create optimizer
    tx = optax.adam(learning_rate=learning_rate)

    # Create train state
    train_state = TrainState.create(
        apply_fn=reward_model.apply,
        params=params,
        tx=tx,
    )

    return train_state, next_rng


@jax.jit
def update_reward_model(
    train_state: Any,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
    """
    Perform one gradient update step for the reward model.

    Args:
        train_state: Current training state
        obs: Observations [batch, num_agents, H, W, C]
        actions: Actions [batch, num_agents]
        rewards: Actual rewards [batch, num_agents]

    Returns:
        new_train_state: Updated training state
        loss: Loss value
        predicted_rewards: Predicted rewards
    """
    from flax.training.train_state import TrainState
    from functools import partial

    def loss_fn(params):
        # We need to access the apply_fn from train_state
        # But this function will be called with just params
        # So we need to use a closure
        raise NotImplementedError("Use train_step below which handles this properly")

    # Define gradient function
    @partial(jax.value_and_grad, has_aux=True)
    def loss_and_grad(params, apply_fn, obs, actions, rewards):
        predicted = apply_fn(params, obs, actions)
        loss = jnp.mean((predicted - rewards) ** 2)
        return loss, predicted

    (loss, predicted), grads = loss_and_grad(
        train_state.params, train_state.apply_fn, obs, actions, rewards
    )

    new_train_state = train_state.apply_gradients(grads=grads)

    return new_train_state, loss, predicted
