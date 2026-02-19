"""MLP network architectures for SocialJax.

This module provides configurable MLP networks for multi-agent reinforcement learning
with vector observations. It includes:

- MLPSmall: A lightweight MLP feature extractor for small observation spaces
- MLPActorCritic: A combined actor-critic network using MLP backbone

All networks support configurable layer sizes and hidden dimensions.

Example usage:
    from socialjax.networks import create_network

    # Create a small MLP actor-critic
    network = create_network("mlp_small", action_dim=8, config_preset="small")

    # Create with custom configuration
    network = create_network(
        "mlp_small",
        action_dim=8,
        layer_sizes=[64, 64],
        activation="tanh"
    )
"""

from typing import Sequence, Tuple, Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from socialjax.networks.registry import register_network


class MLPSmall(nn.Module):
    """A lightweight MLP feature extractor for vector observations.

    This network processes vector observations through a configurable series of
    fully-connected layers.

    Architecture:
        - Configurable Dense layers with specified layer sizes
        - Each layer followed by activation function

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        layer_sizes: List of layer sizes for each hidden layer
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Feature embedding of shape (batch_size, layer_sizes[-1])
    """

    layer_sizes: Sequence[int] = (64, 64)
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Select activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply fully-connected layers
        for size in self.layer_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        return x


@register_network("mlp_small")
class MLPActorCritic(nn.Module):
    """Combined Actor-Critic network with MLP backbone.

    This network uses a shared MLP feature extractor with separate
    actor and critic heads. Suitable for algorithms like IPPO that use a
    shared backbone for both policy and value functions.

    Architecture:
        - Shared MLP layers based on layer_sizes
        - Actor head: Dense(actor_hidden_size) -> Dense(action_dim) -> Categorical
        - Critic head: Dense(critic_hidden_size) -> Dense(1) -> Value

    All layers use orthogonal initialization with appropriate scaling.

    Attributes:
        action_dim: Number of discrete actions
        layer_sizes: List of layer sizes for shared hidden layers
        actor_hidden_size: Size of actor head hidden layer (defaults to last layer size)
        critic_hidden_size: Size of critic head hidden layer (defaults to last layer size)
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Tuple of (Categorical distribution, value estimate)
    """

    action_dim: int
    layer_sizes: Sequence[int] = (64, 64)
    actor_hidden_size: Optional[int] = None
    critic_hidden_size: Optional[int] = None
    activation: str = "relu"

    def setup(self):
        # Use last layer size as default for head-specific sizes
        self._actor_hidden = self.actor_hidden_size or self.layer_sizes[-1]
        self._critic_hidden = self.critic_hidden_size or self.layer_sizes[-1]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        # Select activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply shared MLP layers
        for size in self.layer_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        # Actor head
        actor_hidden = nn.Dense(
            self._actor_hidden,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_hidden = activation_fn(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic head
        critic_hidden = nn.Dense(
            self._critic_hidden,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic_hidden = activation_fn(critic_hidden)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic_hidden)

        return pi, jnp.squeeze(value, axis=-1)


@register_network("mlp_encoder")
class MLPEncoder(nn.Module):
    """MLP encoder that returns intermediate features.

    This network is similar to MLPSmall but can be used as an encoder
    for auxiliary tasks or transfer learning.

    Architecture:
        - Configurable Dense layers with specified layer sizes

    Attributes:
        layer_sizes: List of layer sizes for each hidden layer
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Feature embedding
    """

    layer_sizes: Sequence[int] = (64, 64)
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Select activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply fully-connected layers
        for size in self.layer_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        return x


@register_network("mlp_large")
class MLPLargeActorCritic(nn.Module):
    """Large MLP Actor-Critic network for complex tasks.

    This network uses deeper layers for more complex function approximation.
    Suitable for tasks with larger observation spaces or more complex policies.

    Architecture:
        - Shared MLP layers (3 layers by default)
        - Actor head: Dense(hidden) -> Dense(action_dim) -> Categorical
        - Critic head: Dense(hidden) -> Dense(1) -> Value

    All layers use orthogonal initialization with appropriate scaling.

    Attributes:
        action_dim: Number of discrete actions
        hidden_size: Size of all hidden layers
        num_layers: Number of shared hidden layers
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Tuple of (Categorical distribution, value estimate)
    """

    action_dim: int
    hidden_size: int = 128
    num_layers: int = 3
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        # Select activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply shared MLP layers
        for _ in range(self.num_layers):
            x = nn.Dense(
                features=self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        # Actor head
        actor_hidden = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_hidden = activation_fn(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic head
        critic_hidden = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic_hidden = activation_fn(critic_hidden)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic_hidden)

        return pi, jnp.squeeze(value, axis=-1)
