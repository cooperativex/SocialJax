"""Neural network architectures for MAPPO algorithm.

This module provides separate Actor and Critic network architectures used by the
Multi-Agent PPO algorithm for centralized training with decentralized execution.

Key features:
- Actor network: Takes local observation only (for decentralized execution)
- Critic network: Takes global state (all agent observations concatenated)
- Both use CNN feature extraction for visual observations
"""

from typing import Sequence, Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from socialjax.networks.registry import register_network


class MAPPOCriticCNN(nn.Module):
    """Convolutional Neural Network for MAPPO critic (global state).

    This CNN processes the global state (concatenated observations from all agents)
    for the centralized critic in MAPPO.

    Architecture:
        - Conv2D (32 filters, 5x5 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Dense (64 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class MAPPOActorCNN(nn.Module):
    """Convolutional Neural Network for MAPPO actor (local observation).

    This CNN processes the local observation for the actor in MAPPO.
    Same architecture as the critic CNN for consistency.

    Architecture:
        - Conv2D (32 filters, 5x5 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Dense (64 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


@register_network("mappo_actor")
class MAPPOActor(nn.Module):
    """Actor network for MAPPO algorithm.

    This network uses a CNN backbone for local observation processing
    and outputs a categorical distribution over actions.

    For decentralized execution, this network only receives local observations.

    Architecture:
        - CNN feature extractor (MAPPOActorCNN)
        - Actor head: Dense(hidden_size) -> Dense(action_dim) -> Categorical

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")
        hidden_size: Size of hidden layers (default: 64)

    Returns:
        Categorical distribution over actions
    """
    action_dim: int
    activation: str = "relu"
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = MAPPOActorCNN(self.activation)(x)

        # Actor head
        actor_mean = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


@register_network("mappo_critic")
class MAPPOCritic(nn.Module):
    """Centralized Critic network for MAPPO algorithm.

    This network receives the global state (concatenated observations from all agents)
    and outputs a value estimate. This enables centralized training.

    Architecture:
        - CNN feature extractor (MAPPOCriticCNN)
        - Critic head: Dense(hidden_size) -> Dense(1) -> Value estimate

    Attributes:
        activation: Activation function name ("relu" or "tanh")
        hidden_size: Size of hidden layers (default: 64)

    Returns:
        Scalar value estimate
    """
    activation: str = "relu"
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = MAPPOCriticCNN(self.activation)(x)

        # Critic head
        critic = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return jnp.squeeze(critic, axis=-1)
