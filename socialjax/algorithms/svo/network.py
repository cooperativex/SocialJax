"""Neural network architectures for SVO algorithm.

This module provides the Actor-Critic network architecture used by the
SVO algorithm for multi-agent reinforcement learning with social preferences.
The network architecture is similar to IPPO but can be extended for
SVO-specific features.
"""

from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from socialjax.networks.registry import register_network


class SVOCNN(nn.Module):
    """Convolutional Neural Network for visual feature extraction in SVO.

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


@register_network("svo_actor_critic")
class SVOActorCritic(nn.Module):
    """Combined Actor-Critic network for SVO algorithm.

    This network uses a shared CNN backbone with separate actor and critic heads.
    It extends the IPPO architecture for use with Social Value Orientation
    reward transformations.

    Architecture:
        - Shared CNN feature extractor
        - Actor head: Dense(64) -> Dense(action_dim) -> Categorical distribution
        - Critic head: Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")
        hidden_size: Size of hidden layers (default: 64)

    Returns:
        Tuple of (policy distribution, value estimate)
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

        embedding = SVOCNN(self.activation)(x)

        # Actor head
        actor_mean = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
