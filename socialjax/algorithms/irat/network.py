"""Neural network architectures for IRAT algorithm.

IRAT uses 4 networks:
- Individual Actor: local obs → action distribution (optimized for individual rewards)
- Individual Critic: local obs → value (baseline for individual policy)
- Team Actor: local obs → action distribution (EXECUTED in environment)
- Team Critic: world state → value (baseline for team policy)

The team actor acts in the environment. The individual actor evaluates the same
actions on-policy, enabling individual-reward-assisted learning.
"""

from typing import Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class IRATActorCNN(nn.Module):
    """Actor CNN for IRAT (individual or team actor).

    Processes local observation and outputs a categorical action distribution.

    Architecture:
        - Conv2D (16 filters, 3x3)
        - Dense (16 units)
        - Dense (16 units) actor head
        - Dense (action_dim) → Categorical

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function ("relu" or "tanh")
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # CNN feature extractor
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        # Actor head
        x = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)

        return distrax.Categorical(logits=x)


class IRATCriticCNN(nn.Module):
    """Critic CNN for IRAT.

    For individual critic: takes local observation.
    For team critic: takes world state (all agents' obs stacked along channel dim).

    Architecture:
        - Conv2D (16 filters, 3x3)
        - Dense (16 units)
        - Dense (1) → scalar value

    Attributes:
        activation: Activation function ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # CNN feature extractor
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        # Critic head
        x = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)

        return jnp.squeeze(x, axis=-1)
