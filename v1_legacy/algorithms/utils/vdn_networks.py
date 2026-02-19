"""
VDN-specific neural network architectures.

This module contains network architectures specific to VDN (Value Decomposition Networks).
VDN uses the standard CNN from the shared networks module.
"""

import flax.linen as nn
import jax.numpy as jnp

# Import standard CNN from shared networks
from algorithms.utils.networks import CNN


class QNetwork(nn.Module):
    """
    Q-Network for VDN (Value Decomposition Networks) algorithm.

    This network estimates individual agent Q-values which are then summed
    to produce the total team Q-value in VDN.

    Architecture:
        - Standard CNN feature extractor (shared from networks.py)
        - Dense(hidden_size) -> Dense(action_dim) -> Q-values

    Attributes:
        action_dim: Number of discrete actions
        hidden_size: Size of hidden layer (default: 64)
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Q-values for each action (shape: [batch_size, action_dim])
    """
    action_dim: int
    hidden_size: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)
        # no activation here as a nonlinearity has already
        # been applied to the embedding
        x = nn.Dense(self.hidden_size)(embedding)
        x = activation(x)
        x = nn.Dense(self.action_dim)(x)
        return x
