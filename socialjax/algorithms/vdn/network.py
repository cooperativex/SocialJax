"""Neural network architectures for VDN algorithm.

This module provides the Q-Network architecture used by the
Value Decomposition Networks (VDN) algorithm for cooperative
multi-agent reinforcement learning.

VDN decomposes the team Q-value into individual agent Q-values:
    Q_tot(s, a) = sum_i Q_i(s_i, a_i)

This enables decentralized execution while maintaining coordinated learning.
"""

from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from socialjax.networks.registry import register_network


class VDNCNN(nn.Module):
    """Convolutional Neural Network for visual feature extraction in VDN.

    Architecture:
        - Conv2D (32 filters, 5x5 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Dense (hidden_size units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
        hidden_size: Size of the final dense layer (default: 64)
    """

    activation: str = "relu"
    hidden_size: int = 64

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
            features=self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        return x


@register_network("vdn_q_network")
class VDNQNetwork(nn.Module):
    """Q-Network for VDN (Value Decomposition Networks) algorithm.

    This network estimates individual agent Q-values which are then summed
    to produce the total team Q-value in VDN: Q_tot = sum_i Q_i

    Architecture:
        - VDNCNN feature extractor
        - Dense(hidden_size) with activation
        - Dense(action_dim) without activation -> Q-values

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")
        hidden_size: Size of hidden layer (default: 64)

    Returns:
        Tuple of (policy distribution, max Q-value) for compatibility with PPO trainer
    """

    action_dim: int
    activation: str = "relu"
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN feature extraction
        embedding = VDNCNN(self.activation, self.hidden_size)(x)

        # Q-value head (no activation on output)
        q_values = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(embedding)

        # Create policy distribution from Q-values (treat Q-values as logits)
        # For VDN, we use epsilon-greedy but the trainer samples from pi
        # Using softmax over Q-values gives a Boltzmann policy
        pi = distrax.Categorical(logits=q_values)

        # Value is the max Q-value (for greedy policy)
        value = jnp.max(q_values, axis=-1)

        return pi, value


def compute_q_tot(q_values: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Compute total team Q-value by summing individual agent Q-values.

    VDN decomposition: Q_tot(s, a) = sum_i Q_i(s_i, a_i)

    Args:
        q_values: Individual agent Q-values with shape (num_agents, batch_size, action_dim)
            or (batch_size, num_agents, action_dim)
        axis: Axis along which agents are stacked (default: 0)

    Returns:
        Total Q-value (sum of chosen action Q-values across agents)
    """
    return jnp.sum(q_values, axis=axis)


def compute_vdn_target(
    q_next_target: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
) -> jnp.ndarray:
    """Compute VDN target value for TD learning.

    VDN target: y = r + gamma * sum_i max_a' Q_i^target(s', a')

    Args:
        q_next_target: Target Q-values for next state (num_agents, batch_size, action_dim)
        rewards: Team rewards (batch_size,)
        dones: Done flags (batch_size,)
        gamma: Discount factor

    Returns:
        Target values for TD learning (batch_size,)
    """
    # Get max Q-value for each agent
    q_next_max = jnp.max(q_next_target, axis=-1)  # (num_agents, batch_size)

    # Sum across agents to get Q_tot
    q_tot_next = jnp.sum(q_next_max, axis=0)  # (batch_size,)

    # TD target
    target = rewards + (1 - dones) * gamma * q_tot_next

    return target
