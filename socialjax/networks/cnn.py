"""CNN network architectures for SocialJax.

This module provides configurable CNN networks for multi-agent reinforcement learning
with visual observations. It includes:

- CNNSmall: A lightweight CNN feature extractor for small observation spaces
- CNNActorCritic: A combined actor-critic network using CNN backbone

All networks support configurable channel sizes, kernel sizes, and hidden dimensions.

Example usage:
    from socialjax.networks import create_network

    # Create a small CNN actor-critic
    network = create_network("cnn_actor_critic", action_dim=8, config_preset="small")

    # Create with custom configuration
    network = create_network(
        "cnn_actor_critic",
        action_dim=8,
        channel_sizes=[16, 32, 32],
        kernel_sizes=[5, 3, 3],
        hidden_size=128
    )
"""

from typing import Sequence, Tuple, Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from socialjax.networks.registry import register_network


class CNNSmall(nn.Module):
    """A lightweight CNN feature extractor for small observation spaces.

    This network processes visual observations through a configurable series of
    convolutional layers followed by a fully-connected layer.

    Architecture:
        - Configurable Conv2D layers with specified channel and kernel sizes
        - Flatten layer
        - Fully-connected layer with hidden_size units

    All layers use orthogonal initialization with sqrt(2) scaling for conv layers.

    Attributes:
        channel_sizes: List of output channels for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer (int or tuple)
        hidden_size: Size of the final fully-connected layer
        activation: Activation function name ("relu" or "tanh")
        padding: Padding type for conv layers ("SAME" or "VALID")

    Returns:
        Feature embedding of shape (batch_size, hidden_size)
    """

    channel_sizes: Sequence[int] = (16, 32)
    kernel_sizes: Sequence[int] = (3, 3)
    hidden_size: int = 64
    activation: str = "relu"
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Validate channel and kernel sizes match
        if len(self.channel_sizes) != len(self.kernel_sizes):
            raise ValueError(
                f"channel_sizes ({len(self.channel_sizes)}) and kernel_sizes "
                f"({len(self.kernel_sizes)}) must have the same length"
            )

        # Select activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply convolutional layers
        for channels, kernel in zip(self.channel_sizes, self.kernel_sizes):
            # Handle both int and tuple kernel sizes
            if isinstance(kernel, int):
                kernel_size = (kernel, kernel)
            else:
                kernel_size = kernel

            x = nn.Conv(
                features=channels,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding=self.padding,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        # Flatten spatial dimensions
        x = x.reshape((x.shape[0], -1))

        # Final fully-connected layer
        x = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation_fn(x)

        return x


@register_network("cnn_small")
class CNNActorCritic(nn.Module):
    """Combined Actor-Critic network with CNN backbone.

    This network uses a shared CNN feature extractor (CNNSmall) with separate
    actor and critic heads. Suitable for algorithms like IPPO that use a
    shared backbone for both policy and value functions.

    Architecture:
        - Shared CNNSmall feature extractor
        - Actor head: Dense(hidden_size) -> Dense(action_dim) -> Categorical
        - Critic head: Dense(hidden_size) -> Dense(1) -> Value

    All layers use orthogonal initialization with appropriate scaling.

    Attributes:
        action_dim: Number of discrete actions
        channel_sizes: List of output channels for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        hidden_size: Size of hidden layers in both CNN and heads
        actor_hidden_size: Size of actor head hidden layer (defaults to hidden_size)
        critic_hidden_size: Size of critic head hidden layer (defaults to hidden_size)
        activation: Activation function name ("relu" or "tanh")
        padding: Padding type for conv layers ("SAME" or "VALID")

    Returns:
        Tuple of (Categorical distribution, value estimate)
    """

    action_dim: int
    channel_sizes: Sequence[int] = (16, 32)
    kernel_sizes: Sequence[int] = (3, 3)
    hidden_size: int = 64
    actor_hidden_size: Optional[int] = None
    critic_hidden_size: Optional[int] = None
    activation: str = "relu"
    padding: str = "SAME"

    def setup(self):
        # Use hidden_size as default for head-specific sizes
        self._actor_hidden = self.actor_hidden_size or self.hidden_size
        self._critic_hidden = self.critic_hidden_size or self.hidden_size

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        # Select activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply convolutional layers
        for channels, kernel in zip(self.channel_sizes, self.kernel_sizes):
            if isinstance(kernel, int):
                kernel_size = (kernel, kernel)
            else:
                kernel_size = kernel

            x = nn.Conv(
                features=channels,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding=self.padding,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        # Flatten spatial dimensions
        x = x.reshape((x.shape[0], -1))

        # Shared embedding layer
        embedding = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        embedding = activation_fn(embedding)

        # Actor head
        actor_hidden = nn.Dense(
            self._actor_hidden,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
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
        )(embedding)
        critic_hidden = activation_fn(critic_hidden)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic_hidden)

        return pi, jnp.squeeze(value, axis=-1)


@register_network("cnn_small_encoder")
class CNNSmallEncoder(nn.Module):
    """CNN encoder that returns intermediate features.

    This network is similar to CNNSmall but returns intermediate representations
    that can be used for auxiliary tasks or transfer learning.

    Architecture:
        - Configurable Conv2D layers with specified channel and kernel sizes
        - No final flatten/dense layers (returns conv features)

    Attributes:
        channel_sizes: List of output channels for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        activation: Activation function name ("relu" or "tanh")
        padding: Padding type for conv layers ("SAME" or "VALID")

    Returns:
        Conv feature map
    """

    channel_sizes: Sequence[int] = (16, 32)
    kernel_sizes: Sequence[int] = (3, 3)
    activation: str = "relu"
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Select activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply convolutional layers
        for channels, kernel in zip(self.channel_sizes, self.kernel_sizes):
            if isinstance(kernel, int):
                kernel_size = (kernel, kernel)
            else:
                kernel_size = kernel

            x = nn.Conv(
                features=channels,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding=self.padding,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        return x


@register_network("cnn_impala")
class CNNImpala(nn.Module):
    """IMPALA-style CNN with residual blocks.

    This network follows the IMPALA architecture with residual blocks,
    suitable for more complex visual tasks.

    Architecture:
        - Conv2D (16 filters, 8x8, stride 4)
        - Residual block with 16 filters
        - Conv2D (32 filters, 4x4, stride 2)
        - Residual block with 32 filters
        - Dense (hidden_size)

    Attributes:
        hidden_size: Size of the final fully-connected layer
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Feature embedding of shape (batch_size, hidden_size)
    """

    hidden_size: int = 256
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

        def residual_block(x_in, features):
            """Residual block with two conv layers."""
            residual = x_in
            x = nn.relu(x_in)
            x = nn.Conv(
                features=features,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                features=features,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            return x + residual

        def conv_sequence(x_in, features, kernel_size, stride):
            """Conv sequence with max pooling."""
            x = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                strides=stride,
                padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x_in)
            x = residual_block(x, features)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            return x

        # IMPALA conv sequences
        x = conv_sequence(x, 16, (8, 8), (4, 4))
        x = conv_sequence(x, 32, (4, 4), (2, 2))

        # Flatten and FC
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation_fn(x)

        return x
