"""
Module: Causal Attention Mechanism for Counterfactual Regret
Equation: Attention(Q,K,V) with causal mask (Appendix enhancement)

The CausalRewardModel enhances the standard RewardModel by using multi-head
self-attention to model causal relationships between agents. This allows
the model to better capture how one agent's actions affect others' rewards.

Key Components:
1. CausalMultiHeadAttention: Multi-head attention with causal masking
2. CausalRewardModel: RewardModel with attention-based agent interactions

Reference: Counterfactual/cf_method (Appendix)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Optional, Tuple, Any
import numpy as np


def create_causal_mask(num_agents: int) -> jnp.ndarray:
    """
    Create a causal attention mask for agent interactions.

    In multi-agent settings, we want to model how earlier agents (in some
    ordering) can affect later agents' rewards. The causal mask ensures
    that attention can only flow from earlier to later agents (or bidirectional
    if using non-causal mode).

    Args:
        num_agents: Number of agents in the environment

    Returns:
        mask: Causal mask [num_agents, num_agents]
              mask[i,j] = 1 if agent i can attend to agent j
                         = 0 otherwise (masked out with large negative value)
    """
    # For causal attention: agent i can only attend to agents <= i
    # This creates a lower triangular mask
    mask = jnp.tril(jnp.ones((num_agents, num_agents)))
    return mask


def create_attention_mask_for_causal(
    num_agents: int,
    causal: bool = True
) -> jnp.ndarray:
    """
    Create attention mask for use in scaled dot-product attention.

    Args:
        num_agents: Number of agents
        causal: If True, use causal masking. If False, allow full attention.

    Returns:
        mask: Attention mask [1, 1, num_agents, num_agents]
              Values are 0 for valid attention, -inf for masked positions
    """
    if causal:
        # Causal mask: lower triangular (can attend to self and earlier)
        mask = jnp.tril(jnp.ones((num_agents, num_agents)))
        # Convert to additive mask: 0 for valid, large negative for masked
        additive_mask = jnp.where(mask == 1, 0.0, -1e9)
    else:
        # Full attention mask
        additive_mask = jnp.zeros((num_agents, num_agents))

    # Add batch and head dimensions [1, 1, num_agents, num_agents]
    additive_mask = additive_mask[jnp.newaxis, jnp.newaxis, :, :]

    return additive_mask


class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional causal masking.

    This module computes self-attention over agent embeddings, allowing
    the model to learn how agents influence each other's rewards.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dropout_rate: Dropout rate (0.0 for no dropout)
        causal: Whether to use causal masking
    """
    num_heads: int = 4
    head_dim: int = 16
    dropout_rate: float = 0.0
    causal: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute multi-head self-attention.

        Args:
            x: Input tensor [batch, num_agents, embed_dim]
            mask: Optional attention mask [batch, num_agents, num_agents]
                  or [num_agents, num_agents]
            deterministic: If True, disable dropout

        Returns:
            output: Attended output [batch, num_agents, embed_dim]
            attention_weights: Attention weights [batch, num_heads, num_agents, num_agents]
        """
        batch_size, num_agents, embed_dim = x.shape
        total_key_dim = self.num_heads * self.head_dim

        # Linear projections for Q, K, V
        # Query projection
        q = nn.Dense(
            total_key_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="query"
        )(x)
        q = q.reshape(batch_size, num_agents, self.num_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)  # [batch, num_heads, num_agents, head_dim]

        # Key projection
        k = nn.Dense(
            total_key_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="key"
        )(x)
        k = k.reshape(batch_size, num_agents, self.num_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)  # [batch, num_heads, num_agents, head_dim]

        # Value projection
        v = nn.Dense(
            total_key_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="value"
        )(x)
        v = v.reshape(batch_size, num_agents, self.num_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)  # [batch, num_heads, num_agents, head_dim]

        # Scaled dot-product attention
        # [batch, num_heads, num_agents, num_agents]
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attention_scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

        # Apply causal mask if needed
        if self.causal:
            causal_mask = create_attention_mask_for_causal(num_agents, causal=True)
            attention_scores = attention_scores + causal_mask

        # Apply custom mask if provided
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[jnp.newaxis, jnp.newaxis, :, :]
            elif mask.ndim == 3:
                mask = mask[:, jnp.newaxis, :, :]
            attention_scores = attention_scores + mask

        # Softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            attention_weights = nn.Dropout(self.dropout_rate)(
                attention_weights, deterministic=deterministic
            )

        # Apply attention to values
        # [batch, num_heads, num_agents, head_dim]
        attended = jnp.einsum("bhij,bhjd->bhid", attention_weights, v)

        # Reshape back
        attended = attended.transpose(0, 2, 1, 3)  # [batch, num_agents, num_heads, head_dim]
        attended = attended.reshape(batch_size, num_agents, -1)

        # Output projection
        output = nn.Dense(
            embed_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="output"
        )(attended)

        return output, attention_weights


class AgentFeatureExtractor(nn.Module):
    """
    Extracts features for each agent independently.

    This module processes each agent's observation and action to produce
    an embedding that will be used in the attention mechanism.

    Attributes:
        cnn_features: Number of features in each CNN layer
        cnn_kernels: Kernel sizes for each CNN layer
        hidden_dim: Hidden dimension for embedding
        action_dim: Dimension of discrete action space
        activation: Activation function
    """
    cnn_features: Sequence[int] = (32, 32, 32)
    cnn_kernels: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    action_dim: int = 4
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,  # [batch, num_agents, H, W, C]
        actions: jnp.ndarray,  # [batch, num_agents]
    ) -> jnp.ndarray:
        """
        Extract features for all agents.

        Args:
            obs: Joint observations [batch, num_agents, H, W, C]
            actions: Joint actions [batch, num_agents]

        Returns:
            embeddings: Agent embeddings [batch, num_agents, hidden_dim]
        """
        batch_size, num_agents = obs.shape[:2]

        # Get activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        # Reshape for CNN: [batch * num_agents, H, W, C]
        obs_flat = obs.reshape(-1, *obs.shape[2:])

        # CNN feature extraction
        x = obs_flat
        for features, kernel in zip(self.cnn_features, self.cnn_kernels):
            x = nn.Conv(
                features=features,
                kernel_size=kernel,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)

        # Flatten and project
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation_fn(x)

        # Reshape back: [batch, num_agents, hidden_dim]
        cnn_features = x.reshape(batch_size, num_agents, self.hidden_dim)

        # One-hot encode actions
        actions_onehot = nn.one_hot(actions, self.action_dim)  # [batch, num_agents, action_dim]

        # Project actions to same dimension
        action_embed = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="action_embedding"
        )(actions_onehot)

        # Combine CNN features and action embeddings
        embeddings = cnn_features + action_embed

        return embeddings


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward layers.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mlp_dim: Hidden dimension for MLP
        dropout_rate: Dropout rate
        causal: Whether to use causal masking
        activation: Activation function for MLP
    """
    num_heads: int = 4
    head_dim: int = 16
    mlp_dim: int = 128
    dropout_rate: float = 0.0
    causal: bool = True
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor [batch, num_agents, embed_dim]
            mask: Optional attention mask
            deterministic: If True, disable dropout

        Returns:
            output: Output tensor [batch, num_agents, embed_dim]
            attention_weights: Attention weights
        """
        embed_dim = x.shape[-1]

        # Self-attention with residual connection and LayerNorm
        normed = nn.LayerNorm()(x)
        attention = CausalMultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            causal=self.causal,
        )
        attn_out, attn_weights = attention(normed, mask, deterministic)

        # Dropout on attention output
        if not deterministic and self.dropout_rate > 0:
            attn_out = nn.Dropout(self.dropout_rate)(attn_out, deterministic=deterministic)

        x = x + attn_out  # Residual connection

        # Feed-forward with residual connection and LayerNorm
        normed = nn.LayerNorm()(x)

        # Get activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        ff_out = nn.Dense(
            self.mlp_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(normed)
        ff_out = activation_fn(ff_out)

        # Dropout on FF output
        if not deterministic and self.dropout_rate > 0:
            ff_out = nn.Dropout(self.dropout_rate)(ff_out, deterministic=deterministic)

        ff_out = nn.Dense(
            embed_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(ff_out)

        x = x + ff_out  # Residual connection

        return x, attn_weights


class CausalRewardModel(nn.Module):
    """
    Causal Reward Model with Multi-Head Self-Attention.

    This model enhances the standard RewardModel by using transformer-style
    self-attention to model causal relationships between agents. This allows
    the model to capture how one agent's actions influence others' rewards.

    Architecture:
    1. CNN extracts features from each agent's observation
    2. Action embeddings are added to observation features
    3. Multi-head self-attention over agent embeddings
    4. Feed-forward layers with residual connections
    5. Output projection to reward predictions

    Attributes:
        num_agents: Number of agents in the environment
        action_dim: Dimension of discrete action space
        cnn_features: Number of features in each CNN layer
        cnn_kernels: Kernel sizes for each CNN layer
        hidden_dim: Hidden dimension for embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_dim: Hidden dimension for MLP in transformer
        causal: Whether to use causal masking
        dropout_rate: Dropout rate
        activation: Activation function
    """
    num_agents: int
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32)
    cnn_kernels: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 128
    causal: bool = True
    dropout_rate: float = 0.0
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict rewards for all agents using causal attention.

        Args:
            obs: Joint observations [batch, num_agents, H, W, C]
            actions: Joint actions [batch, num_agents]
            deterministic: If True, disable dropout

        Returns:
            predicted_rewards: Predicted rewards [batch, num_agents]
            attention_weights: Final layer attention weights
                              [batch, num_heads, num_agents, num_agents]
        """
        # Extract agent features
        feature_extractor = AgentFeatureExtractor(
            cnn_features=self.cnn_features,
            cnn_kernels=self.cnn_kernels,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            activation=self.activation,
        )
        x = feature_extractor(obs, actions)  # [batch, num_agents, hidden_dim]

        # Initial layer norm
        x = nn.LayerNorm(name="input_layer_norm")(x)

        # Apply transformer blocks
        attention_weights = None
        for i in range(self.num_layers):
            block = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.hidden_dim // self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                causal=self.causal,
                activation=self.activation,
                name=f"transformer_block_{i}"
            )
            x, attention_weights = block(x, deterministic=deterministic)

        # Final layer norm
        x = nn.LayerNorm(name="output_layer_norm")(x)

        # Flatten across agents: [batch, num_agents * hidden_dim]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Get activation function
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        # MLP to aggregate across agents and predict rewards
        x = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="reward_mlp"
        )(x)
        x = activation_fn(x)

        # Final output: [batch, num_agents]
        rewards = nn.Dense(
            self.num_agents,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="reward_output"
        )(x)

        return rewards, attention_weights

    def get_attention_weights(
        self,
        params: dict,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Get attention weights for visualization/analysis.

        Args:
            params: Model parameters
            obs: Joint observations [batch, num_agents, H, W, C]
            actions: Joint actions [batch, num_agents]

        Returns:
            attention_weights: Final layer attention weights
                              [batch, num_heads, num_agents, num_agents]
        """
        _, attention_weights = self.apply(
            params, obs, actions,
            deterministic=True,
            capture_intermediates=True
        )
        return attention_weights


def compute_causal_reward_model_loss(
    params: dict,
    model: CausalRewardModel,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute the loss for CausalRewardModel.

    Args:
        params: Model parameters
        model: CausalRewardModel instance
        obs: Joint observations [batch, num_agents, H, W, C]
        actions: Joint actions [batch, num_agents]
        rewards: Actual rewards [batch, num_agents]
        mask: Optional mask [batch, num_agents]

    Returns:
        loss: Scalar MSE loss
        predicted_rewards: Predicted rewards [batch, num_agents]
        attention_weights: Attention weights for analysis
    """
    predicted_rewards, attention_weights = model.apply(
        params, obs, actions, deterministic=True
    )

    # Compute MSE loss
    squared_error = (predicted_rewards - rewards) ** 2

    if mask is not None:
        squared_error = squared_error * mask
        loss = jnp.sum(squared_error) / (jnp.sum(mask) + 1e-8)
    else:
        loss = jnp.mean(squared_error)

    return loss, predicted_rewards, attention_weights


def create_causal_reward_model_train_state(
    model: CausalRewardModel,
    rng: jax.random.PRNGKey,
    sample_obs: jnp.ndarray,
    sample_actions: jnp.ndarray,
    learning_rate: float = 0.001,
) -> Tuple[Any, jax.random.PRNGKey]:
    """
    Create training state for CausalRewardModel.

    Args:
        model: CausalRewardModel instance
        rng: JAX random key
        sample_obs: Sample observation for initialization [batch, num_agents, H, W, C]
        sample_actions: Sample actions for initialization [batch, num_agents]
        learning_rate: Learning rate for optimizer

    Returns:
        train_state: Flax TrainState with model params and optimizer
        rng: Remaining random key
    """
    from flax.training.train_state import TrainState
    import optax

    # Initialize model parameters
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, sample_obs, sample_actions, deterministic=True)

    # Create optimizer with weight decay
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=0.01)

    # Create train state
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    return train_state, rng


def verify_attention_weights(
    attention_weights: jnp.ndarray,
    eps: float = 1e-5
) -> Tuple[bool, str]:
    """
    Verify that attention weights satisfy expected properties.

    Args:
        attention_weights: Attention weights [batch, num_heads, num_agents, num_agents]
        eps: Tolerance for numerical checks

    Returns:
        is_valid: True if all checks pass
        message: Description of any issues found
    """
    # Check for NaN/Inf
    if jnp.any(jnp.isnan(attention_weights)):
        return False, "Attention weights contain NaN values"
    if jnp.any(jnp.isinf(attention_weights)):
        return False, "Attention weights contain Inf values"

    # Check that weights sum to 1 along the last dimension (query normalizes over keys)
    weight_sums = jnp.sum(attention_weights, axis=-1)
    if not jnp.allclose(weight_sums, 1.0, atol=eps):
        max_deviation = jnp.max(jnp.abs(weight_sums - 1.0))
        return False, f"Attention weights do not sum to 1 (max deviation: {max_deviation})"

    # Check that all weights are non-negative
    if jnp.any(attention_weights < -eps):
        return False, "Attention weights contain negative values"

    # Check that all weights are <= 1
    if jnp.any(attention_weights > 1.0 + eps):
        return False, "Attention weights exceed 1"

    return True, "All attention weight checks passed"


def get_attention_statistics(
    attention_weights: jnp.ndarray
) -> dict:
    """
    Compute statistics about attention weights for analysis.

    Args:
        attention_weights: Attention weights [batch, num_heads, num_agents, num_agents]

    Returns:
        stats: Dictionary of statistics
    """
    return {
        "mean": float(jnp.mean(attention_weights)),
        "std": float(jnp.std(attention_weights)),
        "min": float(jnp.min(attention_weights)),
        "max": float(jnp.max(attention_weights)),
        "entropy": float(-jnp.sum(
            attention_weights * jnp.log(attention_weights + 1e-10),
            axis=-1
        ).mean()),
    }
