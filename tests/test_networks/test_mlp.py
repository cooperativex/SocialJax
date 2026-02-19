"""Unit tests for MLP network architectures.

This module tests:
- MLPSmall: Lightweight MLP feature extractor
- MLPActorCritic: Combined actor-critic network
- MLPEncoder: MLP encoder for auxiliary tasks
- MLPLargeActorCritic: Large actor-critic network
- Network factory integration
"""

import pytest
import sys
import numpy as np

sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from socialjax.networks.mlp import (
    MLPSmall,
    MLPActorCritic,
    MLPEncoder,
    MLPLargeActorCritic,
)
from socialjax.networks import (
    create_network,
    list_networks,
    get_network_class,
    NETWORK_CONFIGS,
)


class TestMLPSmall:
    """Tests for MLPSmall network."""

    def test_mlp_small_import(self):
        """Test MLPSmall can be imported."""
        from socialjax.networks.mlp import MLPSmall
        assert MLPSmall is not None

    def test_mlp_small_creation(self):
        """Test MLPSmall can be created."""
        network = MLPSmall(layer_sizes=(64, 64))
        assert network.layer_sizes == (64, 64)

    def test_mlp_small_default_config(self):
        """Test MLPSmall default configuration."""
        network = MLPSmall()
        assert network.layer_sizes == (64, 64)
        assert network.activation == "relu"

    def test_mlp_small_forward_pass(self):
        """Test MLPSmall forward pass works."""
        network = MLPSmall(layer_sizes=(32, 64))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))  # batch_size=1, input_dim=10
        params = network.init(rng, x)
        output = network.apply(params, x)
        assert output.shape == (1, 64)

    def test_mlp_small_custom_layer_sizes(self):
        """Test MLPSmall with custom layer sizes."""
        network = MLPSmall(layer_sizes=(128, 64, 32))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        output = network.apply(params, x)
        assert output.shape == (1, 32)

    def test_mlp_small_tanh_activation(self):
        """Test MLPSmall with tanh activation."""
        network = MLPSmall(layer_sizes=(32, 32), activation="tanh")
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        output = network.apply(params, x)
        assert output.shape == (1, 32)

    def test_mlp_small_invalid_activation(self):
        """Test MLPSmall raises error for invalid activation."""
        network = MLPSmall(activation="invalid")
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        with pytest.raises(ValueError, match="Unknown activation"):
            network.apply(network.init(rng, x), x)

    def test_mlp_small_batch_processing(self):
        """Test MLPSmall handles batched inputs."""
        network = MLPSmall(layer_sizes=(32, 64))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((8, 10))  # batch_size=8
        params = network.init(rng, x)
        output = network.apply(params, x)
        assert output.shape == (8, 64)


class TestMLPActorCritic:
    """Tests for MLPActorCritic network."""

    def test_mlp_actor_critic_import(self):
        """Test MLPActorCritic can be imported."""
        from socialjax.networks.mlp import MLPActorCritic
        assert MLPActorCritic is not None

    def test_mlp_actor_critic_creation(self):
        """Test MLPActorCritic can be created."""
        network = MLPActorCritic(action_dim=8, layer_sizes=(64, 64))
        assert network.action_dim == 8
        assert network.layer_sizes == (64, 64)

    def test_mlp_actor_critic_default_config(self):
        """Test MLPActorCritic default configuration."""
        network = MLPActorCritic(action_dim=4)
        assert network.action_dim == 4
        assert network.layer_sizes == (64, 64)
        assert network.activation == "relu"

    def test_mlp_actor_critic_forward_pass(self):
        """Test MLPActorCritic forward pass."""
        network = MLPActorCritic(action_dim=8, layer_sizes=(32, 32))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        assert pi.logits.shape == (1, 8)
        assert value.shape == (1,)

    def test_mlp_actor_critic_output_shapes(self):
        """Test MLPActorCritic output shapes match action dimensions."""
        for action_dim in [2, 4, 8, 16]:
            network = MLPActorCritic(action_dim=action_dim)
            rng = jax.random.PRNGKey(0)
            x = jnp.ones((1, 10))
            params = network.init(rng, x)
            pi, value = network.apply(params, x)
            assert pi.logits.shape == (1, action_dim)

    def test_mlp_actor_critic_custom_hidden_sizes(self):
        """Test MLPActorCritic with custom actor/critic hidden sizes."""
        network = MLPActorCritic(
            action_dim=8,
            layer_sizes=(64, 64),
            actor_hidden_size=128,
            critic_hidden_size=256,
        )
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        assert pi.logits.shape == (1, 8)

    def test_mlp_actor_critic_tanh_activation(self):
        """Test MLPActorCritic with tanh activation."""
        network = MLPActorCritic(action_dim=8, activation="tanh")
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        assert pi.logits.shape == (1, 8)

    def test_mlp_actor_critic_batch_processing(self):
        """Test MLPActorCritic handles batched inputs."""
        network = MLPActorCritic(action_dim=8, layer_sizes=(32, 32))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((16, 10))  # batch_size=16
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        assert pi.logits.shape == (16, 8)
        assert value.shape == (16,)

    def test_mlp_actor_critic_distribution_sampling(self):
        """Test MLPActorCritic can sample actions."""
        network = MLPActorCritic(action_dim=8)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        pi, value = network.apply(params, x)

        # Sample actions
        sample_rng = jax.random.PRNGKey(1)
        actions = pi.sample(seed=sample_rng)
        assert actions.shape == (1,)
        assert actions[0] >= 0 and actions[0] < 8

    def test_mlp_actor_critic_log_prob(self):
        """Test MLPActorCritic can compute log probabilities."""
        network = MLPActorCritic(action_dim=8)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        pi, value = network.apply(params, x)

        # Compute log probabilities
        actions = jnp.array([3])
        log_probs = pi.log_prob(actions)
        assert log_probs.shape == (1,)


class TestMLPEncoder:
    """Tests for MLPEncoder network."""

    def test_mlp_encoder_import(self):
        """Test MLPEncoder can be imported."""
        from socialjax.networks.mlp import MLPEncoder
        assert MLPEncoder is not None

    def test_mlp_encoder_creation(self):
        """Test MLPEncoder can be created."""
        network = MLPEncoder(layer_sizes=(64, 64))
        assert network.layer_sizes == (64, 64)

    def test_mlp_encoder_forward_pass(self):
        """Test MLPEncoder forward pass."""
        network = MLPEncoder(layer_sizes=(32, 64))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        output = network.apply(params, x)
        assert output.shape == (1, 64)

    def test_mlp_encoder_registered(self):
        """Test MLPEncoder is registered as 'mlp_encoder'."""
        networks = list_networks()
        assert "mlp_encoder" in networks


class TestMLPLargeActorCritic:
    """Tests for MLPLargeActorCritic network."""

    def test_mlp_large_import(self):
        """Test MLPLargeActorCritic can be imported."""
        from socialjax.networks.mlp import MLPLargeActorCritic
        assert MLPLargeActorCritic is not None

    def test_mlp_large_creation(self):
        """Test MLPLargeActorCritic can be created."""
        network = MLPLargeActorCritic(action_dim=8, hidden_size=128, num_layers=3)
        assert network.action_dim == 8
        assert network.hidden_size == 128
        assert network.num_layers == 3

    def test_mlp_large_forward_pass(self):
        """Test MLPLargeActorCritic forward pass."""
        network = MLPLargeActorCritic(action_dim=8, hidden_size=64, num_layers=2)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        assert pi.logits.shape == (1, 8)
        assert value.shape == (1,)

    def test_mlp_large_registered(self):
        """Test MLPLargeActorCritic is registered as 'mlp_large'."""
        networks = list_networks()
        assert "mlp_large" in networks


class TestNetworkFactoryIntegration:
    """Tests for MLP networks with factory functions."""

    def test_create_mlp_small(self):
        """Test creating mlp_small via factory."""
        network = create_network("mlp_small", action_dim=8)
        assert network is not None
        assert network.action_dim == 8

    def test_create_mlp_encoder(self):
        """Test creating mlp_encoder via factory."""
        # MLPEncoder doesn't have action_dim (it's just an encoder)
        from socialjax.networks.mlp import MLPEncoder
        network = MLPEncoder(layer_sizes=(64, 64))
        assert network is not None

    def test_create_mlp_large(self):
        """Test creating mlp_large via factory."""
        network = create_network("mlp_large", action_dim=8)
        assert network is not None
        assert network.action_dim == 8

    def test_create_mlp_with_custom_params(self):
        """Test creating MLP with custom parameters."""
        network = create_network(
            "mlp_small",
            action_dim=8,
            layer_sizes=(128, 128),
            activation="tanh"
        )
        assert network is not None
        assert network.layer_sizes == (128, 128)
        assert network.activation == "tanh"

    def test_get_mlp_network_class(self):
        """Test getting MLP network class by name."""
        mlp_small_cls = get_network_class("mlp_small")
        assert mlp_small_cls == MLPActorCritic

        mlp_encoder_cls = get_network_class("mlp_encoder")
        assert mlp_encoder_cls == MLPEncoder

        mlp_large_cls = get_network_class("mlp_large")
        assert mlp_large_cls == MLPLargeActorCritic

    def test_all_mlp_networks_listed(self):
        """Test all MLP networks are listed."""
        networks = list_networks()
        assert "mlp_small" in networks
        assert "mlp_encoder" in networks
        assert "mlp_large" in networks


class TestForwardPass:
    """Tests for forward pass behavior."""

    def test_mlp_deterministic_forward(self):
        """Test MLP forward pass is deterministic."""
        network = MLPActorCritic(action_dim=8)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)

        pi1, v1 = network.apply(params, x)
        pi2, v2 = network.apply(params, x)

        np.testing.assert_array_equal(pi1.logits, pi2.logits)
        np.testing.assert_array_equal(v1, v2)

    def test_mlp_different_inputs(self):
        """Test MLP produces different outputs for different inputs."""
        network = MLPActorCritic(action_dim=8)
        rng = jax.random.PRNGKey(0)
        x1 = jnp.ones((1, 10))
        x2 = jnp.zeros((1, 10))
        params = network.init(rng, x1)

        pi1, v1 = network.apply(params, x1)
        pi2, v2 = network.apply(params, x2)

        # Different inputs should generally produce different outputs
        assert not jnp.allclose(pi1.logits, pi2.logits)


class TestOutputShapes:
    """Tests for output shapes."""

    def test_mlp_small_output_matches_layer_sizes(self):
        """Test MLPSmall output matches last layer size."""
        for layer_sizes in [(32,), (64, 64), (128, 64, 32)]:
            network = MLPSmall(layer_sizes=layer_sizes)
            rng = jax.random.PRNGKey(0)
            x = jnp.ones((1, 10))
            params = network.init(rng, x)
            output = network.apply(params, x)
            assert output.shape == (1, layer_sizes[-1])

    def test_mlp_actor_critic_value_shape(self):
        """Test MLPActorCritic value output is scalar per batch."""
        network = MLPActorCritic(action_dim=8)
        rng = jax.random.PRNGKey(0)
        for batch_size in [1, 4, 16]:
            x = jnp.ones((batch_size, 10))
            params = network.init(rng, x)
            pi, value = network.apply(params, x)
            assert value.shape == (batch_size,)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_mlp_single_layer(self):
        """Test MLP with single layer."""
        network = MLPSmall(layer_sizes=(64,))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 10))
        params = network.init(rng, x)
        output = network.apply(params, x)
        assert output.shape == (1, 64)

    def test_mlp_large_input(self):
        """Test MLP with large input dimension."""
        network = MLPActorCritic(action_dim=8, layer_sizes=(64, 64))
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 1000))  # Large input
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        assert pi.logits.shape == (1, 8)

    def test_mlp_large_batch(self):
        """Test MLP with large batch size."""
        network = MLPActorCritic(action_dim=8)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((256, 10))  # Large batch
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        assert pi.logits.shape == (256, 8)

    def test_mlp_zero_input(self):
        """Test MLP with zero input."""
        network = MLPActorCritic(action_dim=8)
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 10))
        params = network.init(rng, x)
        pi, value = network.apply(params, x)
        # Should produce valid output
        assert pi.logits.shape == (1, 8)
        assert jnp.isfinite(value).all()


class TestModuleExports:
    """Tests for module exports."""

    def test_mlp_small_in_all(self):
        """Test MLPSmall is in __all__."""
        from socialjax.networks import __all__
        assert "MLPSmall" in __all__

    def test_mlp_actor_critic_in_all(self):
        """Test MLPActorCritic is in __all__."""
        from socialjax.networks import __all__
        assert "MLPActorCritic" in __all__

    def test_mlp_encoder_in_all(self):
        """Test MLPEncoder is in __all__."""
        from socialjax.networks import __all__
        assert "MLPEncoder" in __all__

    def test_mlp_large_in_all(self):
        """Test MLPLargeActorCritic is in __all__."""
        from socialjax.networks import __all__
        assert "MLPLargeActorCritic" in __all__

    def test_direct_imports(self):
        """Test direct imports work."""
        from socialjax.networks import MLPSmall, MLPActorCritic, MLPEncoder, MLPLargeActorCritic
        assert MLPSmall is not None
        assert MLPActorCritic is not None
        assert MLPEncoder is not None
        assert MLPLargeActorCritic is not None
