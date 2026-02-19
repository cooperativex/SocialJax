"""Unit tests for SVO algorithm networks."""

import pytest
import sys

sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np

from socialjax.algorithms.svo.network import SVOCNN, SVOActorCritic
from socialjax.networks.registry import get_network_class, is_network_registered


class TestSVOCNNImport:
    """Tests for SVOCNN imports."""

    def test_import_svo_cnn(self):
        """Test SVOCNN can be imported."""
        assert SVOCNN is not None

    def test_import_svo_actor_critic(self):
        """Test SVOActorCritic can be imported."""
        assert SVOActorCritic is not None


class TestSVOCNNForward:
    """Tests for SVOCNN forward pass."""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(0)

    def test_forward_pass_shape(self, rng):
        """Test that forward pass produces correct output shape."""
        network = SVOCNN(activation="relu")
        # Input: (batch, height, width, channels)
        dummy_input = jnp.zeros((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)
        # Output shape should be (batch, 64) after flattening
        assert output.shape == (1, 64)

    def test_forward_pass_batch(self, rng):
        """Test forward pass with batch of observations."""
        network = SVOCNN(activation="relu")
        batch_size = 8
        dummy_input = jnp.zeros((batch_size, 11, 11, 19))
        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)
        assert output.shape == (batch_size, 64)

    def test_forward_pass_different_input_sizes(self, rng):
        """Test forward pass with different input sizes."""
        network = SVOCNN(activation="relu")
        for h, w, c in [(15, 15, 3), (11, 11, 19), (84, 84, 4)]:
            dummy_input = jnp.zeros((1, h, w, c))
            params = network.init(rng, dummy_input)
            output = network.apply(params, dummy_input)
            assert output.ndim == 2, f"Failed for input size {(h, w, c)}"

    def test_relu_activation(self, rng):
        """Test that relu activation works correctly."""
        network = SVOCNN(activation="relu")
        dummy_input = jnp.ones((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)
        # Output should have non-negative values after ReLU (before final dense)
        assert output.shape == (1, 64)

    def test_tanh_activation(self, rng):
        """Test that tanh activation works correctly."""
        network = SVOCNN(activation="tanh")
        dummy_input = jnp.ones((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)
        # Tanh outputs should be in [-1, 1]
        assert output.shape == (1, 64)


class TestSVOActorCriticForward:
    """Tests for SVOActorCritic forward pass."""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(0)

    def test_forward_pass_returns_tuple(self, rng):
        """Test that forward pass returns (pi, value) tuple."""
        action_dim = 8
        network = SVOActorCritic(action_dim=action_dim)
        dummy_input = jnp.zeros((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)
        assert len(output) == 2

    def test_forward_pass_value_shape(self, rng):
        """Test that value output has correct shape."""
        action_dim = 8
        network = SVOActorCritic(action_dim=action_dim)
        dummy_input = jnp.zeros((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        # Value should be scalar per batch element
        assert value.shape == (1,)

    def test_forward_pass_batch_values(self, rng):
        """Test forward pass with batch produces batch of values."""
        action_dim = 8
        batch_size = 16
        network = SVOActorCritic(action_dim=action_dim)
        dummy_input = jnp.zeros((batch_size, 11, 11, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        assert value.shape == (batch_size,)

    def test_policy_distribution(self, rng):
        """Test that policy is a valid Categorical distribution."""
        action_dim = 8
        network = SVOActorCritic(action_dim=action_dim)
        dummy_input = jnp.zeros((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)

        # Test sampling
        key = jax.random.PRNGKey(1)
        action = pi.sample(seed=key)
        assert action.shape == (1,)

        # Test log_prob
        log_prob = pi.log_prob(action)
        assert log_prob.shape == (1,)

    def test_policy_logits_shape(self, rng):
        """Test that policy logits have correct shape."""
        action_dim = 8
        network = SVOActorCritic(action_dim=action_dim)
        dummy_input = jnp.zeros((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        assert pi.logits.shape == (1, action_dim)

    def test_different_action_dims(self, rng):
        """Test network with different action dimensions."""
        for action_dim in [2, 4, 8, 16]:
            network = SVOActorCritic(action_dim=action_dim)
            dummy_input = jnp.zeros((1, 11, 11, 19))
            params = network.init(rng, dummy_input)
            pi, value = network.apply(params, dummy_input)
            assert pi.logits.shape == (1, action_dim), f"Failed for action_dim={action_dim}"

    def test_different_hidden_sizes(self, rng):
        """Test network with different hidden layer sizes."""
        action_dim = 8
        for hidden_size in [32, 64, 128, 256]:
            network = SVOActorCritic(action_dim=action_dim, hidden_size=hidden_size)
            dummy_input = jnp.zeros((1, 11, 11, 19))
            params = network.init(rng, dummy_input)
            pi, value = network.apply(params, dummy_input)
            assert value.shape == (1,), f"Failed for hidden_size={hidden_size}"


class TestSVOActorCriticRegistration:
    """Tests for SVOActorCritic network registration."""

    def test_network_registered(self):
        """Test that SVOActorCritic is registered in the network registry."""
        assert is_network_registered("svo_actor_critic")

    def test_get_network_class(self):
        """Test that SVOActorCritic can be retrieved from registry."""
        network_class = get_network_class("svo_actor_critic")
        assert network_class == SVOActorCritic

    def test_registered_network_works(self):
        """Test that registered network class works correctly."""
        NetworkClass = get_network_class("svo_actor_critic")
        network = NetworkClass(action_dim=8)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        assert pi.logits.shape == (1, 8)


class TestSVOActorCriticDeterministic:
    """Tests for deterministic behavior of SVOActorCritic."""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def test_deterministic_forward(self, rng):
        """Test that forward pass is deterministic with same parameters."""
        network = SVOActorCritic(action_dim=8)
        dummy_input = jnp.ones((1, 11, 11, 19))
        params = network.init(rng, dummy_input)

        output1 = network.apply(params, dummy_input)
        output2 = network.apply(params, dummy_input)

        pi1, value1 = output1
        pi2, value2 = output2

        np.testing.assert_array_equal(pi1.logits, pi2.logits)
        np.testing.assert_array_equal(value1, value2)

    def test_same_init_same_output(self, rng):
        """Test that same init produces same outputs."""
        network1 = SVOActorCritic(action_dim=8)
        network2 = SVOActorCritic(action_dim=8)

        dummy_input = jnp.ones((1, 11, 11, 19))
        params1 = network1.init(rng, dummy_input)
        params2 = network2.init(rng, dummy_input)

        pi1, value1 = network1.apply(params1, dummy_input)
        pi2, value2 = network2.apply(params2, dummy_input)

        np.testing.assert_array_equal(pi1.logits, pi2.logits)
        np.testing.assert_array_equal(value1, value2)


class TestSVONetworkEdgeCases:
    """Tests for edge cases in SVO networks."""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(0)

    def test_single_action(self, rng):
        """Test network with single action (binary)."""
        network = SVOActorCritic(action_dim=2)
        dummy_input = jnp.zeros((1, 11, 11, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        assert pi.logits.shape == (1, 2)

    def test_large_batch(self, rng):
        """Test network with large batch size."""
        network = SVOActorCritic(action_dim=8)
        batch_size = 128
        dummy_input = jnp.zeros((batch_size, 11, 11, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        assert value.shape == (batch_size,)

    def test_non_square_input(self, rng):
        """Test network with non-square input."""
        network = SVOActorCritic(action_dim=8)
        # 15x10 input
        dummy_input = jnp.zeros((1, 15, 10, 19))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        assert value.shape == (1,)

    def test_single_channel_input(self, rng):
        """Test network with single channel input."""
        network = SVOActorCritic(action_dim=8)
        dummy_input = jnp.zeros((1, 11, 11, 1))
        params = network.init(rng, dummy_input)
        pi, value = network.apply(params, dummy_input)
        assert value.shape == (1,)
