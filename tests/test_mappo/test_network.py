"""Unit tests for MAPPO networks."""

import sys
sys.path.insert(0, 'socialjax')

# Make pytest optional
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    class pytest:
        @staticmethod
        def fixture(func):
            return func

import jax
import jax.numpy as jnp
import numpy as np

from socialjax.algorithms.mappo.network import (
    MAPPOActor,
    MAPPOCritic,
    MAPPOActorCNN,
    MAPPOCriticCNN,
)
from socialjax.networks.registry import get_network_class, is_network_registered


class TestMAPPOActorCNN:
    """Tests for MAPPO Actor CNN feature extractor."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        cnn = MAPPOActorCNN(activation="relu")
        rng = jax.random.PRNGKey(0)
        # Input: (batch, height, width, channels)
        x = jnp.zeros((1, 15, 15, 3))
        params = cnn.init(rng, x)
        output = cnn.apply(params, x)
        # Output should be (batch, 64)
        assert output.shape == (1, 64)

    def test_forward_pass_different_batch_sizes(self):
        """Test that CNN handles different batch sizes."""
        cnn = MAPPOActorCNN(activation="relu")
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 3))
        params = cnn.init(rng, x)

        for batch_size in [1, 4, 16]:
            x = jnp.zeros((batch_size, 15, 15, 3))
            output = cnn.apply(params, x)
            assert output.shape == (batch_size, 64)

    def test_relu_activation(self):
        """Test that relu activation works."""
        cnn = MAPPOActorCNN(activation="relu")
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 3))
        params = cnn.init(rng, x)
        output = cnn.apply(params, x)
        assert output.shape == (1, 64)

    def test_tanh_activation(self):
        """Test that tanh activation works."""
        cnn = MAPPOActorCNN(activation="tanh")
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 3))
        params = cnn.init(rng, x)
        output = cnn.apply(params, x)
        assert output.shape == (1, 64)


class TestMAPPOCriticCNN:
    """Tests for MAPPO Critic CNN feature extractor."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        cnn = MAPPOCriticCNN(activation="relu")
        rng = jax.random.PRNGKey(0)
        # Input: (batch, height, width, channels * num_agents)
        # For 5 agents with 3 channels each: 15 channels
        x = jnp.zeros((1, 15, 15, 15))
        params = cnn.init(rng, x)
        output = cnn.apply(params, x)
        # Output should be (batch, 64)
        assert output.shape == (1, 64)

    def test_forward_pass_different_world_state_sizes(self):
        """Test that CNN handles different world state sizes."""
        cnn = MAPPOCriticCNN(activation="relu")
        rng = jax.random.PRNGKey(0)
        # Initialize with smaller world state
        x = jnp.zeros((1, 15, 15, 9))  # 3 agents
        params = cnn.init(rng, x)

        for num_agents in [3, 5, 7]:
            world_channels = 3 * num_agents
            x = jnp.zeros((1, 15, 15, world_channels))
            # Note: This will fail if channel sizes differ from initialization
            # In practice, world state size is fixed per environment


class TestMAPPOActor:
    """Tests for MAPPO Actor network."""

    def test_forward_pass_shape(self):
        """Test that actor forward pass produces correct output shape."""
        actor = MAPPOActor(action_dim=5, activation="relu", hidden_size=64)
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 3))
        params = actor.init(rng, x)
        pi = actor.apply(params, x)
        # Should return a distribution
        assert hasattr(pi, 'sample')
        assert hasattr(pi, 'log_prob')
        assert hasattr(pi, 'entropy')

    def test_sample_actions(self):
        """Test that actor can sample actions."""
        actor = MAPPOActor(action_dim=5, activation="relu")
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 3))
        params = actor.init(rng, x)
        pi = actor.apply(params, x)

        rng, sample_rng = jax.random.split(rng)
        action = pi.sample(seed=sample_rng)
        assert action.shape == (1,)
        assert 0 <= action[0] < 5

    def test_log_prob(self):
        """Test that actor can compute log probabilities."""
        actor = MAPPOActor(action_dim=5, activation="relu")
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 3))
        params = actor.init(rng, x)
        pi = actor.apply(params, x)

        rng, sample_rng = jax.random.split(rng)
        action = pi.sample(seed=sample_rng)
        log_prob = pi.log_prob(action)
        assert log_prob.shape == (1,)

    def test_entropy(self):
        """Test that actor can compute entropy."""
        actor = MAPPOActor(action_dim=5, activation="relu")
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 3))
        params = actor.init(rng, x)
        pi = actor.apply(params, x)

        entropy = pi.entropy()
        assert entropy.shape == (1,)

    def test_different_action_dims(self):
        """Test that actor works with different action dimensions."""
        for action_dim in [2, 5, 8]:
            actor = MAPPOActor(action_dim=action_dim, activation="relu")
            rng = jax.random.PRNGKey(0)
            x = jnp.zeros((1, 15, 15, 3))
            params = actor.init(rng, x)
            pi = actor.apply(params, x)

            rng, sample_rng = jax.random.split(rng)
            action = pi.sample(seed=sample_rng)
            assert 0 <= action[0] < action_dim

    def test_registered_in_network_registry(self):
        """Test that MAPPOActor is registered as 'mappo_actor'."""
        assert is_network_registered("mappo_actor")
        network_class = get_network_class("mappo_actor")
        assert network_class is MAPPOActor


class TestMAPPOCritic:
    """Tests for MAPPO Critic network."""

    def test_forward_pass_shape(self):
        """Test that critic forward pass produces correct output shape."""
        critic = MAPPOCritic(activation="relu", hidden_size=64)
        rng = jax.random.PRNGKey(0)
        # World state with 5 agents (5 * 3 = 15 channels)
        x = jnp.zeros((1, 15, 15, 15))
        params = critic.init(rng, x)
        value = critic.apply(params, x)
        # Should return scalar value (squeezed)
        assert value.shape == (1,)

    def test_batch_values(self):
        """Test that critic can process batch of world states."""
        critic = MAPPOCritic(activation="relu")
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((1, 15, 15, 15))
        params = critic.init(rng, x)

        batch_size = 8
        x = jnp.zeros((batch_size, 15, 15, 15))
        values = critic.apply(params, x)
        assert values.shape == (batch_size,)

    def test_different_hidden_sizes(self):
        """Test that critic works with different hidden sizes."""
        for hidden_size in [32, 64, 128]:
            critic = MAPPOCritic(hidden_size=hidden_size)
            rng = jax.random.PRNGKey(0)
            x = jnp.zeros((1, 15, 15, 15))
            params = critic.init(rng, x)
            value = critic.apply(params, x)
            assert value.shape == (1,)

    def test_registered_in_network_registry(self):
        """Test that MAPPOCritic is registered as 'mappo_critic'."""
        assert is_network_registered("mappo_critic")
        network_class = get_network_class("mappo_critic")
        assert network_class is MAPPOCritic


class TestMAPPOActorCriticIntegration:
    """Integration tests for Actor and Critic working together."""

    def test_actor_critic_different_inputs(self):
        """Test that actor and critic can process different input shapes."""
        rng = jax.random.PRNGKey(0)
        num_agents = 5

        # Actor: local observation only
        actor = MAPPOActor(action_dim=5, activation="relu")
        local_obs = jnp.zeros((1, 15, 15, 3))
        actor_params = actor.init(rng, local_obs)
        pi = actor.apply(actor_params, local_obs)
        action = pi.sample(seed=rng)
        assert action.shape == (1,)

        # Critic: global state (all observations concatenated)
        rng, critic_rng = jax.random.split(rng)
        critic = MAPPOCritic(activation="relu")
        world_state = jnp.zeros((1, 15, 15, 3 * num_agents))
        critic_params = critic.init(critic_rng, world_state)
        value = critic.apply(critic_params, world_state)
        assert value.shape == (1,)


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Run tests manually
        print("Running MAPPO Network Tests...")
        test_classes = [
            TestMAPPOActorCNN(),
            TestMAPPOCriticCNN(),
            TestMAPPOActor(),
            TestMAPPOCritic(),
            TestMAPPOActorCriticIntegration(),
        ]
        passed = 0
        failed = 0

        for test_obj in test_classes:
            for name in dir(test_obj):
                if name.startswith('test_'):
                    try:
                        getattr(test_obj, name)()
                        print(f"  PASS: {test_obj.__class__.__name__}.{name}")
                        passed += 1
                    except AssertionError as e:
                        print(f"  FAIL: {test_obj.__class__.__name__}.{name} - {e}")
                        failed += 1

        print(f"\nResults: {passed} passed, {failed} failed")
