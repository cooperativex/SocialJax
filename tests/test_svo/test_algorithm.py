"""Unit tests for SVO algorithm implementation."""

import pytest
import sys

sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np

from socialjax.algorithms.svo import (
    SVOAlgorithm,
    SVOAlgorithmState,
    get_svo_config,
    compute_svo_reward,
    compute_batch_svo_reward,
)
from socialjax.algorithms import get_algorithm, is_algorithm_registered
from socialjax.core.base_algorithm import BaseAlgorithm


class TestSVOAlgorithmImport:
    """Tests for SVO algorithm imports."""

    def test_import_svo_algorithm(self):
        """Test SVOAlgorithm can be imported."""
        assert SVOAlgorithm is not None

    def test_import_svo_algorithm_state(self):
        """Test SVOAlgorithmState can be imported."""
        assert SVOAlgorithmState is not None

    def test_import_compute_svo_reward(self):
        """Test compute_svo_reward can be imported."""
        assert callable(compute_svo_reward)

    def test_import_compute_batch_svo_reward(self):
        """Test compute_batch_svo_reward can be imported."""
        assert callable(compute_batch_svo_reward)


class TestSVOAlgorithmRegistration:
    """Tests for SVO algorithm registration."""

    def test_algorithm_registered(self):
        """Test that SVO is registered in the algorithm registry."""
        assert is_algorithm_registered("svo")

    def test_get_algorithm_returns_svo(self):
        """Test that get_algorithm returns SVOAlgorithm class."""
        algo_class = get_algorithm("svo")
        assert algo_class == SVOAlgorithm

    def test_algorithm_inherits_from_base(self):
        """Test that SVOAlgorithm inherits from BaseAlgorithm."""
        assert issubclass(SVOAlgorithm, BaseAlgorithm)


class TestSVOAlgorithmInit:
    """Tests for SVO algorithm initialization."""

    @pytest.fixture
    def observation_space(self):
        """Create a mock observation space."""
        class MockSpace:
            shape = (11, 11, 19)
        return MockSpace()

    @pytest.fixture
    def action_space(self):
        """Create a mock action space."""
        class MockSpace:
            n = 8
        return MockSpace()

    def test_init_with_defaults(self, observation_space, action_space):
        """Test initialization with default configuration."""
        algo = SVOAlgorithm(observation_space, action_space)
        assert algo is not None
        assert algo.config is not None

    def test_init_with_custom_config(self, observation_space, action_space):
        """Test initialization with custom configuration."""
        config = {"LR": 1e-3, "SVO_ANGLE": 30.0}
        algo = SVOAlgorithm(observation_space, action_space, config)
        assert algo.config["LR"] == 1e-3
        assert algo.config["SVO_ANGLE"] == 30.0

    def test_init_creates_network(self, observation_space, action_space):
        """Test that initialization creates a network."""
        algo = SVOAlgorithm(observation_space, action_space)
        assert algo.network is not None

    def test_init_creates_optimizer(self, observation_space, action_space):
        """Test that initialization creates an optimizer."""
        algo = SVOAlgorithm(observation_space, action_space)
        assert algo.optimizer is not None


class TestSVOAlgorithmInitState:
    """Tests for SVO algorithm state initialization."""

    @pytest.fixture
    def algorithm(self):
        """Create an SVO algorithm instance."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        return SVOAlgorithm(MockObsSpace(), MockActSpace())

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(0)

    def test_init_state_returns_state(self, algorithm, rng):
        """Test that init_state returns SVOAlgorithmState."""
        state = algorithm.init_state(rng)
        assert isinstance(state, SVOAlgorithmState)

    def test_init_state_has_params(self, algorithm, rng):
        """Test that state contains network parameters."""
        state = algorithm.init_state(rng)
        assert state.params is not None

    def test_init_state_has_optimizer_state(self, algorithm, rng):
        """Test that state contains optimizer state."""
        state = algorithm.init_state(rng)
        assert state.optimizer_state is not None

    def test_init_state_has_rng(self, algorithm, rng):
        """Test that state contains random key."""
        state = algorithm.init_state(rng)
        assert state.rng is not None

    def test_init_state_timestep_zero(self, algorithm, rng):
        """Test that state starts with timestep 0."""
        state = algorithm.init_state(rng)
        assert state.timestep == 0

    def test_init_state_default_svo_angle(self, algorithm, rng):
        """Test that state has default SVO angle."""
        state = algorithm.init_state(rng)
        assert state.svo_angle == 45.0


class TestSVOAlgorithmComputeAction:
    """Tests for SVO algorithm compute_action method."""

    @pytest.fixture
    def algorithm_and_state(self):
        """Create algorithm and initialized state."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        algo = SVOAlgorithm(MockObsSpace(), MockActSpace())
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)
        return algo, state

    def test_compute_action_returns_action(self, algorithm_and_state):
        """Test that compute_action returns an action."""
        algo, state = algorithm_and_state
        observation = jnp.zeros((11, 11, 19))
        rng = jax.random.PRNGKey(1)
        action, info = algo.compute_action(state, observation, rng)
        assert action is not None
        assert isinstance(action, (int, np.ndarray, jnp.ndarray))

    def test_compute_action_returns_info(self, algorithm_and_state):
        """Test that compute_action returns info dict."""
        algo, state = algorithm_and_state
        observation = jnp.zeros((11, 11, 19))
        rng = jax.random.PRNGKey(1)
        action, info = algo.compute_action(state, observation, rng)
        assert "log_prob" in info
        assert "value" in info

    def test_compute_action_deterministic(self, algorithm_and_state):
        """Test deterministic action selection."""
        algo, state = algorithm_and_state
        observation = jnp.zeros((11, 11, 19))
        rng = jax.random.PRNGKey(1)
        action, info = algo.compute_action(state, observation, rng, deterministic=True)
        # Deterministic should give same action for same observation
        assert action is not None

    def test_compute_action_batch(self, algorithm_and_state):
        """Test compute_action with batch of observations."""
        algo, state = algorithm_and_state
        batch_size = 4
        observations = jnp.zeros((batch_size, 11, 11, 19))
        rng = jax.random.PRNGKey(1)
        actions, info = algo.compute_action(state, observations, rng)
        assert actions.shape[0] == batch_size


class TestSVOAlgorithmComputeValue:
    """Tests for SVO algorithm compute_value method."""

    @pytest.fixture
    def algorithm_and_state(self):
        """Create algorithm and initialized state."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        algo = SVOAlgorithm(MockObsSpace(), MockActSpace())
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)
        return algo, state

    def test_compute_value_returns_value(self, algorithm_and_state):
        """Test that compute_value returns a value."""
        algo, state = algorithm_and_state
        observation = jnp.zeros((11, 11, 19))
        value = algo.compute_value(state, observation)
        assert value is not None

    def test_compute_value_shape(self, algorithm_and_state):
        """Test that compute_value returns scalar for single obs."""
        algo, state = algorithm_and_state
        observation = jnp.zeros((11, 11, 19))
        value = algo.compute_value(state, observation)
        # Should be scalar
        assert value.ndim == 0

    def test_compute_value_batch_shape(self, algorithm_and_state):
        """Test compute_value with batch of observations."""
        algo, state = algorithm_and_state
        batch_size = 4
        observations = jnp.zeros((batch_size, 11, 11, 19))
        values = algo.compute_value(state, observations)
        assert values.shape == (batch_size,)


class TestSVORewardComputation:
    """Tests for SVO reward transformation."""

    def test_compute_svo_reward_basic(self):
        """Test basic SVO reward computation."""
        rewards = jnp.array([1.0, 2.0, 3.0])
        svo_rewards = compute_svo_reward(rewards, svo_angle=45.0)
        assert svo_rewards.shape == rewards.shape

    def test_compute_svo_reward_selfish(self):
        """Test that 0 degree angle gives selfish rewards."""
        rewards = jnp.array([1.0, 2.0, 3.0])
        svo_rewards = compute_svo_reward(rewards, svo_angle=0.0, use_fairness=False)
        # With angle=0, svo_rewards should equal original rewards
        np.testing.assert_array_almost_equal(svo_rewards, rewards)

    def test_compute_svo_reward_altruistic(self):
        """Test that 90 degree angle gives altruistic rewards."""
        rewards = jnp.array([1.0, 2.0, 3.0])
        svo_rewards = compute_svo_reward(rewards, svo_angle=90.0, use_fairness=False)
        # With angle=90, svo_rewards should equal mean of others' rewards
        # For agent 0: mean of [2, 3] = 2.5
        # For agent 1: mean of [1, 3] = 2.0
        # For agent 2: mean of [1, 2] = 1.5
        expected = jnp.array([2.5, 2.0, 1.5])
        np.testing.assert_array_almost_equal(svo_rewards, expected)

    def test_compute_svo_reward_cooperative(self):
        """Test that 45 degree angle gives cooperative rewards."""
        rewards = jnp.array([3.0, 0.0, 0.0])
        svo_rewards = compute_svo_reward(rewards, svo_angle=45.0, use_fairness=False)
        # cos(45) = sin(45) = sqrt(2)/2
        # For agent 0: sqrt(2)/2 * 3 + sqrt(2)/2 * 0 = 3*sqrt(2)/2
        # For agent 1: sqrt(2)/2 * 0 + sqrt(2)/2 * 1.5 = 1.5*sqrt(2)/2
        # For agent 2: sqrt(2)/2 * 0 + sqrt(2)/2 * 1.5 = 1.5*sqrt(2)/2
        import math
        w = math.sqrt(2) / 2
        expected = jnp.array([3.0 * w, 1.5 * w, 1.5 * w])
        np.testing.assert_array_almost_equal(svo_rewards, expected, decimal=5)

    def test_compute_svo_reward_with_fairness(self):
        """Test SVO reward with fairness penalty."""
        rewards_equal = jnp.array([1.0, 1.0, 1.0])
        rewards_unequal = jnp.array([3.0, 0.0, 0.0])

        svo_equal = compute_svo_reward(rewards_equal, svo_angle=45.0, use_fairness=True, fairness_weight=0.1)
        svo_unequal = compute_svo_reward(rewards_unequal, svo_angle=45.0, use_fairness=True, fairness_weight=0.1)

        # Unequal rewards should have lower transformed rewards due to fairness penalty
        # (they have higher variance which is penalized)
        assert jnp.mean(svo_equal) > jnp.mean(svo_unequal)

    def test_compute_batch_svo_reward_shape(self):
        """Test batch SVO reward computation preserves shape."""
        batch_rewards = jnp.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ])
        svo_rewards = compute_batch_svo_reward(batch_rewards, svo_angle=45.0)
        assert svo_rewards.shape == batch_rewards.shape

    def test_compute_batch_svo_reward_consistency(self):
        """Test batch SVO reward is consistent with single computation."""
        rewards = jnp.array([1.0, 2.0, 3.0])
        batch_rewards = rewards[jnp.newaxis, :]  # Shape (1, 3)

        single_result = compute_svo_reward(rewards, svo_angle=45.0, use_fairness=False)
        batch_result = compute_batch_svo_reward(batch_rewards, svo_angle=45.0, use_fairness=False)

        np.testing.assert_array_almost_equal(single_result, batch_result[0])


class TestSVOAlgorithmUpdate:
    """Tests for SVO algorithm update method."""

    @pytest.fixture
    def algorithm_and_state(self):
        """Create algorithm and initialized state."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        algo = SVOAlgorithm(MockObsSpace(), MockActSpace())
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)
        return algo, state

    def test_update_returns_new_state(self, algorithm_and_state):
        """Test that update returns new state."""
        algo, state = algorithm_and_state

        batch = {
            "obs": jnp.zeros((32, 11, 11, 19)),
            "actions": jnp.zeros((32,), dtype=jnp.int32),
            "advantages": jnp.zeros((32,)),
            "targets": jnp.zeros((32,)),
            "old_log_probs": jnp.zeros((32,)),
            "values": jnp.zeros((32,)),
        }

        new_state, metrics = algo.update(state, batch)
        assert isinstance(new_state, SVOAlgorithmState)

    def test_update_returns_metrics(self, algorithm_and_state):
        """Test that update returns metrics."""
        algo, state = algorithm_and_state

        batch = {
            "obs": jnp.zeros((32, 11, 11, 19)),
            "actions": jnp.zeros((32,), dtype=jnp.int32),
            "advantages": jnp.zeros((32,)),
            "targets": jnp.zeros((32,)),
            "old_log_probs": jnp.zeros((32,)),
            "values": jnp.zeros((32,)),
        }

        new_state, metrics = algo.update(state, batch)
        assert isinstance(metrics, dict)
        assert "total_loss" in metrics
        assert "value_loss" in metrics
        assert "actor_loss" in metrics
        assert "entropy" in metrics

    def test_update_increments_update_step(self, algorithm_and_state):
        """Test that update increments update_step counter."""
        algo, state = algorithm_and_state

        batch = {
            "obs": jnp.zeros((32, 11, 11, 19)),
            "actions": jnp.zeros((32,), dtype=jnp.int32),
            "advantages": jnp.zeros((32,)),
            "targets": jnp.zeros((32,)),
            "old_log_probs": jnp.zeros((32,)),
            "values": jnp.zeros((32,)),
        }

        initial_step = state.update_step
        new_state, metrics = algo.update(state, batch)
        assert new_state.update_step == initial_step + 1

    def test_update_metrics_contain_svo_angle(self, algorithm_and_state):
        """Test that update metrics contain SVO angle."""
        algo, state = algorithm_and_state

        batch = {
            "obs": jnp.zeros((32, 11, 11, 19)),
            "actions": jnp.zeros((32,), dtype=jnp.int32),
            "advantages": jnp.zeros((32,)),
            "targets": jnp.zeros((32,)),
            "old_log_probs": jnp.zeros((32,)),
            "values": jnp.zeros((32,)),
        }

        new_state, metrics = algo.update(state, batch)
        assert "svo_angle" in metrics


class TestSVOAlgorithmGAE:
    """Tests for SVO algorithm GAE computation."""

    @pytest.fixture
    def algorithm(self):
        """Create an SVO algorithm instance."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        return SVOAlgorithm(MockObsSpace(), MockActSpace())

    def test_compute_gae_returns_advantages(self, algorithm):
        """Test that compute_gae returns advantages."""
        from socialjax.algorithms.svo.algorithm import Transition

        # Create dummy trajectory
        num_steps = 10
        num_agents = 5
        traj = Transition(
            done=jnp.zeros((num_steps, num_agents)),
            action=jnp.zeros((num_steps, num_agents), dtype=jnp.int32),
            value=jnp.zeros((num_steps, num_agents)),
            reward=jnp.zeros((num_steps, num_agents)),
            original_reward=jnp.zeros((num_steps, num_agents)),
            log_prob=jnp.zeros((num_steps, num_agents)),
            obs=jnp.zeros((num_steps, num_agents, 11, 11, 19)),
            info={}
        )

        last_value = jnp.zeros((num_agents,))
        advantages, targets = algorithm.compute_gae(traj, last_value)

        assert advantages.shape == (num_steps, num_agents)
        assert targets.shape == (num_steps, num_agents)


class TestSVOAlgorithmSaveLoad:
    """Tests for SVO algorithm save/load functionality."""

    @pytest.fixture
    def algorithm_and_state(self):
        """Create algorithm and initialized state."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        algo = SVOAlgorithm(MockObsSpace(), MockActSpace())
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)
        return algo, state

    def test_save_creates_checkpoint(self, algorithm_and_state, tmp_path):
        """Test that save creates a checkpoint file."""
        algo, state = algorithm_and_state
        # Set the state on the algorithm
        algo._state = state
        checkpoint_path = str(tmp_path / "checkpoint")
        algo.save(checkpoint_path)
        import os
        assert os.path.exists(os.path.join(checkpoint_path, "checkpoint.pkl"))

    def test_save_load_roundtrip(self, algorithm_and_state, tmp_path):
        """Test that save/load preserves state."""
        algo, state = algorithm_and_state
        # Set the state on the algorithm
        algo._state = state
        checkpoint_path = str(tmp_path / "checkpoint")
        algo.save(checkpoint_path)
        loaded_state = algo.load(checkpoint_path)
        assert loaded_state.params is not None

    def test_state_contains_svo_angle(self, algorithm_and_state):
        """Test that state contains SVO angle."""
        algo, state = algorithm_and_state
        assert hasattr(state, "svo_angle")
        assert state.svo_angle == 45.0


class TestSVOAlgorithmDifferentAngles:
    """Tests for SVO algorithm with different angle configurations."""

    def test_selfish_angle_config(self):
        """Test SVO algorithm with selfish (0 degree) angle."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        algo = SVOAlgorithm(MockObsSpace(), MockActSpace(), {"SVO_ANGLE": 0.0})
        assert algo.config["SVO_ANGLE"] == 0.0

    def test_altruistic_angle_config(self):
        """Test SVO algorithm with altruistic (90 degree) angle."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        algo = SVOAlgorithm(MockObsSpace(), MockActSpace(), {"SVO_ANGLE": 90.0})
        assert algo.config["SVO_ANGLE"] == 90.0

    def test_custom_angle_config(self):
        """Test SVO algorithm with custom angle."""
        class MockObsSpace:
            shape = (11, 11, 19)
        class MockActSpace:
            n = 8
        algo = SVOAlgorithm(MockObsSpace(), MockActSpace(), {"SVO_ANGLE": 22.5})
        assert algo.config["SVO_ANGLE"] == 22.5
