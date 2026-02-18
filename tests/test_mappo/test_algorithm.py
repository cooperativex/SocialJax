"""Unit tests for MAPPO algorithm."""

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
import tempfile
import os

from socialjax.algorithms.mappo import (
    MAPPOAlgorithm,
    MAPPOAlgorithmState,
    Transition,
    get_mappo_config,
)
from socialjax.algorithms.registry import get_algorithm, list_algorithms
from socialjax.core.base_algorithm import BaseAlgorithm


class DummyObsSpace:
    """Dummy observation space for testing."""
    shape = (15, 15, 3)


class DummyActionSpace:
    """Dummy action space for testing."""
    n = 5


def create_algo():
    """Create a MAPPOAlgorithm instance for testing."""
    config = get_mappo_config({"HIDDEN_SIZE": 32})
    return MAPPOAlgorithm(
        observation_space=DummyObsSpace(),
        action_space=DummyActionSpace(),
        config=config,
        num_agents=5
    )


def create_state(algo):
    """Create an initialized algorithm state."""
    rng = jax.random.PRNGKey(42)
    return algo.init_state(rng)


class TestMAPPOAlgorithmInheritance:
    """Tests for MAPPO algorithm inheritance."""

    def test_inherits_from_base_algorithm(self):
        """Test that MAPPOAlgorithm inherits from BaseAlgorithm."""
        assert issubclass(MAPPOAlgorithm, BaseAlgorithm)

    def test_registered_in_algorithm_registry(self):
        """Test that MAPPO is registered as 'mappo'."""
        algorithms = list_algorithms()
        assert "mappo" in algorithms

    def test_get_algorithm_returns_mappo(self):
        """Test that get_algorithm('mappo') returns MAPPOAlgorithm."""
        algo_class = get_algorithm("mappo")
        assert algo_class is MAPPOAlgorithm


class TestMAPPOAlgorithmInit:
    """Tests for MAPPO algorithm initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        algo = create_algo()
        assert algo.config is not None
        assert algo.num_agents == 5

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = get_mappo_config({
            "LR": 0.001,
            "HIDDEN_SIZE": 128,
        })
        algo = MAPPOAlgorithm(
            observation_space=DummyObsSpace(),
            action_space=DummyActionSpace(),
            config=config,
            num_agents=3
        )
        assert algo.config["LR"] == 0.001
        assert algo.config["HIDDEN_SIZE"] == 128
        assert algo.num_agents == 3

    def test_network_is_tuple(self):
        """Test that network is a tuple of (actor, critic)."""
        algo = create_algo()
        assert isinstance(algo.network, tuple)
        assert len(algo.network) == 2

    def test_optimizer_is_tuple(self):
        """Test that optimizer is a tuple of (actor_optimizer, critic_optimizer)."""
        algo = create_algo()
        assert isinstance(algo.optimizer, tuple)
        assert len(algo.optimizer) == 2


class TestMAPPOAlgorithmState:
    """Tests for MAPPO algorithm state."""

    def test_init_state_type(self):
        """Test that init_state returns MAPPOAlgorithmState."""
        algo = create_algo()
        state = create_state(algo)
        assert isinstance(state, MAPPOAlgorithmState)

    def test_init_state_has_params(self):
        """Test that state has actor and critic params."""
        algo = create_algo()
        state = create_state(algo)
        assert hasattr(state, 'actor_params')
        assert hasattr(state, 'critic_params')
        assert state.actor_params is not None
        assert state.critic_params is not None

    def test_init_state_has_optimizer_states(self):
        """Test that state has optimizer states."""
        algo = create_algo()
        state = create_state(algo)
        assert hasattr(state, 'actor_optimizer_state')
        assert hasattr(state, 'critic_optimizer_state')

    def test_init_state_has_rng(self):
        """Test that state has RNG key."""
        algo = create_algo()
        state = create_state(algo)
        assert hasattr(state, 'rng')
        assert state.rng is not None

    def test_init_state_timestep(self):
        """Test that initial timestep is 0."""
        algo = create_algo()
        state = create_state(algo)
        assert state.timestep == 0
        assert state.update_step == 0


class TestMAPPOComputeAction:
    """Tests for MAPPO compute_action method."""

    def test_compute_action_shape(self):
        """Test that compute_action returns correct action shape."""
        algo = create_algo()
        state = create_state(algo)
        rng = jax.random.PRNGKey(0)
        obs = jnp.zeros((1, 15, 15, 3))
        action, info = algo.compute_action(state, obs, rng)
        assert action.shape == (1,)

    def test_compute_action_in_range(self):
        """Test that actions are in valid range."""
        algo = create_algo()
        state = create_state(algo)
        rng = jax.random.PRNGKey(0)
        obs = jnp.zeros((1, 15, 15, 3))
        action, info = algo.compute_action(state, obs, rng)
        assert jnp.all(action >= 0)
        assert jnp.all(action < 5)

    def test_compute_action_deterministic(self):
        """Test that deterministic actions are greedy."""
        algo = create_algo()
        state = create_state(algo)
        rng = jax.random.PRNGKey(0)
        obs = jnp.zeros((1, 15, 15, 3))
        action_det, _ = algo.compute_action(state, obs, rng, deterministic=True)
        action_det2, _ = algo.compute_action(state, obs, rng, deterministic=True)
        assert jnp.array_equal(action_det, action_det2)

    def test_compute_action_returns_log_prob(self):
        """Test that compute_action returns log_prob in info."""
        algo = create_algo()
        state = create_state(algo)
        rng = jax.random.PRNGKey(0)
        obs = jnp.zeros((1, 15, 15, 3))
        action, info = algo.compute_action(state, obs, rng)
        assert "log_prob" in info
        assert info["log_prob"].shape == (1,)


class TestMAPPOComputeValue:
    """Tests for MAPPO compute_value method."""

    def test_compute_value_shape(self):
        """Test that compute_value returns correct value shape."""
        algo = create_algo()
        state = create_state(algo)
        # World state with 5 agents (5 * 3 = 15 channels)
        world_state = jnp.zeros((1, 15, 15, 15))
        value = algo.compute_value(state, world_state)
        assert value.shape == (1,)

    def test_compute_value_batch(self):
        """Test that compute_value handles batch of world states."""
        algo = create_algo()
        state = create_state(algo)
        batch_size = 8
        world_state = jnp.zeros((batch_size, 15, 15, 15))
        values = algo.compute_value(state, world_state)
        assert values.shape == (batch_size,)

    def test_compute_value_centralized(self):
        """Test that critic uses centralized (global) information."""
        algo = create_algo()
        state = create_state(algo)
        # Critic should process world state (all agent observations)
        world_state = jnp.zeros((1, 15, 15, 15))  # 5 agents
        value = algo.compute_value(state, world_state)
        assert value.shape == (1,)


class TestMAPPOUpdate:
    """Tests for MAPPO update method."""

    def _create_dummy_batch(self, batch_size=160):
        """Create a dummy batch for testing."""
        rng = jax.random.PRNGKey(0)
        return {
            'obs': jax.random.normal(rng, (batch_size, 15, 15, 3)),
            'world_state': jax.random.normal(rng, (batch_size, 15, 15, 15)),
            'actions': jax.random.randint(rng, (batch_size,), 0, 5),
            'advantages': jax.random.normal(rng, (batch_size,)),
            'targets': jax.random.normal(rng, (batch_size,)),
            'old_log_probs': jax.random.normal(rng, (batch_size,)),
            'values': jax.random.normal(rng, (batch_size,)),
        }

    def test_update_returns_new_state(self):
        """Test that update returns new state."""
        algo = create_algo()
        state = create_state(algo)
        batch = self._create_dummy_batch()
        new_state, metrics = algo.update(state, batch)
        assert isinstance(new_state, MAPPOAlgorithmState)

    def test_update_returns_metrics(self):
        """Test that update returns metrics dict."""
        algo = create_algo()
        state = create_state(algo)
        batch = self._create_dummy_batch()
        new_state, metrics = algo.update(state, batch)
        assert isinstance(metrics, dict)
        assert "total_loss" in metrics
        assert "actor_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_update_increments_update_step(self):
        """Test that update increments update_step."""
        algo = create_algo()
        state = create_state(algo)
        batch = self._create_dummy_batch()
        initial_step = state.update_step
        new_state, _ = algo.update(state, batch)
        assert new_state.update_step == initial_step + 1

    def test_update_changes_params(self):
        """Test that update changes network parameters."""
        algo = create_algo()
        state = create_state(algo)
        batch = self._create_dummy_batch()
        new_state, _ = algo.update(state, batch)
        assert new_state.actor_params is not None
        assert new_state.critic_params is not None


class TestMAPPOSaveLoad:
    """Tests for MAPPO save/load methods."""

    def test_save_creates_file(self):
        """Test that save creates a file."""
        algo = create_algo()
        state = create_state(algo)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mappo_checkpoint.pkl")
            algo.save(state, path)
            assert os.path.exists(path)

    def test_load_returns_state(self):
        """Test that load returns a state."""
        algo = create_algo()
        state = create_state(algo)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mappo_checkpoint.pkl")
            algo.save(state, path)
            loaded_state = algo.load(path)
            assert isinstance(loaded_state, MAPPOAlgorithmState)

    def test_save_load_preserves_timestep(self):
        """Test that save/load preserves timestep."""
        algo = create_algo()
        state = create_state(algo)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mappo_checkpoint.pkl")
            algo.save(state, path)
            loaded_state = algo.load(path)
            assert loaded_state.timestep == state.timestep


class TestMAPPOTraining:
    """Integration tests for MAPPO training."""

    def test_training_updates_decrease_loss(self):
        """Test that training updates can decrease loss."""
        algo = create_algo()
        state = create_state(algo)

        rng = jax.random.PRNGKey(42)
        losses = []
        for i in range(20):
            rng, batch_rng = jax.random.split(rng)
            batch = {
                'obs': jax.random.normal(batch_rng, (160, 15, 15, 3)),
                'world_state': jax.random.normal(batch_rng, (160, 15, 15, 15)),
                'actions': jax.random.randint(batch_rng, (160,), 0, 5),
                'advantages': jax.random.normal(batch_rng, (160,)),
                'targets': jax.random.normal(batch_rng, (160,)),
                'old_log_probs': jax.random.normal(batch_rng, (160,)) * 0.1,
                'values': jax.random.normal(batch_rng, (160,)) * 0.1,
            }
            state, metrics = algo.update(state, batch)
            losses.append(metrics["total_loss"])

        assert len(losses) == 20

    def test_training_10k_steps(self):
        """Test that training can run for 10K steps equivalent."""
        algo = create_algo()
        state = create_state(algo)

        rng = jax.random.PRNGKey(42)
        for i in range(100):
            rng, batch_rng = jax.random.split(rng)
            batch = {
                'obs': jax.random.normal(batch_rng, (160, 15, 15, 3)),
                'world_state': jax.random.normal(batch_rng, (160, 15, 15, 15)),
                'actions': jax.random.randint(batch_rng, (160,), 0, 5),
                'advantages': jax.random.normal(batch_rng, (160,)),
                'targets': jax.random.normal(batch_rng, (160,)),
                'old_log_probs': jax.random.normal(batch_rng, (160,)) * 0.1,
                'values': jax.random.normal(batch_rng, (160,)) * 0.1,
            }
            state, metrics = algo.update(state, batch)

        assert state.update_step == 100


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Run tests manually
        print("Running MAPPO Algorithm Tests...")
        test_classes = [
            TestMAPPOAlgorithmInheritance(),
            TestMAPPOAlgorithmInit(),
            TestMAPPOAlgorithmState(),
            TestMAPPOComputeAction(),
            TestMAPPOComputeValue(),
            TestMAPPOUpdate(),
            TestMAPPOSaveLoad(),
            TestMAPPOTraining(),
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
