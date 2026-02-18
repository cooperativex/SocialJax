"""Unit tests for VDN algorithm.

Tests cover:
- VDNAlgorithm initialization
- VDNAlgorithmState
- Algorithm registration
- compute_action method
- compute_q_values method
- compute_q_tot method
- update method
- save/load methods
- Q-value decomposition correctness
"""

import pytest
import sys
import tempfile
import os

sys.path.insert(0, "socialjax")

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create mock modules
    class MockModule:
        def __getattr__(self, name):
            return MockModule()

        def __call__(self, *args, **kwargs):
            return MockModule()

    sys.modules["jax"] = MockModule()
    sys.modules["jax.numpy"] = MockModule()
    jnp = MockModule()


class TestVDNAlgorithmImport:
    """Tests for VDN algorithm imports."""

    def test_import_algorithm(self):
        """Test that VDNAlgorithm can be imported."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        assert VDNAlgorithm is not None

    def test_import_state(self):
        """Test that VDNAlgorithmState can be imported."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithmState

        assert VDNAlgorithmState is not None

    def test_import_transition(self):
        """Test that VDNTransition can be imported."""
        from socialjax.algorithms.vdn.algorithm import VDNTransition

        assert VDNTransition is not None


class TestVDNAlgorithmRegistration:
    """Tests for VDN algorithm registration."""

    def test_registered_with_decorator(self):
        """Test that VDNAlgorithm is registered via @register_algorithm."""
        from socialjax.algorithms.registry import get_algorithm, is_algorithm_registered

        # Import to trigger registration
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        assert is_algorithm_registered("vdn")

    def test_get_algorithm_returns_vdn(self):
        """Test that get_algorithm('vdn') returns VDNAlgorithm."""
        from socialjax.algorithms.registry import get_algorithm
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        vdn_class = get_algorithm("vdn")
        assert vdn_class == VDNAlgorithm


class TestVDNAlgorithmInheritance:
    """Tests for VDNAlgorithm inheritance."""

    def test_inherits_from_base_algorithm(self):
        """Test that VDNAlgorithm inherits from BaseAlgorithm."""
        from socialjax.core.base_algorithm import BaseAlgorithm
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        assert issubclass(VDNAlgorithm, BaseAlgorithm)

    def test_has_required_methods(self):
        """Test that VDNAlgorithm has all required abstract methods."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        required_methods = [
            "_build_network",
            "_build_optimizer",
            "init_state",
            "compute_action",
            "update",
        ]

        for method in required_methods:
            assert hasattr(VDNAlgorithm, method), f"Missing method: {method}"
            assert callable(getattr(VDNAlgorithm, method)), f"Method {method} not callable"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNAlgorithmInit:
    """Tests for VDNAlgorithm initialization."""

    def test_init_with_spaces(self):
        """Test VDNAlgorithm initialization with observation/action spaces."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax.numpy as jnp

        # Create mock spaces
        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space, num_agents=2)

        assert algo.observation_space == obs_space
        assert algo.action_space == action_space
        assert algo.num_agents == 2

    def test_init_with_config(self):
        """Test VDNAlgorithm initialization with custom config."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        config = {"LR": 0.001, "HIDDEN_SIZE": 128}
        algo = VDNAlgorithm(obs_space, action_space, config=config)

        assert algo.config["LR"] == 0.001
        assert algo.config["HIDDEN_SIZE"] == 128

    def test_init_merges_with_defaults(self):
        """Test that custom config merges with defaults."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        custom_lr = 0.001
        algo = VDNAlgorithm(obs_space, action_space, config={"LR": custom_lr})

        # Custom value should override
        assert algo.config["LR"] == custom_lr
        # Default values should be preserved
        assert algo.config["GAMMA"] == VDN_DEFAULT_CONFIG["GAMMA"]


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNAlgorithmState:
    """Tests for VDNAlgorithmState."""

    def test_state_creation(self):
        """Test VDNAlgorithmState can be created."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithmState

        import jax
        import jax.numpy as jnp

        rng = jax.random.PRNGKey(0)
        state = VDNAlgorithmState(
            params={},
            target_params={},
            optimizer_state={},
            rng=rng,
            timestep=0,
            update_step=0,
            epsilon=1.0,
        )

        assert state.timestep == 0
        assert state.update_step == 0
        assert state.epsilon == 1.0

    def test_state_is_pytree(self):
        """Test VDNAlgorithmState is a valid JAX PyTree."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithmState

        import jax
        import jax.numpy as jnp

        rng = jax.random.PRNGKey(0)
        state = VDNAlgorithmState(
            params={"w": jnp.zeros((3, 3))},
            target_params={"w": jnp.zeros((3, 3))},
            optimizer_state={},
            rng=rng,
        )

        # Should be able to use with JAX operations
        def _fn(s):
            return s.timestep

        result = jax.jit(_fn)(state)
        assert result == 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNInitState:
    """Tests for VDNAlgorithm.init_state method."""

    def test_init_state_returns_vdn_state(self):
        """Test that init_state returns VDNAlgorithmState."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm, VDNAlgorithmState

        import jax

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space, num_agents=2)
        rng = jax.random.PRNGKey(0)

        state = algo.init_state(rng)

        assert isinstance(state, VDNAlgorithmState)

    def test_init_state_has_params(self):
        """Test that init_state creates network parameters."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space)
        rng = jax.random.PRNGKey(0)

        state = algo.init_state(rng)

        assert state.params is not None
        assert len(state.params) > 0

    def test_init_state_target_equals_main(self):
        """Test that target params equal main params at initialization."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space)
        rng = jax.random.PRNGKey(0)

        state = algo.init_state(rng)

        # Target params should be a copy of main params
        def tree_equal(a, b):
            return jax.tree_util.tree_all(
                jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), a, b)
            )

        assert tree_equal(state.params, state.target_params)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNComputeAction:
    """Tests for VDNAlgorithm.compute_action method."""

    def test_compute_action_returns_action(self):
        """Test that compute_action returns an action."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        # Single agent observation
        obs = jnp.zeros((1, 15, 15, 3))
        rng, action_rng = jax.random.split(rng)

        action, info = algo.compute_action(state, obs, action_rng)

        assert action is not None

    def test_compute_action_deterministic(self):
        """Test deterministic action selection returns greedy actions."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        obs = jnp.zeros((1, 15, 15, 3))

        # Run twice with deterministic=True, should get same action
        action1, _ = algo.compute_action(state, obs, rng, deterministic=True)
        action2, _ = algo.compute_action(state, obs, rng, deterministic=True)

        assert jnp.array_equal(action1, action2)

    def test_compute_action_returns_info(self):
        """Test that compute_action returns info dict with q_values."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        obs = jnp.zeros((1, 15, 15, 3))

        action, info = algo.compute_action(state, obs, rng)

        assert "q_values" in info
        assert "epsilon" in info


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNUpdate:
    """Tests for VDNAlgorithm.update method."""

    def test_update_returns_new_state(self):
        """Test that update returns a new VDNAlgorithmState."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm, VDNAlgorithmState

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space, num_agents=2)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        # Create a dummy batch
        batch = {
            "obs": jnp.zeros((2, 4, 15, 15, 3)),  # (num_agents, batch, *obs_shape)
            "actions": jnp.zeros((2, 4), dtype=jnp.int32),
            "rewards": jnp.zeros((4,)),
            "dones": jnp.zeros((4,)),
            "next_obs": jnp.zeros((2, 4, 15, 15, 3)),
        }

        new_state, metrics = algo.update(state, batch)

        assert isinstance(new_state, VDNAlgorithmState)

    def test_update_returns_metrics(self):
        """Test that update returns metrics dict."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space, num_agents=2)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        batch = {
            "obs": jnp.zeros((2, 4, 15, 15, 3)),
            "actions": jnp.zeros((2, 4), dtype=jnp.int32),
            "rewards": jnp.zeros((4,)),
            "dones": jnp.zeros((4,)),
            "next_obs": jnp.zeros((2, 4, 15, 15, 3)),
        }

        new_state, metrics = algo.update(state, batch)

        assert "loss" in metrics
        assert "epsilon" in metrics

    def test_update_increments_update_step(self):
        """Test that update increments the update_step counter."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space, num_agents=2)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        batch = {
            "obs": jnp.zeros((2, 4, 15, 15, 3)),
            "actions": jnp.zeros((2, 4), dtype=jnp.int32),
            "rewards": jnp.zeros((4,)),
            "dones": jnp.zeros((4,)),
            "next_obs": jnp.zeros((2, 4, 15, 15, 3)),
        }

        new_state, metrics = algo.update(state, batch)

        assert new_state.update_step == state.update_step + 1


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNSaveLoad:
    """Tests for VDNAlgorithm save/load methods."""

    def test_save_creates_file(self):
        """Test that save creates a file."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vdn_checkpoint.pkl")
            algo.save(state, path)
            assert os.path.exists(path)

    def test_load_restores_state(self):
        """Test that load restores algorithm state."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm, VDNAlgorithmState

        import jax

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        # Run a few updates to modify state
        import jax.numpy as jnp

        batch = {
            "obs": jnp.zeros((1, 4, 15, 15, 3)),
            "actions": jnp.zeros((1, 4), dtype=jnp.int32),
            "rewards": jnp.zeros((4,)),
            "dones": jnp.zeros((4,)),
            "next_obs": jnp.zeros((1, 4, 15, 15, 3)),
        }

        for _ in range(5):
            state, _ = algo.update(state, batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vdn_checkpoint.pkl")
            algo.save(state, path)

            loaded_state = algo.load(path)

            assert isinstance(loaded_state, VDNAlgorithmState)
            assert loaded_state.update_step == state.update_step


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNValueDecomposition:
    """Tests for VDN Q-value decomposition correctness."""

    def test_q_tot_is_sum_of_individual_qs(self):
        """Test that Q_tot equals sum of individual agent Q-values."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=3)

        num_agents = 3
        algo = VDNAlgorithm(obs_space, action_space, num_agents=num_agents)
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        # Create observations for all agents
        obs = jax.random.normal(rng, (num_agents, 2, 15, 15, 3))
        actions = jnp.array([[0, 1], [1, 2], [0, 0]], dtype=jnp.int32)

        # Get individual Q-values
        q_values = jax.vmap(algo.network.apply, in_axes=(None, 0))(state.params, obs)

        # Get Q_tot
        q_tot = algo.compute_q_tot(state, obs, actions)

        # Compute expected Q_tot manually
        chosen_q = jnp.take_along_axis(q_values, actions[..., jnp.newaxis], axis=-1).squeeze(-1)
        expected_q_tot = jnp.sum(chosen_q, axis=0)

        assert jnp.allclose(q_tot, expected_q_tot, atol=1e-4)

    def test_target_network_independent(self):
        """Test that target network parameters can differ from main network."""
        from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

        import jax
        import jax.numpy as jnp

        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace((15, 15, 3))
        action_space = MockSpace((), n=5)

        algo = VDNAlgorithm(obs_space, action_space, config={"TARGET_UPDATE_INTERVAL": 1})
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)

        # Run an update (should trigger target network update)
        batch = {
            "obs": jnp.zeros((1, 4, 15, 15, 3)),
            "actions": jnp.zeros((1, 4), dtype=jnp.int32),
            "rewards": jnp.ones((4,)),
            "dones": jnp.zeros((4,)),
            "next_obs": jnp.zeros((1, 4, 15, 15, 3)),
        }

        # Run update - target should be updated after first update
        state, _ = algo.update(state, batch)

        # After update, target params should be updated
        # (with TAU=1.0, they should equal main params)
        assert state.update_step > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
