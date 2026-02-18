"""Unit tests for VDN neural network architectures.

Tests cover:
- VDNCNN feature extractor
- VDNQNetwork Q-value estimation
- compute_q_tot value decomposition
- compute_vdn_target target computation
"""

import pytest
import sys
import numpy as np

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


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNCNN:
    """Tests for VDNCNN feature extractor."""

    def test_import(self):
        """Test that VDNCNN can be imported."""
        from socialjax.algorithms.vdn.network import VDNCNN

        assert VDNCNN is not None

    def test_network_creation(self):
        """Test VDNCNN can be created."""
        from socialjax.algorithms.vdn.network import VDNCNN

        network = VDNCNN(activation="relu", hidden_size=64)
        assert network is not None

    def test_forward_pass_shape(self):
        """Test VDNCNN forward pass produces correct output shape."""
        from socialjax.algorithms.vdn.network import VDNCNN

        import jax
        import jax.numpy as jnp

        # Create network
        network = VDNCNN(activation="relu", hidden_size=64)

        # Create dummy input (batch=2, height=15, width=15, channels=3)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((2, 15, 15, 3))

        # Initialize and run forward pass
        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)

        # Output should be (batch, hidden_size)
        assert output.shape == (2, 64), f"Expected shape (2, 64), got {output.shape}"

    def test_relu_activation(self):
        """Test VDNCNN with relu activation."""
        from socialjax.algorithms.vdn.network import VDNCNN

        import jax
        import jax.numpy as jnp

        network = VDNCNN(activation="relu", hidden_size=64)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 15, 15, 3))

        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)

        assert output.shape == (1, 64)

    def test_tanh_activation(self):
        """Test VDNCNN with tanh activation."""
        from socialjax.algorithms.vdn.network import VDNCNN

        import jax
        import jax.numpy as jnp

        network = VDNCNN(activation="tanh", hidden_size=32)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 15, 15, 3))

        params = network.init(rng, dummy_input)
        output = network.apply(params, dummy_input)

        assert output.shape == (1, 32)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVDNQNetwork:
    """Tests for VDNQNetwork."""

    def test_import(self):
        """Test that VDNQNetwork can be imported."""
        from socialjax.algorithms.vdn.network import VDNQNetwork

        assert VDNQNetwork is not None

    def test_network_registered(self):
        """Test that VDNQNetwork is registered in network registry."""
        from socialjax.networks.registry import get_network_class, is_network_registered

        # Import to trigger registration
        from socialjax.algorithms.vdn.network import VDNQNetwork

        assert is_network_registered("vdn_q_network")

    def test_network_creation(self):
        """Test VDNQNetwork can be created."""
        from socialjax.algorithms.vdn.network import VDNQNetwork

        network = VDNQNetwork(action_dim=5, activation="relu", hidden_size=64)
        assert network is not None
        assert network.action_dim == 5

    def test_forward_pass_shape(self):
        """Test VDNQNetwork forward pass produces Q-values with correct shape."""
        from socialjax.algorithms.vdn.network import VDNQNetwork

        import jax
        import jax.numpy as jnp

        action_dim = 8
        network = VDNQNetwork(action_dim=action_dim, activation="relu", hidden_size=64)

        # Create dummy input
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((4, 15, 15, 3))

        # Initialize and run forward pass
        params = network.init(rng, dummy_input)
        q_values = network.apply(params, dummy_input)

        # Output should be (batch, action_dim)
        assert q_values.shape == (4, action_dim), (
            f"Expected shape (4, {action_dim}), got {q_values.shape}"
        )

    def test_q_values_finite(self):
        """Test that Q-values are finite (no NaN or Inf)."""
        from socialjax.algorithms.vdn.network import VDNQNetwork

        import jax
        import jax.numpy as jnp

        network = VDNQNetwork(action_dim=5, activation="relu", hidden_size=64)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 15, 15, 3))

        params = network.init(rng, dummy_input)
        q_values = network.apply(params, dummy_input)

        assert jnp.all(jnp.isfinite(q_values)), "Q-values contain NaN or Inf"

    def test_different_input_shapes(self):
        """Test VDNQNetwork with different input shapes."""
        from socialjax.algorithms.vdn.network import VDNQNetwork

        import jax
        import jax.numpy as jnp

        network = VDNQNetwork(action_dim=4, hidden_size=32)
        rng = jax.random.PRNGKey(0)

        # Test different batch sizes
        for batch_size in [1, 8, 32]:
            dummy_input = jnp.zeros((batch_size, 15, 15, 3))
            params = network.init(rng, dummy_input)
            q_values = network.apply(params, dummy_input)
            assert q_values.shape == (batch_size, 4)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestComputeQTot:
    """Tests for compute_q_tot function."""

    def test_import(self):
        """Test that compute_q_tot can be imported."""
        from socialjax.algorithms.vdn.network import compute_q_tot

        assert callable(compute_q_tot)

    def test_sum_across_agents(self):
        """Test that compute_q_tot correctly sums Q-values across agents."""
        from socialjax.algorithms.vdn.network import compute_q_tot

        import jax.numpy as jnp

        # Q-values for 3 agents, batch of 2, 4 actions each
        q_values = jnp.ones((3, 2, 4))

        q_tot = compute_q_tot(q_values, axis=0)

        # Should sum to 3 for each batch element
        assert q_tot.shape == (2, 4)
        assert jnp.allclose(q_tot, 3.0)

    def test_different_values(self):
        """Test compute_q_tot with different Q-values per agent."""
        from socialjax.algorithms.vdn.network import compute_q_tot

        import jax.numpy as jnp

        # Agent 0: all 1s, Agent 1: all 2s
        q_values = jnp.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], dtype=jnp.float32)

        q_tot = compute_q_tot(q_values, axis=0)

        expected = jnp.array([[3, 3], [3, 3]], dtype=jnp.float32)
        assert jnp.allclose(q_tot, expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestComputeVDNTarget:
    """Tests for compute_vdn_target function."""

    def test_import(self):
        """Test that compute_vdn_target can be imported."""
        from socialjax.algorithms.vdn.network import compute_vdn_target

        assert callable(compute_vdn_target)

    def test_target_computation(self):
        """Test basic VDN target computation."""
        from socialjax.algorithms.vdn.network import compute_vdn_target

        import jax.numpy as jnp

        # Q-next values for 2 agents, batch of 2, 3 actions each
        q_next = jnp.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Agent 0
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Agent 1
            ]
        )

        rewards = jnp.array([1.0, 2.0])
        dones = jnp.array([0.0, 0.0])
        gamma = 0.99

        target = compute_vdn_target(q_next, rewards, dones, gamma)

        # Max Q for each agent: [3, 6] for agent 0, [3, 6] for agent 1
        # Sum across agents: [6, 12]
        # Target: r + gamma * sum = [1 + 0.99*6, 2 + 0.99*12] = [6.94, 13.88]
        expected = jnp.array([1.0 + 0.99 * 6.0, 2.0 + 0.99 * 12.0])

        assert jnp.allclose(target, expected, atol=1e-4)

    def test_target_with_dones(self):
        """Test VDN target computation with terminal states."""
        from socialjax.algorithms.vdn.network import compute_vdn_target

        import jax.numpy as jnp

        q_next = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]])

        rewards = jnp.array([1.0, 2.0])
        dones = jnp.array([0.0, 1.0])  # Second sample is terminal
        gamma = 0.99

        target = compute_vdn_target(q_next, rewards, dones, gamma)

        # First sample: not done, target = r + gamma * sum(max_q)
        # Second sample: done, target = r
        expected_first = 1.0 + 0.99 * (2.0 + 2.0)  # 1 + 0.99 * 4
        expected_second = 2.0  # No bootstrapping

        assert jnp.isclose(target[0], expected_first, atol=1e-4)
        assert jnp.isclose(target[1], expected_second, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
