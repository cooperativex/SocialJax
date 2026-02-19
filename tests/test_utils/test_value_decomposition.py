"""Unit tests for value decomposition utilities.

Tests cover:
- vdn_decomposition for VDN-style value decomposition
- vdn_target for computing TD targets
- compute_td_loss for TD loss computation
- epsilon_greedy_action for exploration
- soft_target_update and hard_target_update for target network updates
- Edge cases and numerical stability
"""

import pytest
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'socialjax')

from socialjax.algorithms.utils.value_decomposition import (
    vdn_decomposition,
    vdn_target,
    qmix_mixing_network,
    compute_td_loss,
    epsilon_greedy_action,
    soft_target_update,
    hard_target_update,
    create_vdn_loss_fn,
    ValueDecompositionOutput,
)


class TestVDNDecomposition:
    """Tests for vdn_decomposition function."""

    def test_basic_computation(self):
        """Test basic VDN decomposition."""
        num_agents = 3
        batch_size = 4
        action_dim = 5

        q_values = jnp.ones((num_agents, batch_size, action_dim))
        actions = jnp.array([[0, 1, 2, 3],
                             [1, 2, 3, 4],
                             [2, 3, 4, 0]])

        output = vdn_decomposition(q_values, actions)

        assert isinstance(output, ValueDecompositionOutput)
        assert output.q_tot.shape == (batch_size,)
        assert output.individual_q.shape == (num_agents, batch_size, action_dim)
        assert output.chosen_q_values.shape == (num_agents, batch_size)

    def test_q_tot_is_sum(self):
        """Test that Q_tot is the sum of individual Q-values."""
        num_agents = 3
        batch_size = 4
        action_dim = 5

        # Use different Q-values for each agent
        q_values = jnp.arange(num_agents * batch_size * action_dim).reshape(
            num_agents, batch_size, action_dim
        ).astype(float)
        actions = jnp.array([[0, 1, 2, 3],
                             [1, 2, 3, 4],
                             [2, 3, 4, 0]])

        output = vdn_decomposition(q_values, actions)

        # Manually compute expected Q_tot
        expected_chosen_q = jnp.take_along_axis(
            q_values, actions[..., jnp.newaxis], axis=-1
        ).squeeze(-1)
        expected_q_tot = jnp.sum(expected_chosen_q, axis=0)

        np.testing.assert_allclose(output.q_tot, expected_q_tot, rtol=1e-5)
        np.testing.assert_allclose(output.chosen_q_values, expected_chosen_q, rtol=1e-5)

    def test_single_agent(self):
        """Test VDN decomposition with single agent."""
        num_agents = 1
        batch_size = 4
        action_dim = 5

        q_values = jnp.ones((num_agents, batch_size, action_dim))
        actions = jnp.array([[0, 1, 2, 3]])

        output = vdn_decomposition(q_values, actions)

        # For single agent, Q_tot = Q_1
        np.testing.assert_allclose(output.q_tot, output.chosen_q_values[0], rtol=1e-5)

    def test_different_q_values(self):
        """Test with different Q-values per agent."""
        num_agents = 2
        batch_size = 2
        action_dim = 3

        # Agent 0: higher Q-values for action 0
        # Agent 1: higher Q-values for action 2
        q_values = jnp.array([
            [[2.0, 1.0, 0.5], [2.0, 1.0, 0.5]],  # Agent 0
            [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]],  # Agent 1
        ])
        actions = jnp.array([[0, 0], [2, 2]])

        output = vdn_decomposition(q_values, actions)

        # Q_tot should be 2.0 + 2.0 = 4.0 for each sample
        np.testing.assert_allclose(output.q_tot, jnp.array([4.0, 4.0]), rtol=1e-5)

    def test_jit_compatibility(self):
        """Test that vdn_decomposition is JIT-compatible."""
        num_agents = 3
        batch_size = 4
        action_dim = 5

        q_values = jnp.ones((num_agents, batch_size, action_dim))
        actions = jnp.array([[0, 1, 2, 3],
                             [1, 2, 3, 4],
                             [2, 3, 4, 0]])

        jitted_vdn = jax.jit(vdn_decomposition)
        output = jitted_vdn(q_values, actions)

        assert not jnp.any(jnp.isnan(output.q_tot))


class TestVDNTarget:
    """Tests for vdn_target function."""

    def test_basic_computation(self):
        """Test basic VDN target computation."""
        num_agents = 3
        batch_size = 4
        action_dim = 5

        q_target = jnp.ones((num_agents, batch_size, action_dim))
        rewards = jnp.array([1.0, 0.5, 0.0, -0.5])
        dones = jnp.array([0.0, 0.0, 1.0, 0.0])

        targets = vdn_target(q_target, rewards, dones)

        assert targets.shape == (batch_size,)
        assert not jnp.any(jnp.isnan(targets))

    def test_target_formula(self):
        """Test VDN target formula: y = r + gamma * sum_i max_a' Q_i^target(s', a')."""
        num_agents = 2
        batch_size = 2
        action_dim = 3

        q_target = jnp.array([
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],  # Agent 0
            [[2.0, 1.0, 0.5], [1.5, 2.0, 1.0]],  # Agent 1
        ])
        rewards = jnp.array([1.0, 0.5])
        dones = jnp.array([0.0, 0.0])
        gamma = 0.99

        targets = vdn_target(q_target, rewards, dones, gamma=gamma)

        # Expected: y = r + gamma * (max Q_0 + max Q_1)
        # Sample 0: 1.0 + 0.99 * (3.0 + 2.0) = 1.0 + 4.95 = 5.95
        # Sample 1: 0.5 + 0.99 * (1.5 + 2.0) = 0.5 + 3.465 = 3.965
        expected = jnp.array([5.95, 3.965])
        np.testing.assert_allclose(targets, expected, rtol=1e-5)

    def test_with_termination(self):
        """Test VDN target with episode termination."""
        num_agents = 2
        batch_size = 2
        action_dim = 3

        q_target = jnp.ones((num_agents, batch_size, action_dim))
        rewards = jnp.array([1.0, 0.5])
        dones = jnp.array([1.0, 0.0])  # First sample terminates

        targets = vdn_target(q_target, rewards, dones, gamma=0.99)

        # When done=1, target = r (no bootstrapping)
        np.testing.assert_allclose(float(targets[0]), 1.0, rtol=1e-5)

    def test_different_gamma(self):
        """Test VDN target with different discount factors."""
        num_agents = 2
        batch_size = 2
        action_dim = 3

        q_target = jnp.ones((num_agents, batch_size, action_dim)) * 2.0
        rewards = jnp.array([1.0, 0.5])
        dones = jnp.zeros(2)

        targets_low_gamma = vdn_target(q_target, rewards, dones, gamma=0.5)
        targets_high_gamma = vdn_target(q_target, rewards, dones, gamma=0.99)

        # Higher gamma should give higher targets
        assert float(targets_high_gamma[0]) > float(targets_low_gamma[0])

    def test_jit_compatibility(self):
        """Test that vdn_target is JIT-compatible."""
        num_agents = 3
        batch_size = 4
        action_dim = 5

        q_target = jnp.ones((num_agents, batch_size, action_dim))
        rewards = jnp.array([1.0, 0.5, 0.0, -0.5])
        dones = jnp.array([0.0, 0.0, 1.0, 0.0])

        jitted_target = jax.jit(vdn_target)
        targets = jitted_target(q_target, rewards, dones)

        assert not jnp.any(jnp.isnan(targets))


class TestComputeTDLoss:
    """Tests for compute_td_loss function."""

    def test_mse_loss(self):
        """Test MSE TD loss computation."""
        q_tot = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.1, 1.9, 3.2, 3.8])

        loss = compute_td_loss(q_tot, targets, loss_type="mse")

        expected = jnp.mean(jnp.square(q_tot - targets))
        np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)

    def test_huber_loss(self):
        """Test Huber TD loss computation."""
        q_tot = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.1, 1.9, 3.2, 3.8])

        loss_mse = compute_td_loss(q_tot, targets, loss_type="mse")
        loss_huber = compute_td_loss(q_tot, targets, loss_type="huber")

        # Both should be valid
        assert float(loss_mse) >= 0.0
        assert float(loss_huber) >= 0.0

    def test_perfect_prediction(self):
        """Test TD loss with perfect predictions."""
        q_tot = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = q_tot

        loss = compute_td_loss(q_tot, targets)

        np.testing.assert_allclose(float(loss), 0.0, atol=1e-5)

    def test_invalid_loss_type(self):
        """Test that invalid loss type raises error."""
        q_tot = jnp.array([1.0, 2.0])
        targets = jnp.array([1.1, 1.9])

        with pytest.raises(ValueError):
            compute_td_loss(q_tot, targets, loss_type="invalid")

    def test_jit_compatibility(self):
        """Test that compute_td_loss is JIT-compatible when wrapped."""
        q_tot = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.1, 1.9, 3.2, 3.8])

        # JIT with loss_type as static argument
        jitted_mse = jax.jit(lambda q, t: compute_td_loss(q, t, "mse"))
        loss = jitted_mse(q_tot, targets)

        assert not jnp.isnan(loss)
        assert float(loss) >= 0.0


class TestEpsilonGreedyAction:
    """Tests for epsilon_greedy_action function."""

    def test_greedy_selection(self):
        """Test greedy action selection (epsilon=0)."""
        q_values = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])
        key = jax.random.PRNGKey(0)

        actions = epsilon_greedy_action(q_values, key, epsilon=0.0)

        # Should select argmax for each row
        expected = jnp.array([1, 2])  # Index of max Q for each
        np.testing.assert_array_equal(actions, expected)

    def test_random_selection(self):
        """Test random action selection (epsilon=1)."""
        q_values = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])
        key = jax.random.PRNGKey(0)

        # With epsilon=1, actions should be random (not deterministic)
        actions1 = epsilon_greedy_action(q_values, key, epsilon=1.0)
        actions2 = epsilon_greedy_action(q_values, jax.random.PRNGKey(1), epsilon=1.0)

        # Different keys should give different actions (with high probability)
        # Just verify shape and valid range
        assert actions1.shape == (2,)
        assert jnp.all(actions1 >= 0) and jnp.all(actions1 < 3)

    def test_intermediate_epsilon(self):
        """Test with intermediate epsilon values."""
        q_values = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])
        key = jax.random.PRNGKey(0)

        actions = epsilon_greedy_action(q_values, key, epsilon=0.5)

        assert actions.shape == (2,)
        assert jnp.all(actions >= 0) and jnp.all(actions < 3)

    def test_output_shape(self):
        """Test that output shape matches input."""
        q_values = jnp.ones((3, 4, 5))  # 3 samples, 4 sub-samples, 5 actions
        key = jax.random.PRNGKey(0)

        actions = epsilon_greedy_action(q_values, key, epsilon=0.1)

        assert actions.shape == (3, 4)

    def test_jit_compatibility(self):
        """Test that epsilon_greedy_action is JIT-compatible."""
        q_values = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])
        key = jax.random.PRNGKey(0)

        jitted_action = jax.jit(epsilon_greedy_action)
        actions = jitted_action(q_values, key, epsilon=0.5)

        assert actions.shape == (2,)


class TestTargetNetworkUpdates:
    """Tests for soft_target_update and hard_target_update functions."""

    def test_soft_update_formula(self):
        """Test soft target update formula: target = tau * params + (1-tau) * target."""
        params = {'w': jnp.array([1.0, 2.0, 3.0])}
        target_params = {'w': jnp.array([0.0, 0.0, 0.0])}
        tau = 0.1

        new_target = soft_target_update(params, target_params, tau)

        expected = {'w': jnp.array([0.1, 0.2, 0.3])}
        np.testing.assert_allclose(new_target['w'], expected['w'], rtol=1e-5)

    def test_soft_update_different_tau(self):
        """Test soft update with different tau values."""
        params = {'w': jnp.array([1.0, 1.0, 1.0])}
        target_params = {'w': jnp.array([0.0, 0.0, 0.0])}

        new_target_low = soft_target_update(params, target_params, tau=0.1)
        new_target_high = soft_target_update(params, target_params, tau=0.9)

        # Higher tau means faster convergence to params
        assert float(new_target_high['w'][0]) > float(new_target_low['w'][0])

    def test_soft_update_nested_params(self):
        """Test soft update with nested parameter structure."""
        params = {
            'layer1': {'w': jnp.array([1.0, 2.0])},
            'layer2': {'w': jnp.array([3.0, 4.0])},
        }
        target_params = {
            'layer1': {'w': jnp.array([0.0, 0.0])},
            'layer2': {'w': jnp.array([0.0, 0.0])},
        }

        new_target = soft_target_update(params, target_params, tau=0.5)

        np.testing.assert_allclose(new_target['layer1']['w'], jnp.array([0.5, 1.0]), rtol=1e-5)
        np.testing.assert_allclose(new_target['layer2']['w'], jnp.array([1.5, 2.0]), rtol=1e-5)

    def test_hard_update(self):
        """Test hard target update: target = params."""
        params = {'w': jnp.array([1.0, 2.0, 3.0])}
        target_params = {'w': jnp.array([0.0, 0.0, 0.0])}

        new_target = hard_target_update(params, target_params)

        np.testing.assert_allclose(new_target['w'], params['w'], rtol=1e-5)

    def test_jit_compatibility(self):
        """Test that target updates are JIT-compatible."""
        params = {'w': jnp.array([1.0, 2.0, 3.0])}
        target_params = {'w': jnp.array([0.0, 0.0, 0.0])}

        jitted_soft = jax.jit(soft_target_update)
        jitted_hard = jax.jit(hard_target_update)

        new_soft = jitted_soft(params, target_params, 0.1)
        new_hard = jitted_hard(params, target_params)

        assert not jnp.any(jnp.isnan(new_soft['w']))
        assert not jnp.any(jnp.isnan(new_hard['w']))


class TestCreateVDNLossFn:
    """Tests for create_vdn_loss_fn factory function."""

    def test_create_function(self):
        """Test creating VDN loss function."""
        def mock_network_apply(params, obs):
            return jnp.ones((obs.shape[0], 5))  # Fixed Q-values

        loss_fn = create_vdn_loss_fn(
            network_apply_fn=mock_network_apply,
            target_network_apply_fn=mock_network_apply,
            gamma=0.99,
        )

        assert callable(loss_fn)

    def test_function_output(self):
        """Test that created function returns correct outputs."""
        def mock_network_apply(params, obs):
            batch_size = obs.shape[0]
            return jnp.ones((batch_size, 3)) * params['scale']

        loss_fn = create_vdn_loss_fn(
            network_apply_fn=mock_network_apply,
            target_network_apply_fn=mock_network_apply,
            gamma=0.99,
        )

        params = {'scale': 1.0}
        target_params = {'scale': 1.0}
        obs = jnp.ones((2, 4, 3))  # 2 agents, batch size 4, obs dim 3
        actions = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        rewards = jnp.array([0.1, 0.2, 0.3, 0.4])
        dones = jnp.zeros(4)
        next_obs = jnp.ones((2, 4, 3))

        loss, aux = loss_fn(params, obs, actions, rewards, dones, next_obs, target_params)

        assert isinstance(loss, jnp.ndarray)
        assert 'q_tot_mean' in aux
        assert 'target_mean' in aux

    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        def mock_network_apply(params, obs):
            # Simple linear transformation
            return params['w'] @ obs.T  # [action_dim, batch] -> need to transpose

        # Better mock that handles batching
        def mock_network_apply(params, obs):
            batch_size = obs.shape[0]
            return jnp.ones((batch_size, 3)) * params['scale']

        loss_fn = create_vdn_loss_fn(
            network_apply_fn=mock_network_apply,
            target_network_apply_fn=mock_network_apply,
            gamma=0.99,
        )

        params = {'scale': jnp.array(1.0)}
        target_params = {'scale': jnp.array(1.0)}
        obs = jnp.ones((2, 4, 3))
        actions = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        rewards = jnp.array([0.1, 0.2, 0.3, 0.4])
        dones = jnp.zeros(4)
        next_obs = jnp.ones((2, 4, 3))

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, obs, actions, rewards, dones, next_obs, target_params
        )

        assert 'scale' in grads


class TestValueDecompositionOutput:
    """Tests for ValueDecompositionOutput NamedTuple."""

    def test_create_output(self):
        """Test creating ValueDecompositionOutput."""
        output = ValueDecompositionOutput(
            q_tot=jnp.array([1.0, 2.0, 3.0]),
            individual_q=jnp.ones((2, 3, 4)),
            chosen_q_values=jnp.ones((2, 3)),
        )

        assert output.q_tot.shape == (3,)
        assert output.individual_q.shape == (2, 3, 4)
        assert output.chosen_q_values.shape == (2, 3)

    def test_field_access(self):
        """Test accessing fields of ValueDecompositionOutput."""
        output = ValueDecompositionOutput(
            q_tot=jnp.array([1.0, 2.0, 3.0]),
            individual_q=jnp.ones((2, 3, 4)),
            chosen_q_values=jnp.ones((2, 3)),
        )

        _ = output.q_tot
        _ = output.individual_q
        _ = output.chosen_q_values


class TestValueDecompositionNumericalStability:
    """Tests for numerical stability of value decomposition."""

    def test_large_q_values(self):
        """Test with large Q-values."""
        num_agents = 3
        batch_size = 4
        action_dim = 5

        q_values = jnp.ones((num_agents, batch_size, action_dim)) * 1000.0
        actions = jnp.array([[0, 1, 2, 3],
                             [1, 2, 3, 4],
                             [2, 3, 4, 0]])

        output = vdn_decomposition(q_values, actions)

        assert not jnp.any(jnp.isnan(output.q_tot))
        assert not jnp.any(jnp.isinf(output.q_tot))

    def test_negative_q_values(self):
        """Test with negative Q-values."""
        num_agents = 3
        batch_size = 4
        action_dim = 5

        q_values = -jnp.ones((num_agents, batch_size, action_dim))
        actions = jnp.array([[0, 1, 2, 3],
                             [1, 2, 3, 4],
                             [2, 3, 4, 0]])

        output = vdn_decomposition(q_values, actions)

        assert not jnp.any(jnp.isnan(output.q_tot))
        # Q_tot should be negative
        assert jnp.all(output.q_tot < 0)

    def test_mixed_q_values(self):
        """Test with mixed positive and negative Q-values."""
        num_agents = 2
        batch_size = 2
        action_dim = 3

        q_values = jnp.array([
            [[1.0, -1.0, 0.5], [-2.0, 2.0, -0.5]],
            [[-1.0, 1.0, -0.5], [2.0, -2.0, 0.5]],
        ])
        actions = jnp.array([[0, 1], [1, 0]])

        output = vdn_decomposition(q_values, actions)

        assert not jnp.any(jnp.isnan(output.q_tot))


class TestQMIXMixingNetwork:
    """Tests for qmix_mixing_network function."""

    def test_basic_computation(self):
        """Test basic QMIX mixing computation."""
        num_agents = 3
        batch_size = 4

        individual_q = jnp.ones((num_agents, batch_size))
        state = jnp.zeros((batch_size, 10))  # State dimension 10

        # Dummy hypernetwork params (not used in simplified implementation)
        hyper_w1 = jnp.zeros((10, num_agents * 32))
        hyper_w2 = jnp.zeros((10, 32 * 1))
        hyper_b1 = jnp.zeros((10, 32))
        hyper_b2 = jnp.zeros((10, 1))

        q_tot = qmix_mixing_network(
            individual_q, state, hyper_w1, hyper_w2, hyper_b1, hyper_b2,
            mixing_embed_dim=32
        )

        assert q_tot.shape == (batch_size,)

    def test_fallback_to_vdn(self):
        """Test that simplified QMIX falls back to VDN (sum)."""
        num_agents = 3
        batch_size = 4

        individual_q = jnp.array([
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
            [2.0, 1.0, 0.5, 0.0],
        ])
        state = jnp.zeros((batch_size, 10))

        q_tot = qmix_mixing_network(
            individual_q, state, None, None, None, None,
            mixing_embed_dim=32
        )

        # Should be sum of individual Q-values (VDN fallback)
        expected = jnp.sum(individual_q, axis=0)
        np.testing.assert_allclose(q_tot, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
