"""Unit tests for GAE (Generalized Advantage Estimation) utilities.

Tests cover:
- compute_gae function with various trajectory configurations
- compute_gae_batched for batched computations
- normalize_advantages for advantage normalization
- compute_returns for Monte Carlo returns
- Edge cases and numerical stability
"""

import pytest
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'socialjax')

from socialjax.algorithms.utils.gae import (
    compute_gae,
    compute_gae_batched,
    normalize_advantages,
    compute_returns,
    GAETransition,
)


class TestGAETransition:
    """Tests for GAETransition NamedTuple."""

    def test_create_transition(self):
        """Test creating a GAETransition."""
        dones = jnp.zeros(10)
        values = jnp.ones(10)
        rewards = jnp.ones(10) * 0.1

        traj = GAETransition(done=dones, value=values, reward=rewards)

        assert traj.done.shape == (10,)
        assert traj.value.shape == (10,)
        assert traj.reward.shape == (10,)

    def test_transition_attributes(self):
        """Test accessing transition attributes."""
        traj = GAETransition(
            done=jnp.array([0.0, 1.0]),
            value=jnp.array([1.0, 2.0]),
            reward=jnp.array([0.1, 0.2]),
        )

        assert float(traj.done[1]) == 1.0
        assert float(traj.value[0]) == 1.0
        np.testing.assert_allclose(float(traj.reward[0]), 0.1, rtol=1e-5)


class TestComputeGAE:
    """Tests for compute_gae function."""

    def test_basic_computation(self):
        """Test basic GAE computation."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(1.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value)

        assert advantages.shape == (T,)
        assert targets.shape == (T,)
        assert not jnp.any(jnp.isnan(advantages))
        assert not jnp.any(jnp.isnan(targets))

    def test_episode_termination(self):
        """Test GAE with episode termination."""
        # Episode ends at step 5
        dones = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        values = jnp.ones(10)
        rewards = jnp.ones(10) * 0.1
        last_value = jnp.array(1.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value)

        # Advantage at episode end should not include future values
        # (after done, the bootstrap value is 0)
        assert not jnp.any(jnp.isnan(advantages))

    def test_zero_lambda(self):
        """Test GAE with lambda=0 (TD(0) advantage)."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(1.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value, gae_lambda=0.0)

        # With lambda=0, A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # All values are 1, so A_t = 0.1 + 0.99 * 1 - 1 = 0.09
        expected_advantage = 0.1 + 0.99 * 1 - 1
        np.testing.assert_allclose(
            float(advantages[-1]),
            expected_advantage,
            rtol=1e-5
        )

    def test_lambda_one(self):
        """Test GAE with lambda=1 (Monte Carlo advantage)."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(1.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value, gae_lambda=1.0)

        # With lambda=1, advantages should accumulate all future rewards
        assert not jnp.any(jnp.isnan(advantages))
        assert float(advantages[0]) > float(advantages[-1])  # Earlier = more accumulated

    def test_different_gamma(self):
        """Test GAE with different discount factors."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(1.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)

        # Low gamma
        adv_low, _ = compute_gae(traj, last_value, gamma=0.5, gae_lambda=0.95)
        # High gamma
        adv_high, _ = compute_gae(traj, last_value, gamma=0.99, gae_lambda=0.95)

        # Higher gamma should give larger advantages (more future weight)
        assert float(adv_high[0]) > float(adv_low[0])

    def test_batched_computation(self):
        """Test GAE with batched inputs (2D arrays)."""
        T, B = 20, 8
        dones = jnp.zeros((T, B))
        values = jnp.ones((T, B))
        rewards = jnp.ones((T, B)) * 0.1
        last_values = jnp.ones(B)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_values)

        assert advantages.shape == (T, B)
        assert targets.shape == (T, B)

    def test_jit_compatibility(self):
        """Test that compute_gae is JIT-compatible."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(1.0)
        traj = GAETransition(done=dones, value=values, reward=rewards)

        # Should not raise an error
        jitted_gae = jax.jit(compute_gae)
        advantages, targets = jitted_gae(traj, last_value)

        assert advantages.shape == (T,)
        assert not jnp.any(jnp.isnan(advantages))

    def test_value_targets_equal_advantage_plus_value(self):
        """Test that targets = advantages + values."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(11.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value)

        np.testing.assert_allclose(
            targets,
            advantages + values,
            rtol=1e-5
        )


class TestComputeGAEBatched:
    """Tests for compute_gae_batched function."""

    def test_basic_batched(self):
        """Test basic batched GAE computation."""
        T, B = 10, 4
        dones = jnp.zeros((T, B))
        values = jnp.ones((T, B))
        rewards = jnp.ones((T, B)) * 0.1
        last_values = jnp.ones(B)

        advantages, targets = compute_gae_batched(
            dones, values, rewards, last_values
        )

        assert advantages.shape == (T, B)
        assert targets.shape == (T, B)

    def test_matches_compute_gae(self):
        """Test that batched version matches compute_gae."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(1.0)

        # Using compute_gae
        traj = GAETransition(done=dones, value=values, reward=rewards)
        adv1, targets1 = compute_gae(traj, last_value)

        # Using compute_gae_batched
        adv2, targets2 = compute_gae_batched(dones, values, rewards, last_value)

        np.testing.assert_allclose(adv1, adv2, rtol=1e-5)
        np.testing.assert_allclose(targets1, targets2, rtol=1e-5)

    def test_with_custom_gamma_lambda(self):
        """Test batched GAE with custom gamma and lambda."""
        T, B = 10, 4
        dones = jnp.zeros((T, B))
        values = jnp.ones((T, B))
        rewards = jnp.ones((T, B)) * 0.1
        last_values = jnp.ones(B)

        advantages, targets = compute_gae_batched(
            dones, values, rewards, last_values,
            gamma=0.95, gae_lambda=0.9
        )

        assert advantages.shape == (T, B)


class TestNormalizeAdvantages:
    """Tests for normalize_advantages function."""

    def test_basic_normalization(self):
        """Test basic advantage normalization."""
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_advantages(advantages)

        # Should have zero mean and unit variance
        np.testing.assert_allclose(float(normalized.mean()), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(normalized.std()), 1.0, atol=1e-5)

    def test_preserves_order(self):
        """Test that normalization preserves ordering."""
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_advantages(advantages)

        # Relative ordering should be preserved
        assert float(normalized[0]) < float(normalized[1])
        assert float(normalized[3]) < float(normalized[4])

    def test_constant_advantages(self):
        """Test normalization with constant advantages (zero std)."""
        advantages = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        normalized = normalize_advantages(advantages)

        # Should not produce NaN or Inf due to epsilon
        assert not jnp.any(jnp.isnan(normalized))
        assert not jnp.any(jnp.isinf(normalized))

    def test_negative_advantages(self):
        """Test normalization with negative advantages."""
        advantages = jnp.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        normalized = normalize_advantages(advantages)

        np.testing.assert_allclose(float(normalized.mean()), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(normalized.std()), 1.0, atol=1e-5)

    def test_batched_normalization(self):
        """Test normalization with batched advantages."""
        advantages = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized = normalize_advantages(advantages)

        assert normalized.shape == (3, 2)
        np.testing.assert_allclose(float(normalized.mean()), 0.0, atol=1e-5)

    def test_custom_epsilon(self):
        """Test normalization with custom epsilon."""
        advantages = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        normalized = normalize_advantages(advantages, eps=1e-5)

        assert not jnp.any(jnp.isnan(normalized))

    def test_jit_compatibility(self):
        """Test that normalize_advantages is JIT-compatible."""
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        jitted_norm = jax.jit(normalize_advantages)
        normalized = jitted_norm(advantages)

        assert not jnp.any(jnp.isnan(normalized))


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_basic_returns(self):
        """Test basic discounted return computation."""
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        dones = jnp.zeros(5)

        returns = compute_returns(rewards, dones, gamma=0.99)

        # G_0 = 1 + 0.99*1 + 0.99^2*1 + 0.99^3*1 + 0.99^4*1
        expected_first = sum([0.99**i for i in range(5)])
        np.testing.assert_allclose(float(returns[0]), expected_first, rtol=1e-5)

    def test_returns_with_termination(self):
        """Test returns with episode termination."""
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        dones = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])

        returns = compute_returns(rewards, dones, gamma=0.99)

        # After done=1 at index 2, return at index 3 should restart
        assert float(returns[2]) < float(returns[1])  # Episode ends
        np.testing.assert_allclose(float(returns[3]), 1.99, rtol=1e-5)  # 1.0 + 0.99

    def test_zero_gamma(self):
        """Test returns with gamma=0 (immediate rewards only)."""
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dones = jnp.zeros(5)

        returns = compute_returns(rewards, dones, gamma=0.0)

        np.testing.assert_allclose(returns, rewards, rtol=1e-5)

    def test_batched_returns(self):
        """Test returns with batched inputs."""
        T, B = 10, 4
        rewards = jnp.ones((T, B))
        dones = jnp.zeros((T, B))

        returns = compute_returns(rewards, dones, gamma=0.99)

        assert returns.shape == (T, B)


class TestGAENumericalStability:
    """Tests for numerical stability of GAE computations."""

    def test_large_values(self):
        """Test GAE with large value estimates."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T) * 1000.0
        rewards = jnp.ones(T) * 100.0
        last_value = jnp.array(1000.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value)

        assert not jnp.any(jnp.isnan(advantages))
        assert not jnp.any(jnp.isinf(advantages))

    def test_small_rewards(self):
        """Test GAE with very small rewards."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.ones(T) * 1e-6
        last_value = jnp.array(1.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value)

        assert not jnp.any(jnp.isnan(advantages))

    def test_mixed_positive_negative_rewards(self):
        """Test GAE with mixed positive and negative rewards."""
        T = 10
        dones = jnp.zeros(T)
        values = jnp.ones(T)
        rewards = jnp.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5])
        last_value = jnp.array(1.0)

        traj = GAETransition(done=dones, value=values, reward=rewards)
        advantages, targets = compute_gae(traj, last_value)

        assert not jnp.any(jnp.isnan(advantages))
        assert not jnp.any(jnp.isnan(targets))


class TestGAEIntegration:
    """Integration tests for GAE utilities."""

    def test_full_workflow(self):
        """Test full GAE workflow from trajectory to normalized advantages."""
        T = 20

        # Simulate a trajectory
        dones = jnp.zeros(T)
        dones = dones.at[10].set(1.0)  # Episode ends at step 10
        values = jnp.linspace(0, 1, T)
        rewards = jnp.ones(T) * 0.1
        last_value = jnp.array(1.0)

        # Create trajectory
        traj = GAETransition(done=dones, value=values, reward=rewards)

        # Compute GAE
        advantages, targets = compute_gae(traj, last_value, gamma=0.99, gae_lambda=0.95)

        # Normalize advantages
        normalized = normalize_advantages(advantages)

        # Verify all outputs are valid
        assert not jnp.any(jnp.isnan(advantages))
        assert not jnp.any(jnp.isnan(targets))
        assert not jnp.any(jnp.isnan(normalized))
        np.testing.assert_allclose(float(normalized.mean()), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(normalized.std()), 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
