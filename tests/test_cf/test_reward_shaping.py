"""
Unit tests for Reward Shaping module (M6)

Tests for Eq.11: r̂ = r^{ex} + α * r^{in}
"""

import pytest
import sys
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
from jax import grad
from functools import partial

from socialjax.algorithms.cf.reward_shaping import (
    compute_shaped_reward,
    compute_shaped_reward_from_regret,
    compute_alpha_n_minus_1,
    compute_shaped_reward_auto_alpha,
    normalize_shaped_reward,
    compute_shaped_reward_normalized,
    get_shaped_reward_statistics,
    compute_shaped_reward_with_components,
    verify_shaped_reward_properties,
    compute_shaped_reward_gradient,
    compute_shaped_reward_jit,
    DEFAULT_ALPHA,
)


class TestComputeShapedReward:
    """Test shaped reward computation (Eq.11)"""

    def test_basic_formula(self):
        """Test basic formula: shaped = extrinsic + alpha * intrinsic"""
        extrinsic = jnp.array([[1.0, 0.5]])
        intrinsic = jnp.array([[0.0, -0.5]])
        alpha = 2.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        # Expected: [1.0 + 0*2, 0.5 + (-0.5)*2] = [1.0, -0.5]
        expected = jnp.array([[1.0, -0.5]])
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_alpha_zero(self):
        """When alpha=0, shaped = extrinsic"""
        extrinsic = jnp.array([[1.0, 2.0, 3.0]])
        intrinsic = jnp.array([[-1.0, -2.0, -3.0]])
        alpha = 0.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        assert jnp.allclose(shaped, extrinsic, atol=1e-6)

    def test_alpha_one(self):
        """When alpha=1, shaped = extrinsic + intrinsic"""
        extrinsic = jnp.array([[1.0, 2.0]])
        intrinsic = jnp.array([[-0.5, -1.0]])
        alpha = 1.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        expected = extrinsic + intrinsic
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_large_alpha(self):
        """Test with large alpha value"""
        extrinsic = jnp.array([[1.0, 1.0]])
        intrinsic = jnp.array([[-0.1, -0.1]])
        alpha = 10.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        expected = jnp.array([[0.0, 0.0]])  # 1.0 + 10*(-0.1) = 0
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_batch_processing(self):
        """Test with batch of rewards"""
        extrinsic = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        intrinsic = jnp.array([[-0.5, -1.0], [-1.5, -2.0], [-2.5, -3.0]])
        alpha = 1.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        assert shaped.shape == (3, 2)
        expected = extrinsic + intrinsic
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_multi_agent(self):
        """Test with different number of agents"""
        # 5 agents
        extrinsic = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        intrinsic = jnp.array([[0.0, -0.5, -1.0, -1.5, -2.0]])
        alpha = 1.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        assert shaped.shape == (1, 5)
        expected = jnp.array([[1.0, 1.5, 2.0, 2.5, 3.0]])
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_positive_intrinsic(self):
        """Test with positive intrinsic reward (edge case)"""
        # Although intrinsic should be negative, function should still work
        extrinsic = jnp.array([[1.0, 2.0]])
        intrinsic = jnp.array([[0.5, 1.0]])
        alpha = 2.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        expected = jnp.array([[2.0, 4.0]])  # 1+2*0.5=2, 2+2*1=4
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_negative_extrinsic(self):
        """Test with negative extrinsic reward"""
        extrinsic = jnp.array([[-1.0, -2.0]])
        intrinsic = jnp.array([[0.0, -0.5]])
        alpha = 1.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        expected = jnp.array([[-1.0, -2.5]])
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_default_alpha(self):
        """Test with default alpha value"""
        extrinsic = jnp.array([[1.0]])
        intrinsic = jnp.array([[-0.5]])

        shaped = compute_shaped_reward(extrinsic, intrinsic)

        # With default alpha=1.0
        expected = jnp.array([[0.5]])
        assert jnp.allclose(shaped, expected, atol=1e-6)


class TestComputeShapedRewardFromRegret:
    """Test shaped reward computation from regret"""

    def test_basic_from_regret(self):
        """Test shaped = extrinsic + alpha * (-regret)"""
        extrinsic = jnp.array([[1.0, 2.0]])
        regret = jnp.array([[0.5, 1.0]])
        alpha = 2.0

        shaped = compute_shaped_reward_from_regret(extrinsic, regret, alpha)

        # intrinsic = -regret = [-0.5, -1.0]
        # shaped = [1-1, 2-2] = [0, 0]
        expected = jnp.array([[0.0, 0.0]])
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_zero_regret(self):
        """When regret=0, shaped = extrinsic"""
        extrinsic = jnp.array([[1.0, 2.0, 3.0]])
        regret = jnp.array([[0.0, 0.0, 0.0]])
        alpha = 2.0

        shaped = compute_shaped_reward_from_regret(extrinsic, regret, alpha)

        assert jnp.allclose(shaped, extrinsic, atol=1e-6)

    def test_matches_shaped_reward(self):
        """Verify it matches compute_shaped_reward with negative regret"""
        extrinsic = jnp.array([[1.0, 2.0, 3.0]])
        regret = jnp.array([[0.5, 1.0, 1.5]])
        alpha = 3.0

        shaped_from_regret = compute_shaped_reward_from_regret(
            extrinsic, regret, alpha
        )
        shaped_direct = compute_shaped_reward(
            extrinsic, -regret, alpha
        )

        assert jnp.allclose(shaped_from_regret, shaped_direct, atol=1e-6)


class TestComputeAlphaNMinus1:
    """Test alpha = N-1 computation"""

    def test_three_agents(self):
        """With 3 agents, alpha = 2"""
        alpha = compute_alpha_n_minus_1(3)
        assert alpha == 2.0

    def test_seven_agents(self):
        """With 7 agents, alpha = 6"""
        alpha = compute_alpha_n_minus_1(7)
        assert alpha == 6.0

    def test_two_agents(self):
        """With 2 agents, alpha = 1"""
        alpha = compute_alpha_n_minus_1(2)
        assert alpha == 1.0

    def test_one_agent(self):
        """With 1 agent, alpha = 0 (edge case)"""
        alpha = compute_alpha_n_minus_1(1)
        assert alpha == 0.0


class TestComputeShapedRewardAutoAlpha:
    """Test shaped reward with automatic alpha"""

    def test_auto_alpha_n_minus_1(self):
        """Test that auto alpha = N-1 is used"""
        extrinsic = jnp.array([[1.0, 2.0, 3.0]])
        intrinsic = jnp.array([[0.0, -0.5, -1.0]])
        num_agents = 3

        shaped = compute_shaped_reward_auto_alpha(
            extrinsic, intrinsic, num_agents
        )

        # With alpha = 2 (N-1 = 2)
        # shaped = [1+0, 2+2*(-0.5), 3+2*(-1)] = [1, 1, 1]
        expected = jnp.array([[1.0, 1.0, 1.0]])
        assert jnp.allclose(shaped, expected, atol=1e-6)


class TestNormalizeShapedReward:
    """Test shaped reward normalization"""

    def test_normalization_zero_mean(self):
        """Normalized rewards should have approximately zero mean"""
        shaped = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        normalized = normalize_shaped_reward(shaped)

        mean = jnp.mean(normalized, axis=0)
        assert jnp.allclose(mean, 0.0, atol=1e-5)

    def test_normalization_unit_variance(self):
        """Normalized rewards should have approximately unit variance"""
        shaped = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        normalized = normalize_shaped_reward(shaped)

        std = jnp.std(normalized, axis=0)
        assert jnp.allclose(std, 1.0, atol=1e-5)

    def test_normalization_preserves_shape(self):
        """Normalization should preserve shape"""
        shaped = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        normalized = normalize_shaped_reward(shaped)

        assert normalized.shape == shaped.shape

    def test_normalization_single_value(self):
        """Test normalization with single value (edge case)"""
        shaped = jnp.array([[1.0]])

        normalized = normalize_shaped_reward(shaped)

        # With single value, should return zeros
        assert jnp.allclose(normalized, 0.0, atol=1e-5)


class TestComputeShapedRewardNormalized:
    """Test combined shaped reward computation with normalization"""

    def test_normalized_combined(self):
        """Test that combined function produces normalized output"""
        extrinsic = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        intrinsic = jnp.array([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]])
        alpha = 1.0

        normalized = compute_shaped_reward_normalized(
            extrinsic, intrinsic, alpha
        )

        # Check zero mean
        mean = jnp.mean(normalized, axis=0)
        assert jnp.allclose(mean, 0.0, atol=1e-5)


class TestGetShapedRewardStatistics:
    """Test statistics computation"""

    def test_statistics_shape(self):
        """Test that statistics have correct shape"""
        shaped = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        mean, std, min_r, max_r = get_shaped_reward_statistics(shaped)

        assert mean.shape == (2,)
        assert std.shape == (2,)
        assert min_r.shape == (2,)
        assert max_r.shape == (2,)

    def test_statistics_values(self):
        """Test that statistics are computed correctly"""
        shaped = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        mean, std, min_r, max_r = get_shaped_reward_statistics(shaped)

        # Mean: [3.0, 4.0]
        assert jnp.allclose(mean, jnp.array([3.0, 4.0]), atol=1e-6)
        # Min: [1.0, 2.0]
        assert jnp.allclose(min_r, jnp.array([1.0, 2.0]), atol=1e-6)
        # Max: [5.0, 6.0]
        assert jnp.allclose(max_r, jnp.array([5.0, 6.0]), atol=1e-6)


class TestComputeShapedRewardWithComponents:
    """Test shaped reward with component breakdown"""

    def test_components_shape(self):
        """Test that all outputs have correct shape"""
        extrinsic = jnp.array([[1.0, 2.0]])
        intrinsic = jnp.array([[-0.5, -1.0]])
        alpha = 2.0

        shaped, ext_comp, int_comp = compute_shaped_reward_with_components(
            extrinsic, intrinsic, alpha
        )

        assert shaped.shape == (1, 2)
        assert ext_comp.shape == (1, 2)
        assert int_comp.shape == (1, 2)

    def test_components_values(self):
        """Test that components are computed correctly"""
        extrinsic = jnp.array([[1.0, 2.0]])
        intrinsic = jnp.array([[-0.5, -1.0]])
        alpha = 2.0

        shaped, ext_comp, int_comp = compute_shaped_reward_with_components(
            extrinsic, intrinsic, alpha
        )

        # Extrinsic component = [1.0, 2.0]
        assert jnp.allclose(ext_comp, extrinsic, atol=1e-6)
        # Intrinsic component = 2 * [-0.5, -1.0] = [-1.0, -2.0]
        assert jnp.allclose(int_comp, jnp.array([[-1.0, -2.0]]), atol=1e-6)
        # Shaped = [1-1, 2-2] = [0, 0]
        assert jnp.allclose(shaped, jnp.array([[0.0, 0.0]]), atol=1e-6)

    def test_components_sum_to_shaped(self):
        """Test that ext + int = shaped"""
        extrinsic = jnp.array([[1.0, 2.0, 3.0]])
        intrinsic = jnp.array([[-0.1, -0.2, -0.3]])
        alpha = 5.0

        shaped, ext_comp, int_comp = compute_shaped_reward_with_components(
            extrinsic, intrinsic, alpha
        )

        assert jnp.allclose(shaped, ext_comp + int_comp, atol=1e-6)


class TestVerifyShapedRewardProperties:
    """Test property verification"""

    def test_valid_properties(self):
        """Test that valid shaped reward passes verification"""
        extrinsic = jnp.array([[1.0, 2.0]])
        intrinsic = jnp.array([[-0.5, -1.0]])
        alpha = 2.0
        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        is_valid, message = verify_shaped_reward_properties(
            shaped, extrinsic, intrinsic, alpha
        )

        assert is_valid
        assert "satisfied" in message.lower()

    def test_invalid_formula(self):
        """Test that wrong formula is detected"""
        extrinsic = jnp.array([[1.0, 2.0]])
        intrinsic = jnp.array([[-0.5, -1.0]])
        alpha = 2.0
        shaped = jnp.array([[100.0, 200.0]])  # Wrong values

        is_valid, message = verify_shaped_reward_properties(
            shaped, extrinsic, intrinsic, alpha
        )

        assert not is_valid
        assert "formula" in message.lower()


class TestGradientFlow:
    """Test gradient flow through shaped reward"""

    def test_gradient_extrinsic(self):
        """Test gradient w.r.t. extrinsic reward"""
        def fn(ext):
            intrinsic = jnp.array([[-1.0]])
            return jnp.sum(compute_shaped_reward(ext, intrinsic, alpha=2.0))

        grad_fn = grad(fn)
        ext = jnp.array([[1.0]])

        g = grad_fn(ext)
        # Gradient should be 1 (from r̂ = r_ex + alpha * r_in)
        assert jnp.allclose(g, 1.0, atol=1e-6)

    def test_gradient_intrinsic(self):
        """Test gradient w.r.t. intrinsic reward"""
        def fn(intr):
            extrinsic = jnp.array([[1.0]])
            return jnp.sum(compute_shaped_reward(extrinsic, intr, alpha=2.0))

        grad_fn = grad(fn)
        intrinsic = jnp.array([[-1.0]])

        g = grad_fn(intrinsic)
        # Gradient should be alpha = 2.0
        assert jnp.allclose(g, 2.0, atol=1e-6)

    def test_gradient_chain_through_regret(self):
        """Test gradient flows through regret -> intrinsic -> shaped"""
        # Simulate: shaped = r_ex + alpha * (-regret)
        def shaped_from_regret(ext, regret):
            intrinsic = -regret
            return jnp.sum(compute_shaped_reward(ext, intrinsic, alpha=2.0))

        grad_fn = jax.grad(shaped_from_regret, argnums=(0, 1))
        ext = jnp.array([[1.0]])
        regret = jnp.array([[0.5]])

        g_ext, g_regret = grad_fn(ext, regret)

        # Gradient w.r.t. extrinsic = 1
        assert jnp.allclose(g_ext, 1.0, atol=1e-6)
        # Gradient w.r.t. regret = -alpha = -2.0
        assert jnp.allclose(g_regret, -2.0, atol=1e-6)


class TestJITCompilation:
    """Test JIT compilation"""

    def test_jit_basic(self):
        """Test that JIT compilation works"""
        extrinsic = jnp.array([[1.0, 2.0]])
        intrinsic = jnp.array([[-0.5, -1.0]])
        alpha = 2.0

        shaped = compute_shaped_reward_jit(extrinsic, intrinsic, alpha)

        expected = jnp.array([[0.0, 0.0]])
        assert jnp.allclose(shaped, expected, atol=1e-6)

    def test_jit_matches_non_jit(self):
        """Test that JIT and non-JIT produce same results"""
        extrinsic = jnp.array([[1.0, 2.0, 3.0]])
        intrinsic = jnp.array([[-0.1, -0.2, -0.3]])
        alpha = 5.0

        shaped_jit = compute_shaped_reward_jit(extrinsic, intrinsic, alpha)
        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        assert jnp.allclose(shaped_jit, shaped, atol=1e-6)


class TestNumericalStability:
    """Test numerical stability with edge cases"""

    def test_large_values(self):
        """Test with large reward values"""
        extrinsic = jnp.array([[1000.0, 2000.0]])
        intrinsic = jnp.array([[-500.0, -1000.0]])
        alpha = 1.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        assert not jnp.any(jnp.isnan(shaped))
        assert not jnp.any(jnp.isinf(shaped))
        expected = jnp.array([[500.0, 1000.0]])
        assert jnp.allclose(shaped, expected, atol=1e-3)

    def test_small_values(self):
        """Test with small reward values"""
        extrinsic = jnp.array([[1e-6, 2e-6]])
        intrinsic = jnp.array([[-0.5e-6, -1e-6]])
        alpha = 1.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        assert not jnp.any(jnp.isnan(shaped))
        assert not jnp.any(jnp.isinf(shaped))

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative values"""
        extrinsic = jnp.array([[-10.0, 10.0, 0.0]])
        intrinsic = jnp.array([[0.0, -5.0, 5.0]])  # positive intrinsic (edge case)
        alpha = 2.0

        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        assert not jnp.any(jnp.isnan(shaped))
        assert not jnp.any(jnp.isinf(shaped))


class TestIntegrationWithPreviousModules:
    """Test integration with M4 (Regret) and M5 (Intrinsic)"""

    def test_pipeline_regret_to_shaped(self):
        """Test complete pipeline from regret to shaped reward"""
        # Import from previous modules
        from socialjax.algorithms.cf.regret import compute_counterfactual_regret
        from socialjax.algorithms.cf.intrinsic_reward import compute_intrinsic_reward

        # Simulate CF rewards and actual collective rewards
        # collective_cf_rewards shape: [num_agents, action_dim, batch]
        # For each agent, we have action_dim=3 possible counterfactual actions
        # and batch=1 sample
        collective_cf_rewards = jnp.array([
            [[1.0], [2.0], [1.5]],  # Agent 0's CF rewards (3 actions, 1 batch)
            [[0.5], [1.0], [0.8]],  # Agent 1's CF rewards (3 actions, 1 batch)
            [[2.0], [3.0], [2.5]],  # Agent 2's CF rewards (3 actions, 1 batch)
        ])  # Shape: [num_agents=3, action_dim=3, batch=1]

        actual_collective = jnp.array([[3.0, 2.0, 4.0]])  # [batch=1, num_agents=3]

        extrinsic = jnp.array([[1.0, 0.5, 2.0]])  # [batch, num_agents]
        alpha = 2.0

        # Step 1: Compute regret (M4)
        regret = compute_counterfactual_regret(
            collective_cf_rewards, actual_collective
        )

        # Step 2: Compute intrinsic reward (M5)
        intrinsic = compute_intrinsic_reward(regret)

        # Step 3: Compute shaped reward (M6)
        shaped = compute_shaped_reward(extrinsic, intrinsic, alpha)

        # Verify pipeline
        assert shaped.shape == (1, 3)
        assert not jnp.any(jnp.isnan(shaped))
        assert not jnp.any(jnp.isinf(shaped))

    def test_shaped_reward_reduces_with_regret(self):
        """Test that higher regret leads to lower shaped reward"""
        from socialjax.algorithms.cf.intrinsic_reward import compute_intrinsic_reward

        extrinsic = jnp.array([[1.0, 1.0]])
        regret_low = jnp.array([[0.1, 0.1]])
        regret_high = jnp.array([[1.0, 1.0]])
        alpha = 2.0

        intrinsic_low = compute_intrinsic_reward(regret_low)
        intrinsic_high = compute_intrinsic_reward(regret_high)

        shaped_low = compute_shaped_reward(extrinsic, intrinsic_low, alpha)
        shaped_high = compute_shaped_reward(extrinsic, intrinsic_high, alpha)

        # Higher regret should lead to lower shaped reward
        assert jnp.all(shaped_high < shaped_low)


class TestDefaultAlpha:
    """Test default alpha value"""

    def test_default_alpha_value(self):
        """Verify DEFAULT_ALPHA constant"""
        assert DEFAULT_ALPHA == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
