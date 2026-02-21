"""
Unit tests for intrinsic reward construction (CF-IMPL-005)

Tests the following test criteria:
- [x] 内在奖励 = -后悔
- [x] 最优动作时内在奖励 = 0
- [x] 非最优动作时内在奖励 < 0
- [x] 梯度流正确

Reference: Eq.10 from Counterfactual/cf_method
"""

import pytest
import sys
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
from functools import partial

from socialjax.algorithms.cf.intrinsic_reward import (
    compute_intrinsic_reward,
    compute_intrinsic_reward_from_cf,
    compute_scaled_intrinsic_reward,
    get_intrinsic_reward_statistics,
    compute_intrinsic_reward_gradient,
)
from socialjax.algorithms.cf.regret import compute_counterfactual_regret


class TestComputeIntrinsicReward:
    """Test the main intrinsic reward computation (Eq.10)"""

    def test_intrinsic_equals_negative_regret(self):
        """内在奖励 = -后悔"""
        regret = jnp.array([[0.0, 0.5, 1.0, 2.0]])
        intrinsic = compute_intrinsic_reward(regret)
        expected = jnp.array([[0.0, -0.5, -1.0, -2.0]])
        assert jnp.allclose(intrinsic, expected)

    def test_intrinsic_zero_when_optimal(self):
        """最优动作时内在奖励 = 0"""
        regret = jnp.array([[0.0, 0.0, 0.0]])  # All optimal
        intrinsic = compute_intrinsic_reward(regret)
        assert jnp.allclose(intrinsic, 0.0, atol=1e-6)

    def test_intrinsic_negative_when_suboptimal(self):
        """非最优动作时内在奖励 < 0"""
        regret = jnp.array([[0.5, 1.0, 2.5]])  # All suboptimal
        intrinsic = compute_intrinsic_reward(regret)
        assert jnp.all(intrinsic < 0)

    def test_intrinsic_always_non_positive(self):
        """内在奖励始终 <= 0"""
        # Test various regret values
        regret = jnp.array([[0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]])
        intrinsic = compute_intrinsic_reward(regret)
        assert jnp.all(intrinsic <= 0)

    def test_output_shape_matches_input(self):
        """输出形状与输入一致"""
        batch_sizes = [1, 8, 32, 64]
        num_agents_list = [2, 3, 4, 5, 7]

        for batch in batch_sizes:
            for num_agents in num_agents_list:
                regret = jnp.ones((batch, num_agents)) * 0.5
                intrinsic = compute_intrinsic_reward(regret)
                assert intrinsic.shape == (batch, num_agents), \
                    f"Shape mismatch for batch={batch}, num_agents={num_agents}"

    def test_batch_computation(self):
        """批处理计算正确"""
        regret = jnp.array([
            [0.0, 0.5],  # batch 0: agent 0 optimal, agent 1 suboptimal
            [1.0, 0.0],  # batch 1: agent 0 suboptimal, agent 1 optimal
            [0.25, 0.75],  # batch 2: both suboptimal
        ])
        intrinsic = compute_intrinsic_reward(regret)
        expected = jnp.array([
            [0.0, -0.5],
            [-1.0, 0.0],
            [-0.25, -0.75],
        ])
        assert jnp.allclose(intrinsic, expected)

    def test_single_agent(self):
        """单agent场景"""
        regret = jnp.array([[0.5]])  # Single agent, batch=1
        intrinsic = compute_intrinsic_reward(regret)
        assert intrinsic.shape == (1, 1)
        assert jnp.allclose(intrinsic, jnp.array([[-0.5]]))

    def test_large_regret_values(self):
        """大后悔值测试"""
        regret = jnp.array([[1e3, 1e4, 1e5]])
        intrinsic = compute_intrinsic_reward(regret)
        expected = jnp.array([[-1e3, -1e4, -1e5]])
        assert jnp.allclose(intrinsic, expected)

    def test_small_regret_values(self):
        """小后悔值测试"""
        regret = jnp.array([[1e-6, 1e-7, 1e-8]])
        intrinsic = compute_intrinsic_reward(regret)
        expected = jnp.array([[-1e-6, -1e-7, -1e-8]])
        assert jnp.allclose(intrinsic, expected, atol=1e-10)

    def test_mixed_optimal_suboptimal(self):
        """混合最优和非最优动作"""
        regret = jnp.array([[0.0, 0.3, 0.0, 0.9]])
        intrinsic = compute_intrinsic_reward(regret)
        # Agent 0, 2 optimal (intrinsic = 0)
        # Agent 1, 3 suboptimal (intrinsic < 0)
        assert jnp.allclose(intrinsic[0, 0], 0.0, atol=1e-6)
        assert jnp.allclose(intrinsic[0, 2], 0.0, atol=1e-6)
        assert intrinsic[0, 1] < 0
        assert intrinsic[0, 3] < 0


class TestIntrinsicRewardFromCF:
    """Test convenience function combining regret and intrinsic computation"""

    def test_matches_separate_computation(self):
        """结果应与分别计算一致"""
        collective_cf_rewards = jnp.array([
            [[1.0, 2.0]],  # agent 0, actions 0,1, batch dim
            [[1.5, 1.5]],  # agent 1
        ])  # Shape: [2, 2, 1]

        actual_collective = jnp.array([[1.0, 1.5]])  # [1, 2]

        regret_separate = compute_counterfactual_regret(
            collective_cf_rewards, actual_collective
        )
        intrinsic_separate = compute_intrinsic_reward(regret_separate)

        regret_combined, intrinsic_combined = compute_intrinsic_reward_from_cf(
            collective_cf_rewards, actual_collective
        )

        assert jnp.allclose(regret_separate, regret_combined)
        assert jnp.allclose(intrinsic_separate, intrinsic_combined)

    def test_pipeline_m4_to_m5(self):
        """测试从M4(后悔)到M5(内在奖励)的流程"""
        # Setup: 2 agents, 3 actions, batch 4
        collective_cf_rewards = jnp.array([
            [[1.0, 2.0, 1.5, 3.0],
             [1.5, 2.5, 2.0, 3.5],
             [1.0, 1.5, 1.2, 2.0]],  # agent 0
            [[2.0, 1.5, 1.8, 2.5],
             [2.5, 2.0, 2.3, 3.0],
             [2.2, 1.8, 2.0, 2.8]],  # agent 1
        ])  # Shape: [2, 3, 4]

        # Agent 0: max CF = [2.0, 2.5, 1.5, 3.5]
        # Agent 1: max CF = [2.5, 3.0, 2.3, 3.0]

        actual_collective = jnp.array([
            [1.0, 2.0],   # batch 0
            [2.5, 3.0],   # batch 1 (optimal for both)
            [1.2, 2.3],   # batch 2
            [3.5, 3.0],   # batch 3 (optimal for both)
        ])  # [4, 2]

        regret, intrinsic = compute_intrinsic_reward_from_cf(
            collective_cf_rewards, actual_collective
        )

        # Verify regret >= 0
        assert jnp.all(regret >= -1e-6)

        # Verify intrinsic = -regret
        assert jnp.allclose(intrinsic, -regret)

        # Verify shapes
        assert regret.shape == (4, 2)
        assert intrinsic.shape == (4, 2)


class TestScaledIntrinsicReward:
    """Test scaled intrinsic reward computation"""

    def test_alpha_scaling(self):
        """测试alpha缩放因子"""
        regret = jnp.array([[1.0]])

        for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
            scaled = compute_scaled_intrinsic_reward(regret, alpha=alpha)
            expected = alpha * (-1.0)
            assert jnp.allclose(scaled, expected), f"Failed for alpha={alpha}"

    def test_default_alpha(self):
        """默认alpha=1.0"""
        regret = jnp.array([[0.5]])
        scaled = compute_scaled_intrinsic_reward(regret)
        expected = -0.5
        assert jnp.allclose(scaled, expected)

    def test_zero_alpha(self):
        """alpha=0时内在奖励为0"""
        regret = jnp.array([[1.0]])
        scaled = compute_scaled_intrinsic_reward(regret, alpha=0.0)
        assert jnp.allclose(scaled, 0.0)

    def test_batch_alpha_scaling(self):
        """批处理alpha缩放"""
        regret = jnp.array([[1.0, 2.0, 3.0]])
        alpha = 2.0
        scaled = compute_scaled_intrinsic_reward(regret, alpha=alpha)
        expected = jnp.array([[-2.0, -4.0, -6.0]])
        assert jnp.allclose(scaled, expected)


class TestIntrinsicRewardStatistics:
    """Test statistics computation"""

    def test_statistics_computation(self):
        """统计量计算"""
        intrinsic = jnp.array([
            [0.0, -0.5, -1.0],   # batch 0
            [-0.25, 0.0, -0.5],  # batch 1
            [0.0, -0.1, -2.0],   # batch 2
        ])  # [3, 3]

        mean_intr, min_intr, zero_ratio = get_intrinsic_reward_statistics(intrinsic)

        # Mean per agent
        expected_mean = jnp.array([
            (-0.25) / 3,  # agent 0: (0 + -0.25 + 0) / 3
            (-0.6) / 3,   # agent 1: (-0.5 + 0 + -0.1) / 3
            (-3.5) / 3,   # agent 2: (-1.0 + -0.5 + -2.0) / 3
        ])
        assert jnp.allclose(mean_intr, expected_mean, atol=1e-6)

        # Min per agent
        expected_min = jnp.array([-0.25, -0.5, -2.0])
        assert jnp.allclose(min_intr, expected_min)

        # Zero ratio (intrinsic ≈ 0 means optimal)
        expected_zero = jnp.array([
            2/3,  # agent 0: 2 out of 3 are zero
            1/3,  # agent 1: 1 out of 3 is zero
            0/3,  # agent 2: none are zero
        ])
        assert jnp.allclose(zero_ratio, expected_zero, atol=1e-6)

    def test_all_optimal_statistics(self):
        """所有动作都最优时的统计"""
        intrinsic = jnp.zeros((10, 3))  # All zero
        mean_intr, min_intr, zero_ratio = get_intrinsic_reward_statistics(intrinsic)

        assert jnp.allclose(mean_intr, 0.0)
        assert jnp.allclose(min_intr, 0.0)
        assert jnp.allclose(zero_ratio, 1.0)

    def test_all_suboptimal_statistics(self):
        """所有动作都非最优时的统计"""
        intrinsic = -jnp.ones((5, 2))  # All -1.0
        mean_intr, min_intr, zero_ratio = get_intrinsic_reward_statistics(intrinsic)

        assert jnp.allclose(mean_intr, -1.0)
        assert jnp.allclose(min_intr, -1.0)
        assert jnp.allclose(zero_ratio, 0.0)


class TestGradientFlow:
    """测试梯度流正确性"""

    def test_gradient_through_intrinsic(self):
        """梯度能正确通过内在奖励"""
        regret = jnp.array([[0.5, 1.0, 2.0]])

        # Define loss as sum of intrinsic rewards
        def loss_fn(regret):
            intrinsic = compute_intrinsic_reward(regret)
            return jnp.sum(intrinsic)

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(regret)

        # Since intrinsic = -regret, gradient should be -1 everywhere
        expected_grad = -jnp.ones_like(regret)
        assert jnp.allclose(grad, expected_grad)

    def test_gradient_through_scaled_intrinsic(self):
        """梯度通过缩放内在奖励正确传递"""
        regret = jnp.array([[1.0]])
        alpha = 2.0

        def loss_fn(regret):
            scaled = compute_scaled_intrinsic_reward(regret, alpha=alpha)
            return jnp.sum(scaled)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(regret)

        # Gradient should be -alpha = -2.0
        assert jnp.allclose(grad, -alpha)

    def test_gradient_through_full_pipeline(self):
        """测试完整流程的梯度: CF奖励 -> 后悔 -> 内在奖励"""
        # Setup simple case
        collective_cf_rewards = jnp.array([
            [[1.0, 2.0, 1.5]],  # agent 0
        ])  # [1, 3, 1]

        actual_collective = jnp.array([[1.0]])  # [1, 1]

        def loss_fn(actual):
            regret = compute_counterfactual_regret(
                collective_cf_rewards, actual
            )
            intrinsic = compute_intrinsic_reward(regret)
            return jnp.sum(intrinsic)

        # This should be differentiable w.r.t. actual_collective
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(actual_collective)

        # Gradient should exist and be valid
        assert jnp.isfinite(grad).all()

    def test_stop_gradient_independence(self):
        """验证梯度是独立的，不受其他变量影响"""
        regret = jnp.array([[0.5, 1.0]])

        # Compute gradient w.r.t. regret only
        def loss_fn(r):
            return jnp.sum(compute_intrinsic_reward(r))

        grad = jax.grad(loss_fn)(regret)

        # Gradient should be constant -1 regardless of regret value
        assert jnp.allclose(grad, -jnp.ones_like(regret))


class TestJITCompilation:
    """测试JIT编译"""

    def test_jit_compute_intrinsic(self):
        """JIT编译内在奖励计算"""
        regret = jnp.array([[0.5, 1.0]])

        jit_intrinsic = jax.jit(compute_intrinsic_reward)
        result = jit_intrinsic(regret)

        expected = jnp.array([[-0.5, -1.0]])
        assert jnp.allclose(result, expected)

    def test_jit_scaled_intrinsic(self):
        """JIT编译缩放内在奖励"""
        regret = jnp.array([[1.0]])

        jit_scaled = jax.jit(compute_scaled_intrinsic_reward)
        result = jit_scaled(regret, alpha=2.0)

        assert jnp.allclose(result, -2.0)

    def test_jit_statistics(self):
        """JIT编译统计计算"""
        intrinsic = jnp.array([[0.0, -0.5], [-0.25, 0.0]])

        jit_stats = jax.jit(get_intrinsic_reward_statistics)
        mean_intr, min_intr, zero_ratio = jit_stats(intrinsic)

        assert jnp.isfinite(mean_intr).all()
        assert jnp.isfinite(min_intr).all()
        assert jnp.isfinite(zero_ratio).all()


class TestEdgeCases:
    """测试边界情况"""

    def test_single_batch(self):
        """batch_size=1"""
        regret = jnp.array([[0.5]])
        intrinsic = compute_intrinsic_reward(regret)
        assert intrinsic.shape == (1, 1)
        assert jnp.allclose(intrinsic, -0.5)

    def test_large_batch(self):
        """大batch测试"""
        batch = 1000
        num_agents = 5
        regret = jnp.ones((batch, num_agents)) * 0.5
        intrinsic = compute_intrinsic_reward(regret)

        assert intrinsic.shape == (batch, num_agents)
        assert jnp.allclose(intrinsic, -0.5)

    def test_many_agents(self):
        """多agent测试"""
        batch = 4
        num_agents = 20
        regret = jnp.ones((batch, num_agents)) * 0.5
        intrinsic = compute_intrinsic_reward(regret)

        assert intrinsic.shape == (batch, num_agents)
        assert jnp.allclose(intrinsic, -0.5)

    def test_zero_regret_everywhere(self):
        """所有后悔为0"""
        regret = jnp.zeros((5, 3))
        intrinsic = compute_intrinsic_reward(regret)
        assert jnp.allclose(intrinsic, 0.0)

    def test_all_same_regret(self):
        """所有后悔相同"""
        regret = jnp.ones((10, 5)) * 2.5
        intrinsic = compute_intrinsic_reward(regret)
        assert jnp.allclose(intrinsic, -2.5)


class TestIntegrationWithRegret:
    """测试与后悔模块的集成"""

    def test_full_m4_to_m5_pipeline(self):
        """完整M4到M5流程测试"""
        # Create CF rewards for 3 agents, 4 actions, batch 2
        collective_cf_rewards = jnp.array([
            # Agent 0
            [[1.0, 2.0],   # action 0
             [1.5, 2.5],   # action 1
             [2.0, 3.0],   # action 2 (max)
             [1.2, 2.2]],  # action 3
            # Agent 1
            [[2.0, 1.5],
             [2.5, 2.0],
             [2.2, 1.8],
             [3.0, 2.5]],  # action 3 (max)
            # Agent 2
            [[1.5, 1.5],
             [2.0, 2.0],
             [2.5, 2.5],   # action 2 (max)
             [1.8, 1.8]],
        ])  # [3, 4, 2]

        actual_collective = jnp.array([
            [1.0, 2.0, 1.5],  # batch 0: suboptimal for all
            [2.0, 2.5, 2.0],  # batch 1: optimal for agent 1
        ])  # [2, 3]

        # Compute regret (M4)
        regret = compute_counterfactual_regret(
            collective_cf_rewards, actual_collective
        )

        # Compute intrinsic reward (M5)
        intrinsic = compute_intrinsic_reward(regret)

        # Verify properties
        assert jnp.all(regret >= -1e-6), "Regret should be non-negative"
        assert jnp.all(intrinsic <= 1e-6), "Intrinsic should be non-positive"
        assert jnp.allclose(intrinsic, -regret), "Intrinsic should equal -regret"

        # Check specific values
        # Batch 0, Agent 0: max CF = 2.0, actual = 1.0, regret = 1.0, intrinsic = -1.0
        assert jnp.allclose(regret[0, 0], 1.0, atol=1e-6)
        assert jnp.allclose(intrinsic[0, 0], -1.0, atol=1e-6)

        # Batch 1, Agent 1: max CF = 2.5, actual = 2.5, regret = 0, intrinsic = 0
        assert jnp.allclose(regret[1, 1], 0.0, atol=1e-6)
        assert jnp.allclose(intrinsic[1, 1], 0.0, atol=1e-6)

    def test_intrinsic_reward_gradient_for_training(self):
        """验证内在奖励的梯度可以用于策略训练"""
        # Simulate a simple policy loss that uses intrinsic reward
        collective_cf_rewards = jnp.array([
            [[1.0, 2.0, 1.5]],
        ])  # [1, 3, 1]

        actual_collective = jnp.array([[1.5]])  # [1, 1]

        def policy_loss(actual):
            regret = compute_counterfactual_regret(
                collective_cf_rewards, actual
            )
            intrinsic = compute_intrinsic_reward(regret)
            # Simple loss: minimize negative intrinsic (maximize intrinsic)
            return -jnp.sum(intrinsic)

        # Compute gradient w.r.t. actual_collective
        grad_fn = jax.grad(policy_loss)
        grad = grad_fn(actual_collective)

        # Gradient should exist and be finite
        assert jnp.isfinite(grad).all()
        assert grad.shape == actual_collective.shape
