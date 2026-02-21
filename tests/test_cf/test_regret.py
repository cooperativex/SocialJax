"""
Tests for Counterfactual Regret Calculation (M4)

Tests the regret.py module implementing Eq.9:
Regret_t^i = max_{a^{cf}}[R^{-i,cf}] - R^{-i}

Test criteria:
- Regret >= -1e-6 (allowing floating point errors)
- Regret ≈ 0 when current action is optimal
- Regret > 0 when better action exists
- Output shape: [batch, num_agents]
"""

import pytest
import sys
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np

from socialjax.algorithms.cf.regret import (
    compute_counterfactual_regret,
    compute_regret_with_best_action,
    compute_normalized_regret,
    get_regret_statistics,
)


class TestComputeCounterfactualRegret:
    """Test the main regret computation function"""

    def test_output_shape(self):
        """Output shape should be [batch, num_agents]"""
        # collective_cf_rewards: [num_agents, action_dim, batch]
        # actual_collective: [batch, num_agents]
        num_agents = 3
        action_dim = 4
        batch_size = 8

        collective_cf = jnp.ones((num_agents, action_dim, batch_size))
        actual_collective = jnp.zeros((batch_size, num_agents))

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert regret.shape == (batch_size, num_agents), \
            f"Expected shape {(batch_size, num_agents)}, got {regret.shape}"

    def test_regret_non_negative(self):
        """Regret should always be >= -1e-6 (allowing floating point errors)"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        # Random collective CF rewards
        key = jax.random.PRNGKey(42)
        collective_cf = jax.random.uniform(key, (num_agents, action_dim, batch_size))

        # Actual collective should be less than or equal to max
        actual_collective = jnp.zeros((batch_size, num_agents))

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert jnp.all(regret >= -1e-6), f"Regret should be non-negative, got min {jnp.min(regret)}"

    def test_regret_zero_when_optimal(self):
        """When current action is optimal, regret should be ≈ 0"""
        num_agents = 2
        action_dim = 3
        batch_size = 4

        # Create collective CF rewards where action 1 is optimal for all
        # collective_cf: [num_agents, action_dim, batch]
        collective_cf = jnp.array([
            [[1.0, 1.0, 1.0, 1.0],   # agent 0, action 0
             [2.0, 2.0, 2.0, 2.0],   # agent 0, action 1 (best)
             [1.5, 1.5, 1.5, 1.5]],  # agent 0, action 2
            [[0.5, 0.5, 0.5, 0.5],   # agent 1, action 0
             [1.0, 1.0, 1.0, 1.0],   # agent 1, action 1 (best)
             [0.8, 0.8, 0.8, 0.8]],  # agent 1, action 2
        ])

        # Actual collective equals the max (agent took optimal action)
        # actual_collective: [batch, num_agents]
        actual_collective = jnp.array([
            [2.0, 1.0],  # batch 0
            [2.0, 1.0],  # batch 1
            [2.0, 1.0],  # batch 2
            [2.0, 1.0],  # batch 3
        ])

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert jnp.allclose(regret, 0.0, atol=1e-6), \
            f"Regret should be ~0 when action is optimal, got {regret}"

    def test_regret_positive_when_suboptimal(self):
        """When better action exists, regret should be > 0"""
        num_agents = 2
        action_dim = 3
        batch_size = 1

        # Create collective CF rewards
        collective_cf = jnp.array([
            [[1.0],   # agent 0, action 0 (current)
             [3.0],   # agent 0, action 1 (best - higher collective reward)
             [2.0]],  # agent 0, action 2
            [[0.5],   # agent 1, action 0 (current)
             [2.0],   # agent 1, action 1 (best)
             [1.0]],  # agent 1, action 2
        ])

        # Actual collective from current suboptimal actions
        actual_collective = jnp.array([[1.0, 0.5]])  # agent 0 got 1.0, agent 1 got 0.5

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        # Regret should be positive (3.0 - 1.0 = 2.0 for agent 0, 2.0 - 0.5 = 1.5 for agent 1)
        expected_regret = jnp.array([[2.0, 1.5]])

        assert jnp.allclose(regret, expected_regret, atol=1e-6), \
            f"Expected regret {expected_regret}, got {regret}"
        assert jnp.all(regret > 0), "Regret should be positive when better action exists"

    def test_different_batch_sizes(self):
        """Should handle different batch sizes"""
        num_agents = 3
        action_dim = 4

        for batch_size in [1, 8, 32, 64]:
            collective_cf = jnp.ones((num_agents, action_dim, batch_size))
            actual_collective = jnp.zeros((batch_size, num_agents))

            regret = compute_counterfactual_regret(collective_cf, actual_collective)

            assert regret.shape == (batch_size, num_agents), \
                f"Failed for batch_size={batch_size}"

    def test_different_num_agents(self):
        """Should handle different number of agents"""
        action_dim = 4
        batch_size = 8

        for num_agents in [2, 3, 4, 5, 7]:
            collective_cf = jnp.ones((num_agents, action_dim, batch_size))
            actual_collective = jnp.zeros((batch_size, num_agents))

            regret = compute_counterfactual_regret(collective_cf, actual_collective)

            assert regret.shape == (batch_size, num_agents), \
                f"Failed for num_agents={num_agents}"

    def test_different_action_dims(self):
        """Should handle different action dimensions"""
        num_agents = 3
        batch_size = 8

        for action_dim in [2, 3, 4, 5, 8]:
            collective_cf = jnp.ones((num_agents, action_dim, batch_size))
            actual_collective = jnp.zeros((batch_size, num_agents))

            regret = compute_counterfactual_regret(collective_cf, actual_collective)

            assert regret.shape == (batch_size, num_agents), \
                f"Failed for action_dim={action_dim}"

    def test_no_nan_or_inf(self):
        """Output should not contain NaN or Inf"""
        key = jax.random.PRNGKey(42)
        num_agents = 3
        action_dim = 4
        batch_size = 8

        collective_cf = jax.random.uniform(key, (num_agents, action_dim, batch_size))
        actual_collective = jax.random.uniform(key, (batch_size, num_agents))

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert jnp.all(jnp.isfinite(regret)), "Output contains NaN or Inf"

    def test_negative_actual_collective(self):
        """Should handle negative actual collective rewards"""
        num_agents = 2
        action_dim = 3
        batch_size = 2

        # Negative rewards possible
        collective_cf = jnp.array([
            [[-1.0, -2.0],   # agent 0, action 0
             [0.0, -1.0],    # agent 0, action 1 (best)
             [-0.5, -1.5]],  # agent 0, action 2
            [[-0.5, -1.0],   # agent 1, action 0
             [0.5, 0.0],     # agent 1, action 1 (best)
             [0.0, -0.5]],   # agent 1, action 2
        ])

        # Actual is worse than optimal
        actual_collective = jnp.array([[-1.0, -0.5], [-2.0, -1.0]])

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        # Regret should still be non-negative
        assert jnp.all(regret >= -1e-6), f"Regret should be non-negative even with negative rewards"

    def test_jit_compilation(self):
        """Function should be JIT compilable"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        collective_cf = jnp.ones((num_agents, action_dim, batch_size))
        actual_collective = jnp.zeros((batch_size, num_agents))

        # JIT compile
        jit_regret = jax.jit(compute_counterfactual_regret)
        regret = jit_regret(collective_cf, actual_collective)

        assert regret.shape == (batch_size, num_agents)
        assert jnp.all(jnp.isfinite(regret))


class TestComputeRegretWithBestAction:
    """Test regret computation with best action identification"""

    def test_output_shapes(self):
        """Should return both regret and best actions"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        collective_cf = jnp.ones((num_agents, action_dim, batch_size))
        actual_collective = jnp.zeros((batch_size, num_agents))
        actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

        regret, best_actions = compute_regret_with_best_action(
            collective_cf, actual_collective, actual_actions
        )

        assert regret.shape == (batch_size, num_agents)
        assert best_actions.shape == (batch_size, num_agents)

    def test_best_action_identification(self):
        """Should correctly identify the best prosocial action"""
        num_agents = 2
        action_dim = 3
        batch_size = 1

        collective_cf = jnp.array([
            [[1.0],   # agent 0, action 0
             [3.0],   # agent 0, action 1 (best)
             [2.0]],  # agent 0, action 2
            [[0.5],   # agent 1, action 0
             [1.0],   # agent 1, action 1
             [2.5]],  # agent 1, action 2 (best)
        ])

        actual_collective = jnp.array([[1.0, 0.5]])
        actual_actions = jnp.array([[0, 0]])

        regret, best_actions = compute_regret_with_best_action(
            collective_cf, actual_collective, actual_actions
        )

        # Best actions should be 1 for agent 0, 2 for agent 1
        expected_best = jnp.array([[1, 2]])
        assert jnp.array_equal(best_actions, expected_best), \
            f"Expected best actions {expected_best}, got {best_actions}"

    def test_jit_compilation(self):
        """Should be JIT compilable"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        collective_cf = jnp.ones((num_agents, action_dim, batch_size))
        actual_collective = jnp.zeros((batch_size, num_agents))
        actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

        jit_func = jax.jit(compute_regret_with_best_action)
        regret, best_actions = jit_func(collective_cf, actual_collective, actual_actions)

        assert jnp.all(jnp.isfinite(regret))


class TestComputeNormalizedRegret:
    """Test normalized regret computation"""

    def test_output_in_range(self):
        """Normalized regret should be in [0, 1]"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        key = jax.random.PRNGKey(42)
        collective_cf = jax.random.uniform(key, (num_agents, action_dim, batch_size), minval=0.1)
        actual_collective = jnp.zeros((batch_size, num_agents))

        normalized_regret = compute_normalized_regret(collective_cf, actual_collective)

        assert jnp.all(normalized_regret >= 0.0), "Normalized regret should be >= 0"
        assert jnp.all(normalized_regret <= 1.0), "Normalized regret should be <= 1"

    def test_zero_when_optimal(self):
        """Normalized regret should be 0 when action is optimal"""
        num_agents = 2
        action_dim = 3
        batch_size = 1

        collective_cf = jnp.array([
            [[1.0], [2.0], [1.5]],  # agent 0, action 1 best
            [[0.5], [1.0], [0.8]],  # agent 1, action 1 best
        ])

        actual_collective = jnp.array([[2.0, 1.0]])  # Optimal

        normalized_regret = compute_normalized_regret(collective_cf, actual_collective)

        assert jnp.allclose(normalized_regret, 0.0, atol=1e-6), \
            f"Normalized regret should be ~0 when optimal, got {normalized_regret}"

    def test_jit_compilation(self):
        """Should be JIT compilable"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        collective_cf = jnp.ones((num_agents, action_dim, batch_size))
        actual_collective = jnp.zeros((batch_size, num_agents))

        jit_func = jax.jit(compute_normalized_regret)
        normalized_regret = jit_func(collective_cf, actual_collective)

        assert jnp.all(jnp.isfinite(normalized_regret))


class TestGetRegretStatistics:
    """Test regret statistics computation"""

    def test_output_shapes(self):
        """Statistics should have correct shapes"""
        batch_size = 8
        num_agents = 3

        regret = jnp.zeros((batch_size, num_agents))
        mean_regret, max_regret, zero_ratio = get_regret_statistics(regret)

        assert mean_regret.shape == (num_agents,)
        assert max_regret.shape == (num_agents,)
        assert zero_ratio.shape == (num_agents,)

    def test_all_zero_regret(self):
        """When all regret is zero, statistics should reflect that"""
        batch_size = 8
        num_agents = 3

        regret = jnp.zeros((batch_size, num_agents))
        mean_regret, max_regret, zero_ratio = get_regret_statistics(regret)

        assert jnp.allclose(mean_regret, 0.0)
        assert jnp.allclose(max_regret, 0.0)
        assert jnp.allclose(zero_ratio, 1.0)  # All zero regret

    def test_mixed_regret(self):
        """Test with mixed regret values"""
        regret = jnp.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
        ])

        mean_regret, max_regret, zero_ratio = get_regret_statistics(regret)

        # Agent 0: [0, 1, 2, 0] -> mean=0.75, max=2, zero_ratio=0.5
        # Agent 1: [1, 0, 2, 1] -> mean=1.0, max=2, zero_ratio=0.25
        # Agent 2: [2, 1, 0, 1] -> mean=1.0, max=2, zero_ratio=0.25
        expected_mean = jnp.array([0.75, 1.0, 1.0])
        expected_max = jnp.array([2.0, 2.0, 2.0])
        expected_zero_ratio = jnp.array([0.5, 0.25, 0.25])

        assert jnp.allclose(mean_regret, expected_mean)
        assert jnp.allclose(max_regret, expected_max)
        assert jnp.allclose(zero_ratio, expected_zero_ratio)


class TestIntegrationWithCounterfactual:
    """Test integration with counterfactual module (M2, M3)"""

    def test_pipeline_from_collective_cf(self):
        """Test that regret computation works with output from M3"""
        from socialjax.algorithms.cf.counterfactual import (
            compute_collective_cf_reward,
            compute_actual_collective_reward,
        )

        # Simulate counterfactual rewards from M2
        # [num_agents, action_dim, batch, num_agents]
        num_agents = 3
        action_dim = 4
        batch_size = 2

        key = jax.random.PRNGKey(42)
        cf_rewards = jax.random.uniform(
            key, (num_agents, action_dim, batch_size, num_agents)
        )

        # Actual rewards
        actual_rewards = jax.random.uniform(key, (batch_size, num_agents))

        # M3: Compute collective CF rewards
        collective_cf = compute_collective_cf_reward(cf_rewards, exclude_self=True)
        assert collective_cf.shape == (num_agents, action_dim, batch_size)

        # M3: Compute actual collective rewards
        actual_collective = compute_actual_collective_reward(actual_rewards)
        assert actual_collective.shape == (batch_size, num_agents)

        # M4: Compute regret
        regret = compute_counterfactual_regret(collective_cf, actual_collective)
        assert regret.shape == (batch_size, num_agents)

        # Verify non-negative
        assert jnp.all(regret >= -1e-6), "Regret should be non-negative"

    def test_end_to_end_jit(self):
        """Test entire M2 -> M3 -> M4 pipeline is JIT compatible"""
        from socialjax.algorithms.cf.counterfactual import (
            compute_collective_cf_reward,
            compute_actual_collective_reward,
        )

        num_agents = 3
        action_dim = 4
        batch_size = 2

        def pipeline(cf_rewards, actual_rewards):
            collective_cf = compute_collective_cf_reward(cf_rewards, exclude_self=True)
            actual_collective = compute_actual_collective_reward(actual_rewards)
            regret = compute_counterfactual_regret(collective_cf, actual_collective)
            return regret

        jit_pipeline = jax.jit(pipeline)

        key = jax.random.PRNGKey(42)
        cf_rewards = jax.random.uniform(key, (num_agents, action_dim, batch_size, num_agents))
        actual_rewards = jax.random.uniform(key, (batch_size, num_agents))

        regret = jit_pipeline(cf_rewards, actual_rewards)

        assert regret.shape == (batch_size, num_agents)
        assert jnp.all(jnp.isfinite(regret))
        assert jnp.all(regret >= -1e-6)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_agent(self):
        """Test with single agent (edge case)"""
        num_agents = 1
        action_dim = 3
        batch_size = 2

        # With single agent, collective reward is 0 (no other agents)
        collective_cf = jnp.zeros((num_agents, action_dim, batch_size))
        actual_collective = jnp.zeros((batch_size, num_agents))

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert regret.shape == (batch_size, num_agents)
        assert jnp.allclose(regret, 0.0, atol=1e-6)

    def test_single_action(self):
        """Test with single action (no choice)"""
        num_agents = 2
        action_dim = 1
        batch_size = 2

        collective_cf = jnp.ones((num_agents, action_dim, batch_size))
        actual_collective = jnp.ones((batch_size, num_agents))

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        # With only one action, regret should be 0
        assert jnp.allclose(regret, 0.0, atol=1e-6)

    def test_very_large_values(self):
        """Test with very large reward values"""
        num_agents = 3
        action_dim = 4
        batch_size = 2

        collective_cf = jnp.full((num_agents, action_dim, batch_size), 1e6)
        actual_collective = jnp.full((batch_size, num_agents), 1e6)

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert jnp.all(jnp.isfinite(regret))
        assert jnp.allclose(regret, 0.0, atol=1e-6)

    def test_very_small_values(self):
        """Test with very small reward values"""
        num_agents = 3
        action_dim = 4
        batch_size = 2

        collective_cf = jnp.full((num_agents, action_dim, batch_size), 1e-6)
        actual_collective = jnp.zeros((batch_size, num_agents))

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert jnp.all(jnp.isfinite(regret))
        assert jnp.all(regret >= -1e-6)


class TestCFDebug003Verification:
    """CF-DEBUG-003: Verify regret non-negativity and numerical stability"""

    def test_regret_strictly_non_negative_with_epsilon(self):
        """Test that regret is always >= -epsilon after clamping"""
        num_agents = 4
        action_dim = 8
        batch_size = 16

        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)

        # Generate random collective CF rewards
        collective_cf = jax.random.uniform(subkey, (num_agents, action_dim, batch_size), minval=-10.0, maxval=10.0)

        # Generate actual collective rewards that could be larger than max (edge case)
        key, subkey = jax.random.split(key)
        actual_collective = jax.random.uniform(subkey, (batch_size, num_agents), minval=-10.0, maxval=10.0)

        epsilon = 1e-6
        regret = compute_counterfactual_regret(collective_cf, actual_collective, epsilon=epsilon)

        # All regret values should be >= -epsilon
        assert jnp.all(regret >= -epsilon), \
            f"Regret should be >= -epsilon ({epsilon}), got min {float(jnp.min(regret))}"

    def test_regret_exactly_zero_at_boundary(self):
        """Test that regret is exactly 0 when max_cf equals actual (within epsilon)"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        # Create collective CF rewards where all actions give same reward
        # This means max_cf == actual_collective for each
        collective_cf = jnp.full((num_agents, action_dim, batch_size), 5.0)

        # Actual collective equals the max
        actual_collective = jnp.full((batch_size, num_agents), 5.0)

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        # All regret should be exactly 0 (within tolerance)
        assert jnp.allclose(regret, 0.0, atol=1e-6), \
            f"Regret should be 0 when max_cf == actual, got {regret}"

    def test_max_operation_identifies_correct_best(self):
        """Test that max operation correctly identifies the best CF reward"""
        num_agents = 2
        action_dim = 5
        batch_size = 1

        # Create known CF rewards where we know the max
        collective_cf = jnp.array([
            [[1.0], [3.0], [5.0], [2.0], [4.0]],  # agent 0: max is action 2 (5.0)
            [[0.5], [1.5], [0.8], [2.0], [1.0]],  # agent 1: max is action 3 (2.0)
        ])

        # Actual collective is lower than max
        actual_collective = jnp.array([[1.0, 0.5]])  # agent 0 took action 0, agent 1 took action 0

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        # Expected regret: max - actual
        # Agent 0: 5.0 - 1.0 = 4.0
        # Agent 1: 2.0 - 0.5 = 1.5
        expected_regret = jnp.array([[4.0, 1.5]])

        assert jnp.allclose(regret, expected_regret, atol=1e-6), \
            f"Expected regret {expected_regret}, got {regret}"

    def test_max_operation_correctness_with_best_action_function(self):
        """Test that compute_regret_with_best_action identifies correct best action"""
        num_agents = 2
        action_dim = 5
        batch_size = 1

        # Create CF rewards where we know the best action
        collective_cf = jnp.array([
            [[1.0], [3.0], [5.0], [2.0], [4.0]],  # agent 0: best is action 2
            [[0.5], [1.5], [0.8], [2.0], [1.0]],  # agent 1: best is action 3
        ])

        actual_collective = jnp.array([[1.0, 0.5]])
        actual_actions = jnp.array([[0, 0]])

        regret, best_actions = compute_regret_with_best_action(
            collective_cf, actual_collective, actual_actions
        )

        # Best actions should be [2, 3]
        expected_best = jnp.array([[2, 3]])
        assert jnp.array_equal(best_actions, expected_best), \
            f"Expected best actions {expected_best}, got {best_actions}"

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values"""
        num_agents = 3
        action_dim = 4
        batch_size = 4

        # Test with very large values
        large_cf = jnp.full((num_agents, action_dim, batch_size), 1e10)
        large_actual = jnp.full((batch_size, num_agents), 1e10)

        regret_large = compute_counterfactual_regret(large_cf, large_actual)
        assert jnp.all(jnp.isfinite(regret_large)), "Regret should be finite with large values"
        assert jnp.all(regret_large >= -1e-6), "Regret should be non-negative with large values"

        # Test with very small values
        small_cf = jnp.full((num_agents, action_dim, batch_size), 1e-10)
        small_actual = jnp.zeros((batch_size, num_agents))

        regret_small = compute_counterfactual_regret(small_cf, small_actual)
        assert jnp.all(jnp.isfinite(regret_small)), "Regret should be finite with small values"
        assert jnp.all(regret_small >= -1e-6), "Regret should be non-negative with small values"

    def test_numerical_stability_mixed_signs(self):
        """Test numerical stability with mixed positive/negative values"""
        num_agents = 3
        action_dim = 4
        batch_size = 4

        # Create CF rewards with mixed signs
        collective_cf = jnp.array([
            [[1.0, -2.0, 3.0, -4.0],
             [2.0, -1.0, 4.0, -3.0],
             [3.0, 0.0, 2.0, -1.0],
             [0.0, 1.0, -1.0, 2.0]],
            [[-1.0, 2.0, -3.0, 4.0],
             [3.0, -2.0, 1.0, 0.0],
             [2.0, 1.0, -2.0, 3.0],
             [1.0, 0.0, -1.0, -2.0]],
            [[0.5, -0.5, 1.0, -1.0],
             [1.5, 0.5, -0.5, 0.0],
             [0.0, 1.0, 0.5, -0.5],
             [-0.5, 0.0, 1.5, 0.5]],
        ])

        # Actual collective with mixed signs
        actual_collective = jnp.array([
            [0.0, -1.0, 0.5],
            [-2.0, 1.0, -0.5],
            [1.0, 0.0, 0.0],
            [-1.0, -2.0, 0.5],
        ])

        regret = compute_counterfactual_regret(collective_cf, actual_collective)

        assert jnp.all(jnp.isfinite(regret)), "Regret should be finite with mixed signs"
        assert jnp.all(regret >= -1e-6), \
            f"Regret should be non-negative, got min {float(jnp.min(regret))}"

    def test_regret_non_negative_stress_test(self):
        """Stress test with many random configurations"""
        key = jax.random.PRNGKey(0)

        for trial in range(100):
            key, subkey1 = jax.random.split(key)
            key, subkey2 = jax.random.split(key)

            # Random configuration
            num_agents = jax.random.randint(subkey1, (), 2, 8)
            action_dim = jax.random.randint(subkey1, (), 2, 10)
            batch_size = jax.random.randint(subkey1, (), 1, 32)

            # Generate random CF rewards and actual collective
            collective_cf = jax.random.uniform(
                subkey1, (int(num_agents), int(action_dim), int(batch_size)),
                minval=-100.0, maxval=100.0
            )
            actual_collective = jax.random.uniform(
                subkey2, (int(batch_size), int(num_agents)),
                minval=-100.0, maxval=100.0
            )

            regret = compute_counterfactual_regret(collective_cf, actual_collective)

            # Verify non-negativity
            min_regret = float(jnp.min(regret))
            assert min_regret >= -1e-6, \
                f"Trial {trial}: Regret should be >= -1e-6, got min {min_regret}"

    def test_epsilon_parameter_effect(self):
        """Test that epsilon parameter correctly clamps small negative values"""
        num_agents = 2
        action_dim = 3
        batch_size = 1

        # Create a scenario where max - actual might be slightly negative due to floating point
        # (this is artificial but tests the epsilon parameter)
        collective_cf = jnp.array([
            [[1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0]],
        ])
        actual_collective = jnp.array([[1.0, 1.0]])

        # With small epsilon
        regret_small = compute_counterfactual_regret(collective_cf, actual_collective, epsilon=1e-8)
        assert jnp.all(regret_small >= -1e-8), "Regret should respect small epsilon"

        # With large epsilon
        regret_large = compute_counterfactual_regret(collective_cf, actual_collective, epsilon=1e-2)
        assert jnp.all(regret_large >= -1e-2), "Regret should respect large epsilon"

    def test_jit_preserves_non_negativity(self):
        """Test that JIT compilation doesn't affect non-negativity"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        key = jax.random.PRNGKey(42)
        collective_cf = jax.random.uniform(key, (num_agents, action_dim, batch_size))
        actual_collective = jax.random.uniform(key, (batch_size, num_agents))

        # Non-JIT version
        regret_regular = compute_counterfactual_regret(collective_cf, actual_collective)

        # JIT version
        jit_regret_fn = jax.jit(compute_counterfactual_regret)
        regret_jit = jit_regret_fn(collective_cf, actual_collective)

        # Both should be non-negative
        assert jnp.all(regret_regular >= -1e-6), "Regular regret should be non-negative"
        assert jnp.all(regret_jit >= -1e-6), "JIT regret should be non-negative"

        # Results should be identical
        assert jnp.allclose(regret_regular, regret_jit, atol=1e-7), \
            "JIT and regular results should be identical"

    def test_vmap_preserves_non_negativity(self):
        """Test that vmap doesn't affect non-negativity"""
        num_agents = 3
        action_dim = 4
        batch_size = 8

        key = jax.random.PRNGKey(42)
        collective_cf = jax.random.uniform(key, (num_agents, action_dim, batch_size))
        actual_collective = jax.random.uniform(key, (batch_size, num_agents))

        # Batch version
        regret_batch = compute_counterfactual_regret(collective_cf, actual_collective)

        # Per-batch-element version using vmap
        def single_element_regret(cf_single, actual_single):
            # cf_single: [num_agents, action_dim]
            # actual_single: [num_agents]
            cf_expanded = cf_single[:, :, jnp.newaxis]  # [num_agents, action_dim, 1]
            actual_expanded = actual_single[jnp.newaxis, :]  # [1, num_agents]
            return compute_counterfactual_regret(cf_expanded, actual_expanded)[0]

        vmapped_regret = jax.vmap(single_element_regret, in_axes=(2, 0))(collective_cf, actual_collective)

        # Both should be non-negative
        assert jnp.all(regret_batch >= -1e-6), "Batch regret should be non-negative"
        assert jnp.all(vmapped_regret >= -1e-6), "Vmapped regret should be non-negative"

        # Results should be similar
        assert jnp.allclose(regret_batch, vmapped_regret, atol=1e-6), \
            f"Batch and vmapped results should be similar. Max diff: {float(jnp.max(jnp.abs(regret_batch - vmapped_regret)))}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
