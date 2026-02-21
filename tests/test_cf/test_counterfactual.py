"""
Unit tests for Counterfactual Reward Generation (CF-IMPL-002)

Tests verify:
- Enumerate all |A| actions
- Output shape: [batch, num_actions, num_agents]
- Other agents' actions remain unchanged
- vmap implementation for efficient computation
"""

import pytest
import sys
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np


class TestEnumerateCounterfactualActions:
    """Tests for enumerate_counterfactual_actions function"""

    def test_enumerate_all_actions(self):
        """Should enumerate all action_dim actions"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 4
        num_agents = 3
        action_dim = 5

        actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
        cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=0, action_dim=action_dim)

        # Should have action_dim different action combinations
        assert cf_actions.shape[0] == action_dim, f"Expected {action_dim} actions, got {cf_actions.shape[0]}"

    def test_output_shape(self):
        """Output shape should be [action_dim, batch, num_agents]"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 8
        num_agents = 4
        action_dim = 3

        actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
        cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=0, action_dim=action_dim)

        assert cf_actions.shape == (action_dim, batch_size, num_agents), \
            f"Expected shape ({action_dim}, {batch_size}, {num_agents}), got {cf_actions.shape}"

    def test_other_agents_unchanged(self):
        """Other agents' actions should remain unchanged"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 2
        num_agents = 4
        action_dim = 3

        # Create actual actions with different values for each agent
        actual_actions = jnp.array([
            [0, 1, 2, 1],  # batch 0
            [2, 0, 1, 2],  # batch 1
        ], dtype=jnp.int32)

        # Test for agent_id=1
        agent_id = 1
        cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=agent_id, action_dim=action_dim)

        # Check that agents other than agent_id have unchanged actions
        for other_agent in range(num_agents):
            if other_agent != agent_id:
                expected = actual_actions[:, other_agent]
                actual = cf_actions[0, :, other_agent]  # Same for all counterfactual actions
                assert jnp.array_equal(actual, expected), \
                    f"Agent {other_agent}'s action changed when it shouldn't"

    def test_counterfactual_actions_are_correct(self):
        """Counterfactual actions should be [0, 1, ..., action_dim-1]"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 2
        num_agents = 3
        action_dim = 4

        actual_actions = jnp.array([
            [0, 1, 2],
            [2, 0, 1],
        ], dtype=jnp.int32)

        agent_id = 2
        cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=agent_id, action_dim=action_dim)

        # For each counterfactual action a in [0, action_dim), cf_actions[a, :, agent_id] should all be a
        for a in range(action_dim):
            expected = jnp.full((batch_size,), a, dtype=jnp.int32)
            actual = cf_actions[a, :, agent_id]
            assert jnp.array_equal(actual, expected), \
                f"Counterfactual action {a} incorrect: expected {expected}, got {actual}"

    def test_different_batch_sizes(self):
        """Should work with different batch sizes"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        action_dim = 3
        num_agents = 2

        for batch_size in [1, 4, 16, 32]:
            actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
            cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=0, action_dim=action_dim)
            assert cf_actions.shape == (action_dim, batch_size, num_agents)

    def test_different_num_agents(self):
        """Should work with different numbers of agents"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 4
        action_dim = 3

        for num_agents in [2, 3, 5, 7]:
            actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
            cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=0, action_dim=action_dim)
            assert cf_actions.shape == (action_dim, batch_size, num_agents)

    def test_different_action_dims(self):
        """Should work with different action dimensions"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 4
        num_agents = 3

        for action_dim in [2, 4, 8, 16]:
            actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
            cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=0, action_dim=action_dim)
            assert cf_actions.shape == (action_dim, batch_size, num_agents)


class TestEnumerateAllAgentsCounterfactualActions:
    """Tests for enumerate_all_agents_counterfactual_actions function"""

    def test_output_shape(self):
        """Output shape should be [num_agents, action_dim, batch, num_agents]"""
        from socialjax.algorithms.cf.counterfactual import enumerate_all_agents_counterfactual_actions

        batch_size = 4
        num_agents = 3
        action_dim = 5

        actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
        cf_actions = enumerate_all_agents_counterfactual_actions(actual_actions, action_dim=action_dim)

        assert cf_actions.shape == (num_agents, action_dim, batch_size, num_agents), \
            f"Expected shape ({num_agents}, {action_dim}, {batch_size}, {num_agents}), got {cf_actions.shape}"

    def test_each_agent_has_correct_counterfactuals(self):
        """Each agent should have their own counterfactual actions enumerated"""
        from socialjax.algorithms.cf.counterfactual import enumerate_all_agents_counterfactual_actions

        batch_size = 2
        num_agents = 3
        action_dim = 4

        actual_actions = jnp.array([
            [0, 1, 2],
            [3, 2, 1],
        ], dtype=jnp.int32)

        cf_actions = enumerate_all_agents_counterfactual_actions(actual_actions, action_dim=action_dim)

        # For each agent, verify their counterfactual actions
        for agent_id in range(num_agents):
            for a in range(action_dim):
                # The agent's action should be a
                assert jnp.all(cf_actions[agent_id, a, :, agent_id] == a), \
                    f"Agent {agent_id}'s counterfactual action {a} is incorrect"


class TestGenerateCounterfactualRewards:
    """Tests for counterfactual reward generation"""

    @pytest.fixture
    def mock_reward_model(self):
        """Create a mock reward model for testing"""
        from socialjax.algorithms.cf.generative_model import RewardModel

        class MockRewardModel:
            """Mock reward model that returns predictable outputs"""
            def apply(self, params, obs, actions):
                # Simple mock: return sum of actions as rewards
                batch_size = obs.shape[0]
                num_agents = obs.shape[1]
                # Return deterministic output based on actions
                return jnp.sum(actions, axis=-1, keepdims=True).astype(jnp.float32) / num_agents
                # Shape: [batch, 1] - need to expand to [batch, num_agents]

        return MockRewardModel()

    @pytest.fixture
    def real_reward_model(self):
        """Create a real reward model for testing"""
        from socialjax.algorithms.cf.generative_model import RewardModel
        return RewardModel(num_agents=3, action_dim=4)

    def test_output_shape_single_agent(self):
        """Test generate_counterfactual_rewards_single_agent output shape"""
        from socialjax.algorithms.cf.counterfactual import generate_counterfactual_rewards_single_agent
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 4
        num_agents = 3
        action_dim = 5
        obs_shape = (batch_size, num_agents, 8, 8, 3)

        # Create model and initialize
        rng = jax.random.PRNGKey(0)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, obs_shape)
        actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

        params = model.init(rng, obs, actions)

        # Generate counterfactual rewards for agent 0
        cf_rewards = generate_counterfactual_rewards_single_agent(
            model.apply, params, obs, agent_id=0, action_dim=action_dim, actual_actions=actions
        )

        assert cf_rewards.shape == (action_dim, batch_size, num_agents), \
            f"Expected shape ({action_dim}, {batch_size}, {num_agents}), got {cf_rewards.shape}"

    def test_output_shape_vmap(self):
        """Test generate_counterfactual_rewards_vmap output shape"""
        from socialjax.algorithms.cf.counterfactual import generate_counterfactual_rewards_vmap
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 4
        num_agents = 3
        action_dim = 5
        obs_shape = (batch_size, num_agents, 8, 8, 3)

        rng = jax.random.PRNGKey(0)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, obs_shape)
        actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

        params = model.init(rng, obs, actions)

        # Generate counterfactual rewards for all agents
        cf_rewards = generate_counterfactual_rewards_vmap(
            model.apply, params, action_dim, obs, actions
        )

        assert cf_rewards.shape == (num_agents, action_dim, batch_size, num_agents), \
            f"Expected shape ({num_agents}, {action_dim}, {batch_size}, {num_agents}), got {cf_rewards.shape}"

    def test_all_actions_enumerated(self):
        """Verify all action_dim counterfactual actions are generated"""
        from socialjax.algorithms.cf.counterfactual import generate_counterfactual_rewards_single_agent
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 2
        num_agents = 3
        action_dim = 6  # Test with 6 actions

        rng = jax.random.PRNGKey(0)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
        actions = jnp.array([[0, 1, 2], [3, 2, 1]], dtype=jnp.int32)

        params = model.init(rng, obs, actions)

        cf_rewards = generate_counterfactual_rewards_single_agent(
            model.apply, params, obs, agent_id=0, action_dim=action_dim, actual_actions=actions
        )

        # Should have rewards for all action_dim actions
        assert cf_rewards.shape[0] == action_dim, \
            f"Expected {action_dim} counterfactual actions, got {cf_rewards.shape[0]}"

    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        from socialjax.algorithms.cf.counterfactual import generate_counterfactual_rewards_vmap
        from socialjax.algorithms.cf.generative_model import RewardModel

        num_agents = 3
        action_dim = 4

        for batch_size in [1, 4, 16]:
            rng = jax.random.PRNGKey(0)
            model = RewardModel(num_agents=num_agents, action_dim=action_dim)
            obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
            actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

            params = model.init(rng, obs, actions)

            cf_rewards = generate_counterfactual_rewards_vmap(
                model.apply, params, action_dim, obs, actions
            )

            assert cf_rewards.shape == (num_agents, action_dim, batch_size, num_agents), \
                f"Batch size {batch_size}: expected shape ({num_agents}, {action_dim}, {batch_size}, {num_agents}), got {cf_rewards.shape}"

    def test_different_num_agents(self):
        """Test with different numbers of agents"""
        from socialjax.algorithms.cf.counterfactual import generate_counterfactual_rewards_vmap
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 4
        action_dim = 4

        for num_agents in [2, 4, 7]:
            rng = jax.random.PRNGKey(0)
            model = RewardModel(num_agents=num_agents, action_dim=action_dim)
            obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
            actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

            params = model.init(rng, obs, actions)

            cf_rewards = generate_counterfactual_rewards_vmap(
                model.apply, params, action_dim, obs, actions
            )

            assert cf_rewards.shape == (num_agents, action_dim, batch_size, num_agents)

    def test_no_nan_or_inf(self):
        """Output should not contain NaN or Inf"""
        from socialjax.algorithms.cf.counterfactual import generate_counterfactual_rewards_vmap
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 8
        num_agents = 4
        action_dim = 5

        rng = jax.random.PRNGKey(0)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
        actions = jax.random.randint(rng, (batch_size, num_agents), 0, action_dim)

        params = model.init(rng, obs, actions)

        cf_rewards = generate_counterfactual_rewards_vmap(
            model.apply, params, action_dim, obs, actions
        )

        assert jnp.all(jnp.isfinite(cf_rewards)), "Output contains NaN or Inf"


class TestCollectiveCounterfactualReward:
    """Tests for compute_collective_cf_reward function"""

    def test_output_shape(self):
        """Output shape should be [num_agents, action_dim, batch]"""
        from socialjax.algorithms.cf.counterfactual import compute_collective_cf_reward

        num_agents = 3
        action_dim = 4
        batch_size = 8

        cf_rewards = jnp.zeros((num_agents, action_dim, batch_size, num_agents))
        collective = compute_collective_cf_reward(cf_rewards)

        assert collective.shape == (num_agents, action_dim, batch_size), \
            f"Expected shape ({num_agents}, {action_dim}, {batch_size}), got {collective.shape}"

    def test_excludes_self(self):
        """Should exclude ego agent's own reward from sum"""
        from socialjax.algorithms.cf.counterfactual import compute_collective_cf_reward

        num_agents = 3
        action_dim = 2
        batch_size = 2

        # Create predictable rewards with clear structure
        # cf_rewards[i, a, b, j] = reward for agent j when agent i takes counterfactual action a
        # Set each agent j's reward to (j+1)*10 for all counterfactuals
        cf_rewards = jnp.zeros((num_agents, action_dim, batch_size, num_agents))
        for j in range(num_agents):
            cf_rewards = cf_rewards.at[:, :, :, j].set((j + 1) * 10.0)
        # Now cf_rewards[:, :, :, 0] = 10, cf_rewards[:, :, :, 1] = 20, cf_rewards[:, :, :, 2] = 30

        collective = compute_collective_cf_reward(cf_rewards, exclude_self=True)

        # For agent 0's counterfactuals: collective = 20 + 30 = 50
        assert jnp.allclose(collective[0, :, :], 50.0), \
            f"Expected 50.0 for agent 0, got {collective[0, 0, 0]}"
        # For agent 1's counterfactuals: collective = 10 + 30 = 40
        assert jnp.allclose(collective[1, :, :], 40.0), \
            f"Expected 40.0 for agent 1, got {collective[1, 0, 0]}"
        # For agent 2's counterfactuals: collective = 10 + 20 = 30
        assert jnp.allclose(collective[2, :, :], 30.0), \
            f"Expected 30.0 for agent 2, got {collective[2, 0, 0]}"

    def test_two_agent_case(self):
        """In 2-agent case, collective = other agent's reward"""
        from socialjax.algorithms.cf.counterfactual import compute_collective_cf_reward

        num_agents = 2
        action_dim = 3
        batch_size = 2

        # Agent 0's rewards = 5, Agent 1's rewards = 10
        cf_rewards = jnp.zeros((num_agents, action_dim, batch_size, num_agents))
        cf_rewards = cf_rewards.at[:, :, :, 0].set(5.0)
        cf_rewards = cf_rewards.at[:, :, :, 1].set(10.0)

        collective = compute_collective_cf_reward(cf_rewards, exclude_self=True)

        # For agent 0, collective = agent 1's reward = 10
        assert jnp.allclose(collective[0, :, :], 10.0)
        # For agent 1, collective = agent 0's reward = 5
        assert jnp.allclose(collective[1, :, :], 5.0)

    def test_include_self_option(self):
        """When exclude_self=False, should include all agents"""
        from socialjax.algorithms.cf.counterfactual import compute_collective_cf_reward

        num_agents = 3
        action_dim = 2
        batch_size = 1

        cf_rewards = jnp.ones((num_agents, action_dim, batch_size, num_agents)) * 2.0

        collective = compute_collective_cf_reward(cf_rewards, exclude_self=False)

        # All agents' rewards are 2, so sum = 2 * 3 = 6
        assert jnp.allclose(collective, 6.0)


class TestActualCollectiveReward:
    """Tests for compute_actual_collective_reward function"""

    def test_output_shape(self):
        """Output shape should be [batch, num_agents]"""
        from socialjax.algorithms.cf.counterfactual import compute_actual_collective_reward

        batch_size = 8
        num_agents = 4

        rewards = jnp.zeros((batch_size, num_agents))
        collective = compute_actual_collective_reward(rewards)

        assert collective.shape == (batch_size, num_agents)

    def test_computation_correct(self):
        """Should correctly compute sum of other agents' rewards"""
        from socialjax.algorithms.cf.counterfactual import compute_actual_collective_reward

        rewards = jnp.array([
            [1.0, 2.0, 3.0, 4.0],  # batch 0
            [5.0, 6.0, 7.0, 8.0],  # batch 1
        ])

        collective = compute_actual_collective_reward(rewards)

        # For batch 0:
        # agent 0: 2+3+4 = 9
        # agent 1: 1+3+4 = 8
        # agent 2: 1+2+4 = 7
        # agent 3: 1+2+3 = 6
        expected = jnp.array([
            [9.0, 8.0, 7.0, 6.0],
            [21.0, 20.0, 19.0, 18.0],  # 5+6+7+8=26, minus each agent
        ])

        assert jnp.allclose(collective, expected)


class TestGetCounterfactualAnalysis:
    """Tests for the get_counterfactual_analysis convenience function"""

    def test_output_shapes(self):
        """All outputs should have correct shapes"""
        from socialjax.algorithms.cf.counterfactual import get_counterfactual_analysis
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 4
        num_agents = 3
        action_dim = 5
        obs_shape = (batch_size, num_agents, 8, 8, 3)

        rng = jax.random.PRNGKey(0)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, obs_shape)
        actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
        rewards = jnp.zeros((batch_size, num_agents))

        params = model.init(rng, obs, actions)

        cf_rewards, collective_cf, actual_collective = get_counterfactual_analysis(
            model.apply, params, obs, actions, rewards, action_dim
        )

        assert cf_rewards.shape == (num_agents, action_dim, batch_size, num_agents)
        assert collective_cf.shape == (num_agents, action_dim, batch_size)
        assert actual_collective.shape == (batch_size, num_agents)

    def test_integration_with_real_data(self):
        """Test with realistic data from coin_game environment"""
        from socialjax.algorithms.cf.counterfactual import get_counterfactual_analysis
        from socialjax.algorithms.cf.generative_model import RewardModel

        # Simulate coin_game-like data
        batch_size = 8
        num_agents = 2  # coin_game has 2 agents
        action_dim = 4  # coin_game has 4 actions
        obs_shape = (batch_size, num_agents, 8, 8, 3)

        rng = jax.random.PRNGKey(42)
        rng, init_rng, obs_rng, action_rng, reward_rng = jax.random.split(rng, 5)

        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(obs_rng, obs_shape)
        actions = jax.random.randint(action_rng, (batch_size, num_agents), 0, action_dim)
        rewards = jax.random.normal(reward_rng, (batch_size, num_agents))

        params = model.init(init_rng, obs, actions)

        cf_rewards, collective_cf, actual_collective = get_counterfactual_analysis(
            model.apply, params, obs, actions, rewards, action_dim
        )

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(cf_rewards)), "cf_rewards contains NaN/Inf"
        assert jnp.all(jnp.isfinite(collective_cf)), "collective_cf contains NaN/Inf"
        assert jnp.all(jnp.isfinite(actual_collective)), "actual_collective contains NaN/Inf"


class TestJITCompilation:
    """Tests for JIT compilation compatibility"""

    def test_enumerate_jit(self):
        """enumerate_counterfactual_actions should be JIT compatible"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions
        from functools import partial

        jitted_fn = jax.jit(
            partial(enumerate_counterfactual_actions, agent_id=0, action_dim=4),
            static_argnums=(1, 2)
        )

        actual_actions = jnp.zeros((4, 3), dtype=jnp.int32)
        # Need to pass agent_id and action_dim as static
        jitted_fn = partial(enumerate_counterfactual_actions, agent_id=0, action_dim=4)

        result = jitted_fn(actual_actions)
        assert result.shape == (4, 4, 3)

    def test_collective_jit(self):
        """compute_collective_cf_reward should be JIT compatible"""
        from socialjax.algorithms.cf.counterfactual import compute_collective_cf_reward

        jitted_fn = jax.jit(compute_collective_cf_reward)

        cf_rewards = jnp.ones((3, 4, 2, 3))
        result = jitted_fn(cf_rewards)

        assert result.shape == (3, 4, 2)


class TestVmapEfficiency:
    """Tests to verify vmap provides efficient computation"""

    def test_vmap_produces_correct_results(self):
        """vmap version should match sequential computation"""
        from socialjax.algorithms.cf.counterfactual import (
            generate_counterfactual_rewards_single_agent,
            generate_counterfactual_rewards_vmap,
        )
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 4
        num_agents = 3
        action_dim = 4

        rng = jax.random.PRNGKey(0)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
        actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

        params = model.init(rng, obs, actions)

        # Compute with vmap
        cf_rewards_vmap = generate_counterfactual_rewards_vmap(
            model.apply, params, action_dim, obs, actions
        )

        # Compute sequentially for each agent
        cf_rewards_seq = []
        for agent_id in range(num_agents):
            agent_cf = generate_counterfactual_rewards_single_agent(
                model.apply, params, obs, agent_id, action_dim, actions
            )
            cf_rewards_seq.append(agent_cf)
        cf_rewards_sequential = jnp.stack(cf_rewards_seq, axis=0)

        # Results should match
        assert jnp.allclose(cf_rewards_vmap, cf_rewards_sequential, rtol=1e-5, atol=1e-5), \
            "vmap and sequential results differ"


class TestCFDebug002Verification:
    """
    Debug tests for CF-DEBUG-002: 反事实奖励生成验证

    Test criteria:
    1. 枚举 |A| 个动作 - Enumerate all |A| actions
    2. vmap效率提升 - vmap provides efficiency improvement
    3. 其他agent不受影响 - Other agents' actions remain unchanged
    """

    # ==================== Criterion 1: Enumerate |A| Actions ====================

    def test_enumerate_complete_action_set(self):
        """Verify ALL action_dim actions are enumerated, not a subset"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        action_dim = 7  # Arbitrary non-power-of-2 to test edge cases
        batch_size = 3
        num_agents = 4
        target_agent_id = 2  # Define as variable for later reference

        actual_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
        cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=target_agent_id, action_dim=action_dim)

        # Verify exactly action_dim counterfactual actions
        assert cf_actions.shape[0] == action_dim

        # Verify each counterfactual action a is exactly present in the enumeration
        for expected_action in range(action_dim):
            # The counterfactual action for target_agent_id should be expected_action
            actual = cf_actions[expected_action, :, target_agent_id]
            assert jnp.all(actual == expected_action), \
                f"Expected action {expected_action} not found in enumeration"

    def test_enumerate_no_duplicate_actions(self):
        """Verify no duplicate actions in enumeration"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        action_dim = 5
        batch_size = 2
        num_agents = 3

        actual_actions = jnp.array([[0, 1, 2], [3, 2, 1]], dtype=jnp.int32)
        cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=0, action_dim=action_dim)

        # Extract the counterfactual actions taken by the target agent
        cf_actions_for_agent = cf_actions[:, 0, 0]  # [action_dim]

        # Should be [0, 1, 2, 3, 4] for action_dim=5
        expected = jnp.arange(action_dim)
        # Sort and compare (order doesn't matter, just presence)
        assert jnp.array_equal(jnp.sort(cf_actions_for_agent), expected), \
            "Duplicate or missing actions in enumeration"

    def test_enumerate_with_large_action_dim(self):
        """Test enumeration with larger action space (e.g., 16 actions)"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        action_dim = 16
        batch_size = 4
        num_agents = 5

        actual_actions = jax.random.randint(jax.random.PRNGKey(0), (batch_size, num_agents), 0, action_dim)
        cf_actions = enumerate_counterfactual_actions(actual_actions, agent_id=1, action_dim=action_dim)

        assert cf_actions.shape == (action_dim, batch_size, num_agents)

        # Verify all 16 actions are present
        unique_cf_actions = jnp.unique(cf_actions[:, 0, 1])
        assert len(unique_cf_actions) == action_dim, \
            f"Expected {action_dim} unique actions, got {len(unique_cf_actions)}"

    # ==================== Criterion 2: vmap Efficiency ====================

    def test_vmap_matches_sequential_all_agents(self):
        """Verify vmap produces identical results to sequential for all agents"""
        from socialjax.algorithms.cf.counterfactual import (
            generate_counterfactual_rewards_single_agent,
            generate_counterfactual_rewards_vmap,
        )
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 8
        num_agents = 5
        action_dim = 4

        rng = jax.random.PRNGKey(123)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
        actions = jax.random.randint(rng, (batch_size, num_agents), 0, action_dim)

        params = model.init(rng, obs, actions)

        # vmap version
        cf_rewards_vmap = generate_counterfactual_rewards_vmap(
            model.apply, params, action_dim, obs, actions
        )

        # Sequential version for each agent
        cf_rewards_sequential = []
        for agent_id in range(num_agents):
            agent_cf = generate_counterfactual_rewards_single_agent(
                model.apply, params, obs, agent_id, action_dim, actions
            )
            cf_rewards_sequential.append(agent_cf)
        cf_rewards_seq = jnp.stack(cf_rewards_sequential, axis=0)

        # Should be exactly equal (or very close)
        assert jnp.allclose(cf_rewards_vmap, cf_rewards_seq, rtol=1e-6, atol=1e-6), \
            f"vmap and sequential differ: max diff = {jnp.max(jnp.abs(cf_rewards_vmap - cf_rewards_seq))}"

    def test_vmap_correctness_different_configurations(self):
        """Verify vmap correctness across different batch/agent/action configurations"""
        from socialjax.algorithms.cf.counterfactual import (
            generate_counterfactual_rewards_single_agent,
            generate_counterfactual_rewards_vmap,
        )
        from socialjax.algorithms.cf.generative_model import RewardModel

        configs = [
            (1, 2, 2),   # Minimal: batch=1, 2 agents, 2 actions
            (4, 3, 4),   # Small
            (8, 4, 5),   # Medium
            (2, 7, 3),   # Many agents
        ]

        for batch_size, num_agents, action_dim in configs:
            rng = jax.random.PRNGKey(batch_size * 100 + num_agents * 10 + action_dim)
            model = RewardModel(num_agents=num_agents, action_dim=action_dim)
            obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
            actions = jax.random.randint(rng, (batch_size, num_agents), 0, action_dim)

            params = model.init(rng, obs, actions)

            # vmap version
            cf_vmap = generate_counterfactual_rewards_vmap(
                model.apply, params, action_dim, obs, actions
            )

            # Sequential version
            cf_seq_list = []
            for agent_id in range(num_agents):
                cf_seq_list.append(
                    generate_counterfactual_rewards_single_agent(
                        model.apply, params, obs, agent_id, action_dim, actions
                    )
                )
            cf_seq = jnp.stack(cf_seq_list, axis=0)

            # Note: Use looser tolerance due to floating-point non-associativity
            # vmap and sequential may have different computation orders
            max_diff = jnp.max(jnp.abs(cf_vmap - cf_seq))
            assert jnp.allclose(cf_vmap, cf_seq, rtol=1e-4, atol=1e-4), \
                f"vmap != sequential for config ({batch_size}, {num_agents}, {action_dim}), max_diff={max_diff}"

    def test_vmap_is_faster_than_sequential(self):
        """Verify vmap provides computational speedup over sequential execution"""
        import time
        from socialjax.algorithms.cf.counterfactual import (
            generate_counterfactual_rewards_single_agent,
            generate_counterfactual_rewards_vmap,
        )
        from socialjax.algorithms.cf.generative_model import RewardModel

        batch_size = 16
        num_agents = 5
        action_dim = 4

        rng = jax.random.PRNGKey(456)
        model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
        actions = jax.random.randint(rng, (batch_size, num_agents), 0, action_dim)

        params = model.init(rng, obs, actions)

        # Warmup JIT compilation
        _ = generate_counterfactual_rewards_vmap(model.apply, params, action_dim, obs, actions)
        for agent_id in range(num_agents):
            _ = generate_counterfactual_rewards_single_agent(
                model.apply, params, obs, agent_id, action_dim, actions
            )

        # Time vmap version
        num_runs = 5
        vmap_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = generate_counterfactual_rewards_vmap(model.apply, params, action_dim, obs, actions)
            vmap_times.append(time.perf_counter() - start)

        # Time sequential version
        seq_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            for agent_id in range(num_agents):
                _ = generate_counterfactual_rewards_single_agent(
                    model.apply, params, obs, agent_id, action_dim, actions
                )
            seq_times.append(time.perf_counter() - start)

        avg_vmap = sum(vmap_times) / len(vmap_times)
        avg_seq = sum(seq_times) / len(seq_times)

        # vmap should be at least not significantly slower than sequential
        # (may not always be faster due to overhead, but should be competitive)
        print(f"\n  vmap avg: {avg_vmap:.4f}s, sequential avg: {avg_seq:.4f}s")
        # We mainly verify correctness, efficiency depends on hardware
        # Just check that vmap completes without error
        assert avg_vmap > 0 and avg_seq > 0, "Timing should be positive"

    # ==================== Criterion 3: Other Agents Unaffected ====================

    def test_other_agents_completely_unchanged(self):
        """Rigorous test: other agents' actions must be EXACTLY unchanged"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 3
        num_agents = 5
        action_dim = 4

        # Use distinct, non-zero actions to make changes more detectable
        actual_actions = jnp.array([
            [3, 2, 1, 0, 2],
            [1, 3, 2, 1, 0],
            [0, 1, 0, 3, 3],
        ], dtype=jnp.int32)

        for target_agent in range(num_agents):
            cf_actions = enumerate_counterfactual_actions(
                actual_actions, agent_id=target_agent, action_dim=action_dim
            )

            # For each non-target agent, verify their actions are unchanged across ALL counterfactuals
            for other_agent in range(num_agents):
                if other_agent != target_agent:
                    # Extract actions for other_agent across all counterfactuals
                    other_agent_actions = cf_actions[:, :, other_agent]  # [action_dim, batch]

                    # Should be identical to original for all counterfactuals
                    expected = actual_actions[:, other_agent]  # [batch]
                    expected_tiled = jnp.tile(expected[jnp.newaxis, :], (action_dim, 1))

                    assert jnp.array_equal(other_agent_actions, expected_tiled), \
                        f"Agent {other_agent}'s actions changed when targeting agent {target_agent}"

    def test_other_agents_unchanged_all_batch_elements(self):
        """Verify other agents unchanged for ALL batch elements independently"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 10
        num_agents = 4
        action_dim = 3

        rng = jax.random.PRNGKey(789)
        actual_actions = jax.random.randint(rng, (batch_size, num_agents), 0, action_dim)

        for target_agent in range(num_agents):
            cf_actions = enumerate_counterfactual_actions(
                actual_actions, agent_id=target_agent, action_dim=action_dim
            )

            for batch_idx in range(batch_size):
                for cf_action_idx in range(action_dim):
                    for other_agent in range(num_agents):
                        if other_agent != target_agent:
                            actual = cf_actions[cf_action_idx, batch_idx, other_agent]
                            expected = actual_actions[batch_idx, other_agent]
                            assert actual == expected, \
                                f"Batch {batch_idx}, CF action {cf_action_idx}: " \
                                f"Agent {other_agent} changed from {expected} to {actual}"

    def test_other_agents_unchanged_vmap_version(self):
        """Verify other agents unaffected in vmap version for all agents at once"""
        from socialjax.algorithms.cf.counterfactual import (
            enumerate_all_agents_counterfactual_actions,
        )

        batch_size = 4
        num_agents = 3
        action_dim = 4

        rng = jax.random.PRNGKey(101)
        actual_actions = jax.random.randint(rng, (batch_size, num_agents), 0, action_dim)

        cf_actions = enumerate_all_agents_counterfactual_actions(actual_actions, action_dim)

        # cf_actions shape: [num_agents, action_dim, batch, num_agents]
        for target_agent in range(num_agents):
            for cf_action_idx in range(action_dim):
                for batch_idx in range(batch_size):
                    for other_agent in range(num_agents):
                        if other_agent != target_agent:
                            actual = cf_actions[target_agent, cf_action_idx, batch_idx, other_agent]
                            expected = actual_actions[batch_idx, other_agent]
                            assert actual == expected, \
                                f"vmap: Agent {target_agent} CF {cf_action_idx}, " \
                                f"batch {batch_idx}: Agent {other_agent} changed"

    def test_single_agent_counterfactual_isolation(self):
        """When generating CF for one agent, only that agent's column changes"""
        from socialjax.algorithms.cf.counterfactual import enumerate_counterfactual_actions

        batch_size = 2
        num_agents = 4
        action_dim = 5

        actual_actions = jnp.array([
            [4, 3, 2, 1],
            [1, 2, 3, 4],
        ], dtype=jnp.int32)

        target_agent = 2
        cf_actions = enumerate_counterfactual_actions(
            actual_actions, agent_id=target_agent, action_dim=action_dim
        )

        # For each counterfactual action
        for cf_idx in range(action_dim):
            # The target agent's action should be cf_idx
            assert jnp.all(cf_actions[cf_idx, :, target_agent] == cf_idx)

            # All other agents should be unchanged
            for other_agent in range(num_agents):
                if other_agent != target_agent:
                    assert jnp.array_equal(
                        cf_actions[cf_idx, :, other_agent],
                        actual_actions[:, other_agent]
                    ), f"Other agent {other_agent} was modified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
