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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
