"""
Tests for CF Environment Adapters (CF-IMPL-010)

Tests the environment adapters for Coin Game, Clean Up, and Harvest environments.
"""

import pytest
import sys
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np

from socialjax.algorithms.cf.env_adapters import (
    CFEnvSpec,
    BaseCFAdapter,
    CoinGameCFAdapter,
    CleanupCFAdapter,
    HarvestCommonCFAdapter,
    create_cf_adapter,
    get_adapter_for_env,
    list_available_adapters,
    get_env_spec,
    verify_adapter_compatibility,
    ADAPTER_REGISTRY,
)


class TestCFEnvSpec:
    """Test CFEnvSpec dataclass."""

    def test_create_spec(self):
        """Test creating an environment spec."""
        spec = CFEnvSpec(
            env_name='coin_game',
            num_agents=3,
            action_dim=7,
            obs_shape=(11, 11, 14),
            num_inner_steps=1000,
            num_outer_steps=1,
            default_alpha=2.0,
        )
        assert spec.env_name == 'coin_game'
        assert spec.num_agents == 3
        assert spec.action_dim == 7
        assert spec.obs_shape == (11, 11, 14)
        assert spec.default_alpha == 2.0


class TestCoinGameCFAdapter:
    """Test CoinGameCFAdapter."""

    def test_create_adapter(self):
        """Test creating a Coin Game adapter."""
        adapter = CoinGameCFAdapter(num_agents=3)
        assert adapter.env_name == 'coin_game'
        assert adapter.num_agents == 3
        assert adapter.action_dim == 7
        assert adapter.obs_shape == (11, 11, 14)

    def test_default_alpha(self):
        """Test default alpha is N-1."""
        adapter = CoinGameCFAdapter(num_agents=3)
        assert adapter.default_alpha == 2.0  # 3 - 1 = 2

    def test_get_spec(self):
        """Test getting environment spec."""
        adapter = CoinGameCFAdapter(num_agents=3)
        spec = adapter.get_spec()
        assert spec.env_name == 'coin_game'
        assert spec.num_agents == 3
        assert spec.action_dim == 7
        assert spec.obs_shape == (11, 11, 14)

    def test_reset(self):
        """Test resetting the environment."""
        adapter = CoinGameCFAdapter(num_agents=3)
        rng = jax.random.PRNGKey(0)
        obs, state = adapter.reset(rng)

        # Check observation shape
        assert obs.shape == (3, 11, 11, 14)

        # Check observations are finite
        assert jnp.all(jnp.isfinite(obs))

    def test_step(self):
        """Test stepping the environment."""
        adapter = CoinGameCFAdapter(num_agents=3)
        rng = jax.random.PRNGKey(0)

        # Reset
        obs, state = adapter.reset(rng)

        # Sample random actions
        rng, action_rng = jax.random.split(rng)
        actions = jax.random.randint(action_rng, (3,), 0, 7)

        # Step
        rng, step_rng = jax.random.split(rng)
        obs_next, state_next, rewards, dones, infos = adapter.step(
            step_rng, state, actions
        )

        # Check shapes
        assert obs_next.shape == (3, 11, 11, 14)
        assert rewards.shape == (3,)

        # Check values are finite
        assert jnp.all(jnp.isfinite(obs_next))
        assert jnp.all(jnp.isfinite(rewards))

    def test_preprocess_obs(self):
        """Test observation preprocessing."""
        adapter = CoinGameCFAdapter(num_agents=3)

        # Create dummy observation
        obs = jnp.zeros((3, 11, 11, 14), dtype=jnp.int8)
        processed = adapter.preprocess_obs(obs)

        assert processed.shape == (3, 11, 11, 14)
        assert processed.dtype == jnp.float32

    def test_convert_actions(self):
        """Test action conversion."""
        adapter = CoinGameCFAdapter(num_agents=3)

        actions = jnp.array([0, 1, 2])
        env_actions = adapter.convert_actions(actions)

        assert len(env_actions) == 3
        assert env_actions[0] == 0
        assert env_actions[1] == 1
        assert env_actions[2] == 2

    def test_process_rewards(self):
        """Test reward processing."""
        adapter = CoinGameCFAdapter(num_agents=3)

        # Test with array rewards
        rewards = jnp.array([1.0, 0.0, 1.0])
        processed = adapter.process_rewards(rewards)
        assert processed.shape == (3,)
        assert jnp.allclose(processed, rewards)

    def test_different_num_agents(self):
        """Test adapter with different number of agents."""
        for num_agents in [2, 3, 4]:
            adapter = CoinGameCFAdapter(num_agents=num_agents)
            assert adapter.num_agents == num_agents
            assert adapter.default_alpha == float(num_agents - 1)

    def test_properties(self):
        """Test adapter properties."""
        adapter = CoinGameCFAdapter(num_agents=3)
        assert adapter.observation_space == (11, 11, 14)
        assert adapter.action_space == 7


class TestCleanupCFAdapter:
    """Test CleanupCFAdapter."""

    def test_create_adapter(self):
        """Test creating a Clean Up adapter."""
        adapter = CleanupCFAdapter(num_agents=5)
        assert adapter.env_name == 'clean_up'
        assert adapter.num_agents == 5
        assert adapter.action_dim == 8
        assert adapter.obs_shape == (11, 11, 19)

    def test_reset(self):
        """Test resetting the environment."""
        adapter = CleanupCFAdapter(num_agents=5)
        rng = jax.random.PRNGKey(0)
        obs, state = adapter.reset(rng)

        # Check observation shape
        assert obs.shape == (5, 11, 11, 19)

        # Check observations are finite
        assert jnp.all(jnp.isfinite(obs))

    def test_step(self):
        """Test stepping the environment."""
        adapter = CleanupCFAdapter(num_agents=5)
        rng = jax.random.PRNGKey(0)

        # Reset
        obs, state = adapter.reset(rng)

        # Sample random actions
        rng, action_rng = jax.random.split(rng)
        actions = jax.random.randint(action_rng, (5,), 0, 8)

        # Step
        rng, step_rng = jax.random.split(rng)
        obs_next, state_next, rewards, dones, infos = adapter.step(
            step_rng, state, actions
        )

        # Check shapes
        assert obs_next.shape == (5, 11, 11, 19)
        assert rewards.shape == (5,)

        # Check values are finite
        assert jnp.all(jnp.isfinite(obs_next))
        assert jnp.all(jnp.isfinite(rewards))

    def test_default_alpha(self):
        """Test default alpha is N-1."""
        adapter = CleanupCFAdapter(num_agents=5)
        assert adapter.default_alpha == 4.0  # 5 - 1 = 4


class TestHarvestCommonCFAdapter:
    """Test HarvestCommonCFAdapter."""

    def test_create_adapter(self):
        """Test creating a Harvest adapter."""
        adapter = HarvestCommonCFAdapter(num_agents=5)
        assert adapter.env_name == 'harvest_common_open'
        assert adapter.num_agents == 5
        assert adapter.action_dim == 8
        assert adapter.obs_shape == (11, 11, 15)

    def test_reset(self):
        """Test resetting the environment."""
        adapter = HarvestCommonCFAdapter(num_agents=5)
        rng = jax.random.PRNGKey(0)
        obs, state = adapter.reset(rng)

        # Check observation shape
        assert obs.shape == (5, 11, 11, 15)

        # Check observations are finite
        assert jnp.all(jnp.isfinite(obs))

    def test_step(self):
        """Test stepping the environment."""
        adapter = HarvestCommonCFAdapter(num_agents=5)
        rng = jax.random.PRNGKey(0)

        # Reset
        obs, state = adapter.reset(rng)

        # Sample random actions
        rng, action_rng = jax.random.split(rng)
        actions = jax.random.randint(action_rng, (5,), 0, 8)

        # Step
        rng, step_rng = jax.random.split(rng)
        obs_next, state_next, rewards, dones, infos = adapter.step(
            step_rng, state, actions
        )

        # Check shapes
        assert obs_next.shape == (5, 11, 11, 15)
        assert rewards.shape == (5,)

        # Check values are finite
        assert jnp.all(jnp.isfinite(obs_next))
        assert jnp.all(jnp.isfinite(rewards))


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_cf_adapter_coin_game(self):
        """Test creating adapter via factory."""
        adapter = create_cf_adapter('coin_game', num_agents=3)
        assert isinstance(adapter, CoinGameCFAdapter)
        assert adapter.num_agents == 3

    def test_create_cf_adapter_clean_up(self):
        """Test creating adapter via factory."""
        adapter = create_cf_adapter('clean_up', num_agents=5)
        assert isinstance(adapter, CleanupCFAdapter)
        assert adapter.num_agents == 5

    def test_create_cf_adapter_harvest(self):
        """Test creating adapter via factory."""
        adapter = create_cf_adapter('harvest_common_open', num_agents=5)
        assert isinstance(adapter, HarvestCommonCFAdapter)
        assert adapter.num_agents == 5

    def test_create_cf_adapter_invalid(self):
        """Test creating adapter with invalid environment."""
        with pytest.raises(ValueError):
            create_cf_adapter('invalid_env', num_agents=3)

    def test_list_available_adapters(self):
        """Test listing available adapters."""
        adapters = list_available_adapters()
        assert 'coin_game' in adapters
        assert 'clean_up' in adapters
        assert 'harvest_common_open' in adapters

    def test_get_env_spec(self):
        """Test getting environment spec without creating environment."""
        spec = get_env_spec('coin_game', num_agents=3)
        assert spec.env_name == 'coin_game'
        assert spec.num_agents == 3
        assert spec.action_dim == 7
        assert spec.obs_shape == (11, 11, 14)

    def test_get_env_spec_invalid(self):
        """Test getting spec with invalid environment."""
        with pytest.raises(ValueError):
            get_env_spec('invalid_env', num_agents=3)


class TestVerifyAdapterCompatibility:
    """Test verify_adapter_compatibility function."""

    def test_verify_all_pass(self):
        """Test verification passes with correct values."""
        adapter = CoinGameCFAdapter(num_agents=3)
        result = verify_adapter_compatibility(
            adapter,
            expected_num_agents=3,
            expected_action_dim=7,
            expected_obs_shape=(11, 11, 14),
        )
        assert result is True

    def test_verify_num_agents_fail(self):
        """Test verification fails with wrong num_agents."""
        adapter = CoinGameCFAdapter(num_agents=3)
        with pytest.raises(AssertionError):
            verify_adapter_compatibility(
                adapter,
                expected_num_agents=5,
            )

    def test_verify_action_dim_fail(self):
        """Test verification fails with wrong action_dim."""
        adapter = CoinGameCFAdapter(num_agents=3)
        with pytest.raises(AssertionError):
            verify_adapter_compatibility(
                adapter,
                expected_action_dim=10,
            )


class TestGetAdapterForEnv:
    """Test get_adapter_for_env function."""

    def test_get_adapter_for_coin_game(self):
        """Test getting adapter for existing coin_game env."""
        import socialjax
        env = socialjax.make('coin_game', num_agents=3)
        adapter = get_adapter_for_env(env)
        assert isinstance(adapter, CoinGameCFAdapter)
        assert adapter.num_agents == 3

    def test_get_adapter_for_clean_up(self):
        """Test getting adapter for existing clean_up env."""
        import socialjax
        env = socialjax.make('clean_up', num_agents=5)
        adapter = get_adapter_for_env(env)
        assert isinstance(adapter, CleanupCFAdapter)
        assert adapter.num_agents == 5


class TestJITCompilation:
    """Test JIT compilation of adapter methods."""

    def test_reset_jit(self):
        """Test that reset can be JIT compiled."""
        adapter = CoinGameCFAdapter(num_agents=3)

        @jax.jit
        def reset_fn(rng):
            return adapter.reset(rng)

        rng = jax.random.PRNGKey(0)
        obs, state = reset_fn(rng)
        assert obs.shape == (3, 11, 11, 14)

    def test_step_jit(self):
        """Test that step can be JIT compiled."""
        adapter = CoinGameCFAdapter(num_agents=3)
        rng = jax.random.PRNGKey(0)
        obs, state = adapter.reset(rng)

        @jax.jit
        def step_fn(rng, state, actions):
            return adapter.step(rng, state, actions)

        actions = jnp.array([0, 1, 2])
        rng, step_rng = jax.random.split(rng)
        obs_next, state_next, rewards, dones, infos = step_fn(
            step_rng, state, actions
        )
        assert obs_next.shape == (3, 11, 11, 14)


class TestIntegrationWithCFTrainer:
    """Test integration with CF Trainer."""

    def test_adapter_with_trainer(self):
        """Test using adapter with CF trainer."""
        from socialjax.algorithms.cf.cf_trainer import CFConfig, CFTrainer

        # Create adapter
        adapter = CoinGameCFAdapter(num_agents=3)

        # Create config
        config = CFConfig(
            env_name='coin_game',
            num_agents=3,
            num_envs=2,
            total_timesteps=100,
            num_steps=10,
            update_epochs=1,
        )

        # Create trainer with adapter's env
        trainer = CFTrainer(config, adapter.env)

        # Verify trainer is created correctly
        assert trainer.num_agents == 3
        assert trainer.action_dim == 7


class TestMultipleEpisodes:
    """Test multiple episode runs."""

    def test_run_multiple_episodes_coin_game(self):
        """Test running multiple episodes in coin game."""
        adapter = CoinGameCFAdapter(num_agents=3, num_inner_steps=100)
        rng = jax.random.PRNGKey(0)

        # Run for multiple steps
        obs, state = adapter.reset(rng)

        for step in range(50):
            rng, action_rng = jax.random.split(rng)
            actions = jax.random.randint(action_rng, (3,), 0, 7)
            rng, step_rng = jax.random.split(rng)
            obs, state, rewards, dones, infos = adapter.step(
                step_rng, state, actions
            )

            # Verify shapes stay consistent
            assert obs.shape == (3, 11, 11, 14)
            assert rewards.shape == (3,)

    def test_run_multiple_episodes_clean_up(self):
        """Test running multiple episodes in clean up."""
        adapter = CleanupCFAdapter(num_agents=5, num_inner_steps=100)
        rng = jax.random.PRNGKey(0)

        obs, state = adapter.reset(rng)

        for step in range(50):
            rng, action_rng = jax.random.split(rng)
            actions = jax.random.randint(action_rng, (5,), 0, 8)
            rng, step_rng = jax.random.split(rng)
            obs, state, rewards, dones, infos = adapter.step(
                step_rng, state, actions
            )

            assert obs.shape == (5, 11, 11, 19)
            assert rewards.shape == (5,)


class TestEdgeCases:
    """Test edge cases."""

    def test_coin_game_min_agents(self):
        """Test coin game with minimum agents (2)."""
        adapter = CoinGameCFAdapter(num_agents=2)
        assert adapter.num_agents == 2
        assert adapter.default_alpha == 1.0

        rng = jax.random.PRNGKey(0)
        obs, state = adapter.reset(rng)
        assert obs.shape == (2, 11, 11, 14)

    def test_clean_up_max_agents(self):
        """Test clean up with max agents (7)."""
        adapter = CleanupCFAdapter(num_agents=7)
        assert adapter.num_agents == 7
        assert adapter.default_alpha == 6.0

        rng = jax.random.PRNGKey(0)
        obs, state = adapter.reset(rng)
        assert obs.shape == (7, 11, 11, 19)

    def test_scalar_reward_broadcasting(self):
        """Test that scalar rewards are broadcast correctly."""
        adapter = CoinGameCFAdapter(num_agents=3)

        # Create a scalar reward
        scalar_reward = jnp.array(1.0)
        processed = adapter.process_rewards(scalar_reward)
        assert processed.shape == (3,)
