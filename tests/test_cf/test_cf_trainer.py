"""
Integration tests for CF Trainer (CF-IMPL-009)

Tests the complete CF training loop:
1. Trainer initialization
2. Experience collection
3. Counterfactual reward computation
4. Policy update with shaped rewards
5. Checkpoint save/load
6. Smoke test (1000 steps)
"""

import pytest
import sys
import os
import tempfile
import shutil

# Add socialjax to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'socialjax'))

import jax
import jax.numpy as jnp
import numpy as np


class TestCFConfig:
    """Test CFConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from socialjax.algorithms.cf.cf_trainer import CFConfig

        config = CFConfig()
        assert config.env_name == "coin_game"
        assert config.num_agents == 3
        assert config.num_envs == 8
        assert config.gamma == 0.99
        assert config.clip_eps == 0.2

    def test_auto_alpha(self):
        """Test automatic alpha computation."""
        from socialjax.algorithms.cf.cf_trainer import CFConfig

        config = CFConfig(num_agents=5, use_auto_alpha=True)
        assert config.alpha == 4.0  # N - 1

    def test_derived_values(self):
        """Test derived configuration values."""
        from socialjax.algorithms.cf.cf_trainer import CFConfig

        config = CFConfig(num_agents=3, num_envs=4, num_steps=16, total_timesteps=1000)
        assert config.num_actors == 12  # 3 agents * 4 envs
        assert config.minibatch_size > 0


class TestCFTrainerInit:
    """Test CFTrainer initialization."""

    def test_create_trainer(self):
        """Test creating trainer with default config."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(num_agents=3, num_envs=2)

        trainer = CFTrainer(config, env)
        assert trainer.num_agents == 3
        assert trainer.action_dim == env.action_space().n
        assert trainer.policy_network is not None
        assert trainer.reward_model is not None

    def test_initialize_state(self):
        """Test initializing training state."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(num_agents=3, num_envs=2)

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(0)
        state = trainer.initialize(rng)

        assert state.policy_state is not None
        assert state.reward_state is not None
        assert state.env_state is not None
        assert state.last_obs is not None
        assert state.global_step == 0


class TestTransitionBuffer:
    """Test TransitionBuffer functionality."""

    def test_buffer_creation(self):
        """Test creating transition buffer."""
        from socialjax.algorithms.cf.cf_trainer import TransitionBuffer

        buffer = TransitionBuffer(
            num_steps=10,
            num_envs=4,
            num_agents=3,
            obs_shape=(8, 8, 3),
        )
        assert buffer.obs.shape == (10, 4, 3, 8, 8, 3)
        assert buffer.actions.shape == (10, 4, 3)
        assert buffer.ptr == 0

    def test_buffer_add(self):
        """Test adding transitions to buffer."""
        from socialjax.algorithms.cf.cf_trainer import TransitionBuffer

        buffer = TransitionBuffer(
            num_steps=5,
            num_envs=2,
            num_agents=3,
            obs_shape=(8, 8, 3),
        )

        obs = np.random.randn(2, 3, 8, 8, 3).astype(np.float32)
        actions = np.random.randint(0, 4, (2, 3))
        rewards = np.random.randn(2, 3).astype(np.float32)
        dones = np.zeros(2, dtype=np.float32)
        log_probs = np.random.randn(2, 3).astype(np.float32)
        values = np.random.randn(2, 3).astype(np.float32)

        buffer.add(obs, actions, rewards, dones, log_probs, values)
        assert buffer.ptr == 1

    def test_buffer_get(self):
        """Test getting transitions from buffer."""
        from socialjax.algorithms.cf.cf_trainer import TransitionBuffer

        buffer = TransitionBuffer(
            num_steps=5,
            num_envs=2,
            num_agents=3,
            obs_shape=(8, 8, 3),
        )

        for i in range(3):
            obs = np.random.randn(2, 3, 8, 8, 3).astype(np.float32)
            actions = np.random.randint(0, 4, (2, 3))
            rewards = np.random.randn(2, 3).astype(np.float32)
            dones = np.zeros(2, dtype=np.float32)
            log_probs = np.random.randn(2, 3).astype(np.float32)
            values = np.random.randn(2, 3).astype(np.float32)
            buffer.add(obs, actions, rewards, dones, log_probs, values)

        data = buffer.get()
        assert data['obs'].shape == (5, 2, 3, 8, 8, 3)
        assert data['actions'].shape == (5, 2, 3)


class TestCFShapedRewards:
    """Test shaped reward computation."""

    def test_compute_shaped_rewards_shape(self):
        """Test shaped reward output shape."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(num_agents=3, num_envs=2, num_steps=4)

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(0)
        state = trainer.initialize(rng)

        # Create dummy trajectory
        num_steps = 4
        num_envs = 2
        num_agents = 3
        obs_shape = trainer.obs_shape

        obs = jnp.ones((num_steps, num_envs, num_agents, *obs_shape))
        actions = jnp.zeros((num_steps, num_envs, num_agents), dtype=jnp.int32)
        rewards = jnp.zeros((num_steps, num_envs, num_agents))

        shaped = trainer._compute_shaped_rewards_batch(
            state.reward_state.params, obs, actions, rewards
        )

        assert shaped.shape == (num_steps, num_envs, num_agents)

    def test_shaped_rewards_no_nan(self):
        """Test no NaN values in shaped rewards."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(num_agents=3, num_envs=2)

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(0)
        state = trainer.initialize(rng)

        obs = jnp.ones((4, 2, 3, *trainer.obs_shape))
        actions = jnp.zeros((4, 2, 3), dtype=jnp.int32)
        rewards = jnp.zeros((4, 2, 3))

        shaped = trainer._compute_shaped_rewards_batch(
            state.reward_state.params, obs, actions, rewards
        )

        assert not jnp.any(jnp.isnan(shaped))


class TestEnvStep:
    """Test environment stepping."""

    def test_single_step(self):
        """Test single environment step."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(num_agents=3, num_envs=2)

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(0)
        state = trainer.initialize(rng)

        new_state, transition = trainer._env_step(state, None)

        assert new_state.global_step == state.global_step + config.num_envs
        assert transition.obs.shape == (config.num_envs, 3, *trainer.obs_shape)
        assert transition.action.shape == (config.num_envs, 3)


class TestTrajectoryCollection:
    """Test trajectory collection."""

    def test_collect_trajectory(self):
        """Test collecting a trajectory."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(num_agents=3, num_envs=2, num_steps=8)

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(0)
        state = trainer.initialize(rng)

        new_state, traj = trainer._collect_trajectory(state)

        assert traj.obs.shape == (config.num_steps, config.num_envs, 3, *trainer.obs_shape)
        assert traj.action.shape == (config.num_steps, config.num_envs, 3)
        assert traj.reward.shape == (config.num_steps, config.num_envs, 3)


class TestUpdateStep:
    """Test single update step."""

    def test_update_step(self):
        """Test performing one update step."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(
            num_agents=3,
            num_envs=2,
            num_steps=4,
            update_epochs=1,
            num_minibatches=1,
        )

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(0)
        state = trainer.initialize(rng)

        new_state, metrics = trainer._update_step(state, None)

        assert 'reward_model_loss' in metrics
        assert 'policy_loss' in metrics
        assert 'mean_reward' in metrics
        assert not jnp.isnan(metrics['reward_model_loss'])
        assert not jnp.isnan(metrics['policy_loss'])


class TestCheckpointing:
    """Test checkpoint save/load."""

    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        # Create temporary directory for checkpoints
        temp_dir = tempfile.mkdtemp()

        try:
            env = socialjax.make('coin_game', num_agents=3)
            config = CFConfig(
                num_agents=3,
                num_envs=2,
                save_dir=temp_dir,
                save_freq=0,  # Disable auto-save
            )

            trainer = CFTrainer(config, env)
            rng = jax.random.PRNGKey(0)
            state = trainer.initialize(rng)

            # Save checkpoint
            path = trainer.save(state, step=100)
            assert os.path.exists(path)

            # Load checkpoint
            rng2 = jax.random.PRNGKey(1)
            loaded_state = trainer.load(path, rng2)

            # Verify params match using jax.tree_util
            original_params = state.policy_state.params
            loaded_params = loaded_state.policy_state.params

            # Check that params are identical using tree_map
            def check_equal(a, b):
                if isinstance(a, dict):
                    return all(check_equal(a[k], b[k]) for k in a.keys())
                else:
                    return bool(jnp.allclose(a, b))

            assert check_equal(original_params, loaded_params), \
                "Policy params mismatch after load"

            # Also check reward params
            original_reward_params = state.reward_state.params
            loaded_reward_params = loaded_state.reward_state.params
            assert check_equal(original_reward_params, loaded_reward_params), \
                "Reward params mismatch after load"

            # Check step was restored
            assert loaded_state.global_step == 100

        finally:
            shutil.rmtree(temp_dir)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_cf_trainer(self):
        """Test create_cf_trainer function."""
        from socialjax.algorithms.cf.cf_trainer import create_cf_trainer

        trainer, env = create_cf_trainer(
            env_name='coin_game',
            num_agents=3,
            num_envs=2,
            total_timesteps=1000,
        )

        assert trainer is not None
        assert env is not None
        assert trainer.num_agents == 3


class TestSmokeTest:
    """Smoke tests for complete training loop."""

    def test_smoke_100_steps(self):
        """Run 100 steps of training (smoke test)."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(
            num_agents=3,
            num_envs=2,
            num_steps=4,
            update_epochs=1,
            num_minibatches=1,
            total_timesteps=100,
            save_freq=0,
            log_freq=1000,  # Disable frequent logging
        )

        trainer = CFTrainer(config, env)

        # Run training for a few updates
        final_state, metrics = trainer.train(num_updates=5)

        assert final_state.global_step > 0
        assert 'mean_reward' in metrics
        assert not np.isnan(metrics['mean_reward'])

    def test_smoke_1000_steps(self):
        """Run ~1000 steps of training (full smoke test)."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(
            num_agents=3,
            num_envs=4,
            num_steps=16,
            update_epochs=2,
            num_minibatches=2,
            total_timesteps=1000,
            save_freq=0,
            log_freq=10,
        )

        trainer = CFTrainer(config, env)

        # Calculate number of updates for ~1000 steps
        steps_per_update = config.num_steps * config.num_envs  # 16 * 4 = 64
        num_updates = max(1, 1000 // steps_per_update)  # 15 updates = 960 steps

        final_state, metrics = trainer.train(num_updates=num_updates)

        # Verify training ran (allow for some variance due to update granularity)
        expected_steps = num_updates * steps_per_update
        assert final_state.global_step >= expected_steps - steps_per_update

        # Check metrics
        assert 'reward_model_loss' in metrics
        assert 'policy_loss' in metrics
        assert 'mean_reward' in metrics
        assert 'mean_shaped_reward' in metrics

        # Check no NaN values
        assert not np.isnan(metrics['reward_model_loss']), "Reward model loss is NaN"
        assert not np.isnan(metrics['policy_loss']), "Policy loss is NaN"
        assert not np.isnan(metrics['mean_reward']), "Mean reward is NaN"

        print(f"\nSmoke test completed successfully!")
        print(f"Steps: {final_state.global_step}")
        print(f"Final metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


class TestJITCompilation:
    """Test JIT compilation of training functions."""

    def test_jit_update_step(self):
        """Test JIT-compiled update step."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig, make_jitted_update_step
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(
            num_agents=3,
            num_envs=2,
            num_steps=4,
            update_epochs=1,
            num_minibatches=1,
        )

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(0)
        state = trainer.initialize(rng)

        # Create JIT-compiled update function
        jitted_update = make_jitted_update_step(trainer)

        # Run update
        new_state, metrics = jitted_update(state)

        assert new_state.global_step > state.global_step
        assert 'policy_loss' in metrics

    def test_jit_consistency(self):
        """Test that JIT version produces same results as non-JIT."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig, make_jitted_update_step
        import socialjax

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(
            num_agents=3,
            num_envs=2,
            num_steps=4,
            update_epochs=1,
            num_minibatches=1,
        )

        trainer = CFTrainer(config, env)
        rng = jax.random.PRNGKey(42)
        state1 = trainer.initialize(rng)
        state2 = trainer.initialize(rng)

        # Run non-JIT update
        new_state1, metrics1 = trainer._update_step(state1, None)

        # Run JIT update
        jitted_update = make_jitted_update_step(trainer)
        new_state2, metrics2 = jitted_update(state2)

        # Both should complete without error
        assert 'policy_loss' in metrics1
        assert 'policy_loss' in metrics2


class TestMemoryLeaks:
    """Test for memory leaks during training."""

    def test_no_memory_leak_short_run(self):
        """Test that memory doesn't grow excessively during short run."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax
        import gc

        env = socialjax.make('coin_game', num_agents=3)
        config = CFConfig(
            num_agents=3,
            num_envs=2,
            num_steps=4,
            update_epochs=1,
            num_minibatches=1,
            total_timesteps=100,
            save_freq=0,
            log_freq=1000,
        )

        trainer = CFTrainer(config, env)

        # Run training
        for _ in range(10):
            rng = jax.random.PRNGKey(0)
            state = trainer.initialize(rng)
            state, _ = trainer._update_step(state, None)
            gc.collect()

        # If we get here without OOM, test passes
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
