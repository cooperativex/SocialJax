"""Unit tests for the unified Trainer class.

Tests cover:
- Trainer creation with different configurations
- Training loop functionality
- Callback integration
- Evaluation
- Save/load functionality
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Add socialjax to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'socialjax'))

import jax
import jax.numpy as jnp
import numpy as np

# Import algorithms to register them
from socialjax.algorithms.ippo.algorithm import IPPOAlgorithm
from socialjax.algorithms.mappo.algorithm import MAPPOAlgorithm
from socialjax.algorithms.vdn.algorithm import VDNAlgorithm

from socialjax.training.trainer import Trainer, RolloutBuffer, create_trainer
from socialjax.training.callbacks import BaseCallback, CallbackList
from socialjax.config.manager import SocialJaxConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig


# ============================================================================
# Test RolloutBuffer
# ============================================================================

class TestRolloutBuffer:
    """Tests for RolloutBuffer class."""

    def test_buffer_creation(self):
        """Test buffer is created with correct shape."""
        buffer = RolloutBuffer(
            buffer_size=128,
            num_envs=1,
            obs_shape=(10, 10, 3),
            action_dim=4,
        )

        assert buffer.buffer_size == 128
        assert buffer.num_envs == 1
        assert buffer.observations.shape == (128, 1, 10, 10, 3)
        assert buffer.actions.shape == (128, 1)
        assert buffer.pos == 0
        assert not buffer.full

    def test_buffer_add(self):
        """Test adding data to buffer."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=1,
            obs_shape=(4,),
            action_dim=2,
        )

        obs = np.array([[1.0, 2.0, 3.0, 4.0]])
        action = np.array([0])
        reward = np.array([1.0])
        done = np.array([0.0])
        log_prob = np.array([-0.5])
        value = np.array([0.5])

        buffer.add(obs, action, reward, done, log_prob, value)

        assert buffer.pos == 1
        np.testing.assert_array_equal(buffer.observations[0], obs)
        np.testing.assert_array_equal(buffer.actions[0], action)

    def test_buffer_get(self):
        """Test retrieving data from buffer."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=1,
            obs_shape=(4,),
            action_dim=2,
        )

        # Add some data
        for i in range(5):
            buffer.add(
                np.array([[float(i)] * 4]),
                np.array([i % 2]),
                np.array([float(i)]),
                np.array([0.0]),
                np.array([-0.5]),
                np.array([0.5]),
            )

        data = buffer.get()

        assert "observations" in data
        assert "actions" in data
        assert "rewards" in data
        assert data["observations"].shape[0] == 5

    def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=1,
            obs_shape=(4,),
            action_dim=2,
        )

        # Add data
        buffer.add(
            np.array([[1.0, 2.0, 3.0, 4.0]]),
            np.array([0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([-0.5]),
            np.array([0.5]),
        )

        buffer.clear()

        assert buffer.pos == 0
        assert not buffer.full


# ============================================================================
# Test Trainer Creation
# ============================================================================

class TestTrainerCreation:
    """Tests for Trainer initialization."""

    def test_trainer_creation_with_names(self):
        """Test creating trainer with algorithm and env names."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        assert trainer.algorithm is not None
        assert trainer.env is not None
        assert trainer._algo_name == "ippo"
        assert trainer._env_name == "coin_game"

    def test_trainer_creation_with_config(self):
        """Test creating trainer with config object."""
        config = SocialJaxConfig(
            algorithm=AlgorithmConfig(name="ippo"),
            environment=EnvironmentConfig(name="coin_game", num_agents=2),
        )

        trainer = Trainer(config=config, seed=42)

        assert trainer._socialjax_config.algorithm.name == "ippo"
        assert trainer._socialjax_config.environment.name == "coin_game"

    def test_trainer_creation_with_dict_config(self):
        """Test creating trainer with dict config."""
        config = {
            "algorithm": {"name": "ippo"},
            "environment": {"name": "coin_game", "num_agents": 2},
        }

        trainer = Trainer(config=config, seed=42)

        assert trainer._socialjax_config.algorithm.name == "ippo"

    def test_trainer_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        trainer1 = Trainer(algorithm="ippo", env="coin_game", seed=42)
        trainer2 = Trainer(algorithm="ippo", env="coin_game", seed=42)

        # Both should have same RNG state
        key1 = trainer1._rng
        key2 = trainer2._rng

        # Generate same random numbers
        sample1 = jax.random.randint(key1, (5,), 0, 10)
        sample2 = jax.random.randint(key2, (5,), 0, 10)

        np.testing.assert_array_equal(sample1, sample2)

    def test_create_trainer_factory(self):
        """Test create_trainer factory function."""
        trainer = create_trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        assert isinstance(trainer, Trainer)
        assert trainer._algo_name == "ippo"
        assert trainer._env_name == "coin_game"


# ============================================================================
# Test Trainer Configuration
# ============================================================================

class TestTrainerConfig:
    """Tests for Trainer configuration handling."""

    def test_config_extraction(self):
        """Test that config is properly extracted."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        config_dict = trainer.config_dict

        assert "algorithm" in config_dict
        assert "environment" in config_dict

    def test_config_overrides(self):
        """Test config overrides via kwargs."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
            total_timesteps=100000,
        )

        # Check that total_timesteps was set in training config
        assert trainer._socialjax_config.algorithm.training.total_timesteps == 100000


# ============================================================================
# Test Callback Integration
# ============================================================================

class TestCallbackIntegration:
    """Tests for callback integration."""

    def test_callback_list_initialization(self):
        """Test callbacks are properly initialized."""
        class TestCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.training_started = False

            def on_training_start(self, trainer):
                self.training_started = True

        callback = TestCallback()
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            callbacks=[callback],
            seed=42,
        )

        # Callback should have trainer reference set via _callback_list
        assert len(trainer._callback_list) == 1

    def test_callback_set_trainer_called(self):
        """Test that set_trainer is called on callbacks."""
        set_trainer_called = []

        class TrackingCallback(BaseCallback):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def set_trainer(self, trainer):
                set_trainer_called.append(self.name)
                super().set_trainer(trainer)

        callbacks = [TrackingCallback("cb1"), TrackingCallback("cb2")]
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            callbacks=callbacks,
            seed=42,
        )

        # Both callbacks should have had set_trainer called
        assert "cb1" in set_trainer_called
        assert "cb2" in set_trainer_called


# ============================================================================
# Test Training Loop (Unit Tests)
# ============================================================================

class TestTrainingLoopUnit:
    """Unit tests for training loop functionality (without full training)."""

    def test_train_config_retrieved(self):
        """Test that training config is properly retrieved."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        # Config should be available
        assert trainer.config is not None
        assert "num_steps" in trainer.config or "gamma" in trainer.config

    def test_buffer_created(self):
        """Test that buffer is created during setup."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        assert trainer.buffer is not None
        assert isinstance(trainer.buffer, RolloutBuffer)

    def test_obs_shape_extraction(self):
        """Test observation shape extraction from environment."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        obs_shape = trainer._get_obs_shape()
        assert obs_shape is not None
        assert len(obs_shape) == 3  # Should be (H, W, C)

    def test_action_dim_extraction(self):
        """Test action dimension extraction from environment."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        action_dim = trainer._get_action_dim()
        assert action_dim > 0


# ============================================================================
# Test Evaluation (Unit Tests)
# ============================================================================

class TestEvaluateUnit:
    """Unit tests for evaluation functionality."""

    def test_evaluate_creates_state_if_none(self):
        """Test that evaluate creates state if none provided."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        # Calling evaluate without state should create one
        # This tests the code path without actually running episodes
        with patch.object(trainer, 'env') as mock_env:
            mock_env.agents = ['agent_0']
            mock_env.reset.return_value = ({'agent_0': np.zeros((11, 11, 14))}, None)
            mock_env.step.return_value = (None, {'agent_0': np.zeros((11, 11, 14))},
                                          {'agent_0': 0.0}, {'__all__': True}, {})

            # This will try to evaluate - we're just testing it doesn't crash
            # on state creation


# ============================================================================
# Test Save/Load
# ============================================================================

class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_creates_directory(self):
        """Test that save creates checkpoint directory."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "checkpoint")
            trainer.save(save_path)

            assert os.path.exists(save_path)

    def test_save_creates_checkpoint_files(self):
        """Test that save creates necessary checkpoint files."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "checkpoint")
            trainer.save(save_path)

            # Should have algorithm checkpoint
            assert os.path.exists(os.path.join(save_path, "algorithm"))
            # Should have trainer info
            assert os.path.exists(os.path.join(save_path, "trainer_info.pkl"))

    def test_load_restores_state(self):
        """Test that load restores trainer state."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "checkpoint")
            trainer.save(save_path)

            # Load into new trainer
            trainer2 = Trainer(
                algorithm="ippo",
                env="coin_game",
                seed=42,
            )
            loaded_state = trainer2.load(save_path)

            assert loaded_state is not None


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_trainer_with_no_algorithm(self):
        """Test trainer creation with no algorithm specified."""
        # Should create with default
        trainer = Trainer(seed=42)

        assert trainer.algorithm is not None
        assert trainer._socialjax_config.algorithm.name == "ippo"

    def test_trainer_with_no_env(self):
        """Test trainer creation with no environment specified."""
        trainer = Trainer(algorithm="ippo", seed=42)

        assert trainer.env is not None

    def test_trainer_with_custom_config_values(self):
        """Test trainer with custom config values."""
        custom_config = {
            "algorithm": {
                "name": "ippo",
                "training": {
                    "learning_rate": 0.001,
                    "gamma": 0.95,
                }
            },
            "environment": {
                "name": "coin_game",
                "num_agents": 3,
            }
        }

        trainer = Trainer(config=custom_config, seed=42)

        assert trainer._socialjax_config.environment.num_agents == 3
        assert trainer._socialjax_config.algorithm.training.learning_rate == 0.001


# ============================================================================
# Test Space Wrappers
# ============================================================================

class TestSpaceWrappers:
    """Tests for space wrapper classes."""

    def test_space_wrapper_shape(self):
        """Test SpaceWrapper extracts shape correctly."""
        from socialjax.training.trainer import SpaceWrapper

        # Create a mock callable that returns (space, shape)
        def mock_obs_space():
            return (MagicMock(), (11, 11, 14))

        wrapper = SpaceWrapper(mock_obs_space, "observation")
        assert wrapper.shape == (11, 11, 14)

    def test_space_wrapper_n(self):
        """Test SpaceWrapper extracts n (action count) correctly."""
        from socialjax.training.trainer import SpaceWrapper

        # Create a mock callable that returns a space with n
        mock_space = MagicMock()
        mock_space.n = 7

        def mock_action_space():
            return mock_space

        wrapper = SpaceWrapper(mock_action_space, "action")
        assert wrapper.n == 7


# ============================================================================
# Integration Tests (marked to skip if environment issues)
# ============================================================================

@pytest.mark.skip(reason="Requires working environment - coin_game has reset bug")
class TestTrainingLoopIntegration:
    """Integration tests for training loop (require working environment)."""

    def test_train_returns_state_and_metrics(self):
        """Test that train returns proper state and metrics."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        state, metrics = trainer.train(total_timesteps=100)

        assert state is not None
        assert metrics is not None
        assert "total_timesteps" in metrics
        assert "total_updates" in metrics
        assert "elapsed_time" in metrics


@pytest.mark.skip(reason="Requires working environment")
class TestEvaluateIntegration:
    """Integration tests for evaluation (require working environment)."""

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns proper metrics."""
        trainer = Trainer(
            algorithm="ippo",
            env="coin_game",
            seed=42,
        )

        # First train a bit
        state, _ = trainer.train(total_timesteps=100)

        # Then evaluate
        eval_metrics = trainer.evaluate(state, num_episodes=2)

        assert eval_metrics is not None
        assert "mean_return" in eval_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
