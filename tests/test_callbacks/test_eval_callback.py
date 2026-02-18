"""Unit tests for EvalCallback.

Test criteria:
- EvalCallback evaluates at correct frequency
- Evaluation runs specified number of episodes
- Mean and std rewards are computed correctly
- Best model is tracked and saved
- Unit tests exist: test_eval_frequency, test_episode_count, test_reward_computation, test_best_model_tracking
"""

import os
import pickle
import pytest
import sys
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any, List

# Set up path for imports
sys.path.insert(0, 'socialjax')


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, num_agents=5, episode_reward=1.0, max_steps=10):
        self.num_agents = num_agents
        self.episode_reward = episode_reward
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self):
        """Reset the environment."""
        self._step_count = 0
        obs = {f"agent_{i}": np.zeros((10, 10, 3)) for i in range(self.num_agents)}
        state = MagicMock()
        return obs, state

    def step(self, state, actions):
        """Step the environment."""
        self._step_count += 1
        done = self._step_count >= self.max_steps

        obs = {f"agent_{i}": np.zeros((10, 10, 3)) for i in range(self.num_agents)}
        rewards = {f"agent_{i}": self.episode_reward for i in range(self.num_agents)}
        dones = {f"agent_{i}": done for i in range(self.num_agents)}

        return obs, state, rewards, dones, {}

    def action_space(self):
        """Return mock action space."""
        mock_space = MagicMock()
        mock_space.sample = lambda x=None: {f"agent_{i}": 0 for i in range(self.num_agents)}
        return mock_space


class TestEvalCallbackImport:
    """Test that EvalCallback can be imported from various locations."""

    def test_import_from_eval_callback(self):
        """Test importing EvalCallback directly from eval_callback module."""
        from socialjax.training.callbacks.eval_callback import EvalCallback
        assert EvalCallback is not None

    def test_import_from_callbacks_init(self):
        """Test importing EvalCallback from callbacks __init__."""
        from socialjax.training.callbacks import EvalCallback
        assert EvalCallback is not None

    def test_import_from_training_init(self):
        """Test importing EvalCallback from training __init__."""
        from socialjax.training import EvalCallback
        assert EvalCallback is not None


class TestEvalCallbackInit:
    """Test EvalCallback initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment()

    def test_default_initialization(self):
        """Test EvalCallback with default parameters."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(eval_env=self.mock_env)
        assert callback.eval_env == self.mock_env
        assert callback.eval_freq == 1000
        assert callback.n_eval_episodes == 10
        assert callback.best_model_save_path is None
        assert callback.deterministic is True
        assert callback.verbose is False
        assert callback.warn is True

    def test_custom_initialization(self):
        """Test EvalCallback with custom parameters."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(
            eval_env=self.mock_env,
            eval_freq=500,
            n_eval_episodes=20,
            best_model_save_path="/tmp/best_models",
            deterministic=False,
            verbose=True,
            warn=False
        )
        assert callback.eval_freq == 500
        assert callback.n_eval_episodes == 20
        assert callback.best_model_save_path == "/tmp/best_models"
        assert callback.deterministic is False
        assert callback.verbose is True
        assert callback.warn is False

    def test_invalid_eval_freq_raises_error(self):
        """Test that invalid eval_freq raises ValueError."""
        from socialjax.training.callbacks import EvalCallback

        with pytest.raises(ValueError, match="eval_freq must be a positive integer"):
            EvalCallback(eval_env=self.mock_env, eval_freq=0)

        with pytest.raises(ValueError, match="eval_freq must be a positive integer"):
            EvalCallback(eval_env=self.mock_env, eval_freq=-1)

    def test_invalid_n_eval_episodes_raises_error(self):
        """Test that invalid n_eval_episodes raises ValueError."""
        from socialjax.training.callbacks import EvalCallback

        with pytest.raises(ValueError, match="n_eval_episodes must be a positive integer"):
            EvalCallback(eval_env=self.mock_env, n_eval_episodes=0)

        with pytest.raises(ValueError, match="n_eval_episodes must be a positive integer"):
            EvalCallback(eval_env=self.mock_env, n_eval_episodes=-5)

    def test_none_eval_env_raises_error(self):
        """Test that None eval_env raises ValueError."""
        from socialjax.training.callbacks import EvalCallback

        with pytest.raises(ValueError, match="eval_env cannot be None"):
            EvalCallback(eval_env=None)

    def test_inherits_from_base_callback(self):
        """Test that EvalCallback inherits from BaseCallback."""
        from socialjax.training.callbacks import EvalCallback, BaseCallback

        callback = EvalCallback(eval_env=self.mock_env)
        assert isinstance(callback, BaseCallback)

    def test_has_required_hook_methods(self):
        """Test that EvalCallback has required hook methods."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(eval_env=self.mock_env)

        assert hasattr(callback, 'on_training_start')
        assert hasattr(callback, 'on_training_end')
        assert hasattr(callback, 'on_step')
        assert hasattr(callback, 'on_rollout_start')
        assert hasattr(callback, 'on_rollout_end')
        assert hasattr(callback, 'on_update_start')
        assert hasattr(callback, 'on_update_end')
        assert hasattr(callback, 'set_trainer')

    def test_initial_tracking_values(self):
        """Test that initial tracking values are set correctly."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(eval_env=self.mock_env)

        assert callback.best_mean_reward == float('-inf')
        assert callback.last_mean_reward is None
        assert callback.last_std_reward is None
        assert callback.get_update_count() == 0


class TestEvalFrequency:
    """Test that evaluations occur at the correct frequency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_env = MockEnvironment(episode_reward=1.0)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_eval_at_correct_frequency(self):
        """Test that evaluations run every eval_freq updates."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(
            eval_env=self.mock_env,
            eval_freq=5,
            n_eval_episodes=1,
            verbose=False
        )

        # Create mock trainer
        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()

        # Set up mock compute_action
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)

        # Simulate 10 updates
        for i in range(1, 11):
            callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should have evaluated at updates 5 and 10
        assert len(callback.get_evaluation_history()) == 2

    def test_eval_at_frequency_one(self):
        """Test that eval_freq=1 evaluates on every update."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(
            eval_env=self.mock_env,
            eval_freq=1,
            n_eval_episodes=1
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)

        # Simulate 3 updates
        for i in range(1, 4):
            callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should have evaluated 3 times
        assert len(callback.get_evaluation_history()) == 3

    def test_no_eval_before_frequency(self):
        """Test that no evaluation runs before eval_freq is reached."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(
            eval_env=self.mock_env,
            eval_freq=100,
            n_eval_episodes=1
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)

        # Simulate 50 updates (less than eval_freq)
        for i in range(1, 51):
            callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should not have evaluated
        assert len(callback.get_evaluation_history()) == 0

    def test_update_count_tracking(self):
        """Test that update count is tracked correctly."""
        from socialjax.training.callbacks import EvalCallback

        callback = EvalCallback(
            eval_env=self.mock_env,
            eval_freq=100
        )

        mock_trainer = MagicMock()
        mock_trainer.algorithm = MagicMock()
        mock_trainer.algorithm_state = MagicMock()

        # Simulate 25 updates
        for i in range(25):
            callback.on_update_end(mock_trainer, {})

        assert callback.get_update_count() == 25


class TestEpisodeCount:
    """Test that the correct number of episodes are run during evaluation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_correct_number_of_episodes(self):
        """Test that evaluation runs the specified number of episodes."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=5
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        history = callback.get_evaluation_history()
        assert len(history) == 1
        assert history[0]['n_episodes'] == 5

    def test_single_episode_evaluation(self):
        """Test evaluation with n_eval_episodes=1."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=5.0, max_steps=3)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        history = callback.get_evaluation_history()
        assert len(history) == 1
        assert history[0]['n_episodes'] == 1
        assert len(history[0]['episode_rewards']) == 1


class TestRewardComputation:
    """Test that mean and std rewards are computed correctly."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_mean_reward_computation(self):
        """Test that mean reward is computed correctly."""
        from socialjax.training.callbacks import EvalCallback

        episode_reward = 2.5
        mock_env = MockEnvironment(episode_reward=episode_reward, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=3
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        # Each episode has 5 steps * 2.5 reward * 5 agents = 62.5 per episode
        expected_reward = episode_reward * 5 * 5  # reward * max_steps * num_agents

        history = callback.get_evaluation_history()
        assert len(history) == 1

        # Check mean reward
        assert history[0]['mean_reward'] == pytest.approx(expected_reward, rel=0.1)

    def test_std_reward_computation(self):
        """Test that std reward is computed correctly."""
        from socialjax.training.callbacks import EvalCallback

        # All episodes will have the same reward, so std should be 0
        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=3
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        history = callback.get_evaluation_history()
        # All rewards are identical, so std should be 0
        assert history[0]['std_reward'] == pytest.approx(0.0, abs=0.1)

    def test_reward_properties_updated(self):
        """Test that last_mean_reward and last_std_reward properties are updated."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=3.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=2
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        # Before evaluation
        assert callback.last_mean_reward is None
        assert callback.last_std_reward is None

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        # After evaluation
        assert callback.last_mean_reward is not None
        assert callback.last_std_reward is not None
        assert callback.last_mean_reward > 0

    def test_episode_rewards_stored(self):
        """Test that individual episode rewards are stored."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=3
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        history = callback.get_evaluation_history()
        episode_rewards = history[0]['episode_rewards']

        assert isinstance(episode_rewards, list)
        assert len(episode_rewards) == 3
        assert all(r > 0 for r in episode_rewards)


class TestBestModelTracking:
    """Test best model tracking and saving."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_best_mean_reward_tracking(self):
        """Test that best_mean_reward is tracked correctly."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)

        # Initially, best_mean_reward is -inf
        assert callback.best_mean_reward == float('-inf')

        # After first evaluation
        callback.on_update_end(mock_trainer, {})
        first_best = callback.best_mean_reward
        assert first_best > float('-inf')

    def test_best_model_saved_when_improved(self):
        """Test that best model is saved when mean reward improves."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=5.0, max_steps=5)
        best_path = os.path.join(self.temp_dir, "best")

        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            best_model_save_path=best_path,
            verbose=False
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()

        # Create mock save method that actually saves
        def mock_save(state, path):
            save_dict = {"params": {"test": 1}, "update_step": 1}
            with open(path, "wb") as f:
                pickle.dump(save_dict, f)

        mock_algorithm.save = mock_save
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        # Check that best model file was created
        best_model_path = os.path.join(best_path, "best_model.pkl")
        assert os.path.exists(best_model_path)

    def test_best_model_not_saved_when_no_improvement(self):
        """Test that best model is not saved when there's no improvement."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        best_path = os.path.join(self.temp_dir, "best_no_improve")

        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            best_model_save_path=best_path
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()

        save_count = [0]

        def mock_save(state, path):
            save_count[0] += 1

        mock_algorithm.save = mock_save
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)

        # First evaluation - should save
        callback.on_update_end(mock_trainer, {})
        assert save_count[0] == 1

        # Second evaluation - same reward, should not save
        callback.on_update_end(mock_trainer, {})
        assert save_count[0] == 1  # Still 1, not 2

    def test_best_model_directory_created(self):
        """Test that best model directory is created on training start."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment()
        new_dir = os.path.join(self.temp_dir, "new_best_dir")

        callback = EvalCallback(
            eval_env=mock_env,
            best_model_save_path=new_dir
        )

        mock_trainer = MagicMock()
        callback.on_training_start(mock_trainer)

        assert os.path.exists(new_dir)

    def test_no_save_when_path_not_specified(self):
        """Test that no save is attempted when path is not specified."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            best_model_save_path=None  # No save path
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.save = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        # Save should not have been called
        mock_algorithm.save.assert_not_called()


class TestDeterministicMode:
    """Test deterministic evaluation mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_deterministic_flag_passed_to_compute_action(self):
        """Test that deterministic flag is passed to compute_action."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            deterministic=True
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        # Verify compute_action was called with deterministic=True
        mock_algorithm.compute_action.assert_called()
        call_kwargs = mock_algorithm.compute_action.call_args[1]
        assert call_kwargs.get('deterministic') is True

    def test_stochastic_mode(self):
        """Test that deterministic=False works."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            deterministic=False
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        # Verify compute_action was called with deterministic=False
        call_kwargs = mock_algorithm.compute_action.call_args[1]
        assert call_kwargs.get('deterministic') is False


class TestVerboseLogging:
    """Test verbose logging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_verbose_prints_on_training_start(self, capsys):
        """Test that verbose=True prints info on training start."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment()
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=100,
            n_eval_episodes=5,
            best_model_save_path=self.temp_dir,
            verbose=True
        )

        mock_trainer = MagicMock()
        callback.on_training_start(mock_trainer)

        captured = capsys.readouterr()
        assert "EvalCallback" in captured.out
        assert "100" in captured.out  # eval_freq
        assert "5" in captured.out  # n_eval_episodes

    def test_verbose_prints_on_evaluation(self, capsys):
        """Test that verbose=True prints evaluation results."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            verbose=True
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        captured = capsys.readouterr()
        assert "Mean reward" in captured.out

    def test_non_verbose_silent(self, capsys):
        """Test that verbose=False produces no output."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            verbose=False
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        captured = capsys.readouterr()
        assert captured.out == ""


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_no_algorithm_state(self):
        """Test handling when trainer has no algorithm_state."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment()
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            warn=True
        )

        mock_trainer = MagicMock()
        mock_trainer.algorithm_state = None
        mock_trainer.algorithm = MagicMock()

        callback.on_training_start(mock_trainer)
        # Should not raise error
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        # No evaluations should have run
        assert len(callback.get_evaluation_history()) == 0

    def test_no_algorithm(self):
        """Test handling when trainer has no algorithm."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment()
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            warn=True
        )

        mock_trainer = MagicMock()
        mock_trainer.algorithm = None
        mock_trainer.algorithm_state = MagicMock()

        callback.on_training_start(mock_trainer)
        # Should not raise error
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        # No evaluations should have run
        assert len(callback.get_evaluation_history()) == 0

    def test_algorithm_without_compute_action(self):
        """Test handling when algorithm doesn't have compute_action method."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1,
            warn=False
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock(spec=[])  # No methods
        mock_state = MagicMock()

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        # Should not raise error, should fall back to random actions
        callback.on_update_end(mock_trainer, {"loss": 0.5})

    def test_reset_clears_state(self):
        """Test that reset() clears internal state."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        # Verify state was accumulated
        assert callback.get_update_count() == 1
        assert len(callback.get_evaluation_history()) == 1
        assert callback.best_mean_reward > float('-inf')

        # Reset
        callback.reset()

        assert callback.get_update_count() == 0
        assert len(callback.get_evaluation_history()) == 0
        assert callback.best_mean_reward == float('-inf')
        assert callback.last_mean_reward is None

    def test_get_evaluation_history_returns_copy(self):
        """Test that get_evaluation_history returns a copy."""
        from socialjax.training.callbacks import EvalCallback

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)
        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})

        history1 = callback.get_evaluation_history()
        history1.append({"fake": "data"})

        history2 = callback.get_evaluation_history()
        assert len(history2) == 1  # Original not modified


class TestCallbackListIntegration:
    """Test EvalCallback integration with CallbackList."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_eval_callback_in_callback_list(self):
        """Test that EvalCallback works in CallbackList."""
        from socialjax.training.callbacks import EvalCallback, CallbackList

        mock_env = MockEnvironment(episode_reward=1.0, max_steps=5)

        callback = EvalCallback(
            eval_env=mock_env,
            eval_freq=1,
            n_eval_episodes=1
        )

        callback_list = CallbackList([callback])

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_algorithm.compute_action.return_value = ({f"agent_{i}": 0 for i in range(5)},)

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback_list.on_training_start(mock_trainer)
        callback_list.on_update_end(mock_trainer, {"loss": 0.5})

        # Verify evaluation ran
        assert len(callback.get_evaluation_history()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
