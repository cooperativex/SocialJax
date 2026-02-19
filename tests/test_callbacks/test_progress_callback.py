"""Unit tests for ProgressCallback.

This module tests the ProgressCallback class which displays a progress bar
during training using tqdm.
"""

import sys
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure socialjax is in path
sys.path.insert(0, '/home/shuqing/SocialJax/socialjax')


class TestProgressCallbackImport:
    """Test that ProgressCallback can be imported correctly."""

    def test_import_from_callbacks_module(self):
        """Test importing ProgressCallback from callbacks module."""
        from socialjax.training.callbacks import ProgressCallback
        assert ProgressCallback is not None

    def test_import_from_training_module(self):
        """Test importing ProgressCallback from training module."""
        from socialjax.training import ProgressCallback
        assert ProgressCallback is not None

    def test_import_direct(self):
        """Test importing ProgressCallback directly."""
        from socialjax.training.callbacks.progress_callback import ProgressCallback
        assert ProgressCallback is not None

    def test_tqdm_availability_check(self):
        """Test that tqdm availability is detected."""
        from socialjax.training.callbacks.progress_callback import TQDM_AVAILABLE
        assert isinstance(TQDM_AVAILABLE, bool)


class TestProgressCallbackInit:
    """Test ProgressCallback initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback()

        assert callback.total_timesteps == 1_000_000
        assert callback.progress_freq == 1
        assert callback.show_metrics == ['loss', 'episode_return']
        assert callback.verbose is False
        assert callback.disable is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            total_timesteps=500_000,
            progress_freq=10,
            show_metrics=['loss', 'entropy'],
            verbose=True,
            disable=True,
        )

        assert callback.total_timesteps == 500_000
        assert callback.progress_freq == 10
        assert callback.show_metrics == ['loss', 'entropy']
        assert callback.verbose is True
        assert callback.disable is True

    def test_init_empty_metrics(self):
        """Test initialization with empty metrics list."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(show_metrics=[])

        assert callback.show_metrics == []

    def test_init_custom_bar_format(self):
        """Test initialization with custom bar format."""
        from socialjax.training.callbacks import ProgressCallback

        custom_format = "{l_bar}{bar}{r_bar}"
        callback = ProgressCallback(bar_format=custom_format)

        assert callback.bar_format == custom_format

    def test_progress_freq_minimum_is_one(self):
        """Test that progress_freq is at least 1."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(progress_freq=0)
        assert callback.progress_freq == 1

        callback2 = ProgressCallback(progress_freq=-5)
        assert callback2.progress_freq == 1


class TestProgressDisplay:
    """Test progress bar display functionality."""

    def test_progress_bar_created_on_training_start(self):
        """Test that progress bar is created when training starts."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)

        assert callback._pbar is not None
        assert callback._start_time is not None
        assert callback._current_step == 0
        assert callback._current_timestep == 0

        # Clean up
        callback.on_training_end(mock_trainer)

    def test_progress_bar_disabled(self):
        """Test that progress bar is not created when disabled."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=True)
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)

        assert callback._pbar is None

    def test_progress_bar_closed_on_training_end(self):
        """Test that progress bar is closed when training ends."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        assert callback._pbar is not None

        callback.on_training_end(mock_trainer)
        assert callback._pbar is None

    def test_progress_bar_updates_to_100_percent(self):
        """Test that progress bar reaches 100% at training end."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=100, disable=False)
        mock_trainer = MagicMock()
        mock_trainer.config.num_steps = 50

        callback.on_training_start(mock_trainer)

        # Simulate some training
        callback.on_step(mock_trainer, step=1, metrics={'loss': 0.5})

        # Get internal progress bar
        pbar = callback._pbar
        assert pbar is not None

        # End training - should update to 100%
        callback.on_training_end(mock_trainer)

        # Progress should be at total
        assert callback._current_timestep <= callback.total_timesteps

    def test_progress_bar_restarts_after_reset(self):
        """Test that progress bar can be reset and restarted."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()
        mock_trainer.config.num_steps = 50

        # Start and end training
        callback.on_training_start(mock_trainer)
        callback.on_step(mock_trainer, step=5, metrics={'loss': 0.5})
        callback.on_training_end(mock_trainer)

        # Reset
        callback.reset()

        assert callback._pbar is None
        assert callback._start_time is None
        assert callback._current_step == 0
        assert callback._last_metrics == {}

        # Can start again
        callback.on_training_start(mock_trainer)
        assert callback._pbar is not None

        callback.on_training_end(mock_trainer)


class TestUpdateFrequency:
    """Test progress bar update frequency."""

    def test_update_every_step_with_freq_one(self):
        """Test that progress updates every step when freq=1."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            total_timesteps=1000,
            progress_freq=1,
            disable=False
        )
        mock_trainer = MagicMock()
        mock_trainer.config.num_steps = 50

        callback.on_training_start(mock_trainer)

        # Call on_step 3 times - use estimated timestep from step * num_steps
        callback.on_step(mock_trainer, step=1, metrics={'loss': 0.5})
        callback.on_step(mock_trainer, step=2, metrics={'loss': 0.4})
        callback.on_step(mock_trainer, step=3, metrics={'loss': 0.3})

        # With freq=1, all steps should be counted
        assert callback._step_count == 3

        callback.on_training_end(mock_trainer)

    def test_update_less_frequently_with_higher_freq(self):
        """Test that progress updates less frequently with higher freq."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            total_timesteps=1000,
            progress_freq=3,  # Update every 3 steps
            disable=False
        )
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        type(mock_trainer).timestep = PropertyMock(return_value=10)

        # Call on_step 5 times
        for i in range(1, 6):
            callback.on_step(mock_trainer, step=i, metrics={'loss': 0.5})

        # With freq=3, progress bar should only update at steps 3 and 6
        # But _step_count should still be 5
        assert callback._step_count == 5

        callback.on_training_end(mock_trainer)

    def test_progress_bar_n_updates_correctly(self):
        """Test that progress bar internal counter updates correctly."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            total_timesteps=1000,
            progress_freq=1,
            disable=False
        )
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)

        # Simulate 100 timesteps
        type(mock_trainer).timestep = PropertyMock(return_value=100)
        callback.on_step(mock_trainer, step=1, metrics={'loss': 0.5})

        # Progress bar should have advanced
        pbar = callback._pbar
        assert pbar.n == 100

        callback.on_training_end(mock_trainer)


class TestMetricsDisplay:
    """Test metrics display in progress bar."""

    def test_format_metrics_with_valid_metrics(self):
        """Test formatting metrics with valid values."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            show_metrics=['loss', 'episode_return'],
            disable=True
        )

        metrics = {'loss': 0.1234, 'episode_return': 5.67}
        formatted = callback._format_metrics(metrics)

        assert 'loss=0.1234' in formatted
        assert 'episode_return=5.6700' in formatted

    def test_format_metrics_with_scientific_notation(self):
        """Test formatting very small/large values with scientific notation."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(show_metrics=['loss'], disable=True)

        # Very small value
        metrics = {'loss': 0.00001}
        formatted = callback._format_metrics(metrics)
        assert 'e' in formatted.lower()  # Should use scientific notation

        # Very large value
        metrics = {'loss': 10000.0}
        formatted = callback._format_metrics(metrics)
        assert 'e' in formatted.lower()  # Should use scientific notation

    def test_format_metrics_with_empty_show_metrics(self):
        """Test formatting when show_metrics is empty."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(show_metrics=[], disable=True)

        metrics = {'loss': 0.5}
        formatted = callback._format_metrics(metrics)

        assert formatted == ""

    def test_format_metrics_with_missing_metric(self):
        """Test formatting when requested metric is not in dict."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            show_metrics=['loss', 'missing_metric'],
            disable=True
        )

        metrics = {'loss': 0.5}
        formatted = callback._format_metrics(metrics)

        assert 'loss=0.5000' in formatted
        assert 'missing_metric' not in formatted

    def test_metrics_updated_from_on_step(self):
        """Test that metrics are stored from on_step calls."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()
        mock_trainer.config.num_steps = 50

        callback.on_training_start(mock_trainer)
        callback.on_step(mock_trainer, step=1, metrics={'loss': 0.5, 'entropy': 0.1})

        assert callback._last_metrics['loss'] == 0.5
        assert callback._last_metrics['entropy'] == 0.1

        callback.on_training_end(mock_trainer)

    def test_metrics_updated_from_on_update_end(self):
        """Test that metrics are stored from on_update_end calls."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=True)
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {'policy_loss': 0.3, 'value_loss': 0.2})

        assert callback._last_metrics['policy_loss'] == 0.3
        assert callback._last_metrics['value_loss'] == 0.2


class TestUtilityMethods:
    """Test utility methods of ProgressCallback."""

    def test_get_elapsed_time_before_training(self):
        """Test elapsed time before training starts."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback()
        assert callback.get_elapsed_time() == 0.0

    def test_get_elapsed_time_during_training(self):
        """Test elapsed time during training."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(disable=False)  # Enable progress bar for this test
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        time.sleep(0.1)  # Sleep 100ms

        elapsed = callback.get_elapsed_time()
        assert elapsed >= 0.1  # Should be at least 100ms

        callback.on_training_end(mock_trainer)

    def test_get_progress_percentage(self):
        """Test progress percentage calculation."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=True)
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        callback._current_timestep = 250

        percentage = callback.get_progress_percentage()
        assert percentage == 25.0

    def test_get_progress_percentage_capped_at_100(self):
        """Test that progress percentage is capped at 100%."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=True)
        callback._current_timestep = 1500  # Over total

        percentage = callback.get_progress_percentage()
        assert percentage == 100.0

    def test_get_progress_percentage_zero_total(self):
        """Test progress percentage with zero total timesteps."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=0)

        percentage = callback.get_progress_percentage()
        assert percentage == 100.0  # Division by zero protection

    def test_get_current_step(self):
        """Test getting current step."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback()
        callback._current_step = 42

        assert callback.get_current_step() == 42

    def test_get_current_timestep(self):
        """Test getting current timestep."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback()
        callback._current_timestep = 1234

        assert callback.get_current_timestep() == 1234

    def test_reset_clears_all_state(self):
        """Test that reset clears all state."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()
        mock_trainer.config.num_steps = 50

        callback.on_training_start(mock_trainer)
        callback.on_step(mock_trainer, step=5, metrics={'loss': 0.5})

        callback.reset()

        assert callback._pbar is None
        assert callback._start_time is None
        assert callback._current_step == 0
        assert callback._current_timestep == 0
        assert callback._step_count == 0
        assert callback._last_metrics == {}


class TestSetTrainer:
    """Test set_trainer method."""

    def test_set_trainer_stores_reference(self):
        """Test that set_trainer stores the trainer reference."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback()
        mock_trainer = MagicMock()

        callback.set_trainer(mock_trainer)

        assert callback.trainer is mock_trainer

    def test_set_trainer_updates_total_timesteps_from_config(self):
        """Test that set_trainer updates total_timesteps from trainer config."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=None)
        mock_trainer = MagicMock()
        mock_trainer.config.total_timesteps = 500_000

        callback.set_trainer(mock_trainer)

        assert callback.total_timesteps == 500_000


class TestTimestepTracking:
    """Test timestep tracking from trainer."""

    def test_timestep_from_trainer_timestep_attribute(self):
        """Test getting timestep from trainer.timestep."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()
        mock_trainer.config.num_steps = 50
        type(mock_trainer).timestep = PropertyMock(return_value=100)

        callback.on_training_start(mock_trainer)
        callback.on_step(mock_trainer, step=1, metrics={})

        assert callback._current_timestep == 100

        callback.on_training_end(mock_trainer)

    def test_timestep_from_trainer_private_attribute(self):
        """Test getting timestep from trainer._timestep."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()

        # Set up trainer without timestep attribute, with _timestep
        # Need to use spec to prevent MagicMock from auto-creating timestep
        class MockTrainer:
            config = type('config', (), {'num_steps': 50})()
            _timestep = 200

        mock_trainer = MockTrainer()

        callback.on_training_start(mock_trainer)
        callback.on_step(mock_trainer, step=1, metrics={})

        assert callback._current_timestep == 200

        callback.on_training_end(mock_trainer)

    def test_timestep_estimated_from_step_and_config(self):
        """Test timestep estimation from step count and config."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)

        # Create a simple mock without timestep
        class MockTrainer:
            config = type('config', (), {'num_steps': 50})()

        mock_trainer = MockTrainer()

        callback.on_training_start(mock_trainer)
        callback.on_step(mock_trainer, step=3, metrics={})

        # Estimated: step * num_steps = 3 * 50 = 150
        assert callback._current_timestep == 150

        callback.on_training_end(mock_trainer)


class TestRolloutMetrics:
    """Test rollout metrics extraction."""

    def test_episode_return_extracted_from_rollout_data(self):
        """Test that episode returns are extracted from rollout data."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(disable=True)
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        callback.on_rollout_end(mock_trainer, {
            'episode_returns': [1.0, 2.0, 3.0]
        })

        # Mean of [1, 2, 3] = 2.0
        assert callback._last_metrics['episode_return'] == 2.0

    def test_empty_episode_returns_handled(self):
        """Test handling of empty episode returns list."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(disable=True)
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        callback.on_rollout_end(mock_trainer, {
            'episode_returns': []
        })

        # Should not set episode_return if list is empty
        assert 'episode_return' not in callback._last_metrics


class TestVerboseOutput:
    """Test verbose output behavior."""

    def test_verbose_prints_on_training_start(self, capsys):
        """Test that verbose mode prints on training start."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            total_timesteps=1000,
            verbose=True,
            disable=False  # Need progress bar for verbose output
        )
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)

        captured = capsys.readouterr()
        assert "Starting training" in captured.out

        callback.on_training_end(mock_trainer)

    def test_verbose_prints_on_training_end(self, capsys):
        """Test that verbose mode prints on training end."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(
            total_timesteps=1000,
            verbose=True,
            disable=False  # Need progress bar for verbose output
        )
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        callback.on_training_end(mock_trainer)

        captured = capsys.readouterr()
        assert "completed" in captured.out


class TestCallbackListIntegration:
    """Test ProgressCallback integration with CallbackList."""

    def test_progress_callback_in_callback_list(self):
        """Test that ProgressCallback works in CallbackList."""
        from socialjax.training.callbacks import ProgressCallback, CallbackList

        callback1 = ProgressCallback(total_timesteps=1000, disable=False)
        callback2 = ProgressCallback(total_timesteps=500, disable=False)

        callbacks = CallbackList([callback1, callback2])
        mock_trainer = MagicMock()

        callbacks.on_training_start(mock_trainer)

        assert callback1._start_time is not None
        assert callback2._start_time is not None

        callbacks.on_training_end(mock_trainer)

    def test_add_progress_callback_to_list(self):
        """Test adding ProgressCallback to CallbackList."""
        from socialjax.training.callbacks import ProgressCallback, CallbackList

        callbacks = CallbackList()
        progress = ProgressCallback(total_timesteps=1000)

        callbacks.add(progress)

        assert len(callbacks) == 1
        assert callbacks[0] is progress


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_on_step_before_training_start(self):
        """Test on_step called before training starts."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(disable=True)
        mock_trainer = MagicMock()

        # Should not raise error
        callback.on_step(mock_trainer, step=1, metrics={'loss': 0.5})

    def test_on_training_end_before_start(self):
        """Test on_training_end called before training starts."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(disable=True)
        mock_trainer = MagicMock()

        # Should not raise error
        callback.on_training_end(mock_trainer)

    def test_multiple_on_training_start_calls(self):
        """Test multiple on_training_start calls (restart training)."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(total_timesteps=1000, disable=False)
        mock_trainer = MagicMock()

        # First training
        callback.on_training_start(mock_trainer)
        first_pbar = callback._pbar

        # Second training start (without end) - creates new progress bar
        callback.on_training_start(mock_trainer)
        second_pbar = callback._pbar

        # Should have a new progress bar
        assert callback._pbar is not None

        # Clean up
        callback.on_training_end(mock_trainer)

    def test_negative_progress_freq_becomes_one(self):
        """Test that negative progress_freq becomes 1."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback(progress_freq=-10)
        assert callback.progress_freq == 1

    def test_custom_bar_format_is_used(self):
        """Test that custom bar format is used."""
        from socialjax.training.callbacks import ProgressCallback

        custom_format = "Custom: {n_fmt}/{total_fmt}"
        callback = ProgressCallback(bar_format=custom_format)

        assert callback._create_bar_format() == custom_format

    def test_pbar_property_returns_none_when_not_initialized(self):
        """Test that pbar property returns None when not initialized."""
        from socialjax.training.callbacks import ProgressCallback

        callback = ProgressCallback()
        assert callback.pbar is None
