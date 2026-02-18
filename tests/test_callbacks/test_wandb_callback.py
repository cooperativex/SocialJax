"""Unit tests for WandbCallback.

Test criteria:
- WandbCallback initializes wandb correctly
- Metrics are logged to wandb dashboard
- Config is saved to wandb
- wandb.finish() is called on training end
- Unit tests exist: test_wandb_init, test_metric_logging, test_config_save, test_wandb_finish
"""

import pytest
import sys
from unittest.mock import MagicMock, patch, call

# Set up path for imports
sys.path.insert(0, 'socialjax')


class TestWandbCallbackImport:
    """Test that WandbCallback can be imported from various locations."""

    def test_import_from_wandb_callback(self):
        """Test importing WandbCallback directly from wandb_callback module."""
        from socialjax.training.callbacks.wandb_callback import WandbCallback
        assert WandbCallback is not None

    def test_import_from_callbacks_init(self):
        """Test importing WandbCallback from callbacks __init__."""
        from socialjax.training.callbacks import WandbCallback
        assert WandbCallback is not None

    def test_import_from_training_init(self):
        """Test importing WandbCallback from training __init__."""
        from socialjax.training import WandbCallback
        assert WandbCallback is not None


class TestWandbCallbackInit:
    """Test WandbCallback initialization."""

    def test_default_initialization(self):
        """Test WandbCallback with default parameters."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        assert callback.project == "socialjax"
        assert callback.name is None
        assert callback.config == {}
        assert callback.log_freq == 100
        assert callback.verbose is False
        assert callback._initialized is False

    def test_custom_initialization(self):
        """Test WandbCallback with custom parameters."""
        from socialjax.training.callbacks import WandbCallback

        config = {"lr": 0.001, "gamma": 0.99}
        callback = WandbCallback(
            project="my_project",
            name="run_001",
            config=config,
            log_freq=50,
            verbose=True
        )
        assert callback.project == "my_project"
        assert callback.name == "run_001"
        assert callback.config == config
        assert callback.log_freq == 50
        assert callback.verbose is True

    def test_invalid_log_freq_raises_error(self):
        """Test that invalid log_freq raises ValueError."""
        from socialjax.training.callbacks import WandbCallback

        with pytest.raises(ValueError, match="log_freq must be a positive integer"):
            WandbCallback(log_freq=0)

        with pytest.raises(ValueError, match="log_freq must be a positive integer"):
            WandbCallback(log_freq=-1)

    def test_inherits_from_base_callback(self):
        """Test that WandbCallback inherits from BaseCallback."""
        from socialjax.training.callbacks import WandbCallback, BaseCallback

        callback = WandbCallback()
        assert isinstance(callback, BaseCallback)

    def test_has_required_hook_methods(self):
        """Test that WandbCallback has required hook methods."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()

        assert hasattr(callback, 'on_training_start')
        assert hasattr(callback, 'on_training_end')
        assert hasattr(callback, 'on_step')
        assert hasattr(callback, 'on_rollout_start')
        assert hasattr(callback, 'on_rollout_end')
        assert hasattr(callback, 'on_update_start')
        assert hasattr(callback, 'on_update_end')
        assert hasattr(callback, 'set_trainer')


class TestWandbInit:
    """Test wandb initialization."""

    def test_wandb_init_called_on_training_start(self):
        """Test that wandb.init is called with correct parameters on training start."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(
            project="test_project",
            name="test_run",
            config={"lr": 0.001}
        )

        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test_run"
            mock_run.url = "https://wandb.ai/test_project/test_run"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            mock_wandb.init.assert_called_once()
            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs['project'] == "test_project"
            assert call_kwargs['name'] == "test_run"
            assert call_kwargs['config']["lr"] == 0.001

    def test_wandb_init_merges_trainer_config(self):
        """Test that wandb.init merges callback config with trainer config."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(
            project="test_project",
            config={"lr": 0.001}
        )

        mock_trainer = MagicMock()
        mock_trainer.config = {"env": "coin_game", "num_agents": 5}

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test_run"
            mock_run.url = "https://wandb.ai/test_project/test_run"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            call_kwargs = mock_wandb.init.call_args[1]
            assert "lr" in call_kwargs['config']
            assert "env" in call_kwargs['config']
            assert "num_agents" in call_kwargs['config']

    def test_wandb_init_sets_initialized_flag(self):
        """Test that successful init sets _initialized to True."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            assert callback._initialized is True
            assert callback.is_initialized() is True

    def test_wandb_init_handles_import_error(self):
        """Test that import error is handled gracefully."""
        from socialjax.training.callbacks import WandbCallback
        import socialjax.training.callbacks.wandb_callback as wandb_module

        callback = WandbCallback(verbose=True)
        mock_trainer = MagicMock()

        # Mock WANDB_AVAILABLE to False to simulate wandb not being installed
        original_value = wandb_module.WANDB_AVAILABLE
        wandb_module.WANDB_AVAILABLE = False

        try:
            # Should not raise
            callback.on_training_start(mock_trainer)
            assert callback._initialized is False
        finally:
            # Restore original value
            wandb_module.WANDB_AVAILABLE = original_value

    def test_wandb_init_handles_exception(self):
        """Test that general exceptions are handled gracefully."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(verbose=True)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_wandb.init.side_effect = Exception("Connection error")
            # Should not raise
            callback.on_training_start(mock_trainer)
            assert callback._initialized is False


class TestMetricLogging:
    """Test metric logging functionality."""

    def test_log_metrics_at_correct_frequency(self):
        """Test that metrics are logged at the specified frequency."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(log_freq=10)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            # Log at steps that are multiples of log_freq
            for step in [10, 20, 30, 40, 50]:
                callback.on_step(mock_trainer, step, {"loss": 0.5, "reward": 1.0})

            # Should have logged 5 times
            assert mock_wandb.log.call_count == 5

    def test_no_log_before_frequency(self):
        """Test that metrics are not logged before reaching log_freq."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(log_freq=100)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            # Log at steps that are not multiples of log_freq
            for step in [1, 5, 10, 50, 99]:
                callback.on_step(mock_trainer, step, {"loss": 0.5})

            # Should not have logged
            assert mock_wandb.log.call_count == 0

    def test_log_includes_step_in_metrics(self):
        """Test that logged metrics include the step number."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(log_freq=1)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.on_step(mock_trainer, 42, {"loss": 0.5})

            call_args = mock_wandb.log.call_args
            logged_data = call_args[0][0]
            assert logged_data["step"] == 42
            assert logged_data["loss"] == 0.5

    def test_log_all_provided_metrics(self):
        """Test that all provided metrics are logged."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(log_freq=1)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.on_step(mock_trainer, 1, {
                "loss": 0.5,
                "actor_loss": 0.3,
                "critic_loss": 0.2,
                "entropy": 0.1
            })

            call_args = mock_wandb.log.call_args
            logged_data = call_args[0][0]
            assert "loss" in logged_data
            assert "actor_loss" in logged_data
            assert "critic_loss" in logged_data
            assert "entropy" in logged_data

    def test_no_log_if_not_initialized(self):
        """Test that metrics are not logged if wandb is not initialized."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(log_freq=1)
        callback._initialized = False  # Explicitly not initialized
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            callback.on_step(mock_trainer, 100, {"loss": 0.5})
            mock_wandb.log.assert_not_called()

    def test_update_metrics_prefixed(self):
        """Test that update metrics are logged with prefix."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.on_update_end(mock_trainer, {"grad_norm": 0.5})

            call_args = mock_wandb.log.call_args
            logged_data = call_args[0][0]
            assert "update/grad_norm" in logged_data


class TestConfigSave:
    """Test config saving functionality."""

    def test_config_passed_to_wandb_init(self):
        """Test that config is passed to wandb.init."""
        from socialjax.training.callbacks import WandbCallback

        config = {"algorithm": "ippo", "env": "coin_game", "lr": 0.001}
        callback = WandbCallback(config=config)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            call_kwargs = mock_wandb.init.call_args[1]
            # Check that all config keys are present (avoid mock attribute issues)
            assert call_kwargs['config']["algorithm"] == "ippo"
            assert call_kwargs['config']["env"] == "coin_game"
            assert call_kwargs['config']["lr"] == 0.001

    def test_trainer_dict_config_merged(self):
        """Test that trainer config (dict) is merged with callback config."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(config={"lr": 0.001})
        mock_trainer = MagicMock()
        mock_trainer.config = {"env": "coin_game"}

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs['config']["lr"] == 0.001
            assert call_kwargs['config']["env"] == "coin_game"

    def test_trainer_dataclass_config_merged(self):
        """Test that trainer config (dataclass/object) is merged with callback config."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(config={"lr": 0.001})
        mock_trainer = MagicMock()

        # Create a mock config object with __dict__
        class MockConfig:
            def __init__(self):
                self.env = "coin_game"
                self.num_agents = 5

        mock_trainer.config = MockConfig()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs['config']["lr"] == 0.001
            assert call_kwargs['config']["env"] == "coin_game"
            assert call_kwargs['config']["num_agents"] == 5


class TestWandbFinish:
    """Test wandb.finish() functionality."""

    def test_wandb_finish_called_on_training_end(self):
        """Test that wandb.finish() is called on training end."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.on_training_end(mock_trainer)

            mock_wandb.finish.assert_called_once()

    def test_finish_resets_initialized_flag(self):
        """Test that finish resets _initialized to False."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            assert callback._initialized is True

            callback.on_training_end(mock_trainer)
            assert callback._initialized is False
            assert callback.is_initialized() is False

    def test_finish_not_called_if_not_initialized(self):
        """Test that finish is not called if wandb was not initialized."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        callback._initialized = False
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            callback.on_training_end(mock_trainer)
            mock_wandb.finish.assert_not_called()

    def test_finish_handles_exception(self):
        """Test that exceptions during finish are handled gracefully."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(verbose=True)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run
            mock_wandb.finish.side_effect = Exception("Network error")

            callback.on_training_start(mock_trainer)
            # Should not raise
            callback.on_training_end(mock_trainer)


class TestVerboseLogging:
    """Test verbose logging functionality."""

    def test_verbose_prints_on_training_start(self, capsys):
        """Test that verbose=True prints info on training start."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(verbose=True, project="test_project")
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test_run"
            mock_run.url = "https://wandb.ai/test_project/test_run"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            captured = capsys.readouterr()
            assert "WandbCallback" in captured.out
            assert "test_project" in captured.out

    def test_verbose_prints_on_training_end(self, capsys):
        """Test that verbose=True prints info on training end."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(verbose=True)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.on_step(mock_trainer, 100, {"loss": 0.5})  # Set step count
            callback.on_training_end(mock_trainer)

            captured = capsys.readouterr()
            assert "Finishing wandb run" in captured.out

    def test_non_verbose_silent(self, capsys):
        """Test that verbose=False produces no output."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(verbose=False)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.on_training_end(mock_trainer)

            captured = capsys.readouterr()
            assert captured.out == ""


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_run_url(self):
        """Test get_run_url returns correct URL."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test_project/test_run"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            assert callback.get_run_url() == "https://wandb.ai/test_project/test_run"

    def test_get_run_url_none_if_not_initialized(self):
        """Test get_run_url returns None if not initialized."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        assert callback.get_run_url() is None

    def test_get_run_id(self):
        """Test get_run_id returns correct ID."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_run.id = "abc123xyz"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            assert callback.get_run_id() == "abc123xyz"

    def test_get_run_id_none_if_not_initialized(self):
        """Test get_run_id returns None if not initialized."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        assert callback.get_run_id() is None

    def test_log_custom(self):
        """Test log_custom method for logging additional metrics."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback()
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.log_custom({"custom_metric": 42}, step=100)

            call_args = mock_wandb.log.call_args
            logged_data = call_args[0][0]
            assert logged_data["custom_metric"] == 42

    def test_log_custom_uses_step_count_if_not_specified(self):
        """Test log_custom uses current step count if step not provided."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(log_freq=1)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)
            callback.on_step(mock_trainer, 50, {"loss": 0.5})  # Set step count
            callback.log_custom({"custom_metric": 42})

            call_args = mock_wandb.log.call_args
            step_arg = call_args[1]['step']
            assert step_arg == 50


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_trainer_config(self):
        """Test handling when trainer has no config."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(config={"lr": 0.001})
        mock_trainer = MagicMock()
        del mock_trainer.config  # No config attribute

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            # Should not raise
            callback.on_training_start(mock_trainer)

            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs['config'] == {"lr": 0.001}

    def test_wandb_kwargs_passed_to_init(self):
        """Test that extra kwargs are passed to wandb.init."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(
            project="test",
            tags=["experiment", "ippo"],
            notes="Test run",
            entity="my_team"
        )
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback.on_training_start(mock_trainer)

            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs['tags'] == ["experiment", "ippo"]
            assert call_kwargs['notes'] == "Test run"
            assert call_kwargs['entity'] == "my_team"

    def test_log_handles_exception(self):
        """Test that logging exceptions are handled gracefully."""
        from socialjax.training.callbacks import WandbCallback

        callback = WandbCallback(verbose=True, log_freq=1)
        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run
            mock_wandb.log.side_effect = Exception("Network error")

            callback.on_training_start(mock_trainer)
            # Should not raise
            callback.on_step(mock_trainer, 1, {"loss": 0.5})


class TestCallbackListIntegration:
    """Test WandbCallback integration with CallbackList."""

    def test_wandb_callback_in_callback_list(self):
        """Test that WandbCallback works in CallbackList."""
        from socialjax.training.callbacks import WandbCallback, CallbackList

        callback = WandbCallback(
            project="test",
            log_freq=1
        )

        callback_list = CallbackList([callback])

        mock_trainer = MagicMock()

        with patch('socialjax.training.callbacks.wandb_callback.wandb') as mock_wandb:
            mock_run = MagicMock()
            mock_run.name = "test"
            mock_run.url = "https://wandb.ai/test"
            mock_wandb.run = mock_run
            mock_wandb.init.return_value = mock_run

            callback_list.on_training_start(mock_trainer)
            callback_list.on_step(mock_trainer, 1, {"loss": 0.5})
            callback_list.on_training_end(mock_trainer)

            # Verify init, log, and finish were called
            mock_wandb.init.assert_called_once()
            mock_wandb.log.assert_called_once()
            mock_wandb.finish.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
