"""Unit tests for CheckpointCallback.

Test criteria:
- CheckpointCallback saves at correct frequency
- Checkpoint files are created in correct location
- Checkpoint format is loadable
- Verbose logging works correctly
- Unit tests exist: test_save_frequency, test_checkpoint_location, test_checkpoint_format, test_verbose
"""

import os
import pickle
import pytest
import sys
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Set up path for imports
sys.path.insert(0, 'socialjax')


class TestCheckpointCallbackImport:
    """Test that CheckpointCallback can be imported from various locations."""

    def test_import_from_checkpoint_callback(self):
        """Test importing CheckpointCallback directly from checkpoint_callback module."""
        from socialjax.training.callbacks.checkpoint_callback import CheckpointCallback
        assert CheckpointCallback is not None

    def test_import_from_callbacks_init(self):
        """Test importing CheckpointCallback from callbacks __init__."""
        from socialjax.training.callbacks import CheckpointCallback
        assert CheckpointCallback is not None

    def test_import_from_training_init(self):
        """Test importing CheckpointCallback from training __init__."""
        from socialjax.training import CheckpointCallback
        assert CheckpointCallback is not None


class TestCheckpointCallbackInit:
    """Test CheckpointCallback initialization."""

    def test_default_initialization(self):
        """Test CheckpointCallback with default parameters."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback()
        assert callback.save_freq == 1000
        assert callback.save_path == "./checkpoints"
        assert callback.name_prefix == "model"
        assert callback.verbose is False

    def test_custom_initialization(self):
        """Test CheckpointCallback with custom parameters."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=500,
            save_path="/tmp/checkpoints",
            name_prefix="ippo_test",
            verbose=True
        )
        assert callback.save_freq == 500
        assert callback.save_path == "/tmp/checkpoints"
        assert callback.name_prefix == "ippo_test"
        assert callback.verbose is True

    def test_invalid_save_freq_raises_error(self):
        """Test that invalid save_freq raises ValueError."""
        from socialjax.training.callbacks import CheckpointCallback

        with pytest.raises(ValueError, match="save_freq must be a positive integer"):
            CheckpointCallback(save_freq=0)

        with pytest.raises(ValueError, match="save_freq must be a positive integer"):
            CheckpointCallback(save_freq=-1)

    def test_inherits_from_base_callback(self):
        """Test that CheckpointCallback inherits from BaseCallback."""
        from socialjax.training.callbacks import CheckpointCallback, BaseCallback

        callback = CheckpointCallback()
        assert isinstance(callback, BaseCallback)

    def test_has_required_hook_methods(self):
        """Test that CheckpointCallback has required hook methods."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback()

        assert hasattr(callback, 'on_training_start')
        assert hasattr(callback, 'on_training_end')
        assert hasattr(callback, 'on_step')
        assert hasattr(callback, 'on_rollout_start')
        assert hasattr(callback, 'on_rollout_end')
        assert hasattr(callback, 'on_update_start')
        assert hasattr(callback, 'on_update_end')
        assert hasattr(callback, 'set_trainer')


class TestSaveFrequency:
    """Test that checkpoints are saved at the correct frequency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_at_correct_frequency(self):
        """Test that checkpoints are saved every save_freq updates."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=5,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        # Create mock trainer with algorithm
        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 0

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        # Simulate 10 updates
        for i in range(1, 11):
            callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should have called save at updates 5 and 10
        assert mock_algorithm.save.call_count == 2

    def test_save_at_frequency_one(self):
        """Test that save_freq=1 saves on every update."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 0

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        # Simulate 3 updates
        for i in range(1, 4):
            callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should have called save 3 times
        assert mock_algorithm.save.call_count == 3

    def test_no_save_before_frequency(self):
        """Test that no checkpoint is saved before save_freq is reached."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=100,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 0

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        # Simulate 50 updates (less than save_freq)
        for i in range(1, 51):
            callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should not have called save
        assert mock_algorithm.save.call_count == 0

    def test_update_count_tracking(self):
        """Test that update count is tracked correctly."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=100,
            save_path=self.temp_dir
        )

        mock_trainer = MagicMock()
        mock_trainer.algorithm = MagicMock()
        mock_trainer.algorithm_state = MagicMock()

        # Simulate 25 updates
        for i in range(25):
            callback.on_update_end(mock_trainer, {})

        assert callback.get_update_count() == 25


class TestCheckpointLocation:
    """Test that checkpoints are saved in the correct location."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_directory_created(self):
        """Test that checkpoint directory is created on training start."""
        from socialjax.training.callbacks import CheckpointCallback

        new_dir = os.path.join(self.temp_dir, "new_checkpoints")
        callback = CheckpointCallback(
            save_freq=100,
            save_path=new_dir
        )

        mock_trainer = MagicMock()
        callback.on_training_start(mock_trainer)

        assert os.path.exists(new_dir)

    def test_checkpoint_file_location(self):
        """Test that checkpoint files are saved in correct location."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="ippo"
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 100

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        # First create directory
        callback.on_training_start(mock_trainer)

        # Trigger save
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Verify save was called with correct path
        called_path = mock_algorithm.save.call_args[0][1]
        assert called_path.startswith(self.temp_dir)
        assert "ippo" in called_path
        assert called_path.endswith(".pkl")

    def test_checkpoint_filename_format(self):
        """Test that checkpoint filenames have correct format."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="mappo_coins"
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 500

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        called_path = mock_algorithm.save.call_args[0][1]
        filename = os.path.basename(called_path)

        # Should be like "mappo_coins_500.pkl"
        assert filename == "mappo_coins_500.pkl"


class TestCheckpointFormat:
    """Test that checkpoints are saved in a loadable format."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_is_pickle_file(self):
        """Test that checkpoint is a valid pickle file."""
        from socialjax.training.callbacks import CheckpointCallback

        checkpoint_path = os.path.join(self.temp_dir, "test_1.pkl")

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        # Create mock with actual save functionality
        mock_trainer = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 1

        # Create a real save method
        def mock_save(state, path):
            save_dict = {
                "params": {"layer1": [1, 2, 3]},
                "update_step": state.update_step
            }
            with open(path, "wb") as f:
                pickle.dump(save_dict, f)

        mock_algorithm = MagicMock()
        mock_algorithm.save = mock_save

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Verify file is valid pickle
        assert os.path.exists(checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            loaded = pickle.load(f)
        assert "params" in loaded
        assert "update_step" in loaded
        assert loaded["update_step"] == 1

    def test_checkpoint_can_be_loaded(self):
        """Test that saved checkpoints can be loaded."""
        from socialjax.training.callbacks import CheckpointCallback

        checkpoint_path = os.path.join(self.temp_dir, "test_100.pkl")

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        mock_trainer = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 100

        # Create mock save that saves actual data
        def mock_save(state, path):
            save_dict = {
                "params": {"weight": [[1.0, 2.0], [3.0, 4.0]]},
                "optimizer_state": {"lr": 0.001},
                "update_step": state.update_step,
                "timestep": 10000
            }
            with open(path, "wb") as f:
                pickle.dump(save_dict, f)

        mock_algorithm = MagicMock()
        mock_algorithm.save = mock_save

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Load the checkpoint
        with open(checkpoint_path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded["update_step"] == 100
        assert loaded["timestep"] == 10000
        assert "params" in loaded
        assert "optimizer_state" in loaded

    def test_multiple_checkpoints_saved(self):
        """Test that multiple checkpoints can be saved during training."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=2,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        mock_trainer = MagicMock()
        mock_state = MagicMock()

        saved_files = []

        def mock_save(state, path):
            saved_files.append(path)
            save_dict = {"update_step": state.update_step}
            with open(path, "wb") as f:
                pickle.dump(save_dict, f)

        mock_algorithm = MagicMock()
        mock_algorithm.save = mock_save
        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)

        # Simulate 6 updates
        for i in range(1, 7):
            mock_state.update_step = i
            callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should have saved at updates 2, 4, 6
        assert len(saved_files) == 3
        assert "test_2.pkl" in saved_files[0]
        assert "test_4.pkl" in saved_files[1]
        assert "test_6.pkl" in saved_files[2]


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
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=100,
            save_path=self.temp_dir,
            verbose=True
        )

        mock_trainer = MagicMock()
        callback.on_training_start(mock_trainer)

        captured = capsys.readouterr()
        assert "CheckpointCallback" in captured.out
        assert self.temp_dir in captured.out
        assert "100" in captured.out

    def test_verbose_prints_on_save(self, capsys):
        """Test that verbose=True prints info when saving checkpoint."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="test",
            verbose=True
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 1

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.123456})

        captured = capsys.readouterr()
        assert "Saved checkpoint" in captured.out
        assert "update 1" in captured.out
        assert self.temp_dir in captured.out

    def test_verbose_shows_loss_value(self, capsys):
        """Test that verbose=True shows loss value in output."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            verbose=True
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 5

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.987654})

        captured = capsys.readouterr()
        assert "Loss: 0.987654" in captured.out

    def test_non_verbose_silent(self, capsys):
        """Test that verbose=False produces no output."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            verbose=False
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 1

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_warns_no_algorithm_state(self, capsys):
        """Test that verbose warns when no algorithm_state is found."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            verbose=True
        )

        mock_trainer = MagicMock()
        mock_trainer.algorithm_state = None
        mock_trainer.algorithm = MagicMock()

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        captured = capsys.readouterr()
        assert "Warning" in captured.out or "No algorithm_state" in captured.out


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
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir
        )

        mock_trainer = MagicMock()
        mock_trainer.algorithm_state = None
        mock_trainer.algorithm = MagicMock()

        # Should not raise error
        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

    def test_no_algorithm(self):
        """Test handling when trainer has no algorithm."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir
        )

        mock_trainer = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 1

        mock_trainer.algorithm_state = mock_state
        mock_trainer.algorithm = None

        # Should not raise error
        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

    def test_state_without_update_step(self):
        """Test handling when state doesn't have update_step attribute."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock(spec=[])  # State with no attributes

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        # Should use internal update_count instead
        mock_algorithm.save.assert_called_once()

    def test_reset_clears_counters(self):
        """Test that reset() clears internal counters."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=100,
            save_path=self.temp_dir
        )

        # Simulate some updates
        mock_trainer = MagicMock()
        mock_trainer.algorithm = MagicMock()
        mock_trainer.algorithm_state = MagicMock()

        for _ in range(10):
            callback.on_update_end(mock_trainer, {})

        assert callback.get_update_count() == 10

        # Reset
        callback.reset()
        assert callback.get_update_count() == 0
        assert callback.get_last_save_step() == 0

    def test_get_last_save_step(self):
        """Test get_last_save_step returns correct value."""
        from socialjax.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir
        )

        mock_trainer = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 50

        mock_trainer.algorithm = MagicMock()
        mock_trainer.algorithm_state = mock_state

        callback.on_training_start(mock_trainer)
        callback.on_update_end(mock_trainer, {"loss": 0.5})

        assert callback.get_last_save_step() == 50


class TestCallbackListIntegration:
    """Test CheckpointCallback integration with CallbackList."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_callback_in_callback_list(self):
        """Test that CheckpointCallback works in CallbackList."""
        from socialjax.training.callbacks import CheckpointCallback, CallbackList

        callback = CheckpointCallback(
            save_freq=1,
            save_path=self.temp_dir,
            name_prefix="test"
        )

        callback_list = CallbackList([callback])

        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_state = MagicMock()
        mock_state.update_step = 1

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.algorithm_state = mock_state

        callback_list.on_training_start(mock_trainer)
        callback_list.on_update_end(mock_trainer, {"loss": 0.5})

        # Verify save was called
        mock_algorithm.save.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
