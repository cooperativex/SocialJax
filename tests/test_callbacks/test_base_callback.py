"""Unit tests for BaseCallback and CallbackList.

Test criteria:
- BaseCallback can be imported
- All hook methods exist with default (pass) implementation
- set_trainer correctly sets trainer reference
- Callback list can be managed
- Unit tests exist: test_callback_import, test_hook_methods, test_set_trainer, test_callback_list
"""

import pytest
import sys
from typing import Dict, Any
from unittest.mock import MagicMock

# Set up path for imports
sys.path.insert(0, 'socialjax')


class TestBaseCallbackImport:
    """Test that BaseCallback can be imported from various locations."""

    def test_import_from_base_callback(self):
        """Test importing BaseCallback directly from base_callback module."""
        from socialjax.training.callbacks.base_callback import BaseCallback
        assert BaseCallback is not None

    def test_import_from_callbacks_init(self):
        """Test importing BaseCallback from callbacks __init__."""
        from socialjax.training.callbacks import BaseCallback
        assert BaseCallback is not None

    def test_import_from_training_init(self):
        """Test importing BaseCallback from training __init__."""
        from socialjax.training import BaseCallback
        assert BaseCallback is not None

    def test_import_callback_list_from_callbacks(self):
        """Test importing CallbackList from callbacks module."""
        from socialjax.training.callbacks import CallbackList
        assert CallbackList is not None

    def test_import_callback_list_from_training(self):
        """Test importing CallbackList from training module."""
        from socialjax.training import CallbackList
        assert CallbackList is not None


class TestBaseCallbackHookMethods:
    """Test that all hook methods exist with default (pass) implementation."""

    def test_all_hook_methods_exist(self):
        """Test that all required hook methods exist."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()

        # Check all required methods exist
        assert hasattr(callback, 'on_training_start')
        assert hasattr(callback, 'on_training_end')
        assert hasattr(callback, 'on_step')
        assert hasattr(callback, 'on_rollout_start')
        assert hasattr(callback, 'on_rollout_end')
        assert hasattr(callback, 'on_update_start')
        assert hasattr(callback, 'on_update_end')
        assert hasattr(callback, 'set_trainer')

    def test_hook_methods_callable(self):
        """Test that all hook methods are callable."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()

        assert callable(callback.on_training_start)
        assert callable(callback.on_training_end)
        assert callable(callback.on_step)
        assert callable(callback.on_rollout_start)
        assert callable(callback.on_rollout_end)
        assert callable(callback.on_update_start)
        assert callable(callback.on_update_end)
        assert callable(callback.set_trainer)

    def test_hook_methods_return_none(self):
        """Test that default hook methods return None (pass implementation)."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()
        mock_trainer = MagicMock()

        # All default methods should return None
        assert callback.on_training_start(mock_trainer) is None
        assert callback.on_training_end(mock_trainer) is None
        assert callback.on_step(mock_trainer, 0, {}) is None
        assert callback.on_rollout_start(mock_trainer) is None
        assert callback.on_rollout_end(mock_trainer, {}) is None
        assert callback.on_update_start(mock_trainer) is None
        assert callback.on_update_end(mock_trainer, {}) is None

    def test_hook_methods_accept_correct_arguments(self):
        """Test that hook methods accept the correct arguments."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()
        mock_trainer = MagicMock()

        # These should not raise errors
        callback.on_training_start(mock_trainer)
        callback.on_training_end(mock_trainer)
        callback.on_step(mock_trainer, step=100, metrics={"loss": 0.5})
        callback.on_rollout_start(mock_trainer)
        callback.on_rollout_end(mock_trainer, rollout_data={"obs": [1, 2, 3]})
        callback.on_update_start(mock_trainer)
        callback.on_update_end(mock_trainer, update_metrics={"loss": 0.3, "grad_norm": 1.5})


class TestBaseCallbackSetTrainer:
    """Test set_trainer method correctly sets trainer reference."""

    def test_set_trainer_sets_internal_reference(self):
        """Test that set_trainer sets the _trainer attribute."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()
        mock_trainer = MagicMock()

        # Initially trainer should be None
        assert callback.trainer is None

        # After set_trainer, trainer should be set
        callback.set_trainer(mock_trainer)
        assert callback.trainer is mock_trainer

    def test_trainer_property_returns_correct_value(self):
        """Test that the trainer property returns the correct value."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()
        mock_trainer = MagicMock()

        # Property should return None initially
        assert callback.trainer is None

        # After setting, should return the trainer
        callback.set_trainer(mock_trainer)
        assert callback.trainer is mock_trainer

    def test_set_trainer_can_be_called_multiple_times(self):
        """Test that set_trainer can be called multiple times."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()
        mock_trainer1 = MagicMock()
        mock_trainer2 = MagicMock()

        callback.set_trainer(mock_trainer1)
        assert callback.trainer is mock_trainer1

        callback.set_trainer(mock_trainer2)
        assert callback.trainer is mock_trainer2


class TestBaseCallbackVerbose:
    """Test verbose flag functionality."""

    def test_verbose_default_false(self):
        """Test that verbose defaults to False."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback()
        assert callback.verbose is False

    def test_verbose_can_be_set(self):
        """Test that verbose can be set in constructor."""
        from socialjax.training.callbacks import BaseCallback

        callback = BaseCallback(verbose=True)
        assert callback.verbose is True

        callback = BaseCallback(verbose=False)
        assert callback.verbose is False


class TestCallbackList:
    """Test CallbackList management functionality."""

    def test_empty_callback_list(self):
        """Test creating an empty callback list."""
        from socialjax.training.callbacks import CallbackList

        callback_list = CallbackList()
        assert len(callback_list) == 0

    def test_callback_list_with_initial_callbacks(self):
        """Test creating a callback list with initial callbacks."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        cb1 = BaseCallback()
        cb2 = BaseCallback()
        callback_list = CallbackList([cb1, cb2])

        assert len(callback_list) == 2

    def test_add_callback(self):
        """Test adding a callback to the list."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        callback_list = CallbackList()
        cb = BaseCallback()

        callback_list.add(cb)
        assert len(callback_list) == 1
        assert callback_list[0] is cb

    def test_remove_callback(self):
        """Test removing a callback from the list."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        cb = BaseCallback()
        callback_list = CallbackList([cb])

        result = callback_list.remove(cb)
        assert result is True
        assert len(callback_list) == 0

    def test_remove_nonexistent_callback(self):
        """Test removing a callback that doesn't exist."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        cb1 = BaseCallback()
        cb2 = BaseCallback()
        callback_list = CallbackList([cb1])

        result = callback_list.remove(cb2)
        assert result is False
        assert len(callback_list) == 1

    def test_callback_list_iteration(self):
        """Test iterating over callback list."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        cb1 = BaseCallback()
        cb2 = BaseCallback()
        callback_list = CallbackList([cb1, cb2])

        callbacks = list(callback_list)
        assert len(callbacks) == 2
        assert callbacks[0] is cb1
        assert callbacks[1] is cb2

    def test_callback_list_getitem(self):
        """Test getting callback by index."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        cb1 = BaseCallback()
        cb2 = BaseCallback()
        callback_list = CallbackList([cb1, cb2])

        assert callback_list[0] is cb1
        assert callback_list[1] is cb2


class TestCallbackListHookInvocation:
    """Test that CallbackList correctly invokes all hooks."""

    def test_on_training_start_invokes_all(self):
        """Test that on_training_start invokes all callbacks."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        # Create callbacks that track if they were called
        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.training_start_called = False

            def on_training_start(self, trainer):
                self.training_start_called = True

        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        callback_list.on_training_start(mock_trainer)

        assert cb1.training_start_called
        assert cb2.training_start_called

    def test_on_training_end_invokes_all(self):
        """Test that on_training_end invokes all callbacks."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.training_end_called = False

            def on_training_end(self, trainer):
                self.training_end_called = True

        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        callback_list.on_training_end(mock_trainer)

        assert cb1.training_end_called
        assert cb2.training_end_called

    def test_on_step_invokes_all(self):
        """Test that on_step invokes all callbacks with correct arguments."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.last_step = None
                self.last_metrics = None

            def on_step(self, trainer, step, metrics):
                self.last_step = step
                self.last_metrics = metrics

        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        test_metrics = {"loss": 0.5, "accuracy": 0.9}
        callback_list.on_step(mock_trainer, step=100, metrics=test_metrics)

        assert cb1.last_step == 100
        assert cb1.last_metrics == test_metrics
        assert cb2.last_step == 100
        assert cb2.last_metrics == test_metrics

    def test_on_rollout_start_invokes_all(self):
        """Test that on_rollout_start invokes all callbacks."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.rollout_start_called = False

            def on_rollout_start(self, trainer):
                self.rollout_start_called = True

        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        callback_list.on_rollout_start(mock_trainer)

        assert cb1.rollout_start_called
        assert cb2.rollout_start_called

    def test_on_rollout_end_invokes_all(self):
        """Test that on_rollout_end invokes all callbacks with data."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.rollout_data = None

            def on_rollout_end(self, trainer, rollout_data):
                self.rollout_data = rollout_data

        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        test_data = {"observations": [1, 2, 3], "rewards": [0.1, 0.2, 0.3]}
        callback_list.on_rollout_end(mock_trainer, test_data)

        assert cb1.rollout_data == test_data
        assert cb2.rollout_data == test_data

    def test_on_update_start_invokes_all(self):
        """Test that on_update_start invokes all callbacks."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.update_start_called = False

            def on_update_start(self, trainer):
                self.update_start_called = True

        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        callback_list.on_update_start(mock_trainer)

        assert cb1.update_start_called
        assert cb2.update_start_called

    def test_on_update_end_invokes_all(self):
        """Test that on_update_end invokes all callbacks with metrics."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.update_metrics = None

            def on_update_end(self, trainer, update_metrics):
                self.update_metrics = update_metrics

        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        test_metrics = {"loss": 0.3, "grad_norm": 1.5}
        callback_list.on_update_end(mock_trainer, test_metrics)

        assert cb1.update_metrics == test_metrics
        assert cb2.update_metrics == test_metrics


class TestCallbackListSetTrainer:
    """Test CallbackList.set_trainer functionality."""

    def test_set_trainer_calls_all_callbacks(self):
        """Test that set_trainer propagates to all callbacks."""
        from socialjax.training.callbacks import BaseCallback, CallbackList

        cb1 = BaseCallback()
        cb2 = BaseCallback()
        callback_list = CallbackList([cb1, cb2])
        mock_trainer = MagicMock()

        callback_list.set_trainer(mock_trainer)

        assert cb1.trainer is mock_trainer
        assert cb2.trainer is mock_trainer


class TestCustomCallback:
    """Test creating custom callbacks by subclassing BaseCallback."""

    def test_custom_callback_overrides(self):
        """Test that custom callbacks can override methods."""
        from socialjax.training.callbacks import BaseCallback

        class CustomCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.events = []

            def on_training_start(self, trainer):
                self.events.append("training_start")

            def on_training_end(self, trainer):
                self.events.append("training_end")

            def on_step(self, trainer, step, metrics):
                self.events.append(f"step_{step}")

        callback = CustomCallback()
        mock_trainer = MagicMock()

        callback.on_training_start(mock_trainer)
        callback.on_step(mock_trainer, 100, {})
        callback.on_training_end(mock_trainer)

        assert "training_start" in callback.events
        assert "step_100" in callback.events
        assert "training_end" in callback.events

    def test_custom_callback_inherits_default_methods(self):
        """Test that custom callback inherits default (pass) methods."""
        from socialjax.training.callbacks import BaseCallback

        class CustomCallback(BaseCallback):
            def on_training_start(self, trainer):
                pass  # Only override one method

        callback = CustomCallback()

        # All other methods should still exist and work
        assert hasattr(callback, 'on_training_end')
        assert hasattr(callback, 'on_step')
        assert hasattr(callback, 'on_rollout_start')
        assert hasattr(callback, 'on_rollout_end')
        assert hasattr(callback, 'on_update_start')
        assert hasattr(callback, 'on_update_end')

        # These should not raise errors
        mock_trainer = MagicMock()
        callback.on_training_end(mock_trainer)
        callback.on_step(mock_trainer, 0, {})
        callback.on_rollout_start(mock_trainer)
        callback.on_rollout_end(mock_trainer, {})
        callback.on_update_start(mock_trainer)
        callback.on_update_end(mock_trainer, {})


class TestCallbackProtocolCompliance:
    """Test that BaseCallback complies with the Callback Protocol."""

    def test_matches_callback_protocol(self):
        """Test that BaseCallback implements the Callback protocol."""
        from socialjax.training.callbacks import BaseCallback
        from socialjax.core.base_trainer import Callback

        # BaseCallback should be usable as a Callback
        callback = BaseCallback()

        # Verify all protocol methods are present
        assert hasattr(callback, 'on_training_start')
        assert hasattr(callback, 'on_training_end')
        assert hasattr(callback, 'on_step')
        assert hasattr(callback, 'on_rollout_start')
        assert hasattr(callback, 'on_rollout_end')
        assert hasattr(callback, 'on_update_start')
        assert hasattr(callback, 'on_update_end')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
