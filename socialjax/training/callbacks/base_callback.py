"""Base callback class for SocialJax training hooks.

This module provides the BaseCallback class that all custom callbacks should
inherit from. It provides default (no-op) implementations for all callback hooks.

Callbacks are used to hook into the training process at various points:
- on_training_start: Called at the start of training
- on_training_end: Called at the end of training
- on_step: Called after each training step
- on_rollout_start: Called at the start of a rollout
- on_rollout_end: Called at the end of a rollout
- on_update_start: Called before parameter update
- on_update_end: Called after parameter update

Example:
    >>> class MyCallback(BaseCallback):
    ...     def on_training_start(self, trainer):
    ...         print("Training started!")
    ...
    ...     def on_step(self, trainer, step, metrics):
    ...         if step % 100 == 0:
    ...             print(f"Step {step}: loss={metrics.get('loss', 'N/A')}")
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from socialjax.core.base_trainer import BaseTrainer


class BaseCallback:
    """Base class for training callbacks.

    All callbacks should inherit from this class and override the methods
    they need. The default implementations do nothing (pass).

    Callbacks are called at specific points during training:
        - on_training_start: Once at the start of training
        - on_training_end: Once at the end of training
        - on_rollout_start: Before each rollout collection
        - on_rollout_end: After each rollout collection
        - on_update_start: Before each parameter update
        - on_update_end: After each parameter update
        - on_step: After each training step (includes update)

    Attributes:
        trainer: Reference to the trainer (set via set_trainer).
        verbose: If True, callbacks may print additional information.
    """

    def __init__(self, verbose: bool = False):
        """Initialize the callback.

        Args:
            verbose: If True, callbacks may print additional information.
        """
        self._trainer: Optional["BaseTrainer"] = None
        self.verbose = verbose

    @property
    def trainer(self) -> Optional["BaseTrainer"]:
        """Get the trainer reference.

        Returns:
            The trainer instance, or None if not set.
        """
        return self._trainer

    def set_trainer(self, trainer: "BaseTrainer") -> None:
        """Set the trainer reference.

        This method is called by the trainer when the callback is attached.
        Subclasses can override this to perform additional initialization
        that requires access to the trainer.

        Args:
            trainer: The trainer instance.
        """
        self._trainer = trainer

    def on_training_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of training.

        Override this method to perform initialization at the start of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_training_end(self, trainer: "BaseTrainer") -> None:
        """Called at the end of training.

        Override this method to perform cleanup or final logging at the end
        of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_step(
        self,
        trainer: "BaseTrainer",
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called after each training step.

        A training step typically includes collecting a rollout and performing
        a parameter update. Override this method to log or track progress.

        Args:
            trainer: The trainer instance.
            step: Current step number (update count).
            metrics: Dictionary of metrics from this step (e.g., loss values).
        """
        pass

    def on_rollout_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of a rollout.

        Override this method to perform actions before collecting experience
        from the environment.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_rollout_end(
        self,
        trainer: "BaseTrainer",
        rollout_data: Dict[str, Any]
    ) -> None:
        """Called at the end of a rollout.

        Override this method to process or log the collected rollout data.

        Args:
            trainer: The trainer instance.
            rollout_data: Dictionary containing the collected experience data.
                Typically includes observations, actions, rewards, etc.
        """
        pass

    def on_update_start(self, trainer: "BaseTrainer") -> None:
        """Called before parameter update.

        Override this method to perform actions before the gradient update.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_update_end(
        self,
        trainer: "BaseTrainer",
        update_metrics: Dict[str, float]
    ) -> None:
        """Called after parameter update.

        Override this method to log or track update metrics.

        Args:
            trainer: The trainer instance.
            update_metrics: Dictionary of metrics from the update step
                (e.g., loss values, gradient norms).
        """
        pass


class CallbackList:
    """Container for managing multiple callbacks.

    This class provides a convenient way to manage and invoke multiple
    callbacks at once. It ensures all callbacks are called in order.

    Example:
        >>> callbacks = CallbackList([
        ...     CheckpointCallback(save_freq=1000),
        ...     EvalCallback(eval_freq=500),
        ... ])
        >>> callbacks.on_training_start(trainer)
    """

    def __init__(self, callbacks: Optional[list] = None):
        """Initialize the callback list.

        Args:
            callbacks: Optional list of callback instances.
        """
        self.callbacks = callbacks or []

    def add(self, callback: BaseCallback) -> None:
        """Add a callback to the list.

        Args:
            callback: The callback instance to add.
        """
        self.callbacks.append(callback)

    def remove(self, callback: BaseCallback) -> bool:
        """Remove a callback from the list.

        Args:
            callback: The callback instance to remove.

        Returns:
            True if the callback was found and removed, False otherwise.
        """
        try:
            self.callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def set_trainer(self, trainer: "BaseTrainer") -> None:
        """Set the trainer reference for all callbacks.

        Args:
            trainer: The trainer instance.
        """
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_training_start(self, trainer: "BaseTrainer") -> None:
        """Invoke on_training_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(trainer)

    def on_training_end(self, trainer: "BaseTrainer") -> None:
        """Invoke on_training_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(trainer)

    def on_step(
        self,
        trainer: "BaseTrainer",
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Invoke on_step for all callbacks."""
        for callback in self.callbacks:
            callback.on_step(trainer, step, metrics)

    def on_rollout_start(self, trainer: "BaseTrainer") -> None:
        """Invoke on_rollout_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_rollout_start(trainer)

    def on_rollout_end(
        self,
        trainer: "BaseTrainer",
        rollout_data: Dict[str, Any]
    ) -> None:
        """Invoke on_rollout_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_rollout_end(trainer, rollout_data)

    def on_update_start(self, trainer: "BaseTrainer") -> None:
        """Invoke on_update_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_update_start(trainer)

    def on_update_end(
        self,
        trainer: "BaseTrainer",
        update_metrics: Dict[str, float]
    ) -> None:
        """Invoke on_update_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_update_end(trainer, update_metrics)

    def __len__(self) -> int:
        """Return the number of callbacks."""
        return len(self.callbacks)

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)

    def __getitem__(self, index: int) -> BaseCallback:
        """Get callback by index."""
        return self.callbacks[index]
