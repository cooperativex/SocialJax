"""Checkpoint callback for saving model states during training.

This module provides the CheckpointCallback class that periodically saves
model checkpoints during training at specified intervals.

Example:
    >>> from socialjax.training.callbacks import CheckpointCallback
    >>> callback = CheckpointCallback(
    ...     save_freq=1000,
    ...     save_path="./checkpoints",
    ...     name_prefix="ippo",
    ...     verbose=True
    ... )
    >>> # Add to trainer's callback list
"""

import os
from typing import TYPE_CHECKING, Optional

from socialjax.training.callbacks.base_callback import BaseCallback

if TYPE_CHECKING:
    from socialjax.core.base_trainer import BaseTrainer


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints during training.

    This callback saves the algorithm state to disk at regular intervals
    during training. Checkpoints can be used to resume training or for
    evaluation.

    Attributes:
        save_freq: Number of updates between checkpoint saves.
        save_path: Directory where checkpoints will be saved.
        name_prefix: Prefix for checkpoint filenames.
        verbose: If True, print information when saving checkpoints.
        update_count: Internal counter tracking number of updates.

    Example:
        >>> callback = CheckpointCallback(
        ...     save_freq=1000,
        ...     save_path="./checkpoints/ippo",
        ...     name_prefix="ippo_coin_game",
        ...     verbose=True
        ... )
        >>> # This will create files like:
        >>> # ./checkpoints/ippo/ippo_coin_game_1000.pkl
        >>> # ./checkpoints/ippo/ippo_coin_game_2000.pkl
    """

    def __init__(
        self,
        save_freq: int = 1000,
        save_path: str = "./checkpoints",
        name_prefix: str = "model",
        verbose: bool = False
    ):
        """Initialize the checkpoint callback.

        Args:
            save_freq: Number of updates between checkpoint saves.
                A checkpoint will be saved every `save_freq` updates.
            save_path: Directory path where checkpoints will be saved.
                The directory will be created if it doesn't exist.
            name_prefix: Prefix for checkpoint filenames.
                Files will be named: {name_prefix}_{update_step}.pkl
            verbose: If True, print information when saving checkpoints.

        Raises:
            ValueError: If save_freq is not a positive integer.
        """
        super().__init__(verbose=verbose)

        if save_freq <= 0:
            raise ValueError(f"save_freq must be a positive integer, got {save_freq}")

        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._update_count = 0
        self._last_save_step = 0

    def on_training_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of training.

        Creates the checkpoint directory if it doesn't exist.

        Args:
            trainer: The trainer instance.
        """
        # Create checkpoint directory
        os.makedirs(self.save_path, exist_ok=True)

        if self.verbose:
            print(f"CheckpointCallback: Saving checkpoints to {self.save_path}")
            print(f"CheckpointCallback: Save frequency = {self.save_freq} updates")

    def on_update_end(
        self,
        trainer: "BaseTrainer",
        update_metrics: dict
    ) -> None:
        """Called after each parameter update.

        Saves a checkpoint if the update count is a multiple of save_freq.

        Args:
            trainer: The trainer instance.
            update_metrics: Dictionary of metrics from the update step.
        """
        self._update_count += 1

        # Check if we should save a checkpoint
        if self._update_count % self.save_freq == 0:
            self._save_checkpoint(trainer, update_metrics)

    def _save_checkpoint(
        self,
        trainer: "BaseTrainer",
        update_metrics: dict
    ) -> None:
        """Save a checkpoint.

        Args:
            trainer: The trainer instance.
            update_metrics: Dictionary of metrics from the update step.
        """
        # Get the algorithm state from the trainer
        if hasattr(trainer, 'algorithm_state') and trainer.algorithm_state is not None:
            state = trainer.algorithm_state

            # Get update step from state or use update_count
            update_step = getattr(state, 'update_step', self._update_count)
            if update_step == 0:
                update_step = self._update_count

            # Generate checkpoint filename
            checkpoint_name = f"{self.name_prefix}_{update_step}.pkl"
            checkpoint_path = os.path.join(self.save_path, checkpoint_name)

            # Save checkpoint using algorithm's save method
            if hasattr(trainer, 'algorithm') and trainer.algorithm is not None:
                trainer.algorithm.save(state, checkpoint_path)
                self._last_save_step = update_step

                if self.verbose:
                    loss_val = update_metrics.get('loss', 'N/A')
                    print(f"CheckpointCallback: Saved checkpoint at update {update_step}")
                    print(f"  Path: {checkpoint_path}")
                    if loss_val != 'N/A':
                        print(f"  Loss: {loss_val:.6f}")
            elif self.verbose:
                print("CheckpointCallback: Warning - No algorithm found in trainer")
        elif self.verbose:
            print("CheckpointCallback: Warning - No algorithm_state found in trainer")

    def get_last_save_step(self) -> int:
        """Get the update step of the last saved checkpoint.

        Returns:
            The update step of the last saved checkpoint, or 0 if none saved.
        """
        return self._last_save_step

    def get_update_count(self) -> int:
        """Get the current update count.

        Returns:
            The number of updates that have occurred.
        """
        return self._update_count

    def reset(self) -> None:
        """Reset the callback state.

        This can be called when restarting training to reset counters.
        """
        self._update_count = 0
        self._last_save_step = 0
