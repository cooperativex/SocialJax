"""Progress callback for SocialJax training visualization.

This module provides the ProgressCallback class that displays a progress bar
during training using tqdm. It shows training progress, metrics, and elapsed time.

Example:
    >>> from socialjax.training.callbacks import ProgressCallback
    >>> callback = ProgressCallback(
    ...     total_timesteps=1_000_000,
    ...     progress_freq=1,  # Update every step
    ...     show_metrics=['loss', 'episode_return'],
    ... )
    >>> trainer = Trainer(algorithm='ippo', env='coin_game', callbacks=[callback])
    >>> trainer.train()
"""

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

if TYPE_CHECKING:
    from socialjax.core.base_trainer import BaseTrainer


class ProgressCallback:
    """Callback for displaying training progress with a progress bar.

    This callback displays a tqdm progress bar during training, showing:
    - Current step and total steps
    - Elapsed time
    - Current metrics (loss, episode return, etc.)
    - Training progress percentage

    The progress bar is updated at a configurable frequency to avoid
    excessive console output that can slow down training.

    Attributes:
        total_timesteps: Total number of timesteps for training.
        progress_freq: How often to update the progress bar (in steps).
        show_metrics: List of metric names to display in the progress bar.
        verbose: If True, print additional information.
        pbar: The tqdm progress bar instance.

    Example:
        >>> callback = ProgressCallback(
        ...     total_timesteps=1_000_000,
        ...     progress_freq=10,  # Update every 10 steps
        ... )
    """

    def __init__(
        self,
        total_timesteps: int = 1_000_000,
        progress_freq: int = 1,
        show_metrics: Optional[List[str]] = None,
        bar_format: Optional[str] = None,
        verbose: bool = False,
        disable: bool = False,
    ):
        """Initialize the ProgressCallback.

        Args:
            total_timesteps: Total number of timesteps for training.
                Used to calculate progress percentage.
            progress_freq: How often to update the progress bar (in steps).
                Default is 1 (update every step). Higher values reduce
                console output overhead.
            show_metrics: List of metric names to display in the progress bar.
                If None, defaults to ['loss', 'episode_return'].
                Pass empty list [] to show no metrics.
            bar_format: Custom format string for the progress bar.
                If None, uses a default format showing step count,
                percentage, elapsed time, and metrics.
            verbose: If True, print additional information about callback
                initialization and completion.
            disable: If True, disable the progress bar completely.
                Useful for running in non-interactive environments.
        """
        self._trainer: Optional["BaseTrainer"] = None
        self.total_timesteps = total_timesteps
        self.progress_freq = max(1, progress_freq)
        self.show_metrics = show_metrics if show_metrics is not None else ['loss', 'episode_return']
        self.bar_format = bar_format
        self.verbose = verbose
        self.disable = disable

        # Internal state
        self._pbar: Optional[tqdm] = None
        self._start_time: Optional[float] = None
        self._current_step: int = 0
        self._current_timestep: int = 0
        self._last_metrics: Dict[str, float] = {}
        self._step_count: int = 0

        # Check tqdm availability
        if not TQDM_AVAILABLE and not disable:
            if verbose:
                print("ProgressCallback: tqdm not available, progress bar disabled")
            self.disable = True

    @property
    def trainer(self) -> Optional["BaseTrainer"]:
        """Get the trainer reference."""
        return self._trainer

    def set_trainer(self, trainer: "BaseTrainer") -> None:
        """Set the trainer reference.

        Args:
            trainer: The trainer instance.
        """
        self._trainer = trainer

        # Try to get total_timesteps from trainer config if not set
        if self.total_timesteps is None and hasattr(trainer, 'config'):
            if hasattr(trainer.config, 'total_timesteps'):
                self.total_timesteps = trainer.config.total_timesteps

    def _create_bar_format(self) -> str:
        """Create the progress bar format string.

        Returns:
            A format string for tqdm progress bar.
        """
        if self.bar_format is not None:
            return self.bar_format

        # Default format: shows steps, percentage, elapsed time, and metrics
        # Format: "Step {step}/{total} [{percentage}%] Elapsed: {elapsed} {metrics}"
        return (
            "Step {n_fmt}/{total_fmt} [{percentage:3.0f}%] "
            "Elapsed: {elapsed} {postfix}"
        )

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display in progress bar.

        Args:
            metrics: Dictionary of metric names and values.

        Returns:
            Formatted string of metrics.
        """
        if not self.show_metrics or not metrics:
            return ""

        parts = []
        for name in self.show_metrics:
            if name in metrics:
                value = metrics[name]
                # Format based on value magnitude
                if abs(value) < 0.01 or abs(value) >= 1000:
                    parts.append(f"{name}={value:.2e}")
                else:
                    parts.append(f"{name}={value:.4f}")

        if parts:
            return "Metrics: " + ", ".join(parts)
        return ""

    def on_training_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of training.

        Creates and initializes the progress bar.

        Args:
            trainer: The trainer instance.
        """
        if self.disable:
            return

        self._start_time = time.time()
        self._current_step = 0
        self._current_timestep = 0
        self._step_count = 0
        self._last_metrics = {}

        # Create the progress bar
        self._pbar = tqdm(
            total=self.total_timesteps,
            bar_format=self._create_bar_format(),
            disable=self.disable,
            leave=True,  # Leave progress bar after completion
            dynamic_ncols=True,  # Adapt to terminal width
        )

        if self.verbose:
            print(f"ProgressCallback: Starting training with {self.total_timesteps:,} timesteps")

    def on_training_end(self, trainer: "BaseTrainer") -> None:
        """Called at the end of training.

        Closes the progress bar and prints completion message if verbose.

        Args:
            trainer: The trainer instance.
        """
        if self.disable or self._pbar is None:
            return

        # Update to final position
        if self._current_timestep < self.total_timesteps:
            self._pbar.update(self.total_timesteps - self._current_timestep)

        # Close the progress bar
        self._pbar.close()
        self._pbar = None

        if self.verbose:
            elapsed = time.time() - self._start_time if self._start_time else 0
            print(f"ProgressCallback: Training completed in {elapsed:.1f}s")

    def on_step(
        self,
        trainer: "BaseTrainer",
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called after each training step.

        Updates the progress bar with current step and metrics.

        Args:
            trainer: The trainer instance.
            step: Current step number (update count).
            metrics: Dictionary of metrics from this step.
        """
        if self.disable or self._pbar is None:
            return

        self._current_step = step
        self._step_count += 1

        # Get current timestep from trainer if available
        timestep_found = False
        if hasattr(trainer, 'timestep') and not callable(getattr(trainer, 'timestep', None)):
            try:
                ts = trainer.timestep
                if isinstance(ts, int):
                    self._current_timestep = ts
                    timestep_found = True
            except (TypeError, AttributeError):
                pass

        if not timestep_found and hasattr(trainer, '_timestep'):
            try:
                ts = trainer._timestep
                if isinstance(ts, int):
                    self._current_timestep = ts
                    timestep_found = True
            except (TypeError, AttributeError):
                pass

        if not timestep_found:
            # Estimate timestep based on step and num_steps per update
            num_steps = 128  # default
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'num_steps'):
                try:
                    ns = trainer.config.num_steps
                    if isinstance(ns, int):
                        num_steps = ns
                except (TypeError, AttributeError):
                    pass
            self._current_timestep = step * num_steps

        # Store metrics
        self._last_metrics.update(metrics)

        # Only update progress bar at specified frequency
        if self._step_count % self.progress_freq != 0:
            return

        # Update progress bar
        progress = min(self._current_timestep, self.total_timesteps)
        delta = progress - self._pbar.n

        if delta > 0:
            self._pbar.update(delta)

        # Update postfix with metrics
        metrics_str = self._format_metrics(self._last_metrics)
        if metrics_str:
            self._pbar.set_postfix_str(metrics_str)

    def on_rollout_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of a rollout.

        Args:
            trainer: The trainer instance.
        """
        pass  # No action needed for progress tracking

    def on_rollout_end(
        self,
        trainer: "BaseTrainer",
        rollout_data: Dict[str, any]
    ) -> None:
        """Called at the end of a rollout.

        Args:
            trainer: The trainer instance.
            rollout_data: Dictionary containing the collected experience data.
        """
        # Extract episode returns if available
        if 'episode_returns' in rollout_data:
            returns = rollout_data['episode_returns']
            if returns:
                mean_return = sum(returns) / len(returns)
                self._last_metrics['episode_return'] = mean_return

    def on_update_start(self, trainer: "BaseTrainer") -> None:
        """Called before parameter update.

        Args:
            trainer: The trainer instance.
        """
        pass  # No action needed for progress tracking

    def on_update_end(
        self,
        trainer: "BaseTrainer",
        update_metrics: Dict[str, float]
    ) -> None:
        """Called after parameter update.

        Updates metrics from the update step.

        Args:
            trainer: The trainer instance.
            update_metrics: Dictionary of metrics from the update step.
        """
        # Store update metrics for display
        self._last_metrics.update(update_metrics)

    # Utility methods

    def get_elapsed_time(self) -> float:
        """Get the elapsed training time in seconds.

        Returns:
            Elapsed time since training started, or 0 if not started.
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_progress_percentage(self) -> float:
        """Get the current progress as a percentage.

        Returns:
            Progress percentage (0-100).
        """
        if self.total_timesteps == 0:
            return 100.0
        return min(100.0, 100.0 * self._current_timestep / self.total_timesteps)

    def get_current_step(self) -> int:
        """Get the current step number.

        Returns:
            Current step (update count).
        """
        return self._current_step

    def get_current_timestep(self) -> int:
        """Get the current timestep.

        Returns:
            Current timestep (environment steps).
        """
        return self._current_timestep

    def reset(self) -> None:
        """Reset the callback state.

        Useful for restarting training with the same callback instance.
        """
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        self._start_time = None
        self._current_step = 0
        self._current_timestep = 0
        self._step_count = 0
        self._last_metrics = {}

    @property
    def pbar(self) -> Optional[tqdm]:
        """Get the tqdm progress bar instance.

        Returns:
            The tqdm progress bar, or None if not initialized.
        """
        return self._pbar
