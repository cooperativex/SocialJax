"""Wandb callback for logging training metrics to Weights & Biases.

This module provides the WandbCallback class that logs training metrics
to Weights & Biases (wandb) for experiment tracking and visualization.

Example:
    >>> from socialjax.training.callbacks import WandbCallback
    >>> callback = WandbCallback(
    ...     project="socialjax",
    ...     name="ippo_coin_game",
    ...     config={"lr": 0.001, "gamma": 0.99},
    ...     log_freq=100
    ... )
    >>> # Add to trainer's callback list
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from socialjax.training.callbacks.base_callback import BaseCallback

if TYPE_CHECKING:
    from socialjax.core.base_trainer import BaseTrainer

# Try to import wandb at module level for easier testing and usage
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


class WandbCallback(BaseCallback):
    """Callback for logging training metrics to Weights & Biases.

    This callback initializes a wandb run at the start of training, logs
    metrics periodically during training, and finishes the run at the end.

    Attributes:
        project: The wandb project name.
        name: The wandb run name.
        config: Configuration dictionary to log to wandb.
        log_freq: Number of steps between logging.
        verbose: If True, print information about wandb operations.
        _initialized: Whether wandb has been initialized.

    Example:
        >>> callback = WandbCallback(
        ...     project="socialjax",
        ...     name="ippo_coin_game_exp1",
        ...     config={"algorithm": "ippo", "env": "coin_game", "lr": 0.001},
        ...     log_freq=100,
        ...     verbose=True
        ... )
    """

    def __init__(
        self,
        project: str = "socialjax",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_freq: int = 100,
        verbose: bool = False,
        **kwargs
    ):
        """Initialize the wandb callback.

        Args:
            project: The wandb project name. Default is "socialjax".
            name: The wandb run name. If None, wandb generates a random name.
            config: Configuration dictionary to log to wandb. This can include
                hyperparameters, environment settings, etc.
            log_freq: Number of steps between logging. Metrics will be logged
                every `log_freq` steps to avoid overwhelming the wandb backend.
            verbose: If True, print information about wandb operations.
            **kwargs: Additional keyword arguments passed to wandb.init().

        Raises:
            ValueError: If log_freq is not a positive integer.
        """
        super().__init__(verbose=verbose)

        if log_freq <= 0:
            raise ValueError(f"log_freq must be a positive integer, got {log_freq}")

        self.project = project
        self.name = name
        self.config = config or {}
        self.log_freq = log_freq
        self._wandb_kwargs = kwargs
        self._initialized = False
        self._step_count = 0

    def on_training_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of training.

        Initializes the wandb run with the specified project, name, and config.

        Args:
            trainer: The trainer instance.
        """
        if not WANDB_AVAILABLE:
            if self.verbose:
                print("WandbCallback: Warning - wandb not installed. Skipping logging.")
            self._initialized = False
            return

        try:
            # Build config from callback config and trainer config if available
            full_config = dict(self.config)

            # Try to get additional config from trainer
            if hasattr(trainer, 'config') and trainer.config is not None:
                if isinstance(trainer.config, dict):
                    full_config.update(trainer.config)
                elif hasattr(trainer.config, '__dict__'):
                    full_config.update(vars(trainer.config))

            # Initialize wandb
            wandb.init(
                project=self.project,
                name=self.name,
                config=full_config,
                **self._wandb_kwargs
            )
            self._initialized = True

            if self.verbose:
                print(f"WandbCallback: Initialized wandb run")
                print(f"  Project: {self.project}")
                print(f"  Name: {self.name or wandb.run.name}")
                print(f"  URL: {wandb.run.url}")
                print(f"  Log frequency: {self.log_freq} steps")

        except Exception as e:
            if self.verbose:
                print(f"WandbCallback: Warning - Failed to initialize wandb: {e}")
            self._initialized = False

    def on_step(
        self,
        trainer: "BaseTrainer",
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called after each training step.

        Logs metrics to wandb at the specified frequency.

        Args:
            trainer: The trainer instance.
            step: Current step number (update count).
            metrics: Dictionary of metrics from this step.
        """
        if not self._initialized:
            return

        self._step_count = step

        # Only log at specified frequency
        if step % self.log_freq != 0:
            return

        try:
            # Add step to metrics
            log_data = {"step": step}

            # Add all provided metrics
            log_data.update(metrics)

            # Log to wandb
            wandb.log(log_data, step=step)

            if self.verbose and step % (self.log_freq * 10) == 0:
                print(f"WandbCallback: Logged metrics at step {step}")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.6f}")

        except Exception as e:
            if self.verbose:
                print(f"WandbCallback: Warning - Failed to log metrics: {e}")

    def on_update_end(
        self,
        trainer: "BaseTrainer",
        update_metrics: Dict[str, float]
    ) -> None:
        """Called after parameter update.

        Logs update metrics to wandb. This is useful for logging metrics
        that are computed at update time rather than step time.

        Args:
            trainer: The trainer instance.
            update_metrics: Dictionary of metrics from the update step.
        """
        if not self._initialized:
            return

        try:
            # Get update count for step
            update_step = self._step_count

            # Log update metrics
            log_data = {}
            for key, value in update_metrics.items():
                # Prefix update metrics to distinguish from step metrics
                log_data[f"update/{key}"] = value

            if log_data:
                wandb.log(log_data, step=update_step)

        except Exception as e:
            if self.verbose:
                print(f"WandbCallback: Warning - Failed to log update metrics: {e}")

    def on_training_end(self, trainer: "BaseTrainer") -> None:
        """Called at the end of training.

        Finishes the wandb run.

        Args:
            trainer: The trainer instance.
        """
        if not self._initialized:
            return

        try:
            if self.verbose:
                print(f"WandbCallback: Finishing wandb run after {self._step_count} steps")

            wandb.finish()
            self._initialized = False

        except Exception as e:
            if self.verbose:
                print(f"WandbCallback: Warning - Failed to finish wandb run: {e}")

    def log_custom(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log custom metrics to wandb.

        This method can be called manually to log additional metrics
        outside of the normal callback hooks.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number. If None, uses current step count.
        """
        if not self._initialized:
            return

        try:
            step = step if step is not None else self._step_count
            wandb.log(metrics, step=step)

        except Exception as e:
            if self.verbose:
                print(f"WandbCallback: Warning - Failed to log custom metrics: {e}")

    def is_initialized(self) -> bool:
        """Check if wandb has been initialized.

        Returns:
            True if wandb.run is active, False otherwise.
        """
        return self._initialized

    def get_run_url(self) -> Optional[str]:
        """Get the URL of the current wandb run.

        Returns:
            The URL of the wandb run, or None if not initialized.
        """
        if not self._initialized:
            return None

        try:
            return wandb.run.url if wandb.run else None
        except Exception:
            return None

    def get_run_id(self) -> Optional[str]:
        """Get the ID of the current wandb run.

        Returns:
            The ID of the wandb run, or None if not initialized.
        """
        if not self._initialized:
            return None

        try:
            return wandb.run.id if wandb.run else None
        except Exception:
            return None
