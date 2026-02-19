"""Base trainer abstract class for SocialJax.

This module provides the abstract base class that all trainers in SocialJax
must inherit from. It defines the interface for:
- Training loop management
- Buffer creation
- Callback integration
- Training metrics collection
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Protocol
import time

import jax
import jax.numpy as jnp
from flax import struct

from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState


class Callback(Protocol):
    """Protocol defining the callback interface.

    Callbacks are used to hook into the training process at various points
    (start, end, each step, etc.) for logging, checkpointing, evaluation, etc.
    """

    def on_training_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of training."""
        ...

    def on_training_end(self, trainer: "BaseTrainer") -> None:
        """Called at the end of training."""
        ...

    def on_step(self, trainer: "BaseTrainer", step: int, metrics: Dict[str, float]) -> None:
        """Called after each training step."""
        ...

    def on_rollout_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of a rollout."""
        ...

    def on_rollout_end(self, trainer: "BaseTrainer", rollout_data: Dict[str, Any]) -> None:
        """Called at the end of a rollout."""
        ...

    def on_update_start(self, trainer: "BaseTrainer") -> None:
        """Called before parameter update."""
        ...

    def on_update_end(self, trainer: "BaseTrainer", update_metrics: Dict[str, float]) -> None:
        """Called after parameter update."""
        ...


@struct.dataclass
class TrainerState:
    """Immutable state container for trainer runtime state.

    This dataclass uses Flax struct.dataclass decorator to create a
    JAX-compatible immutable data structure.

    Attributes:
        algorithm_state: The current algorithm state (params, optimizer, rng).
        timestep: Current total timesteps across all environments.
        update_step: Current number of parameter updates performed.
        episode_count: Total number of episodes completed.
        start_time: Training start time (for elapsed time tracking).
    """
    algorithm_state: AlgorithmState
    timestep: int = 0
    update_step: int = 0
    episode_count: int = 0
    start_time: float = 0.0


class TrainingMetrics:
    """Container for training metrics collected during training.

    Attributes:
        episode_returns: List of episode returns.
        episode_lengths: List of episode lengths.
        losses: Dictionary of loss values over time.
        custom_metrics: Dictionary of custom metrics.
    """

    def __init__(self):
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: Dict[str, List[float]] = {}
        self.custom_metrics: Dict[str, List[float]] = {}

    def add_episode(self, return_value: float, length: int) -> None:
        """Add an episode return and length."""
        self.episode_returns.append(return_value)
        self.episode_lengths.append(length)

    def add_loss(self, name: str, value: float) -> None:
        """Add a loss value."""
        if name not in self.losses:
            self.losses[name] = []
        self.losses[name].append(value)

    def add_metric(self, name: str, value: float) -> None:
        """Add a custom metric value."""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        import numpy as np

        summary = {}

        if self.episode_returns:
            summary["mean_episode_return"] = float(np.mean(self.episode_returns))
            summary["std_episode_return"] = float(np.std(self.episode_returns))
            summary["min_episode_return"] = float(np.min(self.episode_returns))
            summary["max_episode_return"] = float(np.max(self.episode_returns))

        if self.episode_lengths:
            summary["mean_episode_length"] = float(np.mean(self.episode_lengths))

        for name, values in self.losses.items():
            if values:
                summary[f"mean_{name}"] = float(np.mean(values[-100:]))

        return summary


class BaseTrainer(ABC):
    """Abstract base class for all training loops in SocialJax.

    This class defines the interface that all trainers must implement to work
    with the SocialJax framework. It provides:
    - Standard training loop structure
    - Callback system integration
    - Metrics collection and reporting
    - Component initialization pattern

    Subclasses must implement the abstract methods, particularly _create_buffer()
    to specify the type of experience buffer to use.

    The training loop follows this structure:
    1. on_training_start callbacks
    2. Loop until total_timesteps:
        a. on_rollout_start callbacks
        b. Collect rollout data
        c. on_rollout_end callbacks
        d. on_update_start callbacks
        e. Update parameters
        f. on_update_end callbacks
        g. on_step callbacks
    3. on_training_end callbacks

    Example:
        >>> class IPPOTrainer(BaseTrainer):
        ...     def _create_buffer(self):
        ...         return RolloutBuffer(
        ...             buffer_size=self.config["NUM_STEPS"],
        ...             num_envs=self.config["NUM_ENVS"],
        ...         )
        ...
        ...     def _collect_rollout(self, state):
        ...         # Collect experience from environment
        ...         return rollout_data, new_state

    Attributes:
        algorithm: The algorithm to train (BaseAlgorithm instance).
        env: The environment to train on.
        config: Training configuration dictionary.
        callbacks: List of callbacks to invoke during training.
        metrics: TrainingMetrics instance for collecting metrics.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: Any,
        config: Dict[str, Any],
        callbacks: Optional[List[Callback]] = None,
    ):
        """Initialize the trainer.

        Args:
            algorithm: The algorithm to train.
            env: The environment to train on.
            config: Training configuration dictionary with keys like:
                - total_timesteps: Total number of environment steps.
                - num_steps: Number of steps per rollout.
                - num_envs: Number of parallel environments.
                - update_epochs: Number of update epochs per rollout.
            callbacks: Optional list of callbacks for training hooks.
        """
        self.algorithm = algorithm
        self.env = env
        self.config = config
        self.callbacks = callbacks or []
        self.metrics = TrainingMetrics()
        self._setup()

    def _setup(self) -> None:
        """Initialize training components.

        Called automatically during __init__. Sets up the buffer and
        any other necessary components. Subclasses can override to add
        additional setup logic, but should call super()._setup().
        """
        self.buffer = self._create_buffer()

    @abstractmethod
    def _create_buffer(self) -> Any:
        """Create and return the experience buffer.

        Subclasses must implement this to specify the type of buffer
        to use (e.g., RolloutBuffer for on-policy, ReplayBuffer for off-policy).

        Returns:
            A buffer instance for storing experience data.
        """
        pass

    def _on_training_start(self) -> None:
        """Invoke on_training_start callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, "on_training_start"):
                callback.on_training_start(self)

    def _on_training_end(self) -> None:
        """Invoke on_training_end callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, "on_training_end"):
                callback.on_training_end(self)

    def _on_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Invoke on_step callbacks.

        Args:
            step: Current step number.
            metrics: Dictionary of metrics from this step.
        """
        for callback in self.callbacks:
            if hasattr(callback, "on_step"):
                callback.on_step(self, step, metrics)

    def _on_rollout_start(self) -> None:
        """Invoke on_rollout_start callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, "on_rollout_start"):
                callback.on_rollout_start(self)

    def _on_rollout_end(self, rollout_data: Dict[str, Any]) -> None:
        """Invoke on_rollout_end callbacks.

        Args:
            rollout_data: Dictionary containing the collected rollout data.
        """
        for callback in self.callbacks:
            if hasattr(callback, "on_rollout_end"):
                callback.on_rollout_end(self, rollout_data)

    def _on_update_start(self) -> None:
        """Invoke on_update_start callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, "on_update_start"):
                callback.on_update_start(self)

    def _on_update_end(self, update_metrics: Dict[str, float]) -> None:
        """Invoke on_update_end callbacks.

        Args:
            update_metrics: Dictionary of metrics from the update step.
        """
        for callback in self.callbacks:
            if hasattr(callback, "on_update_end"):
                callback.on_update_end(self, update_metrics)

    def train(
        self,
        total_timesteps: int,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[TrainerState, Dict[str, Any]]:
        """Run the training loop.

        This is the main entry point for training. It initializes the trainer
        state and runs the training loop until total_timesteps is reached.

        Args:
            total_timesteps: Total number of environment steps to train for.
            rng: Optional JAX random key. If None, creates a new one.

        Returns:
            Tuple of:
                - final_state: The final TrainerState.
                - metrics: Dictionary of training metrics and summary.

        Note:
            Subclasses may override this method to customize the training loop,
            but should follow the callback invocation pattern for consistency.
        """
        # Initialize RNG if not provided
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Initialize algorithm state
        rng, algo_rng = jax.random.split(rng)
        algorithm_state = self.algorithm.init_state(algo_rng)

        # Create trainer state
        state = TrainerState(
            algorithm_state=algorithm_state,
            timestep=0,
            update_step=0,
            episode_count=0,
            start_time=time.time(),
        )

        # Invoke training start callbacks
        self._on_training_start()

        try:
            # Run training loop
            while state.timestep < total_timesteps:
                # Rollout phase
                self._on_rollout_start()
                rollout_data, state = self._collect_rollout(state)
                self._on_rollout_end(rollout_data)

                # Update phase
                self._on_update_start()
                state, update_metrics = self._update(state, rollout_data)
                self._on_update_end(update_metrics)

                # Record metrics
                for name, value in update_metrics.items():
                    self.metrics.add_loss(name, value)

                # Invoke step callbacks
                self._on_step(state.update_step, update_metrics)

        except KeyboardInterrupt:
            print(f"\nTraining interrupted at timestep {state.timestep}")

        # Invoke training end callbacks
        self._on_training_end()

        # Return final state and metrics
        final_metrics = {
            "training_summary": self.metrics.get_summary(),
            "total_timesteps": state.timestep,
            "total_updates": state.update_step,
            "elapsed_time": time.time() - state.start_time,
        }

        return state, final_metrics

    @abstractmethod
    def _collect_rollout(
        self,
        state: TrainerState,
    ) -> Tuple[Dict[str, Any], TrainerState]:
        """Collect a rollout of experience from the environment.

        Subclasses must implement this to collect experience data according
        to their algorithm needs.

        Args:
            state: Current trainer state.

        Returns:
            Tuple of:
                - rollout_data: Dictionary containing the collected experience.
                - new_state: Updated trainer state.
        """
        pass

    @abstractmethod
    def _update(
        self,
        state: TrainerState,
        rollout_data: Dict[str, Any],
    ) -> Tuple[TrainerState, Dict[str, float]]:
        """Update algorithm parameters using collected rollout data.

        Subclasses must implement this to perform the parameter update
        according to their algorithm needs.

        Args:
            state: Current trainer state.
            rollout_data: Dictionary containing the collected experience.

        Returns:
            Tuple of:
                - new_state: Updated trainer state with new params.
                - metrics: Dictionary of training metrics from the update.
        """
        pass

    def evaluate(
        self,
        state: TrainerState,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the current policy.

        Args:
            state: Current trainer state.
            num_episodes: Number of episodes to run.
            deterministic: Whether to use deterministic action selection.

        Returns:
            Dictionary of evaluation metrics (mean_return, std_return, etc.).
        """
        import numpy as np

        returns = []
        lengths = []

        for _ in range(num_episodes):
            rng, eval_rng = jax.random.split(state.algorithm_state.rng)
            obs, env_state = self.env.reset(eval_rng)
            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                # Compute actions for all agents
                actions = {}
                for agent in self.env.agents:
                    rng, action_rng = jax.random.split(rng)
                    action, _ = self.algorithm.compute_action(
                        state.algorithm_state,
                        obs[agent],
                        action_rng,
                        deterministic=deterministic,
                    )
                    actions[agent] = action

                # Step environment
                rng, step_rng = jax.random.split(rng)
                env_state, obs, rewards, dones, info = self.env.step(
                    env_state, actions
                )

                # Accumulate rewards
                episode_return += sum(rewards.values())
                episode_length += 1
                done = dones.get("__all__", False)

            returns.append(episode_return)
            lengths.append(episode_length)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "num_episodes": num_episodes,
        }

    def save(self, path: str) -> None:
        """Save trainer state to disk.

        Args:
            path: Directory path to save the checkpoint.
        """
        import os
        import pickle
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save via algorithm
        self.algorithm.save(str(path / "algorithm"))

        # Save trainer-specific state
        trainer_info = {
            "config": self.config,
            "metrics": {
                "episode_returns": self.metrics.episode_returns,
                "episode_lengths": self.metrics.episode_lengths,
                "losses": self.metrics.losses,
                "custom_metrics": self.metrics.custom_metrics,
            },
        }

        with open(path / "trainer_info.pkl", "wb") as f:
            pickle.dump(trainer_info, f)

    def load(self, path: str) -> TrainerState:
        """Load trainer state from disk.

        Args:
            path: Directory path containing the checkpoint.

        Returns:
            Loaded TrainerState.
        """
        import pickle
        from pathlib import Path

        path = Path(path)

        # Load algorithm state
        algorithm_state = self.algorithm.load(str(path / "algorithm"))

        # Load trainer info
        with open(path / "trainer_info.pkl", "rb") as f:
            trainer_info = pickle.load(f)

        # Restore metrics
        self.metrics.episode_returns = trainer_info["metrics"]["episode_returns"]
        self.metrics.episode_lengths = trainer_info["metrics"]["episode_lengths"]
        self.metrics.losses = trainer_info["metrics"]["losses"]
        self.metrics.custom_metrics = trainer_info["metrics"]["custom_metrics"]

        # Create and return trainer state
        return TrainerState(
            algorithm_state=algorithm_state,
            timestep=len(self.metrics.episode_returns) * 100,  # Approximate
            update_step=len(self.metrics.losses.get("total_loss", [])),
            episode_count=len(self.metrics.episode_returns),
            start_time=time.time(),
        )
