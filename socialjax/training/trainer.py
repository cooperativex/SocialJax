"""Unified Trainer class for SocialJax.

This module provides the main Trainer class that combines all components
(algorithm, environment, config, callbacks) into a unified training interface.

Example usage:
    >>> from socialjax.training import Trainer
    >>> trainer = Trainer(algorithm="ippo", env="coin_game")
    >>> state, metrics = trainer.train(total_timesteps=1_000_000)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import time

import jax
import jax.numpy as jnp
import numpy as np

from socialjax.core.base_trainer import (
    BaseTrainer,
    TrainerState,
    TrainingMetrics,
    Callback,
)
from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
from socialjax.algorithms.registry import get_algorithm, list_algorithms
from socialjax.config.manager import ConfigManager, SocialJaxConfig, create_default_config
from socialjax.training.callbacks.base_callback import BaseCallback, CallbackList


class SpaceWrapper:
    """Wrapper to make callable spaces behave like gym-style spaces."""

    def __init__(self, space_callable, space_type="observation"):
        """Initialize wrapper.

        Args:
            space_callable: The callable that returns (space, shape) or space
            space_type: "observation" or "action"
        """
        self._callable = space_callable
        self._space_type = space_type
        self._cached_result = None

    def _get_result(self):
        """Get and cache the space result."""
        if self._cached_result is None:
            result = self._callable()
            if isinstance(result, tuple) and len(result) == 2:
                # (Box, shape) tuple
                self._cached_result = (result[0], result[1])
            else:
                # Just the space
                self._cached_result = result
        return self._cached_result

    @property
    def shape(self):
        """Get shape of the space."""
        result = self._get_result()
        if isinstance(result, tuple):
            return result[1]
        elif hasattr(result, "shape"):
            return result.shape
        return None

    @property
    def n(self):
        """Get number of actions (for Discrete spaces)."""
        result = self._get_result()
        if isinstance(result, tuple):
            space = result[0]
        else:
            space = result

        if hasattr(space, "n"):
            return space.n
        return None


class DummyObservationSpace:
    """Dummy observation space for algorithms."""

    def __init__(self, shape):
        self.shape = shape


class DummyActionSpace:
    """Dummy action space for algorithms."""

    def __init__(self, n):
        self.n = n


class RolloutBuffer:
    """Simple rollout buffer for on-policy algorithms.

    Stores trajectories collected during rollouts for later use in updates.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
    ):
        """Initialize the rollout buffer.

        Args:
            buffer_size: Number of steps to store per environment
            num_envs: Number of parallel environments
            obs_shape: Shape of observations
            action_dim: Dimension of actions
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        # Initialize storage
        self.observations = np.zeros(
            (buffer_size, num_envs) + obs_shape, dtype=np.float32
        )
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Add a single step to the buffer.

        Args:
            obs: Observation array
            action: Action array
            reward: Reward array
            done: Done flag array
            log_prob: Log probability array
            value: Value estimate array
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored data.

        Returns:
            Dictionary with all stored arrays
        """
        indices = np.arange(self.pos if not self.full else self.buffer_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "log_probs": self.log_probs[indices],
            "values": self.values[indices],
        }

    def clear(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False


class Trainer(BaseTrainer):
    """Unified trainer for training RL algorithms in SocialJax.

    This class provides a high-level interface for training algorithms on
    environments, with support for:
    - Automatic algorithm and environment creation via registry
    - Configuration management via ConfigManager
    - Callback integration for logging, checkpointing, evaluation
    - Standard training loop with on-policy rollouts

    The Trainer can be created in several ways:
        1. With algorithm and env names (uses defaults):
            >>> trainer = Trainer(algorithm="ippo", env="coin_game")

        2. With custom config:
            >>> config = create_default_config(algorithm="ippo", env="coin_game")
            >>> trainer = Trainer(config=config)

        3. With pre-built algorithm and environment:
            >>> algo = IPPOAlgorithm(obs_space, action_space)
            >>> env = socialjax.make("coin_game")
            >>> trainer = Trainer(algorithm=algo, env=env)

    Attributes:
        algorithm: The algorithm to train (BaseAlgorithm or str)
        env: The environment to train on (environment instance or str)
        config: Training configuration (dict or SocialJaxConfig)
        callbacks: List of callbacks for training hooks
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        algorithm: Optional[Union[str, BaseAlgorithm]] = None,
        env: Optional[Union[str, Any]] = None,
        config: Optional[Union[Dict[str, Any], SocialJaxConfig]] = None,
        callbacks: Optional[List[Callback]] = None,
        seed: int = 42,
        **kwargs,
    ):
        """Initialize the Trainer.

        Args:
            algorithm: Algorithm name (str) or instance (BaseAlgorithm).
                If str, will be loaded from registry.
            env: Environment name (str) or instance. If str, will be created
                via socialjax.make().
            config: Training configuration. Can be:
                - SocialJaxConfig instance
                - Dict with algorithm/env config
                - None (uses defaults from algorithm/env names)
            callbacks: List of callbacks for training hooks.
            seed: Random seed for reproducibility.
            **kwargs: Additional config overrides.
        """
        self.seed = seed
        self._rng = jax.random.PRNGKey(seed)

        # Initialize config
        self._socialjax_config = self._init_config(algorithm, env, config, **kwargs)

        # Initialize environment
        self._env_name = None
        if isinstance(env, str):
            self._env_name = env
        self.env = self._init_env(env)

        # Initialize algorithm
        self._algo_name = None
        if isinstance(algorithm, str):
            self._algo_name = algorithm
        self.algorithm = self._init_algorithm(algorithm)

        # Initialize callbacks
        self._callback_list = CallbackList(callbacks or [])

        # Convert SocialJaxConfig to dict for BaseTrainer
        train_config = self._extract_train_config()

        # Initialize BaseTrainer
        super().__init__(
            algorithm=self.algorithm,
            env=self.env,
            config=train_config,
            callbacks=callbacks or [],
        )

    def _init_config(
        self,
        algorithm: Optional[Union[str, BaseAlgorithm]],
        env: Optional[Union[str, Any]],
        config: Optional[Union[Dict[str, Any], SocialJaxConfig]],
        **kwargs,
    ) -> SocialJaxConfig:
        """Initialize configuration.

        Args:
            algorithm: Algorithm name or instance
            env: Environment name or instance
            config: Provided config
            **kwargs: Config overrides (will be mapped to appropriate config sections)

        Returns:
            SocialJaxConfig instance
        """
        if isinstance(config, SocialJaxConfig):
            return config

        # Get algorithm and env names
        algo_name = algorithm if isinstance(algorithm, str) else None
        env_name = env if isinstance(env, str) else None

        # Separate kwargs into training vs other config
        training_keys = {
            "total_timesteps", "num_envs", "num_steps", "update_epochs",
            "num_minibatches", "gamma", "gae_lambda", "learning_rate",
            "clip_eps", "ent_coef", "vf_coef", "max_grad_norm", "seed",
            "anneal_lr", "wandb_project", "wandb_entity",
        }

        # Build training overrides from kwargs
        training_overrides = {}
        other_overrides = {}

        for key, value in kwargs.items():
            if key in training_keys:
                training_overrides[key] = value
            elif key not in ("algorithm", "environment"):
                other_overrides[key] = value

        # Create base config
        if config is None:
            base_config = create_default_config(
                algorithm=algo_name or "ippo",
                environment=env_name or "coin_game",
            )
        elif isinstance(config, dict):
            if "algorithm" in config and "environment" in config:
                base_config = SocialJaxConfig.from_dict(config)
            else:
                base_config = create_default_config(
                    algorithm=algo_name or "ippo",
                    environment=env_name or "coin_game",
                    **config,
                )
        else:
            base_config = config

        # Apply training overrides if any
        if training_overrides:
            from socialjax.config.manager import ConfigManager
            manager = ConfigManager()

            # Build override structure
            overrides = {}
            if training_overrides:
                overrides["algorithm"] = {"training": training_overrides}
            if other_overrides:
                for k, v in other_overrides.items():
                    overrides[k] = v

            # Merge overrides into base config
            merged_dict = manager._merge_dicts(base_config.to_dict(), overrides)
            base_config = SocialJaxConfig.from_dict(merged_dict)

        return base_config

    def _init_env(self, env: Optional[Union[str, Any]]) -> Any:
        """Initialize environment.

        Args:
            env: Environment name or instance

        Returns:
            Environment instance
        """
        if env is None:
            # Create default environment
            import socialjax
            return socialjax.make(
                self._socialjax_config.environment.name,
                num_agents=self._socialjax_config.environment.num_agents,
            )

        if isinstance(env, str):
            import socialjax
            return socialjax.make(
                env,
                num_agents=self._socialjax_config.environment.num_agents,
            )

        return env

    def _init_algorithm(self, algorithm: Optional[Union[str, BaseAlgorithm]]) -> BaseAlgorithm:
        """Initialize algorithm.

        Args:
            algorithm: Algorithm name or instance

        Returns:
            BaseAlgorithm instance
        """
        # Get wrapped spaces for algorithm
        obs_space = self._get_obs_space_wrapper()
        action_space = self._get_action_space_wrapper()

        if algorithm is None:
            # Create default algorithm
            algo_name = self._socialjax_config.algorithm.name
            algo_class = get_algorithm(algo_name)
            return algo_class(
                observation_space=obs_space,
                action_space=action_space,
                config=self._socialjax_config.algorithm.to_dict(),
            )

        if isinstance(algorithm, str):
            algo_class = get_algorithm(algorithm)
            return algo_class(
                observation_space=obs_space,
                action_space=action_space,
                config=self._socialjax_config.algorithm.to_dict(),
            )

        return algorithm

    def _get_obs_space_wrapper(self) -> Any:
        """Get observation space wrapper.

        Returns:
            SpaceWrapper, DummyObservationSpace, or raw space
        """
        if hasattr(self.env, "observation_space"):
            obs_space = self.env.observation_space
            if callable(obs_space):
                return SpaceWrapper(obs_space, "observation")
            elif hasattr(obs_space, "shape"):
                return obs_space

        # Fallback: create dummy space
        return DummyObservationSpace(self._get_obs_shape())

    def _get_action_space_wrapper(self) -> Any:
        """Get action space wrapper.

        Returns:
            SpaceWrapper, DummyActionSpace, or raw space
        """
        if hasattr(self.env, "action_space"):
            action_space = self.env.action_space
            if callable(action_space):
                return SpaceWrapper(action_space, "action")
            elif hasattr(action_space, "n"):
                return action_space

        # Fallback: create dummy space
        return DummyActionSpace(self._get_action_dim())

    def _extract_train_config(self) -> Dict[str, Any]:
        """Extract training config as dict for BaseTrainer.

        Returns:
            Dictionary with training configuration
        """
        return self._socialjax_config.algorithm.training.to_dict()

    def _setup(self) -> None:
        """Setup trainer components.

        Creates the rollout buffer and initializes any additional components.
        """
        self.buffer = self._create_buffer()

        # Set trainer reference for all callbacks
        self._callback_list.set_trainer(self)
        for callback in self.callbacks:
            if hasattr(callback, "set_trainer"):
                callback.set_trainer(self)

    def _create_buffer(self) -> RolloutBuffer:
        """Create the rollout buffer.

        Returns:
            RolloutBuffer instance configured for the environment
        """
        # Get observation shape
        obs_shape = self._get_obs_shape()

        # Get action dimension
        action_dim = self._get_action_dim()

        num_steps = self.config.get("num_steps", 128)
        num_envs = self.config.get("num_envs", 1)

        return RolloutBuffer(
            buffer_size=num_steps,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_dim=action_dim,
        )

    def _get_obs_shape(self) -> Tuple[int, ...]:
        """Get observation shape from environment.

        Returns:
            Tuple of observation dimensions
        """
        if hasattr(self.env, "observation_space"):
            obs_space = self.env.observation_space
            # Handle callable observation_space (JaxMARL style)
            if callable(obs_space):
                result = obs_space()
                # Returns (Box, shape) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    return result[1]
                elif hasattr(result, "shape"):
                    return result.shape
            elif hasattr(obs_space, "shape"):
                return obs_space.shape

        # Fallback for environments without gym-style spaces
        return (10, 10, 3)  # Default CNN input

    def _get_action_dim(self) -> int:
        """Get action dimension from environment.

        Returns:
            Number of actions
        """
        if hasattr(self.env, "action_space"):
            action_space = self.env.action_space
            # Handle callable action_space (JaxMARL style)
            if callable(action_space):
                space = action_space()
                if hasattr(space, "n"):
                    return space.n
                elif hasattr(space, "shape"):
                    return space.shape[0]
            elif hasattr(action_space, "n"):
                return action_space.n
            elif hasattr(action_space, "shape"):
                return action_space.shape[0]

        return 4  # Default

    def _collect_rollout(
        self,
        state: TrainerState,
    ) -> Tuple[Dict[str, Any], TrainerState]:
        """Collect a rollout of experience from the environment.

        Args:
            state: Current trainer state

        Returns:
            Tuple of (rollout_data, new_state)
        """
        num_steps = self.config.get("num_steps", 128)
        gamma = self.config.get("gamma", 0.99)
        gae_lambda = self.config.get("gae_lambda", 0.95)

        # Reset environment
        self._rng, reset_rng = jax.random.split(self._rng)
        obs, env_state = self.env.reset(reset_rng)

        # Storage for trajectory
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        episode_returns = []
        episode_lengths = []

        current_episode_return = 0.0
        current_episode_length = 0

        # Collect trajectory
        for step in range(num_steps):
            # Compute action for each agent
            step_actions = {}
            step_log_probs = {}
            step_values = {}

            for agent in self.env.agents:
                self._rng, action_rng = jax.random.split(self._rng)
                action, info = self.algorithm.compute_action(
                    state.algorithm_state,
                    obs[agent],
                    action_rng,
                    deterministic=False,
                )
                step_actions[agent] = np.array(action)
                step_log_probs[agent] = np.array(info["log_prob"])
                step_values[agent] = np.array(info["value"])

            # Convert to arrays (assuming shared parameters, use first agent)
            first_agent = self.env.agents[0]
            action_arr = step_actions[first_agent]
            log_prob_arr = step_log_probs[first_agent]
            value_arr = step_values[first_agent]

            # Store current obs
            obs_arr = np.array(obs[first_agent])

            # Step environment
            self._rng, step_rng = jax.random.split(self._rng)
            env_state, next_obs, step_rewards, step_dones, info = self.env.step(
                env_state, step_actions
            )

            # Extract reward and done for first agent (parameter sharing)
            reward_arr = np.array(step_rewards[first_agent])
            done_arr = np.array(step_dones.get("__all__", step_dones[first_agent]))

            # Accumulate episode statistics
            current_episode_return += float(reward_arr)
            current_episode_length += 1

            # Check for episode end
            if done_arr:
                episode_returns.append(current_episode_return)
                episode_lengths.append(current_episode_length)
                current_episode_return = 0.0
                current_episode_length = 0

            # Store transition
            observations.append(obs_arr)
            actions.append(action_arr)
            rewards.append(reward_arr)
            dones.append(done_arr)
            log_probs.append(log_prob_arr)
            values.append(value_arr)

            # Update obs for next step
            obs = next_obs

        # Get last value for GAE computation
        self._rng, last_rng = jax.random.split(self._rng)
        _, last_info = self.algorithm.compute_action(
            state.algorithm_state,
            obs[first_agent],
            last_rng,
            deterministic=False,
        )
        last_value = np.array(last_info["value"])

        # Compute advantages using GAE
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        log_probs = np.array(log_probs)
        values = np.array(values)

        # Compute GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        targets = advantages + values

        # Create rollout data dict
        rollout_data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "log_probs": log_probs,
            "values": values,
            "advantages": advantages,
            "targets": targets,
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
        }

        # Update trainer state
        new_timestep = state.timestep + num_steps
        new_episode_count = state.episode_count + len(episode_returns)

        new_state = TrainerState(
            algorithm_state=state.algorithm_state,
            timestep=new_timestep,
            update_step=state.update_step,
            episode_count=new_episode_count,
            start_time=state.start_time,
        )

        return rollout_data, new_state

    def _update(
        self,
        state: TrainerState,
        rollout_data: Dict[str, Any],
    ) -> Tuple[TrainerState, Dict[str, float]]:
        """Update algorithm parameters using rollout data.

        Args:
            state: Current trainer state
            rollout_data: Collected rollout data

        Returns:
            Tuple of (new_state, update_metrics)
        """
        update_epochs = self.config.get("update_epochs", 4)
        num_minibatches = self.config.get("num_minibatches", 4)

        # Prepare batch
        obs = jnp.array(rollout_data["observations"])
        actions = jnp.array(rollout_data["actions"])
        advantages = jnp.array(rollout_data["advantages"])
        targets = jnp.array(rollout_data["targets"])
        old_log_probs = jnp.array(rollout_data["log_probs"])
        values = jnp.array(rollout_data["values"])

        # Flatten batch
        batch_size = obs.shape[0]
        obs = obs.reshape(batch_size, -1)
        actions = actions.reshape(batch_size)
        advantages = advantages.reshape(batch_size)
        targets = targets.reshape(batch_size)
        old_log_probs = old_log_probs.reshape(batch_size)
        values = values.reshape(batch_size)

        # Track metrics across epochs
        total_loss_sum = 0.0
        value_loss_sum = 0.0
        actor_loss_sum = 0.0
        entropy_sum = 0.0

        algorithm_state = state.algorithm_state

        # Multiple update epochs
        for epoch in range(update_epochs):
            # Shuffle data
            self._rng, shuffle_rng = jax.random.split(self._rng)
            indices = jax.random.permutation(shuffle_rng, batch_size)

            obs_shuffled = obs[indices]
            actions_shuffled = actions[indices]
            advantages_shuffled = advantages[indices]
            targets_shuffled = targets[indices]
            old_log_probs_shuffled = old_log_probs[indices]
            values_shuffled = values[indices]

            # Split into minibatches
            minibatch_size = max(1, batch_size // num_minibatches)

            for i in range(num_minibatches):
                start_idx = i * minibatch_size
                end_idx = min(start_idx + minibatch_size, batch_size)

                if start_idx >= batch_size:
                    break

                batch = {
                    "obs": obs_shuffled[start_idx:end_idx],
                    "actions": actions_shuffled[start_idx:end_idx],
                    "advantages": advantages_shuffled[start_idx:end_idx],
                    "targets": targets_shuffled[start_idx:end_idx],
                    "old_log_probs": old_log_probs_shuffled[start_idx:end_idx],
                    "values": values_shuffled[start_idx:end_idx],
                }

                # Update algorithm
                algorithm_state, metrics = self.algorithm.update(algorithm_state, batch)

                total_loss_sum += metrics["total_loss"]
                value_loss_sum += metrics["value_loss"]
                actor_loss_sum += metrics["actor_loss"]
                entropy_sum += metrics["entropy"]

        # Average metrics
        n_updates = update_epochs * num_minibatches
        avg_metrics = {
            "total_loss": total_loss_sum / n_updates,
            "value_loss": value_loss_sum / n_updates,
            "actor_loss": actor_loss_sum / n_updates,
            "entropy": entropy_sum / n_updates,
        }

        # Update episode metrics
        for ret in rollout_data.get("episode_returns", []):
            self.metrics.add_episode(ret, 0)

        # Create new trainer state
        new_state = TrainerState(
            algorithm_state=algorithm_state,
            timestep=state.timestep,
            update_step=state.update_step + 1,
            episode_count=state.episode_count,
            start_time=state.start_time,
        )

        return new_state, avg_metrics

    def train(
        self,
        total_timesteps: Optional[int] = None,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[TrainerState, Dict[str, Any]]:
        """Run the training loop.

        Args:
            total_timesteps: Total number of environment steps to train for.
                If None, uses config value.
            rng: Optional JAX random key.

        Returns:
            Tuple of (final_state, metrics_dict)
        """
        if total_timesteps is None:
            total_timesteps = self.config.get("total_timesteps", 1_000_000)

        return super().train(total_timesteps, rng)

    def evaluate(
        self,
        state: Optional[TrainerState] = None,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the current policy.

        Args:
            state: Trainer state to evaluate. If None, must train first.
            num_episodes: Number of episodes to run.
            deterministic: Whether to use deterministic actions.

        Returns:
            Dictionary of evaluation metrics
        """
        if state is None:
            # Create a temporary state from the current algorithm
            self._rng, algo_rng = jax.random.split(self._rng)
            algorithm_state = self.algorithm.init_state(algo_rng)
            state = TrainerState(
                algorithm_state=algorithm_state,
                timestep=0,
                update_step=0,
                episode_count=0,
                start_time=time.time(),
            )

        return super().evaluate(state, num_episodes, deterministic)

    def save(self, path: str) -> None:
        """Save trainer state to disk.

        Args:
            path: Directory path for saving
        """
        super().save(path)

    def load(self, path: str) -> TrainerState:
        """Load trainer state from disk.

        Args:
            path: Directory path containing checkpoint

        Returns:
            Loaded TrainerState
        """
        return super().load(path)

    @classmethod
    def from_config(
        cls,
        config: Union[Dict[str, Any], SocialJaxConfig],
        callbacks: Optional[List[Callback]] = None,
        seed: int = 42,
    ) -> "Trainer":
        """Create a Trainer from configuration.

        Args:
            config: Training configuration
            callbacks: Optional list of callbacks
            seed: Random seed

        Returns:
            Trainer instance
        """
        return cls(config=config, callbacks=callbacks, seed=seed)

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Get the full config as a dictionary.

        Returns:
            Dictionary with full configuration
        """
        return self._socialjax_config.to_dict()


def create_trainer(
    algorithm: str,
    env: str,
    config: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[Callback]] = None,
    seed: int = 42,
    **kwargs,
) -> Trainer:
    """Factory function to create a Trainer.

    This is a convenience function for creating trainers with minimal code.

    Args:
        algorithm: Algorithm name (e.g., "ippo", "mappo")
        env: Environment name (e.g., "coin_game", "clean_up")
        config: Optional configuration overrides
        callbacks: Optional list of callbacks
        seed: Random seed
        **kwargs: Additional config overrides

    Returns:
        Configured Trainer instance

    Example:
        >>> trainer = create_trainer("ippo", "coin_game", total_timesteps=1_000_000)
        >>> state, metrics = trainer.train()
    """
    return Trainer(
        algorithm=algorithm,
        env=env,
        config=config,
        callbacks=callbacks,
        seed=seed,
        **kwargs,
    )
