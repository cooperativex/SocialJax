"""Evaluation callback for periodic model evaluation during training.

This module provides the EvalCallback class that periodically evaluates
the model during training and optionally saves the best performing model.

Example:
    >>> from socialjax.training.callbacks import EvalCallback
    >>> import socialjax
    >>> eval_env = socialjax.make('coin_game', num_agents=5)
    >>> callback = EvalCallback(
    ...     eval_env=eval_env,
    ...     eval_freq=1000,
    ...     n_eval_episodes=10,
    ...     best_model_save_path="./best_models",
    ...     deterministic=True,
    ...     verbose=True
    ... )
    >>> # Add to trainer's callback list
"""

import os
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from socialjax.training.callbacks.base_callback import BaseCallback

if TYPE_CHECKING:
    from socialjax.core.base_trainer import BaseTrainer
    from socialjax.environments.multi_agent_env import MultiAgentEnv


class EvalCallback(BaseCallback):
    """Callback for evaluating the model during training.

    This callback periodically evaluates the model on a separate environment
    and tracks the best performing model based on mean episode reward.
    The best model can be automatically saved.

    Attributes:
        eval_env: Environment to use for evaluation.
        eval_freq: Number of updates between evaluations.
        n_eval_episodes: Number of episodes to run during evaluation.
        best_model_save_path: Directory to save best models (optional).
        deterministic: If True, use deterministic actions during evaluation.
        verbose: If True, print evaluation results.
        best_mean_reward: Best mean reward achieved during evaluation.
        last_mean_reward: Mean reward from the last evaluation.
        update_count: Internal counter tracking number of updates.

    Example:
        >>> import socialjax
        >>> eval_env = socialjax.make('coin_game', num_agents=5)
        >>> callback = EvalCallback(
        ...     eval_env=eval_env,
        ...     eval_freq=1000,
        ...     n_eval_episodes=10,
        ...     best_model_save_path="./checkpoints/best",
        ...     deterministic=True,
        ...     verbose=True
        ... )
    """

    def __init__(
        self,
        eval_env: "MultiAgentEnv",
        eval_freq: int = 1000,
        n_eval_episodes: int = 10,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: bool = False,
        warn: bool = True
    ):
        """Initialize the evaluation callback.

        Args:
            eval_env: Environment to use for evaluation. Should be separate
                from the training environment to avoid contamination.
            eval_freq: Number of updates between evaluations. An evaluation
                will be performed every `eval_freq` updates.
            n_eval_episodes: Number of episodes to run during each evaluation.
                More episodes give more stable metrics but take longer.
            best_model_save_path: Directory path where best models will be saved.
                If None, best models are not saved. The directory will be
                created if it doesn't exist.
            deterministic: If True, use deterministic (greedy) actions during
                evaluation. If False, use stochastic actions from the policy.
            verbose: If True, print evaluation results to stdout.
            warn: If True, warn if no episodes completed during evaluation.

        Raises:
            ValueError: If eval_freq or n_eval_episodes is not a positive integer.
            ValueError: If eval_env is None.
        """
        super().__init__(verbose=verbose)

        if eval_freq <= 0:
            raise ValueError(f"eval_freq must be a positive integer, got {eval_freq}")
        if n_eval_episodes <= 0:
            raise ValueError(f"n_eval_episodes must be a positive integer, got {n_eval_episodes}")
        if eval_env is None:
            raise ValueError("eval_env cannot be None")

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.warn = warn

        # Tracking variables
        self._update_count = 0
        self._best_mean_reward = -np.inf
        self._last_mean_reward = None
        self._last_std_reward = None
        self._evaluations: List[Dict[str, Any]] = []

    @property
    def best_mean_reward(self) -> float:
        """Get the best mean reward achieved during evaluation.

        Returns:
            The best mean reward, or -inf if no evaluation has been run.
        """
        return self._best_mean_reward

    @property
    def last_mean_reward(self) -> Optional[float]:
        """Get the mean reward from the last evaluation.

        Returns:
            The last mean reward, or None if no evaluation has been run.
        """
        return self._last_mean_reward

    @property
    def last_std_reward(self) -> Optional[float]:
        """Get the std of rewards from the last evaluation.

        Returns:
            The last std reward, or None if no evaluation has been run.
        """
        return self._last_std_reward

    def on_training_start(self, trainer: "BaseTrainer") -> None:
        """Called at the start of training.

        Creates the best model save directory if specified.

        Args:
            trainer: The trainer instance.
        """
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
            if self.verbose:
                print(f"EvalCallback: Best models will be saved to {self.best_model_save_path}")

        if self.verbose:
            print(f"EvalCallback: Evaluation frequency = {self.eval_freq} updates")
            print(f"EvalCallback: Episodes per evaluation = {self.n_eval_episodes}")
            print(f"EvalCallback: Deterministic mode = {self.deterministic}")

    def on_update_end(
        self,
        trainer: "BaseTrainer",
        update_metrics: Dict[str, float]
    ) -> None:
        """Called after each parameter update.

        Triggers evaluation if the update count is a multiple of eval_freq.

        Args:
            trainer: The trainer instance.
            update_metrics: Dictionary of metrics from the update step.
        """
        self._update_count += 1

        # Check if we should run evaluation
        if self._update_count % self.eval_freq == 0:
            self._run_evaluation(trainer, update_metrics)

    def _run_evaluation(
        self,
        trainer: "BaseTrainer",
        update_metrics: Dict[str, float]
    ) -> None:
        """Run evaluation on the current model.

        Args:
            trainer: The trainer instance.
            update_metrics: Dictionary of metrics from the update step.
        """
        if self.verbose:
            print(f"\nEvalCallback: Running evaluation at update {self._update_count}...")

        # Run evaluation episodes
        episode_rewards = self._evaluate_episodes(trainer)

        if len(episode_rewards) == 0:
            if self.warn:
                print("EvalCallback: Warning - No episodes completed during evaluation")
            return

        # Compute statistics
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        self._last_mean_reward = mean_reward
        self._last_std_reward = std_reward

        # Store evaluation results
        eval_result = {
            "update": self._update_count,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_episodes": len(episode_rewards),
            "episode_rewards": episode_rewards,
        }
        self._evaluations.append(eval_result)

        # Print results
        if self.verbose:
            print(f"EvalCallback: Evaluation results:")
            print(f"  Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")
            print(f"  Episodes completed: {len(episode_rewards)}")
            if len(episode_rewards) > 0:
                print(f"  Min reward: {min(episode_rewards):.4f}")
                print(f"  Max reward: {max(episode_rewards):.4f}")

        # Check if this is the best model
        if mean_reward > self._best_mean_reward:
            self._best_mean_reward = mean_reward
            if self.verbose:
                print(f"  New best mean reward: {mean_reward:.4f}!")

            # Save best model
            if self.best_model_save_path is not None:
                self._save_best_model(trainer)

    def _evaluate_episodes(self, trainer: "BaseTrainer") -> List[float]:
        """Run evaluation episodes and collect rewards.

        Args:
            trainer: The trainer instance.

        Returns:
            List of total episode rewards.
        """
        episode_rewards = []

        # Check if trainer has algorithm and algorithm_state
        if not hasattr(trainer, 'algorithm') or trainer.algorithm is None:
            if self.warn:
                print("EvalCallback: Warning - No algorithm found in trainer")
            return episode_rewards

        if not hasattr(trainer, 'algorithm_state') or trainer.algorithm_state is None:
            if self.warn:
                print("EvalCallback: Warning - No algorithm_state found in trainer")
            return episode_rewards

        algorithm = trainer.algorithm
        algorithm_state = trainer.algorithm_state

        # Get a random key for evaluation
        key = getattr(algorithm_state, 'rng', None)

        # Check if key is a valid JAX array
        is_jax_key = False
        if key is not None:
            try:
                import jax
                if isinstance(key, jax.Array):
                    is_jax_key = True
            except (ImportError, AttributeError):
                pass

        if not is_jax_key:
            try:
                import jax.random as random
                key = random.PRNGKey(0)
                is_jax_key = True
            except (ImportError, RuntimeError):
                key = None

        for ep in range(self.n_eval_episodes):
            # Reset environment
            obs, state = self.eval_env.reset()

            # Split key for this episode if using JAX
            ep_key = None
            if is_jax_key and key is not None:
                try:
                    import jax.random as random
                    key, ep_key = random.split(key)
                except Exception:
                    ep_key = None

            done = False
            episode_reward = 0.0
            step_count = 0
            max_steps = 1000  # Prevent infinite loops

            while not done and step_count < max_steps:
                # Get actions from algorithm
                try:
                    # Try to get actions using the algorithm
                    if hasattr(algorithm, 'compute_action'):
                        action_result = algorithm.compute_action(
                            algorithm_state,
                            obs,
                            deterministic=self.deterministic
                        )

                        # Handle different return types
                        if isinstance(action_result, tuple):
                            actions = action_result[0]
                        else:
                            actions = action_result

                        # Convert to numpy if needed
                        if hasattr(actions, 'device_buffer'):
                            actions = np.array(actions)
                    else:
                        # Fallback to random actions
                        sample_arg = ep_key if ep_key is not None else step_count
                        actions = self.eval_env.action_space().sample(sample_arg)

                except Exception as e:
                    if self.warn:
                        print(f"EvalCallback: Error computing action: {e}")
                    # Fallback to random actions
                    actions = self.eval_env.action_space().sample(step_count)

                # Step environment
                try:
                    step_result = self.eval_env.step(state, actions)

                    # Handle different return formats
                    if len(step_result) == 5:
                        obs, state, rewards, dones, infos = step_result
                    elif len(step_result) == 4:
                        obs, state, rewards, dones = step_result
                        infos = {}
                    else:
                        obs, state = step_result[0], step_result[1]
                        rewards = step_result[2] if len(step_result) > 2 else 0
                        dones = step_result[3] if len(step_result) > 3 else False
                        infos = {}

                    # Sum rewards across agents (or handle single agent case)
                    if isinstance(rewards, dict):
                        episode_reward += sum(rewards.values())
                    elif isinstance(rewards, (list, np.ndarray)):
                        episode_reward += float(np.sum(rewards))
                    else:
                        episode_reward += float(rewards)

                    # Check if episode is done
                    if isinstance(dones, dict):
                        done = any(dones.values())
                    elif isinstance(dones, (list, np.ndarray)):
                        done = bool(np.any(dones))
                    else:
                        done = bool(dones)

                    step_count += 1

                except Exception as e:
                    if self.warn:
                        print(f"EvalCallback: Error stepping environment: {e}")
                    break

            if step_count > 0:
                episode_rewards.append(episode_reward)

        return episode_rewards

    def _save_best_model(self, trainer: "BaseTrainer") -> None:
        """Save the best model.

        Args:
            trainer: The trainer instance.
        """
        if trainer.algorithm is None or trainer.algorithm_state is None:
            return

        checkpoint_path = os.path.join(
            self.best_model_save_path,
            "best_model.pkl"
        )

        try:
            trainer.algorithm.save(trainer.algorithm_state, checkpoint_path)
            if self.verbose:
                print(f"  Saved best model to: {checkpoint_path}")
        except Exception as e:
            if self.warn:
                print(f"EvalCallback: Error saving best model: {e}")

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get the history of all evaluations.

        Returns:
            List of evaluation result dictionaries, each containing:
            - update: Update step when evaluation was run
            - mean_reward: Mean episode reward
            - std_reward: Standard deviation of episode rewards
            - n_episodes: Number of episodes completed
            - episode_rewards: List of individual episode rewards
        """
        return self._evaluations.copy()

    def get_update_count(self) -> int:
        """Get the current update count.

        Returns:
            The number of updates that have occurred.
        """
        return self._update_count

    def reset(self) -> None:
        """Reset the callback state.

        This can be called when restarting training to reset counters and history.
        """
        self._update_count = 0
        self._best_mean_reward = -np.inf
        self._last_mean_reward = None
        self._last_std_reward = None
        self._evaluations = []
