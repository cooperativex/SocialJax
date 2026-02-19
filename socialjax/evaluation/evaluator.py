"""Evaluation system for trained multi-agent policies.

This module provides the Evaluator class for running evaluation episodes
and collecting metrics on trained policies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import time
import json

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from socialjax.evaluation.metrics import (
    EpisodeMetrics,
    EvaluationMetrics,
    compute_episode_return,
    aggregate_episode_metrics,
    identify_cooperative_action,
)


@dataclass
class EvaluatorConfig:
    """Configuration for the Evaluator.

    Attributes:
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        max_steps_per_episode: Maximum steps per episode (None for unlimited)
        seed: Random seed for evaluation
        capture_frames: Whether to capture frames during evaluation
        capture_frequency: How often to capture frames (1 = every step)
        verbose: Verbosity level (0=silent, 1=normal, 2=debug)
    """
    num_episodes: int = 10
    deterministic: bool = True
    max_steps_per_episode: Optional[int] = None
    seed: int = 42
    capture_frames: bool = False
    capture_frequency: int = 1
    verbose: int = 1


class Evaluator:
    """Evaluator for running evaluation episodes on trained policies.

    This class provides methods for evaluating trained multi-agent policies
    on environments and collecting comprehensive metrics.

    Example usage:
        ```python
        import socialjax
        from socialjax.evaluation import Evaluator

        # Create environment
        env = socialjax.make('clean_up', num_agents=7)

        # Create evaluator
        evaluator = Evaluator(
            env=env,
            algorithm=algorithm,
            config=EvaluatorConfig(num_episodes=50)
        )

        # Run evaluation
        metrics = evaluator.evaluate()

        # Print results
        print(f"Mean return: {metrics.mean_return:.2f} +/- {metrics.std_return:.2f}")
        print(f"Cooperation rate: {metrics.cooperation_rate:.2%}")
        ```
    """

    def __init__(
        self,
        env: Any,
        algorithm: Any,
        config: Optional[EvaluatorConfig] = None,
    ):
        """Initialize the Evaluator.

        Args:
            env: Environment to evaluate on
            algorithm: Algorithm/policy to evaluate
            config: Evaluator configuration (uses defaults if None)
        """
        self.env = env
        self.algorithm = algorithm
        self.config = config or EvaluatorConfig()

    def evaluate(
        self,
        algorithm_state: Optional[Any] = None,
        num_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> EvaluationMetrics:
        """Run evaluation episodes and return aggregated metrics.

        Args:
            algorithm_state: Optional algorithm state (uses current if None)
            num_episodes: Override config num_episodes
            deterministic: Override config deterministic
            seed: Override config seed

        Returns:
            Aggregated evaluation metrics
        """
        num_episodes = num_episodes or self.config.num_episodes
        deterministic = deterministic if deterministic is not None else self.config.deterministic
        seed = seed or self.config.seed

        if JAX_AVAILABLE:
            rng = jax.random.PRNGKey(seed)
        else:
            np.random.seed(seed)
            rng = None

        episode_metrics = []

        if self.config.verbose >= 1:
            print(f"Running {num_episodes} evaluation episodes...")

        for episode in range(num_episodes):
            if rng is not None:
                rng, episode_rng = jax.random.split(rng)
            else:
                episode_rng = None

            metrics = self._run_episode(
                algorithm_state=algorithm_state,
                rng=episode_rng,
                deterministic=deterministic,
                episode_num=episode,
            )
            episode_metrics.append(metrics)

            if self.config.verbose >= 1:
                print(f"  Episode {episode + 1}/{num_episodes}: "
                      f"Return={metrics.episode_return:.2f}, "
                      f"Length={metrics.episode_length}")

        return aggregate_episode_metrics(episode_metrics)

    def evaluate_with_frames(
        self,
        algorithm_state: Optional[Any] = None,
        num_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        seed: Optional[int] = None,
        capture_frequency: int = 1,
    ) -> Tuple[EvaluationMetrics, List[np.ndarray]]:
        """Run evaluation with frame capture.

        Args:
            algorithm_state: Optional algorithm state
            num_episodes: Override config num_episodes
            deterministic: Override config deterministic
            seed: Override config seed
            capture_frequency: Frame capture frequency

        Returns:
            Tuple of (metrics, frames)
        """
        num_episodes = num_episodes or self.config.num_episodes
        deterministic = deterministic if deterministic is not None else self.config.deterministic
        seed = seed or self.config.seed

        if JAX_AVAILABLE:
            rng = jax.random.PRNGKey(seed)
        else:
            np.random.seed(seed)
            rng = None

        episode_metrics = []
        all_frames = []

        for episode in range(num_episodes):
            if rng is not None:
                rng, episode_rng = jax.random.split(rng)
            else:
                episode_rng = None

            metrics, frames = self._run_episode_with_frames(
                algorithm_state=algorithm_state,
                rng=episode_rng,
                deterministic=deterministic,
                capture_frequency=capture_frequency,
            )
            episode_metrics.append(metrics)
            all_frames.extend(frames)

        return aggregate_episode_metrics(episode_metrics), all_frames

    def _run_episode(
        self,
        algorithm_state: Optional[Any],
        rng: Any,
        deterministic: bool,
        episode_num: int = 0,
    ) -> EpisodeMetrics:
        """Run a single evaluation episode.

        Args:
            algorithm_state: Algorithm state for policy
            rng: Random key for JAX
            deterministic: Whether to use deterministic actions
            episode_num: Episode number for logging

        Returns:
            EpisodeMetrics for this episode
        """
        # Reset environment
        if rng is not None:
            obs, env_state = self.env.reset(rng)
        else:
            obs, env_state = self.env.reset()

        episode_return = 0.0
        episode_length = 0
        agent_returns: Dict[str, float] = {}
        agent_rewards: Dict[str, List[float]] = {}
        cooperation_actions = 0
        total_actions = 0
        episode_rng = rng

        done = False
        max_steps = self.config.max_steps_per_episode

        while not done and (max_steps is None or episode_length < max_steps):
            # Get actions from policy
            actions = self._get_actions(
                obs=obs,
                algorithm_state=algorithm_state,
                rng=episode_rng,
                deterministic=deterministic,
            )

            # Count cooperative actions
            for agent_id, action in enumerate(actions):
                total_actions += 1
                if identify_cooperative_action(action, env_state, self.env, str(agent_id)):
                    cooperation_actions += 1

            # Step environment
            if JAX_AVAILABLE and episode_rng is not None:
                episode_rng, step_rng = jax.random.split(episode_rng)
            else:
                step_rng = None

            if step_rng is not None:
                step_result = self.env.step(step_rng, env_state, actions)
            else:
                step_result = self.env.step(env_state, actions)

            # Parse step result
            if len(step_result) == 4:
                next_obs, env_state, rewards, dones = step_result
                info = {}
            else:
                next_obs, env_state, rewards, dones, info = step_result

            # Track rewards
            step_return = compute_episode_return(rewards)
            episode_return += step_return

            if isinstance(rewards, dict):
                for agent_id, reward in rewards.items():
                    if agent_id != "__all__":
                        if agent_id not in agent_returns:
                            agent_returns[agent_id] = 0.0
                            agent_rewards[agent_id] = []
                        agent_returns[agent_id] += float(reward)
                        agent_rewards[agent_id].append(float(reward))
            elif isinstance(rewards, (list, np.ndarray)):
                for i, reward in enumerate(rewards):
                    agent_id = str(i)
                    if agent_id not in agent_returns:
                        agent_returns[agent_id] = 0.0
                        agent_rewards[agent_id] = []
                    agent_returns[agent_id] += float(reward)
                    agent_rewards[agent_id].append(float(reward))

            # Check done
            if isinstance(dones, dict):
                done = dones.get("__all__", False)
            else:
                done = bool(dones)

            episode_length += 1
            obs = next_obs

        return EpisodeMetrics(
            episode_return=episode_return,
            episode_length=episode_length,
            agent_returns=agent_returns,
            agent_rewards=agent_rewards,
            cooperation_actions=cooperation_actions,
            total_actions=total_actions,
            info={"episode_num": episode_num},
        )

    def _run_episode_with_frames(
        self,
        algorithm_state: Optional[Any],
        rng: Any,
        deterministic: bool,
        capture_frequency: int = 1,
    ) -> Tuple[EpisodeMetrics, List[np.ndarray]]:
        """Run a single evaluation episode with frame capture.

        Args:
            algorithm_state: Algorithm state for policy
            rng: Random key for JAX
            deterministic: Whether to use deterministic actions
            capture_frequency: How often to capture frames

        Returns:
            Tuple of (EpisodeMetrics, frames)
        """
        # Reset environment
        if rng is not None:
            obs, env_state = self.env.reset(rng)
        else:
            obs, env_state = self.env.reset()

        frames = []
        episode_return = 0.0
        episode_length = 0
        agent_returns: Dict[str, float] = {}
        agent_rewards: Dict[str, List[float]] = {}
        cooperation_actions = 0
        total_actions = 0
        episode_rng = rng

        done = False
        max_steps = self.config.max_steps_per_episode

        while not done and (max_steps is None or episode_length < max_steps):
            # Capture frame
            if episode_length % capture_frequency == 0:
                frame = self._capture_frame(env_state)
                if frame is not None:
                    frames.append(frame)

            # Get actions from policy
            actions = self._get_actions(
                obs=obs,
                algorithm_state=algorithm_state,
                rng=episode_rng,
                deterministic=deterministic,
            )

            # Count cooperative actions
            for agent_id, action in enumerate(actions):
                total_actions += 1
                if identify_cooperative_action(action, env_state, self.env, str(agent_id)):
                    cooperation_actions += 1

            # Step environment
            if JAX_AVAILABLE and episode_rng is not None:
                episode_rng, step_rng = jax.random.split(episode_rng)
            else:
                step_rng = None

            if step_rng is not None:
                step_result = self.env.step(step_rng, env_state, actions)
            else:
                step_result = self.env.step(env_state, actions)

            # Parse step result
            if len(step_result) == 4:
                next_obs, env_state, rewards, dones = step_result
                info = {}
            else:
                next_obs, env_state, rewards, dones, info = step_result

            # Track rewards
            step_return = compute_episode_return(rewards)
            episode_return += step_return

            if isinstance(rewards, dict):
                for agent_id, reward in rewards.items():
                    if agent_id != "__all__":
                        if agent_id not in agent_returns:
                            agent_returns[agent_id] = 0.0
                            agent_rewards[agent_id] = []
                        agent_returns[agent_id] += float(reward)
                        agent_rewards[agent_id].append(float(reward))
            elif isinstance(rewards, (list, np.ndarray)):
                for i, reward in enumerate(rewards):
                    agent_id = str(i)
                    if agent_id not in agent_returns:
                        agent_returns[agent_id] = 0.0
                        agent_rewards[agent_id] = []
                    agent_returns[agent_id] += float(reward)
                    agent_rewards[agent_id].append(float(reward))

            # Check done
            if isinstance(dones, dict):
                done = dones.get("__all__", False)
            else:
                done = bool(dones)

            episode_length += 1
            obs = next_obs

        # Capture final frame
        frame = self._capture_frame(env_state)
        if frame is not None:
            frames.append(frame)

        metrics = EpisodeMetrics(
            episode_return=episode_return,
            episode_length=episode_length,
            agent_returns=agent_returns,
            agent_rewards=agent_rewards,
            cooperation_actions=cooperation_actions,
            total_actions=total_actions,
            info={},
        )

        return metrics, frames

    def _get_actions(
        self,
        obs: Any,
        algorithm_state: Optional[Any],
        rng: Any,
        deterministic: bool,
    ) -> List[Any]:
        """Get actions from the policy.

        Args:
            obs: Current observations
            algorithm_state: Algorithm state
            rng: Random key
            deterministic: Whether to use deterministic actions

        Returns:
            List of actions for all agents
        """
        actions = []

        # Handle different observation formats
        if isinstance(obs, dict):
            agent_obs = [obs.get(agent) for agent in self.env.agents]
        elif isinstance(obs, (list, np.ndarray)):
            agent_obs = obs
        else:
            agent_obs = [obs]

        for i, agent_ob in enumerate(agent_obs):
            if algorithm_state is not None:
                if rng is not None:
                    rng, action_rng = jax.random.split(rng)
                else:
                    action_rng = None

                try:
                    action, _ = self.algorithm.compute_action(
                        algorithm_state,
                        agent_ob,
                        action_rng,
                        deterministic=deterministic,
                    )
                except Exception:
                    # Fallback to random action
                    action = self._get_random_action(action_rng)

            else:
                # No algorithm state, use random actions
                if rng is not None:
                    rng, action_rng = jax.random.split(rng)
                else:
                    action_rng = None
                action = self._get_random_action(action_rng)

            actions.append(np.array(action) if hasattr(action, '__array__') else action)

        return actions

    def _get_random_action(self, rng: Any) -> int:
        """Get a random action.

        Args:
            rng: Random key

        Returns:
            Random action
        """
        try:
            action_space = self.env.action_space()
            if hasattr(action_space, 'n'):
                n_actions = action_space.n
            else:
                n_actions = 8  # Default for many SocialJax environments

            if rng is not None and JAX_AVAILABLE:
                return int(jax.random.randint(rng, (), 0, n_actions))
            else:
                return np.random.randint(0, n_actions)
        except Exception:
            return 0

    def _capture_frame(self, env_state: Any) -> Optional[np.ndarray]:
        """Capture a frame from the environment.

        Args:
            env_state: Current environment state

        Returns:
            Frame as numpy array or None if rendering not available
        """
        if not hasattr(self.env, 'render'):
            return None

        try:
            frame = self.env.render(env_state)
            if isinstance(frame, np.ndarray):
                return frame
            return None
        except Exception:
            return None


def save_evaluation_results(
    metrics: EvaluationMetrics,
    output_path: str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Save evaluation results to JSON file.

    Args:
        metrics: Evaluation metrics to save
        output_path: Path to save JSON file
        additional_info: Optional additional information to include

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = metrics.to_dict()

    if additional_info:
        results["additional_info"] = additional_info

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return str(output_path)


def load_evaluation_results(input_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary of evaluation results
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def print_evaluation_summary(metrics: EvaluationMetrics, verbose: int = 1):
    """Print a formatted summary of evaluation metrics.

    Args:
        metrics: Evaluation metrics to print
        verbose: Verbosity level
    """
    if verbose < 1:
        return

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes:          {metrics.num_episodes}")
    print(f"Mean Return:       {metrics.mean_return:.4f} +/- {metrics.std_return:.4f}")
    print(f"Min Return:        {metrics.min_return:.4f}")
    print(f"Max Return:        {metrics.max_return:.4f}")
    print(f"Mean Length:       {metrics.mean_length:.1f} +/- {metrics.std_length:.1f}")
    print(f"Cooperation Rate:  {metrics.cooperation_rate:.2%}")
    print(f"Social Welfare:    {metrics.social_welfare:.4f}")
    print(f"Gini Coefficient:  {metrics.gini_coefficient:.4f}")

    if verbose >= 2 and metrics.mean_agent_returns:
        print(f"\nPer-Agent Returns:")
        for agent_id, mean_ret in metrics.mean_agent_returns.items():
            std_ret = metrics.std_agent_returns.get(agent_id, 0.0)
            print(f"  {agent_id}: {mean_ret:.4f} +/- {std_ret:.4f}")

    print("=" * 60)
