"""Evaluation metrics for multi-agent reinforcement learning.

This module provides metrics computation for evaluating multi-agent policies,
including episode returns, cooperation rates, and social welfare measures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected during a single evaluation episode.

    Attributes:
        episode_return: Total episode return (sum of all agent rewards)
        episode_length: Number of steps in the episode
        agent_returns: Dictionary mapping agent_id to total return
        agent_rewards: Dictionary mapping agent_id to list of per-step rewards
        cooperation_actions: Number of cooperative actions taken
        total_actions: Total number of actions taken
        info: Additional episode information
    """
    episode_return: float = 0.0
    episode_length: int = 0
    agent_returns: Dict[str, float] = field(default_factory=dict)
    agent_rewards: Dict[str, List[float]] = field(default_factory=dict)
    cooperation_actions: int = 0
    total_actions: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "agent_returns": self.agent_returns.copy(),
            "agent_rewards": {k: v.copy() for k, v in self.agent_rewards.items()},
            "cooperation_actions": self.cooperation_actions,
            "total_actions": self.total_actions,
            "info": self.info.copy(),
        }


@dataclass
class EvaluationMetrics:
    """Aggregated metrics across multiple evaluation episodes.

    Attributes:
        num_episodes: Number of episodes evaluated
        mean_return: Mean episode return across all episodes
        std_return: Standard deviation of episode returns
        min_return: Minimum episode return
        max_return: Maximum episode return
        mean_length: Mean episode length
        std_length: Standard deviation of episode lengths
        mean_agent_returns: Mean return per agent
        std_agent_returns: Standard deviation of returns per agent
        cooperation_rate: Fraction of cooperative actions (0.0 to 1.0)
        social_welfare: Sum of all agent returns
        gini_coefficient: Inequality measure (0 = equal, 1 = unequal)
        episode_returns: List of individual episode returns
        episode_lengths: List of individual episode lengths
    """
    num_episodes: int = 0
    mean_return: float = 0.0
    std_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0
    mean_length: float = 0.0
    std_length: float = 0.0
    mean_agent_returns: Dict[str, float] = field(default_factory=dict)
    std_agent_returns: Dict[str, float] = field(default_factory=dict)
    cooperation_rate: float = 0.0
    social_welfare: float = 0.0
    gini_coefficient: float = 0.0
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_episodes": self.num_episodes,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "mean_length": self.mean_length,
            "std_length": self.std_length,
            "mean_agent_returns": self.mean_agent_returns.copy(),
            "std_agent_returns": self.std_agent_returns.copy(),
            "cooperation_rate": self.cooperation_rate,
            "social_welfare": self.social_welfare,
            "gini_coefficient": self.gini_coefficient,
            "episode_returns": self.episode_returns.copy(),
            "episode_lengths": self.episode_lengths.copy(),
        }


def compute_episode_return(
    rewards: Union[Dict[str, float], List[float], np.ndarray],
) -> float:
    """Compute total episode return from rewards.

    Args:
        rewards: Either a dict mapping agent_id to reward, or a list/array of rewards

    Returns:
        Total sum of all rewards
    """
    if isinstance(rewards, dict):
        return sum(float(r) for k, r in rewards.items() if k != "__all__")
    elif isinstance(rewards, (list, np.ndarray)):
        return float(np.sum(rewards))
    else:
        return float(rewards)


def compute_agent_returns(
    episode_rewards: Dict[str, List[float]],
) -> Dict[str, float]:
    """Compute total return for each agent from per-step rewards.

    Args:
        episode_rewards: Dictionary mapping agent_id to list of rewards

    Returns:
        Dictionary mapping agent_id to total return
    """
    return {agent_id: sum(rewards) for agent_id, rewards in episode_rewards.items()}


def compute_cooperation_rate(
    cooperation_actions: int,
    total_actions: int,
) -> float:
    """Compute cooperation rate from action counts.

    Args:
        cooperation_actions: Number of cooperative actions
        total_actions: Total number of actions

    Returns:
        Cooperation rate as a fraction (0.0 to 1.0)
    """
    if total_actions == 0:
        return 0.0
    return cooperation_actions / total_actions


def compute_gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient for measuring inequality.

    A Gini coefficient of 0 indicates perfect equality, while 1 indicates
    maximum inequality.

    Args:
        values: List of values (e.g., agent returns)

    Returns:
        Gini coefficient (0.0 to 1.0)
    """
    if not values or len(values) < 2:
        return 0.0

    values = [float(v) for v in values]
    n = len(values)

    # Handle negative values by shifting
    min_val = min(values)
    if min_val < 0:
        values = [v - min_val for v in values]

    # Handle all zeros
    if sum(values) == 0:
        return 0.0

    # Sort values
    sorted_values = sorted(values)

    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_values)
    n_values = len(sorted_values)
    gini = (n_values + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n_values

    return float(max(0.0, min(1.0, gini)))


def compute_social_welfare(agent_returns: Dict[str, float]) -> float:
    """Compute social welfare as sum of all agent returns.

    Args:
        agent_returns: Dictionary mapping agent_id to return

    Returns:
        Total social welfare
    """
    return sum(agent_returns.values())


def aggregate_episode_metrics(
    episode_metrics: List[EpisodeMetrics],
) -> EvaluationMetrics:
    """Aggregate metrics from multiple episodes.

    Args:
        episode_metrics: List of EpisodeMetrics from individual episodes

    Returns:
        Aggregated EvaluationMetrics
    """
    if not episode_metrics:
        return EvaluationMetrics()

    num_episodes = len(episode_metrics)

    # Extract episode returns and lengths
    episode_returns = [em.episode_return for em in episode_metrics]
    episode_lengths = [em.episode_length for em in episode_metrics]

    # Compute aggregate statistics
    returns_array = np.array(episode_returns)
    lengths_array = np.array(episode_lengths)

    mean_return = float(np.mean(returns_array))
    std_return = float(np.std(returns_array))
    min_return = float(np.min(returns_array))
    max_return = float(np.max(returns_array))
    mean_length = float(np.mean(lengths_array))
    std_length = float(np.std(lengths_array))

    # Aggregate per-agent returns
    all_agent_ids = set()
    for em in episode_metrics:
        all_agent_ids.update(em.agent_returns.keys())

    mean_agent_returns = {}
    std_agent_returns = {}
    for agent_id in all_agent_ids:
        agent_returns = [
            em.agent_returns.get(agent_id, 0.0)
            for em in episode_metrics
        ]
        mean_agent_returns[agent_id] = float(np.mean(agent_returns))
        std_agent_returns[agent_id] = float(np.std(agent_returns))

    # Compute cooperation rate
    total_coop = sum(em.cooperation_actions for em in episode_metrics)
    total_actions = sum(em.total_actions for em in episode_metrics)
    cooperation_rate = compute_cooperation_rate(total_coop, total_actions)

    # Compute social welfare and Gini
    # Use mean agent returns for these metrics
    if mean_agent_returns:
        social_welfare = sum(mean_agent_returns.values())
        gini_coefficient = compute_gini_coefficient(list(mean_agent_returns.values()))
    else:
        social_welfare = mean_return
        gini_coefficient = 0.0

    return EvaluationMetrics(
        num_episodes=num_episodes,
        mean_return=mean_return,
        std_return=std_return,
        min_return=min_return,
        max_return=max_return,
        mean_length=mean_length,
        std_length=std_length,
        mean_agent_returns=mean_agent_returns,
        std_agent_returns=std_agent_returns,
        cooperation_rate=cooperation_rate,
        social_welfare=social_welfare,
        gini_coefficient=gini_coefficient,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )


def identify_cooperative_action(
    action: int,
    env_state: Any,
    env: Any,
    agent_id: str = None,
) -> bool:
    """Determine if an action is cooperative.

    This is a general implementation that can be overridden for specific
    environments. By default, it uses heuristics based on common
    multi-agent environment patterns.

    Args:
        action: The action taken
        env_state: Current environment state
        env: Environment instance
        agent_id: Optional agent identifier

    Returns:
        True if the action is considered cooperative
    """
    # Default implementation - override for specific environments
    # Many social dilemma environments have:
    # - Action 0 as "do nothing" or "wait" (often cooperative)
    # - Higher actions as more "selfish" actions

    # This is a placeholder - specific environments should override this
    return action == 0


def compute_metrics_from_episodes(
    episode_data: List[Dict[str, Any]],
    cooperation_threshold: float = 0.5,
) -> EvaluationMetrics:
    """Compute evaluation metrics from raw episode data.

    Args:
        episode_data: List of episode dictionaries containing:
            - rewards: Dict or list of rewards per step
            - actions: Optional list of actions for cooperation rate
            - length: Episode length
            - info: Additional info
        cooperation_threshold: Threshold for considering an action cooperative

    Returns:
        Aggregated EvaluationMetrics
    """
    episode_metrics = []

    for data in episode_data:
        rewards = data.get("rewards", {})
        actions = data.get("actions", [])
        length = data.get("length", 0)
        info = data.get("info", {})

        # Compute episode return
        if isinstance(rewards, list):
            episode_return = sum(rewards)
            agent_returns = {}
            agent_rewards = {}
        else:
            episode_return = compute_episode_return(rewards)
            agent_returns = compute_agent_returns(
                {k: [v] for k, v in rewards.items() if k != "__all__"}
            )
            agent_rewards = {k: [v] for k, v in rewards.items() if k != "__all__"}

        # Count cooperative actions (if action data available)
        cooperation_actions = 0
        total_actions = 0
        if actions:
            for step_actions in actions:
                if isinstance(step_actions, (list, np.ndarray)):
                    total_actions += len(step_actions)
                    # Simple heuristic: lower action values = more cooperative
                    for a in step_actions:
                        if a <= cooperation_threshold * max(step_actions) if max(step_actions) > 0 else True:
                            cooperation_actions += 1

        episode_metrics.append(EpisodeMetrics(
            episode_return=episode_return,
            episode_length=length,
            agent_returns=agent_returns,
            agent_rewards=agent_rewards,
            cooperation_actions=cooperation_actions,
            total_actions=total_actions,
            info=info,
        ))

    return aggregate_episode_metrics(episode_metrics)
