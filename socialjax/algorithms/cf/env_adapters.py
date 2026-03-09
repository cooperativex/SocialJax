"""
Module: Environment Adapters for CF Algorithm
Equation: N/A (Infrastructure)

Provides unified interface for different SocialJax environments to work with
the Counterfactual Regret algorithm. Handles observation space normalization,
action space conversion, and environment-specific configurations.

Supported Environments:
- coin_game: 2-4 agents, 7 actions, (11, 11, 14) obs
- clean_up: 5-7 agents, 8 actions, (11, 11, 19) obs
- harvest_common_open: 5-7 agents, 8 actions, (11, 11, 15) obs

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Any, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import chex


@dataclass
class CFEnvSpec:
    """Specification for a CF-compatible environment."""
    env_name: str
    num_agents: int
    action_dim: int
    obs_shape: Tuple[int, ...]
    num_inner_steps: int
    num_outer_steps: int
    default_alpha: float  # Suggested alpha = N-1


class BaseCFAdapter(ABC):
    """
    Abstract base class for CF environment adapters.

    Provides a unified interface for different SocialJax environments to work
    with the Counterfactual Regret algorithm. Subclasses handle environment-specific
    observation and action space conversions.

    Attributes:
        env: The wrapped SocialJax environment
        spec: Environment specification
    """

    def __init__(
        self,
        env: Any,
        env_name: str,
        num_agents: int,
        action_dim: int,
        num_inner_steps: int = 1000,
        num_outer_steps: int = 1,
    ):
        """
        Initialize the adapter.

        Args:
            env: SocialJax environment instance
            env_name: Name of the environment
            num_agents: Number of agents
            action_dim: Number of discrete actions
            num_inner_steps: Steps per episode
            num_outer_steps: Number of outer episodes
        """
        self.env = env
        self.env_name = env_name
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

        # Get observation shape from environment
        _, self.obs_shape = env.observation_space()

        # Compute default alpha = N-1 (paper suggestion)
        self.default_alpha = float(num_agents - 1)

        # Create spec
        self.spec = CFEnvSpec(
            env_name=env_name,
            num_agents=num_agents,
            action_dim=action_dim,
            obs_shape=self.obs_shape,
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            default_alpha=self.default_alpha,
        )

    @abstractmethod
    def preprocess_obs(
        self,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Preprocess observations for the CF algorithm.

        Args:
            obs: Raw observations from environment [num_agents, H, W, C] or [H, W, C]

        Returns:
            Preprocessed observations ready for CF algorithm
        """
        pass

    @abstractmethod
    def convert_actions(
        self,
        actions: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """
        Convert actions from policy format to environment format.

        Args:
            actions: Actions from policy [num_agents] or [batch, num_agents]

        Returns:
            Actions in environment format (list of per-agent actions)
        """
        pass

    @abstractmethod
    def process_rewards(
        self,
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Process rewards from environment for CF algorithm.

        Args:
            rewards: Raw rewards from environment

        Returns:
            Processed rewards [num_agents] or [batch, num_agents]
        """
        pass

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[jnp.ndarray, Any]:
        """
        Reset the environment and return preprocessed observations.

        Args:
            key: JAX random key

        Returns:
            (preprocessed_obs, env_state)
        """
        obs, state = self.env.reset(key)
        obs = self.preprocess_obs(obs)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: Any,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, Dict[str, bool], Dict]:
        """
        Step the environment with preprocessed outputs.

        Args:
            key: JAX random key
            state: Current environment state
            actions: Actions from policy [num_agents]

        Returns:
            (preprocessed_obs, new_state, processed_rewards, dones, infos)
        """
        # Convert actions to environment format
        env_actions = self.convert_actions(actions)

        # Step environment
        obs, new_state, rewards, dones, infos = self.env.step(
            key, state, env_actions
        )

        # Preprocess outputs
        obs = self.preprocess_obs(obs)
        rewards = self.process_rewards(rewards)

        return obs, new_state, rewards, dones, infos

    def get_spec(self) -> CFEnvSpec:
        """Get environment specification."""
        return self.spec

    @property
    def observation_space(self) -> Tuple[int, ...]:
        """Observation space shape."""
        return self.obs_shape

    @property
    def action_space(self) -> int:
        """Action space dimension."""
        return self.action_dim


class CoinGameCFAdapter(BaseCFAdapter):
    """
    CF Adapter for Coin Game environment.

    Coin Game features:
    - 2-4 agents (typically 2 or 3)
    - 7 actions: turn_left, turn_right, left, right, up, down, stay
    - Observation shape: (11, 11, 14)
    - Social dilemma: agents can cooperate or defect when collecting coins
    """

    def __init__(
        self,
        num_agents: int = 3,
        num_inner_steps: int = 1000,
        num_outer_steps: int = 1,
        shared_rewards: bool = True,
        payoff_matrix: List[List[int]] = None,
        **kwargs,
    ):
        """
        Initialize Coin Game adapter.

        Args:
            num_agents: Number of agents (default: 3)
            num_inner_steps: Steps per episode (default: 1000)
            num_outer_steps: Number of outer episodes (default: 1)
            shared_rewards: Whether to use shared rewards (default: True)
            payoff_matrix: Reward matrix for coin collection
            **kwargs: Additional environment arguments
        """
        import socialjax

        # Default payoff matrix for coin game
        if payoff_matrix is None:
            payoff_matrix = [[1, 1, -2], [1, 1, -2]]

        env = socialjax.make(
            'coin_game',
            num_agents=num_agents,
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            shared_rewards=shared_rewards,
            payoff_matrix=payoff_matrix,
            **kwargs,
        )

        super().__init__(
            env=env,
            env_name='coin_game',
            num_agents=num_agents,
            action_dim=7,  # 7 actions in coin game
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
        )

        self.shared_rewards = shared_rewards
        self.payoff_matrix = payoff_matrix

    def preprocess_obs(
        self,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Preprocess Coin Game observations.

        Args:
            obs: Raw observations [num_agents, H, W, C]

        Returns:
            Preprocessed observations [num_agents, H, W, C]
            - Normalized to float32
            - Already in correct format from environment
        """
        # Ensure float32 and normalize if needed
        obs = jnp.asarray(obs, dtype=jnp.float32)

        # Coin game observations are already properly formatted
        # Shape should be [num_agents, 11, 11, 14]
        return obs

    def convert_actions(
        self,
        actions: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """
        Convert actions to Coin Game format.

        Args:
            actions: Actions from policy [num_agents]

        Returns:
            List of actions per agent
        """
        # Coin game expects list of actions
        return [actions[i] for i in range(self.num_agents)]

    def process_rewards(
        self,
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Process Coin Game rewards.

        Args:
            rewards: Raw rewards from environment [num_agents] or scalar

        Returns:
            Processed rewards [num_agents] as float32
        """
        rewards = jnp.asarray(rewards, dtype=jnp.float32)

        # Ensure shape is [num_agents]
        if rewards.ndim == 0:
            rewards = jnp.broadcast_to(rewards, (self.num_agents,))

        return rewards


class CleanupCFAdapter(BaseCFAdapter):
    """
    CF Adapter for Clean Up environment.

    Clean Up features:
    - 5-7 agents (typically 5)
    - 8 actions: turn_left, turn_right, forward, backward, stay, fire, etc.
    - Observation shape: (11, 11, 19)
    - Social dilemma: agents must clean pollution to spawn apples
    """

    def __init__(
        self,
        num_agents: int = 5,
        num_inner_steps: int = 1000,
        num_outer_steps: int = 1,
        **kwargs,
    ):
        """
        Initialize Clean Up adapter.

        Args:
            num_agents: Number of agents (default: 5)
            num_inner_steps: Steps per episode (default: 1000)
            num_outer_steps: Number of outer episodes (default: 1)
            **kwargs: Additional environment arguments
        """
        import socialjax

        env = socialjax.make(
            'clean_up',
            num_agents=num_agents,
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            **kwargs,
        )

        super().__init__(
            env=env,
            env_name='clean_up',
            num_agents=num_agents,
            action_dim=8,  # 8 actions in clean up
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
        )

    def preprocess_obs(
        self,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Preprocess Clean Up observations.

        Args:
            obs: Raw observations [num_agents, H, W, C]

        Returns:
            Preprocessed observations [num_agents, H, W, C]
        """
        obs = jnp.asarray(obs, dtype=jnp.float32)
        return obs

    def convert_actions(
        self,
        actions: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """
        Convert actions to Clean Up format.

        Args:
            actions: Actions from policy [num_agents]

        Returns:
            List of actions per agent
        """
        return [actions[i] for i in range(self.num_agents)]

    def process_rewards(
        self,
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Process Clean Up rewards.

        Args:
            rewards: Raw rewards from environment

        Returns:
            Processed rewards [num_agents] as float32
        """
        rewards = jnp.asarray(rewards, dtype=jnp.float32)

        if rewards.ndim == 0:
            rewards = jnp.broadcast_to(rewards, (self.num_agents,))

        return rewards


class HarvestCommonCFAdapter(BaseCFAdapter):
    """
    CF Adapter for Harvest (Commons) Open environment.

    Harvest features:
    - 5-7 agents (typically 5)
    - 8 actions: movement + fire/interact
    - Observation shape: (11, 11, 15)
    - Social dilemma: over-harvesting depletes resources
    """

    def __init__(
        self,
        num_agents: int = 5,
        num_inner_steps: int = 1000,
        num_outer_steps: int = 1,
        **kwargs,
    ):
        """
        Initialize Harvest Common adapter.

        Args:
            num_agents: Number of agents (default: 5)
            num_inner_steps: Steps per episode (default: 1000)
            num_outer_steps: Number of outer episodes (default: 1)
            **kwargs: Additional environment arguments
        """
        import socialjax

        env = socialjax.make(
            'harvest_common_open',
            num_agents=num_agents,
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            **kwargs,
        )

        super().__init__(
            env=env,
            env_name='harvest_common_open',
            num_agents=num_agents,
            action_dim=8,  # 8 actions in harvest
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
        )

    def preprocess_obs(
        self,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Preprocess Harvest observations.

        Args:
            obs: Raw observations [num_agents, H, W, C]

        Returns:
            Preprocessed observations [num_agents, H, W, C]
        """
        obs = jnp.asarray(obs, dtype=jnp.float32)
        return obs

    def convert_actions(
        self,
        actions: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """
        Convert actions to Harvest format.

        Args:
            actions: Actions from policy [num_agents]

        Returns:
            List of actions per agent
        """
        return [actions[i] for i in range(self.num_agents)]

    def process_rewards(
        self,
        rewards: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Process Harvest rewards.

        Args:
            rewards: Raw rewards from environment

        Returns:
            Processed rewards [num_agents] as float32
        """
        rewards = jnp.asarray(rewards, dtype=jnp.float32)

        if rewards.ndim == 0:
            rewards = jnp.broadcast_to(rewards, (self.num_agents,))

        return rewards


# ============================================================================
# Factory Functions
# ============================================================================

# Registry of available adapters
ADAPTER_REGISTRY = {
    'coin_game': CoinGameCFAdapter,
    'clean_up': CleanupCFAdapter,
    'harvest_common_open': HarvestCommonCFAdapter,
}


def create_cf_adapter(
    env_name: str,
    num_agents: int,
    num_inner_steps: int = 1000,
    num_outer_steps: int = 1,
    **kwargs,
) -> BaseCFAdapter:
    """
    Factory function to create a CF adapter for a given environment.

    Args:
        env_name: Name of the environment
        num_agents: Number of agents
        num_inner_steps: Steps per episode (default: 1000)
        num_outer_steps: Number of outer episodes (default: 1)
        **kwargs: Additional environment arguments

    Returns:
        CF adapter instance

    Raises:
        ValueError: If environment is not supported
    """
    if env_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Environment '{env_name}' not supported. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )

    adapter_cls = ADAPTER_REGISTRY[env_name]
    return adapter_cls(
        num_agents=num_agents,
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        **kwargs,
    )


def get_adapter_for_env(env: Any) -> BaseCFAdapter:
    """
    Create a CF adapter from an existing environment instance.

    Args:
        env: Existing SocialJax environment

    Returns:
        CF adapter wrapping the environment

    Raises:
        ValueError: If environment type is not recognized
    """
    env_type_str = str(type(env)).lower()
    env_module = type(env).__module__.lower()

    # Try to match environment type based on module path
    if 'coin_game' in env_module or 'coin' in env_type_str:
        return CoinGameCFAdapter(num_agents=env.num_agents)
    elif 'cleanup' in env_module or 'clean_up' in env_module or 'clean' in env_type_str:
        return CleanupCFAdapter(num_agents=env.num_agents)
    elif 'harvest' in env_module or 'common_harvest' in env_module or 'harvest' in env_type_str:
        return HarvestCommonCFAdapter(num_agents=env.num_agents)
    else:
        raise ValueError(
            f"Could not determine adapter for environment type: {type(env)}"
        )


def list_available_adapters() -> List[str]:
    """
    List all available environment adapters.

    Returns:
        List of adapter names
    """
    return list(ADAPTER_REGISTRY.keys())


# ============================================================================
# Utility Functions
# ============================================================================

def get_env_spec(env_name: str, num_agents: int) -> CFEnvSpec:
    """
    Get environment specification without creating the full environment.

    Args:
        env_name: Name of the environment
        num_agents: Number of agents

    Returns:
        CFEnvSpec with environment details

    Raises:
        ValueError: If environment is not supported
    """
    # Define static specs for each environment type
    specs = {
        'coin_game': {
            'action_dim': 7,
            'obs_shape': (11, 11, 14),
        },
        'clean_up': {
            'action_dim': 8,
            'obs_shape': (11, 11, 19),
        },
        'harvest_common_open': {
            'action_dim': 8,
            'obs_shape': (11, 11, 15),
        },
    }

    if env_name not in specs:
        raise ValueError(
            f"Environment '{env_name}' not supported. "
            f"Available: {list(specs.keys())}"
        )

    spec_info = specs[env_name]

    return CFEnvSpec(
        env_name=env_name,
        num_agents=num_agents,
        action_dim=spec_info['action_dim'],
        obs_shape=spec_info['obs_shape'],
        num_inner_steps=1000,
        num_outer_steps=1,
        default_alpha=float(num_agents - 1),
    )


def verify_adapter_compatibility(
    adapter: BaseCFAdapter,
    expected_num_agents: Optional[int] = None,
    expected_action_dim: Optional[int] = None,
    expected_obs_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """
    Verify that an adapter meets expected specifications.

    Args:
        adapter: CF adapter to verify
        expected_num_agents: Expected number of agents
        expected_action_dim: Expected action dimension
        expected_obs_shape: Expected observation shape

    Returns:
        True if all checks pass

    Raises:
        AssertionError: If any check fails
    """
    if expected_num_agents is not None:
        assert adapter.num_agents == expected_num_agents, \
            f"Expected {expected_num_agents} agents, got {adapter.num_agents}"

    if expected_action_dim is not None:
        assert adapter.action_dim == expected_action_dim, \
            f"Expected action_dim {expected_action_dim}, got {adapter.action_dim}"

    if expected_obs_shape is not None:
        assert adapter.obs_shape == expected_obs_shape, \
            f"Expected obs_shape {expected_obs_shape}, got {adapter.obs_shape}"

    return True
