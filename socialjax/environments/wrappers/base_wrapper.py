"""Base wrapper class for SocialJax environments.

This module provides the BaseWrapper class which serves as the foundation
for all environment wrappers. It follows the decorator pattern to wrap
MultiAgentEnv instances and modify their behavior.
"""

from typing import Dict, Tuple, Optional, Any, Union
import jax
import jax.numpy as jnp
import chex
from functools import partial
from flax import struct

from socialjax.environments.multi_agent_env import MultiAgentEnv, State


@struct.dataclass
class WrapperState(State):
    """Extended state for wrapper environments.

    Attributes:
        env_state: The wrapped environment's state
    """
    env_state: State = None  # type: ignore


class BaseWrapper(MultiAgentEnv):
    """Base class for all environment wrappers.

    This class wraps a MultiAgentEnv and forwards all calls to it by default.
    Subclasses should override specific methods to modify behavior.

    The wrapper follows the decorator pattern and maintains a reference to
    the wrapped environment. All attribute access is forwarded to the wrapped
    environment unless explicitly overridden.

    Example:
        >>> from socialjax import make
        >>> from socialjax.environments.wrappers import BaseWrapper
        >>> env = make('coin_game', num_agents=5)
        >>> wrapped = BaseWrapper(env)
        >>> # wrapped behaves exactly like env

    Attributes:
        env: The wrapped environment
        _observation_spaces: Cached observation spaces
        _action_spaces: Cached action spaces
    """

    def __init__(self, env: MultiAgentEnv):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap
        """
        super().__init__(num_agents=env.num_agents)
        self.env = env
        self._observation_spaces = env.observation_spaces.copy()
        self._action_spaces = env.action_spaces.copy()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset the wrapped environment.

        Args:
            key: PRNG key for random number generation

        Returns:
            Tuple of (observations, state)
        """
        return self.env.reset(key)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        timestep: int = 0,
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Step the wrapped environment.

        Args:
            key: PRNG key for random number generation
            state: Current environment state
            actions: Dictionary of actions for each agent
            timestep: Current timestep
            reset_state: Optional state to reset to

        Returns:
            Tuple of (observations, state, rewards, dones, infos)
        """
        return self.env.step(key, state, actions, timestep, reset_state)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        timestep: int = 0
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Step the wrapped environment without auto-reset.

        Args:
            key: PRNG key for random number generation
            state: Current environment state
            actions: Dictionary of actions for each agent
            timestep: Current timestep

        Returns:
            Tuple of (observations, state, rewards, dones, infos)
        """
        return self.env.step_env(key, state, actions, timestep)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Get observations from state.

        Args:
            state: Environment state

        Returns:
            Dictionary of observations for each agent
        """
        return self.env.get_obs(state)

    def observation_space(self, agent: str):
        """Get observation space for an agent.

        Args:
            agent: Agent identifier

        Returns:
            Observation space for the agent
        """
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        """Get action space for an agent.

        Args:
            agent: Agent identifier

        Returns:
            Action space for the agent
        """
        return self._action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Get available actions for each agent.

        Args:
            state: Environment state

        Returns:
            Dictionary of available action masks for each agent
        """
        return self.env.get_avail_actions(state)

    @property
    def name(self) -> str:
        """Get the wrapper name."""
        return f"{type(self).__name__}<{self.env.name}>"

    @property
    def agent_classes(self) -> dict:
        """Get agent classes from wrapped environment."""
        return self.env.agent_classes

    def unwrapped(self) -> MultiAgentEnv:
        """Get the underlying unwrapped environment.

        Recursively unwraps all wrappers to return the base environment.

        Returns:
            The underlying MultiAgentEnv
        """
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment.

        Args:
            name: Attribute name to access

        Returns:
            The attribute from the wrapped environment
        """
        # Avoid infinite recursion for internal attributes
        if name in ['env', '_observation_spaces', '_action_spaces']:
            return super().__getattribute__(name)

        return getattr(self.env, name)

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"<{type(self).__name__}: {self.env}>"
