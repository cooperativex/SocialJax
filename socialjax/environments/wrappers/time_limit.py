"""Time limit wrapper for SocialJax environments.

This module provides a wrapper that enforces a maximum episode length.
When the episode reaches the time limit, it is terminated and a
truncation signal is added to the info dictionary.
"""

from typing import Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import chex
from functools import partial
from flax import struct

from socialjax.environments.multi_agent_env import MultiAgentEnv, State
from socialjax.environments.wrappers.base_wrapper import BaseWrapper


@struct.dataclass
class TimeLimitState(State):
    """State for TimeLimitWrapper.

    Attributes:
        done: Whether episode is done
        step: Current step count (alias for steps for State compatibility)
        env_state: Wrapped environment state
        steps: Number of steps taken in current episode
    """
    done: chex.Array = None  # type: ignore
    step: int = 0
    env_state: State = None  # type: ignore
    steps: chex.Array = None  # type: ignore


class TimeLimitWrapper(BaseWrapper):
    """Wrapper that enforces a maximum episode length.

    This wrapper terminates episodes after a fixed number of steps,
    adding a truncation signal to the info dictionary. This is useful
    for ensuring consistent episode lengths during training.

    The wrapper distinguishes between truncation (reached time limit)
    and termination (environment-specific termination condition).

    Example:
        >>> from socialjax import make
        >>> from socialjax.environments.wrappers import TimeLimitWrapper
        >>> env = make('coin_game', num_agents=5)
        >>> env = TimeLimitWrapper(env, max_steps=1000)
        >>> obs, state = env.reset(key)
        >>> # Episode will end after 1000 steps max

    Attributes:
        max_steps: Maximum number of steps per episode
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        max_steps: int = 1000,
    ):
        """Initialize the time limit wrapper.

        Args:
            env: Environment to wrap
            max_steps: Maximum number of steps per episode

        Raises:
            ValueError: If max_steps is less than 1
        """
        super().__init__(env)

        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        self.max_steps = max_steps

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], TimeLimitState]:
        """Reset the environment and step counter.

        Args:
            key: PRNG key

        Returns:
            Tuple of (observations, state with step counter = 0)
        """
        obs, env_state = self.env.reset(key)

        state = TimeLimitState(
            done=jnp.array(False),
            step=0,
            env_state=env_state,
            steps=jnp.array(0, dtype=jnp.int32),
        )

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: TimeLimitState,
        actions: Dict[str, chex.Array],
        timestep: int = 0,
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], TimeLimitState, Dict[str, float], Dict[str, bool], Dict]:
        """Step the environment with time limit enforcement.

        Args:
            key: PRNG key
            state: Current time limit state
            actions: Actions for each agent
            timestep: Current timestep (unused, kept for API compatibility)
            reset_state: Optional state to reset to

        Returns:
            Tuple of (obs, state, rewards, dones with time limit, infos with truncation)
        """
        # Step the wrapped environment
        obs, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions, timestep, reset_state
        )

        # Increment step counter
        new_steps = state.steps + 1

        # Check if we've reached the time limit
        time_limit_reached = new_steps >= self.max_steps

        # Update dones to include time limit
        # If time limit is reached, all agents are done
        new_dones = {}
        for agent, done in dones.items():
            if agent == "__all__":
                new_dones[agent] = done | time_limit_reached
            else:
                new_dones[agent] = done | time_limit_reached

        # Add truncation info
        new_infos = dict(infos)
        new_infos["TimeLimit.truncated"] = time_limit_reached & (~dones.get("__all__", False))
        new_infos["TimeLimit.steps"] = new_steps

        new_state = TimeLimitState(
            done=new_dones.get("__all__", jnp.array(False)),
            step=0,  # Use steps array for JIT compatibility
            env_state=env_state,
            steps=new_steps,
        )

        return obs, new_state, rewards, new_dones, new_infos

    def step_env(
        self,
        key: chex.PRNGKey,
        state: TimeLimitState,
        actions: Dict[str, chex.Array],
        timestep: int = 0
    ) -> Tuple[Dict[str, chex.Array], TimeLimitState, Dict[str, float], Dict[str, bool], Dict]:
        """Step without auto-reset, but with time limit enforcement.

        Args:
            key: PRNG key
            state: Current time limit state
            actions: Actions for each agent
            timestep: Current timestep

        Returns:
            Tuple of (obs, state, rewards, dones, infos)
        """
        obs, env_state, rewards, dones, infos = self.env.step_env(
            key, state.env_state, actions, timestep
        )

        new_steps = state.steps + 1
        time_limit_reached = new_steps >= self.max_steps

        new_dones = {}
        for agent, done in dones.items():
            if agent == "__all__":
                new_dones[agent] = done | time_limit_reached
            else:
                new_dones[agent] = done | time_limit_reached

        new_infos = dict(infos)
        new_infos["TimeLimit.truncated"] = time_limit_reached & (~dones.get("__all__", False))
        new_infos["TimeLimit.steps"] = new_steps

        new_state = TimeLimitState(
            done=new_dones.get("__all__", jnp.array(False)),
            step=0,  # Use steps array for JIT compatibility
            env_state=env_state,
            steps=new_steps,
        )

        return obs, new_state, rewards, new_dones, new_infos

    def get_obs(self, state: TimeLimitState) -> Dict[str, chex.Array]:
        """Get observations from state.

        Args:
            state: Time limit state

        Returns:
            Observations from the wrapped environment
        """
        return self.env.get_obs(state.env_state)

    @property
    def current_steps(self) -> int:
        """Get the current step count (requires state access)."""
        return 0  # Placeholder, actual value is in state
