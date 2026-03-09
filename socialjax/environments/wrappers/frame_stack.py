"""Frame stacking wrapper for SocialJax environments.

This module provides a wrapper that stacks multiple consecutive frames
to provide temporal information to the agent. This is particularly useful
for environments where motion information is important (e.g., games).
"""

from typing import Dict, Tuple, Optional, List
import jax
import jax.numpy as jnp
import chex
from functools import partial
from flax import struct

from socialjax.environments.multi_agent_env import MultiAgentEnv, State
from socialjax.environments.wrappers.base_wrapper import BaseWrapper


@struct.dataclass
class FrameStackState(State):
    """State for FrameStackWrapper.

    Attributes:
        done: Whether episode is done
        step: Current step count
        env_state: Wrapped environment state
        frames: Stacked frames for each agent
    """
    done: chex.Array = None  # type: ignore
    step: int = 0
    env_state: State = None  # type: ignore
    frames: Dict[str, chex.Array] = None  # type: ignore


class FrameStackWrapper(BaseWrapper):
    """Wrapper that stacks multiple consecutive frames.

    This wrapper stacks the last n frames together to provide temporal
    information. This is useful for environments where motion information
    is important, such as games or physics simulations.

    The stacked frames are concatenated along the last channel dimension.
    For example, with num_frames=4 and an observation shape of (84, 84, 3),
    the output shape will be (84, 84, 12).

    Example:
        >>> from socialjax import make
        >>> from socialjax.environments.wrappers import FrameStackWrapper
        >>> env = make('coin_game', num_agents=5)
        >>> env = FrameStackWrapper(env, num_frames=4)
        >>> obs, state = env.reset(key)
        >>> # obs now contains 4 stacked frames per agent

    Attributes:
        num_frames: Number of frames to stack
        _obs_shape: Original observation shape
        _stacked_obs_shape: Shape after stacking
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        num_frames: int = 4,
    ):
        """Initialize the frame stacking wrapper.

        Args:
            env: Environment to wrap
            num_frames: Number of frames to stack

        Raises:
            ValueError: If num_frames is less than 1
        """
        super().__init__(env)

        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")

        self.num_frames = num_frames

        # Get observation shape from the first agent
        first_agent = f"agent_{0}"
        if first_agent in env.observation_spaces:
            obs_space = env.observation_spaces[first_agent]
            if hasattr(obs_space, 'shape'):
                self._obs_shape = obs_space.shape
            else:
                self._obs_shape = ()
        else:
            self._obs_shape = ()

        # Calculate stacked observation shape
        # Assume stacking along the last dimension (channel dimension)
        if len(self._obs_shape) >= 3:
            # Image observation: stack along channel dimension
            self._stacked_obs_shape = (
                *self._obs_shape[:-1],
                self._obs_shape[-1] * num_frames,
            )
        elif len(self._obs_shape) == 2:
            # 2D observation: add channel dimension and stack
            self._stacked_obs_shape = (*self._obs_shape, num_frames)
        elif len(self._obs_shape) == 1:
            # 1D observation: stack along new dimension
            self._stacked_obs_shape = (*self._obs_shape, num_frames)
        else:
            # Scalar observation: stack along new dimension
            self._stacked_obs_shape = (num_frames,)

        # Update observation spaces
        for agent in self._observation_spaces:
            space = self._observation_spaces[agent]
            if hasattr(space, 'shape'):
                # Create a new space with stacked shape
                # This is a simple wrapper - actual Space implementation may vary
                space.shape = self._stacked_obs_shape

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], FrameStackState]:
        """Reset the environment and initialize frame buffer.

        Args:
            key: PRNG key

        Returns:
            Tuple of (stacked observations, state)
        """
        obs, env_state = self.env.reset(key)

        # Initialize frame buffer with the first observation repeated
        frames = {
            agent: jnp.stack([o] * self.num_frames, axis=-1)
            for agent, o in obs.items()
        }

        state = FrameStackState(
            done=jnp.array(False),
            step=0,
            env_state=env_state,
            frames=frames,
        )

        # Return stacked observation
        stacked_obs = self._stack_frames(frames)

        return stacked_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: FrameStackState,
        actions: Dict[str, chex.Array],
        timestep: int = 0,
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], FrameStackState, Dict[str, float], Dict[str, bool], Dict]:
        """Step the environment with frame stacking.

        Args:
            key: PRNG key
            state: Current frame stack state
            actions: Actions for each agent
            timestep: Current timestep
            reset_state: Optional state to reset to

        Returns:
            Tuple of (stacked obs, state, rewards, dones, infos)
        """
        # Step the wrapped environment
        obs, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions, timestep, reset_state
        )

        # Update frame buffer
        new_frames = self._update_frames(state.frames, obs, dones)

        new_state = FrameStackState(
            done=dones.get("__all__", jnp.array(False)),
            step=state.step + 1,
            env_state=env_state,
            frames=new_frames,
        )

        # Return stacked observation
        stacked_obs = self._stack_frames(new_frames)

        return stacked_obs, new_state, rewards, dones, infos

    def _update_frames(
        self,
        frames: Dict[str, chex.Array],
        new_obs: Dict[str, chex.Array],
        dones: Dict[str, bool],
    ) -> Dict[str, chex.Array]:
        """Update the frame buffer with new observations.

        On episode termination, the frame buffer is reset with the
        new observation repeated.

        Args:
            frames: Current frame buffer
            new_obs: New observations
            dones: Done flags for each agent

        Returns:
            Updated frame buffer
        """
        new_frames = {}
        for agent in frames:
            # Roll frames and add new observation
            old_frames = frames[agent]
            new_o = new_obs[agent]

            # Add channel dimension if needed
            if len(new_o.shape) < len(old_frames.shape):
                new_o = jnp.expand_dims(new_o, axis=-1)

            # Handle done: reset frame buffer on episode termination
            episode_done = dones.get("__all__", False)

            # Roll and append new frame
            rolled = jnp.roll(old_frames, shift=-1, axis=-1)
            updated = rolled.at[..., -1].set(new_o.squeeze(-1))

            # Reset on done
            reset_frames = jnp.stack([new_o.squeeze(-1)] * self.num_frames, axis=-1)
            new_frames[agent] = jax.lax.select(
                episode_done,
                reset_frames,
                updated,
            )

        return new_frames

    def _stack_frames(self, frames: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        """Stack frames into a single observation.

        Args:
            frames: Frame buffer

        Returns:
            Stacked observations
        """
        return frames

    def get_obs(self, state: FrameStackState) -> Dict[str, chex.Array]:
        """Get stacked observations from state.

        Args:
            state: Frame stack state

        Returns:
            Stacked observations
        """
        return self._stack_frames(state.frames)

    @property
    def stacked_obs_shape(self) -> Tuple[int, ...]:
        """Get the shape of stacked observations."""
        return self._stacked_obs_shape


class LazyFrames:
    """Lazy container for stacked frames.

    This class efficiently stores stacked frames without actually
    concatenating them until needed. This saves memory when frames
    are accessed multiple times.

    Attributes:
        frames: List of frame arrays
        axis: Axis to concatenate along
    """

    def __init__(self, frames: List[chex.Array], axis: int = -1):
        """Initialize LazyFrames.

        Args:
            frames: List of frames to stack
            axis: Axis to concatenate along
        """
        self.frames = frames
        self.axis = axis
        self._cached_array = None

    def __array__(self) -> chex.Array:
        """Convert to array, caching the result."""
        if self._cached_array is None:
            self._cached_array = jnp.concatenate(self.frames, axis=self.axis)
        return self._cached_array

    def __getitem__(self, index):
        """Get item from stacked array."""
        return self.__array__()[index]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the stacked array."""
        return self.__array__().shape
