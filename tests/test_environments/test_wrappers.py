"""Unit tests for SocialJax environment wrappers.

Tests cover:
- BaseWrapper: Forwarding and unwrapping
- NormalizationWrapper: Observation and reward normalization
- FrameStackWrapper: Frame stacking
- TimeLimitWrapper: Episode time limits
- Wrapper chaining: Combining multiple wrappers
"""

import pytest
import sys
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional
from flax import struct
import chex

# Add socialjax to path
sys.path.insert(0, 'socialjax')

from socialjax.environments.multi_agent_env import MultiAgentEnv, State
from socialjax.environments.spaces import Discrete, Box


# Create a simple mock environment for testing
@struct.dataclass
class MockState(State):
    """Simple mock state for testing."""
    done: chex.Array
    step: int
    obs: chex.Array
    episode_return: chex.Array


class MockEnv(MultiAgentEnv):
    """Simple mock environment for testing wrappers.

    This environment has:
    - 3 agents
    - 2D observation space (4, 4, 3)
    - Discrete action space (5 actions)
    - Step counter and done signal
    """

    def __init__(self, num_agents: int = 3, obs_shape: Tuple[int, ...] = (4, 4, 3)):
        super().__init__(num_agents)
        self._obs_shape = obs_shape

        # Create observation and action spaces
        for i in range(num_agents):
            agent = f"agent_{i}"
            self.observation_spaces[agent] = Box(
                low=0.0, high=1.0, shape=obs_shape, dtype=jnp.float32
            )
            self.action_spaces[agent] = Discrete(5)

    @property
    def agent_classes(self) -> dict:
        return {"agent": [f"agent_{i}" for i in range(self.num_agents)]}

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], MockState]:
        """Reset to initial state."""
        obs = {}
        for i in range(self.num_agents):
            agent = f"agent_{i}"
            obs[agent] = jax.random.uniform(key, self._obs_shape, dtype=jnp.float32)

        state = MockState(
            done=jnp.array(False),
            step=0,
            obs=jax.random.uniform(key, self._obs_shape, dtype=jnp.float32),
            episode_return=jnp.array(0.0),
        )

        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: MockState,
        actions: Dict[str, chex.Array],
        timestep: int = 0
    ) -> Tuple[Dict[str, chex.Array], MockState, Dict[str, float], Dict[str, bool], Dict]:
        """Step the environment."""
        new_step = state.step + 1

        # Generate new observations
        obs = {}
        for i in range(self.num_agents):
            agent = f"agent_{i}"
            obs[agent] = jax.random.uniform(key, self._obs_shape, dtype=jnp.float32)

        # Generate rewards (1.0 per step)
        rewards = {f"agent_{i}": 1.0 for i in range(self.num_agents)}

        # Episode ends after 100 steps
        done = new_step >= 100
        dones = {f"agent_{i}": done for i in range(self.num_agents)}
        dones["__all__"] = done

        # Update episode return
        new_return = state.episode_return + 1.0

        new_state = MockState(
            done=jnp.array(done),
            step=new_step,
            obs=jax.random.uniform(key, self._obs_shape, dtype=jnp.float32),
            episode_return=new_return,
        )

        infos = {"episode_return": new_return}

        return obs, new_state, rewards, dones, infos

    def get_obs(self, state: MockState) -> Dict[str, chex.Array]:
        """Get observations from state."""
        return {f"agent_{i}": state.obs for i in range(self.num_agents)}

    def get_avail_actions(self, state: MockState) -> Dict[str, chex.Array]:
        """Get available actions (all actions available)."""
        return {
            f"agent_{i}": jnp.ones(5, dtype=jnp.int32)
            for i in range(self.num_agents)
        }


# ============= BaseWrapper Tests =============

class TestBaseWrapper:
    """Tests for BaseWrapper class."""

    def test_import(self):
        """Test that BaseWrapper can be imported."""
        from socialjax.environments.wrappers import BaseWrapper
        assert BaseWrapper is not None

    def test_wrap_environment(self):
        """Test wrapping an environment."""
        from socialjax.environments.wrappers import BaseWrapper

        env = MockEnv()
        wrapped = BaseWrapper(env)

        assert wrapped.env is env
        assert wrapped.num_agents == env.num_agents

    def test_forward_reset(self):
        """Test that reset is forwarded correctly."""
        from socialjax.environments.wrappers import BaseWrapper

        env = MockEnv()
        wrapped = BaseWrapper(env)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)

        assert isinstance(obs, dict)
        assert len(obs) == env.num_agents

    def test_forward_step(self):
        """Test that step is forwarded correctly."""
        from socialjax.environments.wrappers import BaseWrapper

        env = MockEnv()
        wrapped = BaseWrapper(env)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)

        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}
        obs2, state2, rewards, dones, infos = wrapped.step(key, state, actions)

        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)

    def test_unwrapped(self):
        """Test getting unwrapped environment."""
        from socialjax.environments.wrappers import BaseWrapper

        env = MockEnv()
        wrapped = BaseWrapper(env)

        assert wrapped.unwrapped() is env

    def test_nested_unwrapped(self):
        """Test unwrapping nested wrappers."""
        from socialjax.environments.wrappers import BaseWrapper

        env = MockEnv()
        wrapped1 = BaseWrapper(env)
        wrapped2 = BaseWrapper(wrapped1)
        wrapped3 = BaseWrapper(wrapped2)

        assert wrapped3.unwrapped() is env

    def test_name_property(self):
        """Test wrapper name property."""
        from socialjax.environments.wrappers import BaseWrapper

        env = MockEnv()
        wrapped = BaseWrapper(env)

        assert "BaseWrapper" in wrapped.name
        assert "MockEnv" in wrapped.name

    def test_repr(self):
        """Test string representation."""
        from socialjax.environments.wrappers import BaseWrapper

        env = MockEnv()
        wrapped = BaseWrapper(env)

        repr_str = repr(wrapped)
        assert "BaseWrapper" in repr_str


# ============= NormalizationWrapper Tests =============

class TestNormalizationWrapper:
    """Tests for NormalizationWrapper class."""

    def test_import(self):
        """Test that NormalizationWrapper can be imported."""
        from socialjax.environments.wrappers import NormalizationWrapper
        assert NormalizationWrapper is not None

    def test_wrap_environment(self):
        """Test wrapping an environment."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(env)

        assert wrapped.normalize_obs is True
        assert wrapped.normalize_reward is True
        assert wrapped.env is env

    def test_custom_parameters(self):
        """Test custom wrapper parameters."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(
            env,
            normalize_obs=False,
            normalize_reward=True,
            clip_obs=5.0,
            clip_reward=5.0,
        )

        assert wrapped.normalize_obs is False
        assert wrapped.normalize_reward is True
        assert wrapped.clip_obs == 5.0
        assert wrapped.clip_reward == 5.0

    def test_reset(self):
        """Test that reset initializes statistics."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(env)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)

        assert isinstance(obs, dict)
        assert hasattr(state, 'obs_stats')
        assert hasattr(state, 'reward_stats')

    def test_step_updates_stats(self):
        """Test that step updates running statistics."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(env)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        # Take a step
        obs2, state2, rewards, dones, infos = wrapped.step(key, state, actions)

        # Statistics should be updated
        assert state2.obs_stats.count > state.obs_stats.count

    def test_normalize_obs_only(self):
        """Test normalizing only observations."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(env, normalize_obs=True, normalize_reward=False)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        obs2, state2, rewards, dones, infos = wrapped.step(key, state, actions)

        # Rewards should be unchanged (not normalized)
        for agent in rewards:
            assert rewards[agent] == 1.0  # Original reward value

    def test_normalize_reward_only(self):
        """Test normalizing only rewards."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(env, normalize_obs=False, normalize_reward=True)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        obs2, state2, rewards, dones, infos = wrapped.step(key, state, actions)

        # Observations should be in original range [0, 1]
        for agent in obs2:
            assert jnp.all(obs2[agent] >= 0.0)
            assert jnp.all(obs2[agent] <= 1.0)

    def test_set_training_mode(self):
        """Test setting training mode."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(env)

        assert wrapped._training is True

        wrapped.set_training(False)
        assert wrapped._training is False

        wrapped.set_training(True)
        assert wrapped._training is True

    def test_get_stats(self):
        """Test getting normalization statistics."""
        from socialjax.environments.wrappers import NormalizationWrapper

        env = MockEnv()
        wrapped = NormalizationWrapper(env)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        stats = wrapped.get_stats(state)

        assert 'obs' in stats
        assert 'reward' in stats


# ============= FrameStackWrapper Tests =============

class TestFrameStackWrapper:
    """Tests for FrameStackWrapper class."""

    def test_import(self):
        """Test that FrameStackWrapper can be imported."""
        from socialjax.environments.wrappers import FrameStackWrapper
        assert FrameStackWrapper is not None

    def test_wrap_environment(self):
        """Test wrapping an environment."""
        from socialjax.environments.wrappers import FrameStackWrapper

        env = MockEnv()
        wrapped = FrameStackWrapper(env, num_frames=4)

        assert wrapped.num_frames == 4
        assert wrapped.env is env

    def test_invalid_num_frames(self):
        """Test that invalid num_frames raises error."""
        from socialjax.environments.wrappers import FrameStackWrapper

        env = MockEnv()

        with pytest.raises(ValueError):
            FrameStackWrapper(env, num_frames=0)

        with pytest.raises(ValueError):
            FrameStackWrapper(env, num_frames=-1)

    def test_reset_stacks_frames(self):
        """Test that reset stacks initial frames."""
        from socialjax.environments.wrappers import FrameStackWrapper

        env = MockEnv(obs_shape=(4, 4, 3))
        wrapped = FrameStackWrapper(env, num_frames=4)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)

        # Stacked shape should be (4, 4, 3, 4) - stacking along new dimension
        for agent in obs:
            assert obs[agent].shape == (4, 4, 3, 4)

    def test_step_updates_frames(self):
        """Test that step updates the frame buffer."""
        from socialjax.environments.wrappers import FrameStackWrapper

        env = MockEnv(obs_shape=(4, 4, 3))
        wrapped = FrameStackWrapper(env, num_frames=4)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        obs2, state2, rewards, dones, infos = wrapped.step(key, state, actions)

        # Frame buffer should be updated
        assert state2.frames is not None

    def test_stacked_obs_shape(self):
        """Test the stacked_obs_shape property."""
        from socialjax.environments.wrappers import FrameStackWrapper

        env = MockEnv(obs_shape=(4, 4, 3))
        wrapped = FrameStackWrapper(env, num_frames=4)

        # Shape stacks along new dimension
        assert wrapped.stacked_obs_shape == (4, 4, 12)

    def test_single_frame(self):
        """Test with num_frames=1 (no actual stacking)."""
        from socialjax.environments.wrappers import FrameStackWrapper

        env = MockEnv(obs_shape=(4, 4, 3))
        wrapped = FrameStackWrapper(env, num_frames=1)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)

        # With 1 frame, shape should be (4, 4, 3, 1) - stacking adds dimension
        for agent in obs:
            assert obs[agent].shape == (4, 4, 3, 1)


# ============= TimeLimitWrapper Tests =============

class TestTimeLimitWrapper:
    """Tests for TimeLimitWrapper class."""

    def test_import(self):
        """Test that TimeLimitWrapper can be imported."""
        from socialjax.environments.wrappers import TimeLimitWrapper
        assert TimeLimitWrapper is not None

    def test_wrap_environment(self):
        """Test wrapping an environment."""
        from socialjax.environments.wrappers import TimeLimitWrapper

        env = MockEnv()
        wrapped = TimeLimitWrapper(env, max_steps=100)

        assert wrapped.max_steps == 100
        assert wrapped.env is env

    def test_invalid_max_steps(self):
        """Test that invalid max_steps raises error."""
        from socialjax.environments.wrappers import TimeLimitWrapper

        env = MockEnv()

        with pytest.raises(ValueError):
            TimeLimitWrapper(env, max_steps=0)

        with pytest.raises(ValueError):
            TimeLimitWrapper(env, max_steps=-1)

    def test_reset_initializes_counter(self):
        """Test that reset initializes step counter."""
        from socialjax.environments.wrappers import TimeLimitWrapper

        env = MockEnv()
        wrapped = TimeLimitWrapper(env, max_steps=10)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)

        assert state.steps == 0

    def test_step_increments_counter(self):
        """Test that step increments the counter."""
        from socialjax.environments.wrappers import TimeLimitWrapper

        env = MockEnv()
        wrapped = TimeLimitWrapper(env, max_steps=10)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        obs2, state2, rewards, dones, infos = wrapped.step(key, state, actions)

        assert state2.steps == 1

    def test_time_limit_reached(self):
        """Test that time limit triggers done."""
        from socialjax.environments.wrappers import TimeLimitWrapper

        env = MockEnv()
        wrapped = TimeLimitWrapper(env, max_steps=3)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        # Take 3 steps
        for _ in range(3):
            obs, state, rewards, dones, infos = wrapped.step(key, state, actions)

        # Should be done due to time limit
        assert dones["__all__"] == True
        assert infos.get("TimeLimit.truncated", False) == True

    def test_time_limit_not_reached(self):
        """Test behavior before time limit."""
        from socialjax.environments.wrappers import TimeLimitWrapper

        env = MockEnv()
        wrapped = TimeLimitWrapper(env, max_steps=10)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        # Take 2 steps (less than limit)
        for _ in range(2):
            obs, state, rewards, dones, infos = wrapped.step(key, state, actions)

        # Should not be done yet
        assert dones["__all__"] == False
        assert infos.get("TimeLimit.truncated", False) == False

    def test_infos_contain_steps(self):
        """Test that infos contain step count."""
        from socialjax.environments.wrappers import TimeLimitWrapper

        env = MockEnv()
        wrapped = TimeLimitWrapper(env, max_steps=10)
        key = jax.random.PRNGKey(0)

        obs, state = wrapped.reset(key)
        actions = {f"agent_{i}": 0 for i in range(env.num_agents)}

        obs, state, rewards, dones, infos = wrapped.step(key, state, actions)

        assert "TimeLimit.steps" in infos
        assert infos["TimeLimit.steps"] == 1


# ============= Wrapper Chaining Tests =============

class TestWrapperChaining:
    """Tests for chaining multiple wrappers."""

    def test_chain_time_limit_and_normalization(self):
        """Test chaining TimeLimit and Normalization wrappers."""
        from socialjax.environments.wrappers import (
            TimeLimitWrapper,
            NormalizationWrapper,
        )

        env = MockEnv()
        env = TimeLimitWrapper(env, max_steps=10)
        env = NormalizationWrapper(env)

        assert env.env.max_steps == 10

    def test_chain_all_wrappers(self):
        """Test chaining all wrappers together."""
        from socialjax.environments.wrappers import (
            BaseWrapper,
            TimeLimitWrapper,
            NormalizationWrapper,
            FrameStackWrapper,
        )

        env = MockEnv(obs_shape=(4, 4, 3))
        env = FrameStackWrapper(env, num_frames=4)
        env = TimeLimitWrapper(env, max_steps=100)
        env = NormalizationWrapper(env)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Check that all wrappers are working
        assert obs is not None

    def test_unwrapped_through_chain(self):
        """Test unwrapping through multiple wrappers."""
        from socialjax.environments.wrappers import (
            TimeLimitWrapper,
            NormalizationWrapper,
            FrameStackWrapper,
        )

        base_env = MockEnv()
        env = FrameStackWrapper(base_env, num_frames=4)
        env = TimeLimitWrapper(env, max_steps=100)
        env = NormalizationWrapper(env)

        # Get base environment through all wrappers
        unwrapped = env.unwrapped()
        assert unwrapped is base_env


# ============= RunningMeanStd Tests =============

class TestRunningMeanStd:
    """Tests for RunningMeanStd dataclass."""

    def test_create(self):
        """Test creating RunningMeanStd."""
        from socialjax.environments.wrappers.normalization import RunningMeanStd

        stats = RunningMeanStd.create(shape=(4, 4, 3))

        assert stats.mean.shape == (4, 4, 3)
        assert stats.var.shape == (4, 4, 3)
        assert jnp.allclose(stats.mean, 0.0)
        assert jnp.allclose(stats.var, 1.0)

    def test_update_running_stats(self):
        """Test updating running statistics."""
        from socialjax.environments.wrappers.normalization import (
            RunningMeanStd,
            update_running_stats,
        )

        stats = RunningMeanStd.create(shape=(3,))
        batch = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        new_stats = update_running_stats(stats, batch)

        # Mean should be [2.5, 3.5, 4.5]
        expected_mean = jnp.array([2.5, 3.5, 4.5])
        assert jnp.allclose(new_stats.mean, expected_mean, atol=0.1)


# ============= Module Exports Tests =============

class TestModuleExports:
    """Tests for module exports."""

    def test_all_wrappers_exported(self):
        """Test that all wrappers are exported from the module."""
        from socialjax.environments import wrappers

        assert hasattr(wrappers, 'BaseWrapper')
        assert hasattr(wrappers, 'NormalizationWrapper')
        assert hasattr(wrappers, 'FrameStackWrapper')
        assert hasattr(wrappers, 'TimeLimitWrapper')

    def test_import_from_module(self):
        """Test importing wrappers from the module."""
        from socialjax.environments.wrappers import (
            BaseWrapper,
            NormalizationWrapper,
            FrameStackWrapper,
            TimeLimitWrapper,
        )

        assert BaseWrapper is not None
        assert NormalizationWrapper is not None
        assert FrameStackWrapper is not None
        assert TimeLimitWrapper is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
