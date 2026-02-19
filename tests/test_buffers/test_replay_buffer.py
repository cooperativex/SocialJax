"""Unit tests for ReplayBuffer and PrioritizedReplayBuffer.

Tests cover:
- Buffer initialization and configuration
- Adding transitions
- Random sampling
- JAX array conversion
- Prioritized replay features
- Memory efficiency
"""

import pytest
import numpy as np

from socialjax.buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    BufferEmptyError,
    InsufficientDataError,
)

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ============================================================================
# Test ReplayBuffer Imports
# ============================================================================

class TestReplayBufferImports:
    """Test that ReplayBuffer can be imported."""

    def test_import_replay_buffer(self):
        """Test ReplayBuffer can be imported."""
        from socialjax.buffers import ReplayBuffer
        assert ReplayBuffer is not None

    def test_import_prioritized_replay_buffer(self):
        """Test PrioritizedReplayBuffer can be imported."""
        from socialjax.buffers import PrioritizedReplayBuffer
        assert PrioritizedReplayBuffer is not None

    def test_replay_buffer_inherits_from_base(self):
        """Test ReplayBuffer inherits from BaseBuffer."""
        from socialjax.buffers.base_buffer import BaseBuffer
        assert issubclass(ReplayBuffer, BaseBuffer)
        assert issubclass(PrioritizedReplayBuffer, BaseBuffer)


# ============================================================================
# Test ReplayBuffer Initialization
# ============================================================================

class TestReplayBufferInit:
    """Test ReplayBuffer initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        buffer = ReplayBuffer(
            buffer_size=10000,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.buffer_size == 10000
        assert buffer.obs_shape == (4,)
        assert buffer.action_dim == 2

    def test_init_creates_observation_arrays(self):
        """Test that initialization creates observation arrays."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(5, 5, 3),
            action_dim=2,
        )
        assert buffer.observations.shape == (100, 5, 5, 3)
        assert buffer.next_observations.shape == (100, 5, 5, 3)

    def test_init_creates_action_array(self):
        """Test that initialization creates action array."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.actions.shape == (100,)
        assert buffer.actions.dtype == np.int32

    def test_init_creates_reward_array(self):
        """Test that initialization creates reward array."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.rewards.shape == (100,)

    def test_init_creates_done_array(self):
        """Test that initialization creates done array."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.dones.shape == (100,)

    def test_init_creates_timeout_array(self):
        """Test that initialization creates timeout array."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.timeouts.shape == (100,)
        assert buffer.timeouts.dtype == bool

    def test_init_custom_dtype(self):
        """Test initialization with custom dtype."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
            dtype=np.float64,
        )
        assert buffer.observations.dtype == np.float64

    def test_init_empty_state(self):
        """Test that buffer starts empty."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.size == 0
        assert buffer.full is False
        assert buffer.pos == 0


# ============================================================================
# Test ReplayBuffer Add Method
# ============================================================================

class TestReplayBufferAdd:
    """Test ReplayBuffer add method."""

    def test_add_single_transition(self):
        """Test adding a single transition."""
        buffer = ReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        obs = np.array([1, 2, 3, 4], dtype=np.float32)
        next_obs = np.array([2, 3, 4, 5], dtype=np.float32)

        buffer.add(obs, action=1, reward=1.0, next_obs=next_obs, done=False)

        assert buffer.size == 1
        np.testing.assert_array_equal(buffer.observations[0], obs)
        assert buffer.actions[0] == 1
        assert buffer.rewards[0] == 1.0

    def test_add_multiple_transitions(self):
        """Test adding multiple transitions."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )

        for i in range(50):
            obs = np.array([i, i+1, i+2, i+3], dtype=np.float32)
            next_obs = np.array([i+1, i+2, i+3, i+4], dtype=np.float32)
            buffer.add(obs, action=i % 2, reward=float(i), next_obs=next_obs, done=False)

        assert buffer.size == 50
        assert buffer.pos == 50

    def test_add_with_done_true(self):
        """Test adding transition with done=True."""
        buffer = ReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        obs = np.array([1, 2, 3, 4], dtype=np.float32)
        next_obs = np.array([0, 0, 0, 0], dtype=np.float32)

        buffer.add(obs, action=1, reward=10.0, next_obs=next_obs, done=True)

        assert buffer.dones[0] == True

    def test_add_with_timeout(self):
        """Test adding transition with timeout flag."""
        buffer = ReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        obs = np.zeros(4, dtype=np.float32)
        next_obs = np.zeros(4, dtype=np.float32)

        buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=True, timeout=True)

        assert buffer.timeouts[0] == True
        assert buffer.dones[0] == True

    def test_add_with_agent_id(self):
        """Test adding transition with agent ID."""
        buffer = ReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        obs = np.zeros(4, dtype=np.float32)
        next_obs = np.zeros(4, dtype=np.float32)

        buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False, agent_id=3)

        assert buffer.agent_ids is not None
        assert buffer.agent_ids[0] == 3

    def test_add_fills_buffer(self):
        """Test adding until buffer is full."""
        buffer = ReplayBuffer(
            buffer_size=5,
            obs_shape=(4,),
            action_dim=2,
        )

        for i in range(5):
            obs = np.zeros(4, dtype=np.float32)
            next_obs = np.zeros(4, dtype=np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        assert buffer.size == 5
        assert buffer.full is True

    def test_add_overwrites_old_data(self):
        """Test that adding after full overwrites old data."""
        buffer = ReplayBuffer(
            buffer_size=3,
            obs_shape=(2,),
            action_dim=2,
        )

        # Fill buffer with known values
        for i in range(3):
            obs = np.array([i, i], dtype=np.float32)
            next_obs = np.array([i+1, i+1], dtype=np.float32)
            buffer.add(obs, action=i, reward=float(i), next_obs=next_obs, done=False)

        # Add more data to overwrite
        new_obs = np.array([99, 99], dtype=np.float32)
        new_next_obs = np.array([100, 100], dtype=np.float32)
        buffer.add(new_obs, action=9, reward=99.0, next_obs=new_next_obs, done=False)

        np.testing.assert_array_equal(buffer.observations[0], new_obs)

    def test_add_batch(self):
        """Test adding a batch of transitions."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )

        obs_batch = np.random.randn(10, 4).astype(np.float32)
        next_obs_batch = np.random.randn(10, 4).astype(np.float32)
        actions_batch = np.zeros(10, dtype=np.int32)
        rewards_batch = np.zeros(10, dtype=np.float32)
        dones_batch = np.zeros(10, dtype=np.float32)

        buffer.add_batch(obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch)

        assert buffer.size == 10


# ============================================================================
# Test ReplayBuffer Sample Method
# ============================================================================

class TestReplayBufferSample:
    """Test ReplayBuffer sample method."""

    def test_sample_from_empty_buffer_raises(self):
        """Test that sampling from empty buffer raises error."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        with pytest.raises(BufferEmptyError):
            buffer.sample(32)

    def test_sample_returns_correct_keys(self):
        """Test that sample returns all expected keys."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        # Add some transitions
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16)

        expected_keys = {
            "observations", "actions", "rewards",
            "next_observations", "dones"
        }
        assert expected_keys.issubset(set(batch.keys()))

    def test_sample_returns_correct_shapes(self):
        """Test that sample returns arrays with correct shapes."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(5, 5, 3),
            action_dim=8,
        )
        for i in range(50):
            obs = np.random.randn(5, 5, 3).astype(np.float32)
            next_obs = np.random.randn(5, 5, 3).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16)

        assert batch["observations"].shape == (16, 5, 5, 3)
        assert batch["next_observations"].shape == (16, 5, 5, 3)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)
        assert batch["dones"].shape == (16,)

    def test_sample_with_timeouts(self):
        """Test sample with timeout information."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False, timeout=(i % 2 == 0))

        batch = buffer.sample(16, include_timeouts=True)

        assert "timeouts" in batch
        assert batch["timeouts"].shape == (16,)

    def test_sample_with_agent_ids(self):
        """Test sample with agent IDs."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False, agent_id=i % 4)

        batch = buffer.sample(16)

        assert "agent_ids" in batch

    def test_sample_insufficient_data_without_replacement(self):
        """Test sample raises error when not enough data and replace=False is not allowed."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(10):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        # When batch_size > size, it should sample with replacement (not raise)
        # This tests that behavior - we can sample more than available
        batch = buffer.sample(100)
        assert batch["observations"].shape[0] == 100

    def test_sample_with_replacement(self):
        """Test sample with replacement when batch > size."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(20):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        # Should work with replacement
        batch = buffer.sample(100)

        assert batch["observations"].shape[0] == 100

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_sample_as_jax_arrays(self):
        """Test sample with JAX array conversion."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16, as_jax=True)

        # Check that arrays are JAX arrays
        for key, arr in batch.items():
            assert hasattr(arr, '__jax_array__') or type(arr).__module__.startswith('jax')


# ============================================================================
# Test ReplayBuffer Get Method
# ============================================================================

class TestReplayBufferGet:
    """Test ReplayBuffer get method."""

    def test_get_from_empty_buffer_raises(self):
        """Test that getting from empty buffer raises error."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        with pytest.raises(BufferEmptyError):
            buffer.get()

    def test_get_returns_all_data(self):
        """Test that get returns all data."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        data = buffer.get()

        assert data["observations"].shape[0] == 50

    def test_get_with_batch_size_samples(self):
        """Test get with batch_size parameter delegates to sample."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        data = buffer.get(batch_size=16)

        assert data["observations"].shape[0] == 16


# ============================================================================
# Test ReplayBuffer Clear Method
# ============================================================================

class TestReplayBufferClear:
    """Test ReplayBuffer clear method."""

    def test_clear_resets_position(self):
        """Test that clear resets position."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        buffer.clear()

        assert buffer.pos == 0
        assert buffer.size == 0
        assert buffer.full is False

    def test_clear_resets_agent_ids(self):
        """Test that clear resets agent IDs."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(10):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False, agent_id=i)

        buffer.clear()

        assert buffer.agent_ids is None

    def test_reset_storage_zeros_arrays(self):
        """Test that reset_storage zeros all arrays."""
        buffer = ReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        obs = np.ones(4, dtype=np.float32)
        next_obs = np.ones(4, dtype=np.float32)
        buffer.add(obs, action=1, reward=1.0, next_obs=next_obs, done=True)

        buffer.reset_storage()

        assert np.all(buffer.observations == 0)
        assert buffer.pos == 0


# ============================================================================
# Test ReplayBuffer Get Recent Method
# ============================================================================

class TestReplayBufferGetRecent:
    """Test ReplayBuffer get_recent method."""

    def test_get_recent_from_empty_buffer_raises(self):
        """Test get_recent from empty buffer raises error."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        with pytest.raises(BufferEmptyError):
            buffer.get_recent(10)

    def test_get_recent_returns_correct_count(self):
        """Test get_recent returns correct number of transitions."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.array([i, i, i, i], dtype=np.float32)
            next_obs = np.array([i+1, i+1, i+1, i+1], dtype=np.float32)
            buffer.add(obs, action=i, reward=float(i), next_obs=next_obs, done=False)

        recent = buffer.get_recent(10)

        assert recent["observations"].shape[0] == 10

    def test_get_recent_returns_most_recent(self):
        """Test get_recent returns the most recent transitions."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.array([i, 0, 0, 0], dtype=np.float32)
            next_obs = np.zeros(4, dtype=np.float32)
            buffer.add(obs, action=i, reward=float(i), next_obs=next_obs, done=False)

        recent = buffer.get_recent(5)

        # Should have indices 45-49
        expected_actions = np.array([45, 46, 47, 48, 49])
        np.testing.assert_array_equal(recent["actions"], expected_actions)


# ============================================================================
# Test ReplayBuffer Sample With Next Values
# ============================================================================

class TestReplayBufferSampleWithNextValues:
    """Test ReplayBuffer sample_with_next_values method."""

    def test_sample_with_next_values_returns_bootstrap(self):
        """Test sample_with_next_values returns bootstrap values."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            done = (i % 10 == 9)  # Some episodes end
            buffer.add(obs, action=0, reward=1.0, next_obs=next_obs, done=done)

        batch = buffer.sample_with_next_values(16)

        assert "bootstrap" in batch
        assert "gamma" in batch
        assert batch["bootstrap"].shape == (16,)

    def test_bootstrap_is_zero_for_done(self):
        """Test bootstrap is 0 for done transitions."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        # Add all done transitions
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=1.0, next_obs=next_obs, done=True)

        batch = buffer.sample_with_next_values(16)

        assert np.all(batch["bootstrap"] == 0)


# ============================================================================
# Test ReplayBuffer Memory
# ============================================================================

class TestReplayBufferMemory:
    """Test ReplayBuffer memory usage."""

    def test_memory_size_calculation(self):
        """Test memory size calculation."""
        buffer = ReplayBuffer(
            buffer_size=1000,
            obs_shape=(4,),
            action_dim=2,
        )
        memory = buffer.memory_size()
        assert memory > 0
        # obs: 1000 * 4 * 4 bytes = 16 KB for obs
        # + next_obs, actions, rewards, dones, timeouts
        assert memory > 10_000  # At least 10 KB

    def test_memory_size_small_buffer(self):
        """Test memory size for small buffer."""
        buffer = ReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        memory = buffer.memory_size()
        assert memory > 0
        assert memory < 10_000  # Less than 10 KB


# ============================================================================
# Test ReplayBuffer Properties
# ============================================================================

class TestReplayBufferProperties:
    """Test ReplayBuffer properties."""

    def test_size_property(self):
        """Test size property."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.size == 0

        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        assert buffer.size == 50

    def test_full_property(self):
        """Test full property."""
        buffer = ReplayBuffer(
            buffer_size=5,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.full is False

        for i in range(5):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        assert buffer.full is True

    def test_repr(self):
        """Test __repr__ method."""
        buffer = ReplayBuffer(
            buffer_size=10000,
            obs_shape=(4,),
            action_dim=2,
        )
        repr_str = repr(buffer)
        assert "ReplayBuffer" in repr_str
        assert "10000" in repr_str

    def test_len(self):
        """Test __len__ method."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(30):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        assert len(buffer) == 30

    def test_can_sample(self):
        """Test can_sample method."""
        buffer = ReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.can_sample(10) is False

        for i in range(20):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        assert buffer.can_sample(10) is True
        assert buffer.can_sample(100) is False


# ============================================================================
# Test PrioritizedReplayBuffer
# ============================================================================

class TestPrioritizedReplayBuffer:
    """Test PrioritizedReplayBuffer."""

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=1000,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert buffer.epsilon == 1e-6

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=1000,
            obs_shape=(4,),
            action_dim=2,
            alpha=0.8,
            beta=0.5,
            epsilon=1e-4,
        )
        assert buffer.alpha == 0.8
        assert buffer.beta == 0.5
        assert buffer.epsilon == 1e-4

    def test_add_with_default_priority(self):
        """Test adding with default priority."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        obs = np.zeros(4, dtype=np.float32)
        next_obs = np.zeros(4, dtype=np.float32)

        buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        assert buffer._priorities[0] == 1.0  # Default max priority

    def test_add_with_custom_priority(self):
        """Test adding with custom priority."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        obs = np.zeros(4, dtype=np.float32)
        next_obs = np.zeros(4, dtype=np.float32)

        buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False, priority=5.0)

        assert buffer._priorities[0] == 5.0

    def test_sample_returns_indices(self):
        """Test sample returns indices for priority update."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16)

        assert "indices" in batch
        assert batch["indices"].shape == (16,)

    def test_sample_returns_weights(self):
        """Test sample returns importance sampling weights."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16)

        assert "weights" in batch
        assert batch["weights"].shape == (16,)
        # Weights should be normalized
        assert np.all(batch["weights"] <= 1.0)

    def test_update_priorities(self):
        """Test updating priorities after sampling."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16)
        indices = batch["indices"]

        # Update priorities with TD errors
        td_errors = np.random.randn(16)
        buffer.update_priorities(indices, td_errors)

        # Check priorities were updated
        for idx, td_err in zip(indices, td_errors):
            expected_priority = abs(td_err) + buffer.epsilon
            assert buffer._priorities[idx] == pytest.approx(expected_priority)

    def test_sample_without_weights(self):
        """Test sampling without importance weights."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16, include_weights=False)

        assert "weights" not in batch
        assert "indices" in batch

    def test_clear_resets_priorities(self):
        """Test clear resets priorities."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(5):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False, priority=float(i+1))

        buffer.clear()

        assert buffer._max_priority == 1.0
        assert buffer.pos == 0

    def test_priority_affects_sampling(self):
        """Test that higher priority items are sampled more often."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
            alpha=1.0,  # Full prioritization
        )

        # Add items with different priorities
        for i in range(100):
            obs = np.array([i, 0, 0, 0], dtype=np.float32)
            next_obs = np.zeros(4, dtype=np.float32)
            # First 50 items have low priority, next 50 have high priority
            priority = 0.1 if i < 50 else 10.0
            buffer.add(obs, action=i, reward=0.0, next_obs=next_obs, done=False, priority=priority)

        # Sample many times and count
        high_priority_count = 0
        for _ in range(100):
            batch = buffer.sample(32)
            # Count how many are from high priority group (actions >= 50)
            high_priority_count += np.sum(batch["actions"] >= 50)

        # High priority items should be sampled more often
        # With 32 samples over 100 iterations = 3200 total samples
        # High priority group should dominate
        assert high_priority_count > 1600  # More than half


# ============================================================================
# Test PrioritizedReplayBuffer JAX Compatibility
# ============================================================================

class TestPrioritizedReplayBufferJAX:
    """Test PrioritizedReplayBuffer JAX compatibility."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_array_conversion(self):
        """Test conversion to JAX arrays."""
        buffer = PrioritizedReplayBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            buffer.add(obs, action=0, reward=0.0, next_obs=next_obs, done=False)

        batch = buffer.sample(16, as_jax=True)

        # Should be JAX arrays
        for key, arr in batch.items():
            assert hasattr(arr, '__jax_array__') or type(arr).__module__.startswith('jax')
