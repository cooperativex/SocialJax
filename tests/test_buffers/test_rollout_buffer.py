"""Unit tests for RolloutBuffer.

Tests cover:
- Buffer initialization and configuration
- Adding transitions
- Retrieving data
- JAX array conversion
- Advantage computation integration
- Memory efficiency
"""

import pytest
import numpy as np

from socialjax.buffers import RolloutBuffer, BufferEmptyError, InsufficientDataError

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ============================================================================
# Test RolloutBuffer Imports
# ============================================================================

class TestRolloutBufferImports:
    """Test that RolloutBuffer can be imported."""

    def test_import_rollout_buffer(self):
        """Test RolloutBuffer can be imported."""
        from socialjax.buffers import RolloutBuffer
        assert RolloutBuffer is not None

    def test_rollout_buffer_inherits_from_base(self):
        """Test RolloutBuffer inherits from BaseBuffer."""
        from socialjax.buffers.base_buffer import BaseBuffer
        assert issubclass(RolloutBuffer, BaseBuffer)


# ============================================================================
# Test RolloutBuffer Initialization
# ============================================================================

class TestRolloutBufferInit:
    """Test RolloutBuffer initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        buffer = RolloutBuffer(
            buffer_size=128,
            num_envs=8,
            obs_shape=(15, 15, 3),
            action_dim=8,
        )
        assert buffer.buffer_size == 128
        assert buffer.num_envs == 8
        assert buffer.obs_shape == (15, 15, 3)
        assert buffer.action_dim == 8

    def test_init_creates_observation_array(self):
        """Test that initialization creates observation array."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5, 5, 3),
            action_dim=2,
        )
        assert buffer.observations.shape == (10, 4, 5, 5, 3)
        assert buffer.observations.dtype == np.float32

    def test_init_creates_action_array(self):
        """Test that initialization creates action array."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.actions.shape == (10, 4)
        assert buffer.actions.dtype == np.int32

    def test_init_creates_reward_array(self):
        """Test that initialization creates reward array."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.rewards.shape == (10, 4)
        assert buffer.rewards.dtype == np.float32

    def test_init_creates_done_array(self):
        """Test that initialization creates done array."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.dones.shape == (10, 4)

    def test_init_creates_log_prob_array(self):
        """Test that initialization creates log probability array."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.log_probs.shape == (10, 4)

    def test_init_creates_value_array(self):
        """Test that initialization creates value array."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.values.shape == (10, 4)

    def test_init_creates_advantage_return_arrays(self):
        """Test that initialization creates advantage and return arrays."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.advantages.shape == (10, 4)
        assert buffer.returns.shape == (10, 4)

    def test_init_custom_dtype(self):
        """Test initialization with custom dtype."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
            dtype=np.float64,
        )
        assert buffer.observations.dtype == np.float64
        assert buffer.rewards.dtype == np.float64

    def test_init_empty_state(self):
        """Test that buffer starts empty."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.size == 0
        assert buffer.full is False
        assert buffer.pos == 0


# ============================================================================
# Test RolloutBuffer Add Method
# ============================================================================

class TestRolloutBufferAdd:
    """Test RolloutBuffer add method."""

    def test_add_single_step(self):
        """Test adding a single step."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        obs = np.random.randn(4, 5).astype(np.float32)
        action = np.array([0, 1, 0, 1])
        reward = np.array([1.0, 0.5, -0.5, 0.0])
        done = np.array([0, 0, 0, 1])
        log_prob = np.array([-0.5, -0.3, -0.7, -0.2])
        value = np.array([0.1, 0.2, -0.1, 0.0])

        buffer.add(obs, action, reward, done, log_prob, value)

        assert buffer.size == 1
        assert buffer.pos == 1
        np.testing.assert_array_equal(buffer.observations[0], obs)
        np.testing.assert_array_equal(buffer.actions[0], action)
        np.testing.assert_array_equal(buffer.rewards[0], reward)

    def test_add_multiple_steps(self):
        """Test adding multiple steps."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )

        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            action = np.array([0, 1, 0, 1])
            reward = np.array([1.0] * 4)
            done = np.zeros(4)
            log_prob = np.zeros(4)
            value = np.zeros(4)
            buffer.add(obs, action, reward, done, log_prob, value)

        assert buffer.size == 5
        assert buffer.pos == 5

    def test_add_fills_buffer(self):
        """Test adding until buffer is full."""
        buffer = RolloutBuffer(
            buffer_size=5,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )

        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            action = np.zeros(4, dtype=np.int32)
            reward = np.zeros(4)
            done = np.zeros(4)
            log_prob = np.zeros(4)
            value = np.zeros(4)
            buffer.add(obs, action, reward, done, log_prob, value)

        assert buffer.size == 5
        assert buffer.full is True
        assert buffer.pos == 0  # Wrapped around

    def test_add_overwrites_old_data(self):
        """Test that adding after full overwrites old data."""
        buffer = RolloutBuffer(
            buffer_size=3,
            num_envs=2,
            obs_shape=(2,),
            action_dim=2,
        )

        # Fill buffer with known values
        for i in range(3):
            obs = np.ones((2, 2)) * i
            action = np.array([i, i])
            reward = np.array([float(i), float(i)])
            done = np.zeros(2)
            log_prob = np.zeros(2)
            value = np.zeros(2)
            buffer.add(obs, action, reward, done, log_prob, value)

        # Add more data to overwrite
        obs_new = np.ones((2, 2)) * 99
        buffer.add(obs_new, np.array([9, 9]), np.array([9.0, 9.0]),
                   np.zeros(2), np.zeros(2), np.zeros(2))

        np.testing.assert_array_equal(buffer.observations[0], obs_new)

    def test_add_with_cnn_obs_shape(self):
        """Test adding with CNN observation shape."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=8,
            obs_shape=(15, 15, 3),
            action_dim=8,
        )
        obs = np.random.randn(8, 15, 15, 3).astype(np.float32)
        action = np.zeros(8, dtype=np.int32)
        reward = np.zeros(8)
        done = np.zeros(8)
        log_prob = np.zeros(8)
        value = np.zeros(8)

        buffer.add(obs, action, reward, done, log_prob, value)

        assert buffer.size == 1
        np.testing.assert_array_equal(buffer.observations[0], obs)


# ============================================================================
# Test RolloutBuffer Get Method
# ============================================================================

class TestRolloutBufferGet:
    """Test RolloutBuffer get method."""

    def test_get_from_empty_buffer_raises(self):
        """Test that getting from empty buffer raises error."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        with pytest.raises(BufferEmptyError):
            buffer.get()

    def test_get_returns_correct_keys(self):
        """Test that get returns all expected keys."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        obs = np.random.randn(4, 5).astype(np.float32)
        buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                   np.zeros(4), np.zeros(4))

        data = buffer.get()

        expected_keys = {
            "observations", "actions", "rewards", "dones",
            "log_probs", "values", "advantages", "returns"
        }
        assert set(data.keys()) == expected_keys

    def test_get_returns_correct_shapes(self):
        """Test that get returns arrays with correct shapes."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        data = buffer.get()

        assert data["observations"].shape == (5, 4, 5)
        assert data["actions"].shape == (5, 4)
        assert data["rewards"].shape == (5, 4)

    def test_get_without_advantages(self):
        """Test get without advantages and returns."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        obs = np.random.randn(4, 5).astype(np.float32)
        buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                   np.zeros(4), np.zeros(4))

        data = buffer.get(include_advantages=False)

        assert "advantages" not in data
        assert "returns" not in data

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_get_as_jax_arrays(self):
        """Test get with JAX array conversion."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        obs = np.random.randn(4, 5).astype(np.float32)
        buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                   np.zeros(4), np.zeros(4))

        data = buffer.get(as_jax=True)

        # Check that arrays are JAX arrays
        import jax.numpy as jnp
        for key, arr in data.items():
            assert isinstance(arr, jnp.ndarray) or hasattr(arr, '__jax_array__')


# ============================================================================
# Test RolloutBuffer Clear Method
# ============================================================================

class TestRolloutBufferClear:
    """Test RolloutBuffer clear method."""

    def test_clear_resets_position(self):
        """Test that clear resets position."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        buffer.clear()

        assert buffer.pos == 0
        assert buffer.size == 0
        assert buffer.full is False

    def test_clear_does_not_zero_arrays(self):
        """Test that clear doesn't zero arrays (for efficiency)."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        # Add some non-zero data
        obs = np.ones((4, 5)).astype(np.float32)
        buffer.add(obs, np.ones(4), np.ones(4), np.ones(4),
                   np.ones(4), np.ones(4))

        buffer.clear()

        # Data should still be there (just position reset)
        np.testing.assert_array_equal(buffer.observations[0], obs)

    def test_reset_storage_zeros_arrays(self):
        """Test that reset_storage zeros all arrays."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        obs = np.ones((4, 5)).astype(np.float32)
        buffer.add(obs, np.ones(4), np.ones(4), np.ones(4),
                   np.ones(4), np.ones(4))

        buffer.reset_storage()

        assert np.all(buffer.observations == 0)
        assert np.all(buffer.actions == 0)
        assert buffer.pos == 0


# ============================================================================
# Test RolloutBuffer Batch Operations
# ============================================================================

class TestRolloutBufferBatch:
    """Test RolloutBuffer batch operations."""

    def test_get_batch_returns_correct_size(self):
        """Test get_batch returns correct batch size."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        # Add enough steps so that size >= batch_size
        # can_sample uses self.size (number of steps), not samples
        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        # Request a batch size that's <= size and <= size * num_envs
        batch = buffer.get_batch(batch_size=4)

        assert batch["observations"].shape[0] == 4

    def test_get_batch_raises_on_empty(self):
        """Test get_batch raises on empty buffer."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        with pytest.raises(BufferEmptyError):
            buffer.get_batch(batch_size=8)

    def test_get_batch_raises_on_insufficient_data(self):
        """Test get_batch raises when not enough data."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        # Add only 2 steps = 8 samples
        for i in range(2):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        with pytest.raises(InsufficientDataError):
            buffer.get_batch(batch_size=100)

    def test_get_flattened_shape(self):
        """Test get_flattened returns correct shape."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        for i in range(3):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        flat = buffer.get_flattened()

        # 3 steps * 4 envs = 12 samples
        assert flat["observations"].shape == (12, 5)
        assert flat["actions"].shape == (12,)


# ============================================================================
# Test RolloutBuffer Advantage Setting
# ============================================================================

class TestRolloutBufferAdvantages:
    """Test RolloutBuffer advantage setting."""

    def test_set_advantages(self):
        """Test setting advantages and returns."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        advantages = np.random.randn(5, 4).astype(np.float32)
        returns = np.random.randn(5, 4).astype(np.float32)

        buffer.set_advantages(advantages, returns)

        np.testing.assert_array_equal(buffer.advantages[:5], advantages)
        np.testing.assert_array_equal(buffer.returns[:5], returns)

    def test_set_advantages_validates_shape(self):
        """Test that set_advantages validates shape."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )

        # Wrong num_envs
        advantages = np.random.randn(5, 8).astype(np.float32)
        returns = np.random.randn(5, 8).astype(np.float32)

        with pytest.raises(ValueError):
            buffer.set_advantages(advantages, returns)


# ============================================================================
# Test RolloutBuffer Memory
# ============================================================================

class TestRolloutBufferMemory:
    """Test RolloutBuffer memory usage."""

    def test_memory_size_calculation(self):
        """Test memory size calculation."""
        buffer = RolloutBuffer(
            buffer_size=100,
            num_envs=8,
            obs_shape=(15, 15, 3),
            action_dim=8,
        )
        memory = buffer.memory_size()
        assert memory > 0
        # Memory should be at least 2 MB for this buffer configuration
        # obs: 100 * 8 * 15 * 15 * 3 * 4 bytes + other arrays
        assert memory > 1_000_000  # At least 1 MB

    def test_memory_size_small_buffer(self):
        """Test memory size for small buffer."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=1,
            obs_shape=(4,),
            action_dim=2,
        )
        memory = buffer.memory_size()
        assert memory > 0
        assert memory < 1_000_000  # Less than 1 MB


# ============================================================================
# Test RolloutBuffer Properties
# ============================================================================

class TestRolloutBufferProperties:
    """Test RolloutBuffer properties."""

    def test_size_property(self):
        """Test size property."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.size == 0

        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        assert buffer.size == 5

    def test_full_property(self):
        """Test full property."""
        buffer = RolloutBuffer(
            buffer_size=3,
            num_envs=2,
            obs_shape=(2,),
            action_dim=2,
        )
        assert buffer.full is False

        for i in range(3):
            obs = np.random.randn(2, 2).astype(np.float32)
            buffer.add(obs, np.zeros(2), np.zeros(2), np.zeros(2),
                       np.zeros(2), np.zeros(2))

        assert buffer.full is True

    def test_repr(self):
        """Test __repr__ method."""
        buffer = RolloutBuffer(
            buffer_size=128,
            num_envs=8,
            obs_shape=(15, 15, 3),
            action_dim=8,
        )
        repr_str = repr(buffer)
        assert "RolloutBuffer" in repr_str
        assert "128" in repr_str

    def test_len(self):
        """Test __len__ method."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        assert len(buffer) == 5

    def test_can_sample(self):
        """Test can_sample method."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        assert buffer.can_sample(1) is False

        for i in range(5):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        assert buffer.can_sample(1) is True
        assert buffer.can_sample(20) is False  # 5 * 4 = 20 samples, but can_sample checks size not samples


# ============================================================================
# Test RolloutBuffer JAX Compatibility
# ============================================================================

class TestRolloutBufferJAX:
    """Test RolloutBuffer JAX compatibility."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_array_conversion(self):
        """Test conversion to JAX arrays."""
        import jax.numpy as jnp

        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        obs = np.random.randn(4, 5).astype(np.float32)
        buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                   np.zeros(4), np.zeros(4))

        data = buffer.get(as_jax=True)

        # Should be JAX arrays
        assert isinstance(data["observations"], jnp.ndarray) or type(data["observations"]).__module__ == 'jaxlib.xla_extension'

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_flattened_conversion(self):
        """Test flattened data conversion to JAX arrays."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_envs=4,
            obs_shape=(5,),
            action_dim=2,
        )
        for i in range(3):
            obs = np.random.randn(4, 5).astype(np.float32)
            buffer.add(obs, np.zeros(4), np.zeros(4), np.zeros(4),
                       np.zeros(4), np.zeros(4))

        flat = buffer.get_flattened(as_jax=True)

        assert flat["observations"].shape == (12, 5)
