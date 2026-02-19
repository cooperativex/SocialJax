"""Unit tests for BaseBuffer abstract class.

Tests cover:
- Abstract method requirements
- Initialization and validation
- Property access
- Exception classes
"""

import pytest
import numpy as np
from abc import ABC
from typing import Dict

from socialjax.buffers.base_buffer import (
    BaseBuffer,
    BufferError,
    BufferEmptyError,
    BufferFullError,
    InsufficientDataError,
)


# ============================================================================
# Concrete implementation for testing
# ============================================================================

class ConcreteBuffer(BaseBuffer):
    """Concrete buffer implementation for testing."""

    def __init__(self, buffer_size, obs_shape, action_dim, num_envs=1):
        super().__init__(buffer_size, obs_shape, action_dim, num_envs)
        self.data = []

    def add(self, item) -> None:
        """Add item to buffer."""
        self.data.append(item)
        self._pos = (self._pos + 1) % self.buffer_size
        if self._pos == 0:
            self._full = True

    def get(self) -> Dict[str, np.ndarray]:
        """Get all data."""
        return {"data": np.array(self.data[:self.size])}

    def clear(self) -> None:
        """Clear buffer."""
        self.data = []
        self._pos = 0
        self._full = False


# ============================================================================
# Test BaseBuffer Imports
# ============================================================================

class TestBaseBufferImports:
    """Test that buffer classes can be imported."""

    def test_import_base_buffer(self):
        """Test BaseBuffer can be imported."""
        from socialjax.buffers import BaseBuffer
        assert BaseBuffer is not None

    def test_import_buffer_exceptions(self):
        """Test buffer exceptions can be imported."""
        from socialjax.buffers import (
            BufferError,
            BufferEmptyError,
            BufferFullError,
            InsufficientDataError,
        )
        assert BufferError is not None
        assert BufferEmptyError is not None
        assert BufferFullError is not None
        assert InsufficientDataError is not None


# ============================================================================
# Test BaseBuffer Abstract Nature
# ============================================================================

class TestBaseBufferAbstract:
    """Test that BaseBuffer is abstract and requires implementation."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseBuffer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBuffer(buffer_size=100, obs_shape=(4,), action_dim=2)

    def test_is_abstract(self):
        """Test that BaseBuffer is an abstract class."""
        assert issubclass(BaseBuffer, ABC)

    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be created."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer is not None


# ============================================================================
# Test BaseBuffer Initialization
# ============================================================================

class TestBaseBufferInit:
    """Test BaseBuffer initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
            num_envs=8,
        )
        assert buffer.buffer_size == 100
        assert buffer.obs_shape == (4,)
        assert buffer.action_dim == 2
        assert buffer.num_envs == 8

    def test_init_with_tuple_obs_shape(self):
        """Test initialization with tuple observation shape."""
        buffer = ConcreteBuffer(
            buffer_size=128,
            obs_shape=(15, 15, 3),
            action_dim=8,
        )
        assert buffer.obs_shape == (15, 15, 3)

    def test_init_with_single_obs_shape(self):
        """Test initialization with single dimension obs shape."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.obs_shape == (4,)

    def test_init_default_num_envs(self):
        """Test that num_envs defaults to 1."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.num_envs == 1

    def test_init_invalid_buffer_size_zero(self):
        """Test that buffer_size=0 raises error."""
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            ConcreteBuffer(buffer_size=0, obs_shape=(4,), action_dim=2)

    def test_init_invalid_buffer_size_negative(self):
        """Test that negative buffer_size raises error."""
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            ConcreteBuffer(buffer_size=-10, obs_shape=(4,), action_dim=2)

    def test_init_invalid_num_envs_zero(self):
        """Test that num_envs=0 raises error."""
        with pytest.raises(ValueError, match="num_envs must be positive"):
            ConcreteBuffer(
                buffer_size=100,
                obs_shape=(4,),
                action_dim=2,
                num_envs=0,
            )

    def test_init_invalid_num_envs_negative(self):
        """Test that negative num_envs raises error."""
        with pytest.raises(ValueError, match="num_envs must be positive"):
            ConcreteBuffer(
                buffer_size=100,
                obs_shape=(4,),
                action_dim=2,
                num_envs=-1,
            )


# ============================================================================
# Test BaseBuffer Properties
# ============================================================================

class TestBaseBufferProperties:
    """Test BaseBuffer properties."""

    def test_size_empty_buffer(self):
        """Test size property on empty buffer."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.size == 0

    def test_size_after_add(self):
        """Test size property after adding items."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(5):
            buffer.add(i)
        assert buffer.size == 5

    def test_size_full_buffer(self):
        """Test size property when buffer is full."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(15):
            buffer.add(i)
        assert buffer.size == 10  # Max size

    def test_full_property_empty(self):
        """Test full property on empty buffer."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.full is False

    def test_full_property_partial(self):
        """Test full property with partial fill."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(5):
            buffer.add(i)
        assert buffer.full is False

    def test_full_property_full(self):
        """Test full property when buffer is full."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(10):
            buffer.add(i)
        assert buffer.full is True

    def test_pos_property(self):
        """Test pos property tracks write position."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.pos == 0
        buffer.add(1)
        assert buffer.pos == 1
        buffer.add(2)
        assert buffer.pos == 2

    def test_pos_property_wraps(self):
        """Test pos property wraps around."""
        buffer = ConcreteBuffer(
            buffer_size=5,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(5):
            buffer.add(i)
        assert buffer.pos == 0  # Wrapped around

    def test_len_method(self):
        """Test __len__ returns size."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(7):
            buffer.add(i)
        assert len(buffer) == 7


# ============================================================================
# Test BaseBuffer Methods
# ============================================================================

class TestBaseBufferMethods:
    """Test BaseBuffer methods."""

    def test_can_sample_empty(self):
        """Test can_sample on empty buffer."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer.can_sample(10) is False

    def test_can_sample_partial(self):
        """Test can_sample with partial fill."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(20):
            buffer.add(i)
        assert buffer.can_sample(10) is True
        assert buffer.can_sample(30) is False

    def test_can_sample_full(self):
        """Test can_sample with full buffer."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(15):
            buffer.add(i)
        assert buffer.can_sample(10) is True

    def test_repr(self):
        """Test __repr__ returns useful string."""
        buffer = ConcreteBuffer(
            buffer_size=100,
            obs_shape=(4, 5, 3),
            action_dim=8,
        )
        repr_str = repr(buffer)
        assert "ConcreteBuffer" in repr_str
        assert "100" in repr_str
        assert "(4, 5, 3)" in repr_str
        assert "8" in repr_str

    def test_clear_resets_state(self):
        """Test clear method resets buffer state."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        for i in range(5):
            buffer.add(i)
        buffer.clear()
        assert buffer.size == 0
        assert buffer.full is False
        assert buffer.pos == 0


# ============================================================================
# Test Buffer Exceptions
# ============================================================================

class TestBufferExceptions:
    """Test buffer exception classes."""

    def test_buffer_error_is_exception(self):
        """Test BufferError is an exception."""
        assert issubclass(BufferError, Exception)

    def test_buffer_empty_error_inherits(self):
        """Test BufferEmptyError inherits from BufferError."""
        assert issubclass(BufferEmptyError, BufferError)

    def test_buffer_full_error_inherits(self):
        """Test BufferFullError inherits from BufferError."""
        assert issubclass(BufferFullError, BufferError)

    def test_insufficient_data_error_inherits(self):
        """Test InsufficientDataError inherits from BufferError."""
        assert issubclass(InsufficientDataError, BufferError)

    def test_buffer_empty_error_can_be_raised(self):
        """Test BufferEmptyError can be raised and caught."""
        with pytest.raises(BufferEmptyError):
            raise BufferEmptyError("Buffer is empty")

    def test_buffer_error_can_be_caught_as_base(self):
        """Test specific errors can be caught as BufferError."""
        with pytest.raises(BufferError):
            raise BufferEmptyError("Buffer is empty")

    def test_exception_messages(self):
        """Test exception messages are preserved."""
        try:
            raise BufferEmptyError("Test message")
        except BufferEmptyError as e:
            assert "Test message" in str(e)


# ============================================================================
# Test Abstract Method Requirements
# ============================================================================

class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation cannot be instantiated."""
        class IncompleteBuffer(BaseBuffer):
            def __init__(self, buffer_size, obs_shape, action_dim):
                super().__init__(buffer_size, obs_shape, action_dim)
            # Missing: add, get, clear

        with pytest.raises(TypeError):
            IncompleteBuffer(buffer_size=10, obs_shape=(4,), action_dim=2)

    def test_partial_implementation_fails(self):
        """Test that partial implementation cannot be instantiated."""
        class PartialBuffer(BaseBuffer):
            def __init__(self, buffer_size, obs_shape, action_dim):
                super().__init__(buffer_size, obs_shape, action_dim)

            def add(self, *args, **kwargs):
                pass
            # Missing: get, clear

        with pytest.raises(TypeError):
            PartialBuffer(buffer_size=10, obs_shape=(4,), action_dim=2)

    def test_full_implementation_succeeds(self):
        """Test that full implementation can be instantiated."""
        buffer = ConcreteBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_dim=2,
        )
        # Should have all required methods
        assert hasattr(buffer, 'add')
        assert hasattr(buffer, 'get')
        assert hasattr(buffer, 'clear')
        assert callable(buffer.add)
        assert callable(buffer.get)
        assert callable(buffer.clear)
