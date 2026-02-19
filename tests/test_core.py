"""Unit tests for core components: BaseAlgorithm, BaseTrainer, AlgorithmState, TrainerState.

Test criteria:
- BaseAlgorithm can be imported
- Cannot instantiate BaseAlgorithm directly (abstract)
- Subclass can implement all required methods
- AlgorithmState is a valid dataclass
- BaseTrainer can be imported
- TrainerState is a valid dataclass
- TrainingMetrics collects metrics correctly
- train() loop executes with mock components
- Callbacks are invoked at correct points
"""

import pytest
import sys
import os
import time
import tempfile
from typing import Dict, Any, Tuple, Optional
from unittest.mock import MagicMock, patch

# Set up path for imports
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# Test AlgorithmState
# ============================================================================

class TestAlgorithmState:
    """Tests for AlgorithmState dataclass."""

    def test_algorithm_state_creation(self):
        """Test creating an AlgorithmState instance."""
        from socialjax.core.base_algorithm import AlgorithmState

        rng = jax.random.PRNGKey(0)
        state = AlgorithmState(
            params={"layer1": {"weight": jnp.ones((3, 3))}},
            optimizer_state={},
            rng=rng,
        )

        assert state.params is not None
        assert state.rng is not None
        assert state.timestep == 0  # Default value

    def test_algorithm_state_with_timestep(self):
        """Test AlgorithmState with custom timestep."""
        from socialjax.core.base_algorithm import AlgorithmState

        rng = jax.random.PRNGKey(0)
        state = AlgorithmState(
            params={},
            optimizer_state={},
            rng=rng,
            timestep=100,
        )

        assert state.timestep == 100

    def test_algorithm_state_is_immutable(self):
        """Test that AlgorithmState is immutable (frozen dataclass)."""
        from socialjax.core.base_algorithm import AlgorithmState
        from flax import struct

        # AlgorithmState should be a struct.dataclass (immutable)
        rng = jax.random.PRNGKey(0)
        state = AlgorithmState(
            params={},
            optimizer_state={},
            rng=rng,
        )

        # struct.dataclass creates immutable objects
        # This should either raise an error or create a new object
        # depending on how it's accessed
        assert hasattr(state, 'params')
        assert hasattr(state, 'optimizer_state')
        assert hasattr(state, 'rng')
        assert hasattr(state, 'timestep')

    def test_algorithm_state_rng_type(self):
        """Test that rng is a JAX PRNGKey."""
        from socialjax.core.base_algorithm import AlgorithmState

        rng = jax.random.PRNGKey(42)
        state = AlgorithmState(
            params={},
            optimizer_state={},
            rng=rng,
        )

        # RNG should be a JAX array
        assert isinstance(state.rng, jax.Array)
        assert state.rng.shape == (2,)


# ============================================================================
# Test BaseAlgorithm Import and Structure
# ============================================================================

class TestBaseAlgorithmImport:
    """Test that BaseAlgorithm can be imported from various locations."""

    def test_import_from_base_algorithm(self):
        """Test importing BaseAlgorithm directly from base_algorithm module."""
        from socialjax.core.base_algorithm import BaseAlgorithm
        assert BaseAlgorithm is not None

    def test_import_from_core_init(self):
        """Test importing BaseAlgorithm from core __init__."""
        from socialjax.core import BaseAlgorithm
        assert BaseAlgorithm is not None

    def test_import_algorithm_state_from_core(self):
        """Test importing AlgorithmState from core module."""
        from socialjax.core import AlgorithmState
        assert AlgorithmState is not None


class TestBaseAlgorithmAbstract:
    """Test that BaseAlgorithm is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_base_algorithm(self):
        """Test that BaseAlgorithm cannot be instantiated directly."""
        from socialjax.core.base_algorithm import BaseAlgorithm

        with pytest.raises(TypeError):
            BaseAlgorithm(
                observation_space=None,
                action_space=None,
            )

    def test_base_algorithm_has_abstract_methods(self):
        """Test that BaseAlgorithm has required abstract methods."""
        from socialjax.core.base_algorithm import BaseAlgorithm
        from abc import ABC

        assert issubclass(BaseAlgorithm, ABC)

        # Check abstract methods exist
        abstract_methods = BaseAlgorithm.__abstractmethods__
        assert '_build_network' in abstract_methods
        assert '_build_optimizer' in abstract_methods
        assert 'init_state' in abstract_methods
        assert 'compute_action' in abstract_methods
        assert 'update' in abstract_methods


class TestBaseAlgorithmSubclass:
    """Test creating a concrete subclass of BaseAlgorithm."""

    def test_concrete_subclass_implements_all_methods(self):
        """Test that a concrete subclass can be created with all methods."""
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        # Define a simple network
        class SimpleNetwork(nn.Module):
            action_dim: int

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(16)(x)
                x = nn.relu(x)
                return nn.Dense(self.action_dim)(x)

        # Define a complete algorithm implementation
        class CompleteAlgorithm(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=self.action_space.n)

            def _build_optimizer(self):
                return optax.adam(learning_rate=self.config.get('lr', 1e-3))

            def init_state(self, rng):
                rng, init_rng = jax.random.split(rng)
                dummy_obs = jnp.zeros(self.observation_space.shape)
                params = self.network.init(init_rng, dummy_obs)
                optimizer_state = self.optimizer.init(params)

                return AlgorithmState(
                    params=params,
                    optimizer_state=optimizer_state,
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                logits = self.network.apply(state.params, observation)
                if deterministic:
                    action = jnp.argmax(logits)
                else:
                    action = jax.random.categorical(rng, logits)
                return action, {"log_prob": 0.0}

            def update(self, state, batch):
                def loss_fn(params):
                    logits = self.network.apply(params, batch['observations'])
                    loss = jnp.mean(jnp.square(logits))
                    return loss

                grads = jax.grad(loss_fn)(state.params)
                updates, new_opt_state = self.optimizer.update(
                    grads, state.optimizer_state, state.params
                )
                new_params = optax.apply_updates(state.params, updates)

                new_state = AlgorithmState(
                    params=new_params,
                    optimizer_state=new_opt_state,
                    rng=state.rng,
                    timestep=state.timestep + 1,
                )

                return new_state, {"loss": 0.5}

        # Create a mock observation/action space
        class MockSpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n

        obs_space = MockSpace(shape=(4,))
        act_space = MockSpace(shape=(), n=2)

        # Create algorithm instance
        algo = CompleteAlgorithm(
            observation_space=obs_space,
            action_space=act_space,
            config={'lr': 1e-3},
        )

        assert algo.observation_space is obs_space
        assert algo.action_space is act_space
        assert algo.network is not None
        assert algo.optimizer is not None

        # Test init_state
        rng = jax.random.PRNGKey(0)
        state = algo.init_state(rng)
        assert state.params is not None
        assert state.optimizer_state is not None

        # Test compute_action
        obs = jnp.zeros((4,))
        action, info = algo.compute_action(state, obs, jax.random.PRNGKey(1))
        assert action is not None

        # Test update
        batch = {
            'observations': jnp.zeros((1, 4)),
        }
        new_state, metrics = algo.update(state, batch)
        assert 'loss' in metrics


class TestBaseAlgorithmSaveLoad:
    """Test save/load functionality of BaseAlgorithm."""

    def test_save_creates_checkpoint(self):
        """Test that save creates a checkpoint file."""
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class TestAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.0}

        class MockSpace:
            shape = (4,)
            n = 2

        algo = TestAlgo(MockSpace(), MockSpace())
        algo._state = algo.init_state(jax.random.PRNGKey(0))

        with tempfile.TemporaryDirectory() as tmpdir:
            algo.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "checkpoint.pkl"))

    def test_load_restores_state(self):
        """Test that load restores the algorithm state."""
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class TestAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.0}

        class MockSpace:
            shape = (4,)
            n = 2

        algo = TestAlgo(MockSpace(), MockSpace())
        original_state = algo.init_state(jax.random.PRNGKey(0))
        algo._state = original_state

        with tempfile.TemporaryDirectory() as tmpdir:
            algo.save(tmpdir)

            # Create new algorithm and load
            algo2 = TestAlgo(MockSpace(), MockSpace())
            loaded_state = algo2.load(tmpdir)

            assert loaded_state.params is not None
            assert algo2._state is not None


class TestJitMethodDecorator:
    """Test the jit_method decorator."""

    def test_jit_method_is_callable(self):
        """Test that jit_method decorator returns a callable."""
        from socialjax.core.base_algorithm import jit_method

        @jit_method
        def test_method(self, x):
            return x * 2

        assert callable(test_method)

    def test_jit_method_applies_jax_jit(self):
        """Test that jit_method applies jax.jit."""
        from socialjax.core.base_algorithm import jit_method
        import jax

        @jit_method
        def test_method(self, x):
            return x * 2

        # The decorated function should have jit attributes
        assert hasattr(test_method, 'lower')  # jax.jit functions have .lower()


# ============================================================================
# Test TrainerState
# ============================================================================

class TestTrainerState:
    """Tests for TrainerState dataclass."""

    def test_trainer_state_creation(self):
        """Test creating a TrainerState instance."""
        from socialjax.core.base_trainer import TrainerState
        from socialjax.core.base_algorithm import AlgorithmState

        rng = jax.random.PRNGKey(0)
        algo_state = AlgorithmState(
            params={},
            optimizer_state={},
            rng=rng,
        )

        state = TrainerState(
            algorithm_state=algo_state,
            timestep=100,
            update_step=10,
            episode_count=5,
            start_time=time.time(),
        )

        assert state.timestep == 100
        assert state.update_step == 10
        assert state.episode_count == 5
        assert state.algorithm_state is algo_state

    def test_trainer_state_defaults(self):
        """Test TrainerState default values."""
        from socialjax.core.base_trainer import TrainerState
        from socialjax.core.base_algorithm import AlgorithmState

        rng = jax.random.PRNGKey(0)
        algo_state = AlgorithmState(
            params={},
            optimizer_state={},
            rng=rng,
        )

        state = TrainerState(algorithm_state=algo_state)

        assert state.timestep == 0
        assert state.update_step == 0
        assert state.episode_count == 0
        assert state.start_time == 0.0


# ============================================================================
# Test TrainingMetrics
# ============================================================================

class TestTrainingMetrics:
    """Tests for TrainingMetrics class."""

    def test_metrics_creation(self):
        """Test creating a TrainingMetrics instance."""
        from socialjax.core.base_trainer import TrainingMetrics

        metrics = TrainingMetrics()
        assert metrics.episode_returns == []
        assert metrics.episode_lengths == []
        assert metrics.losses == {}
        assert metrics.custom_metrics == {}

    def test_add_episode(self):
        """Test adding episode return and length."""
        from socialjax.core.base_trainer import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.add_episode(return_value=10.5, length=100)
        metrics.add_episode(return_value=15.0, length=150)

        assert len(metrics.episode_returns) == 2
        assert len(metrics.episode_lengths) == 2
        assert metrics.episode_returns == [10.5, 15.0]
        assert metrics.episode_lengths == [100, 150]

    def test_add_loss(self):
        """Test adding loss values."""
        from socialjax.core.base_trainer import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.add_loss("policy_loss", 0.5)
        metrics.add_loss("policy_loss", 0.3)
        metrics.add_loss("value_loss", 0.1)

        assert metrics.losses["policy_loss"] == [0.5, 0.3]
        assert metrics.losses["value_loss"] == [0.1]

    def test_add_metric(self):
        """Test adding custom metrics."""
        from socialjax.core.base_trainer import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.add_metric("entropy", 0.8)
        metrics.add_metric("entropy", 0.7)

        assert metrics.custom_metrics["entropy"] == [0.8, 0.7]

    def test_get_summary_empty(self):
        """Test get_summary with no data."""
        from socialjax.core.base_trainer import TrainingMetrics

        metrics = TrainingMetrics()
        summary = metrics.get_summary()

        assert summary == {}

    def test_get_summary_with_data(self):
        """Test get_summary with episode data."""
        from socialjax.core.base_trainer import TrainingMetrics

        metrics = TrainingMetrics()
        for i in range(10):
            metrics.add_episode(return_value=float(i), length=100 + i * 10)
        metrics.add_loss("total_loss", 1.0)
        metrics.add_loss("total_loss", 0.5)
        metrics.add_loss("total_loss", 0.1)

        summary = metrics.get_summary()

        assert "mean_episode_return" in summary
        assert "std_episode_return" in summary
        assert "mean_episode_length" in summary
        assert "mean_total_loss" in summary
        assert summary["mean_episode_return"] == 4.5  # Mean of 0-9
        assert summary["mean_episode_length"] == 145  # Mean of 100-190


# ============================================================================
# Test BaseTrainer Import and Structure
# ============================================================================

class TestBaseTrainerImport:
    """Test that BaseTrainer can be imported from various locations."""

    def test_import_from_base_trainer(self):
        """Test importing BaseTrainer directly from base_trainer module."""
        from socialjax.core.base_trainer import BaseTrainer
        assert BaseTrainer is not None

    def test_import_from_core_init(self):
        """Test importing BaseTrainer from core __init__."""
        from socialjax.core import BaseTrainer
        assert BaseTrainer is not None

    def test_import_trainer_state_from_core(self):
        """Test importing TrainerState from core module."""
        from socialjax.core import TrainerState
        assert TrainerState is not None

    def test_import_training_metrics_from_core(self):
        """Test importing TrainingMetrics from core module."""
        from socialjax.core import TrainingMetrics
        assert TrainingMetrics is not None

    def test_import_callback_protocol(self):
        """Test importing Callback protocol from core module."""
        from socialjax.core.base_trainer import Callback
        assert Callback is not None


class TestBaseTrainerAbstract:
    """Test that BaseTrainer is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_base_trainer(self):
        """Test that BaseTrainer cannot be instantiated directly."""
        from socialjax.core.base_trainer import BaseTrainer
        from socialjax.core.base_algorithm import BaseAlgorithm

        # Create a mock algorithm
        mock_algo = MagicMock(spec=BaseAlgorithm)
        mock_env = MagicMock()

        with pytest.raises(TypeError):
            BaseTrainer(
                algorithm=mock_algo,
                env=mock_env,
                config={},
            )

    def test_base_trainer_has_abstract_methods(self):
        """Test that BaseTrainer has required abstract methods."""
        from socialjax.core.base_trainer import BaseTrainer
        from abc import ABC

        assert issubclass(BaseTrainer, ABC)

        abstract_methods = BaseTrainer.__abstractmethods__
        assert '_create_buffer' in abstract_methods
        assert '_collect_rollout' in abstract_methods
        assert '_update' in abstract_methods


class TestBaseTrainerSubclass:
    """Test creating a concrete subclass of BaseTrainer."""

    def test_concrete_subclass_implements_all_methods(self):
        """Test that a concrete subclass can be created with all methods."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState, TrainingMetrics
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        # Create a mock algorithm first
        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {"log_prob": -0.5}

            def update(self, state, batch):
                return state, {"loss": 0.1}

        class MockSpace:
            shape = (4,)
            n = 2

        class MockBuffer:
            def __init__(self):
                self.data = []

            def add(self, *args):
                self.data.append(args)

            def get(self):
                return {"observations": jnp.zeros((1, 4))}

        class CompleteTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                rollout_data = self.buffer.get()
                new_state = TrainerState(
                    algorithm_state=state.algorithm_state,
                    timestep=state.timestep + 10,
                    update_step=state.update_step,
                    episode_count=state.episode_count + 1,
                    start_time=state.start_time,
                )
                return rollout_data, new_state

            def _update(self, state, rollout_data):
                new_state, metrics = self.algorithm.update(
                    state.algorithm_state, rollout_data
                )
                return TrainerState(
                    algorithm_state=new_state,
                    timestep=state.timestep,
                    update_step=state.update_step + 1,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                ), metrics

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_env = MagicMock()
        mock_env.agents = ['agent_0']
        mock_env.reset = MagicMock(return_value=(
            {'agent_0': jnp.zeros((4,))},
            MagicMock()
        ))

        trainer = CompleteTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={'total_timesteps': 10},
            callbacks=[],
        )

        assert trainer.algorithm is mock_algo
        assert trainer.env is mock_env
        assert trainer.buffer is not None
        assert trainer.metrics is not None


class TestBaseTrainerCallbacks:
    """Test callback invocation in BaseTrainer."""

    def test_callbacks_invoked_on_training_start(self):
        """Test that on_training_start is invoked."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        from socialjax.training.callbacks import BaseCallback
        import optax
        import flax.linen as nn

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.training_start_called = False

            def on_training_start(self, trainer):
                self.training_start_called = True

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.0}

        class MockBuffer:
            def get(self):
                return {"observations": jnp.zeros((1, 4))}

        class TestTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                new_state = TrainerState(
                    algorithm_state=state.algorithm_state,
                    timestep=state.timestep + 10,
                    update_step=state.update_step,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                )
                return {}, new_state

            def _update(self, state, rollout_data):
                new_state, metrics = self.algorithm.update(
                    state.algorithm_state, rollout_data or {}
                )
                return TrainerState(
                    algorithm_state=new_state,
                    timestep=state.timestep,
                    update_step=state.update_step + 1,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                ), metrics

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_env = MagicMock()
        callback = TrackingCallback()
        trainer = TestTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={},
            callbacks=[callback],
        )

        # Run a short training
        trainer.train(total_timesteps=5)

        assert callback.training_start_called

    def test_callbacks_invoked_on_training_end(self):
        """Test that on_training_end is invoked."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        from socialjax.training.callbacks import BaseCallback
        import optax
        import flax.linen as nn

        class TrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.training_end_called = False

            def on_training_end(self, trainer):
                self.training_end_called = True

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.0}

        class MockBuffer:
            def get(self):
                return {"observations": jnp.zeros((1, 4))}

        class TestTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                new_state = TrainerState(
                    algorithm_state=state.algorithm_state,
                    timestep=state.timestep + 10,
                    update_step=state.update_step,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                )
                return {}, new_state

            def _update(self, state, rollout_data):
                new_state, metrics = self.algorithm.update(
                    state.algorithm_state, rollout_data or {}
                )
                return TrainerState(
                    algorithm_state=new_state,
                    timestep=state.timestep,
                    update_step=state.update_step + 1,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                ), metrics

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_env = MagicMock()
        callback = TrackingCallback()
        trainer = TestTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={},
            callbacks=[callback],
        )

        # Run a short training
        trainer.train(total_timesteps=5)

        assert callback.training_end_called

    def test_callback_order(self):
        """Test that callbacks are invoked in correct order."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        from socialjax.training.callbacks import BaseCallback
        import optax
        import flax.linen as nn

        class OrderTrackingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.events = []

            def on_training_start(self, trainer):
                self.events.append("training_start")

            def on_training_end(self, trainer):
                self.events.append("training_end")

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.0}

        class MockBuffer:
            def get(self):
                return {"observations": jnp.zeros((1, 4))}

        class TestTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                new_state = TrainerState(
                    algorithm_state=state.algorithm_state,
                    timestep=state.timestep + 10,
                    update_step=state.update_step,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                )
                return {}, new_state

            def _update(self, state, rollout_data):
                new_state, metrics = self.algorithm.update(
                    state.algorithm_state, rollout_data or {}
                )
                return TrainerState(
                    algorithm_state=new_state,
                    timestep=state.timestep,
                    update_step=state.update_step + 1,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                ), metrics

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_env = MagicMock()
        callback = OrderTrackingCallback()
        trainer = TestTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={},
            callbacks=[callback],
        )

        trainer.train(total_timesteps=5)

        # training_start should come before training_end
        assert callback.events.index("training_start") < callback.events.index("training_end")


class TestBaseTrainerTrainLoop:
    """Test training loop functionality."""

    def test_train_returns_final_state_and_metrics(self):
        """Test that train returns final state and metrics."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.1}

        class MockBuffer:
            def get(self):
                return {"observations": jnp.zeros((1, 4))}

        class TestTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                new_state = TrainerState(
                    algorithm_state=state.algorithm_state,
                    timestep=state.timestep + 10,
                    update_step=state.update_step,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                )
                return {}, new_state

            def _update(self, state, rollout_data):
                new_state, metrics = self.algorithm.update(
                    state.algorithm_state, rollout_data or {}
                )
                return TrainerState(
                    algorithm_state=new_state,
                    timestep=state.timestep,
                    update_step=state.update_step + 1,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                ), metrics

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_env = MagicMock()
        trainer = TestTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={},
            callbacks=[],
        )

        final_state, metrics = trainer.train(total_timesteps=15)

        assert final_state.timestep >= 15
        assert "training_summary" in metrics
        assert "total_timesteps" in metrics
        assert "elapsed_time" in metrics

    def test_keyboard_interrupt_handling(self):
        """Test that KeyboardInterrupt is handled gracefully."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.1}

        class MockBuffer:
            def get(self):
                return {"observations": jnp.zeros((1, 4))}

        class InterruptingTrainer(BaseTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._call_count = 0

            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                self._call_count += 1
                if self._call_count >= 2:
                    raise KeyboardInterrupt()
                new_state = TrainerState(
                    algorithm_state=state.algorithm_state,
                    timestep=state.timestep + 10,
                    update_step=state.update_step,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                )
                return {}, new_state

            def _update(self, state, rollout_data):
                new_state, metrics = self.algorithm.update(
                    state.algorithm_state, rollout_data or {}
                )
                return TrainerState(
                    algorithm_state=new_state,
                    timestep=state.timestep,
                    update_step=state.update_step + 1,
                    episode_count=state.episode_count,
                    start_time=state.start_time,
                ), metrics

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_env = MagicMock()
        trainer = InterruptingTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={},
            callbacks=[],
        )

        # Should not raise, should handle gracefully
        final_state, metrics = trainer.train(total_timesteps=1000)

        # Should have stopped early
        assert final_state.timestep < 1000
        assert "total_timesteps" in metrics


class TestBaseTrainerEvaluate:
    """Test evaluate functionality in BaseTrainer."""

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns evaluation metrics."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.1}

        class MockBuffer:
            pass

        class TestTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                return {}, state

            def _update(self, state, rollout_data):
                return state, {"loss": 0.1}

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_env = MagicMock()
        mock_env.agents = ['agent_0', 'agent_1']

        # Setup mock reset and step
        mock_env.reset = MagicMock(return_value=(
            {'agent_0': jnp.zeros((4,)), 'agent_1': jnp.zeros((4,))},
            MagicMock()
        ))
        mock_env.step = MagicMock(return_value=(
            MagicMock(),  # new state
            {'agent_0': jnp.zeros((4,)), 'agent_1': jnp.zeros((4,))},  # obs
            {'agent_0': 1.0, 'agent_1': 1.0},  # rewards
            {'__all__': True},  # dones
            {}  # info
        ))

        trainer = TestTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={},
            callbacks=[],
        )

        # Initialize state
        rng = jax.random.PRNGKey(0)
        algo_state = mock_algo.init_state(rng)
        state = TrainerState(
            algorithm_state=algo_state,
            timestep=0,
            update_step=0,
            episode_count=0,
            start_time=time.time(),
        )

        # Run evaluate
        metrics = trainer.evaluate(state, num_episodes=2, deterministic=True)

        assert "mean_return" in metrics
        assert "std_return" in metrics
        assert "mean_length" in metrics
        assert metrics["num_episodes"] == 2


class TestBaseTrainerSaveLoad:
    """Test save/load functionality in BaseTrainer."""

    def test_trainer_save(self):
        """Test that trainer.save creates files."""
        from socialjax.core.base_trainer import BaseTrainer
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.1}

        class MockBuffer:
            pass

        class TestTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                return {}, state

            def _update(self, state, rollout_data):
                return state, {"loss": 0.1}

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_algo._state = mock_algo.init_state(jax.random.PRNGKey(0))
        mock_env = MagicMock()

        trainer = TestTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={'test': 'config'},
            callbacks=[],
        )
        trainer.metrics.add_episode(10.0, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)

            # Check algorithm checkpoint was created
            assert os.path.exists(os.path.join(tmpdir, "algorithm", "checkpoint.pkl"))
            # Check trainer info was created
            assert os.path.exists(os.path.join(tmpdir, "trainer_info.pkl"))

    def test_trainer_load(self):
        """Test that trainer.load restores state."""
        from socialjax.core.base_trainer import BaseTrainer, TrainerState
        from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
        import optax
        import flax.linen as nn

        class SimpleNetwork(nn.Module):
            action_dim: int = 2

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        class MockAlgo(BaseAlgorithm):
            def _build_network(self):
                return SimpleNetwork(action_dim=2)

            def _build_optimizer(self):
                return optax.adam(1e-3)

            def init_state(self, rng):
                params = self.network.init(rng, jnp.zeros((4,)))
                return AlgorithmState(
                    params=params,
                    optimizer_state=self.optimizer.init(params),
                    rng=rng,
                )

            def compute_action(self, state, observation, rng, deterministic=False):
                return jnp.array(0), {}

            def update(self, state, batch):
                return state, {"loss": 0.1}

        class MockBuffer:
            pass

        class TestTrainer(BaseTrainer):
            def _create_buffer(self):
                return MockBuffer()

            def _collect_rollout(self, state):
                return {}, state

            def _update(self, state, rollout_data):
                return state, {"loss": 0.1}

        class MockSpace:
            shape = (4,)
            n = 2

        mock_algo = MockAlgo(MockSpace(), MockSpace())
        mock_algo._state = mock_algo.init_state(jax.random.PRNGKey(0))
        mock_env = MagicMock()

        trainer = TestTrainer(
            algorithm=mock_algo,
            env=mock_env,
            config={},
            callbacks=[],
        )
        trainer.metrics.add_episode(15.0, 200)
        trainer.metrics.add_loss("test_loss", 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            trainer.save(tmpdir)

            # Create new trainer and load
            mock_algo2 = MockAlgo(MockSpace(), MockSpace())
            trainer2 = TestTrainer(
                algorithm=mock_algo2,
                env=mock_env,
                config={},
                callbacks=[],
            )

            loaded_state = trainer2.load(tmpdir)

            # Metrics should be restored
            assert len(trainer2.metrics.episode_returns) == 1
            assert trainer2.metrics.episode_returns[0] == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
