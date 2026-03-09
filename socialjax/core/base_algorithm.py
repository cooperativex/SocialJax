"""Base algorithm abstract class for SocialJax.

This module provides the abstract base class that all algorithms in SocialJax
must inherit from. It defines the interface for:
- Network and optimizer creation
- State initialization
- Action computation
- Parameter updates
- Serialization (save/load)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class AlgorithmState:
    """Immutable state container for algorithm parameters and runtime state.

    This dataclass uses Flax struct.dataclass decorator to create a
    JAX-compatible immutable data structure that can be used with JAX
    transformations like jit, grad, and vmap.

    Attributes:
        params: Dictionary of network parameters (weights and biases).
        optimizer_state: Optimizer state (e.g., momentum buffers for Adam).
        rng: JAX random number generator key for stochastic operations.
        timestep: Current training timestep (optional, defaults to 0).
    """
    params: Dict[str, Any]
    optimizer_state: Any
    rng: jax.random.PRNGKey
    timestep: int = 0


class BaseAlgorithm(ABC):
    """Abstract base class for all multi-agent reinforcement learning algorithms.

    This class defines the interface that all algorithms must implement to work
    with the SocialJax training framework. It provides:
    - Standard initialization pattern for networks and optimizers
    - Abstract methods for action computation and parameter updates
    - Serialization interface for saving and loading models

    Subclasses must implement all abstract methods. The framework expects
    algorithms to be JAX-compatible and support JIT compilation for
    performance-critical operations.

    Example:
        >>> class IPPOAlgorithm(BaseAlgorithm):
        ...     def _build_network(self):
        ...         return IPPOActorCritic(action_dim=self.action_space.n)
        ...
        ...     def _build_optimizer(self):
        ...         return optax.adam(learning_rate=self.config["LR"])
        ...
        ...     # ... implement other abstract methods

    Attributes:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        config: Algorithm configuration dictionary.
        network: The neural network (set in _setup).
        optimizer: The optimizer (set in _setup).
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the algorithm.

        Args:
            observation_space: Environment observation space.
            action_space: Environment action space.
            config: Algorithm configuration dictionary. If None, uses defaults.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}
        self._setup()

    def _setup(self) -> None:
        """Initialize network and optimizer.

        Called automatically during __init__. Subclasses can override
        to add additional setup logic, but should call super()._setup().
        """
        self.network = self._build_network()
        self.optimizer = self._build_optimizer()

    @abstractmethod
    def _build_network(self) -> Any:
        """Build and return the neural network architecture.

        This method should create and return the network that will be used
        by the algorithm. The network should be compatible with Flax
        nn.Module interface.

        Returns:
            A Flax nn.Module or compatible network instance.
        """
        pass

    @abstractmethod
    def _build_optimizer(self) -> Any:
        """Build and return the optimizer.

        This method should create and return an optimizer from optax or
        a compatible library.

        Returns:
            An optax optimizer or compatible optimizer instance.
        """
        pass

    @abstractmethod
    def init_state(self, rng: jax.random.PRNGKey) -> AlgorithmState:
        """Initialize and return the algorithm state.

        This method should:
        1. Initialize network parameters with the given RNG key
        2. Initialize optimizer state
        3. Return an AlgorithmState containing all initialized components

        Args:
            rng: JAX random number generator key for initialization.

        Returns:
            An AlgorithmState containing initialized params, optimizer_state, and rng.
        """
        pass

    @abstractmethod
    def compute_action(
        self,
        state: AlgorithmState,
        observation: jnp.ndarray,
        rng: jax.random.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute action(s) given observation(s).

        This method should be JIT-compilable and handle both single
        observations and batches as appropriate.

        Args:
            state: Current algorithm state containing params.
            observation: Observation array from the environment.
            rng: JAX random key for stochastic action selection.
            deterministic: If True, select the most likely action.
                          If False, sample from the action distribution.

        Returns:
            Tuple of:
                - action: Selected action(s) as jnp.ndarray.
                - info: Dictionary with additional info (e.g., log_prob, value).
        """
        pass

    @abstractmethod
    def update(
        self,
        state: AlgorithmState,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[AlgorithmState, Dict[str, float]]:
        """Update algorithm parameters using a batch of experience.

        This method should be JIT-compilable and perform a single
        gradient update step.

        Args:
            state: Current algorithm state.
            batch: Dictionary containing:
                - observations: Array of observations
                - actions: Array of actions taken
                - rewards: Array of rewards received
                - dones: Array of done flags
                - Additional algorithm-specific data (e.g., old_log_probs, values)

        Returns:
            Tuple of:
                - new_state: Updated AlgorithmState with new params.
                - metrics: Dictionary of training metrics (e.g., loss, entropy).
        """
        pass

    def save(self, path: str) -> None:
        """Save algorithm state to disk.

        Saves the network parameters and optimizer state to allow
        resuming training or running evaluation later.

        Args:
            path: Directory path to save the checkpoint.

        Note:
            Subclasses may override to save additional state.
        """
        import os
        import pickle
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save params and optimizer state using pickle for JAX arrays
        checkpoint = {
            "params": self._state.params if hasattr(self, "_state") else None,
            "optimizer_state": self._state.optimizer_state if hasattr(self, "_state") else None,
            "config": self.config,
        }

        with open(path / "checkpoint.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str) -> AlgorithmState:
        """Load algorithm state from disk.

        Args:
            path: Directory path containing the checkpoint.

        Returns:
            Loaded AlgorithmState.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        checkpoint_file = path / "checkpoint.pkl"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        # Reconstruct state
        rng = jax.random.PRNGKey(0)  # Placeholder, should be set by caller
        state = AlgorithmState(
            params=checkpoint["params"],
            optimizer_state=checkpoint["optimizer_state"],
            rng=rng,
        )

        self._state = state
        return state


# Decorator for marking methods as JIT-compilable
def jit_method(method):
    """Decorator to mark a method for JIT compilation.

    Usage:
        @jit_method
        def compute_action(self, state, observation, rng, deterministic=False):
            ...
    """
    return jax.jit(method, static_argnums=(0, 4))  # static: self, deterministic
