"""
Input/Output utilities for saving and loading model parameters.

This module provides functions for persisting and restoring trained model parameters
across different MARL algorithms. All functions use pickle serialization and JAX
tree mapping for efficient parameter conversion.
"""

import os
import pickle
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState


def save_params(train_state: TrainState, save_path: str) -> None:
    """
    Save model parameters to disk.

    This function extracts parameters from a Flax TrainState, converts them
    to numpy arrays for serialization, and saves them as a pickle file.

    Args:
        train_state: Flax TrainState containing the model parameters to save.
        save_path: Path where the parameters will be saved (typically .pkl file).
                   Parent directories will be created if they don't exist.

    Example:
        >>> train_state = TrainState.create(...)
        >>> save_params(train_state, "./checkpoints/model_seed42.pkl")

    Note:
        The function creates the directory structure if it doesn't exist.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)


def load_params(load_path: str) -> Dict[str, Any]:
    """
    Load model parameters from disk.

    This function loads parameters from a pickle file and converts them
    back to JAX arrays for use in model inference or further training.

    Args:
        load_path: Path to the saved parameters file (.pkl file).

    Returns:
        Dictionary containing the loaded model parameters as JAX arrays,
        structured according to the original model architecture.

    Example:
        >>> params = load_params("./checkpoints/model_seed42.pkl")
        >>> network = ActorCritic(...)
        >>> pi, value = network.apply(params, obs)

    Note:
        The returned parameters are in JAX array format and ready for use
        with network.apply() or other JAX operations.
    """
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)
