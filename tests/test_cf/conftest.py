"""
Test configuration and fixtures for CF algorithm tests.
"""

import pytest
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add socialjax to path
sys.path.insert(0, 'socialjax')


# Check JAX availability
def jax_available():
    """Check if JAX is available with GPU"""
    try:
        devices = jax.devices()
        return len(devices) > 0
    except:
        return False


@pytest.fixture
def key():
    """JAX random key"""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_observation():
    """Sample observation tensor [batch, num_agents, H, W, C]"""
    return jnp.zeros((4, 3, 15, 15, 4))  # batch=4, agents=3


@pytest.fixture
def sample_actions():
    """Sample action tensor [batch, num_agents]"""
    return jnp.zeros((4, 3), dtype=jnp.int32)


@pytest.fixture
def sample_rewards():
    """Sample reward tensor [batch, num_agents]"""
    return jnp.array([
        [1.0, 0.5, -0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
        [-1.0, 2.0, 0.0]
    ])


@pytest.fixture
def sample_cf_rewards():
    """Sample counterfactual rewards [batch, num_actions, num_agents]"""
    # 4 actions, 3 agents
    return jnp.array([
        [[1.0, 0.5, 0.5], [2.0, 1.0, 1.0], [1.5, 0.8, 0.8], [0.5, 0.3, 0.3]],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[0.0, 2.0, 0.0], [1.0, 3.0, 1.0], [0.5, 2.5, 0.5], [0.2, 1.5, 0.2]],
    ])


@pytest.fixture
def env_config():
    """Default environment configuration"""
    return {
        "env_name": "coin_game",
        "num_agents": 3,
        "max_steps": 100,
    }


@pytest.fixture
def cf_config():
    """Default CF algorithm configuration"""
    return {
        "alpha": 2.0,
        "generative_lr": 0.001,
        "policy_lr": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
    }


@pytest.fixture(scope="session")
def coin_game_env():
    """Create coin_game environment for testing"""
    try:
        import socialjax
        env = socialjax.make('coin_game', num_agents=3)
        return env
    except Exception as e:
        pytest.skip(f"Environment not available: {e}")
