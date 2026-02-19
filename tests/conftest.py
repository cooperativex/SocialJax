"""Pytest configuration and fixtures for SocialJax tests."""

import os
import sys
from pathlib import Path

# Add socialjax to path for all tests - MUST be done before any imports
_project_root = Path(__file__).parent.parent.resolve()
_socialjax_path = _project_root / "socialjax"
_socialjax_path_str = str(_socialjax_path)

# Insert at the beginning of sys.path to ensure it takes precedence
if _socialjax_path_str not in sys.path:
    sys.path.insert(0, _socialjax_path_str)

# Also add to PYTHONPATH for subprocess
os.environ['PYTHONPATH'] = _socialjax_path_str + os.pathsep + os.environ.get('PYTHONPATH', '')

import pytest

# Check JAX availability
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_jax: mark test as requiring JAX installation"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and environment."""
    # Skip JAX-requiring tests if JAX not available
    if not JAX_AVAILABLE:
        skip_no_jax = pytest.mark.skip(reason="JAX not available")
        for item in items:
            if "requires_jax" in item.keywords:
                item.add_marker(skip_no_jax)


@pytest.fixture
def clean_algorithm_registry():
    """Fixture to provide a clean algorithm registry for tests that need it."""
    from socialjax.algorithms.registry import clear_registry, list_algorithms
    # Store existing algorithms
    existing = list_algorithms()
    # Clear registry
    clear_registry()
    yield
    # Clear again after test
    clear_registry()


@pytest.fixture
def clean_network_registry():
    """Fixture to provide a clean network registry for tests that need it."""
    from socialjax.networks.registry import clear_registry, list_networks
    # Store existing networks
    existing = list_networks()
    # Clear registry
    clear_registry()
    yield
    # Clear again after test
    clear_registry()
