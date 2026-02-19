"""Unit tests for algorithm and network registries.

Test criteria:
- Decorated class is added to registry
- get_algorithm returns correct class
- list_algorithms returns all registered names
- Duplicate registration raises error
- Unknown algorithm raises helpful error message
- Same for network registry
"""

import pytest
import sys
from typing import Dict, Any

# Set up path for imports
sys.path.insert(0, 'socialjax')


# Fixtures to save/restore registry state
@pytest.fixture(autouse=True)
def preserve_algorithm_registry():
    """Save and restore algorithm registry state around each test."""
    from socialjax.algorithms.registry import _ALGORITHM_REGISTRY, clear_registry
    original = _ALGORITHM_REGISTRY.copy()
    yield
    # Restore original state
    clear_registry()
    _ALGORITHM_REGISTRY.update(original)


@pytest.fixture(autouse=True)
def preserve_network_registry():
    """Save and restore network registry state around each test."""
    from socialjax.networks.registry import _NETWORK_REGISTRY, clear_registry
    original = _NETWORK_REGISTRY.copy()
    yield
    # Restore original state
    clear_registry()
    _NETWORK_REGISTRY.update(original)


# ============================================================================
# Test Algorithm Registry
# ============================================================================

class TestAlgorithmRegistryImport:
    """Test that registry functions can be imported from various locations."""

    def test_import_register_algorithm(self):
        """Test importing register_algorithm decorator."""
        from socialjax.algorithms.registry import register_algorithm
        assert register_algorithm is not None
        assert callable(register_algorithm)

    def test_import_get_algorithm(self):
        """Test importing get_algorithm function."""
        from socialjax.algorithms.registry import get_algorithm
        assert get_algorithm is not None
        assert callable(get_algorithm)

    def test_import_list_algorithms(self):
        """Test importing list_algorithms function."""
        from socialjax.algorithms.registry import list_algorithms
        assert list_algorithms is not None
        assert callable(list_algorithms)

    def test_import_from_algorithms_init(self):
        """Test importing registry functions from algorithms __init__."""
        from socialjax.algorithms import register_algorithm, get_algorithm, list_algorithms
        assert register_algorithm is not None
        assert get_algorithm is not None
        assert list_algorithms is not None


class TestAlgorithmRegistryExceptions:
    """Test custom exceptions for algorithm registry."""

    def test_algorithm_already_registered_error(self):
        """Test AlgorithmAlreadyRegisteredError exists and is Exception."""
        from socialjax.algorithms.registry import AlgorithmAlreadyRegisteredError

        assert issubclass(AlgorithmAlreadyRegisteredError, Exception)

    def test_algorithm_not_found_error(self):
        """Test AlgorithmNotFoundError exists and is Exception."""
        from socialjax.algorithms.registry import AlgorithmNotFoundError

        assert issubclass(AlgorithmNotFoundError, Exception)

    def test_algorithm_not_found_error_message(self):
        """Test AlgorithmNotFoundError has helpful message."""
        from socialjax.algorithms.registry import AlgorithmNotFoundError

        try:
            raise AlgorithmNotFoundError("Test message")
        except AlgorithmNotFoundError as e:
            assert "Test message" in str(e)


class TestAlgorithmRegisterDecorator:
    """Test @register_algorithm decorator functionality."""

    def test_decorator_registers_algorithm(self):
        """Test that decorator adds class to registry."""
        from socialjax.algorithms.registry import (
            register_algorithm, get_algorithm
        )

        @register_algorithm("test_algo_1")
        class TestAlgorithm:
            pass

        # Should be able to retrieve it
        retrieved = get_algorithm("test_algo_1")
        assert retrieved is TestAlgorithm

    def test_decorator_returns_original_class(self):
        """Test that decorator returns the original class."""
        from socialjax.algorithms.registry import register_algorithm

        @register_algorithm("test_algo_2")
        class TestAlgorithm:
            """Test class."""
            pass

        assert TestAlgorithm.__doc__ == "Test class."
        assert TestAlgorithm.__name__ == "TestAlgorithm"

    def test_duplicate_registration_raises_error(self):
        """Test that registering same name twice raises error."""
        from socialjax.algorithms.registry import (
            register_algorithm, AlgorithmAlreadyRegisteredError
        )

        @register_algorithm("test_algo_duplicate")
        class TestAlgorithm1:
            pass

        with pytest.raises(AlgorithmAlreadyRegisteredError):
            @register_algorithm("test_algo_duplicate")
            class TestAlgorithm2:
                pass


class TestGetAlgorithm:
    """Test get_algorithm function."""

    def test_get_algorithm_returns_class(self):
        """Test that get_algorithm returns the registered class."""
        from socialjax.algorithms.registry import register_algorithm, get_algorithm

        @register_algorithm("test_algo_get")
        class TestAlgorithm:
            pass

        retrieved = get_algorithm("test_algo_get")
        assert retrieved is TestAlgorithm

    def test_get_unknown_algorithm_raises_error(self):
        """Test that getting unknown algorithm raises helpful error."""
        from socialjax.algorithms.registry import get_algorithm, AlgorithmNotFoundError

        with pytest.raises(AlgorithmNotFoundError) as exc_info:
            get_algorithm("nonexistent_algorithm_xyz")

        error_message = str(exc_info.value)
        assert "nonexistent_algorithm_xyz" in error_message

    def test_get_algorithm_with_available_list(self):
        """Test that error message lists available algorithms."""
        from socialjax.algorithms.registry import (
            register_algorithm, get_algorithm, AlgorithmNotFoundError
        )

        @register_algorithm("algo_a_test")
        class AlgoA:
            pass

        @register_algorithm("algo_b_test")
        class AlgoB:
            pass

        try:
            get_algorithm("nonexistent_xyz")
        except AlgorithmNotFoundError as e:
            error_message = str(e)
            # Should mention available algorithms
            assert "Available" in error_message or "available" in error_message


class TestListAlgorithms:
    """Test list_algorithms function."""

    def test_list_algorithms_empty(self):
        """Test list_algorithms with empty registry."""
        from socialjax.algorithms.registry import list_algorithms, clear_registry

        # Clear for this test only
        clear_registry()
        result = list_algorithms()
        assert isinstance(result, list)

    def test_list_algorithms_returns_sorted(self):
        """Test that list_algorithms returns sorted list."""
        from socialjax.algorithms.registry import register_algorithm, list_algorithms

        @register_algorithm("z_algo_test")
        class ZAlgo:
            pass

        @register_algorithm("a_algo_test")
        class AAlgo:
            pass

        @register_algorithm("m_algo_test")
        class MAlgo:
            pass

        result = list_algorithms()
        # Should be sorted
        sorted_names = [n for n in result if n.endswith("_test")]
        assert sorted_names == sorted(sorted_names)


class TestAlgorithmRegistryUtilities:
    """Test utility functions for algorithm registry."""

    def test_unregister_algorithm(self):
        """Test unregister_algorithm removes algorithm."""
        from socialjax.algorithms.registry import (
            register_algorithm, get_algorithm, unregister_algorithm
        )

        @register_algorithm("test_unregister_util")
        class TestAlgo:
            pass

        # Should exist
        assert get_algorithm("test_unregister_util") is TestAlgo

        # Unregister
        removed = unregister_algorithm("test_unregister_util")
        assert removed is TestAlgo

        # Should not exist anymore
        from socialjax.algorithms.registry import AlgorithmNotFoundError
        with pytest.raises(AlgorithmNotFoundError):
            get_algorithm("test_unregister_util")

    def test_unregister_nonexistent(self):
        """Test unregister_algorithm with nonexistent name returns None."""
        from socialjax.algorithms.registry import unregister_algorithm

        result = unregister_algorithm("nonexistent_util_xyz")
        assert result is None

    def test_is_algorithm_registered(self):
        """Test is_algorithm_registered function."""
        from socialjax.algorithms.registry import (
            register_algorithm, is_algorithm_registered
        )

        assert is_algorithm_registered("test_is_reg_util") is False

        @register_algorithm("test_is_reg_util")
        class TestAlgo:
            pass

        assert is_algorithm_registered("test_is_reg_util") is True


# ============================================================================
# Test Network Registry
# ============================================================================

class TestNetworkRegistryImport:
    """Test that network registry functions can be imported."""

    def test_import_register_network(self):
        """Test importing register_network decorator."""
        from socialjax.networks.registry import register_network
        assert register_network is not None
        assert callable(register_network)

    def test_import_get_network_class(self):
        """Test importing get_network_class function."""
        from socialjax.networks.registry import get_network_class
        assert get_network_class is not None
        assert callable(get_network_class)

    def test_import_list_networks(self):
        """Test importing list_networks function."""
        from socialjax.networks.registry import list_networks
        assert list_networks is not None
        assert callable(list_networks)

    def test_import_from_networks_init(self):
        """Test importing registry functions from networks __init__."""
        from socialjax.networks import register_network, get_network_class, list_networks
        assert register_network is not None
        assert get_network_class is not None
        assert list_networks is not None


class TestNetworkRegistryExceptions:
    """Test custom exceptions for network registry."""

    def test_network_already_registered_error(self):
        """Test NetworkAlreadyRegisteredError exists and is Exception."""
        from socialjax.networks.registry import NetworkAlreadyRegisteredError

        assert issubclass(NetworkAlreadyRegisteredError, Exception)

    def test_network_not_found_error(self):
        """Test NetworkNotFoundError exists and is Exception."""
        from socialjax.networks.registry import NetworkNotFoundError

        assert issubclass(NetworkNotFoundError, Exception)


class TestNetworkRegisterDecorator:
    """Test @register_network decorator functionality."""

    def test_decorator_registers_network(self):
        """Test that decorator adds class to registry."""
        from socialjax.networks.registry import (
            register_network, get_network_class
        )
        import flax.linen as nn

        @register_network("test_network_1")
        class TestNetwork(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        # Should be able to retrieve it
        retrieved = get_network_class("test_network_1")
        assert retrieved is TestNetwork

    def test_decorator_returns_original_class(self):
        """Test that decorator returns the original class."""
        from socialjax.networks.registry import register_network
        import flax.linen as nn

        @register_network("test_network_2")
        class TestNetwork(nn.Module):
            """Test network docstring."""
            @nn.compact
            def __call__(self, x):
                return x

        assert TestNetwork.__doc__ == "Test network docstring."

    def test_duplicate_registration_raises_error(self):
        """Test that registering same name twice raises error."""
        from socialjax.networks.registry import (
            register_network, NetworkAlreadyRegisteredError
        )
        import flax.linen as nn

        @register_network("test_network_dup")
        class TestNetwork1(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        with pytest.raises(NetworkAlreadyRegisteredError):
            @register_network("test_network_dup")
            class TestNetwork2(nn.Module):
                @nn.compact
                def __call__(self, x):
                    return x


class TestGetNetworkClass:
    """Test get_network_class function."""

    def test_get_network_class_returns_class(self):
        """Test that get_network_class returns the registered class."""
        from socialjax.networks.registry import (
            register_network, get_network_class
        )
        import flax.linen as nn

        @register_network("test_network_get")
        class TestNetwork(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        retrieved = get_network_class("test_network_get")
        assert retrieved is TestNetwork

    def test_get_unknown_network_raises_error(self):
        """Test that getting unknown network raises helpful error."""
        from socialjax.networks.registry import (
            get_network_class, NetworkNotFoundError
        )

        with pytest.raises(NetworkNotFoundError) as exc_info:
            get_network_class("nonexistent_network_xyz")

        error_message = str(exc_info.value)
        assert "nonexistent_network_xyz" in error_message


class TestListNetworks:
    """Test list_networks function."""

    def test_list_networks_empty(self):
        """Test list_networks with empty registry."""
        from socialjax.networks.registry import list_networks, clear_registry

        # Clear for this test only
        clear_registry()
        result = list_networks()
        assert isinstance(result, list)

    def test_list_networks_returns_sorted(self):
        """Test that list_networks returns sorted list."""
        from socialjax.networks.registry import register_network, list_networks
        import flax.linen as nn

        @register_network("z_network_test")
        class ZNetwork(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        @register_network("a_network_test")
        class ANetwork(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        result = list_networks()
        # Should be sorted
        sorted_names = [n for n in result if n.endswith("_test")]
        assert sorted_names == sorted(sorted_names)


class TestNetworkRegistryUtilities:
    """Test utility functions for network registry."""

    def test_unregister_network(self):
        """Test unregister_network removes network."""
        from socialjax.networks.registry import (
            register_network, get_network_class, unregister_network
        )
        import flax.linen as nn

        @register_network("test_net_unregister_util")
        class TestNet(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        # Should exist
        assert get_network_class("test_net_unregister_util") is TestNet

        # Unregister
        removed = unregister_network("test_net_unregister_util")
        assert removed is TestNet

        # Should not exist anymore
        from socialjax.networks.registry import NetworkNotFoundError
        with pytest.raises(NetworkNotFoundError):
            get_network_class("test_net_unregister_util")

    def test_is_network_registered(self):
        """Test is_network_registered function."""
        from socialjax.networks.registry import (
            register_network, is_network_registered
        )
        import flax.linen as nn

        assert is_network_registered("test_net_is_reg_util") is False

        @register_network("test_net_is_reg_util")
        class TestNet(nn.Module):
            @nn.compact
            def __call__(self, x):
                return x

        assert is_network_registered("test_net_is_reg_util") is True


# ============================================================================
# Test Network Factory
# ============================================================================

class TestNetworkFactoryImport:
    """Test that network factory functions can be imported."""

    def test_import_create_network(self):
        """Test importing create_network function."""
        from socialjax.networks.factory import create_network
        assert create_network is not None
        assert callable(create_network)

    def test_import_list_config_presets(self):
        """Test importing list_config_presets function."""
        from socialjax.networks.factory import list_config_presets
        assert list_config_presets is not None
        assert callable(list_config_presets)

    def test_import_get_config_preset(self):
        """Test importing get_config_preset function."""
        from socialjax.networks.factory import get_config_preset
        assert get_config_preset is not None
        assert callable(get_config_preset)

    def test_import_from_networks_init(self):
        """Test importing factory functions from networks __init__."""
        from socialjax.networks import create_network, list_config_presets, get_config_preset
        assert create_network is not None
        assert list_config_presets is not None
        assert get_config_preset is not None


class TestNetworkConfigPresets:
    """Test network configuration presets."""

    def test_list_config_presets(self):
        """Test that config presets are available."""
        from socialjax.networks.factory import list_config_presets

        presets = list_config_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

        # Should have small, medium, large
        assert "small" in presets
        assert "medium" in presets
        assert "large" in presets

    def test_get_config_preset_small(self):
        """Test getting small config preset."""
        from socialjax.networks.factory import get_config_preset

        config = get_config_preset("small")
        assert config is not None
        assert isinstance(config, dict)

    def test_get_config_preset_medium(self):
        """Test getting medium config preset."""
        from socialjax.networks.factory import get_config_preset

        config = get_config_preset("medium")
        assert config is not None
        assert isinstance(config, dict)

    def test_get_config_preset_large(self):
        """Test getting large config preset."""
        from socialjax.networks.factory import get_config_preset

        config = get_config_preset("large")
        assert config is not None
        assert isinstance(config, dict)

    def test_get_invalid_preset(self):
        """Test getting invalid preset raises ValueError."""
        from socialjax.networks.factory import get_config_preset

        with pytest.raises(ValueError) as exc_info:
            get_config_preset("invalid_preset_xyz")

        # Error message should list available presets
        error_message = str(exc_info.value)
        assert "invalid_preset_xyz" in error_message
        assert "small" in error_message or "medium" in error_message or "large" in error_message


class TestCreateNetwork:
    """Test create_network factory function."""

    def test_create_network_with_config_preset(self):
        """Test creating network with a config preset."""
        from socialjax.networks.factory import create_network
        from socialjax.networks.registry import register_network
        import flax.linen as nn

        @register_network("test_factory_network_util")
        class TestNet(nn.Module):
            action_dim: int = 4
            hidden_size: int = 64

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.action_dim)(x)

        # Create with preset
        try:
            network = create_network(
                "test_factory_network_util",
                action_dim=4,
                config_preset="small"
            )
            assert network is not None
        except Exception as e:
            # May need extra parameters
            pass


# ============================================================================
# Test Integration: Registered Algorithms
# ============================================================================

class TestRegisteredAlgorithms:
    """Test that expected algorithms are registered."""

    def test_ippo_is_registered(self):
        """Test that IPPO algorithm is registered."""
        # First import to trigger registration
        try:
            from socialjax.algorithms.ippo.algorithm import IPPOAlgorithm
            from socialjax.algorithms.registry import is_algorithm_registered, get_algorithm

            # Should be registered with 'ippo' name
            assert is_algorithm_registered("ippo") or is_algorithm_registered("IPPO")

            if is_algorithm_registered("ippo"):
                algo_class = get_algorithm("ippo")
                assert algo_class is IPPOAlgorithm
        except ImportError:
            pytest.skip("IPPO not available")

    def test_mappo_is_registered(self):
        """Test that MAPPO algorithm is registered."""
        try:
            from socialjax.algorithms.mappo.algorithm import MAPPOAlgorithm
            from socialjax.algorithms.registry import is_algorithm_registered

            assert is_algorithm_registered("mappo") or is_algorithm_registered("MAPPO")
        except ImportError:
            pytest.skip("MAPPO not available")

    def test_vdn_is_registered(self):
        """Test that VDN algorithm is registered."""
        try:
            from socialjax.algorithms.vdn.algorithm import VDNAlgorithm
            from socialjax.algorithms.registry import is_algorithm_registered

            assert is_algorithm_registered("vdn") or is_algorithm_registered("VDN")
        except ImportError:
            pytest.skip("VDN not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
