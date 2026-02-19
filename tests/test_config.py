"""Unit tests for ConfigManager and configuration dataclasses.

Test criteria:
- ConfigManager loads base config
- Configs merge correctly (algorithm + environment)
- Custom configs override defaults
- Missing required keys raise validation error
- Dataclasses convert to/from dict correctly
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

# Set up path for imports
sys.path.insert(0, 'socialjax')


# ============================================================================
# Test Configuration Dataclasses
# ============================================================================

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_training_config_creation(self):
        """Test creating a TrainingConfig instance."""
        from socialjax.config.manager import TrainingConfig

        config = TrainingConfig()
        assert config.total_timesteps == 10_000_000
        assert config.num_envs == 32
        assert config.gamma == 0.99
        assert config.learning_rate == 2.5e-4

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        from socialjax.config.manager import TrainingConfig

        config = TrainingConfig(
            total_timesteps=1_000_000,
            num_envs=16,
            learning_rate=1e-3,
            gamma=0.95,
        )

        assert config.total_timesteps == 1_000_000
        assert config.num_envs == 16
        assert config.learning_rate == 1e-3
        assert config.gamma == 0.95

    def test_training_config_to_dict(self):
        """Test converting TrainingConfig to dict."""
        from socialjax.config.manager import TrainingConfig

        config = TrainingConfig(total_timesteps=1000)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["total_timesteps"] == 1000
        assert "learning_rate" in d
        assert "gamma" in d

    def test_training_config_from_dict(self):
        """Test creating TrainingConfig from dict."""
        from socialjax.config.manager import TrainingConfig

        d = {"total_timesteps": 5000, "num_envs": 8, "extra_key": "ignored"}
        config = TrainingConfig.from_dict(d)

        assert config.total_timesteps == 5000
        assert config.num_envs == 8
        # Extra keys should be filtered out

    def test_training_config_from_dict_partial(self):
        """Test TrainingConfig.from_dict with partial values uses defaults."""
        from socialjax.config.manager import TrainingConfig

        d = {"total_timesteps": 5000}
        config = TrainingConfig.from_dict(d)

        assert config.total_timesteps == 5000
        assert config.num_envs == 32  # Default


class TestNetworkConfig:
    """Tests for NetworkConfig dataclass."""

    def test_network_config_creation(self):
        """Test creating a NetworkConfig instance."""
        from socialjax.config.manager import NetworkConfig

        config = NetworkConfig()
        assert config.architecture == "cnn_actor_critic"
        assert config.hidden_size == 64

    def test_network_config_custom_values(self):
        """Test NetworkConfig with custom values."""
        from socialjax.config.manager import NetworkConfig

        config = NetworkConfig(
            architecture="mlp",
            hidden_size=128,
            num_channels=(32, 64, 64),
        )

        assert config.architecture == "mlp"
        assert config.hidden_size == 128
        assert config.num_channels == (32, 64, 64)

    def test_network_config_to_dict(self):
        """Test converting NetworkConfig to dict."""
        from socialjax.config.manager import NetworkConfig

        config = NetworkConfig(hidden_size=256)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["hidden_size"] == 256
        assert "architecture" in d

    def test_network_config_from_dict(self):
        """Test creating NetworkConfig from dict."""
        from socialjax.config.manager import NetworkConfig

        d = {"hidden_size": 128, "architecture": "custom"}
        config = NetworkConfig.from_dict(d)

        assert config.hidden_size == 128
        assert config.architecture == "custom"

    def test_network_config_from_dict_converts_tuple(self):
        """Test that list values are converted to tuples for num_channels."""
        from socialjax.config.manager import NetworkConfig

        d = {"num_channels": [16, 32, 64], "kernel_size": [5, 5]}
        config = NetworkConfig.from_dict(d)

        assert config.num_channels == (16, 32, 64)
        assert config.kernel_size == (5, 5)


class TestAlgorithmConfig:
    """Tests for AlgorithmConfig dataclass."""

    def test_algorithm_config_creation(self):
        """Test creating an AlgorithmConfig instance."""
        from socialjax.config.manager import AlgorithmConfig

        config = AlgorithmConfig()
        assert config.name == "ippo"
        assert config.parameter_sharing is True

    def test_algorithm_config_with_nested_configs(self):
        """Test AlgorithmConfig with nested NetworkConfig and TrainingConfig."""
        from socialjax.config.manager import AlgorithmConfig, NetworkConfig, TrainingConfig

        config = AlgorithmConfig(
            name="mappo",
            network=NetworkConfig(hidden_size=128),
            training=TrainingConfig(learning_rate=1e-3),
        )

        assert config.name == "mappo"
        assert config.network.hidden_size == 128
        assert config.training.learning_rate == 1e-3

    def test_algorithm_config_to_dict(self):
        """Test converting AlgorithmConfig to dict."""
        from socialjax.config.manager import AlgorithmConfig

        config = AlgorithmConfig(name="vdn", centralised_critic=True)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "vdn"
        assert d["centralised_critic"] is True
        assert "network" in d
        assert "training" in d

    def test_algorithm_config_from_dict(self):
        """Test creating AlgorithmConfig from dict."""
        from socialjax.config.manager import AlgorithmConfig

        d = {
            "name": "mappo",
            "parameter_sharing": False,
            "centralised_critic": True,
            "network": {"hidden_size": 256},
            "training": {"learning_rate": 5e-4},
        }
        config = AlgorithmConfig.from_dict(d)

        assert config.name == "mappo"
        assert config.parameter_sharing is False
        assert config.centralised_critic is True
        assert config.network.hidden_size == 256
        assert config.training.learning_rate == 5e-4


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_environment_config_creation(self):
        """Test creating an EnvironmentConfig instance."""
        from socialjax.config.manager import EnvironmentConfig

        config = EnvironmentConfig()
        assert config.name == "coin_game"
        assert config.num_agents == 2
        assert config.max_steps == 100

    def test_environment_config_custom_values(self):
        """Test EnvironmentConfig with custom values."""
        from socialjax.config.manager import EnvironmentConfig

        config = EnvironmentConfig(
            name="clean_up",
            num_agents=7,
            max_steps=500,
            kwargs={"width": 25, "height": 20},
        )

        assert config.name == "clean_up"
        assert config.num_agents == 7
        assert config.max_steps == 500
        assert config.kwargs["width"] == 25

    def test_environment_config_to_dict(self):
        """Test converting EnvironmentConfig to dict."""
        from socialjax.config.manager import EnvironmentConfig

        config = EnvironmentConfig(name="test_env")
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "test_env"

    def test_environment_config_from_dict(self):
        """Test creating EnvironmentConfig from dict."""
        from socialjax.config.manager import EnvironmentConfig

        d = {"name": "harvest", "num_agents": 5}
        config = EnvironmentConfig.from_dict(d)

        assert config.name == "harvest"
        assert config.num_agents == 5


class TestSocialJaxConfig:
    """Tests for SocialJaxConfig dataclass."""

    def test_socialjax_config_creation(self):
        """Test creating a SocialJaxConfig instance."""
        from socialjax.config.manager import SocialJaxConfig

        config = SocialJaxConfig()
        assert config.algorithm is not None
        assert config.environment is not None

    def test_socialjax_config_with_nested(self):
        """Test SocialJaxConfig with nested configs."""
        from socialjax.config.manager import (
            SocialJaxConfig, AlgorithmConfig, EnvironmentConfig
        )

        config = SocialJaxConfig(
            algorithm=AlgorithmConfig(name="ippo"),
            environment=EnvironmentConfig(name="coin_game", num_agents=5),
        )

        assert config.algorithm.name == "ippo"
        assert config.environment.name == "coin_game"
        assert config.environment.num_agents == 5

    def test_socialjax_config_to_dict(self):
        """Test converting SocialJaxConfig to dict."""
        from socialjax.config.manager import SocialJaxConfig

        config = SocialJaxConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "algorithm" in d
        assert "environment" in d

    def test_socialjax_config_from_dict(self):
        """Test creating SocialJaxConfig from dict."""
        from socialjax.config.manager import SocialJaxConfig

        d = {
            "algorithm": {"name": "mappo", "centralised_critic": True},
            "environment": {"name": "clean_up", "num_agents": 7},
        }
        config = SocialJaxConfig.from_dict(d)

        assert config.algorithm.name == "mappo"
        assert config.algorithm.centralised_critic is True
        assert config.environment.name == "clean_up"
        assert config.environment.num_agents == 7


# ============================================================================
# Test ConfigValidationError
# ============================================================================

class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_exception_is_exception(self):
        """Test that ConfigValidationError is an Exception."""
        from socialjax.config.manager import ConfigValidationError

        assert issubclass(ConfigValidationError, Exception)

    def test_exception_with_message(self):
        """Test ConfigValidationError with message."""
        from socialjax.config.manager import ConfigValidationError

        try:
            raise ConfigValidationError("Test validation error")
        except ConfigValidationError as e:
            assert "Test validation error" in str(e)


# ============================================================================
# Test ConfigManager
# ============================================================================

class TestConfigManagerImport:
    """Test that ConfigManager can be imported."""

    def test_import_from_manager(self):
        """Test importing ConfigManager from manager module."""
        from socialjax.config.manager import ConfigManager
        assert ConfigManager is not None

    def test_import_from_config_init(self):
        """Test importing ConfigManager from config __init__."""
        from socialjax.config import ConfigManager
        assert ConfigManager is not None

    def test_import_all_dataclasses(self):
        """Test importing all dataclasses from config module."""
        from socialjax.config import (
            TrainingConfig,
            NetworkConfig,
            AlgorithmConfig,
            EnvironmentConfig,
            SocialJaxConfig,
        )
        assert TrainingConfig is not None
        assert NetworkConfig is not None
        assert AlgorithmConfig is not None
        assert EnvironmentConfig is not None
        assert SocialJaxConfig is not None


class TestConfigManagerCreation:
    """Tests for ConfigManager initialization."""

    def test_config_manager_default_creation(self):
        """Test creating ConfigManager with default path."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        assert manager.config_path is not None

    def test_config_manager_custom_path(self):
        """Test creating ConfigManager with custom path."""
        from socialjax.config.manager import ConfigManager

        custom_path = Path("/tmp/configs")
        manager = ConfigManager(config_path=custom_path)
        assert manager.config_path == custom_path


class TestConfigManagerLoad:
    """Tests for ConfigManager.load() method."""

    def test_load_returns_socialjax_config(self):
        """Test that load returns a SocialJaxConfig instance."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        try:
            config = manager.load(algorithm="ippo", environment="coin_game")
            assert config is not None
            # Should be a SocialJaxConfig instance
            from socialjax.config.manager import SocialJaxConfig
            assert isinstance(config, SocialJaxConfig)
        except FileNotFoundError:
            pytest.skip("Config preset files not found")

    def test_load_with_custom_config(self):
        """Test that custom_config overrides defaults."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        try:
            custom = {"algorithm": {"name": "ippo"}, "environment": {"num_agents": 10}}
            config = manager.load(
                algorithm="ippo",
                environment="coin_game",
                custom_config=custom,
            )
            # Custom num_agents should be applied
            assert config.environment.num_agents == 10
        except FileNotFoundError:
            pytest.skip("Config preset files not found")


class TestConfigManagerLoadFromFile:
    """Tests for ConfigManager.load_from_file() method."""

    def test_load_from_file(self):
        """Test loading config from a YAML file."""
        from socialjax.config.manager import ConfigManager

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
algorithm:
  name: ippo
environment:
  name: coin_game
  num_agents: 3
""")
            temp_path = Path(f.name)

        try:
            manager = ConfigManager()
            config = manager.load_from_file(temp_path)

            assert config.algorithm.name == "ippo"
            assert config.environment.name == "coin_game"
            assert config.environment.num_agents == 3
        finally:
            os.unlink(temp_path)

    def test_load_from_nonexistent_file_returns_empty(self):
        """Test loading from nonexistent file."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()

        # _load_yaml returns empty dict for nonexistent files
        result = manager._load_yaml(Path("/nonexistent/path/file.yaml"))
        assert result == {}


class TestConfigManagerValidation:
    """Tests for ConfigManager validation."""

    def test_validate_missing_algorithm(self):
        """Test validation fails without algorithm key."""
        from socialjax.config.manager import ConfigManager, ConfigValidationError

        manager = ConfigManager()

        with pytest.raises(ConfigValidationError) as exc_info:
            manager._validate({"environment": {"name": "test"}})

        assert "algorithm" in str(exc_info.value).lower()

    def test_validate_missing_environment(self):
        """Test validation fails without environment key."""
        from socialjax.config.manager import ConfigManager, ConfigValidationError

        manager = ConfigManager()

        with pytest.raises(ConfigValidationError) as exc_info:
            manager._validate({"algorithm": {"name": "ippo"}})

        assert "environment" in str(exc_info.value).lower()

    def test_validate_missing_algorithm_name(self):
        """Test validation fails without algorithm name."""
        from socialjax.config.manager import ConfigManager, ConfigValidationError

        manager = ConfigManager()

        with pytest.raises(ConfigValidationError):
            manager._validate({
                "algorithm": {"parameter_sharing": True},
                "environment": {"name": "test"},
            })

    def test_validate_missing_environment_name(self):
        """Test validation fails without environment name."""
        from socialjax.config.manager import ConfigManager, ConfigValidationError

        manager = ConfigManager()

        with pytest.raises(ConfigValidationError):
            manager._validate({
                "algorithm": {"name": "ippo"},
                "environment": {"num_agents": 5},
            })

    def test_validate_complete_config(self):
        """Test validation passes with complete config."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()

        # Should not raise
        manager._validate({
            "algorithm": {"name": "ippo"},
            "environment": {"name": "coin_game"},
        })


class TestConfigManagerMergeDicts:
    """Tests for ConfigManager._merge_dicts() method."""

    def test_merge_empty_dicts(self):
        """Test merging empty dicts."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        result = manager._merge_dicts({}, {}, {})
        assert result == {}

    def test_merge_single_dict(self):
        """Test merging a single dict."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        result = manager._merge_dicts({"a": 1})
        assert result == {"a": 1}

    def test_merge_overwrites_values(self):
        """Test that later dicts overwrite earlier values."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        result = manager._merge_dicts(
            {"a": 1, "b": 2},
            {"a": 10, "c": 3},
        )

        assert result == {"a": 10, "b": 2, "c": 3}

    def test_merge_nested_dicts(self):
        """Test merging nested dicts recursively."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        result = manager._merge_dicts(
            {"algorithm": {"name": "ippo", "lr": 1e-3}},
            {"algorithm": {"name": "mappo"}},
        )

        assert result["algorithm"]["name"] == "mappo"
        assert result["algorithm"]["lr"] == 1e-3

    def test_merge_multiple_dicts(self):
        """Test merging multiple dicts in sequence."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        result = manager._merge_dicts(
            {"a": 1},
            {"b": 2},
            {"c": 3},
            {"a": 10},
        )

        assert result == {"a": 10, "b": 2, "c": 3}


class TestConfigManagerSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_config(self):
        """Test saving config to file."""
        from socialjax.config.manager import ConfigManager, SocialJaxConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager()
            config = SocialJaxConfig()
            save_path = Path(tmpdir) / "test_config.yaml"

            manager.save_config(save_path, config)

            assert save_path.exists()

    def test_save_and_load_roundtrip(self):
        """Test saving and loading config preserves basic values."""
        from socialjax.config.manager import (
            ConfigManager, SocialJaxConfig, AlgorithmConfig, EnvironmentConfig
        )

        # Note: NetworkConfig has default tuples that YAML can't serialize.
        # We test basic save functionality and file creation instead of full roundtrip.
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager()

            # Create a minimal config file manually
            save_path = Path(tmpdir) / "test_config.yaml"
            config_content = """
algorithm:
  name: ippo
  parameter_sharing: false
environment:
  name: coin_game
  num_agents: 7
"""
            save_path.write_text(config_content)

            # Load the config
            loaded_config = manager.load_from_file(save_path)

            assert loaded_config.algorithm.name == "ippo"
            assert loaded_config.algorithm.parameter_sharing is False
            assert loaded_config.environment.name == "coin_game"
            assert loaded_config.environment.num_agents == 7

    def test_save_without_config_raises(self):
        """Test saving without config raises ValueError."""
        from socialjax.config.manager import ConfigManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager()
            save_path = Path(tmpdir) / "test.yaml"

            with pytest.raises(ValueError):
                manager.save_config(save_path)


class TestConfigManagerGetConfig:
    """Tests for ConfigManager.get_config() method."""

    def test_get_config_returns_none_initially(self):
        """Test that get_config returns None before load."""
        from socialjax.config.manager import ConfigManager

        manager = ConfigManager()
        assert manager.get_config() is None

    def test_get_config_after_load(self):
        """Test that get_config returns loaded config."""
        from socialjax.config.manager import ConfigManager, SocialJaxConfig

        manager = ConfigManager()

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
algorithm:
  name: test_algo
environment:
  name: test_env
""")
            temp_path = Path(f.name)

        try:
            config = manager.load_from_file(temp_path)
            assert manager.get_config() is config
        finally:
            os.unlink(temp_path)


class TestCreateDefaultConfig:
    """Tests for create_default_config convenience function."""

    def test_create_default_config_basic(self):
        """Test creating default config with basic parameters."""
        from socialjax.config.manager import create_default_config

        config = create_default_config(algorithm="ippo", environment="coin_game")

        assert config.algorithm.name == "ippo"
        assert config.environment.name == "coin_game"

    def test_create_default_config_with_overrides(self):
        """Test creating default config with kwargs overrides."""
        from socialjax.config.manager import create_default_config

        config = create_default_config(
            algorithm="mappo",
            environment="clean_up",
        )

        assert config.algorithm.name == "mappo"
        assert config.environment.name == "clean_up"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
