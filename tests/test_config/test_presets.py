"""Unit tests for configuration preset files.

This module tests:
- All preset YAML files are valid
- ConfigManager can load each preset
- Preset values match proposal specifications
- Inheritance chain works correctly
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add socialjax to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "socialjax"))

from config.manager import (
    ConfigManager,
    SocialJaxConfig,
    AlgorithmConfig,
    EnvironmentConfig,
    TrainingConfig,
    NetworkConfig,
)


# Path to presets directory
PRESETS_DIR = Path(__file__).parent.parent.parent / "socialjax" / "config" / "presets"


class TestAllPresetsValid:
    """Test that all preset YAML files are valid."""

    def test_base_yaml_valid(self):
        """Test that base.yaml is valid YAML."""
        base_path = PRESETS_DIR / "base.yaml"
        assert base_path.exists(), "base.yaml does not exist"

        with open(base_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "base.yaml is empty"
        assert "algorithm" in data, "base.yaml missing 'algorithm' key"
        assert "environment" in data, "base.yaml missing 'environment' key"

    def test_ippo_yaml_valid(self):
        """Test that IPPO preset is valid YAML."""
        ippo_path = PRESETS_DIR / "algorithms" / "ippo.yaml"
        assert ippo_path.exists(), "ippo.yaml does not exist"

        with open(ippo_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "ippo.yaml is empty"
        assert "algorithm" in data
        assert data["algorithm"]["name"] == "ippo"

    def test_mappo_yaml_valid(self):
        """Test that MAPPO preset is valid YAML."""
        mappo_path = PRESETS_DIR / "algorithms" / "mappo.yaml"
        assert mappo_path.exists(), "mappo.yaml does not exist"

        with open(mappo_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "mappo.yaml is empty"
        assert "algorithm" in data
        assert data["algorithm"]["name"] == "mappo"

    def test_vdn_yaml_valid(self):
        """Test that VDN preset is valid YAML."""
        vdn_path = PRESETS_DIR / "algorithms" / "vdn.yaml"
        assert vdn_path.exists(), "vdn.yaml does not exist"

        with open(vdn_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "vdn.yaml is empty"
        assert "algorithm" in data
        assert data["algorithm"]["name"] == "vdn"

    def test_svo_yaml_valid(self):
        """Test that SVO preset is valid YAML."""
        svo_path = PRESETS_DIR / "algorithms" / "svo.yaml"
        assert svo_path.exists(), "svo.yaml does not exist"

        with open(svo_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "svo.yaml is empty"
        assert "algorithm" in data
        assert data["algorithm"]["name"] == "svo"

    def test_coin_game_yaml_valid(self):
        """Test that coin_game preset is valid YAML."""
        coin_path = PRESETS_DIR / "environments" / "coin_game.yaml"
        assert coin_path.exists(), "coin_game.yaml does not exist"

        with open(coin_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "coin_game.yaml is empty"
        assert "environment" in data
        assert data["environment"]["name"] == "coin_game"

    def test_cleanup_yaml_valid(self):
        """Test that cleanup preset is valid YAML."""
        cleanup_path = PRESETS_DIR / "environments" / "cleanup.yaml"
        assert cleanup_path.exists(), "cleanup.yaml does not exist"

        with open(cleanup_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "cleanup.yaml is empty"
        assert "environment" in data
        assert data["environment"]["name"] == "clean_up"

    def test_harvest_open_yaml_valid(self):
        """Test that harvest_open preset is valid YAML."""
        harvest_path = PRESETS_DIR / "environments" / "harvest_open.yaml"
        assert harvest_path.exists(), "harvest_open.yaml does not exist"

        with open(harvest_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "harvest_open.yaml is empty"
        assert "environment" in data
        assert data["environment"]["name"] == "harvest_common_open"

    def test_coop_mining_yaml_valid(self):
        """Test that coop_mining preset is valid YAML."""
        mining_path = PRESETS_DIR / "environments" / "coop_mining.yaml"
        assert mining_path.exists(), "coop_mining.yaml does not exist"

        with open(mining_path) as f:
            data = yaml.safe_load(f)

        assert data is not None, "coop_mining.yaml is empty"
        assert "environment" in data
        assert data["environment"]["name"] == "coop_mining"


class TestPresetLoading:
    """Test that ConfigManager can load each preset."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance."""
        return ConfigManager()

    def test_load_ippo_coin_game(self, config_manager):
        """Test loading IPPO with coin_game environment."""
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        assert isinstance(config, SocialJaxConfig)
        assert config.algorithm.name == "ippo"
        assert config.environment.name == "coin_game"

    def test_load_mappo_cleanup(self, config_manager):
        """Test loading MAPPO with cleanup environment."""
        config = config_manager.load(algorithm="mappo", environment="cleanup")

        assert isinstance(config, SocialJaxConfig)
        assert config.algorithm.name == "mappo"
        assert config.environment.name == "clean_up"

    def test_load_vdn_harvest_open(self, config_manager):
        """Test loading VDN with harvest_open environment."""
        config = config_manager.load(algorithm="vdn", environment="harvest_open")

        assert isinstance(config, SocialJaxConfig)
        assert config.algorithm.name == "vdn"
        assert config.environment.name == "harvest_common_open"

    def test_load_svo_coop_mining(self, config_manager):
        """Test loading SVO with coop_mining environment."""
        config = config_manager.load(algorithm="svo", environment="coop_mining")

        assert isinstance(config, SocialJaxConfig)
        assert config.algorithm.name == "svo"
        assert config.environment.name == "coop_mining"

    def test_load_with_custom_config(self, config_manager):
        """Test loading with custom config overrides."""
        custom = {"algorithm": {"training": {"learning_rate": 0.001}}}
        config = config_manager.load(
            algorithm="ippo",
            environment="coin_game",
            custom_config=custom
        )

        assert config.algorithm.training.learning_rate == 0.001


class TestPresetValues:
    """Test that preset values match expected specifications."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance."""
        return ConfigManager()

    def test_ippo_preset_values(self, config_manager):
        """Test IPPO preset has correct values."""
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        # Check IPPO-specific values
        assert config.algorithm.name == "ippo"
        assert config.algorithm.parameter_sharing is True
        assert config.algorithm.centralised_critic is False

        # Check training values
        assert config.algorithm.training.learning_rate == 0.00025
        assert config.algorithm.training.clip_eps == 0.2
        assert config.algorithm.training.ent_coef == 0.01

    def test_mappo_preset_values(self, config_manager):
        """Test MAPPO preset has correct values."""
        config = config_manager.load(algorithm="mappo", environment="cleanup")

        # Check MAPPO-specific values
        assert config.algorithm.name == "mappo"
        assert config.algorithm.centralised_critic is True

        # Check training values
        assert config.algorithm.training.learning_rate == 0.0005

    def test_vdn_preset_values(self, config_manager):
        """Test VDN preset has correct values."""
        config = config_manager.load(algorithm="vdn", environment="harvest_open")

        # Check VDN-specific values
        assert config.algorithm.name == "vdn"
        assert config.algorithm.target_update_freq == 200
        assert config.algorithm.target_update_tau == 1.0

        # Check training values
        assert config.algorithm.training.learning_rate == 0.001
        assert config.algorithm.training.gamma == 0.99

    def test_svo_preset_values(self, config_manager):
        """Test SVO preset has correct values."""
        config = config_manager.load(algorithm="svo", environment="coop_mining")

        # Check SVO-specific values
        assert config.algorithm.name == "svo"

        # Check training values (SVO uses PPO-style training)
        assert config.algorithm.training.learning_rate == 0.00025
        assert config.algorithm.training.clip_eps == 0.2

    def test_coin_game_preset_values(self, config_manager):
        """Test coin_game preset has correct values."""
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        assert config.environment.name == "coin_game"
        assert config.environment.num_agents == 5
        assert config.environment.max_steps == 100

    def test_cleanup_preset_values(self, config_manager):
        """Test cleanup preset has correct values."""
        config = config_manager.load(algorithm="mappo", environment="cleanup")

        assert config.environment.name == "clean_up"
        assert config.environment.num_agents == 7
        assert config.environment.max_steps == 500

    def test_harvest_open_preset_values(self, config_manager):
        """Test harvest_open preset has correct values."""
        config = config_manager.load(algorithm="vdn", environment="harvest_open")

        assert config.environment.name == "harvest_common_open"
        assert config.environment.num_agents == 7
        assert config.environment.max_steps == 1000

    def test_coop_mining_preset_values(self, config_manager):
        """Test coop_mining preset has correct values."""
        config = config_manager.load(algorithm="svo", environment="coop_mining")

        assert config.environment.name == "coop_mining"
        assert config.environment.num_agents == 4
        assert config.environment.max_steps == 500


class TestInheritanceChain:
    """Test that the config inheritance chain works correctly."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance."""
        return ConfigManager()

    def test_base_values_inherited(self, config_manager):
        """Test that base values are inherited when loading algorithm preset."""
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        # These values come from base.yaml
        assert config.algorithm.training.gamma == 0.99
        assert config.algorithm.training.gae_lambda == 0.95
        assert config.algorithm.training.update_epochs == 4
        assert config.algorithm.training.num_minibatches == 4

    def test_algorithm_overrides_base(self, config_manager):
        """Test that algorithm preset values override base values."""
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        # IPPO sets learning_rate to 0.00025
        assert config.algorithm.training.learning_rate == 0.00025

    def test_environment_overrides_base(self, config_manager):
        """Test that environment preset values override base values."""
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        # coin_game sets num_agents to 5
        assert config.environment.num_agents == 5

    def test_custom_overrides_all(self, config_manager):
        """Test that custom config overrides everything."""
        custom = {
            "algorithm": {"training": {"learning_rate": 0.1}},
            "environment": {"num_agents": 10}
        }
        config = config_manager.load(
            algorithm="ippo",
            environment="coin_game",
            custom_config=custom
        )

        assert config.algorithm.training.learning_rate == 0.1
        assert config.environment.num_agents == 10

    def test_mixed_inheritance(self, config_manager):
        """Test inheritance with partial overrides."""
        custom = {"algorithm": {"training": {"ent_coef": 0.05}}}
        config = config_manager.load(
            algorithm="ippo",
            environment="coin_game",
            custom_config=custom
        )

        # Custom override
        assert config.algorithm.training.ent_coef == 0.05
        # Algorithm preset value
        assert config.algorithm.training.learning_rate == 0.00025
        # Base value
        assert config.algorithm.training.gamma == 0.99


class TestConfigManagerEdgeCases:
    """Test edge cases in ConfigManager preset loading."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance."""
        return ConfigManager()

    def test_all_algorithm_environment_combinations(self, config_manager):
        """Test all algorithm-environment combinations load successfully."""
        algorithms = ["ippo", "mappo", "vdn", "svo"]
        environments = ["coin_game", "cleanup", "harvest_open", "coop_mining"]

        for algo in algorithms:
            for env in environments:
                config = config_manager.load(algorithm=algo, environment=env)
                assert isinstance(config, SocialJaxConfig)
                assert config.algorithm.name == algo

    def test_config_to_dict_roundtrip(self, config_manager):
        """Test that config can be converted to dict and back."""
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        # Convert to dict
        config_dict = config.to_dict()

        # Verify dict structure
        assert "algorithm" in config_dict
        assert "environment" in config_dict
        assert "training" in config_dict["algorithm"]
        assert "network" in config_dict["algorithm"]

        # Convert back
        config2 = SocialJaxConfig.from_dict(config_dict)
        assert config2.algorithm.name == config.algorithm.name
        assert config2.environment.name == config.environment.name

    def test_save_and_load_config(self, config_manager, tmp_path):
        """Test saving and loading config from file.

        Note: YAML safe_load cannot deserialize Python tuples. This test
        uses a custom YAML dumper to handle tuples as lists, then the
        from_dict method handles conversion back.
        """
        config = config_manager.load(algorithm="ippo", environment="coin_game")

        # Save to temp file using custom handling for tuples
        save_path = tmp_path / "test_config.yaml"

        # Convert to dict first, then save manually with tuple handling
        config_dict = config.to_dict()

        # Convert tuples to lists for YAML compatibility
        def convert_tuples_to_lists(obj):
            if isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tuples_to_lists(item) for item in obj]
            return obj

        yaml_dict = convert_tuples_to_lists(config_dict)

        import yaml
        with open(save_path, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)

        # Load from file
        loaded_config = config_manager.load_from_file(save_path)

        assert loaded_config.algorithm.name == config.algorithm.name
        assert loaded_config.environment.name == config.environment.name
        # Note: tuple fields (like num_channels, kernel_size) will be lists after roundtrip
        # This is acceptable since from_dict handles list->tuple conversion


class TestPresetFilesExist:
    """Test that all required preset files exist."""

    def test_all_algorithm_presets_exist(self):
        """Test that all algorithm preset files exist."""
        required_algorithms = ["ippo", "mappo", "vdn", "svo"]

        for algo in required_algorithms:
            preset_path = PRESETS_DIR / "algorithms" / f"{algo}.yaml"
            assert preset_path.exists(), f"Missing algorithm preset: {algo}.yaml"

    def test_all_environment_presets_exist(self):
        """Test that all environment preset files exist."""
        required_environments = ["coin_game", "cleanup", "harvest_open", "coop_mining"]

        for env in required_environments:
            preset_path = PRESETS_DIR / "environments" / f"{env}.yaml"
            assert preset_path.exists(), f"Missing environment preset: {env}.yaml"

    def test_base_preset_exists(self):
        """Test that base.yaml preset exists."""
        base_path = PRESETS_DIR / "base.yaml"
        assert base_path.exists(), "Missing base.yaml preset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
