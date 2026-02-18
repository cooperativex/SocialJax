"""Unit tests for MAPPO configuration."""

import sys
sys.path.insert(0, 'socialjax')

# Make pytest optional
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy pytest fixture
    class pytest:
        @staticmethod
        def fixture(func):
            return func

from socialjax.algorithms.mappo.config import (
    MAPPO_DEFAULT_CONFIG,
    get_mappo_config,
)


class TestMAPPOConfig:
    """Tests for MAPPO configuration."""

    def test_default_config_exists(self):
        """Test that default config has required keys."""
        required_keys = [
            "LR",
            "LR_ACTOR",
            "LR_CRITIC",
            "ANNEAL_LR",
            "GAMMA",
            "GAE_LAMBDA",
            "CLIP_EPS",
            "SCALE_CLIP_EPS",
            "VF_COEF",
            "ENT_COEF",
            "MAX_GRAD_NORM",
            "UPDATE_EPOCHS",
            "NUM_MINIBATCHES",
            "NUM_STEPS",
            "ACTIVATION",
            "HIDDEN_SIZE",
            "PARAMETER_SHARING",
            "USE_CENTRALIZED_VALUE",
            "POPULATE_CRITIC_VALUE",
        ]
        for key in required_keys:
            assert key in MAPPO_DEFAULT_CONFIG, f"Missing key: {key}"

    def test_default_config_values(self):
        """Test that default config values are reasonable."""
        assert MAPPO_DEFAULT_CONFIG["LR"] == 2.5e-4
        assert MAPPO_DEFAULT_CONFIG["GAMMA"] == 0.99
        assert MAPPO_DEFAULT_CONFIG["GAE_LAMBDA"] == 0.95
        assert MAPPO_DEFAULT_CONFIG["CLIP_EPS"] == 0.2
        assert MAPPO_DEFAULT_CONFIG["VF_COEF"] == 0.5
        assert MAPPO_DEFAULT_CONFIG["ENT_COEF"] == 0.01
        assert MAPPO_DEFAULT_CONFIG["MAX_GRAD_NORM"] == 0.5
        assert MAPPO_DEFAULT_CONFIG["UPDATE_EPOCHS"] == 4
        assert MAPPO_DEFAULT_CONFIG["NUM_MINIBATCHES"] == 4
        assert MAPPO_DEFAULT_CONFIG["NUM_STEPS"] == 128
        assert MAPPO_DEFAULT_CONFIG["ACTIVATION"] == "relu"
        assert MAPPO_DEFAULT_CONFIG["HIDDEN_SIZE"] == 64
        assert MAPPO_DEFAULT_CONFIG["PARAMETER_SHARING"] is True
        assert MAPPO_DEFAULT_CONFIG["USE_CENTRALIZED_VALUE"] is True

    def test_get_mappo_config_no_overrides(self):
        """Test get_mappo_config without overrides returns copy of defaults."""
        config = get_mappo_config()
        assert config is not MAPPO_DEFAULT_CONFIG  # Should be a copy
        # get_mappo_config sets LR_ACTOR and LR_CRITIC to LR if they were None
        # So the configs won't be exactly equal, but key values should match
        assert config["LR"] == MAPPO_DEFAULT_CONFIG["LR"]
        assert config["GAMMA"] == MAPPO_DEFAULT_CONFIG["GAMMA"]
        assert config["CLIP_EPS"] == MAPPO_DEFAULT_CONFIG["CLIP_EPS"]

    def test_get_mappo_config_with_overrides(self):
        """Test get_mappo_config with overrides applies them correctly."""
        overrides = {
            "LR": 0.001,
            "GAMMA": 0.95,
            "HIDDEN_SIZE": 128,
        }
        config = get_mappo_config(overrides)
        assert config["LR"] == 0.001
        assert config["GAMMA"] == 0.95
        assert config["HIDDEN_SIZE"] == 128
        # Other values should remain default
        assert config["GAE_LAMBDA"] == 0.95
        assert config["CLIP_EPS"] == 0.2

    def test_get_mappo_config_lr_defaults(self):
        """Test that LR_ACTOR and LR_CRITIC default to LR if not specified."""
        # When no separate LR specified
        config = get_mappo_config({"LR": 0.001})
        assert config["LR_ACTOR"] == 0.001
        assert config["LR_CRITIC"] == 0.001

        # When separate LR specified
        config = get_mappo_config({
            "LR": 0.001,
            "LR_ACTOR": 0.0005,
            "LR_CRITIC": 0.001,
        })
        assert config["LR_ACTOR"] == 0.0005
        assert config["LR_CRITIC"] == 0.001

    def test_get_mappo_config_none_overrides(self):
        """Test get_mappo_config with None overrides returns valid config."""
        config = get_mappo_config(None)
        # Key values should match defaults
        assert config["LR"] == MAPPO_DEFAULT_CONFIG["LR"]
        assert config["GAMMA"] == MAPPO_DEFAULT_CONFIG["GAMMA"]
        assert config["CLIP_EPS"] == MAPPO_DEFAULT_CONFIG["CLIP_EPS"]


class TestMAPPOConfigValidation:
    """Tests for MAPPO configuration validation."""

    def test_valid_activation_values(self):
        """Test that activation accepts valid values."""
        for activation in ["relu", "tanh"]:
            config = get_mappo_config({"ACTIVATION": activation})
            assert config["ACTIVATION"] == activation

    def test_positive_learning_rate(self):
        """Test that learning rate should be positive."""
        config = get_mappo_config({"LR": 0.001})
        assert config["LR"] > 0

    def test_gamma_in_valid_range(self):
        """Test that gamma is in valid range [0, 1]."""
        config = get_mappo_config()
        assert 0 <= config["GAMMA"] <= 1

    def test_gae_lambda_in_valid_range(self):
        """Test that GAE lambda is in valid range [0, 1]."""
        config = get_mappo_config()
        assert 0 <= config["GAE_LAMBDA"] <= 1

    def test_clip_eps_positive(self):
        """Test that clip epsilon is positive."""
        config = get_mappo_config()
        assert config["CLIP_EPS"] > 0


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Run tests manually
        print("Running MAPPO Config Tests...")
        test_config = TestMAPPOConfig()
        test_validation = TestMAPPOConfigValidation()
        passed = 0
        failed = 0

        for name in dir(test_config):
            if name.startswith('test_'):
                try:
                    getattr(test_config, name)()
                    print(f"  PASS: {name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {name} - {e}")
                    failed += 1

        for name in dir(test_validation):
            if name.startswith('test_'):
                try:
                    getattr(test_validation, name)()
                    print(f"  PASS: {name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {name} - {e}")
                    failed += 1

        print(f"\nResults: {passed} passed, {failed} failed")
