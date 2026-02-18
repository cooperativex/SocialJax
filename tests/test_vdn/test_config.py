"""Unit tests for VDN configuration.

Tests cover:
- VDN_DEFAULT_CONFIG structure and values
- get_vdn_config function with overrides
- Configuration validation
"""

import pytest
import sys

sys.path.insert(0, "socialjax")


class TestVDNDefaultConfig:
    """Tests for VDN_DEFAULT_CONFIG."""

    def test_config_exists(self):
        """Test that VDN_DEFAULT_CONFIG can be imported."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        assert VDN_DEFAULT_CONFIG is not None

    def test_config_is_dict(self):
        """Test that VDN_DEFAULT_CONFIG is a dictionary."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        assert isinstance(VDN_DEFAULT_CONFIG, dict)

    def test_required_keys_exist(self):
        """Test that all required configuration keys exist."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        required_keys = [
            "LR",
            "GAMMA",
            "MAX_GRAD_NORM",
            "EPS_START",
            "EPS_FINISH",
            "EPS_DECAY",
            "TARGET_UPDATE_INTERVAL",
            "TAU",
            "BUFFER_SIZE",
            "BUFFER_BATCH_SIZE",
            "LEARNING_STARTS",
            "NUM_EPOCHS",
            "NUM_STEPS",
            "NUM_ENVS",
            "ACTIVATION",
            "HIDDEN_SIZE",
            "PARAMETER_SHARING",
        ]

        for key in required_keys:
            assert key in VDN_DEFAULT_CONFIG, f"Missing required key: {key}"

    def test_learning_rate_value(self):
        """Test default learning rate is reasonable."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        lr = VDN_DEFAULT_CONFIG["LR"]
        assert 0 < lr < 1, f"Learning rate {lr} should be between 0 and 1"

    def test_gamma_value(self):
        """Test discount factor is valid."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        gamma = VDN_DEFAULT_CONFIG["GAMMA"]
        assert 0 <= gamma <= 1, f"Gamma {gamma} should be between 0 and 1"

    def test_epsilon_values(self):
        """Test epsilon exploration values are valid."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        eps_start = VDN_DEFAULT_CONFIG["EPS_START"]
        eps_finish = VDN_DEFAULT_CONFIG["EPS_FINISH"]
        eps_decay = VDN_DEFAULT_CONFIG["EPS_DECAY"]

        assert 0 <= eps_finish <= eps_start <= 1, "Invalid epsilon values"
        assert 0 <= eps_decay <= 1, f"EPS_DECAY {eps_decay} should be between 0 and 1"

    def test_target_update_interval_positive(self):
        """Test target update interval is positive."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        interval = VDN_DEFAULT_CONFIG["TARGET_UPDATE_INTERVAL"]
        assert interval > 0, f"TARGET_UPDATE_INTERVAL {interval} should be positive"

    def test_tau_value(self):
        """Test soft update coefficient is valid."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        tau = VDN_DEFAULT_CONFIG["TAU"]
        assert 0 < tau <= 1, f"TAU {tau} should be between 0 and 1"

    def test_buffer_size_positive(self):
        """Test buffer sizes are positive."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        buffer_size = VDN_DEFAULT_CONFIG["BUFFER_SIZE"]
        batch_size = VDN_DEFAULT_CONFIG["BUFFER_BATCH_SIZE"]

        assert buffer_size > 0, f"BUFFER_SIZE {buffer_size} should be positive"
        assert batch_size > 0, f"BUFFER_BATCH_SIZE {batch_size} should be positive"

    def test_hidden_size_positive(self):
        """Test hidden layer size is positive."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        hidden_size = VDN_DEFAULT_CONFIG["HIDDEN_SIZE"]
        assert hidden_size > 0, f"HIDDEN_SIZE {hidden_size} should be positive"

    def test_activation_valid(self):
        """Test activation function is valid."""
        from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG

        activation = VDN_DEFAULT_CONFIG["ACTIVATION"]
        assert activation in ["relu", "tanh"], f"Invalid activation: {activation}"


class TestGetVDNConfig:
    """Tests for get_vdn_config function."""

    def test_function_exists(self):
        """Test that get_vdn_config function exists."""
        from socialjax.algorithms.vdn.config import get_vdn_config

        assert callable(get_vdn_config)

    def test_returns_dict(self):
        """Test that get_vdn_config returns a dictionary."""
        from socialjax.algorithms.vdn.config import get_vdn_config

        config = get_vdn_config()
        assert isinstance(config, dict)

    def test_returns_default_when_no_overrides(self):
        """Test that get_vdn_config returns default config when no overrides."""
        from socialjax.algorithms.vdn.config import get_vdn_config, VDN_DEFAULT_CONFIG

        config = get_vdn_config()
        assert config == VDN_DEFAULT_CONFIG

    def test_overrides_single_value(self):
        """Test that get_vdn_config correctly overrides single values."""
        from socialjax.algorithms.vdn.config import get_vdn_config

        config = get_vdn_config({"LR": 0.001})
        assert config["LR"] == 0.001

    def test_overrides_multiple_values(self):
        """Test that get_vdn_config correctly overrides multiple values."""
        from socialjax.algorithms.vdn.config import get_vdn_config

        overrides = {"LR": 0.001, "GAMMA": 0.95, "HIDDEN_SIZE": 128}
        config = get_vdn_config(overrides)

        assert config["LR"] == 0.001
        assert config["GAMMA"] == 0.95
        assert config["HIDDEN_SIZE"] == 128

    def test_preserves_non_overridden_values(self):
        """Test that non-overridden values are preserved."""
        from socialjax.algorithms.vdn.config import get_vdn_config, VDN_DEFAULT_CONFIG

        config = get_vdn_config({"LR": 0.001})

        # Check that other values are unchanged
        assert config["GAMMA"] == VDN_DEFAULT_CONFIG["GAMMA"]
        assert config["EPS_START"] == VDN_DEFAULT_CONFIG["EPS_START"]

    def test_none_overrides_returns_default(self):
        """Test that None overrides returns default config."""
        from socialjax.algorithms.vdn.config import get_vdn_config, VDN_DEFAULT_CONFIG

        config = get_vdn_config(None)
        assert config == VDN_DEFAULT_CONFIG

    def test_empty_overrides_returns_default(self):
        """Test that empty dict overrides returns default config."""
        from socialjax.algorithms.vdn.config import get_vdn_config, VDN_DEFAULT_CONFIG

        config = get_vdn_config({})
        assert config == VDN_DEFAULT_CONFIG

    def test_does_not_modify_default(self):
        """Test that get_vdn_config doesn't modify the default config."""
        from socialjax.algorithms.vdn.config import get_vdn_config, VDN_DEFAULT_CONFIG

        original_lr = VDN_DEFAULT_CONFIG["LR"]
        get_vdn_config({"LR": 999})

        assert VDN_DEFAULT_CONFIG["LR"] == original_lr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
