"""Unit tests for SVO algorithm configuration."""

import pytest
import sys
import math

sys.path.insert(0, 'socialjax')

from socialjax.algorithms.svo.config import (
    SVO_DEFAULT_CONFIG,
    get_svo_config,
    svo_angle_to_radians,
    get_svo_weights,
)


class TestSVOConfigImport:
    """Tests for SVO config imports."""

    def test_import_svo_default_config(self):
        """Test SVO_DEFAULT_CONFIG can be imported."""
        assert SVO_DEFAULT_CONFIG is not None
        assert isinstance(SVO_DEFAULT_CONFIG, dict)

    def test_import_get_svo_config(self):
        """Test get_svo_config function can be imported."""
        assert callable(get_svo_config)

    def test_import_svo_angle_to_radians(self):
        """Test svo_angle_to_radians function can be imported."""
        assert callable(svo_angle_to_radians)

    def test_import_get_svo_weights(self):
        """Test get_svo_weights function can be imported."""
        assert callable(get_svo_weights)


class TestSVOConfigDefaults:
    """Tests for default SVO configuration values."""

    def test_lr_default(self):
        """Test default learning rate."""
        assert SVO_DEFAULT_CONFIG["LR"] == 2.5e-4

    def test_gamma_default(self):
        """Test default discount factor."""
        assert SVO_DEFAULT_CONFIG["GAMMA"] == 0.99

    def test_gae_lambda_default(self):
        """Test default GAE lambda."""
        assert SVO_DEFAULT_CONFIG["GAE_LAMBDA"] == 0.95

    def test_clip_eps_default(self):
        """Test default clip epsilon."""
        assert SVO_DEFAULT_CONFIG["CLIP_EPS"] == 0.2

    def test_svo_angle_default(self):
        """Test default SVO angle is cooperative (45 degrees)."""
        assert SVO_DEFAULT_CONFIG["SVO_ANGLE"] == 45.0

    def test_use_fairness_reward_default(self):
        """Test default fairness reward flag."""
        assert SVO_DEFAULT_CONFIG["USE_FAIRNESS_REWARD"] is True

    def test_fairness_weight_default(self):
        """Test default fairness weight."""
        assert SVO_DEFAULT_CONFIG["FAIRNESS_WEIGHT"] == 0.1


class TestGetSVOConfig:
    """Tests for get_svo_config function."""

    def test_returns_dict(self):
        """Test that get_svo_config returns a dictionary."""
        config = get_svo_config()
        assert isinstance(config, dict)

    def test_returns_default_without_overrides(self):
        """Test that get_svo_config returns defaults when no overrides."""
        config = get_svo_config()
        assert config["LR"] == SVO_DEFAULT_CONFIG["LR"]
        assert config["SVO_ANGLE"] == SVO_DEFAULT_CONFIG["SVO_ANGLE"]

    def test_override_single_value(self):
        """Test overriding a single configuration value."""
        config = get_svo_config({"LR": 1e-3})
        assert config["LR"] == 1e-3
        assert config["GAMMA"] == SVO_DEFAULT_CONFIG["GAMMA"]

    def test_override_multiple_values(self):
        """Test overriding multiple configuration values."""
        config = get_svo_config({"LR": 1e-3, "GAMMA": 0.95, "SVO_ANGLE": 30.0})
        assert config["LR"] == 1e-3
        assert config["GAMMA"] == 0.95
        assert config["SVO_ANGLE"] == 30.0

    def test_override_svo_angle(self):
        """Test overriding SVO angle."""
        config = get_svo_config({"SVO_ANGLE": 60.0})
        assert config["SVO_ANGLE"] == 60.0

    def test_does_not_modify_original(self):
        """Test that get_svo_config doesn't modify the original config."""
        original_lr = SVO_DEFAULT_CONFIG["LR"]
        get_svo_config({"LR": 999.0})
        assert SVO_DEFAULT_CONFIG["LR"] == original_lr

    def test_add_new_key(self):
        """Test adding a new configuration key."""
        config = get_svo_config({"NEW_KEY": "new_value"})
        assert config["NEW_KEY"] == "new_value"

    def test_nested_override(self):
        """Test that nested overrides work correctly."""
        config = get_svo_config({
            "REW_SHAPING_HORIZON": 500000,
            "FAIRNESS_WEIGHT": 0.2
        })
        assert config["REW_SHAPING_HORIZON"] == 500000
        assert config["FAIRNESS_WEIGHT"] == 0.2


class TestSVOAngleToRadians:
    """Tests for svo_angle_to_radians function."""

    def test_zero_degrees(self):
        """Test conversion of 0 degrees."""
        result = svo_angle_to_radians(0)
        assert result == 0.0

    def test_45_degrees(self):
        """Test conversion of 45 degrees."""
        result = svo_angle_to_radians(45)
        assert abs(result - math.pi / 4) < 1e-10

    def test_90_degrees(self):
        """Test conversion of 90 degrees."""
        result = svo_angle_to_radians(90)
        assert abs(result - math.pi / 2) < 1e-10

    def test_180_degrees(self):
        """Test conversion of 180 degrees."""
        result = svo_angle_to_radians(180)
        assert abs(result - math.pi) < 1e-10

    def test_negative_angle(self):
        """Test conversion of negative angle."""
        result = svo_angle_to_radians(-45)
        assert abs(result - (-math.pi / 4)) < 1e-10


class TestGetSVOWeights:
    """Tests for get_svo_weights function."""

    def test_zero_degrees_selfish(self):
        """Test that 0 degrees gives purely selfish weights."""
        w_self, w_other = get_svo_weights(0)
        assert abs(w_self - 1.0) < 1e-10
        assert abs(w_other - 0.0) < 1e-10

    def test_90_degrees_altruistic(self):
        """Test that 90 degrees gives purely altruistic weights."""
        w_self, w_other = get_svo_weights(90)
        assert abs(w_self - 0.0) < 1e-10
        assert abs(w_other - 1.0) < 1e-10

    def test_45_degrees_cooperative(self):
        """Test that 45 degrees gives equal weights."""
        w_self, w_other = get_svo_weights(45)
        # cos(45) = sin(45) = sqrt(2)/2
        expected = math.sqrt(2) / 2
        assert abs(w_self - expected) < 1e-10
        assert abs(w_other - expected) < 1e-10

    def test_weights_sum_to_one_for_45(self):
        """Test that weights don't necessarily sum to 1."""
        w_self, w_other = get_svo_weights(45)
        # For 45 degrees, sum is sqrt(2) ~= 1.414
        assert abs(w_self + w_other - math.sqrt(2)) < 1e-10

    def test_weights_are_unit_vectors(self):
        """Test that (w_self, w_other) forms a unit vector."""
        for angle in [0, 15, 30, 45, 60, 75, 90]:
            w_self, w_other = get_svo_weights(angle)
            magnitude = math.sqrt(w_self**2 + w_other**2)
            assert abs(magnitude - 1.0) < 1e-10, f"Failed for angle {angle}"


class TestSVOConfigValidation:
    """Tests for SVO configuration validation."""

    def test_svo_angle_range_valid(self):
        """Test that typical SVO angles are valid."""
        for angle in [0, 15, 30, 45, 60, 75, 90]:
            config = get_svo_config({"SVO_ANGLE": angle})
            assert config["SVO_ANGLE"] == angle

    def test_fairness_weight_range(self):
        """Test fairness weight in valid range."""
        for weight in [0.0, 0.1, 0.5, 1.0]:
            config = get_svo_config({"FAIRNESS_WEIGHT": weight})
            assert config["FAIRNESS_WEIGHT"] == weight

    def test_boolean_flags(self):
        """Test boolean configuration flags."""
        config_true = get_svo_config({"USE_FAIRNESS_REWARD": True})
        assert config_true["USE_FAIRNESS_REWARD"] is True

        config_false = get_svo_config({"USE_FAIRNESS_REWARD": False})
        assert config_false["USE_FAIRNESS_REWARD"] is False

    def test_training_schedule_params(self):
        """Test training schedule parameters."""
        config = get_svo_config({
            "UPDATE_EPOCHS": 8,
            "NUM_MINIBATCHES": 8,
            "NUM_STEPS": 256
        })
        assert config["UPDATE_EPOCHS"] == 8
        assert config["NUM_MINIBATCHES"] == 8
        assert config["NUM_STEPS"] == 256


class TestSVOConfigKeys:
    """Tests for all required config keys."""

    def test_all_ppo_keys_present(self):
        """Test that all PPO-related keys are present."""
        required_keys = [
            "LR", "ANNEAL_LR", "GAMMA", "GAE_LAMBDA",
            "CLIP_EPS", "VF_COEF", "ENT_COEF", "MAX_GRAD_NORM",
            "UPDATE_EPOCHS", "NUM_MINIBATCHES", "NUM_STEPS",
            "ACTIVATION", "HIDDEN_SIZE", "PARAMETER_SHARING"
        ]
        for key in required_keys:
            assert key in SVO_DEFAULT_CONFIG, f"Missing key: {key}"

    def test_all_svo_keys_present(self):
        """Test that all SVO-specific keys are present."""
        svo_keys = [
            "SVO_ANGLE",
            "USE_FAIRNESS_REWARD",
            "FAIRNESS_WEIGHT",
            "REW_SHAPING_HORIZON"
        ]
        for key in svo_keys:
            assert key in SVO_DEFAULT_CONFIG, f"Missing SVO key: {key}"

    def test_config_immutability(self):
        """Test that modifications to returned config don't affect defaults."""
        config1 = get_svo_config()
        config1["LR"] = 999.0

        config2 = get_svo_config()
        assert config2["LR"] == 2.5e-4, "Config was mutated"
