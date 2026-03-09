"""SocialJax Configuration Module.

This module provides a unified configuration system for SocialJax.

Classes:
    TrainingConfig: Training hyperparameters
    NetworkConfig: Neural network architecture settings
    AlgorithmConfig: Algorithm-specific configuration
    EnvironmentConfig: Environment settings
    SocialJaxConfig: Complete configuration combining all components
    ConfigManager: Configuration loader and validator

Functions:
    create_default_config: Create default configuration with overrides

Example:
    >>> from socialjax.config import ConfigManager, create_default_config
    >>> 
    >>> # Using ConfigManager with YAML files
    >>> manager = ConfigManager()
    >>> config = manager.load("ippo", "coin_game")
    >>> 
    >>> # Or create defaults directly
    >>> config = create_default_config(algorithm="ippo", environment="coin_game")
"""

from socialjax.config.manager import (
    TrainingConfig,
    NetworkConfig,
    AlgorithmConfig,
    EnvironmentConfig,
    SocialJaxConfig,
    ConfigManager,
    ConfigValidationError,
    create_default_config,
)

__all__ = [
    # Configuration dataclasses
    "TrainingConfig",
    "NetworkConfig",
    "AlgorithmConfig",
    "EnvironmentConfig",
    "SocialJaxConfig",
    # Manager and utilities
    "ConfigManager",
    "ConfigValidationError",
    "create_default_config",
]
