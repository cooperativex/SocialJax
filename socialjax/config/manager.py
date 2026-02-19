"""SocialJax Configuration Management System.

This module provides a unified configuration system for managing training,
network, algorithm, and environment configurations. It supports:
- Dataclass-based configuration schemas with type hints
- YAML loading and merging using OmegaConf
- Config validation with required key checking
- Custom config overrides

Example usage:
    >>> from socialjax.config import ConfigManager
    >>> manager = ConfigManager()
    >>> config = manager.load(algorithm="ippo", environment="coin_game")
    >>> print(config.training.learning_rate)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    from omegaconf import OmegaConf, DictConfig, MISSING
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    DictConfig = dict
    MISSING = None


# Sentinel for required fields
_REQUIRED = "REQUIRED"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    total_timesteps: int = 10_000_000
    num_envs: int = 32
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    anneal_lr: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    architecture: str = "cnn_actor_critic"
    hidden_size: int = 64
    num_channels: Tuple[int, ...] = (16, 32, 32)
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_size: Tuple[int, int] = (3, 3)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if "num_channels" in filtered and isinstance(filtered["num_channels"], list):
            filtered["num_channels"] = tuple(filtered["num_channels"])
        if "kernel_size" in filtered and isinstance(filtered["kernel_size"], list):
            filtered["kernel_size"] = tuple(filtered["kernel_size"])
        return cls(**filtered)


@dataclass
class AlgorithmConfig:
    """Algorithm-specific configuration."""
    name: str = "ippo"
    parameter_sharing: bool = True
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    centralised_critic: bool = False
    target_update_freq: int = 200
    target_update_tau: float = 0.005

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameter_sharing": self.parameter_sharing,
            "centralised_critic": self.centralised_critic,
            "target_update_freq": self.target_update_freq,
            "target_update_tau": self.target_update_tau,
            "network": self.network.to_dict(),
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmConfig":
        data = data.copy()
        if "network" in data and isinstance(data["network"], dict):
            data["network"] = NetworkConfig.from_dict(data["network"])
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig.from_dict(data["training"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    name: str = "coin_game"
    num_agents: int = 2
    max_steps: int = 100
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SocialJaxConfig:
    """Complete SocialJax configuration combining all components."""
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.to_dict(),
            "environment": self.environment.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SocialJaxConfig":
        data = data.copy()
        if "algorithm" in data and isinstance(data["algorithm"], dict):
            data["algorithm"] = AlgorithmConfig.from_dict(data["algorithm"])
        if "environment" in data and isinstance(data["environment"], dict):
            data["environment"] = EnvironmentConfig.from_dict(data["environment"])
        return cls(**data)


class ConfigManager:
    """Configuration manager for loading, merging, and validating configs."""

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "presets"

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Optional[SocialJaxConfig] = None

    @property
    def config_path(self) -> Path:
        return self._config_path

    def load(
        self,
        algorithm: str,
        environment: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> SocialJaxConfig:
        base_config = self._load_yaml(self._config_path / "base.yaml")
        algo_config = self._load_yaml(self._config_path / "algorithms" / f"{algorithm}.yaml")
        env_config = self._load_yaml(self._config_path / "environments" / f"{environment}.yaml")

        if OMEGACONF_AVAILABLE:
            merged = OmegaConf.merge(base_config, algo_config, env_config)
            if custom_config:
                merged = OmegaConf.merge(merged, custom_config)
            merged_dict = OmegaConf.to_container(merged, resolve=True)
        else:
            merged_dict = self._merge_dicts(base_config, algo_config, env_config, custom_config or {})

        self._validate(merged_dict)
        self._config = SocialJaxConfig.from_dict(merged_dict)
        return self._config

    def load_from_file(self, path: Path) -> SocialJaxConfig:
        config_dict = self._load_yaml(path)
        self._validate(config_dict)
        self._config = SocialJaxConfig.from_dict(config_dict)
        return self._config

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except ImportError:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    def _merge_dicts(self, *dicts: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for d in dicts:
            if not d:
                continue
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_dicts(result[key], value)
                else:
                    result[key] = value
        return result

    def _validate(self, config: Dict[str, Any]) -> None:
        required_keys = ["algorithm", "environment"]
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ConfigValidationError(f"Missing required config keys: {missing}")

        if "algorithm" in config:
            algo_required = ["name"]
            algo_missing = [k for k in algo_required if k not in config.get("algorithm", {})]
            if algo_missing:
                raise ConfigValidationError(f"Missing required algorithm config keys: {algo_missing}")

        if "environment" in config:
            env_required = ["name"]
            env_missing = [k for k in env_required if k not in config.get("environment", {})]
            if env_missing:
                raise ConfigValidationError(f"Missing required environment config keys: {env_missing}")

    def get_config(self) -> Optional[SocialJaxConfig]:
        return self._config

    def save_config(self, path: Path, config: Optional[SocialJaxConfig] = None) -> None:
        config = config or self._config
        if config is None:
            raise ValueError("No configuration to save")
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")


def create_default_config(algorithm: str = "ippo", environment: str = "coin_game", **kwargs) -> SocialJaxConfig:
    config = SocialJaxConfig(
        algorithm=AlgorithmConfig(name=algorithm),
        environment=EnvironmentConfig(name=environment),
    )
    if kwargs:
        manager = ConfigManager()
        merged = manager._merge_dicts(config.to_dict(), kwargs)
        config = SocialJaxConfig.from_dict(merged)
    return config
