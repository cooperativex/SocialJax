"""SocialJax: A Modular Multi-Agent Reinforcement Learning Framework.

SocialJax provides a clean, modular interface for training and evaluating
multi-agent reinforcement learning algorithms on sequential social dilemmas.
Built on JAX/Flax for high-performance GPU acceleration.

Quick Start:
    >>> import socialjax
    >>>
    >>> # Create an environment
    >>> env = socialjax.make('clean_up', num_agents=7)
    >>>
    >>> # Create an algorithm
    >>> algorithm = socialjax.get_algorithm('ippo')(
    ...     observation_space=env.observation_space(),
    ...     action_space=env.action_space(),
    ...     config=socialjax.create_default_config('ippo')
    ... )
    >>>
    >>> # Create a trainer
    >>> trainer = socialjax.Trainer(
    ...     algorithm=algorithm,
    ...     env=env,
    ...     callbacks=[
    ...         socialjax.CheckpointCallback(save_freq=1000, save_path='./checkpoints'),
    ...         socialjax.ProgressCallback(total_timesteps=1_000_000),
    ...     ]
    ... )
    >>>
    >>> # Train the model
    >>> metrics = trainer.train(total_timesteps=1_000_000)
    >>>
    >>> # Evaluate the trained model
    >>> eval_metrics = trainer.evaluate(n_episodes=50)

Available Environments:
    - coin_game: Multi-agent coin collection game
    - clean_up: Public goods game with pollution
    - harvest_common_open: Commons harvesting with open access
    - coop_mining: Cooperative mining scenario
    - territory_open: Territory control game
    - pd_arena: Prisoner's dilemma arena
    - mushrooms: Mushroom foraging game
    - gift: Gift-giving coordination game

Available Algorithms:
    - ippo: Independent PPO (decentralized training and execution)
    - mappo: Multi-Agent PPO (centralized training, decentralized execution)
    - vdn: Value Decomposition Network (off-policy, centralized Q-learning)
    - svo: Social Value Orientation (prosocial reward shaping)
"""

__version__ = "2.0.0"

# =============================================================================
# Environment Creation
# =============================================================================
from socialjax import registration

make = registration.make
registered_envs = registration.REGISTERED_ENVS

# =============================================================================
# Core Components
# =============================================================================
from socialjax.core import (
    # Base classes
    BaseAlgorithm,
    BaseTrainer,
    # State classes
    AlgorithmState,
    TrainerState,
    TrainingMetrics,
    # Utilities
    jit_method,
    Callback,
)

# =============================================================================
# Algorithm Registry and Implementations
# =============================================================================
from socialjax.algorithms import (
    # Registry functions
    register_algorithm,
    get_algorithm,
    list_algorithms,
    unregister_algorithm,
    is_algorithm_registered,
    clear_registry as clear_algorithm_registry,
    # Exceptions
    AlgorithmAlreadyRegisteredError,
    AlgorithmNotFoundError,
)

# =============================================================================
# Network Registry and Factory
# =============================================================================
from socialjax.networks import (
    # Registry functions
    register_network,
    get_network_class,
    list_networks,
    unregister_network,
    is_network_registered,
    clear_registry as clear_network_registry,
    # Factory
    create_network,
    get_config_preset,
    list_config_presets,
    NETWORK_CONFIGS,
    # CNN Networks
    CNNSmall,
    CNNActorCritic,
    CNNSmallEncoder,
    CNNImpala,
    # Exceptions
    NetworkAlreadyRegisteredError,
    NetworkNotFoundError,
)

# =============================================================================
# Experience Buffers
# =============================================================================
from socialjax.buffers import (
    # Base
    BaseBuffer,
    # Buffer implementations
    RolloutBuffer,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    # Exceptions
    BufferError,
    BufferEmptyError,
    BufferFullError,
    InsufficientDataError,
)

# =============================================================================
# Training Utilities
# =============================================================================
from socialjax.training import (
    # Trainer
    Trainer,
    create_trainer,
    RolloutBuffer as TrainingRolloutBuffer,
    # Callbacks
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    ProgressCallback,
    WandbCallback,
)

# =============================================================================
# Evaluation System
# =============================================================================
from socialjax.evaluation import (
    # Metrics
    EpisodeMetrics,
    EvaluationMetrics,
    compute_episode_return,
    compute_agent_returns,
    compute_cooperation_rate,
    compute_gini_coefficient,
    compute_social_welfare,
    aggregate_episode_metrics,
    compute_metrics_from_episodes,
    # Evaluator
    EvaluatorConfig,
    Evaluator,
    save_evaluation_results,
    load_evaluation_results,
    print_evaluation_summary,
    # Visualization
    OutputFormat,
    VisualizationMode,
    save_gif,
    save_mp4,
    save_visualization,
    create_comparison_gif,
    create_episode_grid,
)

# =============================================================================
# Configuration System
# =============================================================================
from socialjax.config import (
    # Configuration dataclasses
    TrainingConfig,
    NetworkConfig,
    AlgorithmConfig,
    EnvironmentConfig,
    SocialJaxConfig,
    # Manager
    ConfigManager,
    ConfigValidationError,
    create_default_config,
)

# =============================================================================
# Public API Exports
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Environment Creation
    "make",
    "registered_envs",
    # Core Components
    "BaseAlgorithm",
    "BaseTrainer",
    "AlgorithmState",
    "TrainerState",
    "TrainingMetrics",
    "jit_method",
    "Callback",
    # Algorithm Registry
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
    "unregister_algorithm",
    "is_algorithm_registered",
    "clear_algorithm_registry",
    "AlgorithmAlreadyRegisteredError",
    "AlgorithmNotFoundError",
    # Network Registry
    "register_network",
    "get_network_class",
    "list_networks",
    "unregister_network",
    "is_network_registered",
    "clear_network_registry",
    "create_network",
    "get_config_preset",
    "list_config_presets",
    "NETWORK_CONFIGS",
    "CNNSmall",
    "CNNActorCritic",
    "CNNSmallEncoder",
    "CNNImpala",
    "NetworkAlreadyRegisteredError",
    "NetworkNotFoundError",
    # Buffers
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "BufferError",
    "BufferEmptyError",
    "BufferFullError",
    "InsufficientDataError",
    # Training
    "Trainer",
    "create_trainer",
    "TrainingRolloutBuffer",
    "BaseCallback",
    "CallbackList",
    "CheckpointCallback",
    "EvalCallback",
    "ProgressCallback",
    "WandbCallback",
    # Evaluation
    "EpisodeMetrics",
    "EvaluationMetrics",
    "compute_episode_return",
    "compute_agent_returns",
    "compute_cooperation_rate",
    "compute_gini_coefficient",
    "compute_social_welfare",
    "aggregate_episode_metrics",
    "compute_metrics_from_episodes",
    "EvaluatorConfig",
    "Evaluator",
    "save_evaluation_results",
    "load_evaluation_results",
    "print_evaluation_summary",
    "OutputFormat",
    "VisualizationMode",
    "save_gif",
    "save_mp4",
    "save_visualization",
    "create_comparison_gif",
    "create_episode_grid",
    # Configuration
    "TrainingConfig",
    "NetworkConfig",
    "AlgorithmConfig",
    "EnvironmentConfig",
    "SocialJaxConfig",
    "ConfigManager",
    "ConfigValidationError",
    "create_default_config",
]
