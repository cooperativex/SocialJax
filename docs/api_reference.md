# SocialJax API Reference

SocialJax is a modular multi-agent reinforcement learning framework built on JAX/Flax for high-performance GPU acceleration. This document provides a complete reference for the V2 API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Creation](#environment-creation)
3. [Algorithms](#algorithms)
4. [Networks](#networks)
5. [Training](#training)
6. [Callbacks](#callbacks)
7. [Evaluation](#evaluation)
8. [Configuration](#configuration)
9. [Buffers](#buffers)

---

## Quick Start

```python
import socialjax

# Create an environment
env = socialjax.make('clean_up', num_agents=7)

# Get an algorithm class
AlgorithmClass = socialjax.get_algorithm('ippo')

# Create algorithm instance
algorithm = AlgorithmClass(
    observation_space=env.observation_space(),
    action_space=env.action_space(),
    config=socialjax.create_default_config('ippo')
)

# Create a trainer with callbacks
trainer = socialjax.Trainer(
    algorithm=algorithm,
    env=env,
    callbacks=[
        socialjax.CheckpointCallback(save_freq=1000, save_path='./checkpoints'),
        socialjax.ProgressCallback(total_timesteps=1_000_000),
    ]
)

# Train the model
state, metrics = trainer.train(total_timesteps=1_000_000)

# Evaluate the trained model
eval_metrics = trainer.evaluate(state, num_episodes=50)
print(f"Mean return: {eval_metrics['mean_return']:.2f}")
```

---

## Environment Creation

### `socialjax.make(name, **kwargs)`

Create a multi-agent environment instance.

**Parameters:**
- `name` (str): Environment name. Available environments:
  - `coin_game`: Multi-agent coin collection game
  - `clean_up`: Public goods game with pollution
  - `harvest_common_open`: Commons harvesting with open access
  - `coop_mining`: Cooperative mining scenario
  - `territory_open`: Territory control game
  - `pd_arena`: Prisoner's dilemma arena
  - `mushrooms`: Mushroom foraging game
  - `gift`: Gift-giving coordination game
- `**kwargs`: Environment-specific parameters (e.g., `num_agents`)

**Returns:**
- Environment instance

**Example:**
```python
import socialjax

# Create clean_up environment with 7 agents
env = socialjax.make('clean_up', num_agents=7)

# Reset and get initial observation
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# Step environment
actions = {agent: env.action_space().sample(key) for agent in env.agents}
state, obs, rewards, dones, info = env.step(state, actions)
```

### `socialjax.registered_envs`

List of all registered environment names.

```python
print(socialjax.registered_envs)
# ['coin_game', 'harvest_common_open', 'clean_up', 'coop_mining', ...]
```

---

## Algorithms

### Algorithm Registry

#### `socialjax.get_algorithm(name)`

Get an algorithm class by name.

**Parameters:**
- `name` (str): Algorithm name (`ippo`, `mappo`, `vdn`, `svo`)

**Returns:**
- Algorithm class (subclass of `BaseAlgorithm`)

**Raises:**
- `AlgorithmNotFoundError`: If algorithm is not registered

**Example:**
```python
# Get IPPO algorithm class
IPPO = socialjax.get_algorithm('ippo')

# Create instance
algorithm = IPPO(observation_space, action_space, config)
```

#### `socialjax.list_algorithms()`

List all registered algorithm names.

**Returns:**
- `List[str]`: Sorted list of algorithm names

```python
print(socialjax.list_algorithms())
# ['ippo', 'mappo', 'svo', 'vdn']
```

#### `socialjax.register_algorithm(name)`

Decorator to register a custom algorithm.

**Parameters:**
- `name` (str): Name to register the algorithm under

**Example:**
```python
@socialjax.register_algorithm('my_custom')
class MyCustomAlgorithm(socialjax.BaseAlgorithm):
    # ... implementation
    pass
```

### Available Algorithms

#### IPPO (Independent PPO)

Decentralized training and execution with independent policy optimization.

```python
algorithm = socialjax.get_algorithm('ippo')(
    observation_space=env.observation_space(),
    action_space=env.action_space(),
    config={
        'LR': 2.5e-4,
        'GAMMA': 0.99,
        'GAE_LAMBDA': 0.95,
        'CLIP_EPS': 0.2,
        'ENT_COEF': 0.01,
        'VF_COEF': 0.5,
        'MAX_GRAD_NORM': 0.5,
    }
)
```

#### MAPPO (Multi-Agent PPO)

Centralized training with decentralized execution using a shared critic.

```python
algorithm = socialjax.get_algorithm('mappo')(
    observation_space=env.observation_space(),
    action_space=env.action_space(),
    config={
        'LR_ACTOR': 2.5e-4,
        'LR_CRITIC': 2.5e-4,
        'GAMMA': 0.99,
        'GAE_LAMBDA': 0.95,
        'CLIP_EPS': 0.2,
        'USE_CENTRALIZED_VALUE': True,
    }
)
```

#### VDN (Value Decomposition Network)

Off-policy algorithm with individual Q-value decomposition.

```python
algorithm = socialjax.get_algorithm('vdn')(
    observation_space=env.observation_space(),
    action_space=env.action_space(),
    config={
        'LR': 1e-4,
        'GAMMA': 0.99,
        'BUFFER_SIZE': 10000,
        'BUFFER_BATCH_SIZE': 32,
        'EPS_START': 1.0,
        'EPS_FINISH': 0.05,
        'EPS_DECAY': 100000,
        'TARGET_UPDATE_INTERVAL': 200,
        'TAU': 0.005,
    }
)
```

#### SVO (Social Value Orientation)

PPO with social value orientation reward transformation.

```python
algorithm = socialjax.get_algorithm('svo')(
    observation_space=env.observation_space(),
    action_space=env.action_space(),
    config={
        'LR': 2.5e-4,
        'SVO_ANGLE': 45.0,  # 0=selfish, 45=cooperative, 90=altruistic
        'USE_FAIRNESS_REWARD': True,
        'FAIRNESS_WEIGHT': 0.1,
    }
)
```

### BaseAlgorithm

Abstract base class for all algorithms.

```python
class BaseAlgorithm(ABC):
    def __init__(self, observation_space, action_space, config=None):
        """Initialize the algorithm."""

    @abstractmethod
    def _build_network(self):
        """Build and return the neural network."""

    @abstractmethod
    def _build_optimizer(self):
        """Build and return the optimizer."""

    @abstractmethod
    def init_state(self, rng):
        """Initialize and return the algorithm state."""

    @abstractmethod
    def compute_action(self, state, observation, rng, deterministic=False):
        """Compute action(s) given observation(s)."""

    @abstractmethod
    def update(self, state, batch):
        """Update algorithm parameters using a batch of experience."""

    def save(self, path):
        """Save algorithm state to disk."""

    def load(self, path):
        """Load algorithm state from disk."""
```

---

## Networks

### Network Factory

#### `socialjax.create_network(name, action_dim, **kwargs)`

Create a network instance by name.

**Parameters:**
- `name` (str): Network name (e.g., `cnn_small`, `cnn_impala`)
- `action_dim` (int): Number of output actions
- `**kwargs`: Additional network configuration

**Returns:**
- Network instance

**Example:**
```python
# Create a small CNN actor-critic
network = socialjax.create_network('cnn_small', action_dim=8)

# Create with custom configuration
network = socialjax.create_network(
    'cnn_small',
    action_dim=8,
    channel_sizes=(32, 64, 64),
    kernel_sizes=(5, 3, 3),
    hidden_size=128,
    activation='tanh'
)
```

#### `socialjax.list_networks()`

List all registered network names.

```python
print(socialjax.list_networks())
# ['cnn_impala', 'cnn_small', 'cnn_small_encoder', ...]
```

### Available Networks

#### CNNSmall

Lightweight CNN for simple environments.

```python
network = socialjax.create_network(
    'cnn_small',
    action_dim=8,
    channel_sizes=(32, 64, 64),
    hidden_size=128,
)
```

#### CNNImpala

IMPALA-style CNN with residual blocks for complex environments.

```python
network = socialjax.create_network(
    'cnn_impala',
    action_dim=8,
)
```

### Network Config Presets

```python
# Available presets: 'small', 'medium', 'large'
config = socialjax.get_config_preset('medium')
network = socialjax.create_network('cnn_small', action_dim=8, **config)
```

---

## Training

### Trainer

The main class for training multi-agent algorithms.

```python
class Trainer:
    def __init__(
        self,
        algorithm: Union[str, BaseAlgorithm],
        env: Union[str, Any],
        config: Optional[Dict] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ):
        """Initialize the trainer.

        Args:
            algorithm: Algorithm instance or name ('ippo', 'mappo', etc.)
            env: Environment instance or name ('clean_up', etc.)
            config: Training configuration dictionary
            callbacks: List of callbacks for training hooks
        """

    def train(self, total_timesteps: int, rng=None):
        """Run the training loop.

        Returns:
            Tuple[TrainerState, Dict]: Final state and training metrics
        """

    def evaluate(self, state, num_episodes=10, deterministic=True):
        """Evaluate the current policy.

        Returns:
            Dict: Evaluation metrics (mean_return, std_return, etc.)
        """

    def save(self, path: str):
        """Save trainer state to disk."""

    def load(self, path: str):
        """Load trainer state from disk."""
```

#### Example Usage

```python
# Create trainer from names
trainer = socialjax.Trainer(
    algorithm='ippo',
    env='clean_up',
    config={'total_timesteps': 1_000_000, 'num_envs': 8},
    callbacks=[
        socialjax.CheckpointCallback(save_freq=1000, save_path='./checkpoints'),
        socialjax.EvalCallback(eval_env=env, eval_freq=5000),
    ]
)

# Train
state, metrics = trainer.train(total_timesteps=1_000_000)

# Evaluate
eval_results = trainer.evaluate(state, num_episodes=50)

# Save checkpoint
trainer.save('./final_model')
```

### create_trainer()

Convenience function for creating trainers.

```python
trainer = socialjax.create_trainer(
    algorithm='ippo',
    env='clean_up',
    num_agents=7,
    callbacks=[...],
)
```

---

## Callbacks

### BaseCallback

Abstract base class for custom callbacks.

```python
class BaseCallback:
    def on_training_start(self, trainer):
        """Called at the start of training."""

    def on_training_end(self, trainer):
        """Called at the end of training."""

    def on_step(self, trainer, step, metrics):
        """Called after each training step."""

    def on_rollout_start(self, trainer):
        """Called at the start of a rollout."""

    def on_rollout_end(self, trainer, rollout_data):
        """Called at the end of a rollout."""

    def on_update_start(self, trainer):
        """Called before parameter update."""

    def on_update_end(self, trainer, update_metrics):
        """Called after parameter update."""
```

### Built-in Callbacks

#### CheckpointCallback

Saves model checkpoints during training.

```python
callback = socialjax.CheckpointCallback(
    save_freq=1000,           # Save every 1000 updates
    save_path='./checkpoints', # Directory to save checkpoints
    name_prefix='ippo',       # Checkpoint filename prefix
    verbose=True,
)
```

#### EvalCallback

Evaluates model periodically during training.

```python
callback = socialjax.EvalCallback(
    eval_env=env,             # Environment for evaluation
    eval_freq=5000,           # Evaluate every 5000 timesteps
    n_eval_episodes=10,       # Number of episodes per evaluation
    best_model_save_path='./best_model',
    deterministic=True,
)
```

#### ProgressCallback

Displays training progress with tqdm.

```python
callback = socialjax.ProgressCallback(
    total_timesteps=1_000_000,
    progress_freq=10,
    show_metrics=['loss', 'episode_return'],
    verbose=True,
)
```

#### WandbCallback

Logs metrics to Weights & Biases.

```python
callback = socialjax.WandbCallback(
    project='socialjax-experiments',
    name='ippo_clean_up',
    config={'lr': 2.5e-4},
    log_freq=100,
)
```

### CallbackList

Manage multiple callbacks.

```python
callbacks = socialjax.CallbackList([
    socialjax.CheckpointCallback(save_freq=1000, save_path='./checkpoints'),
    socialjax.ProgressCallback(total_timesteps=1_000_000),
    socialjax.WandbCallback(project='my-project'),
])

# Add callbacks dynamically
callbacks.add(my_custom_callback)
```

---

## Evaluation

### Evaluator

Run evaluation episodes and compute metrics.

```python
from socialjax.evaluation import Evaluator, EvaluatorConfig

evaluator = socialjax.Evaluator(
    env=env,
    algorithm=algorithm,
    config=EvaluatorConfig(
        num_episodes=50,
        deterministic=True,
        seed=42,
    )
)

# Run evaluation
metrics = evaluator.evaluate()

# Run with frame capture
metrics, frames = evaluator.evaluate_with_frames()

print(f"Mean return: {metrics.mean_return:.2f} +/- {metrics.std_return:.2f}")
print(f"Cooperation rate: {metrics.cooperation_rate:.2%}")
```

### Evaluation Metrics

```python
from socialjax.evaluation import (
    EpisodeMetrics,           # Single episode metrics
    EvaluationMetrics,        # Aggregated metrics
    compute_episode_return,   # Compute total episode return
    compute_agent_returns,    # Per-agent returns
    compute_cooperation_rate, # Cooperation metric
    compute_gini_coefficient, # Inequality measure
    compute_social_welfare,   # Total welfare
)
```

### Visualization

```python
from socialjax.evaluation import (
    save_gif,              # Save frames as GIF
    save_mp4,              # Save frames as MP4
    create_comparison_gif, # Side-by-side comparison
    create_episode_grid,   # Grid of frames
)

# Save evaluation as GIF
metrics, frames = evaluator.evaluate_with_frames()
socialjax.save_gif(frames, 'evaluation.gif', fps=10)

# Save as MP4
socialjax.save_mp4(frames, 'evaluation.mp4', fps=15)
```

---

## Configuration

### ConfigManager

Load and manage configurations.

```python
from socialjax.config import ConfigManager, create_default_config

# Using ConfigManager
manager = socialjax.ConfigManager()
config = manager.load('ippo', 'clean_up')

# Or create defaults directly
config = socialjax.create_default_config(
    algorithm='ippo',
    environment='clean_up',
)
```

### Configuration Classes

```python
from socialjax.config import (
    TrainingConfig,      # Training hyperparameters
    NetworkConfig,       # Network architecture settings
    AlgorithmConfig,     # Algorithm-specific configuration
    EnvironmentConfig,   # Environment settings
    SocialJaxConfig,     # Complete configuration
)

# Create training config
training_config = socialjax.TrainingConfig(
    total_timesteps=1_000_000,
    num_envs=8,
    num_steps=128,
    gamma=0.99,
    learning_rate=2.5e-4,
)
```

### YAML Configuration Files

Load configuration from YAML files:

```yaml
# socialjax/config/presets/algorithms/ippo.yaml
algorithm: ippo
training:
  total_timesteps: 10000000
  num_envs: 8
  num_steps: 128
  gamma: 0.99
  learning_rate: 0.00025
network:
  type: cnn_small
  hidden_size: 128
```

```python
manager = socialjax.ConfigManager()
config = manager.load_from_file('path/to/config.yaml')
```

---

## Buffers

### RolloutBuffer

On-policy buffer for storing trajectories.

```python
from socialjax.buffers import RolloutBuffer

buffer = socialjax.RolloutBuffer(
    buffer_size=128,        # Steps per environment
    num_envs=8,             # Number of parallel environments
    obs_shape=(15, 15, 3),  # Observation shape
    action_dim=8,           # Number of actions
)

# Add transitions
buffer.add(obs, action, reward, done, log_prob, value)

# Get all data
batch = buffer.get()

# Clear for next rollout
buffer.clear()
```

### ReplayBuffer

Off-policy buffer with random sampling.

```python
from socialjax.buffers import ReplayBuffer

buffer = socialjax.ReplayBuffer(
    buffer_size=10000,
    obs_shape=(4,),
    action_dim=2,
)

# Add transitions
buffer.add(obs, action, reward, next_obs, done)

# Sample random batch
batch = buffer.sample(32)
```

### PrioritizedReplayBuffer

Replay buffer with prioritized experience replay.

```python
from socialjax.buffers import PrioritizedReplayBuffer

buffer = socialjax.PrioritizedReplayBuffer(
    buffer_size=10000,
    obs_shape=(4,),
    action_dim=2,
    alpha=0.6,  # Priority exponent
    beta=0.4,   # Importance sampling exponent
)

# Sample with priorities
batch = buffer.sample(32)

# Update priorities after learning
buffer.update_priorities(indices, td_errors)
```

---

## Error Handling

### Algorithm Errors

```python
from socialjax import AlgorithmNotFoundError, AlgorithmAlreadyRegisteredError

try:
    algorithm = socialjax.get_algorithm('unknown')
except AlgorithmNotFoundError as e:
    print(f"Available algorithms: {e.available_algorithms}")
```

### Network Errors

```python
from socialjax import NetworkNotFoundError, NetworkAlreadyRegisteredError

try:
    network = socialjax.create_network('unknown', action_dim=8)
except NetworkNotFoundError as e:
    print(f"Available networks: {e.available_networks}")
```

### Buffer Errors

```python
from socialjax.buffers import BufferError, BufferEmptyError, InsufficientDataError

try:
    batch = buffer.sample(1000)
except InsufficientDataError:
    print("Not enough data in buffer")
```

### Configuration Errors

```python
from socialjax.config import ConfigValidationError

try:
    config = manager.load('unknown', 'unknown')
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

---

## Version

```python
import socialjax
print(socialjax.__version__)  # '2.0.0'
```

---

## See Also

- [Migration Guide](migration_guide.md) - Migrating from V1 to V2
- [Architecture](../ARCHITECTURE.md) - Detailed architecture documentation
- [Examples](../examples/) - Example scripts and notebooks
