# Migration Guide: V1 to V2

This guide helps you migrate your SocialJax code from V1 to the new V2 API. The V2 API provides a more modular, extensible, and easier-to-use interface while maintaining the same underlying JAX performance.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Major Changes Overview](#major-changes-overview)
3. [Training Scripts](#training-scripts)
4. [Algorithm Usage](#algorithm-usage)
5. [Configuration](#configuration)
6. [Callbacks and Logging](#callbacks-and-logging)
7. [Checkpoints](#checkpoints)
8. [Evaluation](#evaluation)
9. [Custom Algorithms](#custom-algorithms)
10. [Common Issues and Solutions](#common-issues-and-solutions)
11. [Backward Compatibility](#backward-compatibility)

---

## Quick Reference

| Feature | V1 Pattern | V2 Pattern |
|---------|-----------|------------|
| **Training** | `make_train(config)` → `train_jit(rng)` | `Trainer(algorithm, env).train(timesteps)` |
| **Algorithm** | Direct network creation | `get_algorithm('ippo')(obs_space, action_space, config)` |
| **Config** | Hydra + OmegaConf YAML | `ConfigManager` + dataclasses + YAML |
| **Logging** | Inline `wandb.log()` | `WandbCallback` in callback system |
| **Checkpoints** | `pickle.dump()` | `trainer.save(path)` / `algorithm.save(path)` |
| **Evaluation** | Custom `evaluate()` function | `Evaluator` class with metrics |

---

## Major Changes Overview

### V1 Architecture (Original)

```python
# V1: Everything in one file
from algorithms.utils import ActorCritic, batchify, save_params
import hydra
from omegaconf import OmegaConf
import wandb

@hydra.main(config_path="config", config_name="ippo_cnn_coins")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(project=config["PROJECT"], config=config)

    def make_train(config):
        env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        network = ActorCritic(env.action_space().n)
        # ... manual training loop with jax.lax.scan ...
        return train

    train_jit = jax.jit(make_train(config))
    out = train_jit(jax.random.PRNGKey(config["SEED"]))
    save_params(out["runner_state"][0].params, "checkpoint.pkl")
```

### V2 Architecture (New)

```python
# V2: Modular components with unified Trainer
import socialjax

# Create trainer with algorithm and environment names
trainer = socialjax.Trainer(
    algorithm='ippo',
    env='clean_up',
    callbacks=[
        socialjax.WandbCallback(project='my-project'),
        socialjax.CheckpointCallback(save_freq=1000, save_path='./checkpoints'),
    ]
)

# Train
state, metrics = trainer.train(total_timesteps=1_000_000)

# Evaluate
eval_metrics = trainer.evaluate(state, num_episodes=50)
```

---

## Training Scripts

### V1 Training Pattern

```python
# V1: algorithms/IPPO/ippo_cnn_coins.py
import sys
sys.path.append('/home/shuqing/SocialJax')
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb
from algorithms.utils import ActorCritic, batchify, save_params

def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

    def train(rng):
        # Manual network initialization
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        network_params = network.init(rng, init_x)

        # Manual optimizer setup
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Manual training loop with jax.lax.scan
        def _update_step(runner_state, unused):
            # ... hundreds of lines of training logic ...
            return runner_state, metric

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_coins")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(project=config["PROJECT"], config=config)

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    save_params(out["runner_state"][0].params, "checkpoint.pkl")

if __name__ == "__main__":
    main()
```

### V2 Training Pattern

```python
# V2: scripts/train.py or your custom script
import socialjax

# Simple: Create trainer from names
trainer = socialjax.Trainer(
    algorithm='ippo',
    env='coin_game',
    config={'total_timesteps': 1_000_000},
    callbacks=[
        socialjax.WandbCallback(project='my-project'),
        socialjax.CheckpointCallback(save_freq=1000, save_path='./checkpoints'),
        socialjax.ProgressCallback(total_timesteps=1_000_000),
    ]
)

# Train
state, metrics = trainer.train(total_timesteps=1_000_000)
print(f"Mean return: {metrics.get('mean_return', 0):.2f}")

# Save final model
trainer.save('./final_model')
```

### Using the CLI Script

```bash
# V2 provides a unified CLI script
python scripts/train.py --algorithm ippo --env coin_game --timesteps 1000000

# With WandB logging
python scripts/train.py --algorithm ippo --env clean_up \
    --timesteps 1000000 \
    --wandb-project socialjax \
    --wandb-name experiment1

# With custom config
python scripts/train.py --algorithm mappo --env clean_up \
    --config configs/mappo_cleanup.yaml

# Override learning rate
python scripts/train.py --algorithm ippo --env coin_game \
    --lr 0.0001 --gamma 0.99
```

---

## Algorithm Usage

### V1: Direct Network Creation

```python
# V1: Create network and manage everything manually
from algorithms.utils import ActorCritic

env = socialjax.make('coin_game', num_agents=5)

# Create network
network = ActorCritic(env.action_space().n, activation='relu')

# Initialize
rng = jax.random.PRNGKey(0)
init_x = jnp.zeros((1, *env.observation_space()[0].shape))
params = network.init(rng, init_x)

# Forward pass
pi, value = network.apply(params, observation)
action = pi.sample(seed=rng)
```

### V2: Registry-Based Algorithm Creation

```python
# V2: Use algorithm registry
import socialjax

env = socialjax.make('coin_game', num_agents=5)

# Get algorithm class from registry
IPPO = socialjax.get_algorithm('ippo')

# Create algorithm instance
algorithm = IPPO(
    observation_space=env.observation_space(),
    action_space=env.action_space(),
    config={
        'LR': 2.5e-4,
        'GAMMA': 0.99,
        'GAE_LAMBDA': 0.95,
    }
)

# Initialize algorithm state
rng = jax.random.PRNGKey(0)
state = algorithm.init_state(rng)

# Compute action
action, new_state = algorithm.compute_action(state, observation, rng)
```

### Available Algorithms

```python
# List all registered algorithms
print(socialjax.list_algorithms())
# ['ippo', 'mappo', 'svo', 'vdn']

# Get specific algorithm
IPPO = socialjax.get_algorithm('ippo')      # Independent PPO
MAPPO = socialjax.get_algorithm('mappo')    # Multi-Agent PPO (centralized critic)
VDN = socialjax.get_algorithm('vdn')        # Value Decomposition Network
SVO = socialjax.get_algorithm('svo')        # Social Value Orientation
```

---

## Configuration

### V1: Hydra + OmegaConf

```yaml
# V1: algorithms/IPPO/config/ippo_cnn_coins.yaml
ENV_NAME: "coin_game"
ENV_KWARGS:
  num_agents: 5
LR: 0.00025
GAMMA: 0.99
GAE_LAMBDA: 0.95
CLIP_EPS: 0.2
ENT_COEF: 0.01
VF_COEF: 0.5
MAX_GRAD_NORM: 0.5
NUM_ENVS: 8
NUM_STEPS: 128
TOTAL_TIMESTEPS: 10000000
SEED: 42
```

```python
# V1: Loading config
@hydra.main(config_path="config", config_name="ippo_cnn_coins")
def main(config):
    config = OmegaConf.to_container(config)
    lr = config["LR"]
```

### V2: ConfigManager with Dataclasses

```yaml
# V2: socialjax/config/presets/algorithms/ippo.yaml
algorithm: ippo
training:
  total_timesteps: 10000000
  num_envs: 8
  num_steps: 128
  gamma: 0.99
  learning_rate: 0.00025
  gae_lambda: 0.95
  clip_eps: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
network:
  type: cnn_small
  hidden_size: 128
```

```python
# V2: Loading config
from socialjax.config import ConfigManager, create_default_config

# Option 1: Load from presets
manager = socialjax.ConfigManager()
config = manager.load('ippo', 'coin_game')

# Option 2: Create default config
config = socialjax.create_default_config(algorithm='ippo')

# Option 3: Load from custom YAML file
config = manager.load_from_file('path/to/config.yaml')

# Access config values
lr = config.algorithm.training.learning_rate
gamma = config.algorithm.training.gamma
```

### V2 Configuration Classes

```python
from socialjax.config import (
    TrainingConfig,
    NetworkConfig,
    AlgorithmConfig,
    EnvironmentConfig,
    SocialJaxConfig,
)

# Create configs programmatically
training = TrainingConfig(
    total_timesteps=1_000_000,
    num_envs=8,
    num_steps=128,
    gamma=0.99,
    learning_rate=2.5e-4,
)

network = NetworkConfig(
    type='cnn_small',
    hidden_size=128,
    channel_sizes=(32, 64, 64),
)

# Build full config
config = SocialJaxConfig(
    algorithm=AlgorithmConfig(
        name='ippo',
        training=training,
        network=network,
    ),
    environment=EnvironmentConfig(
        name='coin_game',
        num_agents=5,
    )
)
```

---

## Callbacks and Logging

### V1: Inline Logging

```python
# V1: Logging embedded in training loop
import wandb

def _update_step(runner_state, unused):
    # ... training logic ...

    def callback(metric):
        wandb.log(metric)

    jax.debug.callback(callback, metric)
    return runner_state, metric
```

### V2: Callback System

```python
# V2: Use built-in callbacks
import socialjax
from socialjax.training.callbacks import (
    CheckpointCallback,
    EvalCallback,
    ProgressCallback,
    WandbCallback,
)

# Create callbacks
callbacks = [
    # Save checkpoints every 1000 updates
    socialjax.CheckpointCallback(
        save_freq=1000,
        save_path='./checkpoints',
        name_prefix='ippo',
        verbose=True,
    ),

    # Evaluate every 5000 timesteps
    socialjax.EvalCallback(
        eval_env=env,
        eval_freq=5000,
        n_eval_episodes=10,
        best_model_save_path='./best_model',
    ),

    # Progress bar
    socialjax.ProgressCallback(
        total_timesteps=1_000_000,
        progress_freq=10,
        show_metrics=['loss', 'episode_return'],
    ),

    # WandB logging
    socialjax.WandbCallback(
        project='socialjax-experiments',
        name='ippo_clean_up',
        config={'lr': 2.5e-4},
        log_freq=100,
    ),
]

# Pass callbacks to trainer
trainer = socialjax.Trainer(
    algorithm='ippo',
    env='clean_up',
    callbacks=callbacks,
)
```

### Custom Callbacks

```python
# V2: Create custom callbacks
from socialjax.training.callbacks import BaseCallback

class MyCustomCallback(BaseCallback):
    def on_training_start(self, trainer):
        print("Training started!")

    def on_update_end(self, trainer, update_metrics):
        if update_metrics.get('loss', 1.0) < 0.1:
            print(f"Loss is low: {update_metrics['loss']:.4f}")

    def on_training_end(self, trainer):
        print("Training completed!")

# Use custom callback
trainer = socialjax.Trainer(
    algorithm='ippo',
    env='clean_up',
    callbacks=[MyCustomCallback()],
)
```

---

## Checkpoints

### V1: Manual Pickle

```python
# V1: Manual save/load with pickle
import pickle

def save_params(params, path):
    with open(path, 'wb') as f:
        pickle.dump(params, f)

def load_params(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Save
save_params(train_state.params, 'checkpoints/model.pkl')

# Load
params = load_params('checkpoints/model.pkl')
```

### V2: Built-in Save/Load

```python
# V2: Use built-in save/load methods

# Option 1: Trainer-level save/load
trainer = socialjax.Trainer(algorithm='ippo', env='clean_up')
state, metrics = trainer.train(total_timesteps=1_000_000)

# Save entire trainer state
trainer.save('./checkpoints/final_model')

# Load and continue training
trainer.load('./checkpoints/final_model')
state, metrics = trainer.train(total_timesteps=500_000)  # Continue training

# Option 2: Algorithm-level save/load
algorithm = socialjax.get_algorithm('ippo')(obs_space, action_space, config)
state = algorithm.init_state(rng)

# Save algorithm state
algorithm.save('./checkpoints/algo_state.pkl')

# Load algorithm state
state = algorithm.load('./checkpoints/algo_state.pkl')
```

### CheckpointCallback

```python
# V2: Automatic checkpoint saving
checkpoint_callback = socialjax.CheckpointCallback(
    save_freq=1000,           # Save every 1000 updates
    save_path='./checkpoints',
    name_prefix='ippo_coins',
    verbose=True,
)

trainer = socialjax.Trainer(
    algorithm='ippo',
    env='coin_game',
    callbacks=[checkpoint_callback],
)
```

---

## Evaluation

### V1: Custom Evaluate Function

```python
# V1: Manual evaluation
def evaluate_ippo(params, env, save_path, config):
    network = ActorCritic(env.action_space().n)
    rng = jax.random.PRNGKey(0)

    episode_returns = []
    for episode in range(10):
        obs, state = env.reset(rng)
        done = False
        total_reward = 0

        while not done:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            pi, _ = network.apply(params, obs_batch)
            actions = pi.sample(seed=rng)

            obs, state, rewards, dones, _ = env.step(rng, state, actions)
            total_reward += sum(rewards.values())
            done = dones["__all__"]

        episode_returns.append(total_reward)

    return np.mean(episode_returns), np.std(episode_returns)
```

### V2: Evaluator Class

```python
# V2: Use Evaluator class
from socialjax.evaluation import Evaluator, EvaluatorConfig

# Create evaluator
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

print(f"Mean return: {metrics.mean_return:.2f} +/- {metrics.std_return:.2f}")
print(f"Cooperation rate: {metrics.cooperation_rate:.2%}")
print(f"Gini coefficient: {metrics.gini_coefficient:.3f}")

# Generate evaluation GIF
metrics, frames = evaluator.evaluate_with_frames()
socialjax.evaluation.save_gif(frames, 'evaluation.gif', fps=10)
```

### Evaluation Metrics

```python
# V2: Available metrics
from socialjax.evaluation import (
    EpisodeMetrics,           # Single episode metrics
    EvaluationMetrics,        # Aggregated metrics across episodes
    compute_episode_return,   # Total episode return
    compute_agent_returns,    # Per-agent returns
    compute_cooperation_rate, # Rate of cooperative actions
    compute_gini_coefficient, # Inequality measure
    compute_social_welfare,   # Total welfare
)

# Compute custom metrics
agent_returns = compute_agent_returns(rewards)
cooperation = compute_cooperation_rate(actions)
gini = compute_gini_coefficient(agent_returns)
```

---

## Custom Algorithms

### V1: Copy Existing Script

```python
# V1: Copy and modify existing algorithm file
# algorithms/IPPO/ippo_cnn_coins.py → algorithms/MY_ALGO/my_algo_env.py

# Modify the network, loss function, training loop, etc.
# Everything is in one file, hard to maintain
```

### V2: Implement BaseAlgorithm

```python
# V2: Create modular algorithm by extending BaseAlgorithm
import socialjax
from socialjax.core import BaseAlgorithm, AlgorithmState
import flax.linen as nn
import jax.numpy as jnp

@socialjax.register_algorithm('my_custom')
class MyCustomAlgorithm(socialjax.BaseAlgorithm):
    """My custom algorithm implementation."""

    def _build_network(self):
        """Build the neural network."""
        return MyCustomNetwork(
            action_dim=self.action_space.n,
            hidden_size=self.config.get('hidden_size', 128),
        )

    def _build_optimizer(self):
        """Build the optimizer."""
        import optax
        return optax.adam(self.config.get('LR', 2.5e-4))

    def init_state(self, rng):
        """Initialize algorithm state."""
        # Create initial params
        dummy_obs = jnp.zeros(self.observation_space.shape)
        params = self.network.init(rng, dummy_obs)
        optimizer_state = self.optimizer.init(params)

        return AlgorithmState(
            params=params,
            optimizer_state=optimizer_state,
            rng=rng,
            timestep=0,
        )

    def compute_action(self, state, observation, rng, deterministic=False):
        """Compute action from observation."""
        pi, value = self.network.apply(state.params, observation)

        if deterministic:
            action = pi.mode()
        else:
            action = pi.sample(seed=rng)

        return action, state

    def update(self, state, batch):
        """Update algorithm parameters."""
        # Compute loss and gradients
        def loss_fn(params):
            pi, value = self.network.apply(params, batch['observations'])
            log_prob = pi.log_prob(batch['actions'])

            # Your custom loss function
            policy_loss = -log_prob * batch['advantages'].mean()
            value_loss = ((value - batch['returns']) ** 2).mean()

            return policy_loss + 0.5 * value_loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, new_opt_state = self.optimizer.update(grads, state.optimizer_state)
        new_params = optax.apply_updates(state.params, updates)

        return state.replace(
            params=new_params,
            optimizer_state=new_opt_state,
        ), {'loss': loss}

# Your algorithm is now registered and can be used:
algorithm = socialjax.get_algorithm('my_custom')(obs_space, action_space, config)
```

---

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem:** V1 code uses direct imports from `algorithms/` directory.

```python
# V1
from algorithms.utils import ActorCritic, batchify
```

**Solution:** Use the socialjax package imports.

```python
# V2
import socialjax
from socialjax.networks import create_network
from socialjax.algorithms import get_algorithm
```

### Issue 2: Config Key Names

**Problem:** V1 uses uppercase config keys, V2 uses lowercase.

```python
# V1
lr = config["LR"]
gamma = config["GAMMA"]

# V2
lr = config.algorithm.training.learning_rate
gamma = config.algorithm.training.gamma
```

**Solution:** Update config access or create a compatibility layer.

```python
# Compatibility helper
def get_config_value(config, v1_key, default=None):
    """Get config value supporting both V1 and V2 key formats."""
    # Try V2 format first
    v2_key = v1_key.lower()
    if hasattr(config, 'algorithm'):
        training = config.algorithm.training
        if hasattr(training, v2_key):
            return getattr(training, v2_key)

    # Fall back to dict access (V1 style)
    if isinstance(config, dict):
        return config.get(v1_key, config.get(v2_key, default))

    return default
```

### Issue 3: Environment Wrappers

**Problem:** V1 uses `LogWrapper` from local wrappers.

```python
# V1
from socialjax.wrappers.baselines import LogWrapper
env = LogWrapper(env)
```

**Solution:** V2 has integrated logging, but wrappers are still available.

```python
# V2 - logging is handled by Trainer
trainer = socialjax.Trainer(algorithm='ippo', env='coin_game')
state, metrics = trainer.train(timesteps)

# Or use wrappers explicitly
from socialjax.environments.wrappers import (
    NormalizationWrapper,
    FrameStackWrapper,
    TimeLimitWrapper,
)
env = socialjax.make('coin_game')
env = NormalizationWrapper(env, normalize_obs=True, normalize_reward=True)
```

### Issue 4: Training Loop Differences

**Problem:** V1 uses JIT-compiled `jax.lax.scan`, V2 Trainer uses Python loops.

```python
# V1 - JIT-compiled scan
train_jit = jax.jit(make_train(config))
out = train_jit(rng)

# V2 - Python loop in Trainer
trainer = socialjax.Trainer(...)
state, metrics = trainer.train(timesteps)
```

**Note:** For maximum performance, you can still use JIT-compiled training loops. The V2 Trainer prioritizes flexibility and ease of use. If you need V1-level performance, you can:

1. JIT-compile your algorithm's update method
2. Use `jax.lax.scan` in custom training loops
3. Profile and optimize specific components

### Issue 5: Batchify Functions

**Problem:** V1 uses custom `batchify` functions for multi-agent handling.

```python
# V1
def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))
```

**Solution:** V2 algorithms handle this internally.

```python
# V2 - handled by algorithm
algorithm = socialjax.get_algorithm('ippo')(obs_space, action_space)
action, state = algorithm.compute_action(state, observations, rng)
```

---

## Backward Compatibility

### V1 Scripts Still Work

Your V1 scripts will continue to work without modification. The V1 algorithm files in `algorithms/` directory are preserved.

```bash
# V1 scripts still work
python algorithms/IPPO/ippo_cnn_coins.py
python algorithms/MAPPO/mappo_cnn_cleanup.py
```

### Gradual Migration

You don't need to migrate everything at once. V1 and V2 can coexist:

```python
# Mix V1 evaluation with V2 training
from algorithms.utils import evaluate_ippo  # V1 evaluation
import socialjax  # V2 training

# Train with V2
trainer = socialjax.Trainer(algorithm='ippo', env='coin_game')
state, metrics = trainer.train(1_000_000)

# Evaluate with V1 function (if you have custom evaluation logic)
# params = convert_v2_to_v1_params(state.params)
# evaluate_ippo(params, env, save_path, config)
```

### Checkpoint Compatibility

V1 and V2 checkpoints have different formats:

```python
# V1 checkpoint
# pickle file with train_state.params

# V2 checkpoint
# Structured checkpoint with AlgorithmState

# Conversion utility (if needed)
def convert_v1_to_v2_checkpoint(v1_path, v2_path):
    import pickle
    with open(v1_path, 'rb') as f:
        v1_params = pickle.load(f)

    # Create V2 algorithm and load params
    algorithm = socialjax.get_algorithm('ippo')(obs_space, action_space)
    state = algorithm.init_state(jax.random.PRNGKey(0))
    state = state.replace(params=v1_params)
    algorithm.save(v2_path)
```

---

## Migration Checklist

Use this checklist when migrating your code:

- [ ] **Update imports**: Change from `algorithms.utils` to `socialjax` package imports
- [ ] **Replace `make_train()`**: Use `Trainer` class instead of `make_train()` function
- [ ] **Update config format**: Migrate from Hydra/OmegaConf to ConfigManager
- [ ] **Add callbacks**: Replace inline logging with `WandbCallback`, `CheckpointCallback`, etc.
- [ ] **Update checkpoint code**: Use `trainer.save()` / `trainer.load()` instead of pickle
- [ ] **Update evaluation**: Use `Evaluator` class for standardized evaluation
- [ ] **Test thoroughly**: Run a short training experiment to verify everything works

---

## Getting Help

If you encounter issues during migration:

1. Check the [API Reference](api_reference.md) for V2 API details
2. Look at example scripts in `scripts/` directory
3. Review unit tests in `tests/` for usage patterns
4. Open an issue on GitHub with your migration question

---

## Summary

| Aspect | V1 Approach | V2 Approach |
|--------|-------------|-------------|
| **Philosophy** | Everything in one file | Modular components |
| **Training** | `make_train(config)` + JIT | `Trainer.train()` |
| **Config** | Hydra + OmegaConf | ConfigManager + dataclasses |
| **Logging** | Inline wandb.log() | Callback system |
| **Algorithms** | Direct instantiation | Registry pattern |
| **Evaluation** | Custom functions | Evaluator class |
| **Extensibility** | Copy-paste | Inheritance + composition |

The V2 API provides:
- **Easier onboarding**: Fewer lines of code to get started
- **Better organization**: Modular components with clear separation
- **More flexibility**: Easy to customize via inheritance and callbacks
- **Standard patterns**: Consistent API across algorithms and environments
- **Maintained performance**: Same JAX/Flax backend for GPU acceleration
