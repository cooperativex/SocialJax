<h1 align="center">SocialJax</h1>


<p align="center">
  <a href="https://arxiv.org/abs/2503.14576">
    <img src="https://img.shields.io/badge/arXiv-2503.14576-B31B1B.svg" alt="arXiv"></a>
  <a href="https://github.com/cooperativex/SocialJax/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0 License"></a>
  <a href="https://github.com/cooperativex/SocialJax/actions/workflows/speet_example.yml">
    <img src="https://github.com/cooperativex/SocialJax/actions/workflows/speet_example.yml/badge.svg" alt="Pylint Status"></a>
</p>

*A suite of sequential social dilemma environments for multi-agent reinforcement learning in JAX*



<p align="center">
  <img src="docs/images/step_150_reward_common_coins.gif" alt="coins_common" width="19.2%">
  <img src="docs/images/step_150_reward_common_harvestopen.gif" alt="harvest_open_common" width="18.5%">
  <img src="docs/images/step_150_reward_common_closed.gif" alt="harvest_closed_common" width="18.5%">
  <img src="docs/images/step_150_reward_common_cleanup.gif" alt="clean_up_common" width="19.8%">
  <img src="docs/images/step_250_reward_common_coop_mining.gif" alt="coop_mining_common" width="14%">
</p>

*Common Rewards* : a scenario where all agents share a single, unified reward signal. This approach ensures that all agents are aligned towards achieving the same objective, promoting collaboration and coordination among them.

<p align="center">
  <img src="docs/images/step_150_reward_individual_coins.gif" alt="coins_individual" width="19.2%">
  <img src="docs/images/step_150_reward_individual_harvestopen.gif" alt="harvest_open_individual" width="18.5%">
  <img src="docs/images/step_150_reward_individual_closed.gif" alt="harvest_closed_individual" width="18.5%">
  <img src="docs/images/step_150_reward_individual_cleanup.gif" alt="clean_up_individual" width="19.8%">
  <img src="docs/images/step_250_reward_individual_coop_mining.gif" alt="coop_mining_individual" width="14%">
</p>


***Individual Rewards***: each agent is assigned its own reward, inherently encouraging selfish behavior.


SocialJax leverages JAX's high-performance GPU capabilities to accelerate multi-agent reinforcement learning in sequential social dilemmas. We are committed to providing a more efficient and diverse suite of environments for studying social dilemmas. We provide JAX implementations of the following environments: Coins, Commons Harvest: Open, Commons Harvest: Closed, Clean Up, Territory, and Coop Mining, which are derived from [Melting Pot 2.0](https://github.com/google-deepmind/meltingpot/) and feature commonly studied mixed incentives.


Our [blog](https://sites.google.com/view/socialjax/home) presents more details and analysis on agents' policy and performance.




## Installation

First: Clone the repository
```bash
git clone https://github.com/cooperativex/SocialJax.git
cd SocialJax
```


Second: Environment Setup.

Option one: Using poetry, make sure you have python 3.10
  1. Install Poetry
       ```bash
       curl -sSL https://install.python-poetry.org | python3 -
       export PATH="$HOME/.local/bin:$PATH"
       ```

  2. Install requirements
       ```bash
       poetry install --no-root
       poetry run pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```
  3. Run code
       ```bash
       poetry run python scripts/train.py --algorithm ippo --env coin_game
       ```

Option two: conda with requirements.txt
  1. Conda
       ```bash
       conda create -n SocialJax python=3.10
       conda activate SocialJax
       ```

  2. Install requirements
       ```bash
       pip install -r requirements.txt
       pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```

  3. Run code
       ```bash
       python scripts/train.py --algorithm ippo --env coin_game
       ```

Option three: conda with environments.yml

  1. Install requirements
       ```bash
       conda env create -f environment.yml
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```

  2. Run code
       ```bash
       python scripts/train.py --algorithm ippo --env coin_game
       ```

## Environments

We introduce the environments and use Schelling diagrams to demonstrate whether the environments are social dilemmas.

| Environment                | Description                                                                                      | Schelling Diagrams Proof |
|----------------------------|--------------------------------------------------------------------------------------------------|:------------------------:|
| Coins                      | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coins)         |&check;                   |
| Commons Harvest: Open      | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest)|&check;                   |
| Commons Harvest: Closed    | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest)|&check;                   |
| Clean Up                   | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/cleanup)       |&check;                   |
| Territory                  | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/territory)     |&cross;                   |
| Coop Mining                | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coop_mining)   |&check;                   |

#### Important Notes:
- *Due to algorithmic limitations, agents may not always learn the optimal actions. As a result, Schelling diagrams can prove that the environment is social dilemmas, but they cannot definitively prove that the environment is not social dilemmas.*

- *Territory might not be Social diagram, but as long as the agents' behaviors are interesting, Territory holds intrinsic value.*

## Quick Start (V2 API)

SocialJax V2 provides a clean, modular interface for training multi-agent RL algorithms.

### Create an Environment

```python
import socialjax

# Create an environment
env = socialjax.make('clean_up', num_agents=7)

print(f"Number of agents: {env.num_agents}")
print(f"Observation shape: {env.observation_space().shape}")
print(f"Number of actions: {env.action_space().n}")
```

### Train with the Trainer API

```python
from socialjax.training import Trainer

# Create a trainer with IPPO on Clean Up
trainer = Trainer(
    algorithm='ippo',
    env='clean_up',
    num_agents=7,
    config={
        'total_timesteps': 100000,
        'learning_rate': 0.0005,
        'gamma': 0.99,
    }
)

# Train the model
metrics = trainer.train(total_timesteps=100000)
print(f"Mean episode return: {metrics.get('mean_episode_return', 'N/A')}")
```

### Command Line Training

```bash
# Train IPPO on coin_game
python scripts/train.py --algorithm ippo --env coin_game --timesteps 1000000

# Train MAPPO on clean_up with WandB logging
python scripts/train.py --algorithm mappo --env clean_up --timesteps 1000000 --wandb-project socialjax

# Evaluate a trained model
python scripts/evaluate.py --checkpoint checkpoints/ippo_final --env coin_game --episodes 50

# Generate visualization
python scripts/visualize.py --checkpoint checkpoints/ippo_final --env coin_game --output output.gif
```

## V2 API Reference

### Available Algorithms

| Algorithm | Description |
|-----------|-------------|
| `ippo` | Independent PPO (decentralized training and execution) |
| `mappo` | Multi-Agent PPO (centralized training, decentralized execution) |
| `vdn` | Value Decomposition Network (off-policy Q-learning) |
| `svo` | Social Value Orientation (prosocial reward shaping) |

### Available Environments

| Environment | Description |
|-------------|-------------|
| `coin_game` | Multi-agent coin collection game |
| `clean_up` | Public goods game with pollution |
| `harvest_common_open` | Commons harvesting with open access |
| `coop_mining` | Cooperative mining scenario |
| `territory_open` | Territory control game |
| `pd_arena` | Prisoner's dilemma arena |
| `mushrooms` | Mushroom foraging game |
| `gift` | Gift-giving coordination game |

### Core Components

```python
# Algorithm Registry
from socialjax.algorithms import (
    get_algorithm,      # Get algorithm class by name
    list_algorithms,    # List all registered algorithms
    register_algorithm, # Register custom algorithm
)

# Network Registry
from socialjax.networks import (
    create_network,     # Create network by name
    list_networks,      # List all registered networks
    CNNSmall,           # CNN feature extractor
    CNNActorCritic,     # CNN actor-critic network
    CNNImpala,          # IMPALA-style CNN
)

# Buffers
from socialjax.buffers import (
    RolloutBuffer,           # On-policy buffer
    ReplayBuffer,            # Off-policy replay buffer
    PrioritizedReplayBuffer, # Prioritized experience replay
)

# Callbacks
from socialjax.training import (
    CheckpointCallback,  # Save checkpoints periodically
    EvalCallback,        # Periodic evaluation
    ProgressCallback,    # Progress bar display
    WandbCallback,       # WandB logging
)
```

### Custom Algorithm Example

```python
from socialjax.core import BaseAlgorithm, AlgorithmState
from socialjax.algorithms import register_algorithm

@register_algorithm('my_algorithm')
class MyAlgorithm(BaseAlgorithm):
    def _build_network(self):
        # Define your network architecture
        pass

    def _build_optimizer(self):
        # Define your optimizer
        pass

    def compute_action(self, state, observation, rng):
        # Compute action given observation
        pass

    def update(self, state, batch):
        # Update algorithm parameters
        pass
```

### Evaluation and Visualization

```python
from socialjax.evaluation import (
    Evaluator,
    save_gif,
    compute_cooperation_rate,
)

# Evaluate trained agent
evaluator = Evaluator(algorithm, env)
metrics = evaluator.evaluate(n_episodes=50)

print(f"Mean return: {metrics.mean_return}")
print(f"Cooperation rate: {compute_cooperation_rate(metrics)}")

# Generate GIF
save_gif(frames, "output.gif", fps=10)
```

## Legacy V1 Code

The original V1 implementation has been moved to `v1_legacy/` for reference. For V1 usage, see:
- `v1_legacy/algorithms/` - Original algorithm implementations
- `v1_legacy/fixed_policy/` - Fixed policy examples
- `v1_legacy/speed_test/` - Speed benchmark scripts

For migration from V1 to V2, see [docs/migration_guide.md](docs/migration_guide.md).

## Tutorials

Interactive tutorials are available in the `tutorials/` directory:
1. `01_quickstart.ipynb` - Getting started with V2 API
2. `02_custom_algorithm.ipynb` - Implementing custom algorithms
3. `03_custom_network.ipynb` - Creating custom network architectures
4. `04_callbacks.ipynb` - Using callbacks for logging and monitoring
5. `05_advanced_config.ipynb` - Advanced configuration options

## See Also

[JaxMARL](https://github.com/flairox/jaxmarl): accelerated MARL environments with baselines in JAX.

[PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
